import json
import os
import secrets
import time
from typing import AsyncIterator, Dict, List, Optional, Tuple

from agent import llm_calls
from agent.agent_prompts import AgentPrompts
from agent.event_stream import EventStream
from agent.llm_client import (
    ContentChunk,
    LlmBadResponse,
    LlmClient,
    LlmError,
    LlmOverloaded,
    LlmUnavailable,
    Message,
    ReasoningChunk,
    ToolCall,
    ToolSpec,
    TurnResult,
)
from agent.llm_provider import CONTEXT_WINDOW, LLM_PROVIDER, TEMPERATURE
from agent.registry import (
    NeedsUserInput,
    NestedEvent,
    ToolContext,
    ToolRegistry,
    language_for_dataset,
)
from agent.run_store import PausedRun, RunStore
from agent.tools.ask_user import ASK_USER
from models.agent_events import (
    AwaitingUser,
    ContentDelta,
    LimitReached,
    Limits,
    LlmFinished,
    LlmRequest,
    QueueStatus,
    ReasoningDelta,
    RunError,
    RunFinished,
    RunStarted,
    SystemPrompt,
    ToolCall as ToolCallEvent,
    ToolFinished,
    ToolResult,
    ToolStarted,
    TurnStarted,
    UserMessage,
)
from services.tokenizer_service import TokenizerService
from utils.logger import get_logger

AGENT_MAX_TURNS = int(os.getenv("AGENT_MAX_TURNS", "12"))

LOGGER = get_logger(__name__)
LOGGER_CHAT = get_logger("agent_chat", "chat.txt")
TOKENIZER = TokenizerService()


class AgentRun:
    def __init__(
        self,
        llm_client: LlmClient,
        registry: ToolRegistry,
        prompts: Dict[str, AgentPrompts],
        run_store: RunStore,
        dataset: str,
        prompt: str,
    ):
        self._llm_client = llm_client
        self._registry = registry
        self._dataset = dataset
        self._language = language_for_dataset(dataset)
        self._prompts = prompts[self._language]
        self._run_store = run_store
        self._prompt = prompt
        self._run_id = secrets.token_hex(8)
        self._events = EventStream(self._run_id)
        self._messages: List[Message] = []
        self._call_counts: Dict[str, int] = {}
        self._turns = 0
        self._turn_offset = 0
        self._segment_counts: Dict[str, int] = {}
        self._answer: Optional[str] = None
        self._paused: Optional[PausedRun] = None
        self._pending_ask: Optional[Tuple[str, ToolCall, float, NeedsUserInput]] = None

    @classmethod
    def from_paused(
        cls,
        llm_client: LlmClient,
        registry: ToolRegistry,
        prompts: Dict[str, AgentPrompts],
        run_store: RunStore,
        paused: PausedRun,
        answer: str,
    ) -> "AgentRun":
        run = cls(
            llm_client=llm_client,
            registry=registry,
            prompts=prompts,
            run_store=run_store,
            dataset=paused.dataset,
            prompt=paused.prompt,
        )
        run._run_id = paused.run_id
        run._events = EventStream(paused.run_id, paused.seq, paused.step_count)
        run._messages = list(paused.messages)
        run._call_counts = dict(paused.call_counts)
        run._turns = paused.turns
        run._turn_offset = paused.turns
        run._answer = answer
        run._paused = paused
        return run

    async def run(self) -> AsyncIterator[str]:
        if self._answer is None:
            LOGGER.info(
                f"[{self._run_id}] Agent run started, dataset <{self._dataset}>"
            )
        else:
            LOGGER.info(f"[{self._run_id}] Agent run resumed after user input")
        try:
            async for frame in self._run():
                yield frame
        except LlmError as e:
            LOGGER.error(f"[{self._run_id}] LLM error: {e}")
            yield self._events.emit(
                RunError(
                    message=self._llm_error_message(e),
                    recoverable=isinstance(e, LlmOverloaded),
                )
            )
            yield self._finish("error", "")
        except Exception:
            LOGGER.exception(f"[{self._run_id}] Agent run failed")
            yield self._events.emit(
                RunError(message="Internal error.", recoverable=False)
            )
            yield self._finish("error", "")

    async def _run(self) -> AsyncIterator[str]:
        if self._answer is None:
            async for frame in self._start():
                yield frame
        else:
            async for frame in self._resume():
                yield frame
        async for frame in self._loop():
            yield frame

    async def _start(self) -> AsyncIterator[str]:
        tools = self._registry.specs(self._language)
        yield self._events.emit(
            RunStarted(
                dataset=self._dataset,
                language=self._language,
                provider=LLM_PROVIDER,
                model=self._llm_client.model,
                temperature=TEMPERATURE,
                context_window=CONTEXT_WINDOW,
                tools=[spec.model_dump() for spec in tools],
                tool_display_names=self._registry.display_names(self._language),
                limits=Limits(
                    max_turns=AGENT_MAX_TURNS,
                    max_calls_per_tool=self._registry.max_calls(),
                ),
            )
        )
        yield self._events.emit(
            SystemPrompt(
                prompt_key=f"agent_{self._language}", text=self._prompts.system
            )
        )
        yield self._events.emit(UserMessage(text=self._prompt))
        self._messages = [
            Message(role="system", content=self._prompts.system),
            Message(role="user", content=self._prompt),
        ]

    async def _resume(self) -> AsyncIterator[str]:
        paused = self._paused
        answer = self._answer
        if paused is None or answer is None:
            raise RuntimeError("Run is not resumable")
        yield self._events.emit(ToolResult(text=answer), paused.ask_step_id)
        yield self._events.emit(
            ToolFinished(duration=time.monotonic() - paused.ask_started),
            paused.ask_step_id,
        )
        self._messages.append(
            Message(
                role="tool",
                content=answer,
                tool_call_id=paused.ask_tool_call_id,
                tool_name=paused.ask_tool_name,
            )
        )

    async def _loop(self) -> AsyncIterator[str]:
        while True:
            self._turns += 1
            last_turn = self._turns - self._turn_offset >= AGENT_MAX_TURNS
            tools = [
                spec
                for spec in self._registry.specs(self._language)
                if spec.name != ASK_USER or self._call_counts.get(ASK_USER, 0) == 0
            ]
            turn_tools = [] if last_turn else tools
            if last_turn:
                yield self._events.emit(
                    LimitReached(
                        limit="max_turns",
                        detail=(
                            f"Turn limit of {AGENT_MAX_TURNS} reached. "
                            "Final turn without tools."
                        ),
                    )
                )
                self._messages.append(
                    Message(role="user", content=self._prompts.max_turns_reached)
                )

            step_id = self._events.new_step()
            yield self._events.emit(
                TurnStarted(
                    turn=self._turns,
                    message_count=len(self._messages),
                    context_tokens=self._context_tokens(turn_tools),
                ),
                step_id,
            )
            yield self._events.emit(
                LlmRequest(
                    model=self._llm_client.model,
                    messages=[message.model_dump() for message in self._messages],
                    tools=[spec.name for spec in turn_tools],
                ),
                step_id,
            )
            yield self._events.emit(
                QueueStatus(llm_calls_in_flight=llm_calls.in_flight())
            )

            result: Optional[TurnResult] = None
            started = time.monotonic()
            with llm_calls.track():
                async for delta in self._llm_client.stream_turn(
                    self._messages, turn_tools
                ):
                    if isinstance(delta, ReasoningChunk):
                        yield self._events.emit(
                            ReasoningDelta(text=delta.text), step_id
                        )
                    elif isinstance(delta, ContentChunk):
                        yield self._events.emit(ContentDelta(text=delta.text), step_id)
                    elif isinstance(delta, TurnResult):
                        result = delta
            duration = time.monotonic() - started
            if result is None:
                raise LlmBadResponse("LLM stream ended without a turn result")

            tool_calls = result.tool_calls if turn_tools else []
            counted: List[Tuple[ToolCall, int, int]] = []
            for call in tool_calls:
                self._call_counts[call.name] = self._call_counts.get(call.name, 0) + 1
                self._segment_counts[call.name] = (
                    self._segment_counts.get(call.name, 0) + 1
                )
                counted.append(
                    (
                        call,
                        self._call_counts[call.name],
                        self._segment_counts[call.name],
                    )
                )
                yield self._events.emit(
                    ToolCallEvent(
                        tool_call_id=call.id,
                        name=call.name,
                        arguments=call.raw_arguments,
                        call_index=self._call_counts[call.name],
                    ),
                    step_id,
                )
            yield self._events.emit(
                LlmFinished(
                    finish_reason=result.finish_reason,
                    content=result.content,
                    reasoning=result.reasoning,
                    usage=result.usage.model_dump(),
                    timings=result.timings.model_dump(),
                    duration=duration,
                    time_to_first_token=result.time_to_first_token,
                    generation_duration=(
                        duration - result.time_to_first_token
                        if result.time_to_first_token is not None
                        else None
                    ),
                ),
                step_id,
            )
            if not tool_calls and not result.content:
                # servers can silently drop malformed tool calls, see agent-issues.md #06
                raise LlmBadResponse(
                    f"LLM returned neither content nor tool calls "
                    f"({result.usage.completion_tokens} completion tokens)"
                )
            if not tool_calls:
                break

            self._messages.append(
                Message(
                    role="assistant",
                    content=result.content,
                    reasoning=result.reasoning,
                    tool_calls=tool_calls,
                )
            )
            for call, call_index, segment_index in self._execution_order(counted):
                async for frame in self._run_tool(call, call_index, segment_index):
                    yield frame
            if self._pending_ask is not None:
                yield self._pause()
                return

        finish_reason = "max_turns" if last_turn else "answer"
        LOGGER_CHAT.info(f"[{self._run_id}] {self._prompt}\n\n")
        if self._answer is not None:
            LOGGER_CHAT.info(f"[{self._run_id}] {self._answer}\n\n")
        LOGGER_CHAT.info(f"[{self._run_id}] {result.content}\n\n\n")
        LOGGER.info(
            f"[{self._run_id}] Agent run finished: {finish_reason} "
            f"after {self._turns} turns"
        )
        yield self._finish(finish_reason, result.content)

    @staticmethod
    def _execution_order(
        counted: List[Tuple[ToolCall, int, int]],
    ) -> List[Tuple[ToolCall, int, int]]:
        others = [entry for entry in counted if entry[0].name != ASK_USER]
        asks = [entry for entry in counted if entry[0].name == ASK_USER]
        return others + asks[1:] + asks[:1]

    async def _run_tool(
        self, call: ToolCall, call_index: int, segment_index: int
    ) -> AsyncIterator[str]:
        step_id = self._events.new_step()
        started = time.monotonic()
        yield self._events.emit(
            ToolStarted(
                tool_call_id=call.id,
                name=call.name,
                arguments=call.arguments,
                call_index=call_index,
            ),
            step_id,
        )
        max_calls = self._registry.max_calls().get(call.name)
        result: Optional[ToolResult] = None
        counted_index = call_index if call.name == ASK_USER else segment_index
        if max_calls is not None and counted_index > max_calls:
            text = (
                f"Call limit reached: tool '{call.name}' is limited to "
                f"{max_calls} calls. Continue without it."
            )
            yield self._events.emit(
                LimitReached(limit="max_calls_per_tool", tool=call.name, detail=text),
                step_id,
            )
            result = ToolResult(text=text, is_error=True)
        else:
            ctx = ToolContext(
                run_id=self._run_id,
                step_id=step_id,
                dataset=self._dataset,
                language=self._language,
                user_prompt=self._prompt,
                new_step=self._events.new_step,
            )
            try:
                async for event in self._registry.invoke(
                    call.name, call.arguments, ctx
                ):
                    if isinstance(event, ToolResult):
                        result = event
                    elif isinstance(event, NestedEvent):
                        yield self._events.emit(event.payload, event.step_id, step_id)
                    elif isinstance(event, NeedsUserInput):
                        self._pending_ask = (step_id, call, started, event)
                        break
                    else:
                        yield self._events.emit(event, step_id)
            except Exception:
                LOGGER.exception(f"[{self._run_id}] Tool <{call.name}> failed")
                result = ToolResult(
                    text=f"Tool '{call.name}' failed with an internal error.",
                    is_error=True,
                )
            if self._pending_ask is not None:
                return
            if result is None:
                result = ToolResult(
                    text=f"Tool '{call.name}' returned no result.", is_error=True
                )
        yield self._events.emit(result, step_id)
        yield self._events.emit(
            ToolFinished(duration=time.monotonic() - started), step_id
        )
        self._messages.append(
            Message(
                role="tool",
                content=result.text,
                tool_call_id=call.id,
                tool_name=call.name,
            )
        )

    def _pause(self) -> str:
        if self._pending_ask is None:
            raise RuntimeError("No pending ask_user call")
        step_id, call, started, ask = self._pending_ask
        # the token must exist before awaiting_user can be emitted, so seq is patched
        # after emit; there is no await in between, so no resume can interleave
        paused = PausedRun(
            run_id=self._run_id,
            dataset=self._dataset,
            language=self._language,
            prompt=self._prompt,
            messages=list(self._messages),
            turns=self._turns,
            call_counts=dict(self._call_counts),
            seq=0,
            step_count=self._events.step_count,
            ask_step_id=step_id,
            ask_tool_call_id=call.id,
            ask_tool_name=call.name,
            ask_started=started,
        )
        token = self._run_store.save(paused)
        frame = self._events.emit(
            AwaitingUser(
                tool_call_id=call.id,
                question=ask.question,
                options=ask.options,
                resume_token=token,
            ),
            step_id,
        )
        paused.seq = self._events.seq
        LOGGER.info(
            f"[{self._run_id}] Agent run paused for user input in turn {self._turns}"
        )
        return frame

    def _context_tokens(self, tools: List[ToolSpec]) -> int:
        serialized = json.dumps(
            [message.model_dump() for message in self._messages]
            + [spec.model_dump() for spec in tools],
            ensure_ascii=False,
        )
        return TOKENIZER.count_tokens(serialized)

    def _finish(self, finish_reason: str, answer: str) -> str:
        return self._events.emit(
            RunFinished(
                finish_reason=finish_reason,
                answer=answer,
                turns=self._turns,
                tool_calls=self._call_counts,
            )
        )

    @staticmethod
    def _llm_error_message(error: LlmError) -> str:
        if isinstance(error, LlmOverloaded):
            return "The language model is overloaded. Please try again later."
        if isinstance(error, LlmUnavailable):
            return "The language model is currently unavailable."
        return "The language model returned an invalid response."
