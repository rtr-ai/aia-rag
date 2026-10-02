import json
import os
import secrets
import time
from typing import AsyncIterator, Dict, List, Optional

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
from agent.registry import NestedEvent, ToolContext, ToolRegistry, language_for_dataset
from models.agent_events import (
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
        dataset: str,
        prompt: str,
    ):
        self._llm_client = llm_client
        self._registry = registry
        self._dataset = dataset
        self._language = language_for_dataset(dataset)
        self._prompts = prompts[self._language]
        self._prompt = prompt
        self._run_id = secrets.token_hex(8)
        self._events = EventStream(self._run_id)
        self._messages: List[Message] = []
        self._call_counts: Dict[str, int] = {}
        self._turns = 0

    async def run(self) -> AsyncIterator[str]:
        LOGGER.info(f"[{self._run_id}] Agent run started, dataset <{self._dataset}>")
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
        while True:
            self._turns += 1
            last_turn = self._turns >= AGENT_MAX_TURNS
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
            call_indexes: List[int] = []
            for call in tool_calls:
                self._call_counts[call.name] = self._call_counts.get(call.name, 0) + 1
                call_indexes.append(self._call_counts[call.name])
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
            for call, call_index in zip(tool_calls, call_indexes):
                async for frame in self._run_tool(call, call_index):
                    yield frame

        finish_reason = "max_turns" if last_turn else "answer"
        LOGGER_CHAT.info(f"[{self._run_id}] {self._prompt}\n\n")
        LOGGER_CHAT.info(f"[{self._run_id}] {result.content}\n\n\n")
        LOGGER.info(
            f"[{self._run_id}] Agent run finished: {finish_reason} "
            f"after {self._turns} turns"
        )
        yield self._finish(finish_reason, result.content)

    async def _run_tool(self, call: ToolCall, call_index: int) -> AsyncIterator[str]:
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
        if max_calls is not None and call_index > max_calls:
            text = (
                f"Call limit reached: tool '{call.name}' is limited to "
                f"{max_calls} calls per run. Continue without it."
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
                    else:
                        yield self._events.emit(event, step_id)
            except Exception:
                LOGGER.exception(f"[{self._run_id}] Tool <{call.name}> failed")
                result = ToolResult(
                    text=f"Tool '{call.name}' failed with an internal error.",
                    is_error=True,
                )
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
