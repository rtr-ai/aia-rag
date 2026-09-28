import contextlib
import json
import secrets
from typing import Any, AsyncIterator, Dict, Generator, List, Optional

import httpx
import ollama
from ollama import ChatResponse, ResponseError
from pydantic import ValidationError

from agent.llm_client import (
    ContentChunk,
    LlmBadResponse,
    LlmOverloaded,
    LlmUnavailable,
    Message,
    ReasoningChunk,
    Timings,
    ToolCall,
    ToolSpec,
    TurnDelta,
    TurnResult,
    Usage,
)
from utils.logger import get_logger

LOGGER = get_logger(__name__)


def _to_seconds(nanoseconds: Optional[int]) -> Optional[float]:
    return nanoseconds / 1_000_000_000 if nanoseconds is not None else None


@contextlib.contextmanager
def _translate_errors() -> Generator[None, None, None]:
    try:
        yield
    except ResponseError as e:
        status = e.status_code
        message = f"Ollama: {e.error} (status {status})"
        if status in (429, 503):
            raise LlmOverloaded(message) from e
        if status == -1 or status in (401, 403) or status >= 500:
            raise LlmUnavailable(message) from e
        raise LlmBadResponse(message) from e
    except (httpx.TransportError, ConnectionError) as e:
        raise LlmUnavailable(f"Ollama: {e}") from e
    except (json.JSONDecodeError, ValidationError) as e:
        raise LlmBadResponse(f"Ollama: malformed response: {e}") from e


def _to_tool_call(call: ollama.Message.ToolCall) -> ToolCall:
    arguments = dict(call.function.arguments)
    return ToolCall(
        id=f"call_{secrets.token_hex(4)}",
        name=call.function.name,
        arguments=arguments,
        raw_arguments=json.dumps(arguments, ensure_ascii=False),
    )


class OllamaLlmClient:
    def __init__(
        self,
        host: str,
        model: str,
        temperature: float,
        context_window: int,
        timeout: float,
    ):
        self.model = model
        self._temperature = temperature
        self._context_window = context_window
        self._client = ollama.AsyncClient(
            host=host, timeout=httpx.Timeout(timeout, connect=10.0)
        )

    async def stream_turn(
        self, messages: List[Message], tools: List[ToolSpec]
    ) -> AsyncIterator[TurnDelta]:
        request_messages = [self._convert_message(message) for message in messages]
        request_tools = [
            {"type": "function", "function": spec.model_dump()} for spec in tools
        ] or None
        content_fragments: List[str] = []
        reasoning_fragments: List[str] = []
        tool_calls: List[ToolCall] = []
        done: Optional[ChatResponse] = None
        with _translate_errors():
            stream = await self._client.chat(
                model=self.model,
                messages=request_messages,
                tools=request_tools,
                options={
                    "temperature": self._temperature,
                    "num_ctx": self._context_window,
                },
                stream=True,
            )
            async for part in stream:
                if part.message.thinking:
                    reasoning_fragments.append(part.message.thinking)
                    yield ReasoningChunk(text=part.message.thinking)
                if part.message.content:
                    content_fragments.append(part.message.content)
                    yield ContentChunk(text=part.message.content)
                if part.message.tool_calls:
                    tool_calls.extend(
                        _to_tool_call(call) for call in part.message.tool_calls
                    )
                if part.done:
                    done = part
        if done is None:
            raise LlmBadResponse("Ollama: stream ended without a final chunk")

        if tool_calls:
            finish_reason = "tool_calls"
        elif done.done_reason == "length":
            finish_reason = "length"
        else:
            finish_reason = "stop"

        yield TurnResult(
            content="".join(content_fragments),
            reasoning="".join(reasoning_fragments) if reasoning_fragments else None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=Usage(
                prompt_tokens=done.prompt_eval_count,
                completion_tokens=done.eval_count,
            ),
            timings=Timings(
                load_duration=_to_seconds(done.load_duration),
                prompt_eval_duration=_to_seconds(done.prompt_eval_duration),
                eval_duration=_to_seconds(done.eval_duration),
                total_duration=_to_seconds(done.total_duration),
            ),
        )

    async def _model_exists(self) -> bool:
        with _translate_errors():
            try:
                await self._client.show(self.model)
            except ResponseError as e:
                if e.status_code == 404:
                    return False
                raise
        return True

    async def ensure_model_available(self) -> None:
        if await self._model_exists():
            return
        LOGGER.info(f"LLM model {self.model} not found. Pulling...")
        with _translate_errors():
            async for _ in await self._client.pull(self.model, stream=True):
                pass
        LOGGER.info(f"LLM model {self.model} pulled successfully.")

    @staticmethod
    def _convert_message(message: Message) -> Dict[str, Any]:
        converted: Dict[str, Any] = {"role": message.role, "content": message.content}
        if message.role == "assistant" and message.tool_calls:
            converted["tool_calls"] = [
                {
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments or {},
                    }
                }
                for tool_call in message.tool_calls
            ]
        if message.tool_name:
            converted["tool_name"] = message.tool_name
        return converted
