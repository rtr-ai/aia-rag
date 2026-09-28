import contextlib
import json
import secrets
from typing import Any, AsyncIterator, Dict, Generator, List, Optional

import httpx

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


def _parse_arguments(raw: str) -> Optional[Dict[str, Any]]:
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


@contextlib.contextmanager
def _translate_errors(what: str) -> Generator[None, None, None]:
    try:
        yield
    except httpx.TransportError as e:
        raise LlmUnavailable(f"OpenAI-compatible endpoint: {e}") from e
    except (json.JSONDecodeError, TypeError, AttributeError, KeyError) as e:
        raise LlmBadResponse(
            f"OpenAI-compatible endpoint: malformed {what}: {e}"
        ) from e


async def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code == 200:
        return
    await response.aread()
    status = response.status_code
    detail = response.text[:500]
    message = f"OpenAI-compatible endpoint: {detail} (status {status})"
    if status in (429, 503):
        raise LlmOverloaded(message)
    if status in (401, 403) or status >= 500:
        raise LlmUnavailable(message)
    raise LlmBadResponse(message)


async def _iter_chunks(response: httpx.Response) -> AsyncIterator[Dict[str, Any]]:
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            return
        chunk = json.loads(payload)
        if chunk.get("error"):
            raise LlmBadResponse(f"OpenAI-compatible endpoint: {chunk['error']}")
        yield chunk


def _parse_usage(chunk: Dict[str, Any]) -> Optional[Usage]:
    usage_data = chunk.get("usage") or {}
    if not usage_data:
        return None
    return Usage(
        prompt_tokens=usage_data.get("prompt_tokens"),
        completion_tokens=usage_data.get("completion_tokens"),
    )


def _merge_tool_call_fragments(
    calls: Dict[int, Dict[str, Any]], fragments: List[Dict[str, Any]]
) -> None:
    for fragment in fragments:
        call = calls.setdefault(
            fragment["index"],
            {"id": None, "name": None, "arguments": ""},
        )
        if fragment.get("id"):
            call["id"] = fragment["id"]
        function = fragment.get("function") or {}
        if function.get("name"):
            call["name"] = function["name"]
        if function.get("arguments"):
            call["arguments"] += function["arguments"]


def _build_tool_calls(calls: Dict[int, Dict[str, Any]]) -> List[ToolCall]:
    tool_calls: List[ToolCall] = []
    for index in sorted(calls):
        call = calls[index]
        if not call["name"]:
            raise LlmBadResponse(
                f"OpenAI-compatible endpoint: tool call {index} has no function name"
            )
        tool_calls.append(
            ToolCall(
                id=call["id"] or f"call_{secrets.token_hex(4)}",
                name=call["name"],
                arguments=_parse_arguments(call["arguments"]),
                raw_arguments=call["arguments"],
            )
        )
    return tool_calls


class OpenAiLlmClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        timeout: float,
        api_key: Optional[str],
    ):
        self.model = model
        self._temperature = temperature
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    async def stream_turn(
        self, messages: List[Message], tools: List[ToolSpec]
    ) -> AsyncIterator[TurnDelta]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [self._convert_message(message) for message in messages],
            "temperature": self._temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            body["tools"] = [
                {"type": "function", "function": spec.model_dump()} for spec in tools
            ]
        content_fragments: List[str] = []
        reasoning_fragments: List[str] = []
        calls: Dict[int, Dict[str, Any]] = {}
        usage = Usage()
        finish_reason: Optional[str] = None
        with _translate_errors("chunk"):
            async with self._client.stream(
                "POST", "/chat/completions", json=body
            ) as response:
                await _raise_for_status(response)
                async for chunk in _iter_chunks(response):
                    usage = _parse_usage(chunk) or usage
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning:
                        reasoning_fragments.append(reasoning)
                        yield ReasoningChunk(text=reasoning)
                    content = delta.get("content")
                    if content:
                        content_fragments.append(content)
                        yield ContentChunk(text=content)
                    _merge_tool_call_fragments(calls, delta.get("tool_calls") or [])
                    finish_reason = choices[0].get("finish_reason") or finish_reason
        if finish_reason is None:
            raise LlmBadResponse(
                "OpenAI-compatible endpoint: stream ended without a finish reason"
            )
        tool_calls = _build_tool_calls(calls)
        if tool_calls:
            finish = "tool_calls"
        elif finish_reason == "length":
            finish = "length"
        else:
            finish = "stop"
        yield TurnResult(
            content="".join(content_fragments),
            reasoning="".join(reasoning_fragments) if reasoning_fragments else None,
            tool_calls=tool_calls,
            finish_reason=finish,
            usage=usage,
            timings=Timings(),
        )

    async def ensure_model_available(self) -> None:
        with _translate_errors("model list"):
            response = await self._client.get("/models")
            await _raise_for_status(response)
            ids = [item["id"] for item in response.json()["data"]]
        if self.model not in ids:
            raise LlmBadResponse(
                f"OpenAI-compatible endpoint: model {self.model} not available"
            )

    @staticmethod
    def _convert_message(message: Message) -> Dict[str, Any]:
        if message.role == "assistant":
            converted: Dict[str, Any] = {
                "role": "assistant",
                "content": message.content,
            }
            if message.tool_calls:
                converted["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.raw_arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ]
            return converted
        if message.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }
        return {"role": message.role, "content": message.content}
