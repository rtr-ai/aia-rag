from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Union,
)

from pydantic import BaseModel


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: Optional[Dict[str, Any]]  # None if raw_arguments is not a JSON object
    raw_arguments: str  # arguments text exactly as the model emitted it


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: List[ToolCall] = []
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class Timings(BaseModel):  # seconds
    load_duration: Optional[float] = None
    prompt_eval_duration: Optional[float] = None
    eval_duration: Optional[float] = None
    total_duration: Optional[float] = None


class ReasoningChunk(BaseModel):
    text: str


class ContentChunk(BaseModel):
    text: str


class TurnResult(BaseModel):
    content: str
    reasoning: Optional[str] = None
    tool_calls: List[ToolCall] = []
    finish_reason: Literal["stop", "tool_calls", "length"]
    usage: Usage
    timings: Timings


TurnDelta = Union[ReasoningChunk, ContentChunk, TurnResult]


class LlmError(Exception): ...


class LlmUnavailable(LlmError): ...


class LlmOverloaded(LlmError): ...


class LlmBadResponse(LlmError): ...


class LlmClient(Protocol):
    model: str

    def stream_turn(
        self, messages: List[Message], tools: List[ToolSpec]
    ) -> AsyncIterator[TurnDelta]: ...

    async def ensure_model_available(self) -> None: ...
