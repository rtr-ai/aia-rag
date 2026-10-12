from typing import Any, ClassVar, Dict, List, Literal, Optional

from models.sources import Source
from pydantic import BaseModel


class Energy(BaseModel):
    cpu_kWh: float
    gpu_kWh: float
    ram_kWh: float
    total_kWh: float
    duration: float
    timing_source: Literal["provider", "wall_clock"]
    scope: Literal["service_host"] = "service_host"
    contended: bool = False


class Usage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class Timings(BaseModel):
    load_duration: Optional[float] = None
    prompt_eval_duration: Optional[float] = None
    eval_duration: Optional[float] = None
    total_duration: Optional[float] = None


class Limits(BaseModel):
    max_turns: int
    max_calls_per_tool: Dict[str, int]


class GuidelineSource(BaseModel):
    id: str
    document: str
    paragraphs: List[str]
    title: str
    content: str
    num_tokens: int
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    embedding_score: Optional[float] = None
    embedding_rank: Optional[int] = None
    fused_score: float
    matched_by: Literal["query", "prompt"]
    skip: bool
    skip_reason: str = ""


class LookupChunk(BaseModel):
    id: str
    corpus: str
    reference: str
    title: str
    content: str
    num_tokens: int
    position: int


class RunStarted(BaseModel):
    EVENT_TYPE: ClassVar[str] = "run_started"

    dataset: str
    language: str
    provider: str
    model: str
    temperature: float
    context_window: int
    tools: List[Dict[str, Any]]
    tool_display_names: Dict[str, str]
    limits: Limits


class SystemPrompt(BaseModel):
    EVENT_TYPE: ClassVar[str] = "system_prompt"

    prompt_key: str
    text: str


class UserMessage(BaseModel):
    EVENT_TYPE: ClassVar[str] = "user_message"

    text: str


class TurnStarted(BaseModel):
    EVENT_TYPE: ClassVar[str] = "turn_started"

    turn: int
    message_count: int
    context_tokens: int


class LlmRequest(BaseModel):
    EVENT_TYPE: ClassVar[str] = "llm_request"

    model: str
    messages: List[Dict[str, Any]]
    tools: List[str]


class ReasoningDelta(BaseModel):
    EVENT_TYPE: ClassVar[str] = "reasoning_delta"

    text: str


class ContentDelta(BaseModel):
    EVENT_TYPE: ClassVar[str] = "content_delta"

    text: str


class ToolCall(BaseModel):
    EVENT_TYPE: ClassVar[str] = "tool_call"

    tool_call_id: str
    name: str
    arguments: str
    call_index: int


class LlmFinished(BaseModel):
    EVENT_TYPE: ClassVar[str] = "llm_finished"

    finish_reason: str
    content: str
    reasoning: Optional[str] = None
    usage: Usage
    timings: Timings
    duration: float
    time_to_first_token: Optional[float] = None
    generation_duration: Optional[float] = None
    energy: Optional[Energy] = None


class ToolStarted(BaseModel):
    EVENT_TYPE: ClassVar[str] = "tool_started"

    tool_call_id: str
    name: str
    arguments: Optional[Dict[str, Any]]
    call_index: int


class ToolSubstep(BaseModel):
    EVENT_TYPE: ClassVar[str] = "tool_substep"

    label: str
    detail: str


class RagSources(BaseModel):
    EVENT_TYPE: ClassVar[str] = "rag_sources"

    sources: List[Source]
    duration: Optional[float] = None
    energy: Optional[Energy] = None


class GuidelineSources(BaseModel):
    EVENT_TYPE: ClassVar[str] = "guideline_sources"

    sources: List[GuidelineSource]
    duration: Optional[float] = None
    energy: Optional[Energy] = None


class LookupChunks(BaseModel):
    EVENT_TYPE: ClassVar[str] = "lookup_chunks"

    chunks: List[LookupChunk]


class ToolResult(BaseModel):
    EVENT_TYPE: ClassVar[str] = "tool_result"

    text: str
    is_error: bool = False


class ToolFinished(BaseModel):
    EVENT_TYPE: ClassVar[str] = "tool_finished"

    duration: float
    energy: Optional[Energy] = None


class QueueStatus(BaseModel):
    EVENT_TYPE: ClassVar[str] = "queue_status"

    llm_calls_in_flight: int


class LimitReached(BaseModel):
    EVENT_TYPE: ClassVar[str] = "limit_reached"

    limit: Literal["max_turns", "max_calls_per_tool"]
    tool: Optional[str] = None
    detail: str


class AwaitingUser(BaseModel):
    EVENT_TYPE: ClassVar[str] = "awaiting_user"

    tool_call_id: str
    question: str
    options: List[str]
    resume_token: str


class RunFinished(BaseModel):
    EVENT_TYPE: ClassVar[str] = "run_finished"

    finish_reason: Literal["answer", "max_turns", "error"]
    answer: str
    turns: int
    tool_calls: Dict[str, int]
    energy: Optional[Energy] = None


class RunError(BaseModel):
    EVENT_TYPE: ClassVar[str] = "error"

    message: str
    recoverable: bool = False
