import os
from typing import AsyncIterator, Dict, List, Optional, Tuple

from agent import llm_calls
from agent.llm_client import ContentChunk, ReasoningChunk
from agent.registry import NestedEvent, ToolContext, ToolEvent
from models.agent_events import (
    ContentDelta,
    LlmFinished,
    LlmRequest,
    QueueStatus,
    RagSources,
    ReasoningDelta,
    SystemPrompt,
    ToolResult,
    ToolSubstep,
)
from models.sources import Source
from pydantic import BaseModel, ConfigDict, Field
from services.embedding_service import DEFAULT_MODEL
from services.index_service import CONTEXT_WINDOW, PROMPT_BUFFER
from services.rag_pipeline import (
    GenerationFinished,
    PromptBuilt,
    RagPipeline,
    SourcesRetrieved,
)
from services.tokenizer_service import TokenizerService
from utils.prompt_utils import used_sources

AGENT_SEARCH_AI_ACT_MAX_CALLS = int(os.getenv("AGENT_SEARCH_AI_ACT_MAX_CALLS", "2"))

_token_limit_raw = os.getenv("AGENT_SEARCH_AI_ACT_TOKEN_LIMIT", "")
AGENT_SEARCH_AI_ACT_TOKEN_LIMIT: Optional[int] = (
    int(_token_limit_raw) if _token_limit_raw else None
)

TOKENIZER = TokenizerService()

TEXTS = {
    "de": {
        "user": "Quellen:\n\n{context_str}\n\n---\n\nSuchanfrage: {query_str}",
        "sources_used": "Verwendete Quellen:",
        "no_sources": "keine",
        "substep_embedding": (
            "Retrieve · ① Embedding der Suchanfrage",
            "Die Suchanfrage wird mit {embedding_model} in einen Vektor überführt.",
        ),
        "substep_search": (
            "Retrieve · ② Semantische Vektorsuche",
            "Abgleich gegen {corpus} Abschnitte nach Kosinus-Ähnlichkeit → die {top_n} ähnlichsten Treffer.",
        ),
        "substep_crossrefs": (
            "Retrieve · ③ Querverweise ergänzen",
            "{relevant} manuell gepflegte Querverweise der Treffer werden hinzugeladen.",
        ),
        "substep_context": (
            "Retrieve · ④ Kontext begrenzen",
            "{used} von {total} Quellen kommen in den Kontext ({used_tokens} von max. {token_limit} Token); {duplicates} Dubletten und {context_window} wegen der Kontextgrenze verworfen.",
        ),
        "substep_prompt": (
            "Augment · ⑤ Prompt aufbauen",
            "System-Prompt, {used} Quellen im Volltext und die Suchanfrage werden zum Prompt zusammengesetzt (ca. {prompt_tokens} Token).",
        ),
        "substep_answer": (
            "Generate · ⑥ Antwort erzeugen",
            "{llm_model} erzeugt aus dem Prompt eine quellengebundene Antwort.",
        ),
    },
    "en": {
        "user": "Sources:\n\n{context_str}\n\n---\n\nSearch query: {query_str}",
        "sources_used": "Sources used:",
        "no_sources": "none",
        "substep_embedding": (
            "Retrieve · ① Embedding of the search query",
            "The search query is converted into a vector with {embedding_model}.",
        ),
        "substep_search": (
            "Retrieve · ② Semantic vector search",
            "Comparison against {corpus} sections by cosine similarity → the {top_n} most similar hits.",
        ),
        "substep_crossrefs": (
            "Retrieve · ③ Add cross-references",
            "{relevant} manually maintained cross-references of the hits are added.",
        ),
        "substep_context": (
            "Retrieve · ④ Limit the context",
            "{used} of {total} sources go into the context ({used_tokens} of max. {token_limit} tokens); {duplicates} duplicates and {context_window} discarded due to the context limit.",
        ),
        "substep_prompt": (
            "Augment · ⑤ Build the prompt",
            "System prompt, {used} sources in full text and the search query are assembled into the prompt (approx. {prompt_tokens} tokens).",
        ),
        "substep_answer": (
            "Generate · ⑥ Generate the answer",
            "{llm_model} generates a source-bound answer from the prompt.",
        ),
    },
}


def _skip_counts(sources: List[Source]) -> Tuple[int, int]:
    duplicates = 0
    context_window = 0
    for source in sources:
        if source.skip_reason == "duplicate":
            duplicates += 1
        elif source.skip_reason == "context_window":
            context_window += 1
        for relevant_chunk in source.relevantChunks:
            if relevant_chunk.skip_reason == "duplicate":
                duplicates += 1
            elif relevant_chunk.skip_reason == "context_window":
                context_window += 1
    return duplicates, context_window


class SearchAiActArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)


class SearchAiActTool:
    name = "search_ai_act"
    args_model = SearchAiActArgs
    max_calls = AGENT_SEARCH_AI_ACT_MAX_CALLS

    def __init__(self, pipeline: RagPipeline, system_prompts: Dict[str, str]):
        self._pipeline = pipeline
        self._system_prompts = system_prompts

    async def invoke(
        self, args: SearchAiActArgs, ctx: ToolContext
    ) -> AsyncIterator[ToolEvent]:
        language = ctx.language
        texts = TEXTS[language]
        system_prompt = self._system_prompts[language]
        nested_step: Optional[str] = None
        retrieved: List[Source] = []

        yield ToolSubstep(
            label=texts["substep_embedding"][0],
            detail=texts["substep_embedding"][1].format(embedding_model=DEFAULT_MODEL),
        )

        async for event in self._pipeline.run(
            dataset_id=ctx.dataset,
            query=args.query,
            prompt_template=texts["user"],
            request_id=ctx.run_id,
            system_prompt=system_prompt,
            token_limit=AGENT_SEARCH_AI_ACT_TOKEN_LIMIT,
        ):
            if isinstance(event, SourcesRetrieved):
                retrieved = event.sources
                yield RagSources(sources=event.sources, duration=event.energy.duration)

                used = used_sources(retrieved)
                used_tokens = sum(source.num_tokens for source in used)
                top_n = len(retrieved)
                relevant = sum(len(source.relevantChunks) for source in retrieved)
                duplicates, context_window = _skip_counts(retrieved)
                token_limit = (
                    AGENT_SEARCH_AI_ACT_TOKEN_LIMIT
                    if AGENT_SEARCH_AI_ACT_TOKEN_LIMIT is not None
                    else CONTEXT_WINDOW - PROMPT_BUFFER
                )
                yield ToolSubstep(
                    label=texts["substep_search"][0],
                    detail=texts["substep_search"][1].format(
                        corpus=self._pipeline.index_service.chunk_count(ctx.dataset),
                        top_n=top_n,
                    ),
                )
                yield ToolSubstep(
                    label=texts["substep_crossrefs"][0],
                    detail=texts["substep_crossrefs"][1].format(relevant=relevant),
                )
                yield ToolSubstep(
                    label=texts["substep_context"][0],
                    detail=texts["substep_context"][1].format(
                        used=len(used),
                        total=top_n + relevant,
                        used_tokens=used_tokens,
                        token_limit=token_limit,
                        duplicates=duplicates,
                        context_window=context_window,
                    ),
                )
            elif isinstance(event, PromptBuilt):
                prompt_tokens = TOKENIZER.count_tokens(
                    "\n\n".join(message.content for message in event.messages)
                )
                yield ToolSubstep(
                    label=texts["substep_prompt"][0],
                    detail=texts["substep_prompt"][1].format(
                        used=len(used), prompt_tokens=prompt_tokens
                    ),
                )
                yield ToolSubstep(
                    label=texts["substep_answer"][0],
                    detail=texts["substep_answer"][1].format(
                        llm_model=self._pipeline.llm_client.model
                    ),
                )
                nested_step = ctx.new_step()
                yield NestedEvent(
                    nested_step,
                    SystemPrompt(
                        prompt_key=f"search_ai_act_{language}", text=system_prompt
                    ),
                )
                yield NestedEvent(
                    nested_step,
                    LlmRequest(
                        model=self._pipeline.llm_client.model,
                        messages=[message.model_dump() for message in event.messages],
                        tools=[],
                    ),
                )
                yield NestedEvent(
                    nested_step,
                    QueueStatus(llm_calls_in_flight=llm_calls.in_flight()),
                )
            elif isinstance(event, ReasoningChunk):
                yield NestedEvent(nested_step, ReasoningDelta(text=event.text))
            elif isinstance(event, ContentChunk):
                yield NestedEvent(nested_step, ContentDelta(text=event.text))
            elif isinstance(event, GenerationFinished):
                result = event.result
                duration = event.duration
                yield NestedEvent(
                    nested_step,
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
                )

                used = used_sources(retrieved)
                if used:
                    sources_text = "\n".join(
                        [texts["sources_used"]] + [f"- {source.title}" for source in used]
                    )
                else:
                    sources_text = f"{texts['sources_used']} {texts['no_sources']}"
                yield ToolResult(text=f"{result.content.strip()}\n\n{sources_text}")
