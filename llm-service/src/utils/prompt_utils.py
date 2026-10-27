import os
from typing import List, Union
from pydantic import TypeAdapter
from models.sources import RelevantChunk, Source
from utils.logger import get_logger


LOGGER = get_logger(__name__)


ORDER_CHUNKS_FROM_SOURCE = TypeAdapter(bool).validate_python(
    os.getenv("ORDER_CHUNKS_FROM_SOURCE", "false")
)


def used_sources(sources: List[Source]) -> List[Union[Source, RelevantChunk]]:
    combined = []  # flatten

    for source in sources:
        if not source.skip:
            combined.append(source)
        for relevant_chunk in source.relevantChunks:
            if not relevant_chunk.skip:
                combined.append(relevant_chunk)

    # sort
    if ORDER_CHUNKS_FROM_SOURCE:
        combined.sort(key=lambda x: x.position)

    return combined


def render_prompt(template: str, sources: List[Source], query: str) -> str:
    chunks = ""
    for source in used_sources(sources):
        chunks += f"Titel: {source.title} \n{source.content}\n"
        chunks += "\n"

    return template.format(context_str=chunks.strip(), query_str=query)
