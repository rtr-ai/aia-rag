import os
from typing import List
from pydantic import TypeAdapter
from services.dataset_configuration import DatasetConfiguration
from models.sources import Source
from utils.logger import get_logger


LOGGER = get_logger(__name__)


ORDER_CHUNKS_FROM_SOURCE = TypeAdapter(bool).validate_python(
    os.getenv("ORDER_CHUNKS_FROM_SOURCE", "false")
)


def generate_prompt(
    prompt: str, sources: List[Source], dataset_id: str, config: DatasetConfiguration
) -> str:
    combined_sources = []  # flatten

    chunks = ""
    for source in sources:
        if not source.skip:
            combined_sources.append(source)
        for relevant_chunk in source.relevantChunks:
            if not relevant_chunk.skip:
                combined_sources.append(relevant_chunk)

    # sort
    if ORDER_CHUNKS_FROM_SOURCE:
        combined_sources.sort(key=lambda x: x.position)

    # output
    for source in combined_sources:
        chunks += f"Titel: {source.title} \n{source.content}\n"
        chunks += "\n"

    system_prompt = config.get_prompt(dataset_id)

    final_prompt = system_prompt.format(context_str=chunks.strip(), query_str=prompt)
    return final_prompt
