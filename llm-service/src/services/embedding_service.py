import os
from typing import List
from ollama import AsyncClient
from utils.logger import get_logger

DEFAULT_MODEL = os.getenv("EMBEDDING_MODELS", "bge-m3").split(",")[0]
LOGGER = get_logger(__name__)


class EmbeddingService:
    def __init__(self):
        self.indices = {}
        self.client = AsyncClient(host=os.getenv("OLLAMA_HOST"))

    async def generate_embedding(self, input: str) -> dict:
        query_prefix = ""
        if "multilingual-e5" in DEFAULT_MODEL:
            query_prefix = "query: "
        response = await self.client.embed(model=DEFAULT_MODEL, input=f"{query_prefix}{input}")

        return response["embeddings"]

    async def generate_embeddings_batch(self, input: List[str], batch_size: int = 10) -> dict:
        passage_prefix = ""
        
        if "multilingual-e5" in DEFAULT_MODEL:
            passage_prefix = "passage: "
        prefixed_input = [f"{passage_prefix}{text}" for text in input]
        batches = [prefixed_input[i:i + batch_size] for i in range(0, len(prefixed_input), batch_size)]
        all_embeddings = []
        for index, batch in enumerate(batches):
            LOGGER.debug(f"Processing embeddings batch <{index+1}> of <{len(batches)}>")
            response = await self.client.embed(model=DEFAULT_MODEL, input=batch)
            all_embeddings.extend(response["embeddings"])
            
        return all_embeddings
