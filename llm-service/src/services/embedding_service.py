import os
from typing import List
from ollama import AsyncClient

DEFAULT_MODEL = os.getenv("EMBEDDING_MODELS", "bge-m3").split(",")[0]


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

    async def generate_embeddings_batch(self, input: List[str]) -> dict:
        passage_prefix = ""
        if "multilingual-e5" in DEFAULT_MODEL:
            passage_prefix = "passage: "
        prefixed_input = [f"{passage_prefix}{text}" for text in input]
        response = await self.client.embed(model=DEFAULT_MODEL, input=prefixed_input)

        return response["embeddings"]
