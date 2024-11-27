import os
from typing import List
from ollama import AsyncClient

DEFAULT_MODEL = os.getenv("EMBEDDING_MODELS", "bge-m3").split(",")[0]


class EmbeddingService:
    def __init__(self):
        self.indices = {}
        self.client = AsyncClient(host=os.getenv("OLLAMA_HOST"))

    async def generate_embedding(self, input: str) -> dict:
        response = await self.client.embed(model=DEFAULT_MODEL, input=input)

        return response["embeddings"]

    async def generate_embeddings_batch(self, input: List[str]) -> dict:
        response = await self.client.embed(model=DEFAULT_MODEL, input=input)

        return response["embeddings"]
