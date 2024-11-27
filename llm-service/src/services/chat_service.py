import json
import os
from typing import AsyncGenerator, List
from fastapi import HTTPException
from ollama import AsyncClient
from services.embedding_service import EmbeddingService
from utils import path_utils
from services.index_service import IndexService
from utils.logger import get_logger
from models.chat_request import ChatRequest
from models.sources import Source, SourceList

STORAGE_PATH = os.path.join(path_utils.get_project_root(), "data", "indices")
LOGGER = get_logger(__name__)
DEFAULT_MODEL = os.getenv("LLM_MODELS", "llama3.1:8b-instruct-fp16").split(",")[0]
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "8000"))


class ChatService:
    def __init__(self):
        self.indices = {}
        self.model = (DEFAULT_MODEL,)
        self.embedding_service = EmbeddingService()
        self.index_service = IndexService()
        self.client = AsyncClient(host=os.getenv("OLLAMA_HOST"))

    async def chat(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        data = json.dumps({"content": "", "type": "heartbeat"})
        try:
            yield f"data: {data}\n\n"
            LOGGER.debug(f"Prompting: <{request.prompt}>")
            self.model = request.model
            chunks = await self.index_service.query_index(
                "main", query=request.prompt, top_k=10
            )
            LOGGER.debug(f"Chunks are {chunks}")
            chunks_printable = "\n".join(
                [f"Score: {node.score}\n{node.content}" for node in chunks]
            )

            LOGGER.debug(f"Received chunks: <{chunks_printable}>")

            async for part in self.__yield_sources__(chunks):
                yield part

            LOGGER.debug("Prompting Ollama")
            response = ""
            async for part in self.prompt_ollama(request.prompt):
                response += part
                data = json.dumps({"content": part, "type": "assistant"})
                yield f"data: {data}\n\n"
            LOGGER.debug(f"Final response: {response}")
        except HTTPException as e:
            data = json.dumps({"content": f"{e.detail}", "type": "error"})
            yield f"data: {data}\n\n"

    async def prompt_ollama(self, prompt: str):
        message = {"role": "user", "content": prompt}
        async for part in await self.client.chat(
            model=self.model,
            messages=[message],
            options={"temperature": TEMPERATURE, "num_ctx": CONTEXT_WINDOW},
            stream=True,
        ):
            yield part["message"]["content"]

    async def __yield_sources__(self, sources: List[Source]):
        data = json.dumps(
            {
                "content": SourceList(root=sources).model_dump_json(),
                "type": "sources",
            }
        )
        yield f"data: {data}\n\n"
