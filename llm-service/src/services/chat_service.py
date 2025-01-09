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
from utils.prompt_utils import generate_prompt
from services.power_meter_service import PowerMeterService

STORAGE_PATH = os.path.join(path_utils.get_project_root(), "data", "indices")
LOGGER = get_logger(__name__)
DEFAULT_MODEL = os.getenv("LLM_MODELS", "llama3.1:8b-instruct-fp16").split(",")[0]
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "8000"))


class ChatService:
    def __init__(self):
        self.indices = {}
        self.model = DEFAULT_MODEL
        self.embedding_service = EmbeddingService()
        self.index_service = IndexService()
        self.client = AsyncClient(host=os.getenv("OLLAMA_HOST"))

    async def chat(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        data = json.dumps({"content": "", "type": "heartbeat"})
        try:
            yield f"data: {data}\n\n"
            LOGGER.debug(f"Prompting: <{request.prompt}>")
            self.model = request.model
            meter = PowerMeterService()
            meter.start()
            
            chunks = await self.index_service.query_index("main", query=request.prompt)
            LOGGER.debug(f"Generating embeddings: Power consumption over {measurement.duration_seconds:.2f} seconds:")
            measurement = meter.stop()
            LOGGER.debug(f"CPU: {measurement.cpu_watts:.2f} W")
            LOGGER.debug(f"GPU: {measurement.gpu_watts:.2f} W")
            LOGGER.debug(f"RAM: {measurement.ram_watts:.2f} W")
            LOGGER.debug(f"Total for generating embeddings: {measurement.total_watts:.2f} W")
            async for part in self.__yield_sources__(chunks):
                yield part

            prompt = generate_prompt(prompt=request.prompt, sources=chunks)
            data = json.dumps({"content": prompt, "type": "user"})
            yield f"data: {data}\n\n"
            LOGGER.debug("Prompting Ollama")
            response = ""
            meter = PowerMeterService()
            meter.start()
            async for part in self.prompt_ollama(prompt):
                response += part
                data = json.dumps({"content": part, "type": "assistant"})
                yield f"data: {data}\n\n"
            measurement = meter.stop()
            LOGGER.debug(f"Final response: {response}")
            LOGGER.debug(f"Generate response: Power consumption over {measurement.duration_seconds:.2f} seconds:")
            LOGGER.debug(f"CPU: {measurement.cpu_watts:.2f} W")
            LOGGER.debug(f"GPU: {measurement.gpu_watts:.2f} W")
            LOGGER.debug(f"RAM: {measurement.ram_watts:.2f} W")
            LOGGER.debug(f"Total for generating response: {measurement.total_watts:.2f} W")
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
