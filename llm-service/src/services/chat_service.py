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
from services.matomo_tracking_service import matomo_service

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
            self.model = DEFAULT_MODEL
            meter = PowerMeterService()
            meter.start()

            index_power_usage = meter.get_initial_power_consumption()
            LOGGER.debug(f"Initial index power usage: {index_power_usage}")
            data = {"type": "power_index", "content": index_power_usage}

            yield f"data: {json.dumps(data)}\n\n"
            chunks = await self.index_service.query_index("main", query=request.prompt)
            measurement = meter.stop()

            async for part in self.__yield_sources__(chunks):
                yield part

            data = {
                "type": "power_prompt",
                "content": {
                    "cpu_kWh": (
                        measurement.cpu_watts
                        * measurement.duration_seconds
                        / 3600
                        / 1000
                    ),
                    "gpu_kWh": (
                        measurement.gpu_watts
                        * measurement.duration_seconds
                        / 3600
                        / 1000
                    ),
                    "ram_kWh": (
                        measurement.ram_watts
                        * measurement.duration_seconds
                        / 3600
                        / 1000
                    ),
                    "total_kWh": (
                        measurement.total_watts
                        * measurement.duration_seconds
                        / 3600
                        / 1000
                    ),
                    "duration": measurement.duration_seconds,
                },
            }
            matomo_service.track_event(action=data["type"], value=data["content"])
            yield f"data: {json.dumps(data)}\n\n"
            LOGGER.debug(f"Power consumption for generating prompt: {data}")

            prompt = generate_prompt(prompt=request.prompt, sources=chunks)
            data = json.dumps({"content": prompt, "type": "user"})
            matomo_service.track_event(action="user", value=request.prompt)
            yield f"data: {data}\n\n"
            LOGGER.debug("Prompting Ollama")
            response = ""
            meter.start()
            power_samples = []
            async for part in self.prompt_ollama(prompt):
                power_samples.append(meter.sample_power())
                response += part
                data = json.dumps({"content": part, "type": "assistant"})
                yield f"data: {data}\n\n"
            measurement = meter.stop()
            median_measurement = meter.get_median_power(power_samples)
            matomo_service.track_event(action="assistant", value=response)
            LOGGER.debug(f"Final response: {response}")
            LOGGER.debug(
                f"Generating response: Median Power consumption over {measurement.duration_seconds:.2f} seconds:"
            )
            data = {
                "type": "power_response",
                "content": {
                    "cpu_kWh": (
                        median_measurement.cpu_watts
                        * measurement.duration_seconds
                        / 3600
                        / 1000
                    ),
                    "gpu_kWh": (
                        median_measurement.gpu_watts
                        * measurement.duration_seconds
                        / 3600
                        / 1000
                    ),
                    "ram_kWh": (
                        median_measurement.ram_watts
                        * measurement.duration_seconds
                        / 3600
                        / 1000
                    ),
                    "total_kWh": (
                        median_measurement.total_watts
                        * measurement.duration_seconds
                        / 3600
                        / 1000
                    ),
                    "duration": measurement.duration_seconds,
                },
            }
            matomo_service.track_event(action=data["type"], value=data["content"])
            yield f"data: {json.dumps(data)}\n\n"

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
        sources_json = SourceList(root=sources).model_dump_json()
        data = json.dumps(
            {
                "content": sources_json,
                "type": "sources",
            }
        )
        matomo_service.track_event(
            action="sources",
            value=sources_json,
        )
        yield f"data: {data}\n\n"
