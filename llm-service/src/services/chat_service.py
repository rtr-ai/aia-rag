import json
import os
from typing import AsyncGenerator, List
from fastapi import HTTPException
from agent.llm_client import ContentChunk, Message, Timings, TurnResult
from agent.llm_provider import create_llm_client
from services.embedding_service import EmbeddingService
from utils import path_utils
from services.index_service import IndexService
from utils.logger import get_logger
from models.chat_request import ChatRequest
from models.sources import Source, SourceList
from utils.prompt_utils import generate_prompt
from services.power_meter_service import PowerMeterService
from services.matomo_tracking_service import matomo_service
from services.dataset_configuration import DatasetConfiguration
import secrets

STORAGE_PATH = os.path.join(path_utils.get_project_root(), "data", "indices")
LOGGER = get_logger(__name__)
LOGGER_CHAT = get_logger("chat", "chat.txt")


class ChatService:
    def __init__(self):
        self.indices = {}
        self.embedding_service = EmbeddingService()
        self.index_service = IndexService()
        self.llm_client = create_llm_client()

    async def chat(
        self, request: ChatRequest, queue_position: int, config: DatasetConfiguration
    ) -> AsyncGenerator[str, None]:
        data = json.dumps({"content": "", "type": "heartbeat"})
        yield f"data: {data}\n\n"
        data = {
            "type": "queue_position",
            "content": {"position": queue_position},
        }
        yield f"data: {json.dumps(data)}\n\n"
        request_id = str(secrets.token_hex(8))
        try:

            LOGGER.debug(f"[{request_id}]   Prompting: <{request.prompt}>")
            meter = PowerMeterService()
            meter.start()

            index_power_usage = meter.get_initial_power_consumption()
            LOGGER.debug(
                f"[{request_id}]    Initial index power usage: {index_power_usage}"
            )
            data = {"type": "power_index", "content": index_power_usage}

            yield f"data: {json.dumps(data)}\n\n"
            chunks, duration = await self.index_service.query_index(
                dataset_id=request.dataset, query=request.prompt, request_id=request_id
            )
            measurement = meter.stop()
            final_duration = duration if duration else measurement.duration_seconds

            async for part in self.__yield_sources__(
                sources=chunks, request_id=request_id
            ):
                yield part

            data = {
                "type": "power_prompt",
                "content": {
                    "cpu_kWh": (measurement.cpu_watts * final_duration / 3600 / 1000),
                    "gpu_kWh": (measurement.gpu_watts * final_duration / 3600 / 1000),
                    "ram_kWh": (measurement.ram_watts * final_duration / 3600 / 1000),
                    "total_kWh": (
                        measurement.total_watts * final_duration / 3600 / 1000
                    ),
                    "duration": final_duration,
                },
            }
            matomo_service.track_event(
                action=data["type"], request_id=request_id, value=data["content"]
            )
            yield f"data: {json.dumps(data)}\n\n"
            LOGGER.debug(
                f"[{request_id}]    Power consumption for generating prompt: {data}"
            )

            prompt = generate_prompt(
                prompt=request.prompt,
                sources=chunks,
                dataset_id=request.dataset,
                config=config,
            )
            data = json.dumps({"content": prompt, "type": "user"})
            matomo_service.track_event(
                action="user", request_id=request_id, value=request.prompt
            )
            yield f"data: {data}\n\n"
            LOGGER.debug(f"[{request_id}]   Prompting LLM")
            response = ""
            meter.start()
            power_samples = []
            timings = Timings()

            async for delta in self.llm_client.stream_turn(
                [Message(role="user", content=prompt)], []
            ):
                power_samples.append(meter.sample_power())
                if isinstance(delta, ContentChunk):
                    response += delta.text
                    data = json.dumps({"content": delta.text, "type": "assistant"})
                    yield f"data: {data}\n\n"
                elif isinstance(delta, TurnResult):
                    timings = delta.timings
            measurement = meter.stop()
            provider_durations = [
                timings.load_duration,
                timings.prompt_eval_duration,
                timings.eval_duration,
            ]
            known = [d for d in provider_durations if d is not None]
            provider_duration = sum(known) if len(known) == len(provider_durations) else 0.0
            final_duration = provider_duration or measurement.duration_seconds
            median_measurement = meter.get_median_power(power_samples)
            matomo_service.track_event(
                action="assistant", request_id=request_id, value=response
            )
            final_log = f"User Prompt: {request.prompt}\n\n\n"
            final_log += f"Dataset: {request.dataset}"
            final_log += f"LLM Response: {response}\n\n\n"
            LOGGER.debug(f"[{request_id}] {final_log}")
            LOGGER_CHAT.info(f"[{request_id}] {request.prompt}\n\n")
            LOGGER_CHAT.info(f"[{request_id}] {response}\n\n\n")
            LOGGER.debug(
                f"[{request_id}]    Generating response: Median Power consumption over {final_duration:.2f} seconds:"
            )
            data = {
                "type": "power_response",
                "content": {
                    "cpu_kWh": (
                        median_measurement.cpu_watts * final_duration / 3600 / 1000
                    ),
                    "gpu_kWh": (
                        median_measurement.gpu_watts * final_duration / 3600 / 1000
                    ),
                    "ram_kWh": (
                        median_measurement.ram_watts * final_duration / 3600 / 1000
                    ),
                    "total_kWh": (
                        median_measurement.total_watts * final_duration / 3600 / 1000
                    ),
                    "duration": final_duration,
                },
            }
            matomo_service.track_event(
                action=data["type"], request_id=request_id, value=data["content"]
            )
            yield f"data: {json.dumps(data)}\n\n"

        except HTTPException as e:
            data = json.dumps({"content": f"{e.detail}", "type": "error"})
            yield f"data: {data}\n\n"

    async def __yield_sources__(self, sources: List[Source], request_id: str):
        sources_json = SourceList(root=sources).model_dump_json()
        data = json.dumps(
            {
                "content": sources_json,
                "type": "sources",
            }
        )
        matomo_service.track_event(
            action="sources",
            request_id=request_id,
            value=sources_json,
        )
        yield f"data: {data}\n\n"
