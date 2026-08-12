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
from services.dataset_configuration import DatasetConfiguration
import secrets

STORAGE_PATH = os.path.join(path_utils.get_project_root(), "data", "indices")
LOGGER = get_logger(__name__)
LOGGER_CHAT = get_logger("chat", "chat.txt")
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
            self.model = DEFAULT_MODEL
            meter = PowerMeterService()
            chunks = []

            if request.skip_retrieval:
                rerank_metadata = {
                    "requested": False,
                    "enabled": False,
                    "applied": False,
                    "reason": "retrieval_skipped",
                }
                prompt = request.final_prompt
                measurement = None
                final_duration = 0.0
            else:
                meter.start()

                index_power_usage = meter.get_initial_power_consumption()
                LOGGER.debug(
                    f"[{request_id}]    Initial index power usage: {index_power_usage}"
                )
                data = {"type": "power_index", "content": index_power_usage}

                yield f"data: {json.dumps(data)}\n\n"
                chunks, duration, rerank_metadata = await self.index_service.query_index(
                    dataset_id=request.dataset,
                    query=request.prompt,
                    request_id=request_id,
                    use_rerank=request.use_rerank,
                )
                measurement = meter.stop()
                final_duration = (
                    duration if duration else measurement.duration_seconds
                )

            metadata = {
                "request_id": request_id, "dataset": request.dataset,
                "llm_model": self.model,
                "embedding_model": (
                    None
                    if request.skip_retrieval
                    else self.index_service.embedding_service.model_for_dataset(
                        request.dataset
                    )
                ),
                "generate_answer": request.generate_answer,
                "llm_used": request.generate_answer,
                "retrieval_skipped": request.skip_retrieval,
                "temperature": TEMPERATURE,
                "context_window": CONTEXT_WINDOW,
                "prompt_buffer": int(os.getenv("PROMPT_BUFFER", "1500")),
                "top_n_chunks": int(os.getenv("TOP_N_CHUNKS", "15")),
                "rerank_top_n": int(os.getenv("RERANK_TOP_N", "25")),
                "rerank": rerank_metadata,
            }
            yield f"data: {json.dumps({'type': 'metadata', 'content': metadata})}\n\n"

            async for part in self.__yield_sources__(
                sources=chunks, request_id=request_id
            ):
                yield part
            if not request.generate_answer:
                LOGGER.info(
                    f"[{request_id}]   Retrieval-only request completed without "
                    "calling the LLM"
                )
                return


            if not request.skip_retrieval:
                data = {
                    "type": "power_prompt",
                    "content": {
                        "cpu_kWh": (
                            measurement.cpu_watts * final_duration / 3600 / 1000
                        ),
                        "gpu_kWh": (
                            measurement.gpu_watts * final_duration / 3600 / 1000
                        ),
                        "ram_kWh": (
                            measurement.ram_watts * final_duration / 3600 / 1000
                        ),
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
            LOGGER.debug(f"[{request_id}]   Prompting Ollama")
            response = ""
            meter.start()
            power_samples = []
            ollama_duration = 0.0

            async for part in self.prompt_ollama(prompt):
                power_samples.append(meter.sample_power())
                # LOGGER.debug(f"Ollama part: {part}")
                ollama_duration = (
                    sum(
                        getattr(part, attr)
                        for attr in [
                            "load_duration",
                            "eval_duration",
                            "prompt_eval_duration",
                        ]
                    )
                    / 1_000_000_000
                    if all(
                        getattr(part, attr, None) is not None
                        for attr in [
                            "load_duration",
                            "eval_duration",
                            "prompt_eval_duration",
                        ]
                    )
                    else None
                )
                if ollama_duration:
                    LOGGER.debug(
                        f"""Total completion duration from Ollama {ollama_duration}"""
                    )

                if "message" in part:
                    message_part = part["message"]["content"]
                    response += message_part
                    data = json.dumps({"content": message_part, "type": "assistant"})
                    yield f"data: {data}\n\n"
            measurement = meter.stop()
            final_duration = (
                ollama_duration if ollama_duration else measurement.duration_seconds
            )
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
        except Exception as e:
            LOGGER.exception(f"[{request_id}]   Chat stream failed: {e}")
            data = json.dumps({"content": "Backend error while generating the answer.", "type": "error"})
            yield f"data: {data}\n\n"

    async def prompt_ollama(self, prompt: str):
        message = {"role": "user", "content": prompt}
        async for part in await self.client.chat(
            model=self.model,
            messages=[message],
            options={"temperature": TEMPERATURE, "num_ctx": CONTEXT_WINDOW},
            stream=True,
        ):
            yield part

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
