import json
import os
import secrets
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
from services.power_meter_service import PowerMeterService, PhasePowerMeter
from services.matomo_tracking_service import matomo_service
from services.dataset_configuration import DatasetConfiguration

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
        yield f"data: {json.dumps({'content': '', 'type': 'heartbeat'})}\n\n"
        yield f"data: {json.dumps({'type': 'queue_position', 'content': {'position': queue_position}})}\n\n"
        request_id = str(secrets.token_hex(8))
        phases = None
        try:
            LOGGER.debug(f"[{request_id}]   Prompting: <{request.prompt}>")
            self.model = DEFAULT_MODEL
            meter = PowerMeterService()
            phases = PhasePowerMeter(meter)
            chunks = []
            embedding_model = None
            embedding_query_prefix = None
            embedding_query_text = None
            if request.skip_retrieval:
                rerank_metadata = {
                    "requested": False, "enabled": False, "applied": False,
                    "reason": "retrieval_skipped", "status": "skipped", "duration_seconds": 0.0,
                }
                prompt = request.final_prompt
            else:
                index_power_usage = meter.get_initial_power_consumption()
                yield f"data: {json.dumps({'type': 'power_index', 'content': index_power_usage})}\n\n"
                phases.switch("retrieval")
                chunks, _, rerank_metadata = await self.index_service.query_index(
                    dataset_id=request.dataset, query=request.prompt,
                    request_id=request_id, use_rerank=request.use_rerank,
                    phase_meter=phases,
                )
                embedding_service = self.index_service.embedding_service
                embedding_model = embedding_service.model_for_dataset(request.dataset)
                embedding_query_prefix = embedding_service.query_prefix_for_model(embedding_model)
                embedding_query_text = embedding_service.query_input_for_model(request.prompt, embedding_model)
                if request.generate_answer:
                    prompt = generate_prompt(
                        prompt=request.prompt, sources=chunks,
                        dataset_id=request.dataset, config=config,
                    )
                phases.stop()

            retrieval_power = phases.payload("retrieval", "skipped" if request.skip_retrieval else "completed")
            rerank_power = phases.payload("rerank", rerank_metadata.get("status", "not_run"))
            rerank_metadata["duration_seconds"] = rerank_power["duration"]
            metadata = {
                "request_id": request_id, "dataset": request.dataset,
                "llm_model": self.model, "embedding_model": embedding_model,
                "embedding_query_prefix": embedding_query_prefix,
                "embedding_query_text": embedding_query_text,
                "generate_answer": request.generate_answer,
                "llm_used": request.generate_answer, "retrieval_skipped": request.skip_retrieval,
                "temperature": TEMPERATURE, "context_window": CONTEXT_WINDOW,
                "prompt_buffer": int(os.getenv("PROMPT_BUFFER", "1500")),
                "top_n_chunks": int(os.getenv("TOP_N_CHUNKS", "15")),
                "rerank_top_n": int(os.getenv("RERANK_TOP_N", "25")),
                "rerank": rerank_metadata, "measurement_version": 2,
            }
            yield f"data: {json.dumps({'type': 'metadata', 'content': metadata})}\n\n"
            for event_type, content in (("power_prompt", retrieval_power), ("power_rerank", rerank_power)):
                matomo_service.track_event(action=event_type, request_id=request_id, value=content)
                yield f"data: {json.dumps({'type': event_type, 'content': content})}\n\n"
            async for part in self.__yield_sources__(sources=chunks, request_id=request_id):
                yield part
            if not request.generate_answer:
                content = phases.payload("generation", "not_run")
                yield f"data: {json.dumps({'type': 'power_response', 'content': content})}\n\n"
                LOGGER.info(f"[{request_id}]   Retrieval-only request completed without calling the LLM")
                return

            matomo_service.track_event(action="user", request_id=request_id, value=request.prompt)
            yield f"data: {json.dumps({'content': prompt, 'type': 'user'})}\n\n"
            LOGGER.debug(f"[{request_id}]   Prompting Ollama")
            response = ""
            phases.switch("generation")
            async for part in self.prompt_ollama(prompt):
                if "message" in part:
                    message_part = part["message"]["content"]
                    response += message_part
                    yield f"data: {json.dumps({'content': message_part, 'type': 'assistant'})}\n\n"
            phases.stop()
            response_power = phases.payload("generation")
            matomo_service.track_event(action="assistant", request_id=request_id, value=response)
            final_log = f"User Prompt: {request.prompt}\n\n\nDataset: {request.dataset}LLM Response: {response}\n\n\n"
            LOGGER.debug(f"[{request_id}] {final_log}")
            LOGGER_CHAT.info(f"[{request_id}] {request.prompt}\n\n")
            LOGGER_CHAT.info(f"[{request_id}] {response}\n\n\n")
            matomo_service.track_event(action="power_response", request_id=request_id, value=response_power)
            yield f"data: {json.dumps({'type': 'power_response', 'content': response_power})}\n\n"
        except HTTPException as e:
            yield f"data: {json.dumps({'content': str(e.detail), 'type': 'error'})}\n\n"
        except Exception as e:
            LOGGER.exception(f"[{request_id}]   Chat stream failed: {e}")
            yield f"data: {json.dumps({'content': 'Backend error while generating the answer.', 'type': 'error'})}\n\n"
        finally:
            if phases is not None:
                phases.stop()

    async def prompt_ollama(self, prompt: str):
        message = {"role": "user", "content": prompt}
        async for part in await self.client.chat(
            model=self.model, messages=[message],
            options={"temperature": TEMPERATURE, "num_ctx": CONTEXT_WINDOW}, stream=True,
        ):
            yield part

    async def __yield_sources__(self, sources: List[Source], request_id: str):
        sources_json = SourceList(root=sources).model_dump_json()
        data = json.dumps({"content": sources_json, "type": "sources"})
        matomo_service.track_event(action="sources", request_id=request_id, value=sources_json)
        yield f"data: {data}\n\n"
