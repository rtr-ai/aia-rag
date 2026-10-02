import json
from typing import AsyncGenerator, List
import secrets
from fastapi import HTTPException
from agent.llm_client import ContentChunk
from agent.llm_provider import create_llm_client
from services.index_service import IndexService
from utils.logger import get_logger
from models.chat_request import ChatRequest
from models.sources import Source, SourceList
from services.power_meter_service import PowerMeterService
from services.matomo_tracking_service import matomo_service
from services.dataset_configuration import DatasetConfiguration
from services.rag_pipeline import (
    GenerationFinished,
    PromptBuilt,
    RagPipeline,
    SourcesRetrieved,
)

LOGGER = get_logger(__name__)
LOGGER_CHAT = get_logger("chat", "chat.txt")


class ChatService:
    def __init__(self):
        self.index_service = IndexService()
        self.llm_client = create_llm_client()
        self.rag_pipeline = RagPipeline(self.index_service, self.llm_client)

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

            index_power_usage = PowerMeterService().get_initial_power_consumption()
            LOGGER.debug(
                f"[{request_id}]    Initial index power usage: {index_power_usage}"
            )
            data = {"type": "power_index", "content": index_power_usage}

            yield f"data: {json.dumps(data)}\n\n"

            response = ""
            async for event in self.rag_pipeline.run(
                dataset_id=request.dataset,
                query=request.prompt,
                prompt_template=config.get_prompt(request.dataset),
                request_id=request_id,
            ):
                if isinstance(event, SourcesRetrieved):
                    async for part in self.__yield_sources__(
                        sources=event.sources, request_id=request_id
                    ):
                        yield part

                    data = {
                        "type": "power_prompt",
                        "content": event.energy.model_dump(),
                    }
                    matomo_service.track_event(
                        action=data["type"],
                        request_id=request_id,
                        value=data["content"],
                    )
                    yield f"data: {json.dumps(data)}\n\n"
                    LOGGER.debug(
                        f"[{request_id}]    Power consumption for generating prompt: {data}"
                    )
                elif isinstance(event, PromptBuilt):
                    data = json.dumps(
                        {"content": event.messages[-1].content, "type": "user"}
                    )
                    matomo_service.track_event(
                        action="user", request_id=request_id, value=request.prompt
                    )
                    yield f"data: {data}\n\n"
                    LOGGER.debug(f"[{request_id}]   Prompting LLM")
                elif isinstance(event, ContentChunk):
                    response += event.text
                    data = json.dumps({"content": event.text, "type": "assistant"})
                    yield f"data: {data}\n\n"
                elif isinstance(event, GenerationFinished):
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
                        f"[{request_id}]    Generating response: Median Power consumption over {event.energy.duration:.2f} seconds:"
                    )
                    data = {
                        "type": "power_response",
                        "content": event.energy.model_dump(),
                    }
                    matomo_service.track_event(
                        action=data["type"],
                        request_id=request_id,
                        value=data["content"],
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
