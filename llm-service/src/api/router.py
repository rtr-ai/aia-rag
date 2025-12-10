import asyncio
import inspect
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from services.chat_service import ChatService
from models.chat_request import ChatRequest
from utils.logger import get_logger
from utils.captcha import verify_captcha
from services.dataset_configuration import DatasetConfiguration
from typing import Protocol

router = APIRouter()
LOGGER = get_logger(__name__)
chat_service = ChatService()


class RequestCounter:
    def __init__(self):
        self.count = 0
        self.lock = asyncio.Lock()

    async def increment(self):
        async with self.lock:
            self.count += 1
            return self.count

    async def decrement(self):
        async with self.lock:
            self.count = max(0, self.count - 1)
            return self.count

    async def get_count(self):
        async with self.lock:
            return self.count


request_counter = RequestCounter()


class AppState(Protocol):
    dataset_config: DatasetConfiguration


def get_dataset_config(request: Request) -> DatasetConfiguration:
    state: AppState = request.app.state
    return state.dataset_config


async def sync_to_async_generator(gen):
    for item in gen:
        yield item
        await asyncio.sleep(0)


async def tracked_streaming(stream_gen):
    try:
        async for item in stream_gen:
            yield item
    finally:
        current_count = await request_counter.decrement()
        LOGGER.info(f"Stream completed. Active requests: {current_count}")


@router.post("/chat")
async def chat(
    req: ChatRequest, config: DatasetConfiguration = Depends(get_dataset_config)
):
    await verify_captcha(req)
    if not config.dataset_exists(req.dataset):
        LOGGER.error(f"Dataset not found: {req.dataset}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{req.dataset}' not found.",
        )

    current_count = await request_counter.increment()
    LOGGER.info(f"Starting chat request. Active requests: {current_count}")

    try:
        stream = chat_service.chat(req, current_count, config)
        is_async_gen = inspect.isasyncgen(stream) or hasattr(stream, "__aiter__")

        if not is_async_gen:
            LOGGER.debug(f"Converting sync generator to async")
            stream = sync_to_async_generator(stream)

        wrapped_stream = tracked_streaming(stream)

        return StreamingResponse(
            wrapped_stream,
            media_type="text/event-stream",
        )
    except Exception as e:
        await request_counter.decrement()
        LOGGER.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
