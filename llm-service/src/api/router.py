import json
from fastapi import APIRouter, HTTPException, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from services.chat_service import ChatService
from models.chat_request import ChatRequest
from utils.logger import get_logger
from uvicorn.protocols.utils import ClientDisconnected

router = APIRouter()
LOGGER = get_logger(__name__)

chat_service = ChatService()


@router.post("/chat")
async def chat(req: ChatRequest):
    try:
        return StreamingResponse(
            chat_service.chat(req),
            media_type="text/event-stream",
        )
    except (ClientDisconnected, WebSocketDisconnect):
        LOGGER.info("Client disconnected. Stopping the stream.")
    except Exception as e:
        LOGGER.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
