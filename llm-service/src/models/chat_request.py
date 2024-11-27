import os
from pydantic import BaseModel

DEFAULT_MODEL = os.getenv("LLM_MODELS", "llama3.1:8b-instruct-fp16").split(",")[0]


class ChatRequest(BaseModel):
    prompt: str
    model: str = DEFAULT_MODEL
