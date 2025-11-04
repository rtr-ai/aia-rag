import os
from typing import Optional
from pydantic import BaseModel, Field

DEFAULT_MODEL = os.getenv("LLM_MODELS", "llama3.1:8b-instruct-fp16").split(",")[0]


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    frc_captcha_solution: Optional[str] = None
