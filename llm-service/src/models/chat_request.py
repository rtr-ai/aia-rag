import os
from typing import Optional
from pydantic import BaseModel, Field, model_validator

DEFAULT_MODEL = os.getenv("LLM_MODELS", "llama3.1:8b-instruct-fp16").split(",")[0]


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    dataset: str = Field(min_length=1, max_length=500, default="ai_act_de")
    frc_captcha_solution: Optional[str] = None
    use_rerank: bool = False
    generate_answer: bool = True
    skip_retrieval: bool = False
    final_prompt: Optional[str] = Field(default=None, min_length=1, max_length=100000)

    @model_validator(mode="after")
    def validate_final_prompt_mode(self):
        if self.skip_retrieval and not self.final_prompt:
            raise ValueError("final_prompt is required when skip_retrieval is true")
        if self.skip_retrieval and not self.generate_answer:
            raise ValueError("generate_answer must be true when skip_retrieval is true")
        if self.final_prompt and not self.skip_retrieval:
            raise ValueError("skip_retrieval must be true when final_prompt is provided")
        return self
