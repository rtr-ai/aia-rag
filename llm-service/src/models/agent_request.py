from typing import Optional

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    language: str = Field(min_length=1, max_length=10)
    frc_captcha_solution: Optional[str] = None


class ResumeRequest(BaseModel):
    resume_token: str = Field(min_length=1, max_length=100)
    answer: str = Field(min_length=1, max_length=500)
