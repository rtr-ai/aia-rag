from typing import Optional

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    dataset: str = Field(min_length=1, max_length=500, default="ai_act_de")
    frc_captcha_solution: Optional[str] = None
