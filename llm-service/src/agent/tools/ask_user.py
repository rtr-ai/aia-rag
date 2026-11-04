from typing import AsyncIterator, List, Optional

from agent.registry import NeedsUserInput, ToolContext, ToolEvent
from pydantic import BaseModel, ConfigDict, Field, field_validator

ASK_USER = "ask_user"
MAX_OPTIONS = 5


class AskUserArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    options: List[str] = []

    @field_validator("question")
    @classmethod
    def clean_question(cls, question: str) -> str:
        stripped = question.strip()
        if not stripped:
            raise ValueError("question must not be empty")
        return stripped

    @field_validator("options", mode="before")
    @classmethod
    def none_to_empty(cls, options: Optional[List[str]]) -> List[str]:
        return [] if options is None else options

    @field_validator("options")
    @classmethod
    def clean_options(cls, options: List[str]) -> List[str]:
        stripped = [option.strip() for option in options]
        return [option for option in stripped if option][:MAX_OPTIONS]


class AskUserTool:
    name = ASK_USER
    args_model = AskUserArgs
    max_calls = 1

    async def invoke(
        self, args: AskUserArgs, ctx: ToolContext
    ) -> AsyncIterator[ToolEvent]:
        yield NeedsUserInput(question=args.question, options=args.options)
