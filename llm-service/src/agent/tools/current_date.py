from datetime import datetime
from typing import AsyncIterator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from agent.registry import AGENT_MAX_CALLS_PER_TOOL, ToolContext, ToolEvent
from models.agent_events import ToolResult
from pydantic import BaseModel, ConfigDict

TEXTS = {
    "de": {
        "weekdays": [
            "Montag",
            "Dienstag",
            "Mittwoch",
            "Donnerstag",
            "Freitag",
            "Samstag",
            "Sonntag",
        ],
        "result": "Heute ist {weekday}, {date} (Zeitzone {timezone}).",
    },
    "en": {
        "weekdays": [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
        "result": "Today is {weekday}, {date} (time zone {timezone}).",
    },
}


class CurrentDateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # required: mistral-small3.2 emits calls without arguments that the server drops
    timezone: str


class CurrentDateTool:
    name = "current_date"
    args_model = CurrentDateArgs
    max_calls = AGENT_MAX_CALLS_PER_TOOL

    async def invoke(
        self, args: CurrentDateArgs, ctx: ToolContext
    ) -> AsyncIterator[ToolEvent]:
        try:
            timezone = ZoneInfo(args.timezone)
        except (ZoneInfoNotFoundError, ValueError):
            yield ToolResult(
                text=(
                    f"Unknown time zone '{args.timezone}'. "
                    "Use an IANA time zone name such as 'Europe/Vienna'."
                ),
                is_error=True,
            )
            return
        now = datetime.now(timezone)
        texts = TEXTS[ctx.language]
        yield ToolResult(
            text=texts["result"].format(
                weekday=texts["weekdays"][now.weekday()],
                date=now.date().isoformat(),
                timezone=args.timezone,
            )
        )
