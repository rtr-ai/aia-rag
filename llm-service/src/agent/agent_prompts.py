import os
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel

from services.dataset_configuration import ConfigError

AGENT_PROMPT_PREFIX = os.getenv("AGENT_PROMPT_PREFIX", "agent_prompt")
AGENT_MAX_TURNS_PREFIX = os.getenv("AGENT_MAX_TURNS_PREFIX", "agent_max_turns")


class AgentPrompts(BaseModel):
    system: str
    max_turns_reached: str


def _read_prompt(prefix: str, language: str) -> str:
    file_path = Path("/app/data") / f"{prefix}_{language}.md"
    if not file_path.is_file():
        raise ConfigError(f"Agent prompt file does not exist: {file_path}")
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ConfigError(f"Agent prompt file is empty: {file_path}")
    return text


def load_agent_prompts(languages: List[str]) -> Dict[str, AgentPrompts]:
    return {
        language: AgentPrompts(
            system=_read_prompt(AGENT_PROMPT_PREFIX, language),
            max_turns_reached=_read_prompt(AGENT_MAX_TURNS_PREFIX, language),
        )
        for language in languages
    }
