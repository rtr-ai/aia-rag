import asyncio
import os
import secrets
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

from agent.llm_client import Message
from models.agent_events import Energy
from utils.logger import get_logger

AGENT_RESUME_TTL = int(os.getenv("AGENT_RESUME_TTL", "900"))
AGENT_RESUME_CLEANUP_INTERVAL = int(os.getenv("AGENT_RESUME_CLEANUP_INTERVAL", "60"))

LOGGER = get_logger(__name__)


@dataclass
class PausedRun:
    run_id: str
    dataset: str
    language: str
    prompt: str
    messages: List[Message]
    turns: int
    call_counts: Dict[str, int]
    seq: int
    step_count: int
    ask_step_id: str
    ask_tool_call_id: str
    ask_tool_name: str
    ask_started: float
    energy: Optional[Energy]
    elapsed: float


class RunStore(Protocol):
    def save(self, run: PausedRun) -> str: ...

    def take(self, token: str) -> Optional[PausedRun]: ...


class InMemoryRunStore:
    def __init__(self):
        self._runs: Dict[str, Tuple[float, PausedRun]] = {}

    def save(self, run: PausedRun) -> str:
        token = secrets.token_urlsafe(32)
        self._runs[token] = (time.monotonic() + AGENT_RESUME_TTL, run)
        return token

    def take(self, token: str) -> Optional[PausedRun]:
        entry = self._runs.pop(token, None)
        if entry is None:
            return None
        expires_at, run = entry
        if expires_at <= time.monotonic():
            return None
        return run

    def remove_expired(self) -> None:
        now = time.monotonic()
        expired = [
            token for token, (expires_at, _) in self._runs.items() if expires_at <= now
        ]
        for token in expired:
            _, run = self._runs.pop(token)
            LOGGER.info(f"[{run.run_id}] Paused run expired, removed from store")

    async def run_cleanup(self) -> None:
        while True:
            await asyncio.sleep(AGENT_RESUME_CLEANUP_INTERVAL)
            try:
                self.remove_expired()
            except Exception:
                LOGGER.exception("Removing expired paused runs failed")
