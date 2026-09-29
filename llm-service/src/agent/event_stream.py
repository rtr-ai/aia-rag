import json
from typing import Optional

from pydantic import BaseModel

PROTOCOL_VERSION = 1


class EventStream:
    def __init__(self, run_id: str, seq: int = 0, step_count: int = 0):
        self.run_id = run_id
        self.seq = seq
        self.step_count = step_count

    def new_step(self) -> str:
        self.step_count += 1
        return f"s{self.step_count}"

    def emit(
        self,
        payload: BaseModel,
        step_id: Optional[str] = None,
        parent_step_id: Optional[str] = None,
    ) -> str:
        self.seq += 1
        envelope = {
            "v": PROTOCOL_VERSION,
            "seq": self.seq,
            "run_id": self.run_id,
            "step_id": step_id,
            "parent_step_id": parent_step_id,
            "type": payload.EVENT_TYPE,
            "payload": payload.model_dump(mode="json"),
        }
        return f"data: {json.dumps(envelope)}\n\n"
