from typing import List
from pydantic import BaseModel, RootModel


class RelevantChunk(BaseModel):
    id: str
    title: str
    content: str
    num_tokens: int
    skip: bool
    position: int
    skip_reason: str = ""


class Source(BaseModel):
    content: str
    score: float
    title: str = ""
    relevantChunks: List[RelevantChunk] = []
    num_tokens: int
    skip: bool
    position: int
    skip_reason: str = ""


class SourceList(RootModel[List[Source]]):
    root: List[Source]
