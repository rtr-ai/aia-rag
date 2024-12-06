from typing import List
from pydantic import BaseModel, RootModel

class RelevantChunk(BaseModel):
    id: str
    title: str

class Source(BaseModel):
    content: str
    score: float
    title: str = ""
    relevantChunks: List[RelevantChunk] = []


class SourceList(RootModel[List[Source]]):
    root: List[Source]
