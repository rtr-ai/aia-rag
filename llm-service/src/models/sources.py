from typing import List, Optional
from pydantic import BaseModel, RootModel


class RelevantChunk(BaseModel):
    id: str
    title: str
    content: str
    num_tokens: int
    score: Optional[float] = None
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None
    retrieval_position: Optional[int] = None
    rerank_position: Optional[int] = None
    vector_retrieval_position: Optional[int] = None
    skip: bool
    position: int
    skip_reason: str = ""


class Source(BaseModel):
    content: str
    score: float
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None
    retrieval_position: Optional[int] = None
    rerank_position: Optional[int] = None
    vector_retrieval_position: Optional[int] = None
    title: str = ""
    relevantChunks: List[RelevantChunk] = []
    num_tokens: int
    skip: bool
    position: int
    skip_reason: str = ""


class SourceList(RootModel[List[Source]]):
    root: List[Source]
