from typing import List, Union, Optional
from pydantic import BaseModel


class ChunkNode(BaseModel):
    id: str
    title: str
    content: str
    keywords: List[str]
    negativeKeywords: List[str]
    relevantChunksIds: List[str]
    parameters: List[str]
    position: int = -1


class TextNode(BaseModel):
    id: str
    content: str


class ManualIndex(BaseModel):
    id: str
    creation_date: int = 0
    last_updated: int = 0
    chunks: List[Union[ChunkNode, TextNode, str]]
