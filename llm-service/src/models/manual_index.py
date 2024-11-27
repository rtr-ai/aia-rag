from typing import List, Union
from pydantic import BaseModel


class ChunkNode(BaseModel):
    id: str
    title: str
    content: str
    keywords: List[str]
    negativeKeywords: List[str]
    relevantChunksIds: List[str]
    parameters: List[str]


class TextNode(BaseModel):
    id: str
    content: str


class ManualIndex(BaseModel):
    id: str
    creation_date: int = 0
    last_updated: int = 0
    chunks: List[Union[ChunkNode, TextNode, str]]
