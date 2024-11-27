from typing import List
from pydantic import BaseModel, RootModel


class Source(BaseModel):
    content: str
    score: float
    title: str = ""


class SourceList(RootModel[List[Source]]):
    root: List[Source]
