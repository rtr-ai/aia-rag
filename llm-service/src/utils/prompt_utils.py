from typing import List
from models.sources import Source

DEFAULT_PROMPT_RAG: str = """
Informationen aus dem EU AI Act sind unten angeführt.
{context_str}
Basierend auf den Informationen und ohne Vorwissen beantworte die Anfrage. Antworte faktenbasiert und ohne Konklusionen.
Antworte in der selben Sprache, in der die Anfrage geschrieben ist.
Anfrage: {query_str}
Antwort:
"""


def generate_prompt(prompt: str, sources: List[Source]) -> str:
    chunks = "\n".join([f"Score: {node.score}\n{node.content}" for node in sources])
    final_prompt = DEFAULT_PROMPT_RAG.format(context_str=chunks, query_str=prompt)
    return final_prompt
