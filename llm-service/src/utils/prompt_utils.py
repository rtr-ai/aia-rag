from typing import List
from models.sources import Source

DEFAULT_PROMPT_RAG: str = """
Informationen aus dem EU AI Act sind unten angeführt.
{context_str}
Basierend auf den Informationen und ohne Vorwissen beantworte die Anfrage. Antworte faktenbasiert und ohne Konklusionen.
Antworte in der selben Sprache, in der die Anfrage geschrieben ist.
Falls diese Begriffe in deiner Antwort vorkommen, beachte das folgende Wörterbuch/Glossar: der EU AI Act, das LLM, das Large Language Modell, die KI, die DSGVO, der AI Act.
Anfrage: {query_str}
Antwort:
"""


def generate_prompt(prompt: str, sources: List[Source]) -> str:
    chunks = ""
    for source in sources:
        chunks += f"Score: {source.score}\n{source.content}\n"
        for relevant_chunk in source.relevantChunks:
            chunks += f"{relevant_chunk.content}\n"
        chunks += "\n"
    
    final_prompt = DEFAULT_PROMPT_RAG.format(context_str=chunks.strip(), query_str=prompt)
    return final_prompt