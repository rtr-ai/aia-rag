from typing import List
from models.sources import Source

DEFAULT_PROMPT_RAG: str = """
Sie sind Senior-Partner in einer östereichischen Großkanzlei.
Sie beantworten Mandantenanfragen höflich, präzise und genau. Sie verwenden ausschließlich die Ihnen übergebenen Quellen um Ihre Subsumptionen durchzuführen.
Folgende Kundenanfrage ist bei Ihnen gestern eingegangen:
{query_str}
Sie verwenden dazu folgende Quellen:
Quellen: {context_str}
Antwort:
"""


def generate_prompt(prompt: str, sources: List[Source]) -> str:
    chunks = "\n".join([f"Score: {node.score}\n{node.content}" for node in sources])
    final_prompt = DEFAULT_PROMPT_RAG.format(context_str=chunks, query_str=prompt)
    return final_prompt
