from typing import List
from models.sources import Source

DEFAULT_PROMPT_RAG: str = """
Informationen aus dem EU AI Act sind unten angeführt.
{context_str}
Aufgabe: Basierend auf den Informationen oben und ohne Vorwissen beantworte die Anfrage. Antworte faktenbasiert und ohne Konklusionen.
Antworte in der selben Sprache, in der die Anfrage geschrieben ist.
Falls diese Begriffe in deiner Antwort vorkommen, beachte das folgende Wörterbuch/Glossar: der EU AI Act, das LLM, das Large Language Modell, die KI, die DSGVO, der AI Act.
Beziehe dich in deiner Antwort immer auf die verwendeten Quellen. 

Beispiel 1: 

Die Veröffentlichung von KI-generierten Videos, sogenannten "Deepfakes" auf der eigenen Plattform unterliegen ab 2.8.2025 den Transparenzpflichten des AI-Acts. [1][2], [3]. Durch die Verwendung von KI-generierten Inhalten (Videos) sind sie nach Betreiber einzustufen und unterliegen Offenlegungspflichten gemäß dem AI Act. [4]

[1] Art 50 Abs 4 AIA
[2] KI-Servicestelle: Transparenzpflichten
[3] KI- Servicestelle: FAQ 
[4] KI- Servicestelle: Betreiberpflichten

Beispiel 2:
„Biometrische Daten“ sind mit speziellen technischen Verfahren gewonnene personenbezogene Daten zu den physischen, physiologischen oder verhaltenstypischen Merkmalen einer natürlichen Person, wie etwa Gesichtsbilder oder daktyloskopische Daten. [1] [2] Der Einsatz von KI-Systemen zur biometrischne Kategorisierung von natürlichen Personen ist verboten. [2][4]

[1] Art 3 Z 34 AIA
[2] Art 5 AIA
[3] ErwG 14
[4] ErwG 30

Beispiel 3:

Als Betreiber von KI-Systemen müssen sie Maßnahmen ergreifen um nach besten Kräften sicherzustellen, dass ihr Personal und andere Personen, die in ihrem Auftrag mit dem Betrieb und der Nutzung von KI-Systemen befasst sind, über ein ausreichendes Maß an KI-Kompetenz verfügen. Dabei sind besonders ihre technischen Kenntnise, ihre Erfahrung, ihre Ausbildung und Schulung und der Kontext, in dem die KI-Systeme eingesetzt werden sollen, sowie die Personen oder -gruppen bei denen die KI-Systeme eingesetzt werden sollen, zu berücksichtigen. [1],[2],[3],[4]

[1] Art 4 AIA
[2] Art 3 AIA
[3] KI-Servicestelle: KI-Kompetenz
[4] KI-Servicestelle: Risikostufen

Anfrage: {query_str}
Antwort:
"""


def generate_prompt(prompt: str, sources: List[Source]) -> str:
    chunks = ""
    for source in sources:
        chunks += f"Titel: {source.title} \n{source.content}\n"
        for relevant_chunk in source.relevantChunks:
            chunks += f"Titel: {source.title} \n{relevant_chunk.content}\n"
            chunks += "\n"
        chunks += "\n"
    
    final_prompt = DEFAULT_PROMPT_RAG.format(context_str=chunks.strip(), query_str=prompt)
    return final_prompt
