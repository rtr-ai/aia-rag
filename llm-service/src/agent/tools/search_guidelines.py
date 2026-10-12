import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import (
    AsyncIterator,
    Dict,
    List,
    Literal,
    Tuple,
    get_args,
)

from agent.registry import AGENT_MAX_CALLS_PER_TOOL, ToolContext, ToolEvent
from models.agent_events import (
    GuidelineSource,
    GuidelineSources,
    ToolResult,
    ToolSubstep,
)
from pydantic import BaseModel, ConfigDict, Field
from services.dataset_configuration import ConfigError
from services.embedding_service import DEFAULT_MODEL
from services.index_service import IndexService
from services.tokenizer_service import TokenizerService
from utils.logger import get_logger

LOGGER = get_logger(__name__)

DATA_DIR = "/app/data"

DEFAULT_GUIDELINES: Dict[str, List[str]] = {
    "de": [
        "prohibited_practices_de.json",
        "ai_system_definition_de.json",
        "gpai_obligations_de.json",
        "transparency_art50_en.json",
    ],
    "en": [
        "prohibited_practices_en.json",
        "ai_system_definition_en.json",
        "gpai_obligations_en.json",
        "transparency_art50_en.json",
    ],
}

_raw_guidelines = os.getenv("GUIDELINES")
if _raw_guidelines is None:
    GUIDELINES: Dict[str, List[str]] = DEFAULT_GUIDELINES
else:
    try:
        GUIDELINES = json.loads(_raw_guidelines)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in GUIDELINES environment variable: {e}")

AGENT_SEARCH_GUIDELINES_TOP_N = int(os.getenv("AGENT_SEARCH_GUIDELINES_TOP_N", "5"))
AGENT_SEARCH_GUIDELINES_TOKEN_LIMIT = int(
    os.getenv("AGENT_SEARCH_GUIDELINES_TOKEN_LIMIT", "2500")
)
AGENT_SEARCH_GUIDELINES_CANDIDATES = int(
    os.getenv("AGENT_SEARCH_GUIDELINES_CANDIDATES", "15")
)
AGENT_SEARCH_GUIDELINES_RRF_K = int(os.getenv("AGENT_SEARCH_GUIDELINES_RRF_K", "60"))

DocumentId = Literal[
    "prohibited_practices",
    "ai_system_definition",
    "gpai_obligations",
    "transparency_art50",
]
DOCUMENT_IDS = get_args(DocumentId)

TOKENIZER = TokenizerService()

BM25_K1 = 1.5
BM25_B = 0.75
_FOLDING = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})
_TOKEN_PATTERN = re.compile(r"\w+")

# Function words only, already in folded form (lowercase, ae/oe/ue/ss) so they
# match _tokenize output; deliberately no domain words like "ki", "act", "art".
STOPWORDS = frozenset(
    {
        "ab",
        "alle",
        "aller",
        "alles",
        "als",
        "am",
        "an",
        "andere",
        "anderen",
        "anderes",
        "auf",
        "aus",
        "bei",
        "beim",
        "bevor",
        "bis",
        "da",
        "dabei",
        "dadurch",
        "dafuer",
        "dagegen",
        "daher",
        "dann",
        "darf",
        "dass",
        "davon",
        "dazu",
        "dem",
        "den",
        "denen",
        "denn",
        "dennoch",
        "der",
        "des",
        "deshalb",
        "dessen",
        "deswegen",
        "die",
        "diese",
        "dieser",
        "dir",
        "doch",
        "dort",
        "du",
        "durch",
        "duerfen",
        "duerfte",
        "duerften",
        "ein",
        "eine",
        "einem",
        "einen",
        "einer",
        "eines",
        "er",
        "es",
        "euch",
        "falls",
        "fuer",
        "gegen",
        "haben",
        "hat",
        "hast",
        "hatte",
        "hatten",
        "haette",
        "haetten",
        "hier",
        "ihm",
        "ihn",
        "ihnen",
        "ihr",
        "im",
        "immer",
        "in",
        "ist",
        "kann",
        "kein",
        "keine",
        "koennen",
        "koennte",
        "koennten",
        "konnte",
        "konnten",
        "man",
        "mich",
        "mir",
        "mit",
        "moechte",
        "muessen",
        "muss",
        "musste",
        "mussten",
        "nach",
        "nachdem",
        "nun",
        "nur",
        "oder",
        "ohne",
        "sei",
        "seid",
        "seien",
        "sein",
        "seine",
        "seit",
        "sich",
        "sie",
        "sind",
        "soll",
        "sollen",
        "sollte",
        "sollten",
        "sondern",
        "sowie",
        "ueber",
        "um",
        "und",
        "uns",
        "unser",
        "unsere",
        "unter",
        "vom",
        "von",
        "vor",
        "war",
        "waren",
        "weil",
        "wenn",
        "werden",
        "wie",
        "wird",
        "wurde",
        "wurden",
        "wuerde",
        "wuerden",
        "zu",
        "zum",
        "zur",
        "zwischen",
        # English
        "about",
        "against",
        "all",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "before",
        "been",
        "between",
        "both",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "during",
        "each",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "hers",
        "here",
        "him",
        "his",
        "if",
        "into",
        "is",
        "it",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "out",
        "over",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "under",
        "until",
        "up",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
        "yours",
        "also",
        "could",
        "hence",
        "however",
        "may",
        "might",
        "must",
        "shall",
        "therefore",
        "thus",
        "whether",
        "would",
        "yet",
    }
)

TEXTS = {
    "de": {
        "header": "Auszüge aus den Leitlinien der Europäischen Kommission (nicht verbindlich):",
        "no_matches": "Keine passenden Abschnitte in den Leitlinien gefunden.",
        "footnotes": "Fußnoten:",
        "substep_embedding": (
            "Retrieve · ① Embedding der Suchanfrage",
            "Die Suchanfrage und die Frage des Nutzers werden mit {embedding_model} in Vektoren überführt.",
        ),
        "substep_keyword": (
            "Retrieve · ② Schlüsselwortsuche (BM25)",
            "{candidates} Abschnitte nach Schlüsselwörtern durchsucht, {hits} (Suchanfrage) bzw. {prompt_hits} (Frage des Nutzers) mit Treffern.",
        ),
        "substep_semantic": (
            "Retrieve · ③ Semantische Vektorsuche",
            "Abgleich der {candidates} Abschnitte nach Kosinus-Ähnlichkeit mit beiden Anfragen → je eine Rangliste der Vektortreffer.",
        ),
        "substep_fusion": (
            "Retrieve · ④ Rangfusion (RRF)",
            "Je Anfrage werden die Ränge aus Schlüsselwortsuche und Vektorsuche mit Reciprocal Rank Fusion (k={rrf_k}) fusioniert; die beiden Ranglisten werden abwechselnd zusammengeführt.",
        ),
        "substep_context": (
            "Retrieve · ⑤ Kontext begrenzen",
            "{used} von {candidates} Abschnitten kommen in den Kontext ({used_tokens} von max. {token_limit} Token).",
        ),
    },
    "en": {
        "header": "Excerpts from the European Commission guidelines (non-binding):",
        "no_matches": "No matching passages found in the guidelines.",
        "footnotes": "Footnotes:",
        "substep_embedding": (
            "Retrieve · ① Embedding of the search query",
            "The search query and the user's question are converted into vectors with {embedding_model}.",
        ),
        "substep_keyword": (
            "Retrieve · ② Keyword search (BM25)",
            "{candidates} passages searched by keywords, {hits} (search query) and {prompt_hits} (user's question) with hits.",
        ),
        "substep_semantic": (
            "Retrieve · ③ Semantic vector search",
            "Comparison of the {candidates} passages by cosine similarity with both inputs → one ranking of the vector hits each.",
        ),
        "substep_fusion": (
            "Retrieve · ④ Rank fusion (RRF)",
            "Per input, the ranks from the keyword search and the vector search are fused with reciprocal rank fusion (k={rrf_k}); the two rankings are then interleaved.",
        ),
        "substep_context": (
            "Retrieve · ⑤ Limit the context",
            "{used} of {candidates} passages go into the context ({used_tokens} of max. {token_limit} tokens).",
        ),
    },
}


def guideline_dataset_id(filename: str) -> str:
    return "guidelines_" + filename.removesuffix(".json")


def _tokenize(text: str) -> List[str]:
    tokens = _TOKEN_PATTERN.findall(text.lower().translate(_FOLDING))
    return [
        token
        for token in tokens
        if token not in STOPWORDS and (len(token) > 1 or not token.isalpha())
    ]


def _ranks(entries: List[Tuple[str, float]]) -> Dict[str, int]:
    return {
        chunk_id: rank
        for rank, (chunk_id, _) in enumerate(
            sorted(entries, key=lambda entry: -entry[1]), start=1
        )
    }


@dataclass
class _ChunkRecord:
    id: str
    dataset_id: str
    document: str
    paragraph: int
    title: str
    content: str
    footnotes: List[dict]


@dataclass
class _Ranking:
    order: List[_ChunkRecord]
    bm25_scores: Dict[str, float]
    bm25_ranks: Dict[str, int]
    embedding_scores: Dict[str, float]
    embedding_ranks: Dict[str, int]
    fused_scores: Dict[str, float]
    bm25_hits: int
    duration: float


class _Bm25Index:
    def __init__(self, records: List[_ChunkRecord]):
        self._doc_term_counts: Dict[str, Counter] = {}
        self._doc_lens: Dict[str, int] = {}
        doc_freq: Counter = Counter()
        total_len = 0
        for record in records:
            tokens = _tokenize(record.content) + [
                token
                for footnote in record.footnotes
                for token in _tokenize(footnote["text"])
            ]
            counts = Counter(tokens)
            self._doc_term_counts[record.id] = counts
            self._doc_lens[record.id] = len(tokens)
            total_len += len(tokens)
            doc_freq.update(counts.keys())
        self._doc_count = len(records)
        self._avg_doc_len = total_len / self._doc_count if self._doc_count else 0.0
        self._idf = {
            term: math.log(1 + (self._doc_count - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def score(self, chunk_id: str, query_tokens: List[str]) -> float:
        counts = self._doc_term_counts.get(chunk_id)
        if counts is None or not query_tokens or self._avg_doc_len == 0.0:
            return 0.0
        doc_len = self._doc_lens.get(chunk_id, 0)
        score = 0.0
        for token in query_tokens:
            tf = counts.get(token)
            if not tf:
                continue
            score += (
                self._idf[token]
                * tf
                * (BM25_K1 + 1)
                / (tf + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / self._avg_doc_len))
            )
        return score


def _load_guideline_chunks(filename: str) -> List[dict]:
    path = os.path.join(DATA_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"Guidelines file does not exist: {path}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in guidelines file {path}: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("chunks"), list):
        raise ConfigError(
            f"Invalid guidelines file {path}: expected an object with a 'chunks' list."
        )
    return data["chunks"]


def _chunk_record(raw: dict, dataset_id: str, filename: str) -> _ChunkRecord:
    document = raw["document"]
    if document not in DOCUMENT_IDS:
        raise ConfigError(
            f"Invalid guideline document '{document}' in {filename}. "
            f"Expected one of: {', '.join(DOCUMENT_IDS)}."
        )
    return _ChunkRecord(
        id=raw["id"],
        dataset_id=dataset_id,
        document=document,
        paragraph=raw["paragraph"],
        title=raw["title"],
        content=raw["content"],
        footnotes=raw.get("footnotes", []),
    )


class SearchGuidelinesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)


class SearchGuidelinesTool:
    name = "search_guidelines"
    args_model = SearchGuidelinesArgs
    max_calls = AGENT_MAX_CALLS_PER_TOOL

    def __init__(self, index_service: IndexService, languages: List[str]):
        self._index_service = index_service
        self._chunks: Dict[str, List[_ChunkRecord]] = {}
        self._bm25: Dict[str, _Bm25Index] = {}
        for language in languages:
            if language not in GUIDELINES:
                raise ConfigError(
                    f"No GUIDELINES entry for dataset language '{language}'."
                )
            records: List[_ChunkRecord] = []
            for filename in GUIDELINES[language]:
                dataset_id = guideline_dataset_id(filename)
                for raw in _load_guideline_chunks(filename):
                    records.append(_chunk_record(raw, dataset_id, filename))
            self._chunks[language] = records
            self._bm25[language] = _Bm25Index(records)
            LOGGER.info(
                f"Loaded {len(records)} guideline chunks for language '{language}'"
            )

    async def _rank(
        self,
        text: str,
        candidates: List[_ChunkRecord],
        language: str,
        run_id: str,
    ) -> _Ranking:
        embedding_response = (
            await self._index_service.embedding_service.generate_embedding(text)
        )
        duration = 0.0
        if "total_duration" in embedding_response:
            duration = embedding_response["total_duration"] / 1_000_000_000
            LOGGER.debug(
                f"[{run_id}] Total duration for embedding from ollama: {duration}"
            )

        tokens = _tokenize(text)
        bm25_scores = {
            record.id: self._bm25[language].score(record.id, tokens)
            for record in candidates
        }
        bm25_ranks = _ranks(
            [(chunk_id, score) for chunk_id, score in bm25_scores.items() if score > 0]
        )

        vectors: Dict[str, list] = {}
        for dataset_id in {record.dataset_id for record in candidates}:
            for chunk in self._index_service.chunks(dataset_id):
                vectors[chunk["id"]] = chunk["vector"]
        similarities = IndexService.cosine_similarity(
            embedding_response.embeddings[0],
            [vectors[record.id] for record in candidates],
        )
        scored = [
            (record.id, float(similarity))
            for record, similarity in zip(candidates, similarities)
        ]
        embedding_scores: Dict[str, float] = dict(scored)
        embedding_ranks = _ranks(scored)

        rrf_k = AGENT_SEARCH_GUIDELINES_RRF_K
        fused_scores = {}
        for record in candidates:
            fused_score = 0.0
            bm25_rank = bm25_ranks.get(record.id)
            embedding_rank = embedding_ranks.get(record.id)
            if bm25_rank is not None:
                fused_score += 1 / (rrf_k + bm25_rank)
            if embedding_rank is not None:
                fused_score += 1 / (rrf_k + embedding_rank)
            fused_scores[record.id] = fused_score

        order = sorted(
            candidates,
            key=lambda record: (
                -fused_scores[record.id],
                -embedding_scores[record.id],
            ),
        )
        return _Ranking(
            order=order,
            bm25_scores=bm25_scores,
            bm25_ranks=bm25_ranks,
            embedding_scores=embedding_scores,
            embedding_ranks=embedding_ranks,
            fused_scores=fused_scores,
            bm25_hits=len(bm25_ranks),
            duration=duration,
        )

    async def invoke(
        self, args: SearchGuidelinesArgs, ctx: ToolContext
    ) -> AsyncIterator[ToolEvent]:
        language = ctx.language
        texts = TEXTS[language]
        candidates = list(self._chunks.get(language, []))

        rrf_k = AGENT_SEARCH_GUIDELINES_RRF_K

        yield ToolSubstep(
            label=texts["substep_embedding"][0],
            detail=texts["substep_embedding"][1].format(embedding_model=DEFAULT_MODEL),
        )
        query_ranking = await self._rank(args.query, candidates, language, ctx.run_id)
        prompt_ranking = await self._rank(
            ctx.user_prompt, candidates, language, ctx.run_id
        )

        yield ToolSubstep(
            label=texts["substep_keyword"][0],
            detail=texts["substep_keyword"][1].format(
                candidates=len(candidates),
                hits=query_ranking.bm25_hits,
                prompt_hits=prompt_ranking.bm25_hits,
            ),
        )
        yield ToolSubstep(
            label=texts["substep_semantic"][0],
            detail=texts["substep_semantic"][1].format(candidates=len(candidates)),
        )

        merged: List[_ChunkRecord] = []
        matched_by: Dict[str, Literal["query", "prompt"]] = {}
        for query_record, prompt_record in zip(
            query_ranking.order, prompt_ranking.order
        ):
            for record, origin in ((query_record, "query"), (prompt_record, "prompt")):
                if record.id in matched_by:
                    continue
                matched_by[record.id] = origin
                merged.append(record)
        ranking = merged[:AGENT_SEARCH_GUIDELINES_CANDIDATES]
        yield ToolSubstep(
            label=texts["substep_fusion"][0],
            detail=texts["substep_fusion"][1].format(rrf_k=rrf_k),
        )

        sources: List[GuidelineSource] = []
        passages: List[str] = []
        used_count = 0
        used_tokens = 0
        for record in ranking:
            passage = _render_passage(record, texts)
            passage_tokens = TOKENIZER.count_tokens(passage)
            if used_count >= AGENT_SEARCH_GUIDELINES_TOP_N:
                skip, skip_reason = True, "top_n"
            elif used_tokens + passage_tokens > AGENT_SEARCH_GUIDELINES_TOKEN_LIMIT:
                skip, skip_reason = True, "context_window"
            else:
                skip, skip_reason = False, ""
                used_count += 1
                used_tokens += passage_tokens
                passages.append(passage)
            contributor = (
                query_ranking if matched_by[record.id] == "query" else prompt_ranking
            )
            bm25_score = contributor.bm25_scores[record.id]
            sources.append(
                GuidelineSource(
                    id=record.id,
                    document=record.document,
                    paragraphs=[str(record.paragraph)],
                    title=record.title,
                    content=record.content,
                    num_tokens=passage_tokens,
                    bm25_score=bm25_score if bm25_score > 0 else None,
                    bm25_rank=contributor.bm25_ranks.get(record.id),
                    embedding_score=contributor.embedding_scores.get(record.id),
                    embedding_rank=contributor.embedding_ranks.get(record.id),
                    fused_score=contributor.fused_scores[record.id],
                    matched_by=matched_by[record.id],
                    skip=skip,
                    skip_reason=skip_reason,
                )
            )
        yield ToolSubstep(
            label=texts["substep_context"][0],
            detail=texts["substep_context"][1].format(
                used=used_count,
                candidates=len(ranking),
                used_tokens=used_tokens,
                token_limit=AGENT_SEARCH_GUIDELINES_TOKEN_LIMIT,
            ),
        )

        yield GuidelineSources(
            sources=sources,
            duration=query_ranking.duration + prompt_ranking.duration,
        )

        if passages:
            result_text = "\n\n".join([texts["header"], *passages])
        else:
            result_text = texts["no_matches"]
        LOGGER.info(
            f"[{ctx.run_id}] search_guidelines: query <{args.query}>, "
            f"{used_count} of {len(ranking)} passages used ({used_tokens} tokens)"
        )
        yield ToolResult(text=result_text)


def _render_passage(record: _ChunkRecord, texts: Dict[str, str]) -> str:
    lines = [f"[{record.title}]", record.content]
    if record.footnotes:
        lines.append(texts["footnotes"])
        lines.extend(f"[{f['number']}] {f['text']}" for f in record.footnotes)
    return "\n".join(lines)
