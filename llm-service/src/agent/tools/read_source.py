import json
import os
import re
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Literal, Optional, Set, Tuple

from agent.registry import AGENT_MAX_CALLS_PER_TOOL, ToolContext, ToolEvent
from models.agent_events import LookupChunk, LookupChunks, ToolResult, ToolSubstep
from pydantic import BaseModel, ConfigDict, Field
from services.dataset_configuration import ConfigError
from services.tokenizer_service import TokenizerService
from utils.logger import get_logger

LOGGER = get_logger(__name__)

DATA_DIR = "/app/data"

AGENT_READ_SOURCE_MAX_TOKENS = int(os.getenv("AGENT_READ_SOURCE_MAX_TOKENS", "2500"))

OUTLINE_WORDS = 12
MAX_LISTED_KEYS = 12

UNIT_LABELS = {
    "article": "label_article",
    "recital": "label_recital",
    "annex": "label_annex",
}

PREFIX_RE = re.compile(
    r"^(?:artikel|article|art|absatz|abs|paragraph|para|nr|rz|erwg|recital|anhang"
    r"|annex|z)(?:\.\s*|\s+|(?=[\d(]))"
)
PARENTHESIZED_RE = re.compile(r"^\((\w+)\)$")
LETTER_REFERENCE_RE = re.compile(
    r"^(\S+?)\s*(?:\(\s*[a-z]+\s*\)|\s(?:lit\.?|buchstabe|letter|point|punkt)\s*"
    r"\(?[a-z]+\)?)"
)
LEADING_MARKER_RE = re.compile(r"^(?:\(\d+[a-z]*\)|\d+[a-z]*\.)\s+")
DEFINED_TERM_RE = re.compile(r"„[^“]+“|‘[^’]+’|‚[^‛]+‛")
ROMAN_NUMERALS = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]

TEXTS = {
    "de": {
        "substep_label": "Fundstellen",
        "law": "AI Act",
        "label_article": "Art",
        "label_recital": "ErwG",
        "label_annex": "Anhang",
        "ref_article": "Art {number}",
        "ref_article_para": "Art {number} Abs {sub}",
        "ref_article_point": "Art {number} Nr. {sub}",
        "ref_recital": "ErwG {number}",
        "ref_annex": "Anhang {number}",
        "ref_annex_point": "Anhang {number} Nr. {sub}",
        "sub_paragraph": "Absatz",
        "sub_point": "Punkt",
        "error": "Fehler bei {reference}: {message}",
        "sub_not_allowed": "Für unit '{unit}' ist kein 'sub' erlaubt; lass es weg.",
        "letter_reference": 'Buchstaben und Unterpunkte sind nicht einzeln abrufbar, sie sind Teil ihres Absatzes bzw. Punktes. Fordere stattdessen den Absatz oder Punkt an, z. B. sub "{paragraph}".',
        "unknown_number": "{requested} gibt es nicht. Gültig: {valid}.",
        "unknown_sub": "{parent} hat keinen {sub_kind} {sub}. Gültig: {valid}.",
        "no_subs": "{parent} hat keine Absätze oder Punkte; lass 'sub' weg.",
        "outline_unit": "Hinweis: {reference} ist zu lang ({tokens} Token, Grenze {limit}); hier nur die Gliederung. Fordere einzelne Teile über 'sub' an.",
    },
    "en": {
        "substep_label": "References",
        "law": "AI Act",
        "label_article": "Art",
        "label_recital": "Recital",
        "label_annex": "Annex",
        "ref_article": "Art {number}",
        "ref_article_para": "Art {number}({sub})",
        "ref_article_point": "Art {number}({sub})",
        "ref_recital": "Recital {number}",
        "ref_annex": "Annex {number}",
        "ref_annex_point": "Annex {number}({sub})",
        "sub_paragraph": "paragraph",
        "sub_point": "point",
        "error": "Error for {reference}: {message}",
        "sub_not_allowed": "Unit '{unit}' takes no 'sub'; omit it.",
        "letter_reference": 'Letters and sub-items cannot be requested individually, they are part of their paragraph or point. Request the paragraph or point instead, e.g. sub "{paragraph}".',
        "unknown_number": "{requested} does not exist. Valid: {valid}.",
        "unknown_sub": "{parent} has no {sub_kind} {sub}. Valid: {valid}.",
        "no_subs": "{parent} has no paragraphs or points; omit 'sub'.",
        "outline_unit": "Note: {reference} is too long ({tokens} tokens, limit {limit}); only the outline is shown. Request individual parts with 'sub'.",
    },
}

TOKENIZER = TokenizerService()


class ReferenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=True)

    unit: Literal["article", "recital", "annex"]
    number: str = Field(min_length=1)
    sub: Optional[str] = None


class ReadSourceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    references: List[ReferenceItem] = Field(min_length=1, max_length=5)


@dataclass(frozen=True)
class _Reference:
    unit: str
    number: str
    sub: Optional[str]


@dataclass
class _Unit:
    id: str
    reference: str
    title: str
    content: str
    position: int
    children: Dict[str, "_Unit"] = field(default_factory=dict)


@dataclass
class _Law:
    units: Dict[Tuple[str, str], _Unit] = field(default_factory=dict)
    numbers: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class _Outcome:
    text: str
    chunks: List[LookupChunk] = field(default_factory=list)
    resolved: bool = False


class _ReferenceError(Exception):
    pass


def _to_roman(value: int) -> str:
    result = ""
    for size, numeral in ROMAN_NUMERALS:
        while value >= size:
            result += numeral
            value -= size
    return result


def _normalize(value: str) -> str:
    text = value.strip().lower()
    while True:
        stripped = PREFIX_RE.sub("", text, count=1).strip()
        parenthesized = PARENTHESIZED_RE.match(stripped)
        if parenthesized:
            stripped = parenthesized.group(1)
        if stripped == text:
            return text.rstrip(".")
        text = stripped


def _normalize_reference(item: ReferenceItem) -> _Reference:
    number = _normalize(item.number)
    if item.unit == "annex":
        number = _to_roman(int(number)) if number.isdecimal() else number.upper()
    sub = _normalize(item.sub) if item.sub is not None else ""
    return _Reference(unit=item.unit, number=number, sub=sub or None)


def _describe(reference: _Reference) -> str:
    description = f"{reference.unit} {reference.number}"
    if reference.sub is not None:
        description += f" sub {reference.sub}"
    return description


def _compact(keys: List[str]) -> str:
    if not keys[0][0].isdecimal():
        return keys[0] if len(keys) == 1 else f"{keys[0]}–{keys[-1]}"
    if len(keys) <= MAX_LISTED_KEYS:
        return ", ".join(keys)
    runs: List[List[int]] = []
    for number in sorted(int(key) for key in keys if key.isdecimal()):
        if runs and number == runs[-1][1] + 1:
            runs[-1][1] = number
        else:
            runs.append([number, number])
    parts = [_join_run(str(first), str(last), last - first + 1) for first, last in runs]
    suffixed: List[List[str]] = []
    for key in keys:
        if key.isdecimal():
            continue
        previous = suffixed[-1][-1] if suffixed else None
        if (
            previous is not None
            and key[:-1] == previous[:-1]
            and ord(key[-1]) == ord(previous[-1]) + 1
        ):
            suffixed[-1].append(key)
        else:
            suffixed.append([key])
    parts.extend(_join_run(run[0], run[-1], len(run)) for run in suffixed)
    return ", ".join(parts)


def _join_run(first: str, last: str, size: int) -> str:
    if size == 1:
        return first
    if size == 2:
        return f"{first}, {last}"
    return f"{first}–{last}"


def _first_words(content: str) -> str:
    words = content.split()
    text = " ".join(words[:OUTLINE_WORDS])
    return f"{text} …" if len(words) > OUTLINE_WORDS else text


def _outline_entry(content: str, definitions: bool) -> str:
    content = LEADING_MARKER_RE.sub("", content)
    if definitions:
        term = DEFINED_TERM_RE.search(content)
        if term:
            return term.group(0)
    return _first_words(content)


def _render_unit(unit: _Unit) -> str:
    return f"[{unit.title}]\n{unit.content}"


def _load_structured(language: str) -> dict:
    path = os.path.join(DATA_DIR, f"ai_act_structured_{language}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"Structured AI Act file does not exist: {path}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in structured AI Act file {path}: {e}") from e
    for section in ("articles", "recitals", "annexes"):
        if not isinstance(data.get(section), list):
            raise ConfigError(
                f"Invalid structured AI Act file {path}: missing '{section}' list."
            )
    return data


def _build_law(data: dict, language: str) -> _Law:
    texts = TEXTS[language]
    law = _Law()
    position = 0

    def add(kind: str, number: str, unit: _Unit) -> None:
        law.units[(kind, number)] = unit
        law.numbers.setdefault(kind, []).append(number)

    def children(
        parent_id: str,
        reference_key: str,
        number: str,
        heading: str,
        raw: List[dict],
    ) -> Dict[str, _Unit]:
        nonlocal position
        result: Dict[str, _Unit] = {}
        for paragraph in raw:
            key = paragraph["number"].lower()
            reference = (
                texts[reference_key].format(number=number, sub=key) + f" {texts['law']}"
            )
            position += 1
            result[key] = _Unit(
                id=f"{parent_id}-{key}",
                reference=reference,
                title=f"{reference} – {heading}",
                content=paragraph["text"],
                position=position,
            )
        return result

    for article in data["articles"]:
        number = article["number"].lower()
        heading = article["heading"]
        reference = f"{texts['ref_article'].format(number=number)} {texts['law']}"
        unit_id = f"ai_act_{language}:art-{number}"
        position += 1
        unit = _Unit(
            id=unit_id,
            reference=reference,
            title=f"{reference} – {heading}",
            content="",
            position=position,
        )
        reference_key = "ref_article_point" if number == "3" else "ref_article_para"
        unit.children = children(
            unit_id, reference_key, number, heading, article["paragraphs"]
        )
        unit.content = "\n".join(
            part
            for part in [article["text"], *(c.content for c in unit.children.values())]
            if part
        )
        add("article", number, unit)

    for annex in data["annexes"]:
        number = annex["number"].upper()
        heading = annex["heading"]
        reference = f"{texts['ref_annex'].format(number=number)} {texts['law']}"
        unit_id = f"ai_act_{language}:annex-{number.lower()}"
        position += 1
        unit = _Unit(
            id=unit_id,
            reference=reference,
            title=f"{reference} – {heading}",
            content="",
            position=position,
        )
        unit.children = children(
            unit_id, "ref_annex_point", number, heading, annex["paragraphs"]
        )
        unit.content = "\n".join(
            part
            for part in [annex["text"], *(c.content for c in unit.children.values())]
            if part
        )
        add("annex", number, unit)

    for recital in data["recitals"]:
        number = recital["number"]
        reference = f"{texts['ref_recital'].format(number=number)} {texts['law']}"
        position += 1
        add(
            "recital",
            number,
            _Unit(
                id=f"ai_act_{language}:recital-{number}",
                reference=reference,
                title=reference,
                content=recital["text"],
                position=position,
            ),
        )
    return law


def _error(reference: _Reference, texts: Dict[str, str], message: str) -> str:
    return texts["error"].format(reference=_describe(reference), message=message)


class ReadSourceTool:
    name = "read_source"
    args_model = ReadSourceArgs
    max_calls = AGENT_MAX_CALLS_PER_TOOL

    def __init__(self, languages: List[str]):
        self._law: Dict[str, _Law] = {}
        for language in languages:
            self._law[language] = _build_law(_load_structured(language), language)
            LOGGER.info(
                f"Loaded {len(self._law[language].units)} AI Act units "
                f"for language '{language}'"
            )

    async def invoke(
        self, args: ReadSourceArgs, ctx: ToolContext
    ) -> AsyncIterator[ToolEvent]:
        language = ctx.language
        texts = TEXTS[language]

        references: List[_Reference] = []
        seen: Set[_Reference] = set()
        for item in args.references:
            reference = _normalize_reference(item)
            if reference not in seen:
                seen.add(reference)
                references.append(reference)

        yield ToolSubstep(
            label=texts["substep_label"],
            detail=", ".join(_describe(reference) for reference in references),
        )

        outcomes: List[_Outcome] = []
        for reference in references:
            try:
                outcomes.append(self._read(reference, language))
            except _ReferenceError as e:
                outcomes.append(_Outcome(text=_error(reference, texts, str(e))))

        yield LookupChunks(chunks=[c for outcome in outcomes for c in outcome.chunks])

        blocks = [o.text for o in outcomes]
        resolved = sum(1 for o in outcomes if o.resolved)
        LOGGER.info(
            f"[{ctx.run_id}] read_source: {resolved} of {len(references)} "
            f"references resolved"
        )
        yield ToolResult(text="\n\n".join(blocks), is_error=resolved == 0)

    def _read(self, reference: _Reference, language: str) -> _Outcome:
        texts = TEXTS[language]
        if reference.sub is not None and reference.unit == "recital":
            raise _ReferenceError(texts["sub_not_allowed"].format(unit=reference.unit))
        if reference.sub is not None and reference.unit in ("article", "annex"):
            letter = LETTER_REFERENCE_RE.match(reference.sub)
            if letter:
                raise _ReferenceError(
                    texts["letter_reference"].format(paragraph=letter.group(1))
                )
        return self._read_law(reference, language)

    def _read_law(self, reference: _Reference, language: str) -> _Outcome:
        texts = TEXTS[language]
        law = self._law[language]
        unit = law.units.get((reference.unit, reference.number))
        if unit is None:
            label = texts[UNIT_LABELS[reference.unit]]
            valid = _compact(law.numbers[reference.unit])
            raise _ReferenceError(
                texts["unknown_number"].format(
                    requested=f"{label} {reference.number}",
                    valid=f"{label} {valid}",
                )
            )
        if reference.sub is not None:
            if not unit.children:
                raise _ReferenceError(texts["no_subs"].format(parent=unit.reference))
            child = unit.children.get(reference.sub)
            if child is None:
                point = reference.unit == "annex" or reference.number == "3"
                raise _ReferenceError(
                    texts["unknown_sub"].format(
                        parent=unit.reference,
                        sub_kind=texts["sub_point" if point else "sub_paragraph"],
                        sub=reference.sub,
                        valid=_compact(list(unit.children)),
                    )
                )
            unit = child

        passage = _render_unit(unit)
        tokens = TOKENIZER.count_tokens(passage)
        content = unit.content
        definitions = reference.unit == "article" and reference.number == "3"
        if unit.children and tokens > AGENT_READ_SOURCE_MAX_TOKENS:
            content = "\n".join(
                [
                    texts["outline_unit"].format(
                        reference=unit.reference,
                        tokens=tokens,
                        limit=AGENT_READ_SOURCE_MAX_TOKENS,
                    ),
                    *(
                        f"- {child.reference}: "
                        f"{_outline_entry(child.content, definitions)}"
                        for child in unit.children.values()
                    ),
                ]
            )
            passage = f"[{unit.title}]\n{content}"
            tokens = TOKENIZER.count_tokens(passage)
        return _Outcome(
            text=passage,
            chunks=[
                LookupChunk(
                    id=unit.id,
                    corpus="ai_act",
                    reference=unit.reference,
                    title=unit.title,
                    content=content,
                    num_tokens=tokens,
                    position=unit.position,
                )
            ],
            resolved=True,
        )
