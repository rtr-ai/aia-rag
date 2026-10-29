import json
import os
import re
from datetime import date as Date
from pathlib import Path
from typing import AsyncIterator, Dict, List, Literal, Optional

from agent.registry import AGENT_MAX_CALLS_PER_TOOL, ToolContext, ToolEvent
from models.agent_events import ToolResult
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from services.dataset_configuration import ConfigError
from utils.logger import get_logger

LOGGER = get_logger(__name__)

AGENT_APPLICABILITY_FILE = os.getenv("AGENT_APPLICABILITY_FILE", "ai_act_geltung.json")

ARTICLE_PREFIX_RE = re.compile(r"^(?:artikel|article|art\.?)\s*", re.IGNORECASE)
PARAGRAPH_PREFIX_RE = re.compile(r"^(?:absatz|abs\.?|para\.?)\s*", re.IGNORECASE)
LETTER_PREFIX_RE = re.compile(r"^(?:buchstabe|letter|lit\.?)\s*", re.IGNORECASE)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

TEXTS = {
    "de": {
        "header": "Geltung von {reference} ({title})",
        "label_date": "Stichtag",
        "label_stand": "Datenstand",
        "whole_article": "(ganzer Artikel)",
        "label_condition": "Bedingung",
        "label_status": "Status",
        "label_basis": "Grundlage",
        "label_remark": "Anmerkung",
        "label_source": "Quelle",
        "self_reference": "in der Bestimmung selbst",
        "applies_since": "gilt seit {date}",
        "applies_from": "gilt noch nicht, erst ab {date}",
        "in_force_since": "in Kraft seit {date}",
        "in_force_from": "noch nicht in Kraft, erst ab {date}",
        "transition_running": "Übergangsfrist läuft noch bis {date}",
        "transition_expired": "Übergangsfrist am {date} abgelaufen",
        "deadline_running": "Frist läuft noch bis {date}",
        "deadline_expired": "Frist am {date} abgelaufen",
        "status_no_date": "kein festes Datum",
        "no_own_entry": (
            "Kein eigener Eintrag für {reference}; es gelten die Einträge zu {parent}."
        ),
        "no_own_entry_all": (
            "Kein eigener Eintrag für {reference}; es folgen alle Einträge zu "
            "{article}."
        ),
        "transition_precedence": (
            "Übergangsfristen gehen dem Geltungsbeginn für die in ihrer Bedingung "
            "genannten Fälle vor."
        ),
        "not_covered": (
            "„{article}“ ist nicht in der Geltungstabelle enthalten. Nicht erfasst "
            "sind die Anhänge, nationale Durchführungsregelungen und Fristen der "
            "Verhaltenskodizes/Praxisleitfäden."
        ),
        "invalid_date": (
            "Ungültiges Datum '{date}'. Gib das Datum im ISO-Format JJJJ-MM-TT an, "
            "z. B. 2026-10-28."
        ),
        "letter_without_paragraph": (
            "Ein Buchstabe kann nur zusammen mit einem Absatz angegeben werden."
        ),
    },
    "en": {
        "header": "Applicability of {reference} ({title})",
        "label_date": "Reference date",
        "label_stand": "Data as of",
        "whole_article": "(whole article)",
        "label_condition": "Condition",
        "label_status": "Status",
        "label_basis": "Basis",
        "label_remark": "Remark",
        "label_source": "Source",
        "self_reference": "in the provision itself",
        "applies_since": "applies since {date}",
        "applies_from": "does not apply yet, only from {date}",
        "in_force_since": "in force since {date}",
        "in_force_from": "not yet in force, only from {date}",
        "transition_running": "transition period runs until {date}",
        "transition_expired": "transition period expired on {date}",
        "deadline_running": "deadline runs until {date}",
        "deadline_expired": "deadline expired on {date}",
        "status_no_date": "no fixed date",
        "no_own_entry": (
            "No separate entry for {reference}; the entries for {parent} apply."
        ),
        "no_own_entry_all": (
            "No separate entry for {reference}; all entries for {article} follow."
        ),
        "transition_precedence": (
            "For the cases named in their condition, transition periods take "
            "precedence over the date of application."
        ),
        "not_covered": (
            "\"{article}\" is not covered by the applicability table. Not covered "
            "are the annexes, national implementing rules and deadlines of codes of "
            "practice."
        ),
        "invalid_date": (
            "Invalid date '{date}'. Provide the date in ISO format YYYY-MM-DD, "
            "e.g. 2026-10-28."
        ),
        "letter_without_paragraph": (
            "A letter can only be given together with a paragraph."
        ),
    },
}


class DateBasis(BaseModel):
    norm: Optional[str]
    datum: Optional[Date]
    inhalt: Optional[str]


class ApplicabilityEntry(BaseModel):
    artikel: str
    absatz: Optional[str]
    litera: Optional[str]
    titel: str
    typ: Literal[
        "Geltungsbeginn", "Übergangsfrist", "Einzelfrist", "Hinweis", "Inkrafttreten"
    ]
    bedingung: Optional[str]
    datum: Optional[Date]
    grundlage: Optional[str]
    anmerkung: Optional[str]
    quelle: Optional[str]


class ApplicabilityTable(BaseModel):
    stand: Date
    grundlagen: Dict[str, DateBasis]
    eintraege: List[ApplicabilityEntry]


def load_applicability_table() -> ApplicabilityTable:
    file_path = Path("/app/data") / AGENT_APPLICABILITY_FILE
    if not file_path.is_file():
        raise ConfigError(f"Applicability table file does not exist: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in applicability table file {file_path}: {e}")
    try:
        table = ApplicabilityTable.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"Invalid applicability table file {file_path}: {e}")
    LOGGER.info(
        f"Loaded applicability table <{file_path}>: {len(table.eintraege)} entries, "
        f"stand <{table.stand}>"
    )
    return table


def _normalize(value: Optional[str], prefix: re.Pattern[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = prefix.sub("", value.strip(), count=1).strip().lower()
    return normalized or None


def _parse_date(value: str) -> Optional[Date]:
    if not DATE_RE.match(value):
        return None
    try:
        return Date.fromisoformat(value)
    except ValueError:
        return None


def _reference(
    artikel: str, absatz: Optional[str], litera: Optional[str], language: str
) -> str:
    if language == "en":
        reference = f"Art {artikel}"
        if absatz is not None:
            reference += f"({absatz})"
        if litera is not None:
            reference += f"({litera})"
        return reference
    reference = f"Art {artikel}"
    if absatz is not None and litera is not None:
        reference += f" Abs {absatz} lit {litera}"
    elif absatz is not None:
        reference += f" Abs {absatz}"
    return reference


def _own_ref(entry: ApplicabilityEntry) -> str:
    return "-".join(
        part for part in (entry.artikel, entry.absatz, entry.litera) if part is not None
    )


def _status(
    entry: ApplicabilityEntry, d: Date, texts: Dict[str, str]
) -> Optional[str]:
    if entry.typ == "Hinweis":
        return None
    if entry.datum is None:
        return texts["status_no_date"]
    if entry.typ == "Geltungsbeginn":
        key = "applies_since" if entry.datum <= d else "applies_from"
    elif entry.typ == "Inkrafttreten":
        key = "in_force_since" if entry.datum <= d else "in_force_from"
    elif entry.typ == "Übergangsfrist":
        key = "transition_running" if d < entry.datum else "transition_expired"
    else:
        key = "deadline_running" if d <= entry.datum else "deadline_expired"
    return texts[key].format(date=entry.datum.isoformat())


def _basis(
    entry: ApplicabilityEntry, table: ApplicabilityTable, texts: Dict[str, str]
) -> Optional[str]:
    code = entry.grundlage
    if code is not None and code in table.grundlagen:
        basis = table.grundlagen[code]
        if basis.norm is not None and basis.inhalt is not None:
            return f"{basis.norm} – {basis.inhalt}"
        if basis.norm is not None:
            return basis.norm
        return basis.inhalt
    if code is not None and code == _own_ref(entry):
        return texts["self_reference"]
    return None


def _entry_lines(
    entry: ApplicabilityEntry,
    d: Date,
    texts: Dict[str, str],
    language: str,
    table: ApplicabilityTable,
    paragraph: Optional[str],
) -> List[str]:
    whole_article = ""
    if paragraph is not None and entry.absatz is None:
        whole_article = f" {texts['whole_article']}"
    reference = _reference(entry.artikel, entry.absatz, entry.litera, language)
    lines = [f"{reference}{whole_article} · {entry.typ}"]
    if entry.bedingung is not None:
        lines.append(f"{texts['label_condition']}: {entry.bedingung}")
    status = _status(entry, d, texts)
    if status is not None:
        lines.append(f"{texts['label_status']}: {status}")
    basis = _basis(entry, table, texts)
    if basis is not None:
        lines.append(f"{texts['label_basis']}: {basis}")
    if entry.anmerkung is not None:
        lines.append(f"{texts['label_remark']}: {entry.anmerkung}")
    if entry.quelle is not None:
        lines.append(f"{texts['label_source']}: {entry.quelle}")
    return lines


class CheckApplicabilityArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    article: str = Field(min_length=1)
    paragraph: Optional[str] = None
    letter: Optional[str] = None
    date: str


class CheckApplicabilityTool:
    name = "check_applicability"
    args_model = CheckApplicabilityArgs
    max_calls = AGENT_MAX_CALLS_PER_TOOL

    def __init__(self, table: ApplicabilityTable):
        self._table = table

    async def invoke(
        self, args: CheckApplicabilityArgs, ctx: ToolContext
    ) -> AsyncIterator[ToolEvent]:
        texts = TEXTS[ctx.language]
        language = ctx.language
        d = _parse_date(args.date.strip())
        if d is None:
            yield ToolResult(
                text=texts["invalid_date"].format(date=args.date.strip()),
                is_error=True,
            )
            return
        paragraph = _normalize(args.paragraph, PARAGRAPH_PREFIX_RE)
        letter = _normalize(args.letter, LETTER_PREFIX_RE)
        if letter is not None and paragraph is None:
            yield ToolResult(text=texts["letter_without_paragraph"], is_error=True)
            return
        article = _normalize(args.article, ARTICLE_PREFIX_RE)
        rows = [
            entry
            for entry in self._table.eintraege
            if article is not None and entry.artikel.lower() == article
        ]
        if not rows:
            yield ToolResult(
                text=texts["not_covered"].format(article=args.article.strip())
            )
            return
        artikel = rows[0].artikel
        selected: List[ApplicabilityEntry] = rows
        specific: List[ApplicabilityEntry] = []
        if paragraph is not None:
            selected = [
                entry
                for entry in rows
                if entry.absatz is None
                or (
                    entry.absatz.lower() == paragraph
                    and (
                        entry.litera is None
                        or letter is None
                        or entry.litera.lower() == letter
                    )
                )
            ]
            specific = [
                entry
                for entry in rows
                if entry.absatz is not None
                and entry.absatz.lower() == paragraph
                and (
                    letter is None
                    or (entry.litera is not None and entry.litera.lower() == letter)
                )
            ]
        requested = _reference(artikel, paragraph, letter, language)
        note: Optional[str] = None
        if paragraph is not None and not specific:
            has_parent_entry = any(
                entry.absatz is not None
                and entry.absatz.lower() == paragraph
                and entry.litera is None
                for entry in rows
            )
            if letter is not None and has_parent_entry:
                parent = _reference(artikel, paragraph, None, language)
            else:
                parent = _reference(artikel, None, None, language)
            if selected:
                note = texts["no_own_entry"].format(reference=requested, parent=parent)
            else:
                selected = rows
                note = texts["no_own_entry_all"].format(
                    reference=requested,
                    article=_reference(artikel, None, None, language),
                )
        LOGGER.info(
            f"[{ctx.run_id}] check_applicability: <{requested}> on <{d}>, "
            f"{len(selected)} entries"
        )
        blocks: List[str] = [
            "\n".join(
                [
                    texts["header"].format(
                        reference=requested, title=selected[0].titel
                    ),
                    f"{texts['label_date']}: {d.isoformat()}",
                    f"{texts['label_stand']}: {self._table.stand.isoformat()}",
                ]
            )
        ]
        if note is not None:
            blocks.append(note)
        for entry in selected:
            blocks.append(
                "\n".join(
                    _entry_lines(entry, d, texts, language, self._table, paragraph)
                )
            )
        if any(entry.typ == "Übergangsfrist" for entry in selected):
            blocks.append(texts["transition_precedence"])
        yield ToolResult(text="\n\n".join(blocks))
