#!/usr/bin/env python3
"""Turn the AI Act chunk files into one structured JSON file per language.

Usage: python3 ingest_ai_act.py [--de combined.json] [--en combined_en_new.json]
                                [--out-dir <dir>]

Only law chunks are read (titles starting with Art/ErwG/Recital/Anhang/Annex);
every other chunk is ignored. Chunks of one article or annex are merged in legal
order, repeated headings, omission markers and overlapping duplicates are removed
and hard-wrapped lines are joined so that each paragraph, point, letter and
sub-item is one line. The output files ai_act_structured_<lang>.json are written
to <out_dir> (default: next to this script) only if all checks pass; the script
and the generated JSON are committed.
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(SCRIPT_DIR)

PAREN_NUM_RE = re.compile(r"^\((\d+[a-z]?)\)(?:\s+|$)")
DOT_NUM_RE = re.compile(r"^(\d+[a-z]?)\.(?:\s+|$)")
ITEM_RE = re.compile(r"^(?:\(([a-z]{1,4})\)|([a-z]{1,4})\))(?:\s+|$)")
INLINE_PAREN_RE = re.compile(r"(?<=[.;:]) \((\d+[a-z]?)\)(?= \S)")
INLINE_DOT_RE = re.compile(r"(?<=[.;:]) (\d+[a-z]?)\.(?= \S)")
ITEM_DOT_RE = re.compile(r"^([a-z])\.(?:\s+|$)")
BULLET_RE = re.compile(r"^[-—–•]\s")
KEY_RE = re.compile(r"^(\d+)([a-z]*)$")
ARTICLE_TITLE_RE = re.compile(r"^Art (\d+[a-z]?)(?::|$)")
ABS_TITLE_RE = re.compile(r"\bAbs (\d+[a-z]?)")
HEADING_RE = re.compile(r"^(#+)\s+(\S.*)$")

ARTICLE_EXTRAS = ["4a", "60a", "75a", "75b", "75c", "75d"]
ART3_POINTS = [str(n) for n in range(1, 69)] + ["14a", "14b"]

# Annex I starts at point 2 (point 1 is absent from the chunks), Annex X
# lacks its point 4 heading, Annex VII lacks point 2 in German and has nested numbers
# (3.1, 4.2), Annexes VIII and XI restart their numbering per section and Annex XIII
# has letters only. These stay plain text, the others are split into numbered points.
ANNEX_POINT_LISTS = {"III", "IV", "V", "VI", "IX", "XII", "XIV"}


class IngestError(Exception):
    pass


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def toline(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def to_roman(number: int) -> str:
    out = ""
    for value, symbol in ((10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")):
        while number >= value:
            out += symbol
            number -= value
    return out


ROMAN_SUBITEMS = [to_roman(n) for n in range(1, 21)]
ANNEX_NUMBERS = [to_roman(n).upper() for n in range(1, 15)]


def split_key(key: str) -> Tuple[int, str]:
    match = KEY_RE.match(key)
    return int(match.group(1)), match.group(2)


def letter_order(letter: str) -> Tuple[int, int]:
    return ord(letter[0]), ord(letter[1]) if len(letter) > 1 else 0


def letter_successors(letter: str) -> List[str]:
    if len(letter) == 1:
        return [chr(ord(letter) + 1), f"{letter}a"]
    return [f"{letter[0]}{chr(ord(letter[1]) + 1)}", chr(ord(letter[0]) + 1)]


def next_roman(current: Optional[str]) -> Optional[str]:
    if current is None:
        return ROMAN_SUBITEMS[0]
    index = ROMAN_SUBITEMS.index(current) + 1
    return ROMAN_SUBITEMS[index] if index < len(ROMAN_SUBITEMS) else None


def next_para_keys(key: str) -> List[str]:
    number, suffix = split_key(key)
    next_suffix = "a" if not suffix else chr(ord(suffix[-1]) + 1)
    return [str(number + 1), f"{number}{next_suffix}"]


class LangCfg:
    def __init__(
        self,
        code: str,
        default_file: str,
        recital: str,
        annex: str,
        article_para_re: "re.Pattern[str]",
        art3_para_re: "re.Pattern[str]",
    ):
        self.code = code
        self.default_file = default_file
        self.recital = recital
        self.annex = annex
        self.article_para_re = article_para_re
        self.art3_para_re = art3_para_re
        self.law_re = re.compile(rf"^(Art \d|{recital} |{annex} )")
        self.recital_re = re.compile(rf"^{recital} (\d+)( Digital Omnibus)?\s*$")
        self.annex_re = re.compile(rf"^{annex} ([IVX]+)(?::\s*Z\d+)?\s*$")


LANGS = [
    LangCfg("de", "combined.json", "ErwG", "Anhang", PAREN_NUM_RE, DOT_NUM_RE),
    LangCfg("en", "combined_en_new.json", "Recital", "Annex", DOT_NUM_RE, PAREN_NUM_RE),
]


class Lines:
    """Lines of one paragraph, letter or sub-item; wrapped lines are joined."""

    def __init__(self, join: bool, bullets: bool = False):
        self.join = join
        self.bullets = bullets
        self.lines: List[str] = []
        self.joinable = False

    def add(self, text: str, gap: bool = False) -> None:
        table = text.startswith("|")
        bullet = self.bullets and BULLET_RE.match(text) is not None
        if (
            self.lines
            and self.join
            and self.joinable
            and not gap
            and not table
            and not bullet
        ):
            self.lines[-1] = f"{self.lines[-1]} {text}"
        else:
            self.lines.append(text)
        self.joinable = not table and not bullet


class Item:
    def __init__(self, lines: Lines):
        self.lines = lines
        self.subs: Dict[str, Lines] = {}

    def render_lines(self) -> List[str]:
        out = list(self.lines.lines)
        for key in sorted(self.subs, key=ROMAN_SUBITEMS.index):
            out.extend(self.subs[key].lines)
        return out


class Block:
    """A paragraph or point: head lines followed by lettered lists with sub-items.

    A paragraph can hold several lists (the letters restart at 'a'); each one is
    a run.
    """

    def __init__(self, key: Optional[str], join: bool):
        self.key = key
        self.fragment = False
        self.head = Lines(join)
        self.runs: List[Dict[str, Item]] = []

    def is_empty(self) -> bool:
        return not self.head.lines and not self.runs

    def render_lines(self) -> List[str]:
        out = list(self.head.lines)
        for run in self.runs:
            for key in sorted(run, key=letter_order):
                out.extend(run[key].render_lines())
        return out

    def render(self) -> str:
        return "\n".join(self.render_lines())

    def merge(self, other: "Block") -> None:
        if other.head.lines:
            if not self.head.lines:
                self.head = other.head
            elif self.head.lines != other.head.lines:
                raise IngestError("two chunks give different text for the same head")
        if len(self.runs) != len(other.runs):
            if not self.runs or not other.runs:
                self.runs = self.runs or other.runs
                return
            raise IngestError("two chunks give different lists of letters")
        for run, other_run in zip(self.runs, other.runs):
            for key, item in other_run.items():
                if key not in run:
                    run[key] = item
                elif run[key].render_lines() != item.render_lines():
                    raise IngestError(f"two chunks give different text for '{key}'")

    def check_items(self) -> None:
        for run in self.runs:
            previous: Optional[str] = None
            for key in sorted(run, key=letter_order):
                if previous is None and key != "a":
                    raise IngestError(
                        f"letter 'a' is missing (first letter is '{key}')"
                    )
                if previous is not None and key not in letter_successors(previous):
                    raise IngestError(
                        f"letter after '{previous}' is missing (found '{key}')"
                    )
                previous = key
                subs = sorted(run[key].subs, key=ROMAN_SUBITEMS.index)
                if subs != ROMAN_SUBITEMS[: len(subs)]:
                    raise IngestError(
                        f"sub-items of '{key}' are not i, ii, ...: {subs}"
                    )


class ParsedChunk:
    def __init__(self) -> None:
        self.blocks: List[Block] = []
        self.heading: Optional[str] = None
        self.chapter: Optional[str] = None


class ChunkParser:
    """Split the lines of one chunk into blocks by paragraph and item markers."""

    def __init__(
        self,
        title: str,
        kind: str,
        para_re: "re.Pattern[str]",
        join: bool,
    ):
        self.title = title
        self.kind = kind
        self.para_re = para_re
        self.join = join
        abs_match = ABS_TITLE_RE.search(title)
        self.title_abs = abs_match.group(1) if abs_match else None
        self.inline_re = INLINE_PAREN_RE if para_re is PAREN_NUM_RE else INLINE_DOT_RE
        self.inline_splits: List[str] = []
        self.parsed = ParsedChunk()
        self.block = Block(None, join)
        self.target = self.block.head
        self.cur_item: Optional[Item] = None
        self.last_para: Optional[str] = None
        self.last_letter: Optional[str] = None
        self.last_roman: Optional[str] = None
        self.gap = False

    def tokenize(self, content: str) -> List[Optional[str]]:
        tokens: List[Optional[str]] = []
        for raw in content.split("\n"):
            text = toline(raw)
            if not text or text == "[...]":
                tokens.append(None)
                continue
            heading = HEADING_RE.match(text)
            if heading is None:
                tokens.append(text)
            elif self.kind == "annex":
                tokens.append(heading.group(2))
            elif len(heading.group(1)) == 1:
                self.parsed.chapter = heading.group(2)
            elif len(heading.group(1)) == 3:
                self.parsed.heading = heading.group(2)
        return tokens

    def parse(self, content: str) -> ParsedChunk:
        tokens = self.tokenize(content)
        index = 0
        while index < len(tokens):
            text = tokens[index]
            if text is None:
                self.gap = True
            else:
                text = self.split_inline(text, index, tokens)
                if not self.try_para(text) and not self.try_item(text, index, tokens):
                    self.target.add(text, self.gap)
                    self.gap = False
            index += 1
        self.close_block()
        return self.parsed

    def split_inline(self, text: str, index: int, tokens: List[Optional[str]]) -> str:
        """Cut off a paragraph marker that is glued to the end of the line."""
        key = self.last_para
        match = self.para_re.match(text)
        if match is not None and self.para_accepted(match.group(1)):
            key = match.group(1)
        if key is None:
            return text
        for inline in self.inline_re.finditer(text):
            if inline.group(1) in next_para_keys(key):
                tokens.insert(index + 1, text[inline.start() + 1 :])
                self.inline_splits.append(f"{self.title}: {inline.group(0).strip()}")
                return text[: inline.start()]
        return text

    def close_block(self) -> None:
        if not self.block.is_empty() or self.block.key is not None:
            self.parsed.blocks.append(self.block)

    def para_accepted(self, key: str) -> bool:
        if self.last_para is None:
            return True
        if split_key(key) <= split_key(self.last_para):
            return False
        return self.gap or key in next_para_keys(self.last_para)

    def try_para(self, text: str) -> bool:
        match = self.para_re.match(text)
        if match is None or not self.para_accepted(match.group(1)):
            return False
        key = match.group(1)
        self.close_block()
        self.block = Block(key, self.join)
        self.block.head.add(text)
        self.target = self.block.head
        self.cur_item = None
        self.last_para = key
        self.last_letter = None
        self.last_roman = None
        self.gap = False
        return True

    def match_item(self, text: str) -> Optional[str]:
        match = ITEM_RE.match(text)
        if match is not None:
            return match.group(1) or match.group(2)
        if self.kind == "annex":
            match = ITEM_DOT_RE.match(text)
            if match is not None:
                return match.group(1)
        return None

    def letter_allowed(self, token: str) -> bool:
        if len(token) > 2:
            return False
        if self.last_letter is None:
            return token == "a" or self.block.fragment
        if token == "a":
            return True
        if self.gap:
            return letter_order(token) > letter_order(self.last_letter)
        return token in letter_successors(self.last_letter)

    def next_marker_token(
        self, index: int, tokens: List[Optional[str]]
    ) -> Optional[str]:
        for text in tokens[index + 1 :]:
            if text is None:
                continue
            token = self.match_item(text)
            if token is not None or self.para_re.match(text):
                return token
        return None

    def classify(
        self, token: str, index: int, tokens: List[Optional[str]]
    ) -> Optional[str]:
        letter_ok = self.letter_allowed(token)
        roman_ok = self.cur_item is not None and token == next_roman(self.last_roman)
        if letter_ok and roman_ok:
            following = self.next_marker_token(index, tokens)
            return "roman" if following == next_roman(token) else "letter"
        if letter_ok:
            return "letter"
        return "roman" if roman_ok else None

    def try_item(self, text: str, index: int, tokens: List[Optional[str]]) -> bool:
        token = self.match_item(text)
        if token is None:
            return False
        if (
            self.block.key is None
            and self.block.is_empty()
            and self.last_para is None
            and self.title_abs is not None
        ):
            self.block.key = self.title_abs
            self.block.fragment = True
        kind = self.classify(token, index, tokens)
        if kind is None:
            raise IngestError(
                f"{self.title}: unexpected marker '{token}' after letter "
                f"'{self.last_letter}': {text[:60]}"
            )
        lines = Lines(self.join)
        lines.add(text)
        if kind == "letter":
            if not self.block.runs or (token == "a" and self.last_letter is not None):
                self.block.runs.append({})
            run = self.block.runs[-1]
            if token in run:
                raise IngestError(f"{self.title}: letter '{token}' appears twice")
            self.cur_item = Item(lines)
            run[token] = self.cur_item
            self.last_letter = token
            self.last_roman = None
        else:
            self.cur_item.subs[token] = lines
            self.last_roman = token
        self.target = lines
        self.gap = False
        return True


def check_numbers(noun: str, keys: List[str]) -> None:
    bases: List[int] = []
    previous_suffix = ""
    for key in sorted(keys, key=split_key):
        number, suffix = split_key(key)
        if suffix:
            if number not in bases:
                raise IngestError(f"{noun} {key} has no base {noun} {number}")
            expected = "a" if not previous_suffix else chr(ord(previous_suffix) + 1)
            if suffix != expected:
                raise IngestError(f"{noun} {number}{expected} is missing (found {key})")
            previous_suffix = suffix
        else:
            expected_number = bases[-1] + 1 if bases else 1
            if number != expected_number:
                previous = f"{bases[-1]}" if bases else "the start"
                raise IngestError(
                    f"{noun} {expected_number} is missing (found {number} after "
                    f"{previous})"
                )
            bases.append(number)
            previous_suffix = ""


class LanguageIngestor:
    def __init__(self, cfg: LangCfg, path: str):
        self.cfg = cfg
        self.path = path
        self.problems: List[str] = []
        self.inline_splits: List[str] = []
        self.used: Set[int] = set()
        self.failed: Set[str] = set()
        self.articles: Dict[str, dict] = {}
        self.recitals: Dict[str, dict] = {}
        self.omnibus: Dict[str, dict] = {}
        self.annexes: Dict[str, dict] = {}

    def problem(self, msg: str) -> None:
        self.problems.append(f"[{self.cfg.code}] {msg}")

    def run(self) -> None:
        with open(self.path, encoding="utf-8") as fh:
            chunks = json.load(fh)["chunks"]
        article_chunks: Dict[str, List[Tuple[int, dict]]] = {}
        annex_chunks: Dict[str, List[Tuple[int, dict]]] = {}
        recital_chunks: List[Tuple[int, dict]] = []
        law_indexes: Set[int] = set()
        for index, chunk in enumerate(chunks):
            title = (chunk.get("title") or "").strip()
            if not self.cfg.law_re.match(title):
                continue
            law_indexes.add(index)
            article = ARTICLE_TITLE_RE.match(title)
            annex = self.cfg.annex_re.match(title)
            if article:
                article_chunks.setdefault(article.group(1), []).append((index, chunk))
            elif annex and annex.group(1) in ANNEX_NUMBERS:
                annex_chunks.setdefault(annex.group(1), []).append((index, chunk))
            elif self.cfg.recital_re.match(title):
                recital_chunks.append((index, chunk))
            else:
                self.problem(f"law chunk with unrecognised title: '{title}'")
        for number, items in article_chunks.items():
            indexes = [index for index, _ in items]
            self.guard(f"article {number}", indexes, self.build_article, number, items)
        for index, chunk in recital_chunks:
            match = self.cfg.recital_re.match(chunk["title"].strip())
            label = "omnibus recital" if match.group(2) else "recital"
            self.guard(
                f"{label} {match.group(1)}", [index], self.build_recital, index, chunk
            )
        for number, items in annex_chunks.items():
            indexes = [index for index, _ in items]
            self.guard(f"annex {number}", indexes, self.build_annex, number, items)
        self.finish(law_indexes)

    def guard(self, unit: str, indexes: List[int], func, *args) -> None:
        try:
            func(*args)
        except IngestError as exc:
            self.failed.add(unit)
            self.used.update(indexes)
            self.problem(f"{unit}: {exc}")

    def finish(self, law_indexes: Set[int]) -> None:
        for index in sorted(law_indexes - self.used):
            self.problem(f"law chunk #{index} was not used")
        expected_articles = [str(n) for n in range(1, 114)] + ARTICLE_EXTRAS
        self.check_set("article", self.articles, expected_articles)
        self.check_set("recital", self.recitals, [str(n) for n in range(1, 181)])
        self.check_set("omnibus recital", self.omnibus, [str(n) for n in range(1, 48)])
        self.check_set("annex", self.annexes, ANNEX_NUMBERS)
        art3 = self.articles.get("3")
        if art3 is not None:
            found = [p["number"] for p in art3["paragraphs"]]
            if sorted(found) != sorted(ART3_POINTS):
                missing = sorted(set(ART3_POINTS) - set(found), key=split_key)
                extra = sorted(set(found) - set(ART3_POINTS), key=split_key)
                self.problem(
                    f"article 3 must have points 1-68 plus 14a, 14b: "
                    f"missing {missing}, unexpected {extra}"
                )
        previous: Optional[str] = None
        for number in sorted(self.articles, key=split_key):
            article = self.articles[number]
            if article["chapter"] is None:
                article["chapter"] = previous
            previous = article["chapter"]

    def check_set(
        self, label: str, found: Dict[str, dict], expected: List[str]
    ) -> None:
        missing = [
            key
            for key in expected
            if key not in found and f"{label} {key}" not in self.failed
        ]
        extra = [key for key in found if key not in expected]
        if missing:
            self.problem(f"missing {label}(s): {', '.join(missing)}")
        if extra:
            self.problem(f"unexpected {label}(s): {', '.join(extra)}")

    # -- articles -------------------------------------------------------------

    def build_article(self, number: str, items: List[Tuple[int, dict]]) -> None:
        kind = "art3" if number == "3" else "art"
        para_re = self.cfg.art3_para_re if number == "3" else self.cfg.article_para_re
        heading: Optional[str] = None
        chapter: Optional[str] = None
        intro: Optional[Block] = None
        paragraphs: Dict[str, Block] = {}
        for index, chunk in items:
            title = chunk["title"].strip()
            parser = ChunkParser(title, kind, para_re, True)
            parsed = parser.parse(chunk["content"])
            self.inline_splits.extend(parser.inline_splits)
            if parsed.heading is not None:
                if heading is not None and heading != parsed.heading:
                    raise IngestError(f"chunks disagree on the heading: '{title}'")
                heading = parsed.heading
            if parsed.chapter is not None:
                if chapter is not None and chapter != parsed.chapter:
                    raise IngestError(f"chunks disagree on the chapter: '{title}'")
                chapter = parsed.chapter
            for block in parsed.blocks:
                try:
                    if block.key is None:
                        if intro is None:
                            intro = block
                        else:
                            intro.merge(block)
                    elif block.key not in paragraphs:
                        paragraphs[block.key] = block
                    else:
                        paragraphs[block.key].merge(block)
                except IngestError as exc:
                    raise IngestError(f"{title}: paragraph {block.key}: {exc}")
            self.used.add(index)
        if heading is None or not re.match(
            rf"^(?:Artikel|Article) {re.escape(number)}\b", heading
        ):
            raise IngestError(f"heading '{heading}' does not name article {number}")
        for key, block in paragraphs.items():
            try:
                block.check_items()
            except IngestError as exc:
                raise IngestError(f"paragraph {key}: {exc}")
        if intro is not None:
            intro.check_items()
        if paragraphs:
            check_numbers("paragraph", list(paragraphs))
        self.articles[number] = {
            "number": number,
            "heading": heading,
            "chapter": chapter,
            "text": intro.render() if intro is not None else "",
            "paragraphs": [
                {"number": key, "text": paragraphs[key].render()}
                for key in sorted(paragraphs, key=split_key)
            ],
        }

    # -- recitals -------------------------------------------------------------

    def build_recital(self, index: int, chunk: dict) -> None:
        title = chunk["title"].strip()
        match = self.cfg.recital_re.match(title)
        number = match.group(1)
        target = self.omnibus if match.group(2) else self.recitals
        lines = Lines(True, bullets=True)
        gap = False
        for raw in chunk["content"].split("\n"):
            text = toline(raw)
            if not text:
                gap = True
                continue
            lines.add(text, gap)
            gap = False
        text = "\n".join(lines.lines)
        if not text.startswith(f"({number})"):
            raise IngestError(f"content does not start with ({number})")
        if number in target:
            raise IngestError("appears twice")
        target[number] = {"number": number, "text": text}
        self.used.add(index)

    # -- annexes --------------------------------------------------------------

    def build_annex(self, number: str, items: List[Tuple[int, dict]]) -> None:
        heading: Optional[str] = None
        intro: Optional[Block] = None
        points: Dict[str, Block] = {}
        plain = Lines(False)
        for index, chunk in items:
            title = chunk["title"].strip()
            raw_lines = chunk["content"].split("\n")
            first = next((i for i, ln in enumerate(raw_lines) if ln.strip()), None)
            if first is None:
                raise IngestError(f"{title}: chunk is empty")
            chunk_heading = HEADING_RE.sub(r"\2", toline(raw_lines[first]))
            if heading is not None and heading != chunk_heading:
                raise IngestError(f"chunks disagree on the heading: '{title}'")
            heading = chunk_heading
            body = "\n".join(raw_lines[first + 1 :])
            if number in ANNEX_POINT_LISTS:
                parser = ChunkParser(title, "annex", DOT_NUM_RE, False)
                parsed = parser.parse(body)
                self.inline_splits.extend(parser.inline_splits)
                for block in parsed.blocks:
                    try:
                        if block.key is None:
                            if intro is None:
                                intro = block
                            else:
                                intro.merge(block)
                        elif block.key not in points:
                            points[block.key] = block
                        else:
                            points[block.key].merge(block)
                    except IngestError as exc:
                        raise IngestError(f"{title}: point {block.key}: {exc}")
            else:
                for raw in body.split("\n"):
                    text = toline(raw)
                    if text and text != "[...]":
                        plain.add(HEADING_RE.sub(r"\2", text))
            self.used.add(index)
        if not re.match(rf"(?i)^{self.cfg.annex} {number}\b", heading or ""):
            raise IngestError(f"heading '{heading}' does not name annex {number}")
        if number in ANNEX_POINT_LISTS:
            if not points:
                raise IngestError("no numbered points found")
            for key, block in points.items():
                try:
                    block.check_items()
                except IngestError as exc:
                    raise IngestError(f"point {key}: {exc}")
            if intro is not None:
                intro.check_items()
            check_numbers("point", list(points))
            text = intro.render() if intro is not None else ""
        else:
            text = "\n".join(plain.lines)
        self.annexes[number] = {
            "number": number,
            "heading": heading,
            "text": text,
            "paragraphs": [
                {"number": key, "text": points[key].render()}
                for key in sorted(points, key=split_key)
            ],
        }

    # -- output ---------------------------------------------------------------

    def build(self) -> dict:
        def by_number(units: Dict[str, dict]) -> List[dict]:
            return [units[key] for key in sorted(units, key=split_key)]

        return {
            "language": self.cfg.code,
            "source": os.path.basename(self.path),
            "articles": by_number(self.articles),
            "recitals": by_number(self.recitals),
            "annexes": [self.annexes[n] for n in ANNEX_NUMBERS if n in self.annexes],
            "omnibus_recitals": by_number(self.omnibus),
        }

    def paragraph_keys(self) -> Dict[str, List[str]]:
        keys = {}
        for number, article in self.articles.items():
            keys[f"article {number}"] = [p["number"] for p in article["paragraphs"]]
        for number, annex in self.annexes.items():
            keys[f"annex {number}"] = [p["number"] for p in annex["paragraphs"]]
        return keys

    def summary(self) -> str:
        articles = list(self.articles.values())
        with_paragraphs = [a for a in articles if a["paragraphs"]]
        total = sum(len(a["paragraphs"]) for a in articles)
        no_points = [
            n for n in ANNEX_NUMBERS if not self.annexes.get(n, {}).get("paragraphs")
        ]
        return (
            f"{self.cfg.code}: articles={len(articles)} "
            f"(with paragraphs={len(with_paragraphs)}, paragraphs/points={total}), "
            f"recitals={len(self.recitals)}, annexes={len(self.annexes)} "
            f"(without points: {', '.join(no_points)}), "
            f"omnibus_recitals={len(self.omnibus)}"
        )


def compare_languages(first: LanguageIngestor, second: LanguageIngestor) -> List[str]:
    problems = []
    keys_a = first.paragraph_keys()
    keys_b = second.paragraph_keys()
    for unit in sorted(set(keys_a) | set(keys_b)):
        if unit in first.failed or unit in second.failed:
            continue
        if unit not in keys_a or unit not in keys_b:
            side = first.cfg.code if unit in keys_b else second.cfg.code
            problems.append(f"{unit} is missing in {side}")
        elif keys_a[unit] != keys_b[unit]:
            problems.append(
                f"{unit}: paragraph keys differ: "
                f"{first.cfg.code}={keys_a[unit]} {second.cfg.code}={keys_b[unit]}"
            )
    for label, left, right in (
        ("recital", first.recitals, second.recitals),
        ("omnibus recital", first.omnibus, second.omnibus),
    ):
        failed = first.failed | second.failed
        diff = sorted(
            (key for key in set(left) ^ set(right) if f"{label} {key}" not in failed),
            key=split_key,
        )
        if diff:
            problems.append(f"{label}s differ between languages: {diff}")
    return [f"[de vs en] {p}" for p in problems]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--de", default=os.path.join(DATA_DIR, "combined.json"))
    parser.add_argument("--en", default=os.path.join(DATA_DIR, "combined_en_new.json"))
    parser.add_argument("--out-dir", default=SCRIPT_DIR)
    args = parser.parse_args()
    paths = {"de": args.de, "en": args.en}

    ingestors: List[LanguageIngestor] = []
    for cfg in LANGS:
        if not os.path.isfile(paths[cfg.code]):
            fail(f"input file not found: {paths[cfg.code]}")
        ingestor = LanguageIngestor(cfg, paths[cfg.code])
        ingestor.run()
        ingestors.append(ingestor)

    for ingestor in ingestors:
        for split in ingestor.inline_splits:
            print(
                f"WARNING: [{ingestor.cfg.code}] paragraph marker glued to the "
                f"previous line, split off: {split}",
                file=sys.stderr,
            )
    problems = [p for ingestor in ingestors for p in ingestor.problems]
    problems.extend(compare_languages(ingestors[0], ingestors[1]))
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        print("no files written", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    for ingestor in ingestors:
        out_path = os.path.join(
            args.out_dir, f"ai_act_structured_{ingestor.cfg.code}.json"
        )
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(ingestor.build(), fh, ensure_ascii=False, indent=2)
        print(f"{out_path}: {ingestor.summary()}")


if __name__ == "__main__":
    main()
