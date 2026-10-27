#!/usr/bin/env python3
"""Turn EU Commission guideline PDFs into per-paragraph chunk JSON files.

Usage: python3 ingest_guidelines.py <pdf_dir> [<out_dir>]

The guideline PDFs are matched by filename substring; every other file in
<pdf_dir> is ignored. Each document is split into one chunk per numbered
paragraph plus the unnumbered text that follows it until the next paragraph
or section heading. Footnotes and page furniture are removed from the text.
Output files <id>_<lang>.json in ManualIndex (ChunkNode) shape are written to
<out_dir>; only the script and the generated JSON are committed.
"""

import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

ROMAN_HEAD_RE = re.compile(r"^\s*([IVXL]{1,5})\.\s+(\S.*)$")
NUMBER_HEAD_RE = re.compile(r"^\s*(\d{1,3}(?:\.\d{1,3})*)\.?\s+(\S.*)$")
ANNEX_HEAD_RE = re.compile(r"^\s*(A(?:\.\d{1,3})+)\.?\s+(\S.*)$")
ANNEX_TOP_RE = re.compile(r"^\s*(Anhang|Annex):\s+(\S.*)$")
LETTER_HEAD_RE = re.compile(r"^\s*([a-z])\)\s+(\S.*)$")
PAREN_PARA_RE = re.compile(r"^\s*\((\d{1,4})\)(?:\s+(.*))?$")
DOT_PARA_RE = re.compile(r"^(\d{1,4})\.(?:\s+(.*))?$")
BARE_FOOT_RE = re.compile(r"^\s{0,5}(\d{1,4})\s{0,5}$")
PAREN_FOOT_RE = re.compile(r"^\((\d{1,4})\)\s+(.*)$")
TOC_LEADER_RE = re.compile(r"\.{5,}")
TOC_NUM_RE = re.compile(r"^((?:[A-Z]\.)?\d{1,3}(?:\.\d{1,3})*)\.?\s+(.*)$")
EXAMPLE_LABEL_RE = re.compile(r"^\s*(Examples?|Beispiele)\s*:\s*$")
LETTER_ITEM_END_RE = re.compile(r"[,;]\s*$")
LIST_BREAK_RE = re.compile(
    r"^\s*(?:[„•–—-]\s+|\(?[a-z]\)\s|\(?[ivx]{1,4}\)\s|o\s+\S)"
)
CONJUNCTIONS = {"und", "oder", "bzw", "and", "or"}
LETTERS = "abcdefghijklmnopqrstuvwxyz"


class DocCfg:
    def __init__(
        self,
        doc_id: str,
        display: Dict[str, str],
        label: Dict[str, str],
        para_style,
        foot_style: str,
        file_stems: Dict[str, str],
        letter_headings: bool = False,
        roman_headings: bool = False,
        annex: bool = False,
    ):
        self.id = doc_id
        self.display = display
        self.label = label
        self.para_style = para_style
        self.foot_style = foot_style
        self.file_stems = file_stems
        self.letter_headings = letter_headings
        self.roman_headings = roman_headings
        self.annex = annex
        self.langs = sorted(file_stems)

    def para_kind(self, lang: str) -> str:
        kind = self.para_style
        if isinstance(kind, dict):
            return kind[lang]
        return kind(lang) if callable(kind) else kind


DOCS = [
    DocCfg(
        "prohibited_practices",
        {"en": "Guidelines on prohibited AI practices", "de": "LL Verbotene Praktiken"},
        {"en": "para", "de": "Rz"},
        "paren",
        "bare",
        {
            "en": "Guidelines_on_prohibited_artificial_intelligence_practices",
            "de": "Guidelines_on_prohibited_artificial_intelligence_practices",
        },
        letter_headings=True,
    ),
    DocCfg(
        "ai_system_definition",
        {
            "en": "Guidelines on the AI system definition",
            "de": "LL KI-System",
        },
        {"en": "para", "de": "Rz"},
        "paren",
        "bare",
        {
            "en": "Commission_Guidelines_on_the_definition_of_an_artificial_intelligence_system",
            "de": "Commission_Guidelines_on_the_definition_of_an_artificial_intelligence_system",
        },
        roman_headings=True,
    ),
    DocCfg(
        "gpai_obligations",
        {"en": "Guidelines on GPAI obligations", "de": "LL GPAI"},
        {"en": "para", "de": "Rz"},
        {"en": "paren", "de": "dot"},
        "bare",
        {
            "en": "Guidelines_on_the_scope_of_the_obligations_for_generalpurpose_AI_models",
            "de": "Guidelines_on_the_scope_of_the_obligations_for_generalpurpose_AI_models",
        },
        annex=True,
    ),
    DocCfg(
        "transparency_art50",
        {"en": "Guidelines on Art 50 transparency"},
        {"en": "para"},
        "paren",
        "paren",
        {
            "en": "Guidelines_on_the_implementation_of_the_transparency_obligations",
        },
    ),
]

FILELANG_TOKENS = {"en": "English", "de": "German"}


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def run_pdftotext(pdf_path: str) -> str:
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", pdf_path, "-"],
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        fail("pdftotext not found, please install poppler-utils")
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
        fail(f"pdftotext failed for {pdf_path}: {stderr}")
    return result.stdout.decode("utf-8", errors="replace")


def find_pdf(pdf_dir: str, cfg: DocCfg, lang: str) -> str:
    matches = [
        name
        for name in sorted(os.listdir(pdf_dir))
        if name.lower().endswith(".pdf") and cfg.file_stems[lang] in name
    ]
    if len(matches) > 1 and len(cfg.langs) > 1:
        token = FILELANG_TOKENS[lang]
        matches = [name for name in matches if token in name]
    if not matches:
        fail(f"no PDF matching '{cfg.file_stems[lang]}' in {pdf_dir}")
    if len(matches) > 1:
        fail(f"ambiguous PDF match for {cfg.id} {lang}: {matches}")
    return os.path.join(pdf_dir, matches[0])


def toline(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def page_lines(page_text: str) -> List[str]:
    return page_text.splitlines()


def is_toc_page(lines: List[str]) -> bool:
    return sum(1 for ln in lines if TOC_LEADER_RE.search(ln)) >= 2


def strip_page_bottom(lines: List[str]) -> List[str]:
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and re.fullmatch(r"\s{6,}(\d{1,4}|[IVXL]{1,6})\s*", lines[-1]):
        lines.pop()
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def extract_footnotes(
    lines: List[str], style: str, expected: int
) -> Tuple[Optional[List[Tuple[int, str]]], int]:
    """Pull the trailing footnote block off one page.

    Returns (units, cut): units is a list of (number, text) in document order
    forming a consecutive run starting at *expected*; the block must be the
    tail of the page. cut is the index of the first block line, so
    lines[:cut] is the page body. (None, len(lines)) means no valid footnote
    block was found and the page is left untouched.
    """
    units_rev: List[Tuple[int, str]] = []
    cut = len(lines)
    stop_num: Optional[int] = None
    i = len(lines)
    foot_marker_re = BARE_FOOT_RE if style == "bare" else PAREN_FOOT_RE
    while True:
        k = i
        while k > 0 and not foot_marker_re.match(lines[k - 1]):
            k -= 1
        if k == 0:
            break
        marker = lines[k - 1]
        first_text = ""
        if style == "bare":
            m = BARE_FOOT_RE.match(marker)
        else:
            m = PAREN_FOOT_RE.match(marker)
            if m:
                first_text = m.group(2)
        if not m:
            break
        num = int(m.group(1))
        text_lines = [ln for ln in lines[k:i] if ln.strip()]
        text = " ".join([first_text] + [toline(ln) for ln in text_lines])
        units_rev.append((num, toline(text)))
        i = k - 1
        if num == expected:
            stop_num = num
            cut = k - 1
            break
    if not units_rev or stop_num != expected:
        return None, len(lines)
    units = list(reversed(units_rev))
    nums = [u[0] for u in units]
    if nums[0] != expected or any(
        nums[p + 1] - nums[p] != 1 for p in range(len(nums) - 1)
    ):
        return None, len(lines)
    return units, cut


def parse_toc(pages: List[str], toc_idx: List[int]) -> Tuple[set, Dict[str, str]]:
    numbers: set = set()
    titles: Dict[str, str] = {}
    pending: Optional[str] = None
    pending_words: List[str] = []
    no_leader: set = set()

    def finish() -> None:
        nonlocal pending, pending_words
        if pending is None:
            return
        title = toline(" ".join(pending_words))
        title = re.sub(r"(\w)- (\w)", r"\1-\2", title)
        title = re.sub(r"\s{1,3}\d{1,4}$", "", title)
        if pending in no_leader:
            title = re.sub(r"([^\W\d_])\d{1,4}$", r"\1", title)
        if title:
            numbers.add(pending)
            titles.setdefault(pending, title)
        pending = None
        pending_words = []

    for idx in toc_idx:
        for raw in page_lines(pages[idx]):
            line = raw.rstrip()
            stripped = line.strip()
            if not stripped:
                continue
            head = TOC_LEADER_RE.split(line)[0].strip()
            entry = TOC_NUM_RE.match(head)
            annex = ANNEX_TOP_RE.match(stripped)
            if entry:
                finish()
                num = entry.group(1).rstrip(".")
                if TOC_LEADER_RE.search(line):
                    numbers.add(num)
                    titles.setdefault(num, toline(entry.group(2)))
                else:
                    pending = num
                    pending_words = [entry.group(2)]
                    no_leader.add(num)
            elif annex:
                finish()
                numbers.add("Anhang")
                titles.setdefault("Anhang", toline(annex.group(2)))
            elif TOC_LEADER_RE.search(line):
                if pending is not None:
                    pending_words.append(head)
                finish()
            elif pending is not None:
                pending_words.append(stripped)
    finish()
    return numbers, titles


def title_startswith(head_text: str, toc_title: str) -> bool:
    """First three words of the heading must match the TOC title."""
    want = toline(toc_title).casefold().split()[:3]
    got = toline(head_text).casefold().split()
    return got[: len(want)] == want


def title_continues(got: str, want: str) -> bool:
    """Accumulated heading text is still a prefix of the TOC title.

    Allows the last word to be a line-wrap split ('KI-' truncating
    'KI-Systems') and tolerates a page number glued to the last TOC title
    word (e.g. 'Act45').
    """
    want_words = toline(want).casefold().split()
    got_words = toline(got).casefold().split()
    if len(got_words) > len(want_words):
        return False
    for i, gw in enumerate(got_words):
        ww = want_words[i]
        if i == len(got_words) - 1:
            if not ww.startswith(gw):
                return False
        elif gw != ww:
            return False
    return True


def assemble(blocks: List[List[str]]) -> str:
    pieces = []
    for block in blocks:
        text = ""
        for raw in block:
            line = re.sub(r"\s+", " ", raw).strip()
            if not line:
                continue
            if text.endswith("-") and line[:1].isalpha():
                first_word = toline(line).split()[0]
                if first_word in CONJUNCTIONS:
                    text = f"{text} {line}"
                    continue
                text = f"{text}{line}"
            elif text:
                text = f"{text} {line}"
            else:
                text = line
        if text:
            pieces.append(text)
    return "\n".join(pieces).strip()


GLUED_FOOT_RE = re.compile(
    r"(?:(?<=[^\W\d_])|(?<=[^\w\s(]))\d{1,4}(?!\d)"
    r"|(?<=\d{4})\d{1,3}(?![\w\d./])"
    r"|(?<=[^\w\s(] )\d{1,4}(?![\w\d)])"
    r"|(?<=\w )\d{1,4}(?!\.?\d)"
)
PAREN_MARK_RE = re.compile(r"(?<=\s)\((\d{1,4})\)")


CITATION_PREFIX_RE = re.compile(
    r"(recital|erwägungsgrund|article|artikel|nummer|absatz|annex|anhang|abschnitt"
    r"|section|paragraph|abl\.|[sb]\.\s)$",
    re.IGNORECASE,
)
RESIDUAL_MARKER_RE = re.compile(r"([a-zäöüß’)])(\d{1,3})\b")


class Ingestor:
    def __init__(self, cfg: DocCfg, lang: str, pdf_path: str):
        self.cfg = cfg
        self.lang = lang
        self.para_kind = cfg.para_kind(lang)
        self.pages = run_pdftotext(pdf_path).split("\f")
        self.expected_para = 1
        self.next_foot = 1
        self.footnotes: Dict[int, str] = {}
        self.footnote_pages: Dict[int, int] = {}
        self.assigned: Dict[int, Tuple[int, int]] = {}
        self.toc_numbers: set = set()
        self.toc_titles: Dict[str, str] = {}
        self.section: Optional[str] = None
        self.numbered_num: Optional[str] = None
        self.expected_letter = 0
        self.paras: List[dict] = []
        self.cur: Optional[dict] = None
        self.orphan: List[List[Tuple[int, str]]] = []
        self.break_next = False
        self.para_indent = 0
        self.flow_indent: Optional[int] = None
        self.after_heading = False
        self.heading_cont: Optional[dict] = None
        self.anomalies: List[str] = []
        self.marker_warnings: List[str] = []
        self.strip_counter = 1
        self.prev_content_line: Optional[str] = None
        self.current_page = 0

    # -- body location ------------------------------------------------------

    def locate_body(self) -> List[Tuple[int, int]]:
        toc_idx = [
            i for i, p in enumerate(self.pages[:15]) if is_toc_page(p.splitlines())
        ]
        first_body = max(toc_idx) + 1 if toc_idx else 0
        if toc_idx:
            self.toc_numbers, self.toc_titles = parse_toc(self.pages, toc_idx)
        for p in range(first_body, len(self.pages)):
            lines = page_lines(self.pages[p])
            for i, line in enumerate(lines):
                regex = PAREN_PARA_RE if self.para_kind == "paren" else DOT_PARA_RE
                m = regex.match(line)
                if m and int(m.group(1)) == 1:
                    start = self.backtrack(lines, i)
                    rest = [(q, 0) for q in range(p + 1, len(self.pages))]
                    return [(p, start)] + rest
        fail(f"[{self.cfg.id}_{self.lang}] paragraph 1 not found")

    def backtrack(self, lines: List[str], at: int) -> int:
        i, kept = at, 0
        while i > 0 and kept < 2:
            i -= 1
            stripped = lines[i].strip()
            if not stripped:
                continue
            if (
                NUMBER_HEAD_RE.match(lines[i])
                or ROMAN_HEAD_RE.match(lines[i])
                or LETTER_HEAD_RE.match(lines[i])
                or DOT_PARA_RE.match(lines[i])
                or PAREN_PARA_RE.match(lines[i])
            ):
                kept += 1
                continue
            i += 1
            break
        return i

    # -- main -------------------------------------------------------------

    def ingest(self) -> None:
        for p, start in self.locate_body():
            self.current_page = p + 1
            lines = page_lines(self.pages[p])
            lines = strip_page_bottom(lines)
            lines = [ln for ln in lines if ln.strip() not in ("EN", "DE")]
            units, cut = extract_footnotes(lines, self.cfg.foot_style, self.next_foot)
            body = lines if units is None else lines[:cut]
            if units is not None:
                for num, text in units:
                    if num in self.footnotes:
                        self.anomalies.append(f"duplicate footnote page text {num}")
                    self.footnotes[num] = text
                    self.footnote_pages[num] = self.current_page
                self.next_foot = units[-1][0] + 1
            if start > 0:
                body = body[start:]
            for line in body:
                self.classify(line)
        self.close_para()

    # -- classification -----------------------------------------------------

    def classify(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            self.break_next = True
            self.heading_cont = None
            return
        if self.para_scan_heading(line):
            return
        if self.heading_cont is not None and self.absorb_heading(line):
            return
        if self.try_numbered_heading(line):
            return
        if self.cfg.roman_headings:
            m = ROMAN_HEAD_RE.match(line)
            if m:
                self.heading_cont = {
                    "mode": "indent",
                    "indent": indent_of(line),
                    "absorbed": 0,
                }
                self.section = f"{m.group(1)} {toline(m.group(2))}"
                self.numbered_num = None
                self.expected_letter = 0
                self.after_heading = True
                return
        if self.cfg.annex:
            m = ANNEX_TOP_RE.match(line)
            if m:
                self.section = toline(m.group(0))
                self.numbered_num = None
                self.expected_letter = 0
                self.after_heading = True
                self.start_block()
                return
            m = ANNEX_HEAD_RE.match(line)
            if m and self.heading_ok(m.group(1), m.group(2)):
                self.section = f"{m.group(1)} {toline(m.group(2))}"
                self.numbered_num = m.group(1)
                self.expected_letter = 0
                self.after_heading = True
                self.heading_cont = (
                    {
                        "mode": "toc",
                        "num": m.group(1),
                        "text": toline(m.group(2)),
                        "absorbed": 0,
                    }
                    if self.toc_numbers
                    else None
                )
                self.start_block()
                return
        if self.cfg.letter_headings:
            m = LETTER_HEAD_RE.match(line)
            if m and LETTERS.index(m.group(1)) == self.expected_letter:
                if (
                    self.prev_content_line
                    and EXAMPLE_LABEL_RE.match(self.prev_content_line)
                ) or LETTER_ITEM_END_RE.search(line):
                    # list item inside a quote or an example box, not a heading
                    self.feed(line)
                    return
                label = toline(m.group(2))
                prefix = (
                    f"{self.numbered_num} {m.group(1)}) "
                    if self.numbered_num
                    else f"{m.group(1)}) "
                )
                self.section = f"{prefix}{label}"
                self.expected_letter += 1
                self.heading_cont = {
                    "mode": "letter",
                    "indent": indent_of(line),
                    "absorbed": 0,
                }
                self.after_heading = True
                self.start_block()
                return
        self.feed(line)

    def para_scan_heading(self, line: str) -> bool:
        regex = PAREN_PARA_RE if self.para_kind == "paren" else DOT_PARA_RE
        m = regex.match(line)
        if m is None:
            return False
        num = int(m.group(1))
        text_after = m.group(2)
        if num == self.expected_para:
            if self.is_exact_toc_heading(line):
                # a section heading can carry the same number as the paragraph
                # that follows it (GPAI German doc); only an exact title match
                # with the TOC entry is a heading, everything else is a paragraph
                return True
            text = (text_after or "").strip()
            self.new_para(num, text, line)
            return True
        if self.is_toc_heading(line):
            m_num = NUMBER_HEAD_RE.match(line)
            self.register_numbered_heading(
                m_num.group(1).rstrip("."), m_num.group(2), line
            )
            return True
        if num > self.expected_para:
            self.anomalies.append(
                f"unexpected paragraph marker {num}: {line.strip()[:80]}"
            )
        return False

    def is_exact_toc_heading(self, line: str) -> bool:
        m_num = NUMBER_HEAD_RE.match(line)
        if m_num is None:
            return False
        key = m_num.group(1).rstrip(".")
        title = self.toc_titles.get(key, "")
        if key not in self.toc_numbers or not title:
            return False
        if toline(m_num.group(2)).casefold() != title.casefold():
            return False
        self.section = f"{key} {title}"
        self.numbered_num = key
        self.expected_letter = 0
        self.after_heading = True
        self.heading_cont = (
            {"mode": "toc", "num": key, "text": title, "absorbed": 0}
            if self.toc_numbers
            else None
        )
        self.start_block()
        return True

    def is_toc_heading(self, line: str) -> bool:
        if not self.toc_numbers:
            return False
        m = NUMBER_HEAD_RE.match(line)
        if m is None:
            return False
        key = m.group(1).rstrip(".")
        if key not in self.toc_numbers:
            return False
        return title_startswith(m.group(2), self.toc_titles.get(key, ""))

    def heading_ok(self, key: str, text: str) -> bool:
        if key not in self.toc_numbers:
            return False
        return title_startswith(text, self.toc_titles.get(key, ""))

    def register_numbered_heading(self, key: str, text: str, line: str) -> None:
        self.section = f"{key} {toline(text)}"
        self.numbered_num = key
        self.expected_letter = 0
        self.after_heading = True
        self.heading_cont = (
            {"mode": "toc", "num": key, "text": toline(text), "absorbed": 0}
            if self.toc_numbers
            else None
        )
        self.start_block()

    def try_numbered_heading(self, line: str) -> bool:
        m = NUMBER_HEAD_RE.match(line)
        if m is None:
            return False
        key = m.group(1).rstrip(".")
        if not self.toc_numbers:
            # without a TOC accept single-level headings only with a trailing
            # dot ("1. Machine-based system"), never wrapped body lines
            # ("1. August 2024 in Kraft. Mit der Verordnung ..."); multi-level
            # numbers ("5.1 KI-Techniken ...") are unambiguous without the dot
            if "." not in key and not line.lstrip().startswith(f"{key}."):
                return False
            if re.search(r"\.\s+[A-ZÄÖÜ]", m.group(2)):
                return False
        elif not self.heading_ok(key, m.group(2)):
            return False
        self.register_numbered_heading(key, m.group(2), line)
        return True

    def absorb_heading(self, line: str) -> bool:
        state = self.heading_cont
        ind = indent_of(line)
        if state["mode"] == "toc":
            num = state["num"]
            tail = line.strip()
            if state["text"].endswith("-"):
                state["text"] = toline(f"{state['text']}{tail}")
            else:
                state["text"] = toline(f"{state['text']} {tail}")
            state["absorbed"] += 1
            if state["absorbed"] > 4 or not title_continues(
                state["text"], self.toc_titles.get(num, "")
            ):
                state["text"] = toline(
                    re.sub(re.escape(tail) + r"$", "", state["text"])
                )
                self.heading_cont = None
                return False
            self.section = f"{num} {state['text']}"
            return True
        if state["absorbed"] >= (3 if state["mode"] == "letter" else 2):
            self.heading_cont = None
            return False
        if LIST_BREAK_RE.match(line) or LETTER_HEAD_RE.match(line):
            self.heading_cont = None
            return False
        if not (ind >= state["indent"] + 2):
            self.heading_cont = None
            return False
        state["absorbed"] += 1
        self.section = toline(f"{self.section} {line.strip()}")
        return True

    def start_block(self) -> None:
        self.break_next = True

    def feed(self, line: str) -> None:
        if self.cur is None and not self.after_heading:
            self.anomalies.append(
                f"body line outside any paragraph dropped: {line.strip()[:60]}"
            )
            return
        sink = self.orphan if self.cur is None else self.cur["blocks"]
        ind = indent_of(line)
        if self.flow_indent is None:
            self.flow_indent = ind
        broken = self.break_next or abs(ind - self.flow_indent) >= 3
        if broken or not sink or LIST_BREAK_RE.match(line):
            sink.append([(self.current_page, line)])
            self.flow_indent = ind
        else:
            sink[-1].append((self.current_page, line))
        self.break_next = False
        self.prev_content_line = line

    # -- paragraphs ----------------------------------------------------------

    def new_para(self, num: int, text: str, line: str) -> None:
        self.close_para()
        self.expected_para = num + 1
        self.heading_cont = None
        blocks: List[List[Tuple[int, str]]] = []
        if text:
            blocks.append([(self.current_page, text)])
        if self.orphan:
            blocks = self.orphan + blocks
            self.orphan = []
        self.after_heading = False
        self.cur = {
            "paragraph": num,
            "blocks": blocks,
            "section": self.section,
        }
        if self.para_kind == "paren":
            marker = f"({num})"
            pos = line.index(marker)
            self.para_indent = pos + len(marker)
        else:
            self.para_indent = 0
        while self.para_indent < len(line) and line[self.para_indent] == " ":
            self.para_indent += 1
        self.flow_indent = None
        self.break_next = False

    def close_para(self) -> None:
        cur, self.cur = self.cur, None
        if cur is None:
            return
        matched: List[int] = []
        stripped_blocks: List[List[str]] = []
        for block in cur["blocks"]:
            stripped_blocks.append(
                [
                    self.strip_markers_line(page, text, cur["paragraph"], matched)
                    for page, text in block
                ]
            )
        content = assemble(stripped_blocks)
        self.paras.append(
            {
                "paragraph": cur["paragraph"],
                "content": content,
                "section": cur["section"],
                "footnotes": matched,
            }
        )

    def strip_markers_line(
        self, page: int, content: str, para: int, matched: List[int]
    ) -> str:
        if not content.strip():
            return ""
        regex = PAREN_MARK_RE if self.cfg.foot_style == "paren" else GLUED_FOOT_RE
        out: List[str] = []
        pos = 0
        prefix_gap = 16
        for m in regex.finditer(content):
            num = int(m.group(0) if self.cfg.foot_style != "paren" else m.group(1))
            if num != self.strip_counter:
                continue
            foot_page = self.footnote_pages.get(num)
            if foot_page is None or page not in (foot_page, foot_page - 1):
                # a real marker sits on the page of its footnote text (or the
                # previous page, when the text starts on the following page);
                # anything else is text, not a marker
                continue
            prefix = content[max(m.start() - prefix_gap, 0) : m.start()]
            if CITATION_PREFIX_RE.search(toline(prefix)):
                continue
            start = m.start()
            if start > 0 and content[start - 1] == " ":
                start -= 1
            out.append(content[pos:start])
            pos = m.end()
            matched.append(num)
            self.assigned[num] = (para, page)
            self.strip_counter = num + 1
        out.append(content[pos:])
        return toline("".join(out))

    # -- output ----------------------------------------------------------

    def build(self) -> dict:
        chunks = []
        for para in self.paras:
            if para["section"] is None:
                self.anomalies.append(
                    f"paragraph {para['paragraph']} has no section heading"
                )
            chunk = {
                "id": f"{self.cfg.id}_{self.lang}_{para['paragraph']}",
                "title": self.chunk_title(para),
                "content": para["content"],
                "keywords": [],
                "negativeKeywords": [],
                "relevantChunksIds": [],
                "parameters": [],
                "position": para["paragraph"] - 1,
                "document": self.cfg.id,
                "section": para["section"],
                "paragraph": para["paragraph"],
                "footnotes": [
                    {"number": num, "text": self.footnotes[num]}
                    for num in sorted(para["footnotes"])
                ],
            }
            chunks.append(chunk)
        return {
            "id": f"{self.cfg.id}_{self.lang}",
            "creation_date": 0,
            "last_updated": 0,
            "document": self.cfg.id,
            "language": self.lang,
            "chunks": chunks,
        }

    def chunk_title(self, para: dict) -> str:
        label = self.cfg.label[self.lang]
        section = para["section"] or "?"
        return f"{self.cfg.display[self.lang]}, {section}, {label} {para['paragraph']}"

    def validate(self, data: dict) -> None:
        for chunk in data["chunks"]:
            if not chunk["content"]:
                self.anomalies.append(
                    f"paragraph {chunk['paragraph']} has empty content"
                )
            if re.match(r"^(\(\d+\)|\d+\.)", chunk["content"]):
                self.anomalies.append(
                    f"paragraph {chunk['paragraph']} content starts with a marker: "
                    f"{chunk['content'][:60]}"
                )
        for num in sorted(self.assigned):
            para, page = self.assigned[num]
            chunk_no = para
            text_page = self.footnote_pages.get(num)
            if text_page is None or page not in (text_page, text_page - 1):
                self.anomalies.append(
                    f"footnote {num} marker on page {page} but its text is on "
                    f"page {text_page}"
                )
            for chunk in data["chunks"]:
                if chunk["paragraph"] == chunk_no:
                    text = toline(self.footnotes.get(num, "")).casefold()
                    if len(text) >= 20 and text in toline(chunk["content"]).casefold():
                        self.anomalies.append(
                            f"paragraph {chunk_no} contains footnote {num} text"
                        )

    def scan_residual_markers(self, data: dict) -> None:
        foot_nums = set(self.footnotes)
        for chunk in data["chunks"]:
            for m in RESIDUAL_MARKER_RE.finditer(chunk["content"]):
                num = int(m.group(2))
                if num not in foot_nums:
                    continue
                if self.assigned.get(num, (0, 0))[0] == chunk["paragraph"]:
                    continue
                start = max(m.start() - 40, 0)
                context = toline(chunk["content"][start : m.end() + 40])
                self.marker_warnings.append(
                    f"paragraph {chunk['paragraph']} ({chunk['section']}): leftover "
                    f"digit run '{m.group(2)}' like a footnote marker "
                    f"(footnote {num} is attached to paragraph "
                    f"{self.assigned.get(num, (0, 0))[0]}): …{context}…"
                )


def main() -> None:
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        fail("usage: ingest_guidelines.py <pdf_dir> [<out_dir>]")
    pdf_dir = sys.argv[1]
    if not os.path.isdir(pdf_dir):
        fail(f"PDF directory not found: {pdf_dir}")
    out_dir = (
        sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
    )
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for cfg in DOCS:
        for lang in cfg.langs:
            ingestor = Ingestor(cfg, lang, find_pdf(pdf_dir, cfg, lang))
            ingestor.ingest()
            data = ingestor.build()
            results.append((cfg, lang, ingestor, data))

    # EN and DE of the same document must have the same paragraph count
    by_doc: Dict[str, Dict[str, int]] = {}
    for cfg, lang, _, data in results:
        by_doc.setdefault(cfg.id, {})[lang] = len(data["chunks"])
    for doc_id, counts in by_doc.items():
        if len(set(counts.values())) > 1:
            fail(
                f"{doc_id}: paragraph counts differ across languages: "
                + ", ".join(f"{lang}={count}" for lang, count in sorted(counts.items()))
            )

    failures: List[str] = []
    for cfg, lang, ingestor, data in results:
        problems = [
            f"[{cfg.id}_{lang}] {anomaly}" for anomaly in ingestor.anomalies
        ]
        paragraphs = [chunk["paragraph"] for chunk in data["chunks"]]
        if paragraphs != list(range(1, len(paragraphs) + 1)):
            problems.append(f"[{cfg.id}_{lang}] paragraph sequence broken")
        foot_nums = set(ingestor.footnotes)
        assigned = set(ingestor.assigned)
        if foot_nums != set(range(1, len(foot_nums) + 1)):
            missing = sorted(set(range(1, max(foot_nums) + 1)) - foot_nums)[:10]
            problems.append(
                f"[{cfg.id}_{lang}] footnote text numbers not contiguous, "
                f"missing: {missing}"
            )
        if assigned != foot_nums:
            problems.append(
                f"[{cfg.id}_{lang}] footnotes not all assigned to exactly one chunk: "
                f"text without marker: {sorted(foot_nums - assigned)[:10]}, "
                f"markers without text: {sorted(assigned - foot_nums)[:10]}"
            )
        if problems:
            failures.extend(problems)
            continue
        out_path = os.path.join(out_dir, f"{cfg.id}_{lang}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        sections = {chunk["section"] for chunk in data["chunks"]}
        max_len = max(len(chunk["content"]) for chunk in data["chunks"])
        print(
            f"{cfg.id}_{lang}: {len(data['chunks'])} paragraphs "
            f"[footnotes: {len(ingestor.footnotes)}, sections: {len(sections)}, "
            f"max content: {max_len} chars]"
        )
        ingestor.scan_residual_markers(data)
        for warning in ingestor.marker_warnings:
            print(f"WARNING: [{cfg.id}_{lang}] {warning}")

    if failures:
        for problem in failures:
            print(f"ERROR: {problem}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
