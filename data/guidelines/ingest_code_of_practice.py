#!/usr/bin/env python3
"""Turn the Code of Practice PDFs into chunk JSON files: the three GPAI Code
chapters and the Art 50 Code on transparency of AI-generated content.

Usage: python3 ingest_code_of_practice.py <pdf_dir> [<out_dir>]

The codes have no numbered paragraphs, so they are split by their own
structure: one chunk per recital, Commitment intro, Measure, Sub-measure,
Glossary term, Appendix and Annex section. The GPAI "Objectives" page,
identical in all chapters, is skipped. Output files <id>_en.json share the
shape of ingest_guidelines.py; "paragraph" is a running chunk number, the
reference lives in the title.
"""

import json
import os
import re
import sys
from typing import List, Optional, Tuple

from ingest_guidelines import assemble, fail, run_pdftotext, toline

HEADING_RE = re.compile(
    r"^(Introductory note\b.*|Objectives|Recitals|Glossary|Appendices"
    r"|Model Documentation Form|Commitment \d+ .*|Measure \d+\.\d+ .*"
    r"|Appendix \d+(?:\.\d+)* .*)$"
)
RECITAL_RE = re.compile(r"^\s{0,8}\(([a-z])\)\s+\S")
PAGE_NUMBER_RE = re.compile(r"^\s{10,}\d{1,3}\s*$")
FIGURE_RE = re.compile(r"^\s*Figure \d+\.")
FOOTNOTE_RE = re.compile(r"^(\d{1,2}) (\S.*)$")
GLOSSARY_HEADER_RE = re.compile(r"^\s*Term\s{3,}Definition\s*$")
GLOSSARY_SPLIT_RE = re.compile(r"^(\s*)(\S.*?)(?:\s{3,}(\S.*))?$")
LIST_ITEM_RE = re.compile(r"^\s*(?:\(\d{1,2}\)|\(?[a-z]\)|\([ivx]{1,4}\)|[A-Z]\.)\s")
# a wrapped line can start with "(a) and (b), ..." too
LIST_END_RE = re.compile(r"(?:[;:.]|\b(?:and|or|and/or|while))\s*$")
CONNECTOR_WORDS = {"and", "of", "the", "for", "to", "when", "on", "in", "with"}


class Unit:
    def __init__(self, section: str, heading: str, part: str = ""):
        self.part = part
        self.section = section
        self.heading = heading
        self.blocks: List[List[str]] = []
        self.footnotes: List[int] = []
        self.new_block = True

    def add(self, line: str) -> None:
        list_start = LIST_ITEM_RE.match(line) and self.blocks and (
            LIST_END_RE.search(self.blocks[-1][-1])
        )
        if self.new_block or list_start or not self.blocks:
            self.blocks.append([line])
        else:
            self.blocks[-1].append(line)
        self.new_block = False


class Chapter:
    heading_re = HEADING_RE
    recital_re = RECITAL_RE

    def __init__(self, doc_id: str, display: str, pdf_path: str):
        self.doc_id = doc_id
        self.display = display
        self.pages = run_pdftotext(pdf_path).split("\f")
        self.units: List[Unit] = []
        self.footnotes: dict = {}
        self.mode: Optional[str] = None
        self.commitment = ""
        self.appendix = ""
        self.cur: Optional[Unit] = None
        self.heading_open = False
        self.next_letter = 0
        self.glossary_term: Optional[Unit] = None
        self.part = ""

    def skip_page(self, page: str) -> bool:
        return False

    # -- page cleanup -------------------------------------------------------

    def page_body(self, page: str) -> List[Tuple[str, List[int]]]:
        lines = page.splitlines()
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and PAGE_NUMBER_RE.match(lines[-1]):
            lines.pop()
        notes: List[Tuple[int, str]] = []
        while lines and (not lines[-1].strip() or FOOTNOTE_RE.match(lines[-1])):
            m = FOOTNOTE_RE.match(lines.pop())
            if m:
                notes.insert(0, (int(m.group(1)), toline(m.group(2))))
        body: List[Tuple[str, List[int]]] = []
        skip_caption = False
        for line in lines:
            if FIGURE_RE.match(line):
                skip_caption = "takes precedence" not in line
                continue
            if skip_caption and "takes precedence" in line:
                skip_caption = False
                continue
            skip_caption = False
            body.append((line, []))
        for num, text in notes:
            if num in self.footnotes:
                fail(f"[{self.doc_id}] duplicate footnote {num}")
            self.footnotes[num] = text
            marker = re.compile(rf"(?<=[a-z,]){num}(?=[\s),;.]|$)")
            for i, (line, attached) in enumerate(body):
                if marker.search(line):
                    body[i] = (marker.sub("", line, count=1), attached + [num])
                    break
            else:
                fail(f"[{self.doc_id}] no marker for footnote {num}")
        return body

    # -- structure ----------------------------------------------------------

    def start(self, section: str, heading: str) -> Unit:
        unit = Unit(section, heading, self.part)
        self.units.append(unit)
        self.cur = unit
        return unit

    def on_heading(self, heading: str) -> None:
        self.heading_open = True
        self.cur = None
        if heading == "Objectives":
            self.mode = "skip"
        elif heading == "Model Documentation Form":
            # the form itself is a separate document; only a pointer to it here
            self.mode = "skip"
        elif heading == "Recitals":
            self.mode = "recitals"
            self.next_letter = 0
        elif heading == "Glossary":
            self.mode = "glossary_intro"
            self.start("Glossary", "Glossary")
        elif heading == "Appendices":
            self.mode = "skip"
        elif heading.startswith("Commitment "):
            self.mode = "body"
            self.commitment = heading
            self.start(heading, heading)
        elif heading.startswith("Measure "):
            self.mode = "body"
            self.start(self.commitment, heading)
        elif heading.startswith("Appendix "):
            self.mode = "body"
            if re.match(r"^Appendix \d+ ", heading):
                self.appendix = heading
            self.start(self.appendix, heading)
        else:
            self.mode = "body"
            self.start(heading, heading)

    def extend_heading(self, line: str) -> bool:
        """Wrapped heading lines stay at column 0 like the heading itself."""
        unit = self.cur
        if not self.heading_open or line[:1] == " ":
            return False
        last = (unit.heading if unit else "").split()[-1:] or [""]
        words = line.split()
        if not (
            line[:1].islower()
            or last[0] in CONNECTOR_WORDS
            or (len(words) <= 2 and not line.rstrip().endswith((".", ":")))
        ):
            return False
        if unit is None:
            return True
        unit.heading = toline(f"{unit.heading} {line}")
        if unit.section and unit.section.split()[:2] == unit.heading.split()[:2]:
            unit.section = unit.heading
        if unit.heading.startswith("Commitment "):
            self.commitment = unit.heading
        if re.match(r"^Appendix \d+ ", unit.heading):
            self.appendix = unit.heading
            unit.section = unit.heading
        return True

    def feed(self, line: str, notes: List[int]) -> None:
        stripped = line.strip()
        if not stripped:
            self.heading_open = False
            if self.cur:
                self.cur.new_block = True
            return
        if self.heading_re.match(line):
            self.on_heading(line.strip())
            return
        if self.extend_heading(line):
            return
        self.heading_open = False
        if self.mode == "skip" or self.mode is None:
            return
        if self.mode == "recitals":
            self.feed_recital(line)
        elif self.mode in ("glossary_intro", "glossary"):
            self.feed_glossary(line)
        else:
            self.cur.add(line)
        if notes:
            self.cur.footnotes.extend(notes)

    def feed_recital(self, line: str) -> None:
        if line.strip() == "Whereas:":
            return
        m = self.recital_re.match(line)
        if m and m.group(1) == "abcdefghijklmnop"[self.next_letter]:
            self.next_letter += 1
            self.start("Recitals", f"Recitals ({m.group(1)})")
        elif line[:1] != " " and (self.cur is None or self.cur.new_block):
            self.start("Recitals", "Recitals")
        if self.cur is None:
            fail(f"[{self.doc_id}] recital text before (a): {line.strip()[:60]}")
        self.cur.add(line)

    def feed_glossary(self, line: str) -> None:
        if GLOSSARY_HEADER_RE.match(line):
            self.mode = "glossary"
            self.cur = None
            return
        if self.mode == "glossary_intro":
            self.cur.add(line)
            return
        m = GLOSSARY_SPLIT_RE.match(line)
        indent = len(m.group(1))
        if indent >= 15:
            term, definition = "", m.group(2)
        else:
            term, definition = m.group(2), m.group(3) or ""
        if term.startswith("‘"):
            self.start("Glossary", f"Glossary {term}")
        elif term:
            joiner = "" if self.cur.heading.endswith("-") else " "
            self.cur.heading = toline(f"{self.cur.heading}{joiner}{term}")
        if self.cur is None:
            fail(f"[{self.doc_id}] glossary text before the first term: {line[:60]}")
        if definition:
            self.cur.add(definition)

    # -- output -------------------------------------------------------------

    def ingest(self) -> dict:
        for page in self.pages[1:]:
            if self.skip_page(page):
                continue
            for line, notes in self.page_body(page):
                self.feed(line, notes)
        # top-level appendices without intro text go straight to Appendix N.1
        self.units = [
            unit
            for unit in self.units
            if unit.blocks or not re.match(r"^Appendix \d+ ", unit.heading)
        ]
        chunks = []
        for number, unit in enumerate(self.units, start=1):
            content = assemble(unit.blocks)
            if not content:
                fail(f"[{self.doc_id}] empty chunk: {unit.heading}")
            prefix = f"{self.display}, {unit.part}" if unit.part else self.display
            title = f"{prefix}, {unit.heading}"
            if not unit.heading.startswith(unit.section):
                title = f"{prefix}, {unit.section}, {unit.heading}"
            chunks.append(
                {
                    "id": f"{self.doc_id}_en_{number}",
                    "title": title,
                    "content": content,
                    "keywords": [],
                    "negativeKeywords": [],
                    "relevantChunksIds": [],
                    "parameters": [],
                    "position": number - 1,
                    "document": self.doc_id,
                    "section": unit.section,
                    "paragraph": number,
                    "footnotes": [
                        {"number": num, "text": self.footnotes[num]}
                        for num in sorted(unit.footnotes)
                    ],
                }
            )
        attached = sorted(n for unit in self.units for n in unit.footnotes)
        if attached != sorted(self.footnotes):
            fail(f"[{self.doc_id}] footnotes not attached exactly once: {attached}")
        if self.next_letter == 0:
            fail(f"[{self.doc_id}] no recitals found")
        return {
            "id": f"{self.doc_id}_en",
            "creation_date": 0,
            "last_updated": 0,
            "document": self.doc_id,
            "language": "en",
            "chunks": chunks,
        }


class SectionedCode(Chapter):
    """The Art 50 Code on AI-generated content: two sections (providers,
    deployers), each with its own Objectives, Recitals and Commitments 1-4."""

    heading_re = re.compile(
        r"^(Section \d: .*|Objectives|Recitals|Commitments|Glossary|Annex"
        r"|Commitment \d+: .*|Measure \d+\.\d+: .*|Sub-measure \d+\.\d+\.\d+[:.] .*"
        r"|Annex \d+: .*)$"
    )
    recital_re = re.compile(r"^([a-z])\)\s+\S")
    parts = {"1": "Section 1 Providers", "2": "Section 2 Deployers"}

    def skip_page(self, page: str) -> bool:
        # cover and section title pages list the chairs; TOC pages have leaders
        return "Working Group" in page or bool(TOC_LINE_RE.search(page))

    def page_body(self, page: str) -> List[Tuple[str, List[int]]]:
        lines = page.splitlines()
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and PAGE_NUMBER_RE.match(lines[-1]):
            lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
        # footnotes: a bare number line, then its text up to the next one
        notes: List[Tuple[int, str]] = []
        end = len(lines)
        while True:
            start = end
            while start > 0 and lines[start - 1].strip():
                start -= 1
                if re.match(r"^\d{1,2}$", lines[start]):
                    break
            if start == end or not re.match(r"^\d{1,2}$", lines[start]):
                break
            text = toline(" ".join(lines[start + 1 : end]))
            notes.insert(0, (int(lines[start]), text))
            end = start
            while end > 0 and not lines[end - 1].strip():
                end -= 1
        lines = lines[:end]
        body: List[Tuple[str, List[int]]] = []
        in_caption = False
        for line in lines:
            if FIGURE_RE.match(line):
                in_caption = True
            elif not line.strip():
                in_caption = False
            if not in_caption:
                body.append((line, []))
        for num, text in notes:
            if num in self.footnotes:
                fail(f"[{self.doc_id}] duplicate footnote {num}")
            self.footnotes[num] = text
            marker = re.compile(rf"(?<=[a-z,]){num}(?=[\s),;.]|$)")
            for i, (line, attached) in enumerate(body):
                if marker.search(line):
                    body[i] = (marker.sub("", line, count=1), attached + [num])
                    break
            else:
                fail(f"[{self.doc_id}] no marker for footnote {num}")
        return body

    def on_heading(self, heading: str) -> None:
        m = re.match(r"^Section (\d): ", heading)
        if m:
            self.heading_open = True
            self.cur = None
            self.part = self.parts[m.group(1)]
            self.mode = "skip"
        elif heading == "Objectives":
            self.heading_open = True
            self.mode = "body"
            self.start(heading, heading)
        elif heading == "Annex":
            self.heading_open = True
            self.cur = None
            self.mode = "skip"
        elif heading.startswith("Annex "):
            self.heading_open = True
            self.part = ""
            self.mode = "body"
            self.start(heading, heading)
        elif heading.startswith("Sub-measure "):
            self.heading_open = True
            self.mode = "body"
            self.start(self.commitment, heading)
        else:
            super().on_heading(heading)

    def feed_glossary(self, line: str) -> None:
        if GLOSSARY_HEADER_RE.match(line) or self.mode == "glossary_intro":
            super().feed_glossary(line)
            return
        m = GLOSSARY_SPLIT_RE.match(line)
        if len(m.group(1)) >= 15:
            term, definition = "", m.group(2)
        else:
            term, definition = m.group(2), m.group(3) or ""
        # terms are plain words; a term line after a finished definition starts
        # a new term, otherwise it continues a wrapped term
        if term and (self.cur is None or self.cur.blocks[-1][-1].endswith(".")):
            self.start("Glossary", f"Glossary {term}")
        elif term:
            self.cur.heading = toline(f"{self.cur.heading} {term}")
        if definition:
            self.cur.add(definition)


TOC_LINE_RE = re.compile(r"\.{10,}\s*\d+\s*$", re.MULTILINE)

CHAPTERS = [
    (
        "cop_gpai_copyright",
        "GPAI CoP Copyright",
        "Code_of_Practice_for_GeneralPurpose_AI_Models_Copyright_Chapter",
        Chapter,
    ),
    (
        "cop_gpai_transparency",
        "GPAI CoP Transparency",
        "Code_of_Practice_for_GeneralPurpose_AI_Models_Transparency_Chapter",
        Chapter,
    ),
    (
        "cop_gpai_safety_security",
        "GPAI CoP Safety and Security",
        "Code_of_Practice_for_GeneralPurpose_AI_Models_Safety_and_Security_Chapter",
        Chapter,
    ),
    (
        "cop_ai_generated_content",
        "CoP AI-generated content",
        "CoP_Transparency_of_AI_generated_content",
        SectionedCode,
    ),
]


def find_pdf(pdf_dir: str, stem: str) -> str:
    matches = [
        name
        for name in sorted(os.listdir(pdf_dir))
        if name.lower().endswith(".pdf") and stem in name
    ]
    if len(matches) != 1:
        fail(f"expected exactly one PDF matching '{stem}' in {pdf_dir}: {matches}")
    return os.path.join(pdf_dir, matches[0])


def main() -> None:
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        fail("usage: ingest_code_of_practice.py <pdf_dir> [<out_dir>]")
    pdf_dir = sys.argv[1]
    if not os.path.isdir(pdf_dir):
        fail(f"PDF directory not found: {pdf_dir}")
    out_dir = (
        sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
    )
    os.makedirs(out_dir, exist_ok=True)
    for doc_id, display, stem, parser in CHAPTERS:
        data = parser(doc_id, display, find_pdf(pdf_dir, stem)).ingest()
        out_path = os.path.join(out_dir, f"{doc_id}_en.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        max_len = max(len(chunk["content"]) for chunk in data["chunks"])
        print(
            f"{doc_id}_en: {len(data['chunks'])} chunks "
            f"[footnotes: {sum(len(c['footnotes']) for c in data['chunks'])}, "
            f"max content: {max_len} chars]"
        )


if __name__ == "__main__":
    main()
