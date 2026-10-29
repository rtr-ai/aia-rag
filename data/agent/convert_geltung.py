#!/usr/bin/env python3
"""Turn the client's applicability workbook into ai_act_geltung.json.

Usage: python3 convert_geltung.py [xlsx] [out]

Both paths default to files next to this script: geltung-ai-act.xlsx and
ai_act_geltung.json. The script is stdlib only and deterministic: the xlsx is
read as a zip of XML parts, so identical input yields byte-identical output.
Update procedure: replace the xlsx with the client's new version, rerun this
script and commit both files together.
"""

import json
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from typing import Dict, List, Optional, Set, Tuple

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"m": MAIN_NS}

XLSX_NAME = "geltung-ai-act.xlsx"
JSON_NAME = "ai_act_geltung.json"
GELTUNG_SHEET = "Geltung"
GRUNDLAGEN_SHEET = "Datumsgrundlagen"
LESEHILFE_SHEET = "Lesehilfe"

GELTUNG_COLUMNS = [
    "ref",
    "artikel",
    "absatz",
    "titel",
    "kapitel",
    "abschnitt",
    "typ",
    "bedingung",
    "geltung_ab",
    "grundlage",
    "anmerkung",
    "anmerkung_quelle",
]
GRUNDLAGEN_COLUMNS = ["grundlage", "Norm", "Datum", "Inhalt"]
TYPOLOGIES = {
    "Geltungsbeginn",
    "Übergangsfrist",
    "Einzelfrist",
    "Hinweis",
    "Inkrafttreten",
}

REF_RE = re.compile(r"^(\d+[a-z]?)(?:-(\d+[a-z]?)(?:-([a-z]+))?)?$")
CELL_REF_RE = re.compile(r"^([A-Za-z]+)(\d+)$")
CELL_LETTER_RE = re.compile(r"^([A-Za-z]+)")
STAND_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})")

Row = Dict[str, Optional[str]]


class ConversionError(Exception):
    pass


def read_part(zf: zipfile.ZipFile, member: str) -> bytes:
    try:
        return zf.read(member)
    except KeyError:
        raise ConversionError(f"Workbook part '{member}' not found") from None


def workbook_parts(zf: zipfile.ZipFile) -> Tuple[Dict[str, str], date]:
    workbook = ET.fromstring(read_part(zf, "xl/workbook.xml"))
    rels = ET.fromstring(read_part(zf, "xl/_rels/workbook.xml.rels"))
    targets = {
        rel.get("Id"): rel.get("Target")
        for rel in rels.findall(f"{{{RELS_NS}}}Relationship")
    }
    parts: Dict[str, str] = {}
    for sheet in workbook.findall(f"{{{MAIN_NS}}}sheets/{{{MAIN_NS}}}sheet"):
        name = sheet.get("name")
        target = targets.get(sheet.get(f"{{{REL_NS}}}id", ""))
        if name is None or not target:
            continue
        if target.startswith("/"):
            parts[name] = target.lstrip("/")
        else:
            parts[name] = "xl/" + target
    epoch = date(1899, 12, 30)  # Excel serial 0, accounting for the 1900 leap-year bug
    pr = workbook.find("m:workbookPr", NS)
    if pr is not None and pr.get("date1904", "") in ("1", "true"):
        epoch = date(1904, 1, 1)
    return parts, epoch


def shared_strings(zf: zipfile.ZipFile) -> List[str]:
    try:
        raw = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(raw)
    return [
        "".join(node.text or "" for node in si.iter(f"{{{MAIN_NS}}}t"))
        for si in root.findall(f"{{{MAIN_NS}}}si")
    ]


def cell_text(cell: ET.Element, shared: List[str]) -> Optional[str]:
    kind = cell.get("t")
    text = ""
    if kind == "inlineStr":
        inline = cell.find("m:is", NS)
        if inline is not None:
            text = "".join(node.text or "" for node in inline.iter(f"{{{MAIN_NS}}}t"))
    elif kind == "s":
        value = cell.find("m:v", NS)
        index = -1
        if value is not None and value.text:
            try:
                index = int(value.text)
            except ValueError:
                index = -1
        if not 0 <= index < len(shared):
            raise ConversionError(f"Shared string index {index} out of range")
        text = shared[index]
    else:
        value = cell.find("m:v", NS)
        if value is not None and value.text is not None:
            text = value.text
    text = text.strip()
    return text or None


def parse_sheet(
    zf: zipfile.ZipFile,
    part: str,
    shared: List[str],
) -> Dict[int, Row]:
    root = ET.fromstring(read_part(zf, part))
    node = root.find("m:sheetData", NS)
    rows: Dict[int, Row] = {}
    if node is None:
        return rows
    for element in node.findall("m:row", NS):
        attr = element.get("r")
        try:
            number = int(attr) if attr else max(rows, default=0) + 1
        except ValueError:
            raise ConversionError(
                f"Sheet part '{part}': invalid row number '{attr}'"
            ) from None
        cells: Row = {}
        for cell in element.findall("m:c", NS):
            ref = cell.get("r")
            match = CELL_LETTER_RE.match(ref) if ref else None
            if match is None:
                raise ConversionError(
                    f"Sheet part '{part}' row {number}: invalid cell reference '{ref}'"
                )
            cells[match.group(1).upper()] = cell_text(cell, shared)
        rows[number] = cells
    return rows


def read_sheet(zf: zipfile.ZipFile, part: str, shared: List[str]) -> List[Row]:
    return list(parse_sheet(zf, part, shared).values())


def read_cell(
    zf: zipfile.ZipFile,
    part: str,
    shared: List[str],
    ref: str,
) -> Optional[str]:
    match = CELL_REF_RE.match(ref)
    if match is None:
        raise ConversionError(f"Invalid cell reference '{ref}'")
    rows = parse_sheet(zf, part, shared)
    return rows.get(int(match.group(2)), {}).get(match.group(1).upper())


def header_columns(row: Row) -> Dict[str, str]:
    columns: Dict[str, str] = {}
    for letter, value in row.items():
        if value:
            columns.setdefault(value, letter)
    return columns


def read_table(
    zf: zipfile.ZipFile,
    name: str,
    parts: Dict[str, str],
    shared: List[str],
    expected: List[str],
) -> Tuple[List[Row], Dict[str, str]]:
    part = parts.get(name)
    if part is None:
        raise ConversionError(f"Sheet '{name}' not found")
    rows = read_sheet(zf, part, shared)
    if not rows:
        raise ConversionError(f"Sheet '{name}' is empty")
    columns = header_columns(rows[0])
    missing = [header for header in expected if header not in columns]
    if missing:
        raise ConversionError(
            f"Sheet '{name}': missing column(s): {', '.join(missing)}"
        )
    return rows[1:], columns


def iso_date(serial: str, epoch: date, context: str) -> str:
    try:
        days = int(float(serial))
    except ValueError:
        raise ConversionError(f"{context}: non-numeric date '{serial}'") from None
    return (epoch + timedelta(days=days)).isoformat()


def convert_grundlagen(
    rows: List[Row],
    columns: Dict[str, str],
    epoch: date,
) -> Dict[str, Dict[str, Optional[str]]]:
    grundlagen: Dict[str, Dict[str, Optional[str]]] = {}
    for excel_row, row in enumerate(rows, start=2):
        if not any(row.values()):
            continue
        key = row.get(columns["grundlage"])
        if key is None:
            raise ConversionError(
                f"Sheet 'Datumsgrundlagen' row {excel_row}: missing grundlage"
            )
        datum_cell = row.get(columns["Datum"])
        datum = (
            iso_date(datum_cell, epoch, f"Sheet 'Datumsgrundlagen' row {excel_row}")
            if datum_cell is not None
            else None
        )
        grundlagen[key] = {
            "norm": row.get(columns["Norm"]),
            "datum": datum,
            "inhalt": row.get(columns["Inhalt"]),
        }
    return grundlagen


def convert_entries(
    rows: List[Row],
    columns: Dict[str, str],
    epoch: date,
    grundlagen: Dict[str, Dict[str, Optional[str]]],
) -> Tuple[List[Dict[str, Optional[str]]], List[Tuple[str, str]]]:
    entries: List[Dict[str, Optional[str]]] = []
    unresolved: List[Tuple[str, str]] = []
    reported: Set[str] = set()
    for excel_row, row in enumerate(rows, start=2):
        if not any(row.values()):
            continue
        ref = row.get(columns["ref"]) or ""
        match = REF_RE.match(ref)
        if match is None:
            raise ConversionError(
                f"Sheet 'Geltung' row {excel_row}: invalid ref '{ref}'"
            )
        artikel = row.get(columns["artikel"])
        absatz = row.get(columns["absatz"])
        expected_absatz = "-".join(
            group for group in (match.group(2), match.group(3)) if group
        )
        if artikel != match.group(1) or (absatz or "") != expected_absatz:
            raise ConversionError(
                f"Sheet 'Geltung' row {excel_row}: ref '{ref}' does not match "
                f"artikel '{artikel or ''}'/absatz '{absatz or ''}'"
            )
        typ = row.get(columns["typ"])
        if typ not in TYPOLOGIES:
            raise ConversionError(
                f"Sheet 'Geltung' row {excel_row}: invalid typ '{typ}'"
            )
        grundlage_cell = row.get(columns["grundlage"])
        serial = row.get(columns["geltung_ab"])
        datum: Optional[str] = None
        if typ == "Hinweis":
            if serial is not None:
                raise ConversionError(
                    f"Sheet 'Geltung' row {excel_row}: typ 'Hinweis' must not have "
                    f"geltung_ab '{serial}'"
                )
        elif serial is None:
            if grundlage_cell != "111-2":
                raise ConversionError(
                    f"Sheet 'Geltung' row {excel_row}: typ '{typ}' without geltung_ab "
                    f"(grundlage '{grundlage_cell or ''}')"
                )
        else:
            datum = iso_date(
                serial, epoch, f"Sheet 'Geltung' row {excel_row} geltung_ab"
            )
        grundlage = grundlage_cell if grundlage_cell not in (None, "-") else None
        entries.append(
            {
                "artikel": match.group(1),
                "absatz": match.group(2),
                "litera": match.group(3),
                "titel": row.get(columns["titel"]),
                "typ": typ,
                "bedingung": row.get(columns["bedingung"]),
                "datum": datum,
                "grundlage": grundlage,
                "anmerkung": row.get(columns["anmerkung"]),
                "quelle": row.get(columns["anmerkung_quelle"]),
            }
        )
        if (
            grundlage is not None
            and grundlage not in grundlagen
            and grundlage != ref
            and grundlage not in reported
        ):
            reported.add(grundlage)
            unresolved.append((grundlage, ref))
    return entries, unresolved


def read_stand(zf: zipfile.ZipFile, parts: Dict[str, str], shared: List[str]) -> str:
    part = parts.get(LESEHILFE_SHEET)
    if part is None:
        raise ConversionError(f"Sheet '{LESEHILFE_SHEET}' not found")
    text = read_cell(zf, part, shared, "B3") or ""
    match = STAND_RE.match(text)
    if match is None:
        preview = text[:80] + ("..." if len(text) > 80 else "")
        raise ConversionError(
            f"Sheet '{LESEHILFE_SHEET}' cell B3: no date at start of '{preview}'"
        )
    day, month, year = match.groups()
    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"


def convert(
    xlsx: str,
) -> Tuple[Dict[str, object], List[Dict[str, Optional[str]]], List[Tuple[str, str]]]:
    with zipfile.ZipFile(xlsx) as zf:
        parts, epoch = workbook_parts(zf)
        shared = shared_strings(zf)
        rows, columns = read_table(
            zf, GRUNDLAGEN_SHEET, parts, shared, GRUNDLAGEN_COLUMNS
        )
        grundlagen = convert_grundlagen(rows, columns, epoch)
        rows, columns = read_table(zf, GELTUNG_SHEET, parts, shared, GELTUNG_COLUMNS)
        entries, unresolved = convert_entries(rows, columns, epoch, grundlagen)
        stand = read_stand(zf, parts, shared)
    data: Dict[str, object] = {
        "stand": stand,
        "grundlagen": grundlagen,
        "eintraege": entries,
    }
    return data, entries, unresolved


def main() -> int:
    if len(sys.argv) > 3:
        print("Usage: python3 convert_geltung.py [xlsx] [out]", file=sys.stderr)
        return 1
    base = os.path.dirname(os.path.abspath(__file__))
    xlsx = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base, XLSX_NAME)
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(base, JSON_NAME)
    try:
        data, entries, unresolved = convert(xlsx)
    except ConversionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    for code, ref in unresolved:
        print(f"Unresolved grundlage code: {code} (ref {ref})")
    print(f"Wrote {len(entries)} entries to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
