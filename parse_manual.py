#!/usr/bin/env python3
"""
Structure-aware PDF manual parser + chunker for RAG.

Goal
----
Parse a truck maintenance manual PDF (tested/targeted for
`Heavy-Duty Trucks Maintenance Manual_.pdf`) and emit retrieval-friendly JSON
chunks that preserve the manual's native hierarchy.

Key properties
--------------
- Extracts text page-by-page using `pdfplumber` (layout aware).
- Builds an intermediate representation: Page -> Blocks (heading, paragraph,
  procedure step block, safety block, table block).
- Detects hierarchy: Group -> Maintenance Operation (MOP) -> Subsections/blocks.
- Chunks primarily by MOP; splits long MOPs by semantic boundaries.
- Keeps tables as separate chunks.
- Keeps WARNING/CAUTION/DANGER/IMPORTANT/NOTE/NOTICE near relevant procedure.
- Preserves page ranges per chunk.

OCR fallback
------------
If a page appears to be "image-only" (text extraction yields almost nothing),
the pipeline can optionally OCR that page if `pytesseract` + `pdf2image` are
installed AND the system has:
- Tesseract (`brew install tesseract`)
- Poppler (`brew install poppler`) for pdf2image on macOS

Usage
-----
python parse_manual.py --input "Heavy-Duty Trucks Maintenance Manual_.pdf" --output chunks.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pdfplumber
from pydantic import BaseModel, Field
from tqdm import tqdm


LOGGER = logging.getLogger("manual_parser")


# ----------------------------
# Models / intermediate blocks
# ----------------------------


class Chunk(BaseModel):
    chunk_id: str
    group_number: Optional[str] = None
    group_title: Optional[str] = None
    mop_number: Optional[str] = None
    mop_title: Optional[str] = None
    vehicle_model_applicability: List[str] = Field(default_factory=list)
    maintenance_interval: Optional[str] = None
    content_type: str
    component: Optional[str] = None
    page_start: int
    page_end: int
    safety_related: bool = False
    table_id: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    contextual_header: str
    text: str


@dataclass(frozen=True)
class PageBlock:
    """
    A block is a contiguous semantic unit inferred from layout + text patterns.
    """

    page: int
    block_type: str  # heading | paragraph | procedure_step_block | safety_block | table | toc_entry
    text: str
    heading_level: Optional[int] = None  # 1=group, 2=mop, 3=subsection, etc.
    table_id: Optional[str] = None
    safety_label: Optional[str] = None  # WARNING | CAUTION | NOTE | ...


@dataclass
class MopContext:
    group_number: Optional[str] = None
    group_title: Optional[str] = None
    mop_number: Optional[str] = None
    mop_title: Optional[str] = None
    vehicle_models: List[str] = None

    def __post_init__(self) -> None:
        if self.vehicle_models is None:
            self.vehicle_models = []


@dataclass
class MopFamily:
    ctx: MopContext
    blocks: List[PageBlock]

    def page_range(self) -> Tuple[int, int]:
        pages = [b.page for b in self.blocks if b.text.strip() or b.table_id]
        if not pages:
            return (0, 0)
        return (min(pages), max(pages))


# ----------------------------
# Regex patterns / heuristics
# ----------------------------


SAFETY_LABELS = ("WARNING", "CAUTION", "DANGER", "IMPORTANT", "NOTE", "NOTICE")
SAFETY_RE = re.compile(rf"^\s*({'|'.join(SAFETY_LABELS)})\b[:\-]?\s*(.*)$", re.IGNORECASE)

# MOP patterns like 00-01, 00–01, 00—01, sometimes with weird OCR spacing.
MOP_RE = re.compile(r"\b(?P<a>\d{2})\s*[-–—]\s*(?P<b>\d{2})\b")

# Group patterns: "Group 00", "Group No. 00", "General Information 00", etc.
GROUP_EXPLICIT_RE = re.compile(r"\bGroup\s*(?:No\.?|Number)?\s*[:\-]?\s*(?P<num>\d{2})\b", re.IGNORECASE)
GROUP_TRAILING_NUM_RE = re.compile(
    r"^(?P<title>[A-Z][A-Za-z0-9 /&,\-]+?)\s+(?P<num>\d{2})\s*$"
)

# Subsection headings often end with ":" or are title cased short lines.
HEADING_LIKE_RE = re.compile(r"^[A-Z][A-Za-z0-9 /&,\-]{2,80}:?\s*$")

# Procedure steps: "1." "1)" "1 -" or "Step 1"
STEP_RE = re.compile(r"^\s*(?:Step\s*)?(?P<n>\d{1,2})\s*[\.\)\-]\s+(?P<body>\S.+)$", re.IGNORECASE)

# Maintenance interval tokens like IM, M1..M4. Keep exact as found.
MAINT_INTERVAL_RE = re.compile(r"\b(IM|M1|M2|M3|M4)\b")

# Model tokens; expandable.
MODEL_TOKENS = [
    "FLA",
    "FLB",
    "FLC 112",
    "FLC",
    "FLD",
    "FLL",
    "COE",
    "Conventional",
]
MODEL_RE = re.compile(r"\b(FLA|FLB|FLC\s*112|FLC|FLD|FLL)\b")


def normalize_mop(mop: str) -> str:
    m = MOP_RE.search(mop)
    if not m:
        return mop.strip()
    return f"{m.group('a')}-{m.group('b')}"


def normalize_ws(s: str) -> str:
    # Normalize weird spacing while preserving technical tokens.
    s = s.replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def looks_like_all_caps(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 6:
        return False
    return sum(1 for c in letters if c.isupper()) / max(1, len(letters)) > 0.85


def infer_models_from_text(text: str) -> List[str]:
    found = sorted(set(m.group(0).replace(" ", "") if " " not in m.group(0) else m.group(0) for m in MODEL_RE.finditer(text)))
    # Normalize common combined forms like "FLA/FLB".
    slash_models = []
    if re.search(r"\bFLA\s*/\s*FLB\b", text):
        slash_models.extend(["FLA", "FLB"])
    if re.search(r"\bFLA\s*/\s*FLB\s*/\s*FLD\b", text):
        slash_models.extend(["FLA", "FLB", "FLD"])
    merged = sorted(set(found + slash_models))
    return merged


def infer_maintenance_interval(text: str) -> Optional[str]:
    m = MAINT_INTERVAL_RE.search(text)
    return m.group(1) if m else None


def infer_component_label(mop_title: Optional[str], group_title: Optional[str]) -> Optional[str]:
    basis = " ".join([x for x in [group_title, mop_title] if x]).lower()
    if not basis:
        return None
    # A small, adaptable keyword map.
    mapping = [
        (("brake", "brakes"), "brakes"),
        (("steer", "steering"), "steering"),
        (("cab", "tilt"), "cab tilt system"),
        (("lubric", "fluid", "oil", "coolant"), "lubrication and fluid service"),
        (("interval", "schedule", "maintenance interval"), "maintenance scheduling"),
        (("fifth wheel",), "fifth wheel"),
        (("clutch",), "clutch"),
        (("transmission",), "transmission"),
        (("engine",), "engine"),
        (("electrical",), "electrical"),
    ]
    for keys, label in mapping:
        if any(k in basis for k in keys):
            return label
    return None


# ----------------------------
# Pipeline stages (modular)
# ----------------------------


def load_pdf(path: str) -> pdfplumber.PDF:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pdfplumber.open(path)


def _extract_layout_lines(page: pdfplumber.page.Page) -> List[str]:
    """
    Extract lines in reading order using word positions.

    This is more robust than page.extract_text() for manuals with columns,
    headers/footers, and variable fonts.
    """
    words = page.extract_words(
        use_text_flow=True,
        keep_blank_chars=False,
        extra_attrs=["x0", "top", "x1", "bottom", "size", "fontname"],
    )
    if not words:
        return []

    # Cluster words by "top" (line). Tolerance tuned for typical PDFs.
    words_sorted = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    lines: List[List[dict]] = []
    current: List[dict] = []
    current_top: Optional[float] = None
    tol = 2.5
    for w in words_sorted:
        top = float(w["top"])
        if current_top is None or abs(top - current_top) <= tol:
            current.append(w)
            current_top = top if current_top is None else (current_top + top) / 2.0
        else:
            lines.append(current)
            current = [w]
            current_top = top
    if current:
        lines.append(current)

    out: List[str] = []
    for line_words in lines:
        line_words = sorted(line_words, key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in line_words)
        text = re.sub(r"\s+", " ", text).strip()
        text = _despace_if_needed(text)
        if text:
            out.append(text)
    return out


def _despace_if_needed(line: str) -> str:
    """
    Some PDFs (including this manual) can yield character-by-character "words"
    from `extract_words()`, producing lines like:
      "D e t e r m i n i n g ... : 0 0 – 0 1"

    This breaks hierarchy detection. If a line appears to be mostly single-character
    tokens, remove inter-character spaces while preserving normal word spacing.
    """
    toks = line.split()
    if not toks:
        return line
    single = sum(1 for t in toks if len(t) == 1)
    if single / max(1, len(toks)) < 0.6:
        return line

    # Remove spaces between adjacent single-character tokens (letters/digits) and
    # between digits around common dash variants.
    s = line
    # Collapse "A B C" -> "ABC" for sequences of single chars.
    s = re.sub(r"(?<=\b\w)\s+(?=\w\b)", "", s)
    # Collapse "00 – 01" patterns where digits were tokenized.
    s = re.sub(r"(?<=\d)\s*([–—-])\s*(?=\d)", r"\1", s)
    # Normalize excessive spaces again.
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


def _page_needs_ocr(lines: List[str]) -> bool:
    joined = " ".join(lines).strip()
    # Heuristic: very little extracted text typically indicates scanned page.
    return len(joined) < 40


def _try_ocr_page(pdf_path: str, page_number_1_indexed: int) -> Optional[List[str]]:
    """
    Best-effort OCR for a single page; returns lines or None if unavailable.
    """
    try:
        from pdf2image import convert_from_path  # type: ignore
        import pytesseract  # type: ignore
    except Exception:
        return None

    try:
        images = convert_from_path(
            pdf_path,
            first_page=page_number_1_indexed,
            last_page=page_number_1_indexed,
            dpi=250,
        )
        if not images:
            return None
        text = pytesseract.image_to_string(images[0])
        text = normalize_ws(text)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines
    except Exception as e:
        LOGGER.warning("OCR failed for page %s: %s", page_number_1_indexed, e)
        return None


def extract_pages(
    pdf: pdfplumber.PDF,
    pdf_path: str,
    *,
    enable_ocr_fallback: bool,
    backend: str = "pdfplumber",
    respacer: bool = True,
) -> List[Tuple[int, List[str], List[List[str]]]]:
    """
    Returns a list of tuples:
      (page_number_1_indexed, lines, tables_as_rows)
    """
    if backend not in ("pdfplumber", "pymupdf", "auto"):
        raise ValueError("backend must be one of: pdfplumber | pymupdf | auto")

    if backend in ("pymupdf", "auto"):
        pymu = _extract_pages_pymupdf(pdf_path, respacer=respacer)
        if pymu is not None and pymu:
            return pymu
        if backend == "pymupdf":
            raise RuntimeError("PyMuPDF backend requested but unavailable/failed.")

    out: List[Tuple[int, List[str], List[List[str]]]] = []
    for i, page in enumerate(tqdm(pdf.pages, desc="Extracting pages")):
        page_num = i + 1
        lines = _extract_layout_lines(page)
        if respacer:
            lines = [_respacer_line(ln) for ln in lines]

        if enable_ocr_fallback and _page_needs_ocr(lines):
            ocr_lines = _try_ocr_page(pdf_path, page_num)
            if ocr_lines:
                LOGGER.info("OCR used for page %s (text extraction was sparse)", page_num)
                lines = ocr_lines

        # Extract tables (best-effort). Keep separate from text chunking.
        tables: List[List[str]] = []
        try:
            raw_tables = page.extract_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "intersection_tolerance": 5,
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                    "edge_min_length": 10,
                }
            )
            for t in raw_tables or []:
                if not t:
                    continue
                for row in t:
                    if row is None:
                        continue
                    tables.append([("" if c is None else str(c).strip()) for c in row])
        except Exception:
            # Some PDFs don't have table lines. We'll still keep going.
            tables = []

        out.append((page_num, lines, tables))
    return out


def _extract_pages_pymupdf(pdf_path: str, *, respacer: bool) -> Optional[List[Tuple[int, List[str], List[List[str]]]]]:
    """
    Alternative extractor using PyMuPDF (fitz). Often preserves word spacing better
    than layout-clustered `extract_words()` for older manuals.
    """
    try:
        import fitz  # PyMuPDF
    except Exception:
        return None

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return None

    out: List[Tuple[int, List[str], List[List[str]]]] = []
    try:
        for i in tqdm(range(doc.page_count), desc="Extracting pages (PyMuPDF)"):
            page = doc.load_page(i)
            # get_text("text") returns a single string with line breaks.
            txt = page.get_text("text") or ""
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            # Apply the same de-spacing + conservative re-spacing.
            lines = [_despace_if_needed(ln) for ln in lines]
            if respacer:
                lines = [_respacer_line(ln) for ln in lines]
            # Tables: leave empty for now (pdfplumber does better on line-tables).
            out.append((i + 1, lines, []))
    finally:
        doc.close()

    return out


_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
_ALNUM_SPLIT_RE = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")


def _respacer_line(line: str) -> str:
    """
    Conservative word-boundary recovery for mashed text.

    We avoid touching lines that already have good spacing, and we avoid breaking:
    - MOP codes (00-11, 26–08)
    - units and abbreviations
    """
    s = line.strip()
    if not s:
        return s

    # If it already has multiple spaces/words, leave it alone.
    if len(s.split()) >= 4:
        return s

    # Only attempt if it looks like a long mashed token segment.
    # Example: "Ifthecabstopsmovingwhileitisbeingraised"
    if " " in s and max((len(t) for t in s.split()), default=0) < 18:
        return s

    # Don't alter pure numeric / MOP-like lines.
    if MOP_RE.fullmatch(s) or re.fullmatch(r"[\d\-/–—]+", s):
        return s

    # Split CamelCase and alnum boundaries, but preserve common all-caps tokens.
    s2 = _CAMEL_SPLIT_RE.sub(" ", s)
    s2 = _ALNUM_SPLIT_RE.sub(" ", s2)

    # Split on common glued stopwords patterns (very conservative).
    # Only applies when there are no spaces and the string is long.
    if " " not in s2 and len(s2) >= 22:
        # Insert spaces before frequent connectors if preceded by letters.
        s2 = re.sub(r"(?i)(?<=[a-z])(?=(the|and|or|for|to|of|with|without|into|from|before|after|while|when|if))", " ", s2)

    s2 = re.sub(r"[ \t]{2,}", " ", s2).strip()
    return s2


def classify_page_regions_or_blocks(
    extracted_pages: List[Tuple[int, List[str], List[List[str]]]]
) -> List[PageBlock]:
    """
    Convert page lines into semantic blocks. Blocks are formed using:
    - heading detection (short, title-like lines; ALL CAPS; endswith ':')
    - safety label detection (WARNING/CAUTION/etc)
    - procedure step sequences
    - paragraph grouping
    - tables are emitted as separate blocks
    """
    blocks: List[PageBlock] = []

    def flush_paragraph(buf: List[str], page: int) -> None:
        text = normalize_ws("\n".join(buf))
        if text:
            blocks.append(PageBlock(page=page, block_type="paragraph", text=text))

    def flush_steps(step_buf: List[str], page: int) -> None:
        text = normalize_ws("\n".join(step_buf))
        if text:
            blocks.append(PageBlock(page=page, block_type="procedure_step_block", text=text))

    for page_num, lines, tables in extracted_pages:
        # Emit table blocks first (they will be associated later by page proximity).
        for ti, rows in enumerate(_group_rows_into_tables(tables)):
            table_text = _format_table(rows)
            blocks.append(
                PageBlock(
                    page=page_num,
                    block_type="table",
                    text=table_text,
                    table_id=f"p{page_num}-t{ti+1}",
                )
            )

        paragraph_buf: List[str] = []
        step_buf: List[str] = []
        in_safety = False
        safety_label: Optional[str] = None
        safety_buf: List[str] = []

        def flush_safety() -> None:
            nonlocal in_safety, safety_label, safety_buf
            if safety_buf:
                blocks.append(
                    PageBlock(
                        page=page_num,
                        block_type="safety_block",
                        text=normalize_ws("\n".join(safety_buf)),
                        safety_label=safety_label,
                    )
                )
            in_safety = False
            safety_label = None
            safety_buf = []

        for ln in lines:
            raw = ln.strip()
            if not raw:
                continue

            # If a line itself looks like a MOP title line, emit it as a heading block.
            # This prevents TOC/procedure pages from collapsing many MOPs into one paragraph.
            if MOP_RE.search(raw) and _looks_like_mop_title_line(raw):
                flush_paragraph(paragraph_buf, page_num)
                paragraph_buf = []
                flush_steps(step_buf, page_num)
                step_buf = []
                flush_safety()
                if _is_toc_entry_line(raw):
                    blocks.append(PageBlock(page=page_num, block_type="toc_entry", text=raw))
                else:
                    blocks.append(PageBlock(page=page_num, block_type="heading", text=raw, heading_level=2))
                continue

            # Safety blocks: start on a label line; continue until a strong boundary.
            m_safety = SAFETY_RE.match(raw)
            if m_safety:
                flush_paragraph(paragraph_buf, page_num)
                paragraph_buf = []
                flush_steps(step_buf, page_num)
                step_buf = []
                flush_safety()
                in_safety = True
                safety_label = m_safety.group(1).upper()
                remainder = m_safety.group(2).strip()
                safety_buf = [safety_label + (": " + remainder if remainder else "")]
                continue

            if in_safety:
                # End safety if we hit a clear heading, MOP, group, or a blank-ish separator.
                if _looks_like_hard_boundary(raw):
                    flush_safety()
                    # Continue processing this line normally after flush.
                else:
                    safety_buf.append(raw)
                    continue

            # Headings: keep as a separate block so chunker can split on it later.
            if _looks_like_heading_line(raw):
                flush_paragraph(paragraph_buf, page_num)
                paragraph_buf = []
                flush_steps(step_buf, page_num)
                step_buf = []
                blocks.append(PageBlock(page=page_num, block_type="heading", text=raw, heading_level=3))
                continue

            # Procedure steps: accumulate consecutive numbered lines.
            if STEP_RE.match(raw):
                flush_paragraph(paragraph_buf, page_num)
                paragraph_buf = []
                step_buf.append(raw)
                continue

            # If we were in a step block and the line no longer looks like a step,
            # flush the step block before continuing.
            if step_buf and not STEP_RE.match(raw):
                flush_steps(step_buf, page_num)
                step_buf = []

            # Default: paragraph text
            paragraph_buf.append(raw)

        flush_safety()
        flush_steps(step_buf, page_num)
        flush_paragraph(paragraph_buf, page_num)

    # Ensure stable order: by page, with tables placed where extracted.
    blocks.sort(key=lambda b: (b.page, _block_sort_key(b)))
    return blocks


def _looks_like_mop_title_line(line: str) -> bool:
    """
    Heuristic: lines that are MOP titles or TOC entries for a MOP.
    Examples:
      "Determining Scheduled Maintenance Intervals: 00–01"
      "Axle Breather and Lubricant Level Checking ... 35–01"
      "13–01 Bendix Air Compressor"
    """
    if len(line) > 160:
        return False
    if not MOP_RE.search(line):
        return False
    # TOC dot leaders or obvious title separators.
    if "..." in line or " . . . " in line or re.search(r"\.{4,}", line):
        return True
    if ":" in line:
        return True
    # MOP at start with enough text after.
    m = MOP_RE.search(line)
    if m and m.start() <= 2 and len(line[m.end() :].strip()) >= 4:
        return True
    # MOP at end with enough text before.
    if m and m.end() >= len(line) - 2 and len(line[: m.start()].strip()) >= 6:
        return True
    return False


def _is_toc_entry_line(line: str) -> bool:
    """
    Detect dot-leader TOC entries like:
      "COECabTilting , FLA/FLB .......... 00–11"
    We don't want these to start real MOP families.
    """
    if not MOP_RE.search(line):
        return False
    if " . . . " in line or re.search(r"\.{8,}", line):
        return True
    if line.count(".") >= 12:
        return True
    return False


def detect_document_metadata(extracted_pages: List[Tuple[int, List[str], List[List[str]]]]) -> Dict[str, Any]:
    """
    Best-effort document-level metadata extraction from early pages.
    """
    first_text = "\n".join(" ".join(lines) for _, lines, _ in extracted_pages[:8])
    title = _find_title(first_text) or "Heavy-Duty Trucks Maintenance Manual"

    manufacturer = None
    for cand in [
        "Daimler Trucks North America",
        "Daimler Truck North America",
        "Freightliner",
    ]:
        if cand.lower() in first_text.lower():
            manufacturer = cand
            break
    if not manufacturer:
        manufacturer = "Daimler Trucks North America"

    models = sorted(set(infer_models_from_text(first_text)))
    if not models:
        # Default from user-provided target schema
        models = ["FLA COE", "FLB COE", "FLC 112 Conventional", "FLD Conventional", "FLL COE"]

    return {
        "document_title": title,
        "document_type": "maintenance_manual",
        "manufacturer": manufacturer,
        "models": models,
    }


def detect_groups_and_mops(blocks: List[PageBlock]) -> List[MopFamily]:
    """
    Walk blocks in reading order, maintain current Group + MOP context, and
    assign blocks to MOP families.
    """
    families: List[MopFamily] = []
    current_ctx = MopContext()
    current_family: Optional[MopFamily] = None

    for b in blocks:
        # Ignore TOC entries for family detection; they are useful for metadata
        # but shouldn't create MOP families.
        if b.block_type == "toc_entry":
            continue

        # Group detection can appear in many places (headers, TOC, etc). We do it on all blocks.
        grp = _detect_group_from_heading(b.text)
        if grp:
            current_ctx.group_number, current_ctx.group_title = grp

        # MOP detection: in this manual, MOP identifiers frequently appear inside long
        # TOC-like lines or normal paragraphs, so we scan all blocks.
        mop = _detect_mop_from_heading(b.text)
        if mop:
            mop_number, mop_title = mop
            current_ctx.mop_number = mop_number
            current_ctx.mop_title = mop_title

            # If group wasn't detected explicitly, infer it from MOP prefix.
            if not current_ctx.group_number:
                current_ctx.group_number = mop_number.split("-")[0]
            if not current_ctx.group_title:
                current_ctx.group_title = "Unknown Group"

            # Infer model applicability from the title line/block, if present.
            current_ctx.vehicle_models = infer_models_from_text(b.text)

            current_family = MopFamily(ctx=MopContext(**current_ctx.__dict__), blocks=[])
            families.append(current_family)
            # Keep the trigger line itself as a MOP heading-like block for context.
            current_family.blocks.append(PageBlock(page=b.page, block_type="heading", text=b.text, heading_level=2))
            continue

        # If we don't have a MOP yet, we skip blocks until we find one.
        if current_family is None:
            continue

        # Update applicability opportunistically for blocks near the top of a MOP.
        if current_family and not current_family.ctx.vehicle_models:
            models = infer_models_from_text(b.text)
            if models:
                current_family.ctx.vehicle_models = models

        current_family.blocks.append(b)

    return families


def segment_semantic_subsections(family: MopFamily) -> List[Tuple[str, List[PageBlock]]]:
    """
    Split a MOP family into semantic sections based on subsection headings and
    block types. Returns (section_label, blocks).

    Section labels are normalized to controlled-ish types (overview/procedure/etc).
    """
    sections: List[Tuple[str, List[PageBlock]]] = []
    buf: List[PageBlock] = []
    current_label = "operation_summary"

    def flush() -> None:
        nonlocal buf, current_label
        if buf:
            sections.append((current_label, buf))
        buf = []

    for b in family.blocks:
        # Keep tables separate regardless of where they appear.
        if b.block_type == "table":
            flush()
            sections.append(("table", [b]))
            continue

        # Safety block: keep it attached to following procedure/paragraph if possible.
        # We'll treat it as part of the buffer; chunk builder may merge it forward.
        if b.block_type == "safety_block":
            buf.append(b)
            continue

        # Headings inside MOP: subsection boundary.
        if b.block_type == "heading" and (b.heading_level or 3) >= 3:
            maybe = _normalize_section_label(b.text)
            if maybe:
                flush()
                current_label = maybe
                buf.append(b)  # keep heading inside the section
                continue

        # Procedure step blocks strongly indicate procedure content.
        if b.block_type == "procedure_step_block":
            if current_label not in ("procedure", "troubleshooting"):
                # Switch to procedure section if we haven't.
                if buf:
                    flush()
                current_label = "procedure"
            buf.append(b)
            continue

        buf.append(b)

    flush()
    return sections


def build_chunks(families: List[MopFamily], *, max_chars: int) -> List[Chunk]:
    """
    Convert MOP families into final chunks. Rules:
    - Base boundary = MOP family.
    - If a MOP is short, keep as one coherent chunk (excluding table chunks).
    - If long, split by section labels and preserve safety blocks with procedure.
    - Tables are always separate chunks.
    """
    chunks: List[Chunk] = []

    for fam in tqdm(families, desc="Building chunks"):
        if not fam.ctx.mop_number:
            continue

        sections = segment_semantic_subsections(fam)

        # Separate tables up front.
        non_table_sections = [(lbl, blks) for (lbl, blks) in sections if lbl != "table"]
        table_sections = [(lbl, blks) for (lbl, blks) in sections if lbl == "table"]

        mop_text = normalize_ws("\n\n".join(_block_to_text(b) for _, blks in non_table_sections for b in blks))

        # Single chunk if short enough and no obvious multiple sections.
        if len(mop_text) <= max_chars and len(non_table_sections) <= 2:
            page_start, page_end = fam.page_range()
            chunks.append(
                _make_chunk(
                    fam=fam,
                    chunk_id=_chunk_id(fam, "overview"),
                    content_type=_best_content_type_from_sections(non_table_sections),
                    text=mop_text,
                    page_start=page_start,
                    page_end=page_end,
                    safety_related=_has_safety(non_table_sections),
                    parent_chunk_id=None,
                )
            )
        else:
            # Build subchunks by sections, with size guard.
            parent_id = _chunk_id(fam, "mop")
            # Parent chunk is not emitted as content; used as link only.

            for idx, (label, blks) in enumerate(non_table_sections, start=1):
                # Merge safety blocks forward: if section starts with safety block(s) and next section is procedure,
                # keep them with the next procedure chunk. We implement a simple local rule.
                if _section_is_only_safety(blks):
                    # Attach to the next section if exists.
                    if idx < len(non_table_sections):
                        next_label, next_blks = non_table_sections[idx]
                        non_table_sections[idx] = (next_label, blks + next_blks)
                    continue

                text = normalize_ws("\n\n".join(_block_to_text(b) for b in blks))
                if not text:
                    continue

                # If a section is still too large, split by block boundaries.
                sub_parts = _split_blocks_by_size(blks, max_chars=max_chars)
                for part_i, part_blks in enumerate(sub_parts, start=1):
                    part_text = normalize_ws("\n\n".join(_block_to_text(b) for b in part_blks))
                    if not part_text:
                        continue
                    ps = min(b.page for b in part_blks)
                    pe = max(b.page for b in part_blks)
                    suffix = label if len(sub_parts) == 1 else f"{label}-{part_i:02d}"
                    chunks.append(
                        _make_chunk(
                            fam=fam,
                            chunk_id=_chunk_id(fam, suffix),
                            content_type=label,
                            text=part_text,
                            page_start=ps,
                            page_end=pe,
                            safety_related=_has_safety([("x", part_blks)]),
                            parent_chunk_id=parent_id,
                            subsection=_extract_first_subheading(part_blks),
                        )
                    )

        # Add tables as separate chunks (always child of the MOP).
        for ti, (_, tbl_blks) in enumerate(table_sections, start=1):
            b = tbl_blks[0]
            ps = pe = b.page
            chunks.append(
                _make_chunk(
                    fam=fam,
                    chunk_id=_chunk_id(fam, f"table-{ti:02d}"),
                    content_type="table",
                    text=b.text,
                    page_start=ps,
                    page_end=pe,
                    safety_related=False,
                    table_id=b.table_id,
                    parent_chunk_id=_chunk_id(fam, "mop"),
                )
            )

    return chunks


def enrich_metadata(chunks: List[Chunk]) -> List[Chunk]:
    """
    Add inferred fields like maintenance_interval, component, applicability.
    """
    for ch in chunks:
        if not ch.vehicle_model_applicability:
            if ch.mop_title:
                models = infer_models_from_text(ch.mop_title)
            else:
                models = []
            ch.vehicle_model_applicability = models or ["all applicable models"]

        if not ch.maintenance_interval:
            ch.maintenance_interval = infer_maintenance_interval(ch.text)

        if not ch.component:
            ch.component = infer_component_label(ch.mop_title, ch.group_title)
    return chunks


def export_json(
    output_path: str,
    doc_meta: Dict[str, Any],
    chunks: List[Chunk],
    *,
    primary_boundary: str = "maintenance_operation",
) -> None:
    payload: Dict[str, Any] = {
        **doc_meta,
        "chunking_strategy": {
            "primary_boundary": primary_boundary,
            "secondary_boundaries": [
                "subsection_heading",
                "procedure_step_block",
                "table",
                "warning_caution_note_block",
            ],
            "preserve_structure": True,
            "attach_contextual_headers": True,
            "keep_tables_separate": True,
            "keep_safety_text_with_procedure": True,
        },
        "metadata_fields": [
            "group_number",
            "group_title",
            "mop_number",
            "mop_title",
            "vehicle_model_applicability",
            "maintenance_interval",
            "content_type",
            "component",
            "page_start",
            "page_end",
            "safety_related",
            "table_id",
            "parent_chunk_id",
        ],
        "chunks": [c.model_dump() for c in chunks],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ----------------------------
# Helpers
# ----------------------------


def _group_rows_into_tables(rows: List[List[str]]) -> List[List[List[str]]]:
    """
    pdfplumber's extract_tables returns separate tables, but our extraction path
    above flattens. Keep this as a simple grouping hook in case future versions
    change. Here we treat it as one table if there are rows.
    """
    if not rows:
        return []
    return [rows]


def _format_table(rows: List[List[str]]) -> str:
    # Use TSV-like format for retrieval while preserving row structure.
    lines = []
    for row in rows:
        cleaned = [re.sub(r"\s+", " ", c).strip() for c in row]
        lines.append("\t".join(cleaned))
    return normalize_ws("\n".join(lines))


def _block_sort_key(b: PageBlock) -> int:
    # Prefer headings first, then safety, then paragraphs/steps, then tables.
    order = {"heading": 0, "toc_entry": 1, "safety_block": 2, "procedure_step_block": 3, "paragraph": 4, "table": 5}
    return order.get(b.block_type, 99)


def _looks_like_hard_boundary(line: str) -> bool:
    if GROUP_EXPLICIT_RE.search(line):
        return True
    if MOP_RE.search(line):
        return True
    if looks_like_all_caps(line) and len(line) < 60:
        return True
    if _looks_like_heading_line(line):
        return True
    return False


def _looks_like_heading_line(line: str) -> bool:
    if len(line) > 90:
        return False
    if GROUP_EXPLICIT_RE.search(line) or MOP_RE.search(line):
        return True
    if looks_like_all_caps(line) and len(line) <= 80:
        return True
    if line.endswith(":") and len(line) <= 80:
        return True
    # Avoid misclassifying de-spaced running text as headings (e.g. "Todeterminethe...").
    # Require some word boundaries (spaces) for generic heading detection.
    if HEADING_LIKE_RE.match(line):
        words = line.split()
        if len(words) >= 2 and len(words) <= 10:
            return True
    return False


def _detect_group_from_heading(line: str) -> Optional[Tuple[str, str]]:
    m = GROUP_EXPLICIT_RE.search(line)
    if m:
        num = m.group("num")
        title = re.sub(GROUP_EXPLICIT_RE, "", line).strip(" :-–—")
        title = title if title else None
        return (num, title or "Unknown Group")

    m2 = GROUP_TRAILING_NUM_RE.match(line)
    if m2:
        title = m2.group("title").strip()
        num = m2.group("num")
        if 0 <= int(num) <= 99 and len(title) >= 3:
            return (num, title)
    return None


def _detect_mop_from_heading(line: str) -> Optional[Tuple[str, str]]:
    m = MOP_RE.search(line)
    if not m:
        return None

    mop = f"{m.group('a')}-{m.group('b')}"
    # Title can be:
    # - "Determining Scheduled Maintenance Intervals: 00–01"  (mop at end)
    # - "13–01 Bendix Air Compressor"                         (mop at start)
    # - "M1 Maintenance Interval Operations Table: 00–07 ..." (mop embedded)
    before = line[: m.start()].strip(" :-–—.\t")
    after = line[m.end() :].strip(" :-–—.\t")
    title = None

    # If MOP appears at the start, title is usually after it.
    if m.start() <= 2 and after and len(after) >= 4:
        title = after
    # If MOP appears at the end, title is usually before it.
    elif m.end() >= len(line) - 2 and before and len(before) >= 4:
        title = before
    # Otherwise pick the larger/cleaner side.
    else:
        if before and len(before) >= 4 and (not after or len(before) >= len(after)):
            title = before
        elif after and len(after) >= 4:
            title = after

    # Clean "MOP" or "Maintenance Operation" tokens.
    if title:
        title = re.sub(r"\b(MOP|Maintenance Operation)\b", "", title, flags=re.IGNORECASE).strip(" :-–—")
    return (mop, title or "Unknown Maintenance Operation")


def _normalize_section_label(heading_text: str) -> Optional[str]:
    t = heading_text.strip().lower().strip(":")
    # Map common subsection terms to controlled-ish types.
    mapping = {
        "description": "system_overview",
        "system description": "system_overview",
        "overview": "overview",
        "introduction": "overview",
        "procedure": "procedure",
        "procedures": "procedure",
        "removal": "procedure",
        "installation": "procedure",
        "repair": "procedure",
        "inspection": "procedure",
        "cleaning": "procedure",
        "adjustment": "procedure",
        "troubleshooting": "troubleshooting",
        "specifications": "specification",
        "specification": "specification",
        "notes": "note_block",
        "warnings": "warning_block",
        "cautions": "caution_block",
    }
    for k, v in mapping.items():
        if t == k:
            return v
    # If it looks like a heading but not known, treat as subsection grouping
    if _looks_like_heading_line(heading_text):
        return "operation_summary"
    return None


def _block_to_text(b: PageBlock) -> str:
    if b.block_type == "heading":
        return b.text
    if b.block_type == "safety_block" and b.safety_label:
        return b.text  # includes label prefix
    return b.text


def _chunk_id(fam: MopFamily, suffix: str) -> str:
    grp = fam.ctx.group_number or "xx"
    mop = (fam.ctx.mop_number or "xx-xx").replace("-", "")
    safe_suffix = re.sub(r"[^a-zA-Z0-9\-]+", "-", suffix.strip().lower())
    safe_suffix = re.sub(r"-{2,}", "-", safe_suffix).strip("-")
    return f"grp{grp}-mop{mop}-{safe_suffix}"


def _make_contextual_header(
    fam: MopFamily, content_type: str, subsection: Optional[str] = None
) -> str:
    parts = []
    if fam.ctx.group_number and fam.ctx.group_title:
        parts.append(f"Group {fam.ctx.group_number} {fam.ctx.group_title}")
    elif fam.ctx.group_number:
        parts.append(f"Group {fam.ctx.group_number}")
    if fam.ctx.mop_number and fam.ctx.mop_title:
        parts.append(f"MOP {fam.ctx.mop_number} {fam.ctx.mop_title}")
    elif fam.ctx.mop_number:
        parts.append(f"MOP {fam.ctx.mop_number}")
    if subsection:
        parts.append(subsection.strip(": ").strip())
    elif content_type and content_type not in ("operation_summary", "overview"):
        parts.append(content_type.replace("_", " ").title())
    return " | ".join(parts)


def _make_chunk(
    fam: MopFamily,
    chunk_id: str,
    content_type: str,
    text: str,
    page_start: int,
    page_end: int,
    safety_related: bool,
    parent_chunk_id: Optional[str],
    table_id: Optional[str] = None,
    subsection: Optional[str] = None,
) -> Chunk:
    header = _make_contextual_header(fam, content_type, subsection=subsection)
    return Chunk(
        chunk_id=chunk_id,
        group_number=fam.ctx.group_number,
        group_title=fam.ctx.group_title,
        mop_number=fam.ctx.mop_number,
        mop_title=fam.ctx.mop_title,
        vehicle_model_applicability=fam.ctx.vehicle_models or [],
        maintenance_interval=None,
        content_type=content_type,
        component=None,
        page_start=page_start,
        page_end=page_end,
        safety_related=safety_related,
        table_id=table_id,
        parent_chunk_id=parent_chunk_id,
        contextual_header=header,
        text=text,
    )


def _find_title(text: str) -> Optional[str]:
    # Pick the first line containing "Maintenance Manual" if present.
    for line in text.splitlines():
        if "maintenance manual" in line.lower() and len(line.strip()) <= 80:
            return normalize_ws(line)
    return None


def _has_safety(sections: List[Tuple[str, List[PageBlock]]]) -> bool:
    for _, blks in sections:
        for b in blks:
            if b.block_type == "safety_block":
                return True
    return False


def _section_is_only_safety(blks: List[PageBlock]) -> bool:
    real = [b for b in blks if b.block_type != "heading" and b.text.strip()]
    return bool(real) and all(b.block_type == "safety_block" for b in real)


def _split_blocks_by_size(blks: List[PageBlock], *, max_chars: int) -> List[List[PageBlock]]:
    parts: List[List[PageBlock]] = []
    buf: List[PageBlock] = []
    buf_len = 0

    for b in blks:
        t = _block_to_text(b)
        add = len(t) + 2
        if buf and buf_len + add > max_chars:
            parts.append(buf)
            buf = [b]
            buf_len = len(t)
        else:
            buf.append(b)
            buf_len += add
    if buf:
        parts.append(buf)
    return parts


def _extract_first_subheading(blks: List[PageBlock]) -> Optional[str]:
    for b in blks:
        if b.block_type == "heading" and (b.heading_level or 3) >= 3:
            # Avoid repeating group/mop headings.
            if GROUP_EXPLICIT_RE.search(b.text) or MOP_RE.search(b.text):
                continue
            return b.text
    return None


def _best_content_type_from_sections(sections: List[Tuple[str, List[PageBlock]]]) -> str:
    labels = [lbl for lbl, _ in sections]
    if "procedure" in labels:
        return "procedure"
    if "system_overview" in labels:
        return "system_overview"
    if "overview" in labels:
        return "overview"
    return "overview"


# ----------------------------
# CLI / main
# ----------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parse and chunk a maintenance manual PDF into structured JSON for RAG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Path to input PDF.")
    p.add_argument("--output", required=True, help="Path to output JSON.")
    p.add_argument("--max-chars", type=int, default=4500, help="Max characters per chunk (structure-aware).")
    p.add_argument("--enable-ocr-fallback", action="store_true", help="Use OCR on sparse pages if deps installed.")
    p.add_argument(
        "--extractor",
        default="auto",
        choices=["auto", "pdfplumber", "pymupdf"],
        help="PDF text extraction backend (auto tries PyMuPDF then falls back).",
    )
    p.add_argument(
        "--no-respacer",
        action="store_true",
        help="Disable word-boundary recovery postprocessing (not recommended for this PDF).",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    LOGGER.info("Loading PDF: %s", args.input)
    with load_pdf(args.input) as pdf:
        extracted_pages = extract_pages(
            pdf,
            args.input,
            enable_ocr_fallback=args.enable_ocr_fallback,
            backend=args.extractor,
            respacer=(not args.no_respacer),
        )

    doc_meta = detect_document_metadata(extracted_pages)
    LOGGER.info("Detected metadata: title=%r manufacturer=%r models=%s", doc_meta["document_title"], doc_meta["manufacturer"], doc_meta["models"])

    blocks = classify_page_regions_or_blocks(extracted_pages)
    LOGGER.info("Built %s page blocks", len(blocks))

    families = detect_groups_and_mops(blocks)
    LOGGER.info("Detected %s MOP families", len(families))
    if not families:
        LOGGER.warning(
            "No MOP families detected. This usually means heading patterns didn't match the PDF text.\n"
            "Tip: run with --log-level DEBUG and inspect extracted headings; we can tune regexes."
        )

    chunks = build_chunks(families, max_chars=args.max_chars)
    chunks = enrich_metadata(chunks)
    LOGGER.info("Emitting %s chunks", len(chunks))

    export_json(args.output, doc_meta, chunks)
    LOGGER.info("Wrote JSON: %s", args.output)

    # Print a small sample snippet for quick inspection.
    if chunks:
        sample = chunks[0].model_dump()
        sample["text"] = (sample["text"][:400] + "…") if len(sample["text"]) > 400 else sample["text"]
        LOGGER.info("Sample chunk:\n%s", json.dumps(sample, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

