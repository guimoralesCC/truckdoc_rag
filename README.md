# Daimler manual PDF → structured JSON chunks (RAG)

This folder contains a **structure-aware PDF parser + chunker** for truck maintenance manuals.
It targets `Heavy-Duty Trucks Maintenance Manual_.pdf` and produces **chunked JSON** suitable for retrieval-augmented generation.

## What it does

- Extracts text **page-by-page** (layout aware).
- Detects hierarchy: **Group → Maintenance Operation (MOP) → subsections / procedures / safety blocks / tables**.
- Chunks primarily by **MOP** (no naive token-window chunking).
- Keeps **tables** as separate chunks.
- Keeps **WARNING/CAUTION/DANGER/IMPORTANT/NOTE/NOTICE** attached near procedure content.
- Preserves **page ranges** for every chunk.

## Setup (virtual environment)

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python parse_manual.py \
  --input "Heavy-Duty Trucks Maintenance Manual_.pdf" \
  --output chunks.json
```

## Search

BM25 (keyword search):

```bash
python bm25_search.py --chunks chunks.json --query "cab tilting FLA WARNING" --topk 5
```

Hybrid (BM25 + vector):

```bash
python hybrid_search.py --chunks chunks.json --query "cab tilting FLA WARNING" --topk 5
```

## Ask questions with an LLM (OpenRouter)

Export your key:

```bash
export OPENROUTER_API_KEY="..."
```

Ask a question (uses hybrid retrieval + `openai/gpt-4o-mini`):

```bash
python answer_manual.py \
  --chunks chunks.json \
  --question "How do I tilt the FLA cab safely?" \
  --topk 8
```

## GUI (Streamlit)

```bash
export OPENROUTER_API_KEY="..."
streamlit run gui_app.py
```

Optional: enable OCR fallback for scanned pages (only used when text extraction is sparse):

```bash
python parse_manual.py \
  --input "Heavy-Duty Trucks Maintenance Manual_.pdf" \
  --output chunks.json \
  --enable-ocr-fallback
```

### OCR dependencies (macOS)

If you use `--enable-ocr-fallback`, you may need:

```bash
brew install tesseract poppler
```

## Output JSON shape

The output file includes:

- document-level metadata
- chunking strategy metadata
- `chunks`: list of chunk objects with fields like:
  - `group_number`, `group_title`
  - `mop_number`, `mop_title`
  - `content_type` (overview/procedure/table/etc)
  - `vehicle_model_applicability`, `maintenance_interval`
  - `page_start`, `page_end`
  - `contextual_header`

## Known limitations / how to adapt

- Manuals vary a lot in typography; if MOP/group detection misses items, tune regexes in `parse_manual.py`:
  - `GROUP_EXPLICIT_RE`, `GROUP_TRAILING_NUM_RE`
  - `MOP_RE`
- Table extraction depends on PDF table lines; some tables may be missed and will remain in text blocks.
- If the PDF is heavily scanned, OCR improves results but requires system dependencies.

