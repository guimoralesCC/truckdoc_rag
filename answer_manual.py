#!/usr/bin/env python3
"""
RAG QA over a maintenance manual using:
- Hybrid retrieval (BM25 + vector) over `chunks.json`
- OpenRouter LLM call (gpt-4o-mini) to generate a grounded answer

Run
---
source .venv/bin/activate
export OPENROUTER_API_KEY="..."

python answer_manual.py \
  --chunks chunks.json \
  --question "How do I tilt the FLA cab safely?" \
  --topk 8

Notes
-----
- This script never sends your whole PDF; it only sends the top retrieved chunks.
- Output includes citations with page ranges and chunk ids.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from typing import Any, Dict, List, Sequence

import requests

from hybrid_search import bm25_index, embed_corpus, faiss_index, hybrid_search, load_chunks


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"


def build_references(results: List[Dict[str, Any]]) -> str:
    """
    Create a compact, citation-friendly references section.
    """
    out = []
    for i, ch in enumerate(results, start=1):
        cid = ch.get("chunk_id")
        pages = f"p{ch.get('page_start')}–p{ch.get('page_end')}"
        header = ch.get("contextual_header") or ""
        text = (ch.get("text") or "").strip()
        out.append(
            "\n".join(
                [
                    f"[{i}] {pages} | {cid}",
                    f"Header: {header}",
                    "Excerpt:",
                    text,
                ]
            )
        )
    return "\n\n".join(out)


def openrouter_chat(
    *,
    api_key: str,
    model: str,
    question: str,
    references: str,
    temperature: float = 0.2,
    strict_citations: bool = False,
) -> str:
    base_rules = """\
    You answer questions about a truck maintenance manual.

    Rules:
    - Use ONLY the provided references as the source of truth.
    - If the references do not contain enough information, say what is missing.
    - Be concise, safety-first, and do not invent steps/specs.
    - Every factual statement MUST be supported by citations like [1], [2] immediately after the sentence.
    """
    strict_rule = """\
    STRICT CITATIONS MODE:
    - Do not write any sentence unless you can cite it.
    - If you are missing information, write a short "Missing info" section with bullets (still cite what you can).
    """

    sys_prompt = textwrap.dedent(base_rules + ("\n" + strict_rule if strict_citations else "")).strip()

    user_prompt = textwrap.dedent(
        f"""\
        Question:
        {question}

        References:
        {references}
        """
    ).strip()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(data, indent=2)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid RAG QA over manual chunks.json via OpenRouter")
    p.add_argument("--chunks", required=True, help="Path to chunks.json")
    p.add_argument("--question", required=True, help="User question")
    p.add_argument("--topk", type=int, default=8, help="How many chunks to send as references")
    p.add_argument("--bm25-k", type=int, default=80, help="BM25 candidate pool")
    p.add_argument("--vec-k", type=int, default=80, help="Vector candidate pool")
    p.add_argument("--alpha", type=float, default=0.6, help="Vector weight in fusion (0..1)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model id")
    p.add_argument("--cache-dir", default=".cache", help="Embedding cache directory")
    p.add_argument("--rebuild-embeddings", action="store_true", help="Force rebuild embeddings cache")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--print-references", action="store_true", help="Print retrieved references before answering")
    p.add_argument("--print-sources", action="store_true", help="Print a short [n] -> source map after answering")
    p.add_argument("--strict-citations", action="store_true", help="Force the model to cite every sentence")
    return p.parse_args(argv)


def _print_sources_map(top_chunks: List[Dict[str, Any]]) -> None:
    print("=== Cited sources map ===")
    for i, ch in enumerate(top_chunks, start=1):
        cid = ch.get("chunk_id")
        pages = f"p{ch.get('page_start')}–p{ch.get('page_end')}"
        header = ch.get("contextual_header") or ""
        print(f"[{i}] {pages} | {cid}")
        print(f"     {header}")
    print()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY in environment.")

    chunks = load_chunks(args.chunks)
    bm25 = bm25_index(chunks)
    emb, _ = embed_corpus(
        chunks,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=args.cache_dir,
        chunks_path=args.chunks,
        rebuild=args.rebuild_embeddings,
    )
    idx = faiss_index(emb)

    ranked = hybrid_search(
        chunks,
        bm25=bm25,
        vec_index=idx,
        corpus_emb=emb,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        query=args.question,
        topk=args.topk,
        bm25_k=args.bm25_k,
        vec_k=args.vec_k,
        alpha=args.alpha,
    )
    top_chunks = [r.chunk for r in ranked]

    refs = build_references(top_chunks)
    if args.print_references:
        print("=== Retrieved references ===")
        print(refs)
        print()

    answer = openrouter_chat(
        api_key=api_key,
        model=args.model,
        question=args.question,
        references=refs,
        temperature=args.temperature,
        strict_citations=args.strict_citations,
    )
    print(answer.strip())
    if args.print_sources:
        print()
        _print_sources_map(top_chunks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

