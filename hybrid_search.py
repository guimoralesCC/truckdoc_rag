#!/usr/bin/env python3
"""
Hybrid retrieval: BM25 + vector search (SentenceTransformers) over `chunks.json`.

-----------
- BM25 is excellent for exact terms (part numbers, MOP numbers, WARNING).
- Embeddings recover relevant chunks when wording differs (semantic recall).

This script:
1) loads `chunks.json`
2) builds BM25 index (header + text)
3) builds/loads embeddings cache (default local model: `all-MiniLM-L6-v2`)
4) runs BM25 + vector search
5) fuses results via weighted score and prints top-k.

Usage
-----
source .venv/bin/activate
python hybrid_search.py --chunks chunks.json --query "cab tilting FLA WARNING" --topk 8

Options
-------
--model: sentence-transformers model name (local)
--alpha: weight for vector score (0..1). BM25 weight = (1-alpha)
--bm25-k / --vec-k: candidate pool sizes before fusion
--rebuild: force rebuild embeddings cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "faiss-cpu is required for hybrid_search.py. Install with: pip install faiss-cpu\n"
        f"Import error: {e}"
    )


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*")


def tokenize(s: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(s or "")]


def load_chunks(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError("Invalid chunks.json: expected top-level key 'chunks' to be a list")
    return chunks


def bm25_index(chunks: List[Dict[str, Any]]) -> BM25Okapi:
    corpus = []
    for ch in chunks:
        doc = f"{ch.get('contextual_header','')}\n{ch.get('text','')}"
        corpus.append(tokenize(doc))
    return BM25Okapi(corpus)


def _cache_key(chunks_path: str, model_name: str, count: int) -> str:
    h = hashlib.sha256()
    h.update(os.path.abspath(chunks_path).encode("utf-8"))
    h.update(model_name.encode("utf-8"))
    h.update(str(count).encode("utf-8"))
    return h.hexdigest()[:16]


def embed_corpus(
    chunks: List[Dict[str, Any]],
    *,
    model_name: str,
    cache_dir: str,
    chunks_path: str,
    rebuild: bool,
) -> Tuple[np.ndarray, str]:
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(chunks_path, model_name, len(chunks))
    cache_path = os.path.join(cache_dir, f"embeddings_{key}.npy")

    if (not rebuild) and os.path.exists(cache_path):
        emb = np.load(cache_path)
        return emb, cache_path

    model = SentenceTransformer(model_name)
    docs = [f"{ch.get('contextual_header','')}\n{ch.get('text','')}" for ch in chunks]
    emb = model.encode(docs, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")
    np.save(cache_path, emb)
    return emb, cache_path


def faiss_index(emb: np.ndarray) -> faiss.Index:
    # emb is normalized => inner product = cosine similarity
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)
    return idx


def minmax_norm(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo = float(scores.min())
    hi = float(scores.max())
    if hi - lo < 1e-9:
        return np.zeros_like(scores, dtype="float32")
    return ((scores - lo) / (hi - lo)).astype("float32")


@dataclass(frozen=True)
class Ranked:
    fused: float
    bm25: float
    vec: float
    idx: int
    chunk: Dict[str, Any]


def hybrid_search(
    chunks: List[Dict[str, Any]],
    *,
    bm25: BM25Okapi,
    vec_index: faiss.Index,
    corpus_emb: np.ndarray,
    model_name: str,
    query: str,
    topk: int,
    bm25_k: int,
    vec_k: int,
    alpha: float,
) -> List[Ranked]:
    # BM25 candidates
    q_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(q_tokens) if q_tokens else np.zeros(len(chunks), dtype="float32")
    bm25_scores = np.asarray(bm25_scores, dtype="float32")
    bm25_top = np.argsort(-bm25_scores)[: min(bm25_k, len(chunks))]

    # Vector candidates
    model = SentenceTransformer(model_name)
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")
    vec_scores, vec_ids = vec_index.search(q_emb, min(vec_k, len(chunks)))
    vec_scores = vec_scores[0].astype("float32")
    vec_ids = vec_ids[0].astype("int64")

    # Candidate union
    cand = set(int(i) for i in bm25_top.tolist()) | set(int(i) for i in vec_ids.tolist() if i >= 0)
    cand_ids = np.array(sorted(cand), dtype="int64")

    # Normalize within candidate set
    bm25_c = bm25_scores[cand_ids]
    bm25_n = minmax_norm(bm25_c)

    # Get vector scores for candidates via dot product with normalized embeddings
    vec_c = (corpus_emb[cand_ids] @ q_emb[0]).astype("float32")
    vec_n = minmax_norm(vec_c)

    fused = (1.0 - alpha) * bm25_n + alpha * vec_n
    order = np.argsort(-fused)[: min(topk, fused.size)]

    out: List[Ranked] = []
    for j in order:
        i = int(cand_ids[j])
        out.append(
            Ranked(
                fused=float(fused[j]),
                bm25=float(bm25_c[j]),
                vec=float(vec_c[j]),
                idx=i,
                chunk=chunks[i],
            )
        )
    return out


def snippet(text: str, width: int = 260) -> str:
    t = " ".join((text or "").split())
    return (t[:width] + "…") if len(t) > width else t


def format_ranked(r: Ranked) -> str:
    ch = r.chunk
    pages = f"p{ch.get('page_start')}–p{ch.get('page_end')}"
    ctype = ch.get("content_type") or ""
    safety = "safety" if ch.get("safety_related") else ""
    mop = f"{ch.get('mop_number') or ''} {ch.get('mop_title') or ''}".strip()
    return "\n".join(
        [
            f"- fused={r.fused:.4f}  bm25={r.bm25:.2f}  vec={r.vec:.4f}  {pages}  {ctype}  {safety}".rstrip(),
            f"  chunk_id: {ch.get('chunk_id')}",
            f"  mop: {mop}",
            f"  header: {ch.get('contextual_header')}",
            f"  text: {snippet(ch.get('text',''))}",
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid BM25 + vector search over manual chunks.json")
    p.add_argument("--chunks", required=True, help="Path to chunks.json")
    p.add_argument("--query", required=True, help="Search query")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--bm25-k", type=int, default=60, help="BM25 candidate pool size")
    p.add_argument("--vec-k", type=int, default=60, help="Vector candidate pool size")
    p.add_argument("--alpha", type=float, default=0.55, help="Vector weight in fusion (0..1)")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model")
    p.add_argument("--cache-dir", default=".cache", help="Embedding cache directory")
    p.add_argument("--rebuild", action="store_true", help="Force rebuild embedding cache")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    chunks = load_chunks(args.chunks)
    bm25 = bm25_index(chunks)
    emb, cache_path = embed_corpus(
        chunks,
        model_name=args.model,
        cache_dir=args.cache_dir,
        chunks_path=args.chunks,
        rebuild=args.rebuild,
    )
    idx = faiss_index(emb)

    results = hybrid_search(
        chunks,
        bm25=bm25,
        vec_index=idx,
        corpus_emb=emb,
        model_name=args.model,
        query=args.query,
        topk=args.topk,
        bm25_k=args.bm25_k,
        vec_k=args.vec_k,
        alpha=args.alpha,
    )

    print(f"Embedding cache: {cache_path}")
    for r in results:
        print(format_ranked(r))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

