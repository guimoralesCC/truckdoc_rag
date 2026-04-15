#!/usr/bin/env python3
"""
Streamlit GUI for manual Q&A (Hybrid RAG + OpenRouter).

Run
---
source .venv/bin/activate
export OPENROUTER_API_KEY="..."
streamlit run gui_app.py
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import streamlit as st

from answer_manual import build_references, openrouter_chat
from hybrid_search import bm25_index, embed_corpus, faiss_index, hybrid_search, load_chunks


st.set_page_config(page_title="Manual Q&A", layout="wide")


@st.cache_data(show_spinner=False)
def _load_chunks(path: str) -> List[Dict[str, Any]]:
    return load_chunks(path)


@st.cache_resource(show_spinner=False)
def _build_retrieval(chunks_path: str, model_name: str, cache_dir: str, rebuild_embeddings: bool):
    chunks = _load_chunks(chunks_path)
    bm25 = bm25_index(chunks)
    emb, cache_path = embed_corpus(
        chunks,
        model_name=model_name,
        cache_dir=cache_dir,
        chunks_path=chunks_path,
        rebuild=rebuild_embeddings,
    )
    idx = faiss_index(emb)
    return chunks, bm25, emb, idx, cache_path


def _sources_table(top_chunks: List[Dict[str, Any]]):
    rows = []
    for i, ch in enumerate(top_chunks, start=1):
        rows.append(
            {
                "cite": f"[{i}]",
                "pages": f"p{ch.get('page_start')}–p{ch.get('page_end')}",
                "chunk_id": ch.get("chunk_id"),
                "mop": f"{ch.get('mop_number') or ''} {ch.get('mop_title') or ''}".strip(),
                "header": ch.get("contextual_header"),
                "content_type": ch.get("content_type"),
                "safety": bool(ch.get("safety_related")),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    for i, ch in enumerate(top_chunks, start=1):
        with st.expander(f"[{i}] {ch.get('chunk_id')}  ({ch.get('page_start')}–{ch.get('page_end')})", expanded=False):
            st.write(ch.get("contextual_header") or "")
            st.code((ch.get("text") or "").strip())


def main() -> None:
    st.title("Truck Manual Q&A (Hybrid RAG)")

    with st.sidebar:
        st.header("Settings")
        chunks_path = st.text_input("chunks.json path", value="chunks.json")
        model_name = st.text_input("Embedding model", value="sentence-transformers/all-MiniLM-L6-v2")
        llm_model = st.text_input("OpenRouter model", value="openai/gpt-4o-mini")
        cache_dir = st.text_input("Embedding cache dir", value=".cache")

        col1, col2 = st.columns(2)
        with col1:
            topk = st.number_input("Top-k refs", min_value=3, max_value=20, value=8, step=1)
            bm25_k = st.number_input("BM25 candidate pool", min_value=10, max_value=500, value=80, step=10)
        with col2:
            vec_k = st.number_input("Vector candidate pool", min_value=10, max_value=500, value=80, step=10)
            alpha = st.slider("Vector weight (alpha)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

        strict = st.checkbox("Strict citations (every sentence must cite)", value=True)
        temp = st.slider("LLM temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        rebuild = st.checkbox("Rebuild embeddings cache", value=False)

        api_key_present = bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
        st.caption(f"OPENROUTER_API_KEY: {'set' if api_key_present else 'NOT set'}")

    question = st.text_area("Ask a question", placeholder="e.g. How do I tilt the FLA cab safely?", height=90)
    ask = st.button("Answer", type="primary", disabled=not question.strip())

    if ask:
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            st.error("Missing OPENROUTER_API_KEY environment variable.")
            return

        with st.spinner("Building/Loading indexes…"):
            chunks, bm25, emb, idx, cache_path = _build_retrieval(chunks_path, model_name, cache_dir, rebuild)

        with st.spinner("Retrieving relevant sections…"):
            ranked = hybrid_search(
                chunks,
                bm25=bm25,
                vec_index=idx,
                corpus_emb=emb,
                model_name=model_name,
                query=question,
                topk=int(topk),
                bm25_k=int(bm25_k),
                vec_k=int(vec_k),
                alpha=float(alpha),
            )
            top_chunks = [r.chunk for r in ranked]
            refs = build_references(top_chunks)

        with st.spinner("Calling LLM…"):
            answer = openrouter_chat(
                api_key=api_key,
                model=llm_model,
                question=question,
                references=refs,
                temperature=float(temp),
                strict_citations=bool(strict),
            )

        left, right = st.columns([1.2, 1.0], gap="large")
        with left:
            st.subheader("Answer")
            st.markdown(answer)
            st.caption(f"Embedding cache: `{cache_path}`")

        with right:
            st.subheader("Sources (pages + excerpts)")
            _sources_table(top_chunks)


if __name__ == "__main__":
    main()

