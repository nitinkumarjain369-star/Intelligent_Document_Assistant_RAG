# Evaluation Report — Intelligent Document Assistant (RAG)

## Summary

This project implements a complete Retrieval-Augmented Generation pipeline
that lets a user upload PDF documents and ask natural-language questions,
receiving grounded, source-cited answers through a Streamlit interface.

## Mapping to project evaluation metrics

| Metric | Weight | How it's addressed |
|---|---|---|
| **Data Preprocessing** | 15% | `src/ingestion.py` extracts text per-page with `pypdf`, then `clean_text()` strips page-break artifacts, collapses whitespace, and normalizes bullets before chunking. |
| **Retrieval Pipeline** | 25% | `src/vector_store.py` builds a FAISS index over `sentence-transformers/all-MiniLM-L6-v2` embeddings; `src/retrieval.py` performs top-k similarity search and ranks results by a normalized similarity score. Chunk-size tradeoffs are benchmarked in the notebook (`compare_chunk_sizes`). |
| **Response Quality** | 20% | `src/generation.py` uses a grounded prompt template (`PROMPT_TEMPLATE`) that instructs the LLM to answer only from retrieved context, cite sources, and explicitly say when the answer isn't in the documents — reducing hallucination. |
| **Streamlit Application** | 15% | `app.py` provides PDF upload, adjustable chunking/retrieval settings, a chat interface, and an expandable "View sources" panel per answer. |
| **Documentation** | 15% | `README.md` (setup + usage), this evaluation report, and the walkthrough notebook together document the system end-to-end. |
| **Code Quality** | 10% | Code is split into single-responsibility modules per pipeline phase (`ingestion`, `chunking`, `vector_store`, `retrieval`, `generation`), tied together by `RAGPipeline` in `rag_pipeline.py`, with type hints and docstrings throughout. |

## Design decisions & justification

**Chunking.** Default configuration: 1000-character chunks with 150-character
overlap. The notebook compares 500/1000/1500-character configurations
(`compare_chunk_sizes`): smaller chunks retrieve more precisely but risk
losing surrounding context; larger chunks preserve context but reduce
retrieval precision and increase prompt length/cost. 1000/150 balances both.

**Embeddings.** `sentence-transformers/all-MiniLM-L6-v2` was chosen because
it's open-source (per project guidelines), runs locally with no API cost,
and is fast enough for interactive use while still giving solid semantic
retrieval quality for general documents (reports, papers, manuals).

**Vector store.** FAISS (`langchain_community.vectorstores.FAISS`), as
required by the project guidelines. The index is persisted to
`data/faiss_index` so it can be reloaded without re-embedding.

**Generation.** OpenAI `gpt-4o-mini` was used for cost/latency, with a
strict grounded prompt that reduces hallucination and requires source
citation. The LLM call is isolated in `src/generation.py`, so swapping to
another provider only requires changing `get_llm()`.

## Known limitations / possible extensions

- No re-ranking model (e.g., cross-encoder) is applied after the initial
  FAISS retrieval — could improve precision on ambiguous queries.
- No persistent chat memory across sessions (Streamlit session state only
  lasts for the browser session).
- Currently supports PDF only; `ingestion.py` could be extended with
  loaders for `.docx` / `.txt` / `.md`.
- No automated evaluation harness (e.g., RAGAS) is included; manual
  spot-checking was used during development. This would be a natural next
  step for measuring retrieval and answer quality quantitatively.

## How to reproduce results

1. Follow setup steps in `README.md`.
2. Run `notebooks/RAG_Pipeline_Walkthrough.ipynb` end-to-end against a
   sample PDF (e.g., an SEC EDGAR annual report or an arXiv paper, per the
   project's recommended datasets).
3. Launch `streamlit run app.py` for the interactive demo.
