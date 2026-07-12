"""
Phase 4: Retrieval Pipeline
-----------------------------
Given a user question, retrieve the most relevant chunks from the FAISS
vector store and rank them by similarity score.
"""

from dataclasses import dataclass
from typing import List

from langchain_community.vectorstores import FAISS


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_id: str
    score: float


def retrieve_relevant_chunks(
    vector_store: FAISS,
    query: str,
    top_k: int = 4,
) -> List[RetrievedChunk]:
    """
    Retrieve the top_k most relevant chunks for a query, ranked by
    similarity score (lower FAISS L2 distance = more relevant, so we
    convert to a similarity-style score for readability).
    """
    results = vector_store.similarity_search_with_score(query, k=top_k)

    ranked: List[RetrievedChunk] = []
    for doc, distance in results:
        similarity = 1 / (1 + distance)  # convert distance -> 0-1 similarity
        ranked.append(
            RetrievedChunk(
                text=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                chunk_id=doc.metadata.get("chunk_id", "unknown"),
                score=round(similarity, 4),
            )
        )

    # Already sorted by FAISS, but re-sort explicitly to make the "ranking"
    # step in the pipeline explicit and easy to point to in an evaluation.
    ranked.sort(key=lambda c: c.score, reverse=True)
    return ranked


def format_context(chunks: List[RetrievedChunk]) -> str:
    """Combine retrieved chunks into a single context block for the LLM prompt."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[Source {i}: {c.source} | relevance={c.score}]\n{c.text}")
    return "\n\n---\n\n".join(blocks)
