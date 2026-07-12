"""
End-to-end RAG Pipeline
--------------------------
Wires together Phases 1-5:
  Ingestion -> Chunking -> Embedding/Indexing -> Retrieval -> Generation

This is the class the Streamlit app (Phase 6) calls directly.
"""

from typing import List

from src.ingestion import load_documents, Document
from src.chunking import chunk_documents, Chunk
from src.vector_store import build_vector_store, get_embedding_model
from src.retrieval import retrieve_relevant_chunks, RetrievedChunk
from src.generation import generate_answer, get_llm


class RAGPipeline:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150, top_k: int = 4):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self.embedding_model = get_embedding_model()
        self.llm = None  # lazily created, since it needs an API key
        self.vector_store = None
        self.documents: List[Document] = []
        self.chunks: List[Chunk] = []

    def ingest(self, file_paths: List[str]) -> None:
        """Phase 1 + 2 + 3: load PDFs, chunk them, and build the FAISS index."""
        self.documents = load_documents(file_paths)
        self.chunks = chunk_documents(
            self.documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.vector_store = build_vector_store(self.chunks, self.embedding_model)

    def add_documents(self, file_paths: List[str]) -> None:
        """Ingest additional PDFs into an already-built index."""
        new_docs = load_documents(file_paths)
        new_chunks = chunk_documents(
            new_docs, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        self.documents.extend(new_docs)
        self.chunks.extend(new_chunks)

        if self.vector_store is None:
            self.vector_store = build_vector_store(new_chunks, self.embedding_model)
        else:
            more = build_vector_store(new_chunks, self.embedding_model)
            self.vector_store.merge_from(more)

    def ask(self, question: str) -> dict:
        """Phase 4 + 5: retrieve relevant chunks, then generate a grounded answer."""
        if self.vector_store is None:
            raise ValueError("No documents have been ingested yet. Call ingest() first.")

        retrieved: List[RetrievedChunk] = retrieve_relevant_chunks(
            self.vector_store, question, top_k=self.top_k
        )

        if self.llm is None:
            self.llm = get_llm()

        answer = generate_answer(question, retrieved, self.llm)

        return {
            "answer": answer,
            "sources": [
                {"source": c.source, "score": c.score, "snippet": c.text[:200]}
                for c in retrieved
            ],
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rag_pipeline.py <pdf_path1> <pdf_path2> ...")
        sys.exit(1)

    pipeline = RAGPipeline()
    pipeline.ingest(sys.argv[1:])
    print(f"Ingested {len(pipeline.documents)} document(s), {len(pipeline.chunks)} chunk(s).")

    while True:
        q = input("\nAsk a question (or 'quit'): ")
        if q.lower() in {"quit", "exit"}:
            break
        result = pipeline.ask(q)
        print("\nANSWER:\n", result["answer"])
        print("\nSOURCES:")
        for s in result["sources"]:
            print(f"  - {s['source']} (score={s['score']}): {s['snippet']}...")
