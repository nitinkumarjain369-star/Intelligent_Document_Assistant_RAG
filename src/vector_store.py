"""
Phase 3: Embedding Generation + Vector Database
-------------------------------------------------
Generates vector embeddings for text chunks using an open-source
sentence-transformers model, and stores/retrieves them with FAISS.
"""

import os
import pickle
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document as LCDocument

from src.chunking import Chunk

# Open-source embedding model (no API key required, runs locally).
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """Load the open-source embedding model."""
    return HuggingFaceEmbeddings(model_name=model_name)


def build_vector_store(chunks: List[Chunk], embedding_model=None) -> FAISS:
    """
    Embed all chunks and build a FAISS index.
    Each chunk becomes a LangChain Document so metadata (source, chunk id)
    travels with it and can be shown alongside answers later.
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()

    lc_documents = [
        LCDocument(
            page_content=chunk.text,
            metadata={
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                **chunk.metadata,
            },
        )
        for chunk in chunks
    ]

    vector_store = FAISS.from_documents(lc_documents, embedding_model)
    return vector_store


def save_vector_store(vector_store: FAISS, path: str = "data/faiss_index") -> None:
    """Persist the FAISS index to disk so it doesn't need to be rebuilt every run."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    vector_store.save_local(path)


def load_vector_store(path: str = "data/faiss_index", embedding_model=None) -> FAISS:
    """Load a previously saved FAISS index from disk."""
    if embedding_model is None:
        embedding_model = get_embedding_model()
    return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)


if __name__ == "__main__":
    from src.ingestion import load_documents
    from src.chunking import chunk_documents
    import sys

    if len(sys.argv) < 2:
        print("Usage: python vector_store.py <pdf_path1> <pdf_path2> ...")
        sys.exit(1)

    docs = load_documents(sys.argv[1:])
    chunks = chunk_documents(docs)
    store = build_vector_store(chunks)
    save_vector_store(store)
    print(f"Indexed {len(chunks)} chunks into FAISS and saved to data/faiss_index")
