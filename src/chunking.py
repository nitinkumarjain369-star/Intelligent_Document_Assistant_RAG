"""
Phase 2: Document Chunking
---------------------------
Splits cleaned document text into overlapping, manageable chunks that will
later be embedded and indexed. Supports comparing multiple chunk-size /
overlap configurations, per the project guidelines.
"""

from dataclasses import dataclass, field
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.ingestion import Document


@dataclass
class Chunk:
    """A single chunk of text ready for embedding."""
    chunk_id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Chunk]:
    """
    Split a list of Documents into Chunks using a recursive character
    splitter (tries to break on paragraph/sentence boundaries first,
    which keeps chunks semantically coherent).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Chunk] = []
    for doc in documents:
        pieces = splitter.split_text(doc.text)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.source}::chunk_{i}",
                    text=piece,
                    source=doc.source,
                    metadata={
                        "chunk_index": i,
                        "chunk_size_setting": chunk_size,
                        "chunk_overlap_setting": chunk_overlap,
                    },
                )
            )
    return chunks


def compare_chunk_sizes(
    documents: List[Document],
    configs: List[dict] = None,
) -> dict:
    """
    Run chunking with several configurations and report basic stats so you
    can justify your final choice in the evaluation report / notebook.

    configs example:
        [{"chunk_size": 500, "chunk_overlap": 50},
         {"chunk_size": 1000, "chunk_overlap": 150},
         {"chunk_size": 1500, "chunk_overlap": 200}]
    """
    if configs is None:
        configs = [
            {"chunk_size": 500, "chunk_overlap": 50},
            {"chunk_size": 1000, "chunk_overlap": 150},
            {"chunk_size": 1500, "chunk_overlap": 200},
        ]

    results = {}
    for cfg in configs:
        chunks = chunk_documents(documents, **cfg)
        lengths = [len(c.text) for c in chunks]
        key = f"size={cfg['chunk_size']}_overlap={cfg['chunk_overlap']}"
        results[key] = {
            "num_chunks": len(chunks),
            "avg_chunk_length": sum(lengths) / len(lengths) if lengths else 0,
            "min_chunk_length": min(lengths) if lengths else 0,
            "max_chunk_length": max(lengths) if lengths else 0,
        }
    return results


if __name__ == "__main__":
    from src.ingestion import load_documents
    import sys

    if len(sys.argv) < 2:
        print("Usage: python chunking.py <pdf_path1> <pdf_path2> ...")
        sys.exit(1)

    docs = load_documents(sys.argv[1:])
    stats = compare_chunk_sizes(docs)
    for config_name, s in stats.items():
        print(config_name, s)
