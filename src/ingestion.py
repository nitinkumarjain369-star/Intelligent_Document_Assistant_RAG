"""
Phase 1: Document Ingestion
----------------------------
Handles uploading PDFs, extracting raw text, and cleaning/preprocessing it
before it moves into the chunking stage.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List

from pypdf import PdfReader


@dataclass
class Document:
    """Simple container for a single ingested document."""
    source: str
    text: str
    metadata: dict = field(default_factory=dict)


def extract_text_from_pdf(file_path: str) -> str:
    """Extract raw text from a PDF file, page by page."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    reader = PdfReader(file_path)
    pages_text = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text)


def clean_text(raw_text: str) -> str:
    """
    Basic cleaning/preprocessing:
      - collapse excessive whitespace
      - remove page-break artifacts and stray control characters
      - normalize bullet characters
      - strip leading/trailing whitespace
    """
    text = raw_text.replace("\x0c", " ")  # form-feed / page break artifact
    text = re.sub(r"[ \t]+", " ", text)  # collapse repeated spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse excessive blank lines
    text = re.sub(r"[\u2022\u25CF\u25AA]", "-", text)  # normalize bullets
    text = text.strip()
    return text


def load_documents(file_paths: List[str]) -> List[Document]:
    """
    Ingest a list of PDF file paths and return cleaned Document objects.
    This is the single entry point Phase 2 (chunking) should call.
    """
    documents = []
    for path in file_paths:
        raw_text = extract_text_from_pdf(path)
        cleaned = clean_text(raw_text)
        documents.append(
            Document(
                source=os.path.basename(path),
                text=cleaned,
                metadata={"file_path": path, "num_chars": len(cleaned)},
            )
        )
    return documents


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingestion.py <pdf_path1> <pdf_path2> ...")
        sys.exit(1)

    docs = load_documents(sys.argv[1:])
    for d in docs:
        print(f"{d.source}: {d.metadata['num_chars']} characters extracted")
