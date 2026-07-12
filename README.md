# Intelligent Document Assistant using RAG

An AI-powered document assistant that lets you upload PDF documents and ask
natural-language questions about them. It retrieves the most relevant
sections of your documents and uses an LLM to generate a grounded,
context-aware answer with cited sources.

Built for the GUVI/HCL capstone project brief: *Intelligent Document
Assistant using Retrieval-Augmented Generation (RAG)*.

## How it works (pipeline overview)

```
PDF Upload → Text Extraction & Cleaning → Chunking → Embedding (FAISS)
    → Retrieval (top-k similarity search) → Grounded Answer Generation (LLM)
    → Streamlit Chat UI
```

See `architecture_diagram.svg` for the visual diagram and
`notebooks/RAG_Pipeline_Walkthrough.ipynb` for a step-by-step walkthrough of
every phase.

## Project structure

```
rag_project/
├── app.py                     # Phase 6: Streamlit application
├── requirements.txt
├── .env.example                # Template for your API key
├── architecture_diagram.svg
├── README.md                   # This file
├── EVALUATION_REPORT.md
├── notebooks/
│   └── RAG_Pipeline_Walkthrough.ipynb
├── data/                       # Put sample PDFs here / FAISS index gets saved here
└── src/
    ├── ingestion.py             # Phase 1: PDF loading, text extraction, cleaning
    ├── chunking.py               # Phase 2: chunking + chunk-size comparison
    ├── vector_store.py           # Phase 3: embeddings + FAISS index
    ├── retrieval.py               # Phase 4: similarity search + ranking
    ├── generation.py              # Phase 5: grounded LLM answer generation
    └── rag_pipeline.py            # Glue class used by app.py and the notebook
```

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your OpenAI API key

Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=sk-...your-key...
```

(You can also just paste the key into the sidebar field when the Streamlit
app is running — it isn't required to be in `.env`.)

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`),
upload one or more PDFs, click **Process Documents**, and start asking
questions in the chat box.

### 4. (Optional) Run the notebook

```bash
jupyter notebook notebooks/RAG_Pipeline_Walkthrough.ipynb
```

Point the `sample_files` variable at a PDF in `data/` to walk through
every phase interactively — useful for demoing the internals during your
live evaluation.

### 5. (Optional) Run from the command line

```bash
python -m src.rag_pipeline data/sample_document.pdf
```

This ingests the file and drops you into an interactive question loop in
the terminal.

## Configuration

You can tune these directly in the Streamlit sidebar, or as arguments to
`RAGPipeline(...)`:

| Setting | Default | What it does |
|---|---|---|
| `chunk_size` | 1000 chars | Size of each text chunk before embedding |
| `chunk_overlap` | 150 chars | Overlap between consecutive chunks, to avoid cutting context at boundaries |
| `top_k` | 4 | Number of chunks retrieved per question |

`src/chunking.py` also includes `compare_chunk_sizes()`, used in the
notebook to justify these defaults against alternative configurations.

## Key design choices

- **Embeddings**: open-source `sentence-transformers/all-MiniLM-L6-v2`
  (runs locally, no API cost, per the project guideline to use an
  open-source embedding model).
- **Vector database**: FAISS (as required by the project guidelines),
  persisted to disk in `data/faiss_index` so it doesn't need to be
  rebuilt on every run.
- **LLM**: OpenAI `gpt-4o-mini` via LangChain, using a strict grounded
  prompt template so answers stick to the retrieved context and the
  model says so explicitly when the documents don't contain the answer.
- **Chunking**: `RecursiveCharacterTextSplitter`, which tries paragraph
  and sentence boundaries first so chunks stay semantically coherent
  rather than cutting mid-sentence.

## Troubleshooting

- **"OPENAI_API_KEY is not set"** — enter your key in the Streamlit
  sidebar, or create a `.env` file from `.env.example`.
- **Slow first run** — the embedding model downloads once (~90MB) the
  first time you run the app; subsequent runs use the cached copy.
- **Answers seem off-topic** — try increasing `top_k`, or reduce
  `chunk_size` so retrieval is more precise.
