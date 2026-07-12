"""
Phase 6: Application Development
------------------------------------
Streamlit interface for the Intelligent Document Assistant.
Run with:  streamlit run app.py
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from src.rag_pipeline import RAGPipeline

load_dotenv()  # picks up OPENAI_API_KEY from a .env file if present

st.set_page_config(page_title="Intelligent Document Assistant", page_icon="📄", layout="wide")

st.title("📄 Intelligent Document Assistant (RAG)")
st.caption("Upload PDF documents, then ask natural-language questions about them.")

# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []

# ---------------------------------------------------------------------------
# Sidebar: settings + API key + upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.environ.get("OPENAI_API_KEY", ""),
        help="Required for answer generation. Stored only for this session.",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    chunk_size = st.slider("Chunk size (characters)", 300, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, 150, step=25)
    top_k = st.slider("Chunks to retrieve per question", 1, 10, 4)

    st.divider()
    st.header("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Process Documents", type="primary", disabled=not uploaded_files):
        if not api_key:
            st.error("Please enter your OpenAI API key first.")
        else:
            with st.spinner("Extracting text, chunking, and building the vector index..."):
                temp_paths = []
                for uploaded in uploaded_files:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    tmp.write(uploaded.getbuffer())
                    tmp.close()
                    temp_paths.append(tmp.name)

                pipeline = RAGPipeline(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=top_k
                )
                pipeline.ingest(temp_paths)
                st.session_state.pipeline = pipeline
                st.session_state.ingested_files = [f.name for f in uploaded_files]
                st.session_state.chat_history = []

            st.success(
                f"Indexed {len(pipeline.chunks)} chunks from "
                f"{len(st.session_state.ingested_files)} file(s)."
            )

    if st.session_state.ingested_files:
        st.divider()
        st.subheader("📚 Indexed files")
        for name in st.session_state.ingested_files:
            st.write(f"- {name}")

# ---------------------------------------------------------------------------
# Main panel: chat interface
# ---------------------------------------------------------------------------
if st.session_state.pipeline is None:
    st.info("👈 Upload PDFs and click **Process Documents** to get started.")
else:
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                with st.expander("View sources"):
                    for s in turn["sources"]:
                        st.markdown(
                            f"**{s['source']}** (relevance: {s['score']})\n\n> {s['snippet']}..."
                        )

    question = st.chat_input("Ask a question about your documents...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant sections and generating an answer..."):
                try:
                    result = st.session_state.pipeline.ask(question)
                    st.markdown(result["answer"])
                    with st.expander("View sources"):
                        for s in result["sources"]:
                            st.markdown(
                                f"**{s['source']}** (relevance: {s['score']})\n\n> {s['snippet']}..."
                            )
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                        }
                    )
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
