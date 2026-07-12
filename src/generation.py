"""
Phase 5: Answer Generation
-----------------------------
Passes retrieved context + the user's question to an LLM (Google Gemini)
using a grounded prompt template, so answers are based on the retrieved
document content rather than the model's general knowledge.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

from src.retrieval import RetrievedChunk, format_context

PROMPT_TEMPLATE = """You are an intelligent document assistant. Answer the
user's question using ONLY the context below, which was retrieved from
their uploaded documents.

Rules:
- Base your answer strictly on the provided context.
- If the answer is not contained in the context, say clearly that the
  documents don't seem to contain that information. Do not make anything up.
- Cite which source(s) you used, e.g. "(Source 1)".
- Be concise and directly answer the question first, then add supporting detail.

Context:
{context}

Question: {question}

Answer:"""


def get_llm(model_name: str = "gemini-1.5-flash", temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """
    Load the Google Gemini chat model. Requires GOOGLE_API_KEY to be set as
    an environment variable (see README for setup instructions). Get a free
    key at https://aistudio.google.com/app/apikey
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Export it or add it to a .env file "
            "before running the app."
        )
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=api_key)


def generate_answer(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    llm: ChatGoogleGenerativeAI = None,
) -> str:
    """Generate a grounded answer from the retrieved chunks."""
    if llm is None:
        llm = get_llm()

    context = format_context(retrieved_chunks)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm

    response = chain.invoke({"context": context, "question": question})
    return response.content
