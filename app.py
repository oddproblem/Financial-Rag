"""
app.py — Financial Document RAG API

FastAPI server with a /ask endpoint that:
  1. Retrieves relevant chunks from a FAISS vector store
  2. Constructs a context window
  3. Sends it to an LLM (GPT-3.5-turbo)
  4. Returns the answer along with source citations

Usage:
    uvicorn app:app --reload
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ─── Load environment variables ───────────────────────────────────────
load_dotenv()

# ─── FastAPI app ──────────────────────────────────────────────────────
app = FastAPI(
    title="Financial Document RAG API",
    description="Ask questions about financial/regulatory documents and get answers with source citations.",
    version="1.0.0",
)

# ─── Load vector store & models at startup ────────────────────────────
VECTOR_STORE_PATH = "vector_store"

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local(
    VECTOR_STORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# ─── Endpoints ────────────────────────────────────────────────────────
@app.get("/")
def root():
    """Health check / welcome message."""
    return {
        "message": "Financial Document RAG API is running.",
        "docs": "Visit /docs for the Swagger UI.",
    }


@app.get("/ask")
def ask(
    question: str = Query(
        ..., description="Your question about the financial documents"
    ),
):
    """
    Ask a question.

    The system retrieves the most relevant document chunks from the
    FAISS vector store, constructs a context, queries the LLM, and
    returns the answer together with the source documents.
    """

    # 1. Retrieve relevant document chunks
    docs = retriever.invoke(question)

    # 2. Build context from retrieved chunks
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    # 3. Extract unique source file names
    sources = list({doc.metadata.get("source", "unknown") for doc in docs})

    # 4. Build prompt and query the LLM
    system_prompt = (
        "You are a helpful financial analyst assistant. "
        "Answer the user's question based ONLY on the provided context. "
        "If the context does not contain enough information, say so. "
        "Be concise and cite relevant details."
    )

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    # 5. Return structured response
    return {
        "question": question,
        "answer": response.content,
        "sources": sources,
    }
