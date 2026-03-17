"""
app.py — Financial Document RAG API

FastAPI server with a /ask endpoint that:
  1. Retrieves relevant chunks from a FAISS vector store
  2. Constructs a context window
  3. Sends it to Google Gemini LLM
  4. Returns the answer along with source citations

Usage:
    uvicorn app:app --reload
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from google import genai
from google.genai import types

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

# Use GEMINI_API_KEY if available, else fallback to GOOGLE_API_KEY
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

def embed_query(text: str) -> list[float]:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values

def embed_documents(texts: list[str]) -> list[list[float]]:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
    )
    return [e.values for e in response.embeddings]

class CustomGoogleEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_documents(texts)
    def embed_query(self, text: str) -> list[float]:
        return embed_query(text)
    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)

embeddings = CustomGoogleEmbeddings()
vectorstore = FAISS.load_local(
    VECTOR_STORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


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
    FAISS vector store, constructs a context, queries the Gemini LLM, and
    returns the answer together with the source documents.
    """
    try:
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

        response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=[
                types.Content(role="user", parts=[types.Part.from_text(text=system_prompt)]),
                types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])
            ],
            config=types.GenerateContentConfig(temperature=0.0)
        )

        # 5. Return structured response
        return {
            "question": question,
            "answer": response.text,
            "sources": sources,
        }
    except Exception as e:
        print(f"Error in /ask: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
