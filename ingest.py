"""
ingest.py — Document Ingestion Pipeline

Loads PDFs from data/pdfs/, splits text into chunks,
creates OpenAI embeddings, and stores them in a local FAISS vector database.

Usage:
    python ingest.py
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# ─── Load environment variables ───────────────────────────────────────
load_dotenv()

PDF_FOLDER = "data/pdfs"
VECTOR_STORE_PATH = "vector_store"


def load_documents(folder: str) -> list:
    """Load all PDF files from the given folder."""
    documents = []
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder, filename)
            print(f"  📄 Loading: {filename}")
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
    return documents


def chunk_documents(documents: list) -> list:
    """Split documents into smaller overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks: list, save_path: str):
    """Embed the chunks and persist a FAISS index to disk."""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    return vectorstore


def main():
    print("\n🔄  Starting document ingestion pipeline …\n")

    # 1. Load
    print("Step 1/3 — Loading PDFs …")
    documents = load_documents(PDF_FOLDER)
    print(f"  ✅ Loaded {len(documents)} pages from PDFs.\n")

    if not documents:
        print("  ⚠️  No PDFs found in data/pdfs/. Add some and re-run.")
        return

    # 2. Chunk
    print("Step 2/3 — Chunking text …")
    chunks = chunk_documents(documents)
    print(f"  ✅ Created {len(chunks)} chunks.\n")

    # 3. Embed & Save
    print("Step 3/3 — Creating embeddings & saving FAISS index …")
    create_vector_store(chunks, VECTOR_STORE_PATH)
    print(f"  ✅ Vector store saved to '{VECTOR_STORE_PATH}/'.\n")

    print("🎉  Ingestion complete! You can now run the API.\n")


if __name__ == "__main__":
    main()
