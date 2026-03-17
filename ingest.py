"""
ingest.py — Document Ingestion Pipeline
"""

import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from google import genai
from langchain_core.documents import Document

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

PDF_FOLDER = "data/pdfs"
VECTOR_STORE_PATH = "vector_store"


def main():
    print("\n[*] Starting document ingestion pipeline ...\n")
    documents = []
    
    # 1. Load
    print("Step 1/3 - Loading PDFs ...")
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            filepath = os.path.join(PDF_FOLDER, filename)
            print(f"  >> Loading: {filename}")
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
            
    if not documents:
        print("  [!] No PDFs found in data/pdfs/. Add some and re-run.")
        return

    # 2. Chunk
    print("Step 2/3 - Chunking text ...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"  [OK] Created {len(chunks)} chunks.\n")

    # 3. Embed & Save
    print("Step 3/3 - Creating Google Gemini embeddings & saving FAISS index ...")
    
    client = genai.Client()
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
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text
            )
            return response.embeddings[0].values

    embeddings = CustomGoogleEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)

    print(f"\n  [OK] Vector store saved to '{VECTOR_STORE_PATH}/'.\n")
    print("[DONE] Ingestion complete! You can now run the API.\n")

if __name__ == "__main__":
    main()
