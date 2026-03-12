# Financial Document RAG Assistant

A retrieval-augmented generation (RAG) system built with **LangChain**, **FAISS**, and **FastAPI** that answers natural-language questions about financial/regulatory documents and returns source citations.

## Architecture

```text
PDF Documents  →  Text Chunking  →  OpenAI Embeddings  →  FAISS Vector DB
                                                                ↓
User Question  →  Retriever  →  Context Assembly  →  LLM (GPT-3.5)  →  Answer + Sources
```

## Tech Stack

| Layer         | Technology                  |
|---------------|-----------------------------|
| Framework     | FastAPI                     |
| LLM           | OpenAI GPT-3.5-Turbo        |
| Embeddings    | OpenAI Embeddings           |
| Vector Store  | FAISS (CPU)                 |
| Orchestration | LangChain                   |
| PDF Parsing   | PyPDF                       |

## Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/oddproblem/Financial-Rag.git
cd Financial-Rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux / Mac
.\venv\Scripts\Activate.ps1     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your OpenAI API key
cp .env.example .env
# Edit .env and paste your key

# 5. Place financial PDFs in data/pdfs/

# 6. Run the ingestion pipeline
python ingest.py

# 7. Start the API server
uvicorn app:app --reload
```

## Usage

Open **http://127.0.0.1:8000/docs** for the Swagger UI, or query directly:

```
GET /ask?question=What are the Basel III capital requirements?
```

### Example Response

```json
{
  "question": "What are the Basel III capital requirements?",
  "answer": "Basel III requires banks to hold a minimum Common Equity Tier 1 (CET1) capital ratio of 4.5% …",
  "sources": ["data/pdfs/basel3_framework.pdf"]
}
```

## Project Structure

```
Financial-Rag/
├── app.py              # FastAPI backend with /ask endpoint
├── ingest.py           # PDF → Chunks → Embeddings → FAISS
├── requirements.txt    # Python dependencies
├── .env.example        # Template for environment variables
├── TASK_SHEET.md       # Elaborated project task sheet
├── data/
│   └── pdfs/           # Place your financial PDFs here
└── vector_store/       # Auto-generated FAISS index (gitignored)
```
