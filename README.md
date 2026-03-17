# Financial Document RAG Assistant

A Retrieval-Augmented Generation (RAG) system developed for high-precision querying of financial and regulatory documents. The system utilizes the Google Gemini 3 API for both semantic embeddings and language generation, indexed within a FAISS vector store.

## Architecture

The data pipeline consists of the following stages:

1. Ingestion: PDF parsing -> Text chunking -> Gemini embedding generation -> FAISS indexing.
2. Retrieval: Query embedding -> Vector similarity search -> Context extraction.
3. Generation: System prompt construction -> Gemini 3 Flash generation -> Structured response with citations.

## Tech Stack

| Component | Technology |
|---|---|
| Runtime | Python 3.10+ |
| Backend Framework | FastAPI |
| LLM | Google Gemini 3 (Flash Preview) |
| Embeddings | Google Gemini Embeddings (001 series) |
| Vector Store | FAISS (CPU-optimized) |
| Documents | PyPDF |
| Env Management | Python Dotenv |

## Prerequisites

- Python 3.10 or higher
- A Google Gemini API Key
- Command Line Interface (CLI) access

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/oddproblem/Financial-Rag.git
   cd Financial-Rag
   ```

2. Initialize virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate.bat
   # On Unix/macOS:
   source venv/bin/activate
   ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   Create a `.env` file in the root directory based on `.env.example`:
   ```text
   GOOGLE_API_KEY=your_gemini_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## Workflow

### 1. Document Ingestion
Place your financial PDF documents in the `data/pdfs/` directory. Run the ingestion pipeline to process and index them:
```bash
python ingest.py
```

### 2. Launching the API
Start the FastAPI server using Uvicorn:
```bash
uvicorn app:app --reload
```

### 3. Querying the System
Access the interactive API documentation at `http://127.0.0.1:8000/docs` or perform a direct GET request:
```bash
curl "http://127.0.0.1:8000/ask?question=What+are+the+capital+requirements"
```

## Project Structure

- `app.py`: Main FastAPI application logic and retrieval chain.
- `ingest.py`: Script for document loading and vector store construction.
- `requirements.txt`: Project dependency declarations.
- `data/pdfs/`: Storage for source financial documents.
- `vector_store/`: Local directory containing the generated FAISS index.
- `TASK_SHEET.md`: Implementation status and feature roadmap.

## Disclaimer

This tool is for informational purposes and should not be considered financial advice. Verify all responses against the original cited source documents.
