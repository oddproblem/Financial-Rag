# Financial Document RAG Assistant - Elaborated Task Sheet

## Phase 1: Environment & Setup (Completed)
- [x] **Project Folder & Env**: Created `financial-rag-assistant` and initialized isolated `venv`.
- [x] **Dependencies**: Installed `langchain`, `langchain-community`, `langchain-openai`, `faiss-cpu`, `pypdf`, `fastapi`, `uvicorn`, `python-dotenv`.
- [x] **Project Structure**: Set up `data/pdfs/` for raw documents, `vector_store/` for FAISS DB, and empty `app.py` & `ingest.py`.

## Phase 2: Data Gathering & Preparation
- [x] **Download PDFs**: 
  - Go to official sources (e.g., BIS for Basel III, RBI, World Bank).
  - Download 3 to 5 financial/regulatory documents in `.pdf` format.
  - Move the downloaded files into the `data/pdfs/` directory in this project.
- [x] **Environment Variables**:
  - Create a `.env` file in the root directory.
  - Add your Gemini API key: `GOOGLE_API_KEY=your_actual_api_key_here`.

## Phase 3: The Ingestion Pipeline (`ingest.py`)
*Goal: Convert PDFs into searchable mathematical vectors.*
- [x] **Import Libraries**: Import `PyPDFLoader`, `RecursiveCharacterTextSplitter`, `FAISS`, `GoogleGenerativeAIEmbeddings`, `dotenv` and `os`.
- [x] **Load Credentials**: Call `load_dotenv()` to pull in your Gemini API Key.
- [x] **Iterate and Load Files**:
  - Loop through `os.listdir("data/pdfs")`.
  - For each `.pdf` file, use `PyPDFLoader(file_path)` and call `.load()`.
  - Accumulate all returned pages into a single `documents` list.
- [x] **Chunk the Text**:
  - Initialize `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`.
  - Pass the `documents` list into `text_splitter.split_documents(documents)`.
- [x] **Create Vector DB**:
  - Initialize the embedding model: `embeddings = GoogleGenerativeAIEmbeddings()`.
  - Pass the chunked text and the embedding model to `FAISS.from_documents()`.
- [x] **Save and Run**:
  - Save the resulting vector DB: `vectorstore.save_local("vector_store")`.
  - Open terminal and run `python ingest.py`. Verify `vector_store/` populates with `.faiss` and `.pkl` files.

## Phase 4: The FastAPI Backend & RAG Chain (`app.py`)
*Goal: Create an API endpoint that handles the retrieval-augmented generation.*
- [x] **Import Libraries**: `FastAPI`, `FAISS`, `GoogleGenerativeAIEmbeddings`, `ChatGoogleGenerativeAI`, `RetrievalQA`, etc.
- [x] **Initialize the Application**: 
  - `app = FastAPI(title="Financial RAG API")`
  - Load environment variables using `dotenv`.
- [x] **Load Vector Store**:
  - Re-initialize `GoogleGenerativeAIEmbeddings()`.
  - Load DB: `vectorstore = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)`.
  - Create retriever: `retriever = vectorstore.as_retriever(search_kwargs={"k": 3})`.
- [x] **Setup the LLM and Chain**:
  - Initialize LLM: `llm = ChatGoogleGenerativeAI(model="gemini-3.0-pro", temperature=0)`.
  - *Option 1 (Simple)*: Use `RetrievalQA.from_chain_type(...)`.
  - *Option 2 (With Citations)*: Skip `RetrievalQA` and build a custom function (see Phase 5).
- [x] **Create the GET Endpoint (`/ask`)**:
  - Define `@app.get("/ask")`.
  - Accept a `question: str` parameter.

## Phase 5: Implementing Source Citations
*Goal: Prove to the user where the information came from.*
- [x] **Custom Retrieval Logic inside `/ask`**:
  - Step 1: Use `docs = retriever.invoke(question)` (or `.get_relevant_documents`).
  - Step 2: Combine the document content: `context = "\n\n".join([d.page_content for d in docs])`.
  - Step 3: Combine sources: Extract `d.metadata["source"]` from the list of retrieved documents.
  - Step 4: Manually prompt the LLM: `prompt = f"Context: {context}\n\nQuestion: {question}"` and `answer = llm.invoke(prompt)`.
- [x] **Return JSON Response**:
  - Return a dictionary: `{"question": question, "answer": answer, "sources": source_list}`.

## Phase 6: Run and Test
- [ ] **Start the API server**: Run `uvicorn app:app --reload` in your terminal.
- [ ] **Access Swagger UI**: Open your browser to `http://127.0.0.1:8000/docs`.
- [ ] **Execute a Test**: Click "Try it out" on the `/ask` endpoint and type "What are the Basel III capital requirements?". Review the returned answer and citations.

## Phase 7: Resume and Portfolio
- [ ] **Commit Code**: Run `git init`, `git add .`, `git commit`, and push to GitHub.
- [ ] **Update your CV**: Copy the exact bullet points from the README:
  - *"Built a retrieval-augmented generation (RAG) system using LangChain and FAISS..."*
  - *"Implemented document ingestion, chunking, and embedding pipelines..."*
  - *"Developed FastAPI endpoints enabling natural language querying with source citations."*

## Bonus / Advanced Elaborations (Optional)
- **LangGraph Workflow**: Convert the linear chain into a state agent using LangGraph to allow query routing or fallback logic.
- **Query Rewriting**: Before passing the raw user question to the retriever, use an LLM call to rewrite it into a better search query.
- **Reranking**: Retrieve top 10 documents, then use a CrossEncoder model (like `Cohere` or `sentence-transformers`) to re-rank and pick the best top 3 context chunks before passing to the final LLM.
