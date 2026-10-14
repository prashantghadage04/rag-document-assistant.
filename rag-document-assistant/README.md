# RAG Document Assistant

A question-answering chatbot over your own documents using **Retrieval-Augmented Generation (RAG)**, with a Python/FastAPI backend, a vector store, and a React frontend. Works out of the box with **no API key**; add one to get LLM-written answers.

## What it demonstrates
| Skill | Where |
|---|---|
| RAG pipeline (load → chunk → embed → retrieve → generate) | `backend/rag.py` |
| Vector database concepts (embeddings, cosine similarity, top-k search) | `VectorStore` class |
| LLM integration (Claude or OpenAI, with graceful fallback) | `generate_answer()` |
| Prompt engineering (grounded answers, "say you don't know") | `generate_answer()` |
| Backend REST API (Python, FastAPI, Pydantic, CORS) | `backend/app.py` |
| Frontend (React hooks, fetch, chat UI) | `frontend/index.html` |
| AI agents foundation (retrieval tool + reasoning step) | easy next step, see below |

## How RAG works here
```
Question ─► embed ─► search vector store ─► top-3 chunks ─┐
                                                            ├─► LLM prompt ─► grounded answer + sources
Documents ─► chunk ─► embed ─► vector store ───────────────┘
```

## Run it
```bash
pip install -r requirements.txt
cd backend
uvicorn app:app --reload --port 8000
```
Open **http://localhost:8000** and ask questions like:
- *How many sick leave days do I get?*
- *What is the password policy?*
- *How much internet reimbursement can I claim?*

API docs are auto-generated at http://localhost:8000/docs.

### Enable LLM answers (optional)
```bash
pip install anthropic          # or: pip install openai
export ANTHROPIC_API_KEY=your_key      # or OPENAI_API_KEY
```
Without a key the app runs in retrieval-only mode and returns the best-matching passage.

## Use your own documents
Drop `.txt` or `.md` files into `backend/docs/` and restart the server.

## Ideas to extend
- Replace TF-IDF with real embeddings (`sentence-transformers`) and a vector DB (ChromaDB, FAISS, Pinecone)
- Add PDF loading with `pypdf`
- Turn it into an agent: give the LLM tools (search docs, calculate, query SQL) and let it decide which to call
- Add conversation memory and streaming responses
- Dockerise and deploy on AWS / GCP / Azure
