# RAG Documentation Assistant

> A self-corrective Retrieval-Augmented Generation system powered by **LangGraph** + **FastAPI**  
> Built for the Express Analytics AI/ML Engineer Intern assignment

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Setup Instructions](#setup-instructions)
5. [Running the Application](#running-the-application)
6. [API Reference & Example Requests](#api-reference--example-requests)
7. [Document Corpus](#document-corpus)
8. [Design Decisions & Tradeoffs](#design-decisions--tradeoffs)
9. [Chunking & Embedding Strategy](#chunking--embedding-strategy)
10. [What I Would Improve With More Time](#what-i-would-improve-with-more-time)
11. [Assumptions](#assumptions)
12. [Running Tests](#running-tests)

---

## Project Overview

This system answers natural-language questions about a set of technical documents using a multi-node LangGraph workflow. The pipeline features:

- **Self-corrective retrieval** — retrieved documents are graded for relevance; if none pass, the query is automatically rewritten and retrieval retried
- **Source-grounded generation** — answers are generated strictly from retrieved context with inline citations
- **Hallucination checking** — a post-generation node verifies the answer is supported by the source context
- **Conversation memory** — follow-up questions are resolved using per-session chat history
- **FastAPI serving** — full REST API with Swagger/ReDoc documentation
- **Zero-cost local option** — works entirely offline using sentence-transformers embeddings + Groq free tier

---

## Architecture

### LangGraph Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        LangGraph StateGraph                       │
│                                                                   │
│   START                                                           │
│     │                                                             │
│     ▼                                                             │
│  ┌──────────────────┐                                            │
│  │  query_analysis  │  ← Rewrite query, classify type            │
│  └────────┬─────────┘                                            │
│           │                                                        │
│           ▼                                                        │
│  ┌──────────────────┐                                            │
│  │    retrieval     │  ← Similarity search (ChromaDB/FAISS)      │
│  └────────┬─────────┘                                            │
│           │                                                        │
│           ▼                                                        │
│  ┌──────────────────┐                                            │
│  │document_grading  │  ← LLM grades each chunk: relevant/not     │
│  └────────┬─────────┘                                            │
│           │                                                        │
│    ┌──────┴──────────────────────┐                               │
│    │    conditional routing      │                               │
│    └─────┬──────────┬────────────┘                               │
│          │          │            │                                │
│    relevant    no relevant   no relevant                          │
│          │    + retries left  + web search                        │
│          │          │            │                                │
│          │    increment_retries  │                                │
│          │          │            ▼                                │
│          │          └──►  web_search_fallback                     │
│          │                       │                                │
│          ▼                       ▼                                │
│  ┌──────────────────────────────────┐                            │
│  │          generation              │  ← Grounded answer + citations│
│  └────────────────┬─────────────────┘                            │
│                   │                                               │
│                   ▼                                               │
│  ┌──────────────────────────────────┐                            │
│  │      hallucination_check         │  ← Verify answer vs context│
│  └────────────────┬─────────────────┘                            │
│                   │                                               │
│                  END                                              │
└─────────────────────────────────────────────────────────────────┘
```

### Node Responsibilities

| Node | Responsibility |
|------|---------------|
| `query_analysis` | Rewrite/expand query for better retrieval; classify query type (conceptual / how-to / troubleshooting / api_reference) |
| `retrieval` | Cosine similarity search against vector store; returns top-K chunks with scores |
| `document_grading` | LLM grades each chunk as relevant/irrelevant; filters out noise |
| `increment_retries` | Bumps the retry counter before re-entering the rewrite loop |
| `web_search` | (Bonus) Tavily web search fallback when vector store has nothing relevant |
| `generation` | Generates grounded answer with inline citations using only relevant context |
| `hallucination_check` | (Bonus) Post-generation LLM verification that every claim is supported |

### State Schema

The graph state (`dict`) carries:

```python
{
    "original_question": str,       # user's raw question
    "rewritten_question": str,      # expanded/clarified for retrieval
    "query_type": QueryType,        # conceptual | how_to | troubleshooting | api_reference
    "retrieved_docs": list[dict],   # raw results from vector store
    "relevant_docs": list[dict],    # graded-relevant subset
    "answer": str,                  # final generated answer
    "sources": list[SourceReference],
    "retries": int,                 # retry counter (max = MAX_RETRIES from settings)
    "used_web_search": bool,
    "hallucination_check_passed": bool | None,
    "session_id": str | None,
    "chat_history": list[dict],     # for conversation memory
    "error": str | None,
    "processing_time_ms": float,
}
```

---

## Project Structure

```
rag-assistant/
├── app/
│   ├── main.py                 # FastAPI app + lifespan
│   ├── config.py               # Pydantic settings from .env
│   ├── api/
│   │   └── routes.py           # All API endpoints
│   ├── core/
│   │   ├── graph.py            # LangGraph StateGraph (all nodes + routing)
│   │   ├── ingestion.py        # Document loading, chunking, embedding
│   │   ├── llm.py              # LLM + embeddings factory
│   │   └── vector_store.py     # ChromaDB / FAISS abstraction
│   └── models/
│       └── schemas.py          # Pydantic request/response models
├── corpus/                     # Technical documentation corpus
│   ├── langchain_overview.md
│   ├── langgraph_guide.md
│   ├── fastapi_reference.md
│   ├── rag_concepts.md
│   └── pydantic_guide.md
├── scripts/
│   └── ingest_corpus.py        # Standalone ingestion CLI
├── tests/
│   └── test_api.py             # pytest test suite
├── .env.example                # Environment variable template
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- An API key for at least one LLM provider (OpenAI, Groq, or Anthropic)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-assistant.git
cd rag-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in **at least one** of these:

```dotenv
# Option A: OpenAI (recommended — gpt-4o-mini is cheap)
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# Option B: Groq (free tier, very fast)
# GROQ_API_KEY=gsk_...
# LLM_PROVIDER=groq
# LLM_MODEL=llama3-8b-8192

# Embeddings — use "local" for zero cost (sentence-transformers)
EMBEDDING_PROVIDER=local
```

> **No API key available?** Set `EMBEDDING_PROVIDER=local` and `LLM_PROVIDER=groq` with a free Groq key (https://console.groq.com). Groq's free tier is generous enough for this assignment.

### 5. Ingest the corpus

```bash
python scripts/ingest_corpus.py
```

This indexes the 5 documents in `corpus/` into ChromaDB. You should see output like:
```
✅ Ingestion complete!
   Documents ingested : 5
   Chunks created     : 87
```

---

## Running the Application

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health check**: http://localhost:8000/api/v1/health

> The application auto-ingests the `corpus/` directory on startup if the vector store is empty.

---

## API Reference & Example Requests

### `POST /api/v1/query` — Ask a question

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does document grading work in CRAG?"}'
```

**Response:**
```json
{
  "answer": "In Corrective RAG (CRAG), document grading is a self-corrective step that evaluates whether retrieved chunks are actually relevant to the question. An LLM grades each document as 'relevant' or 'irrelevant'. If all documents are irrelevant, the query is automatically rewritten and retrieval is retried. [Source: rag_concepts]",
  "sources": [
    {
      "source": "corpus/rag_concepts.md",
      "title": "rag_concepts",
      "chunk_index": 4,
      "relevance_score": 0.91,
      "snippet": "Adds a document grading step after retrieval..."
    }
  ],
  "query_type": "conceptual",
  "rewritten_query": "How does document relevance grading work in Corrective RAG CRAG pipeline?",
  "retries": 0,
  "used_web_search": false,
  "hallucination_check_passed": true,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "processing_time_ms": 2341.5
}
```

**Follow-up question (with session):**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Can you explain the retry mechanism in more detail?",
    "session_id": "550e8400-e29b-41d4-a716-446655440000"
  }'
```

---

### `POST /api/v1/ingest` — Ingest from URLs

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://docs.pydantic.dev/latest/",
      "https://python.langchain.com/docs/introduction/"
    ]
  }'
```

**Response:**
```json
{
  "status": "success",
  "documents_ingested": 2,
  "chunks_created": 34,
  "sources": [
    "https://docs.pydantic.dev/latest/",
    "https://python.langchain.com/docs/introduction/"
  ],
  "message": "Ingested 2 document(s) into 34 chunks."
}
```

---

### `POST /api/v1/ingest/file` — Upload a file

```bash
curl -X POST http://localhost:8000/api/v1/ingest/file \
  -F "file=@my_api_docs.md"
```

---

### `GET /api/v1/documents` — List indexed documents

```bash
curl http://localhost:8000/api/v1/documents
```

**Response:**
```json
{
  "total_documents": 5,
  "total_chunks": 87,
  "documents": [
    {
      "doc_id": "a1b2c3d4e5f6",
      "source": "corpus/langgraph_guide.md",
      "title": "langgraph_guide",
      "chunk_count": 21,
      "ingested_at": "2024-06-15T10:30:00+00:00"
    }
  ]
}
```

---

### `POST /api/v1/feedback` — Submit feedback

```bash
curl -X POST http://localhost:8000/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is LangGraph?",
    "answer": "LangGraph is a library for building stateful LLM workflows...",
    "rating": "thumbs_up",
    "comment": "Clear and accurate answer!"
  }'
```

**Response:**
```json
{
  "status": "recorded",
  "feedback_id": "7f3a9c2e-...",
  "message": "Thank you for your feedback!"
}
```

---

## Document Corpus

The corpus covers the exact stack used in this assignment:

| File | Content | Chunks (approx.) |
|------|---------|-------------------|
| `langchain_overview.md` | LangChain core concepts: chains, LCEL, retrievers, memory, agents | ~18 |
| `langgraph_guide.md` | LangGraph StateGraph, nodes, edges, RAG agent patterns, streaming | ~21 |
| `fastapi_reference.md` | FastAPI path operations, Pydantic integration, middleware, testing | ~20 |
| `rag_concepts.md` | RAG pipeline, CRAG, Self-RAG, Adaptive RAG, evaluation metrics | ~17 |
| `pydantic_guide.md` | Pydantic v2 models, validators, settings management | ~11 |

To ingest additional documents:
```bash
# From local files
python scripts/ingest_corpus.py --files path/to/doc.md

# From URLs
python scripts/ingest_corpus.py --urls https://example.com/docs

# Reset and re-ingest everything
python scripts/ingest_corpus.py --reset
```

---

## Design Decisions & Tradeoffs

### Why LangGraph over a simple chain?
LangGraph's `StateGraph` allows explicit control flow with conditional routing and retry loops — things that are awkward to express in a linear LangChain chain. The state schema (a typed dict flowing between nodes) makes debugging straightforward: you can inspect exactly what each node received and returned.

**Tradeoff:** LangGraph adds complexity. For a simple Q&A without self-correction, a plain `RetrievalQA` chain would be faster to build.

### State as plain dict vs TypedDict vs Pydantic
I chose `dict` for the LangGraph state (with Pydantic models at the API boundary). Pydantic v2 models can't be used directly as LangGraph state types without custom reducers. Using plain dicts inside the graph and converting at the API boundary avoids serialization friction.

**Tradeoff:** Less type safety inside graph nodes. Mitigated by the Pydantic schema (`GraphState`) for documentation purposes.

### ChromaDB as default vector store
ChromaDB runs locally with zero configuration, persists automatically, and integrates cleanly with LangChain. For production or a larger corpus, I would switch to FAISS (faster similarity search) or Pinecone (managed, scalable).

**Tradeoff:** ChromaDB is slower than FAISS for large-scale search but is far easier to set up for this assignment scope.

### Local embeddings as default
`sentence-transformers/all-MiniLM-L6-v2` runs on CPU with no API key. Quality is slightly below OpenAI's `text-embedding-3-small`, but the difference is small for English technical documentation. This makes the system runnable by evaluators who may not have an OpenAI key.

**Tradeoff:** Local embeddings add ~200ms on cold start and ~30ms per query on CPU. Not a concern for a demo but matters at scale.

### Document grading uses an LLM per chunk
This is the correct approach architecturally (matching the CRAG paper) but is expensive — it adds K LLM calls before generation. For low-latency production use, I would batch these calls or use a fine-tuned cross-encoder model for scoring.

**Tradeoff:** Accuracy vs. latency. The grader significantly improves precision at the cost of ~K×100ms extra latency.

### In-memory session store
Sessions are stored in a Python dict (`_sessions` in `routes.py`). This is lost on server restart and doesn't work in multi-process deployments.

**Production alternative:** Replace with Redis using `aioredis` and a TTL-based expiry.

### Feedback stored as JSON Lines
Simple and readable. For production, a proper database (SQLite or PostgreSQL) with a dashboard for reviewing feedback would be better.

---

## Chunking & Embedding Strategy

### Chunking
I use `RecursiveCharacterTextSplitter` with:
- **Chunk size: 512 characters** (~128 tokens) — captures a full explanation without being too noisy
- **Overlap: 64 characters** (~16 tokens) — preserves sentence context across chunk boundaries
- **Separator priority:** Markdown headings first (`## `, `### `), then paragraph breaks, then sentence endings — this keeps semantically coherent sections together

For technical documentation, respecting heading boundaries is crucial because subsections often answer different questions. Splitting in the middle of a code example or parameter list hurts retrieval precision.

### Embeddings
Default: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, runs on CPU)
- Good trade-off between speed and quality for English technical prose
- Free, no API key, works offline
- Normalised embeddings → cosine similarity = dot product (efficient)

Optional: `text-embedding-3-small` (OpenAI) — noticeably better for short/ambiguous queries.

---

## What I Would Improve With More Time

1. **Async graph execution** — Use `async def` for all graph nodes and `await graph.ainvoke()` to reduce latency from serial LLM calls (especially document grading)
2. **Batch document grading** — Send all chunks in a single LLM call with structured output instead of N individual calls
3. **HyDE (Hypothetical Document Embeddings)** — Generate a hypothetical answer and use it for retrieval, which typically improves recall for abstract or conceptual questions
4. **Multi-query retrieval** — Generate 3 query paraphrases, retrieve for each, and deduplicate to improve recall
5. **RAGAS evaluation** — Automated evaluation of faithfulness, answer relevance, and context precision against a held-out question set
6. **Redis session store** — Replace the in-memory dict with Redis for persistence and multi-process support
7. **Streaming endpoint** — `GET /query/stream` using Server-Sent Events for token-by-token response
8. **Streamlit UI** — Interactive chat interface with source preview and feedback buttons
9. **Re-ranking** — Add a cross-encoder re-ranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) after retrieval for better ordering
10. **Document update detection** — Track source file hashes; re-ingest only changed documents

---

## Assumptions

1. **Single-tenant deployment** — No authentication is implemented. For a real deployment, add OAuth2/API key authentication to the FastAPI app.
2. **English-only corpus** — The embedding model and prompts are tuned for English. Multi-language would require a multilingual embedding model.
3. **Static corpus for demo** — The 5 corpus files cover the assignment's stack (LangChain, LangGraph, FastAPI, Pydantic, RAG). A production system would continuously ingest new documentation versions.
4. **LLM grading is binary** — The grader returns relevant/irrelevant. A scored grader (0.0–1.0) would allow soft filtering.
5. **No authentication on /feedback** — In production, feedback should be tied to an authenticated user to prevent spam.

---

## Running Tests

```bash
# Install test dependencies (included in requirements.txt)
pip install pytest httpx

# Run all tests (mocked — no API key needed)
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=app --cov-report=term-missing
```

Tests use `unittest.mock` to patch the LLM pipeline and vector store, so they run instantly without any API credentials.

---

## Environment Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai` / `groq` / `anthropic` |
| `LLM_MODEL` | `gpt-4o-mini` | Model name for the provider |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `GROQ_API_KEY` | — | Groq API key (free tier available) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `EMBEDDING_PROVIDER` | `local` | `local` (sentence-transformers) / `openai` |
| `VECTOR_STORE` | `chroma` | `chroma` / `faiss` |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB persistence directory |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `MAX_RETRIES` | `2` | Max query-rewrite + re-retrieve loops |
| `CHUNK_SIZE` | `512` | Chunk size in characters |
| `CHUNK_OVERLAP` | `64` | Chunk overlap in characters |
| `ENABLE_WEB_SEARCH` | `false` | Enable Tavily web search fallback |
| `TAVILY_API_KEY` | — | Tavily API key (for web search bonus) |

---

*Built by [Asiya Begum] — Express Analytics AI/ML Intern Assignment*
