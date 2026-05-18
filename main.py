"""
app/main.py
FastAPI application entry point.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api.routes import router
from app.config import get_settings

settings = get_settings()

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting %s", settings.app_title)

    # Pre-warm vector store + embeddings on startup
    try:
        from app.core.vector_store import get_vector_store
        get_vector_store()
        logger.info("✅ Vector store ready (%s)", settings.vector_store)
    except Exception as exc:
        logger.warning("⚠️  Vector store init warning (will retry on first request): %s", exc)

    # Auto-ingest corpus if it exists and store is empty
    corpus_dir = os.path.join(os.path.dirname(__file__), "..", "corpus")
    if os.path.isdir(corpus_dir):
        from pathlib import Path
        files = list(Path(corpus_dir).glob("*"))
        if files:
            try:
                from app.core.vector_store import list_all_documents
                existing = list_all_documents()
                real = [d for d in existing if d.get("source") != "__init__"]
                if not real:
                    logger.info("📚 Auto-ingesting corpus (%d files)...", len(files))
                    from app.core.ingestion import ingest_sources
                    result = ingest_sources([str(f) for f in files])
                    logger.info("✅ Corpus ingested: %d docs, %d chunks",
                                result["documents_ingested"], result["chunks_created"])
                else:
                    logger.info("📖 Corpus already indexed (%d documents)", len(real))
            except Exception as exc:
                logger.warning("⚠️  Corpus auto-ingest failed: %s", exc)

    yield
    logger.info("👋 Shutting down")


# ── App instance ──────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_title,
    description=(
        "A Retrieval-Augmented Generation system with a self-corrective LangGraph workflow. "
        "Built for the Express Analytics AI/ML Engineer Intern assignment."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


# ── Root redirect to docs / simple UI ────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>RAG Documentation Assistant</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 700px; margin: 60px auto; padding: 0 20px; }
    h1 { color: #1a1a2e; }
    a { color: #e94560; text-decoration: none; font-weight: 600; }
    a:hover { text-decoration: underline; }
    .links { display: flex; gap: 20px; margin-top: 20px; }
    .card { background: #f5f5f5; border-radius: 10px; padding: 20px; }
  </style>
</head>
<body>
  <h1>🔍 RAG Documentation Assistant</h1>
  <div class="card">
    <p>Self-corrective RAG pipeline powered by LangGraph + FastAPI.</p>
    <div class="links">
      <a href="/docs">📖 Swagger UI</a>
      <a href="/redoc">📚 ReDoc</a>
      <a href="/api/v1/health">💚 Health</a>
    </div>
  </div>
</body>
</html>
"""
