"""
app/api/routes.py
FastAPI router with all required endpoints:
  POST /query
  POST /ingest
  GET  /documents
  POST /feedback
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.config import get_settings
from app.core.graph import run_rag_pipeline
from app.core.ingestion import ingest_sources
from app.core.vector_store import list_all_documents
from app.models.schemas import (
    DocumentListResponse,
    FeedbackRequest,
    FeedbackResponse,
    IngestResponse,
    IngestURLRequest,
    QueryRequest,
    QueryResponse,
    QueryType,
    SourceReference,
)

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

# In-memory session store  (replace with Redis for production)
_sessions: dict[str, list[dict]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# POST /query
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Submit a question to the RAG pipeline",
    tags=["RAG"],
)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Run the full LangGraph RAG workflow:
    query_analysis → retrieval → document_grading → generation → hallucination_check

    Returns the answer with source citations.
    """
    session_id = request.session_id or str(uuid.uuid4())
    chat_history = _sessions.get(session_id, [])

    result = run_rag_pipeline(
        question=request.question,
        session_id=session_id,
        chat_history=chat_history,
        top_k=request.top_k,
    )

    if result.get("error"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"],
        )

    # Update conversation memory
    chat_history.append({"role": "user", "content": request.question})
    chat_history.append({"role": "assistant", "content": result["answer"]})
    _sessions[session_id] = chat_history[-20:]  # keep last 10 turns

    sources = result.get("sources", [])
    # Ensure sources are SourceReference objects
    if sources and isinstance(sources[0], dict):
        sources = [SourceReference(**s) for s in sources]

    qt = result.get("query_type", QueryType.unknown)
    if isinstance(qt, str):
        try:
            qt = QueryType(qt)
        except ValueError:
            qt = QueryType.unknown

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        query_type=qt,
        rewritten_query=result.get("rewritten_question") or None,
        retries=result.get("retries", 0),
        used_web_search=result.get("used_web_search", False),
        hallucination_check_passed=result.get("hallucination_check_passed"),
        session_id=session_id,
        processing_time_ms=round(result.get("processing_time_ms", 0), 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /ingest  (URLs)
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest documents from URLs or uploaded files",
    tags=["Ingestion"],
)
async def ingest_urls(request: IngestURLRequest) -> IngestResponse:
    """Fetch and ingest documents from a list of URLs."""
    if not request.urls:
        raise HTTPException(status_code=400, detail="No URLs provided")

    result = ingest_sources(request.urls)

    if result["documents_ingested"] == 0:
        raise HTTPException(
            status_code=422,
            detail=f"No documents could be ingested. Errors: {result['errors']}",
        )

    return IngestResponse(
        status="success",
        documents_ingested=result["documents_ingested"],
        chunks_created=result["chunks_created"],
        sources=result["sources"],
        message=f"Ingested {result['documents_ingested']} document(s) into {result['chunks_created']} chunks.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /ingest/file  (file uploads)
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/ingest/file",
    response_model=IngestResponse,
    summary="Upload and ingest a document file",
    tags=["Ingestion"],
)
async def ingest_file(file: UploadFile = File(...)) -> IngestResponse:
    """Upload a file (.txt, .md, .html, .pdf) and ingest it into the vector store."""
    allowed_extensions = {".txt", ".md", ".html", ".htm", ".pdf"}
    ext = Path(file.filename or "").suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed_extensions}",
        )

    # Save to a temp path
    tmp_dir = Path("/tmp/rag_uploads")
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / (str(uuid.uuid4()) + ext)

    try:
        content = await file.read()
        tmp_path.write_bytes(content)
        result = ingest_sources([str(tmp_path)])
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    if result["documents_ingested"] == 0:
        raise HTTPException(status_code=422, detail=f"Ingestion failed: {result['errors']}")

    return IngestResponse(
        status="success",
        documents_ingested=result["documents_ingested"],
        chunks_created=result["chunks_created"],
        sources=[file.filename or str(tmp_path)],
        message=f"File '{file.filename}' ingested into {result['chunks_created']} chunks.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /documents
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all indexed documents",
    tags=["Ingestion"],
)
async def list_documents() -> DocumentListResponse:
    """Return a summary of every document currently indexed in the vector store."""
    docs = list_all_documents()
    total_chunks = sum(d.get("chunk_count", 0) for d in docs)
    from app.models.schemas import DocumentInfo
    return DocumentListResponse(
        total_documents=len(docs),
        total_chunks=total_chunks,
        documents=[
            DocumentInfo(
                doc_id=d.get("doc_id", ""),
                source=d.get("source", ""),
                title=d.get("title", ""),
                chunk_count=d.get("chunk_count", 0),
                ingested_at=d.get("ingested_at", ""),
            )
            for d in docs
            if d.get("source") != "__init__"  # filter bootstrap doc
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /feedback
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Submit thumbs-up / thumbs-down feedback on an answer",
    tags=["Feedback"],
)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Store user feedback (rating + optional comment) for quality monitoring."""
    feedback_id = str(uuid.uuid4())
    entry = {
        "feedback_id": feedback_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": request.question,
        "answer": request.answer,
        "rating": request.rating,
        "comment": request.comment,
        "session_id": request.session_id,
    }

    # Append to JSON Lines file
    fb_path = Path(settings.feedback_store_path)
    with fb_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info("[Feedback] %s → %s", feedback_id, request.rating)
    return FeedbackResponse(
        status="recorded",
        feedback_id=feedback_id,
        message="Thank you for your feedback!",
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health", tags=["Utility"], summary="Health check")
async def health():
    return {"status": "ok", "service": settings.app_title}
