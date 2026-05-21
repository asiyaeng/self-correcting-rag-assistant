"""
tests/test_api.py
API tests using FastAPI's TestClient.

Run with:
    pytest tests/ -v

Note: These tests mock LLM calls so no API key is needed.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ─────────────────────────────────────────────────────────────────────────────
# App fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """
    Create a test client with mocked LLM and vector store.
    This ensures tests run without real API keys.
    """
    # Patch the entire RAG pipeline so tests are fast and free
    with patch("app.api.routes.run_rag_pipeline") as mock_pipeline, \
         patch("app.api.routes.ingest_sources") as mock_ingest, \
         patch("app.api.routes.list_all_documents") as mock_list, \
         patch("app.main.get_vector_store"):

        # Default mock return values
        mock_pipeline.return_value = {
            "answer": "LangGraph is a library for building stateful LLM workflows.",
            "sources": [
                {
                    "source": "corpus/langgraph_guide.md",
                    "title": "LangGraph Guide",
                    "chunk_index": 0,
                    "relevance_score": 0.92,
                    "snippet": "LangGraph is a library for building stateful...",
                }
            ],
            "query_type": "conceptual",
            "rewritten_question": "What is LangGraph and what is it used for?",
            "retries": 0,
            "used_web_search": False,
            "hallucination_check_passed": True,
            "session_id": "test-session",
            "processing_time_ms": 1234.5,
            "error": None,
        }

        mock_ingest.return_value = {
            "documents_ingested": 1,
            "chunks_created": 5,
            "sources": ["https://example.com/docs"],
            "errors": [],
        }

        mock_list.return_value = [
            {
                "doc_id": "abc123",
                "source": "corpus/langchain_overview.md",
                "title": "langchain_overview",
                "chunk_count": 12,
                "ingested_at": "2024-01-01T00:00:00+00:00",
            }
        ]

        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c, mock_pipeline, mock_ingest, mock_list


# ─────────────────────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────────────────────

def test_health(client):
    c, *_ = client
    resp = c.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_root_returns_html(client):
    c, *_ = client
    resp = c.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


# ─────────────────────────────────────────────────────────────────────────────
# POST /query
# ─────────────────────────────────────────────────────────────────────────────

def test_query_basic(client):
    c, mock_pipeline, *_ = client
    resp = c.post("/api/v1/query", json={"question": "What is LangGraph?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert isinstance(data["sources"], list)
    assert "session_id" in data
    assert data["retries"] == 0


def test_query_with_session(client):
    c, mock_pipeline, *_ = client
    resp = c.post(
        "/api/v1/query",
        json={"question": "Explain retrieval", "session_id": "my-session"}
    )
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "my-session"


def test_query_with_custom_top_k(client):
    c, mock_pipeline, *_ = client
    resp = c.post("/api/v1/query", json={"question": "What is RAG?", "top_k": 3})
    assert resp.status_code == 200
    # Verify top_k was passed through
    call_kwargs = mock_pipeline.call_args
    assert call_kwargs.kwargs.get("top_k") == 3 or (
        call_kwargs.args and 3 in call_kwargs.args
    )


def test_query_too_short(client):
    c, *_ = client
    resp = c.post("/api/v1/query", json={"question": "Hi"})
    assert resp.status_code == 422  # Pydantic validation: min_length=3


def test_query_missing_question(client):
    c, *_ = client
    resp = c.post("/api/v1/query", json={})
    assert resp.status_code == 422


def test_query_processing_time_included(client):
    c, *_ = client
    resp = c.post("/api/v1/query", json={"question": "What is LangChain?"})
    assert resp.status_code == 200
    assert resp.json()["processing_time_ms"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# POST /ingest
# ─────────────────────────────────────────────────────────────────────────────

def test_ingest_urls(client):
    c, _, mock_ingest, _ = client
    resp = c.post(
        "/api/v1/ingest",
        json={"urls": ["https://example.com/docs"], "overwrite": False}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["documents_ingested"] >= 1
    assert data["chunks_created"] >= 1


def test_ingest_empty_urls(client):
    c, *_ = client
    resp = c.post("/api/v1/ingest", json={"urls": []})
    assert resp.status_code in (400, 422)


def test_ingest_missing_urls(client):
    c, *_ = client
    resp = c.post("/api/v1/ingest", json={})
    assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# POST /ingest/file
# ─────────────────────────────────────────────────────────────────────────────

def test_ingest_file_md(client):
    c, _, mock_ingest, _ = client
    content = b"# Test Document\n\nThis is test content for ingestion."
    resp = c.post(
        "/api/v1/ingest/file",
        files={"file": ("test.md", content, "text/markdown")},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_ingest_file_unsupported_type(client):
    c, *_ = client
    resp = c.post(
        "/api/v1/ingest/file",
        files={"file": ("test.exe", b"binary data", "application/octet-stream")},
    )
    assert resp.status_code == 415


# ─────────────────────────────────────────────────────────────────────────────
# GET /documents
# ─────────────────────────────────────────────────────────────────────────────

def test_list_documents(client):
    c, _, _, mock_list = client
    resp = c.get("/api/v1/documents")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_documents" in data
    assert "documents" in data
    assert isinstance(data["documents"], list)


def test_list_documents_structure(client):
    c, _, _, mock_list = client
    resp = c.get("/api/v1/documents")
    data = resp.json()
    if data["documents"]:
        doc = data["documents"][0]
        assert "doc_id" in doc
        assert "source" in doc
        assert "chunk_count" in doc


# ─────────────────────────────────────────────────────────────────────────────
# POST /feedback
# ─────────────────────────────────────────────────────────────────────────────

def test_feedback_thumbs_up(client, tmp_path):
    c, *_ = client
    # Patch feedback store path to tmp
    with patch("app.api.routes.settings") as mock_settings:
        mock_settings.feedback_store_path = str(tmp_path / "feedback.json")
        resp = c.post("/api/v1/feedback", json={
            "question": "What is LangGraph?",
            "answer": "LangGraph is a library...",
            "rating": "thumbs_up",
            "comment": "Very helpful!",
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "recorded"
    assert "feedback_id" in data


def test_feedback_thumbs_down(client, tmp_path):
    c, *_ = client
    with patch("app.api.routes.settings") as mock_settings:
        mock_settings.feedback_store_path = str(tmp_path / "feedback.json")
        resp = c.post("/api/v1/feedback", json={
            "question": "How do I set up ChromaDB?",
            "answer": "Install with pip...",
            "rating": "thumbs_down",
            "comment": "Answer was too vague",
        })
    assert resp.status_code == 200


def test_feedback_invalid_rating(client):
    c, *_ = client
    resp = c.post("/api/v1/feedback", json={
        "question": "Test?",
        "answer": "Test.",
        "rating": "neutral",  # not a valid FeedbackRating
    })
    assert resp.status_code == 422


def test_feedback_no_comment_required(client, tmp_path):
    c, *_ = client
    with patch("app.api.routes.settings") as mock_settings:
        mock_settings.feedback_store_path = str(tmp_path / "feedback.json")
        resp = c.post("/api/v1/feedback", json={
            "question": "Quick question",
            "answer": "Quick answer",
            "rating": "thumbs_up",
        })
    assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# Graph state tests (unit)
# ─────────────────────────────────────────────────────────────────────────────

def test_graph_state_defaults():
    from app.models.schemas import GraphState, QueryType
    state = GraphState(original_question="What is RAG?")
    assert state.retries == 0
    assert state.used_web_search is False
    assert state.query_type == QueryType.unknown
    assert state.relevant_docs == []


def test_query_response_model():
    from app.models.schemas import QueryResponse, QueryType
    resp = QueryResponse(
        answer="Test answer",
        sources=[],
        query_type=QueryType.conceptual,
    )
    assert resp.retries == 0
    assert resp.used_web_search is False
