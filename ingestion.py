"""
app/core/ingestion.py
Document ingestion pipeline:
  load → split → embed → store
Supports local files (.md, .txt, .html, .pdf) and remote URLs.
"""
from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import get_settings
from app.core.vector_store import add_documents

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking strategy
# ---------------------------------------------------------------------------
# We use RecursiveCharacterTextSplitter with code/markdown-aware separators.
# Chunk size: 512 tokens (~400 words) — large enough to contain a full
# explanation but small enough for precise retrieval.
# Overlap: 64 tokens — preserves sentence context across chunk boundaries.
# ---------------------------------------------------------------------------

def _make_splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "\n## ", "\n### ", "\n#### ",   # Markdown headings first
            "\n\n", "\n",                    # paragraph / line breaks
            ". ", "? ", "! ",               # sentence boundaries
            " ", "",                         # last resort
        ],
        length_function=len,
        is_separator_regex=False,
    )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _doc_id(source: str) -> str:
    return hashlib.md5(source.encode()).hexdigest()[:12]


def _load_text_file(path: str | Path) -> list[Document]:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    suffix = path.suffix.lower()
    if suffix in (".html", ".htm"):
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator="\n")
    return [Document(
        page_content=text,
        metadata={"source": str(path), "title": path.stem},
    )]


def _load_pdf_file(path: str | Path) -> list[Document]:
    from pypdf import PdfReader
    path = Path(path)
    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": str(path), "title": path.stem, "page": i + 1},
            ))
    return docs


def _load_url(url: str) -> list[Document]:
    logger.info("Fetching URL: %s", url)
    resp = requests.get(url, timeout=30, headers={"User-Agent": "RAGBot/1.0"})
    resp.raise_for_status()
    ct = resp.headers.get("content-type", "")
    if "html" in ct:
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove nav / footer clutter
        for tag in soup.find_all(["nav", "footer", "script", "style", "aside"]):
            tag.decompose()
        # Try to grab <main> or <article> first for cleaner content
        main = soup.find("main") or soup.find("article") or soup.body
        text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")
        title = soup.title.string.strip() if soup.title else url
    else:
        text = resp.text
        title = url.split("/")[-1] or url

    return [Document(
        page_content=text,
        metadata={"source": url, "title": title},
    )]


def load_source(source: str) -> list[Document]:
    """
    Load documents from a file path or URL.
    Returns a list of raw (un-chunked) Documents.
    """
    if source.startswith("http://") or source.startswith("https://"):
        return _load_url(source)
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {source}")
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _load_pdf_file(path)
    return _load_text_file(path)


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

def ingest_sources(sources: list[str]) -> dict:
    """
    Full pipeline: load → split → enrich metadata → store.
    Returns a summary dict.
    """
    settings = get_settings()
    splitter = _make_splitter()
    now = datetime.now(timezone.utc).isoformat()

    all_chunks: list[Document] = []
    ingested_sources: list[str] = []
    errors: list[str] = []

    for source in sources:
        try:
            raw_docs = load_source(source)
            chunks = splitter.split_documents(raw_docs)
            doc_id = _doc_id(source)
            title = raw_docs[0].metadata.get("title", "") if raw_docs else ""
            total = len(chunks)

            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "doc_id": doc_id,
                    "source": source,
                    "title": title,
                    "chunk_index": i,
                    "total_chunks": total,
                    "ingested_at": now,
                })
            all_chunks.extend(chunks)
            ingested_sources.append(source)
            logger.info("Loaded '%s' → %d chunks", source, total)

        except Exception as exc:
            logger.error("Failed to ingest '%s': %s", source, exc)
            errors.append(f"{source}: {exc}")

    if all_chunks:
        add_documents(all_chunks)

    return {
        "documents_ingested": len(ingested_sources),
        "chunks_created": len(all_chunks),
        "sources": ingested_sources,
        "errors": errors,
    }
