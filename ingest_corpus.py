#!/usr/bin/env python3
"""
scripts/ingest_corpus.py

Standalone script to ingest documents into the vector store.
Run this before starting the API server if you want to pre-populate the index.

Usage:
    # Ingest the default corpus/ directory
    python scripts/ingest_corpus.py

    # Ingest specific files
    python scripts/ingest_corpus.py --files path/to/doc.md path/to/api.html

    # Ingest from URLs
    python scripts/ingest_corpus.py --urls https://example.com/docs https://example.com/api

    # Ingest everything and reset the store first
    python scripts/ingest_corpus.py --reset
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make sure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store")
    parser.add_argument("--files", nargs="*", help="File paths to ingest")
    parser.add_argument("--urls", nargs="*", help="URLs to fetch and ingest")
    parser.add_argument("--corpus-dir", default="corpus", help="Directory to ingest (default: corpus/)")
    parser.add_argument("--reset", action="store_true", help="Clear the vector store before ingesting")
    args = parser.parse_args()

    from app.config import get_settings
    settings = get_settings()

    if args.reset:
        import shutil, os
        if settings.vector_store == "chroma" and os.path.exists(settings.chroma_persist_dir):
            shutil.rmtree(settings.chroma_persist_dir)
            logger.info("🗑️  Cleared ChromaDB at: %s", settings.chroma_persist_dir)
        elif settings.vector_store == "faiss" and os.path.exists(settings.faiss_index_path):
            shutil.rmtree(settings.faiss_index_path)
            logger.info("🗑️  Cleared FAISS index at: %s", settings.faiss_index_path)
        # Reset the in-memory singleton
        from app.core import vector_store
        vector_store.reset_store()

    sources: list[str] = []

    # Collect sources from --files
    if args.files:
        sources.extend(args.files)

    # Collect sources from --urls
    if args.urls:
        sources.extend(args.urls)

    # Default: ingest the corpus/ directory
    if not sources:
        corpus_dir = Path(args.corpus_dir)
        if not corpus_dir.exists():
            logger.error("Corpus directory not found: %s", corpus_dir)
            sys.exit(1)
        exts = {".md", ".txt", ".html", ".htm", ".pdf"}
        sources = [str(f) for f in corpus_dir.iterdir() if f.suffix.lower() in exts]
        if not sources:
            logger.warning("No documents found in %s", corpus_dir)
            sys.exit(0)

    logger.info("📚 Ingesting %d source(s)...", len(sources))
    for s in sources:
        logger.info("  → %s", s)

    from app.core.ingestion import ingest_sources
    result = ingest_sources(sources)

    logger.info("─" * 50)
    logger.info("✅ Ingestion complete!")
    logger.info("   Documents ingested : %d", result["documents_ingested"])
    logger.info("   Chunks created     : %d", result["chunks_created"])
    logger.info("   Sources            : %s", result["sources"])
    if result["errors"]:
        logger.warning("   Errors             : %s", result["errors"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
