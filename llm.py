"""
app/core/llm.py
Factory that returns a LangChain chat model based on the configured provider.
"""
from __future__ import annotations

import logging
import os

from langchain_core.language_models import BaseChatModel

from app.config import get_settings

logger = logging.getLogger(__name__)


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Return an instantiated LangChain chat model.
    Directly checks os.getenv to bypass settings parsing validation issues.
    """
    # Force read directly from environment variables to bypass config parsing issues
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    model_name = os.getenv("LLM_MODEL", "llama3-8b-8192").strip()

    if provider == "groq":
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError("Install langchain-groq: pip install langchain-groq")
        
        logger.info("Using Groq model: %s", model_name)
        return ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=groq_key,
        )

    elif provider == "openai":
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
        from langchain_openai import ChatOpenAI
        logger.info("Using OpenAI model: %s", model_name)
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=openai_key,
        )

    elif provider == "anthropic":
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError("Install langchain-anthropic: pip install langchain-anthropic")
        logger.info("Using Anthropic model: %s", model_name)
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            api_key=anthropic_key,
        )

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: '{provider}'. "
                         "Choose 'openai', 'groq', or 'anthropic'.")


def get_embeddings():
    """Return a LangChain embeddings model."""
    settings = get_settings()

    # Safely pull embedding provider configuration
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local").lower().strip()

    if embedding_provider == "openai":
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI embeddings.")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=settings.embedding_model_openai,
            api_key=openai_key,
        )

    else:  # local / sentence-transformers
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model_local = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info("Using local embedding model: %s", model_local)
        return HuggingFaceEmbeddings(
            model_name=model_local,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
