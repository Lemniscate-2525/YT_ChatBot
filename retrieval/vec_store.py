import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import settings

PROJECT_ROOT = settings.PROJECT_ROOT
CHROMA_DIR = settings.CHROMA_DIR
COLLECTION_NAME = "youtube_rag"


def get_embeddings():
    """Create the embedding model."""

    api_key = settings.GEMINI_API_KEY

    if not api_key:
        raise ValueError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment.")

    return GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=api_key,
    )


def get_vector_store(require_non_empty: bool = True):
    """Connect to the Chroma collection, creating its directory if needed."""

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings()

    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    if require_non_empty and vector_store._collection.count() == 0:
        raise RuntimeError(
            "The Chroma collection is empty. "
            "Ingestion is required before asking questions."
        )

    return vector_store
