import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "chroma_db"))).expanduser()

COLLECTION_NAME = "youtube_rag"


def get_embeddings():
    """Create the embedding model."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("Set GEMINI_API_KEY in your .env file.")

    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )


def get_vector_store(require_non_empty: bool = True):
    """Connect to the existing Chroma collection."""

    if not CHROMA_DIR.is_dir():
        raise FileNotFoundError(
            f"Chroma database not found: {CHROMA_DIR}"
        )

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