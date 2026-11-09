import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv()


class Settings:
    PROJECT_ROOT: Path = PROJECT_ROOT

    # Storage paths
    CHROMA_DIR: Path = Path(
        os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "chroma_db"))
    ).expanduser()
    KNOWLEDGE_GRAPH_PATH: Path = Path(
        os.getenv("KNOWLEDGE_GRAPH_PATH", str(PROJECT_ROOT / "data" / "knowledge_graph.json"))
    ).expanduser()

    # Redis & rate limiting
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_TIMEOUT: float = float(os.getenv("REDIS_TIMEOUT", "2.0"))
    GLOBAL_LIMIT: int = int(os.getenv("GLOBAL_LIMIT", "100"))
    USER_LIMIT: int = int(os.getenv("USER_LIMIT", "10"))
    RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

    # Transcript acquisition
    WEBSHARE_PROXY_USERNAME: str = os.getenv("WEBSHARE_PROXY_USERNAME", "")
    WEBSHARE_PROXY_PASSWORD: str = os.getenv("WEBSHARE_PROXY_PASSWORD", "")
    PROXY_URL: str = os.getenv("PROXY_URL", "")
    TRANSCRIPT_BLOCK_COOLDOWN_SECONDS: int = int(
        os.getenv("TRANSCRIPT_BLOCK_COOLDOWN_SECONDS", "900")
    )

    # Security & API
    RAG_API_KEY: str = os.getenv("RAG_API_KEY", "")
    ENABLE_API_DOCS: bool = os.getenv("ENABLE_API_DOCS", "false").lower() in ("true", "1", "yes")
    CORS_ORIGINS: list[str] = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]

    # Providers
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

    # Input/output validation
    MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "200"))
    MAX_ANSWER_LENGTH: int = int(os.getenv("MAX_ANSWER_LENGTH", "3000"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


settings = Settings()
