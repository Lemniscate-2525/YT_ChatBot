"""Authenticated FastAPI service for YouTube RAG."""
from __future__ import annotations

import hmac
import logging
import os
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field, HttpUrl

from exceptions import (
    GuardrailViolationError,
    InvalidQueryError,
    PromptInjectionError,
    RateLimitExceededError,
)
from ingestion.ingestion import ingest_video
from knowledge_graph.entity_verification import verify_entity_mention
from knowledge_graph.graph_store import get_knowledge_graph
from rag import ask_question

logger = logging.getLogger(__name__)
app = FastAPI(
    title="YouTube RAG API",
    version="1.1.0",
    docs_url="/docs" if os.getenv("ENABLE_API_DOCS", "false").lower() == "true" else None,
    redoc_url=None,
    openapi_url="/openapi.json" if os.getenv("ENABLE_API_DOCS", "false").lower() == "true" else None,
)

API_KEY_HEADER = "X-API-Key"


def require_api_key(x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER)) -> str:
    """Require a server-side shared secret for all non-health endpoints."""
    expected = os.getenv("RAG_API_KEY", "").strip()
    if not expected:
        logger.error("RAG_API_KEY is not configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured.",
        )
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return "streamlit-service"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=200)


class IngestRequest(BaseModel):
    video_url: HttpUrl
    refresh: bool = False


class EntityVerifyRequest(BaseModel):
    mention: str = Field(min_length=1, max_length=150)
    entity_type: str = Field(default="PERSON", max_length=40)


class EntityRegistration(BaseModel):
    canonical_name: str = Field(min_length=1, max_length=150)
    entity_type: str = Field(default="PERSON", max_length=40)
    aliases: list[str] = Field(default_factory=list, max_length=30)
    evidence: str = Field(min_length=5, max_length=1000)
    verified: bool = False
    source_video_id: str | None = Field(default=None, max_length=20)


@app.get("/health")
def health() -> dict[str, str]:
    """Public liveness endpoint; does not initialize model or database clients."""
    return {"status": "ok", "service": "youtube-rag-api"}


@app.post("/ingest")
def ingest(
    request: IngestRequest,
    _client: str = Depends(require_api_key),
) -> dict[str, Any]:
    try:
        return ingest_video(str(request.video_url), refresh=request.refresh)
    except Exception as exc:
        logger.exception("Ingestion failed")
        # Avoid returning provider, filesystem, or secret-related exception details.
        raise HTTPException(status_code=502, detail="Video ingestion failed.") from exc


@app.post("/ask")
def ask(
    request: AskRequest,
    client_id: str = Depends(require_api_key),
) -> dict[str, Any]:
    try:
        answer, docs = ask_question(request.question, user_id=client_id)
        sources = []
        for doc in docs:
            url = doc.metadata.get("video_url", "")
            timestamp = doc.metadata.get("start_seconds")
            if url and timestamp is not None:
                separator = "&" if "?" in url else "?"
                url = f"{url}{separator}t={int(timestamp)}s"
            sources.append({
                "video_id": doc.metadata.get("video_id"),
                "url": url,
                "start_seconds": timestamp,
                "text": doc.page_content,
            })
        return {"answer": answer, "sources": sources}
    except RateLimitExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except (InvalidQueryError, PromptInjectionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GuardrailViolationError as exc:
        raise HTTPException(status_code=502, detail="Generated answer failed validation.") from exc
    except Exception as exc:
        logger.exception("RAG inference failed")
        raise HTTPException(status_code=502, detail="Inference failed.") from exc


@app.post("/entities/resolve")
def resolve_entity(
    request: EntityVerifyRequest,
    _client: str = Depends(require_api_key),
) -> dict[str, Any]:
    try:
        return verify_entity_mention(request.mention, entity_type=request.entity_type)
    except Exception as exc:
        logger.exception("Entity resolution failed")
        raise HTTPException(status_code=500, detail="Entity resolution failed.") from exc


@app.post("/entities/register")
def register_entity(
    request: EntityRegistration,
    _client: str = Depends(require_api_key),
) -> dict[str, Any]:
    """Register a reviewed or pending canonical entity."""
    try:
        graph = get_knowledge_graph()
        node_id = graph.add_entity(
            request.canonical_name,
            entity_type=request.entity_type,
            aliases=request.aliases,
            source_video_id=request.source_video_id,
            evidence=request.evidence,
            verified=request.verified,
        )
        graph.save()
        return {"registered": True, "entity": graph.get_entity(node_id)}
    except Exception as exc:
        logger.exception("Entity registration failed")
        raise HTTPException(status_code=400, detail="Entity registration failed.") from exc


@app.get("/graph/summary")
def graph_summary(
    _client: str = Depends(require_api_key),
) -> dict[str, int]:
    return get_knowledge_graph().summary()
