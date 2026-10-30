"""FastAPI ingestion and inference service for YouTube RAG."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl

from exceptions import (
    GuardrailViolationError,
    InvalidQueryError,
    PromptInjectionError,
    RateLimitExceededError,
)
from ingestion.ingestion import ingest_video
from knowledge_graph.entity_verification import (
    register_video_entities,
    verify_entity_mention,
)
from knowledge_graph.graph_store import get_knowledge_graph
from rag import ask_question

logger = logging.getLogger(__name__)
app = FastAPI(title="YouTube RAG API", version="1.0.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=200)
    # Demo identity only; production identity must come from authentication.
    user_id: str = Field(default="api-dev-user", min_length=1, max_length=100)


class IngestRequest(BaseModel):
    video_url: HttpUrl
    refresh: bool = False


class EntityVerifyRequest(BaseModel):
    mention: str = Field(min_length=1, max_length=150)
    entity_type: str = Field(default="PERSON", max_length=40)


class EntityRegistration(BaseModel):
    canonical_name: str = Field(min_length=1, max_length=150)
    entity_type: str = Field(default="PERSON", max_length=40)
    aliases: list[str] = Field(default_factory=list)
    evidence: str = Field(min_length=5, max_length=1000)
    verified: bool = False
    source_video_id: str | None = Field(default=None, max_length=20)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "youtube-rag-api"}


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, Any]:
    try:
        return ingest_video(str(request.video_url), refresh=request.refresh)
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    try:
        answer, docs = ask_question(request.question, user_id=request.user_id)
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
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("RAG inference failed")
        raise HTTPException(status_code=502, detail=f"Inference failed: {type(exc).__name__}") from exc


@app.post("/entities/resolve")
def resolve_entity(request: EntityVerifyRequest) -> dict[str, Any]:
    try:
        return verify_entity_mention(request.mention, entity_type=request.entity_type)
    except Exception as exc:
        logger.exception("Entity resolution failed")
        raise HTTPException(status_code=500, detail="Entity resolution failed") from exc


@app.post("/entities/register")
def register_entity(request: EntityRegistration) -> dict[str, Any]:
    """Register a human-reviewed or independently sourced canonical entity."""
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
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/graph/summary")
def graph_summary() -> dict[str, int]:
    return get_knowledge_graph().summary()
