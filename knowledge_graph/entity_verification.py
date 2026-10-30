"""Entity mention detection and conservative graph-backed resolution.

Named-entity extraction is pluggable. This baseline accepts explicit entity hints
from metadata or callers; it does not claim to discover every name in arbitrary text.
"""
from __future__ import annotations

from typing import Any

from knowledge_graph.graph_store import KnowledgeGraph, get_knowledge_graph


def verify_entity_mention(
    mention: str,
    *,
    entity_type: str = "PERSON",
    graph: KnowledgeGraph | None = None,
) -> dict[str, Any]:
    store = graph or get_knowledge_graph()
    return store.resolve_entity(mention, entity_type=entity_type)


def register_video_entities(
    *,
    video_id: str,
    entities: list[dict[str, Any]],
    graph: KnowledgeGraph | None = None,
) -> dict[str, Any]:
    """Register metadata-backed entities for later resolution.

    Each entity expects canonical_name and may include aliases, entity_type,
    evidence, and verified. 'verified' must only be set by trusted metadata
    or a human review—not by fuzzy matching alone.
    """
    store = graph or get_knowledge_graph()
    registered = []
    for item in entities:
        if not item.get("canonical_name"):
            continue
        node_id = store.add_entity(
            item["canonical_name"],
            entity_type=item.get("entity_type", "PERSON"),
            aliases=item.get("aliases", []),
            source_video_id=video_id,
            evidence=item.get("evidence"),
            verified=item.get("verified", False),
        )
        registered.append(node_id)
    store.save()
    return {"registered": registered, "count": len(registered)}
