"""Small persistent knowledge graph with entity aliases and provenance.

This implementation is deliberately local-first: NetworkX stores a directed graph
in JSON. It proposes possible entity matches but never silently rewrites source text.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import networkx as nx
from rapidfuzz import fuzz, process

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "data" / "knowledge_graph.json"


class KnowledgeGraph:
    def __init__(self, path: str | Path = DEFAULT_GRAPH_PATH) -> None:
        self.path = Path(path)
        self.graph = nx.MultiDiGraph()
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.graph = nx.node_link_graph(payload, directed=True, multigraph=True)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = nx.node_link_data(self.graph)
        fd, temporary = tempfile.mkstemp(
            prefix=self.path.name, suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def add_entity(
        self,
        canonical_name: str,
        *,
        entity_type: str = "PERSON",
        aliases: list[str] | None = None,
        source_video_id: str | None = None,
        evidence: str | None = None,
        verified: bool = False,
    ) -> str:
        name = canonical_name.strip()
        if not name:
            raise ValueError("canonical_name cannot be empty")
        node_id = f"{entity_type.upper()}::{name.casefold()}"
        existing = self.graph.nodes.get(node_id, {})
        all_aliases = set(existing.get("aliases", []))
        all_aliases.add(name)
        all_aliases.update(a.strip() for a in (aliases or []) if a and a.strip())
        videos = set(existing.get("source_video_ids", []))
        if source_video_id:
            videos.add(source_video_id)
        evidence_items = list(existing.get("evidence", []))
        if evidence:
            evidence_items.append(evidence)
        self.graph.add_node(
            node_id,
            canonical_name=name,
            entity_type=entity_type.upper(),
            aliases=sorted(all_aliases),
            source_video_ids=sorted(videos),
            evidence=evidence_items[-100:],
            verified=bool(existing.get("verified", False) or verified),
        )
        if source_video_id:
            video_node = f"VIDEO::{source_video_id}"
            self.graph.add_node(video_node, entity_type="VIDEO", video_id=source_video_id)
            self.graph.add_edge(
                video_node,
                node_id,
                relation="MENTIONS",
                provenance=evidence or "Transcript entity extraction",
            )
        return node_id

    def add_relation(
        self,
        subject_id: str,
        relation: str,
        object_id: str,
        *,
        source_video_id: str | None = None,
        evidence: str | None = None,
    ) -> None:
        if subject_id not in self.graph or object_id not in self.graph:
            raise ValueError("Both entities must exist before adding a relation")
        self.graph.add_edge(
            subject_id,
            object_id,
            relation=relation.upper(),
            source_video_id=source_video_id,
            evidence=evidence,
        )

    def resolve_entity(
        self,
        mention: str,
        *,
        entity_type: str = "PERSON",
        threshold: int = 90,
        review_threshold: int = 75,
    ) -> dict[str, Any]:
        """Return a candidate decision, never an automatic destructive correction."""
        query = (mention or "").strip()
        if not query:
            return {"status": "not_found", "mention": mention, "candidate": None}

        choices: dict[str, str] = {}
        for node_id, data in self.graph.nodes(data=True):
            if data.get("entity_type") != entity_type.upper():
                continue
            for alias in data.get("aliases", []):
                choices[alias] = node_id
        if not choices:
            return {"status": "not_found", "mention": query, "candidate": None}

        match = process.extractOne(query, list(choices), scorer=fuzz.WRatio)
        if not match:
            return {"status": "not_found", "mention": query, "candidate": None}
        alias, score, _ = match
        candidate_id = choices[alias]
        candidate = self.graph.nodes[candidate_id]
        if score >= threshold:
            status = "candidate_match"
        elif score >= review_threshold:
            status = "needs_review"
        else:
            status = "no_confident_match"
        return {
            "status": status,
            "mention": query,
            "candidate": candidate.get("canonical_name"),
            "candidate_id": candidate_id,
            "matched_alias": alias,
            "similarity_score": int(score),
            "verified": bool(candidate.get("verified", False)),
            "source_video_ids": candidate.get("source_video_ids", []),
            "evidence": candidate.get("evidence", []),
            "auto_correct": False,
        }

    def get_entity(self, node_id: str) -> dict[str, Any] | None:
        if node_id not in self.graph:
            return None
        return {"id": node_id, **dict(self.graph.nodes[node_id])}

    def summary(self) -> dict[str, int]:
        return {"entities": self.graph.number_of_nodes(), "relationships": self.graph.number_of_edges()}


def get_knowledge_graph() -> KnowledgeGraph:
    return KnowledgeGraph()
