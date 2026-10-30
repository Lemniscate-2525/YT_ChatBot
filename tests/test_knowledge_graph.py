from knowledge_graph.graph_store import KnowledgeGraph


def test_add_entity_persists_alias_and_provenance(tmp_path):
    path = tmp_path / "graph.json"
    graph = KnowledgeGraph(path)
    node_id = graph.add_entity(
        "Alex Hormozi",
        aliases=["Alex Hormozy"],
        source_video_id="video1234567",
        evidence="Reviewed video metadata.",
        verified=True,
    )
    graph.save()
    loaded = KnowledgeGraph(path)
    entity = loaded.get_entity(node_id)
    assert entity["canonical_name"] == "Alex Hormozi"
    assert "Alex Hormozy" in entity["aliases"]
    assert entity["verified"] is True
    assert entity["source_video_ids"] == ["video1234567"]


def test_entity_resolution_proposes_but_never_auto_corrects(tmp_path):
    graph = KnowledgeGraph(tmp_path / "graph.json")
    graph.add_entity("Alex Hormozi", aliases=["Alex Hormozy"], verified=True)
    result = graph.resolve_entity("Alex Hormozy")
    assert result["status"] == "candidate_match"
    assert result["candidate"] == "Alex Hormozi"
    assert result["auto_correct"] is False


def test_ambiguous_name_is_not_silently_corrected(tmp_path):
    graph = KnowledgeGraph(tmp_path / "graph.json")
    graph.add_entity("Alex Hormozi", aliases=["Alex Hormozy"])
    result = graph.resolve_entity("Alex Shomzi")
    assert result["status"] in {"needs_review", "no_confident_match"}
    assert result["auto_correct"] is False


def test_unknown_entity_does_not_invent_candidate(tmp_path):
    graph = KnowledgeGraph(tmp_path / "graph.json")
    result = graph.resolve_entity("Completely Unknown Person")
    assert result["status"] == "not_found"
    assert result["candidate"] is None
