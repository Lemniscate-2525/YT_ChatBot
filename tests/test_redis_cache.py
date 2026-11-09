import fakeredis

from infrastructure import redis_cache


def test_transcript_cache_round_trip(monkeypatch):
    client = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis_cache, "_client", client)
    key = redis_cache.transcript_key("abcdefghijk", "en")
    segments = [{"text": "hello", "start": 1.25, "duration": 2.0}]
    assert redis_cache.set_json(key, segments, 60)
    assert redis_cache.get_json(key) == segments


def test_exact_answer_key_normalizes_whitespace_and_scopes_user():
    a = redis_cache.exact_answer_key("user-a", "  What is RAG? ")
    b = redis_cache.exact_answer_key("user-a", "what   is rag?")
    c = redis_cache.exact_answer_key("user-b", "what is rag?")
    assert a == b
    assert a != c


def test_semantic_cache_hit_is_user_scoped(monkeypatch):
    client = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis_cache, "_client", client)
    payload = {"answer": "cached answer", "sources": []}
    redis_cache.save_semantic_entry("user-a", "What is RAG?", [1.0, 0.0], payload)
    assert redis_cache.find_semantic_match("user-a", [1.0, 0.0], threshold=0.99) == payload
    assert redis_cache.find_semantic_match("user-a", [0.0, 1.0], threshold=0.99) is None
    assert redis_cache.find_semantic_match("user-b", [1.0, 0.0], threshold=0.99) is None


def test_corpus_update_invalidates_exact_and_semantic_entries(monkeypatch):
    client = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis_cache, "_client", client)
    exact = redis_cache.exact_answer_key("user-a", "question")
    redis_cache.set_json(exact, {"answer": "old"}, 60)
    redis_cache.save_semantic_entry("user-a", "question", [1.0, 0.0], {"answer": "old"})
    redis_cache.invalidate_answer_caches()
    assert redis_cache.get_json(exact) is None
    assert redis_cache.find_semantic_match("user-a", [1.0, 0.0], threshold=0.5) is None
