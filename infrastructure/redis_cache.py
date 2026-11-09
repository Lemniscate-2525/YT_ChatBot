"""Redis cache primitives; cache failures fail open."""
from __future__ import annotations
import hashlib, json, logging, math, re
from typing import Any
import redis
from config import settings

logger = logging.getLogger(__name__)
_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True,
    socket_connect_timeout=settings.REDIS_TIMEOUT, socket_timeout=settings.REDIS_TIMEOUT)

def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def _key(namespace: str, value: str) -> str:
    return f"{settings.REDIS_CACHE_PREFIX}:{namespace}:{_digest(value)}"

def get_json(key: str) -> Any | None:
    try:
        raw = _client.get(key)
        return json.loads(raw) if raw else None
    except (redis.RedisError, ValueError, TypeError):
        logger.warning("Redis cache read failed; bypassing cache.", exc_info=True)
        return None

def set_json(key: str, value: Any, ttl_seconds: int) -> bool:
    try:
        _client.setex(key, max(1, int(ttl_seconds)), json.dumps(value, ensure_ascii=False))
        return True
    except (redis.RedisError, TypeError, ValueError):
        logger.warning("Redis cache write failed; bypassing cache.", exc_info=True)
        return False

def transcript_key(video_id: str, language: str = "en") -> str:
    return _key("transcript", f"{video_id}:{language.lower()}")

def exact_answer_key(user_id: str, query: str) -> str:
    normalized = re.sub(r"\s+", " ", query).strip().casefold()
    return _key("answer", f"{user_id}:{normalized}")

def semantic_namespace(user_id: str) -> str:
    return f"{settings.REDIS_CACHE_PREFIX}:semantic:{_digest(user_id)}"

def semantic_entry_key(user_id: str, query: str) -> str:
    return f"{semantic_namespace(user_id)}:{_digest(query.strip().casefold())}"

def save_semantic_entry(user_id: str, query: str, embedding: list[float], payload: dict) -> None:
    set_json(semantic_entry_key(user_id, query),
        {"query": query, "embedding": embedding, "payload": payload},
        settings.SEMANTIC_CACHE_TTL_SECONDS)

def find_semantic_match(user_id: str, query_embedding: list[float],
                        threshold: float | None = None) -> dict | None:
    threshold = settings.SEMANTIC_CACHE_THRESHOLD if threshold is None else threshold
    try:
        keys = list(_client.scan_iter(match=f"{semantic_namespace(user_id)}:*", count=100))
        keys = keys[:settings.SEMANTIC_CACHE_MAX_ENTRIES]
        if not keys:
            return None
        pipe = _client.pipeline(transaction=False)
        for key in keys:
            pipe.get(key)
        records = pipe.execute()
    except redis.RedisError:
        logger.warning("Redis semantic lookup failed; bypassing cache.", exc_info=True)
        return None
    qnorm = math.sqrt(sum(float(x) ** 2 for x in query_embedding))
    if not qnorm:
        return None
    best_score, best_payload = -1.0, None
    for raw in records:
        if not raw:
            continue
        try:
            record = json.loads(raw)
            candidate = record.get("embedding", [])
            if len(candidate) != len(query_embedding):
                continue
            cnorm = math.sqrt(sum(float(x) ** 2 for x in candidate))
            if not cnorm:
                continue
            dot = sum(float(a) * float(b) for a, b in zip(query_embedding, candidate))
            score = dot / (qnorm * cnorm)
            if score > best_score:
                best_score, best_payload = score, record.get("payload")
        except (ValueError, TypeError, KeyError):
            continue
    if best_payload is not None and best_score >= threshold:
        logger.info("Semantic answer cache hit (cosine_similarity=%.4f).", best_score)
        return best_payload
    return None


def invalidate_answer_caches() -> None:
    """Invalidate exact and semantic answers after the indexed corpus changes."""
    try:
        patterns = (
            f"{settings.REDIS_CACHE_PREFIX}:answer:*",
            f"{settings.REDIS_CACHE_PREFIX}:semantic:*",
        )
        for pattern in patterns:
            batch = []
            for key in _client.scan_iter(match=pattern, count=200):
                batch.append(key)
                if len(batch) >= 200:
                    _client.delete(*batch)
                    batch.clear()
            if batch:
                _client.delete(*batch)
    except redis.RedisError:
        logger.warning("Could not invalidate answer caches after corpus update.", exc_info=True)
