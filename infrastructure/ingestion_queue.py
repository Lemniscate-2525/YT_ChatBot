"""Redis Streams queue for local transcript workers."""
from __future__ import annotations

import json
import logging
import time
import uuid

import redis
from config import settings

logger = logging.getLogger(__name__)
_client = redis.Redis.from_url(
    settings.REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=settings.REDIS_TIMEOUT,
    socket_timeout=settings.REDIS_TIMEOUT,
)
STREAM = f"{settings.REDIS_CACHE_PREFIX}:ingestion:stream"
GROUP = f"{settings.REDIS_CACHE_PREFIX}:workers"
STATUS_PREFIX = f"{settings.REDIS_CACHE_PREFIX}:ingestion:job:"
MAX_ATTEMPTS = 3


def _status_key(job_id: str) -> str:
    return f"{STATUS_PREFIX}{job_id}"


def create_job(video_url: str, refresh: bool = False) -> dict:
    job_id = uuid.uuid4().hex
    now = str(time.time())
    payload = {"job_id": job_id, "video_url": video_url, "refresh": json.dumps(refresh), "attempt": "1"}
    _client.hset(_status_key(job_id), mapping={
        "job_id": job_id, "status": "queued", "video_url": video_url,
        "created_at": now, "updated_at": now, "message": "Waiting for local transcript worker.",
    })
    _client.expire(_status_key(job_id), 7 * 24 * 3600)
    _client.xadd(STREAM, payload, maxlen=10000, approximate=True)
    return get_job(job_id)


def get_job(job_id: str) -> dict | None:
    data = _client.hgetall(_status_key(job_id))
    return data or None


def set_job_status(job_id: str, status: str, message: str, **extra) -> None:
    mapping = {"status": status, "message": message, "updated_at": str(time.time())}
    mapping.update({k: str(v) for k, v in extra.items() if v is not None})
    _client.hset(_status_key(job_id), mapping=mapping)
    _client.expire(_status_key(job_id), 7 * 24 * 3600)


def ensure_group() -> None:
    try:
        _client.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    except redis.exceptions.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


def read_jobs(consumer: str, block_ms: int = 5000):
    """Recover abandoned deliveries before reading new jobs."""
    ensure_group()
    try:
        reclaimed = _client.xautoclaim(
            STREAM, GROUP, consumer, min_idle_time=300_000, start_id="0-0", count=1
        )
        messages = reclaimed[1] if reclaimed and len(reclaimed) > 1 else []
        if messages:
            return [(STREAM, messages)]
    except redis.exceptions.ResponseError as exc:
        # Older Redis-compatible providers may not implement XAUTOCLAIM.
        logger.warning("Pending-job auto-claim unavailable: %s", exc)
    return _client.xreadgroup(GROUP, consumer, {STREAM: ">"}, count=1, block=block_ms)


def acknowledge(message_id: str) -> None:
    _client.xack(STREAM, GROUP, message_id)


def retry_or_fail(message_id: str, fields: dict, error: str) -> None:
    attempt = int(fields.get("attempt", "1"))
    job_id = fields["job_id"]
    if attempt < MAX_ATTEMPTS:
        fields["attempt"] = str(attempt + 1)
        _client.xadd(STREAM, fields, maxlen=10000, approximate=True)
        set_job_status(job_id, "queued", f"Retry scheduled ({attempt + 1}/{MAX_ATTEMPTS}).")
    else:
        set_job_status(job_id, "failed", error[:1000])
    acknowledge(message_id)


def redis_client():
    return _client
