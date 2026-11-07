from __future__ import annotations

import logging
import time
import uuid

import redis

from config import settings
from exceptions import RateLimitExceededError, RedisUnavailableError

logger = logging.getLogger(__name__)

REDIS_URL = settings.REDIS_URL
GLOBAL_LIMIT = settings.GLOBAL_LIMIT
USER_LIMIT = settings.USER_LIMIT
WINDOW_SECONDS = settings.RATE_LIMIT_WINDOW

redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=settings.REDIS_TIMEOUT,
    socket_timeout=settings.REDIS_TIMEOUT,
)

LUA_SCRIPT_TEXT = """
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local global_limit = tonumber(ARGV[3])
local user_limit = tonumber(ARGV[4])
local request_id = ARGV[5]

local global_key = KEYS[1]
local user_key = KEYS[2]

local cutoff = now - window

redis.call("ZREMRANGEBYSCORE", global_key, "-inf", cutoff)
redis.call("ZREMRANGEBYSCORE", user_key, "-inf", cutoff)

local global_count = redis.call("ZCARD", global_key)
local user_count = redis.call("ZCARD", user_key)

if global_count >= global_limit then
    return {0, global_limit - global_count, "global"}
end

if user_count >= user_limit then
    return {0, user_limit - user_count, "user"}
end

redis.call("ZADD", global_key, now, request_id)
redis.call("ZADD", user_key, now, request_id)

redis.call("EXPIRE", global_key, window)
redis.call("EXPIRE", user_key, window)

return {1, user_limit - user_count - 1, "allowed"}
"""

RATE_LIMIT_SCRIPT = redis_client.register_script(LUA_SCRIPT_TEXT)


def set_redis_client(client: redis.Redis) -> None:
    """Allow injecting a mock/fake Redis client for testing or custom connections."""
    global redis_client, RATE_LIMIT_SCRIPT
    redis_client = client
    RATE_LIMIT_SCRIPT = client.register_script(LUA_SCRIPT_TEXT)


def is_redis_available() -> bool:
    """Check whether the Redis instance is reachable."""
    try:
        return bool(redis_client.ping())
    except Exception:
        return False


def check_rate_limit(user_id: str) -> dict:
    """Atomically enforce global and per-user sliding-window limits."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string.")

    now = time.time()
    request_id = uuid.uuid4().hex

    try:
        result = RATE_LIMIT_SCRIPT(
            keys=[
                "rate_limit:global",
                f"rate_limit:user:{user_id}",
            ],
            args=[
                now,
                WINDOW_SECONDS,
                GLOBAL_LIMIT,
                USER_LIMIT,
                request_id,
            ],
        )
    except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as exc:
        logger.error("Rate limiter Redis failure: %s", exc)
        raise RedisUnavailableError("Rate limiting service is temporarily unavailable.") from exc

    allowed = bool(int(result[0]))
    remaining = int(result[1])
    scope = result[2]

    if not allowed:
        raise RateLimitExceededError(
            f"Rate limit exceeded ({scope}). Try again later."
        )

    return {
        "allowed": True,
        "user_id": user_id,
        "remaining_user_requests": remaining,
        "window_seconds": WINDOW_SECONDS,
    }
