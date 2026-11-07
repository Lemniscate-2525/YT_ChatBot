import uuid
import pytest

from infrastructure import rate_limiter
from infrastructure.rate_limiter import (
    check_rate_limit,
    is_redis_available,
    RateLimitExceededError,
)
from exceptions import RedisUnavailableError


_IS_LIVE_REDIS: bool | None = None


def _check_live_redis() -> bool:
    global _IS_LIVE_REDIS
    if _IS_LIVE_REDIS is None:
        _IS_LIVE_REDIS = is_redis_available()
    return _IS_LIVE_REDIS


@pytest.fixture(autouse=True)
def ensure_working_redis():
    """Ensure a working Redis instance (live or fake) and reset shared keys."""
    original_client = rate_limiter.redis_client
    is_live = _check_live_redis()

    if not is_live:
        import fakeredis
        fake_server = fakeredis.FakeServer()
        fake_client = fakeredis.FakeRedis(server=fake_server, decode_responses=True)
        rate_limiter.set_redis_client(fake_client)

    # Clean keys before test
    try:
        rate_limiter.redis_client.delete("rate_limit:global")
    except Exception:
        pass

    yield

    # Clean keys after test
    try:
        rate_limiter.redis_client.delete("rate_limit:global")
    except Exception:
        pass

    if not is_live:
        rate_limiter.set_redis_client(original_client)


def test_redis_connection():
    assert rate_limiter.redis_client.ping() is True


def test_request_is_allowed():
    user_id = f"test_{uuid.uuid4().hex}"

    result = check_rate_limit(user_id)

    assert result["allowed"] is True
    assert result["user_id"] == user_id
    assert result["remaining_user_requests"] == 9


def test_different_users_have_independent_limits():
    user_a = f"test_{uuid.uuid4().hex}"
    user_b = f"test_{uuid.uuid4().hex}"

    for _ in range(5):
        check_rate_limit(user_a)

    result = check_rate_limit(user_b)

    assert result["remaining_user_requests"] == 9


def test_user_limit_is_enforced():
    user_id = f"test_{uuid.uuid4().hex}"

    # The configured per-user limit is 10 requests per 60 seconds.
    for _ in range(10):
        check_rate_limit(user_id)

    with pytest.raises(RateLimitExceededError, match="user"):
        check_rate_limit(user_id)


def test_global_limit_is_enforced():
    # Use unique users so the per-user limit doesn't trigger first.
    for i in range(100):
        check_rate_limit(f"global_test_{uuid.uuid4().hex}_{i}")

    with pytest.raises(RateLimitExceededError, match="global"):
        check_rate_limit(f"global_test_{uuid.uuid4().hex}")


def test_empty_user_id_is_rejected():
    with pytest.raises(ValueError):
        check_rate_limit("")


def test_redis_unavailable_raises_clean_error(monkeypatch):
    """Verify that connection failure raises RedisUnavailableError instead of crashing."""
    class BrokenScript:
        def __call__(self, *args, **kwargs):
            import redis
            raise redis.exceptions.ConnectionError("Connection refused")

    monkeypatch.setattr(rate_limiter, "RATE_LIMIT_SCRIPT", BrokenScript())
    with pytest.raises(RedisUnavailableError):
        check_rate_limit("user_test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
