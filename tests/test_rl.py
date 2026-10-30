import uuid

import pytest

from infrastructure.rate_limiter import (
    redis_client,
    check_rate_limit,
    RateLimitExceededError,
)


@pytest.fixture(autouse=True)
def clean_test_keys():
    """Reset shared test counters before and after each test."""
    redis_client.delete("rate_limit:global")

    yield

    redis_client.delete("rate_limit:global")


def test_redis_connection():
    assert redis_client.ping() is True


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])