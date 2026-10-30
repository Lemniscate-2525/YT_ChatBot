class RAGApplicationError(Exception):
    """Base exception for application-level errors."""


class InvalidQueryError(RAGApplicationError):
    """Raised when a user query fails validation."""


class PromptInjectionError(RAGApplicationError):
    """Raised when a potentially malicious query is detected."""


class GuardrailViolationError(RAGApplicationError):
    """Raised when a generated answer violates output rules."""


class RateLimitExceededError(RAGApplicationError):
    """Raised when a request exceeds a configured rate limit."""


class RedisUnavailableError(RAGApplicationError):
    """Raised when Redis cannot be reached."""