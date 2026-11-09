import logging
import random
import re
import time
from threading import Lock
from urllib.parse import parse_qs, urlparse

from requests.exceptions import (
    ConnectTimeout,
    ConnectionError as RequestsConnectionError,
    ProxyError,
    ReadTimeout,
    RetryError,
)

logger = logging.getLogger(__name__)

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    YouTubeRequestFailed,
)


class TranscriptError(Exception):
    """Raised when a YouTube transcript cannot be retrieved."""


def extract_video_id(video_url: str) -> str:
    """Validate a YouTube URL and extract its video ID."""
    if not isinstance(video_url, str) or not video_url.strip():
        raise ValueError("A YouTube URL is required.")

    parsed = urlparse(video_url.strip())
    hostname = (parsed.hostname or "").lower()

    if hostname in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
        elif re.fullmatch(r"/(?:shorts|embed|live)/[^/]+", parsed.path):
            video_id = parsed.path.split("/")[2]
        else:
            raise ValueError("Unsupported YouTube URL format.")
    elif hostname == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    else:
        raise ValueError("The URL must belong to YouTube.")

    if not video_id or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("The URL does not contain a valid YouTube video ID.")
    return video_id


def _exception_chain(exc: Exception):
    """Yield an exception and its chained causes/contexts without looping."""
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _http_status_code(exc: Exception) -> int | None:
    """Extract an HTTP status from requests/library exception chains."""
    for item in _exception_chain(exc):
        response = getattr(item, "response", None)
        status = getattr(response, "status_code", None) or getattr(response, "status", None)
        if status:
            try:
                return int(status)
            except (TypeError, ValueError):
                pass
        text = " ".join(
            str(value) for value in (
                item, getattr(item, "reason", ""), getattr(item, "args", "")
            )
        )
        match = re.search(r"\b([45]\d{2})\b", text)
        if match:
            return int(match.group(1))
    return None


def _is_blocked(exc: Exception) -> bool:
    """Identify rate limits and explicit YouTube/IP blocks."""
    if any(isinstance(item, (IpBlocked, RequestBlocked)) for item in _exception_chain(exc)):
        return True
    return _http_status_code(exc) == 429


def _should_retry(exc: Exception) -> bool:
    """Retry only transient transport/server failures, never rate limits or proxy errors."""
    if _is_blocked(exc):
        return False

    if any(isinstance(item, ProxyError) for item in _exception_chain(exc)):
        return False

    status = _http_status_code(exc)
    if status is not None:
        return status in {408, 425, 500, 502, 503, 504}

    if isinstance(exc, (RetryError,)):
        return False

    return isinstance(exc, (ConnectTimeout, ReadTimeout, TimeoutError, RequestsConnectionError, ConnectionError))


def _wait_before_retry(attempt: int) -> None:
    """Bounded exponential backoff with jitter for genuinely transient errors."""
    delay = min(1.0 * (2 ** (attempt - 1)), 8.0) + random.uniform(0, 0.5)
    time.sleep(delay)


class YouTubeTranscriptClient:
    """Retrieve timestamped YouTube captions with bounded retries and a block cooldown."""

    # Shared by client instances in this Python process; no requests are sent while open.
    _blocked_until = 0.0
    _block_reason = ""
    _circuit_lock = Lock()

    def __init__(
        self,
        enable_audio_fallback: bool = False,
        proxy_username: str | None = None,
        proxy_password: str | None = None,
        proxy_url: str | None = None,
        block_cooldown_seconds: int | None = None,
    ):
        from config import settings

        # Kept as an explicit opt-in for compatibility; production defaults to disabled.
        self.enable_audio_fallback = bool(enable_audio_fallback)
        self.block_cooldown_seconds = (
            block_cooldown_seconds
            if block_cooldown_seconds is not None
            else settings.TRANSCRIPT_BLOCK_COOLDOWN_SECONDS
        )

        if proxy_url:
            self.proxy_url = proxy_url.strip()
            self.proxy_username = (proxy_username or "").strip()
            self.proxy_password = (proxy_password or "").strip()
        else:
            self.proxy_url = (getattr(settings, "PROXY_URL", "") or "").strip()
            self.proxy_username = (
                proxy_username or getattr(settings, "WEBSHARE_PROXY_USERNAME", "") or ""
            ).strip()
            self.proxy_password = (
                proxy_password or getattr(settings, "WEBSHARE_PROXY_PASSWORD", "") or ""
            ).strip()

        proxy_config = None
        self._effective_proxy_url = None

        # Explicit endpoint takes precedence. Webshare's free shared-proxy list
        # provides individual IP:port endpoints; it is not the rotating-residential
        # gateway expected by WebshareProxyConfig.
        if self.proxy_url:
            try:
                from youtube_transcript_api.proxies import GenericProxyConfig
                proxy_config = GenericProxyConfig(
                    http_url=self.proxy_url,
                    https_url=self.proxy_url,
                )
                self._effective_proxy_url = self.proxy_url
                logger.info("Generic proxy endpoint configured.")
            except Exception:
                logger.exception("Failed to initialize generic proxy configuration.")
        elif self.proxy_username and self.proxy_password:
            try:
                from youtube_transcript_api.proxies import WebshareProxyConfig
                proxy_config = WebshareProxyConfig(
                    proxy_username=self.proxy_username,
                    proxy_password=self.proxy_password,
                )
                req_dict = proxy_config.to_requests_dict()
                self._effective_proxy_url = req_dict.get("https") or req_dict.get("http")
                logger.info("Webshare residential proxy configuration initialized.")
            except Exception:
                logger.exception("Failed to initialize Webshare proxy configuration.")

        self._api = (
            YouTubeTranscriptApi(proxy_config=proxy_config)
            if proxy_config is not None
            else YouTubeTranscriptApi()
        )

    @classmethod
    def _open_circuit(cls, reason: str, cooldown_seconds: int) -> None:
        with cls._circuit_lock:
            cls._blocked_until = max(
                cls._blocked_until, time.monotonic() + max(1, cooldown_seconds)
            )
            cls._block_reason = reason

    @classmethod
    def _circuit_state(cls) -> tuple[float, str]:
        with cls._circuit_lock:
            remaining = cls._blocked_until - time.monotonic()
            if remaining <= 0:
                cls._blocked_until = 0.0
                cls._block_reason = ""
                return 0.0, ""
            return remaining, cls._block_reason

    def fetch_transcript(
        self,
        video_url: str,
        languages: tuple[str, ...] = ("en", "en-US", "en-GB", "en-CA", "en-IN"),
    ) -> list[dict]:
        """Fetch captions; do not hammer YouTube after a 429/IP block."""
        video_id = extract_video_id(video_url)

        remaining, reason = self._circuit_state()
        if remaining > 0:
            raise TranscriptError(
                f"YouTube transcript provider is cooling down after {reason}. "
                f"Retry in approximately {int(remaining) + 1} seconds."
            )

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                transcript = None
                try:
                    transcript = self._api.fetch(video_id, languages=list(languages))
                except NoTranscriptFound as not_found_err:
                    try:
                        track_list = self._api.list(video_id)
                        try:
                            track = track_list.find_transcript(list(languages))
                            transcript = track.fetch()
                        except Exception:
                            for track in track_list:
                                if getattr(track, "is_translatable", False):
                                    transcript = track.translate("en").fetch()
                                    break
                    except Exception:
                        pass
                    if transcript is None:
                        raise not_found_err

                segments = [
                    {
                        "text": snippet.text if hasattr(snippet, "text") else snippet.get("text", ""),
                        "start": float(snippet.start if hasattr(snippet, "start") else snippet.get("start", 0)),
                        "duration": float(snippet.duration if hasattr(snippet, "duration") else snippet.get("duration", 0)),
                    }
                    for snippet in transcript
                ]
                if not segments:
                    raise TranscriptError(f"The transcript for video {video_id} is empty.")

                # A successful response closes a previous process-local cooldown.
                with self._circuit_lock:
                    type(self)._blocked_until = 0.0
                    type(self)._block_reason = ""
                return segments

            except TranscriptError:
                raise
            except Exception as exc:
                if _is_blocked(exc):
                    status = _http_status_code(exc)
                    reason = f"HTTP {status}" if status else type(exc).__name__
                    self._open_circuit(reason, self.block_cooldown_seconds)
                    logger.warning(
                        "YouTube transcript provider circuit opened for %s seconds (%s).",
                        self.block_cooldown_seconds, reason,
                    )
                    raise TranscriptError(
                        f"YouTube rejected transcript retrieval for video {video_id} "
                        f"({reason}). Provider cooldown: {self.block_cooldown_seconds} seconds. "
                        "No further requests will be sent during this cooldown."
                    ) from exc

                if attempt < max_attempts and _should_retry(exc):
                    logger.warning(
                        "Transient transcript transport failure (attempt %s/%s): %s",
                        attempt, max_attempts, type(exc).__name__,
                    )
                    _wait_before_retry(attempt)
                    continue

                raise TranscriptError(
                    f"Could not retrieve transcript for video {video_id} "
                    f"after {attempt} attempt(s): {type(exc).__name__}: {exc}"
                ) from exc

        raise TranscriptError(f"Could not retrieve transcript for video {video_id}.")
