import random
import re
import time
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    IpBlocked,
    RequestBlocked,
    YouTubeRequestFailed,
)

class TranscriptError(Exception):
    """Raised when a YouTube transcript cannot be retrieved."""


def extract_video_id(video_url: str) -> str:
    """Validate a YouTube URL and extract its video ID."""

    if not isinstance(video_url, str) or not video_url.strip(): # Input Validation. Strips Whitespaces
        raise ValueError("A YouTube URL is required.")  # Condition is true for either case, if stripped string is empty or if input ain't a string.

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


def _should_retry(exc: Exception) -> bool:
    """Determine whether a transcript-fetching failure is retryable."""

    # Blocking errors should not trigger repeated requests.
    if isinstance(exc, (IpBlocked, RequestBlocked)):
        return False

    if isinstance(exc, YouTubeRequestFailed):
        # Check the original exception when the library exposes it.
        cause = exc.__cause__ or exc.__context__

        if cause is not None and cause is not exc:
            return _should_retry(cause)

        # The library may retain the HTTP status only in its reason string.
        reason = getattr(exc, "reason", "")
        match = re.search(r"\b([45]\d{2})\b", str(reason))

        if match:
            status_code = int(match.group(1))
            return status_code >= 500 or status_code in {408, 425}

        # Unknown HTTP failures are not retried blindly.
        return False

    # Retry temporary network failures.
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    return False


def _wait_before_retry(attempt: int) -> None:
    """Wait using exponential backoff with jitter."""

    base_delay = 1.0
    max_delay = 8.0

    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
    delay += random.uniform(0, 0.5)

    time.sleep(delay)


class YouTubeTranscriptClient:
    """Fetch timestamped transcripts from YouTube."""

    def __init__(self):
        self._api = YouTubeTranscriptApi()

    def fetch_transcript(
        self,
        video_url: str,
        languages: tuple[str, ...] = ("en",),
    ) -> list[dict]:
        """Fetch timestamped transcript snippets with bounded retries."""

        video_id = extract_video_id(video_url)
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            try:
                transcript = self._api.fetch(
                    video_id,
                    languages=list(languages),
                )

                segments = [
                    {
                        "text": snippet.text,
                        "start": float(snippet.start),
                        "duration": float(snippet.duration),
                    }
                    for snippet in transcript
                ]

                if not segments:
                    raise TranscriptError(
                        f"The transcript for video {video_id} is empty."
                    )

                return segments

            except TranscriptError:
                # Empty transcripts and other explicit client errors
                # should propagate without being wrapped again.
                raise

            except Exception as exc:   # If any other exception occurs and retrying is valid as we still have attempts left; and it's valid to retry, then we wait and retry. If not, we stop curr iteration and print TranscriptError. 
                if attempt < max_attempts and _should_retry(exc):
                    _wait_before_retry(attempt)
                    continue  # We need continue here because after waiting we wanna go back to the top of for loop and execute the try block again; if cont was not here; well; then it'd go to the next line and print Transcript Error.

                raise TranscriptError(
                    f"Could not retrieve the transcript for video {video_id} "
                    f"after {attempt} attempt(s). "
                    f"Underlying error: {type(exc).__name__}: {exc}"
                ) from exc

        # Defensive fallback; the loop normally returns or raises.
        raise TranscriptError(
            f"Could not retrieve the transcript for video {video_id}."
        )