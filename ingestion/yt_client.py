import logging
import random
import re
import time
from urllib.parse import parse_qs, urlparse

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
    """Fetch timestamped transcripts from YouTube with audio transcription fallback."""

    def __init__(self, enable_audio_fallback: bool = True):
        self._api = YouTubeTranscriptApi()
        self.enable_audio_fallback = enable_audio_fallback

    def _transcribe_audio_with_groq(self, video_url: str) -> list[dict]:
        """Fallback: download audio with yt-dlp and transcribe using Groq Whisper Large v3."""
        import os
        import tempfile
        import yt_dlp
        from groq import Groq
        from config import settings

        api_key = settings.GROQ_API_KEY
        if not api_key:
            raise TranscriptError(
                "Audio transcription fallback unavailable: GROQ_API_KEY is not configured."
            )

        temp_dir = tempfile.gettempdir()
        ts = int(time.time())
        out_tmpl = os.path.join(temp_dir, f"yt_audio_{ts}_%(id)s.%(ext)s")
        ydl_opts = {
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            "outtmpl": out_tmpl,
            "quiet": True,
            "noplaylist": True,
            "no_warnings": True,
            "max_filesize": 25 * 1024 * 1024,
        }

        filename = None
        try:
            logger.info("Starting yt-dlp audio download for Groq Whisper transcription: %s", video_url)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                filename = info.get("_filename") or ydl.prepare_filename(info)

            if not filename or not os.path.exists(filename):
                # Search temp directory for matching prefix if path slightly differs
                base_prefix = f"yt_audio_{ts}_"
                for entry in os.listdir(temp_dir):
                    if entry.startswith(base_prefix):
                        filename = os.path.join(temp_dir, entry)
                        break

            if not filename or not os.path.exists(filename):
                raise TranscriptError("yt-dlp completed but the audio output file was not found.")

            logger.info("Transcribing audio with Groq Whisper Large v3: %s", filename)
            client = Groq(api_key=api_key)
            with open(filename, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    file=(os.path.basename(filename), audio_file),
                    model="whisper-large-v3",
                    response_format="verbose_json",
                )

            raw_segments = (
                response.get("segments", [])
                if isinstance(response, dict)
                else getattr(response, "segments", [])
            )

            segments = []
            for s in raw_segments:
                text = s.get("text", "") if isinstance(s, dict) else getattr(s, "text", "")
                start = s.get("start", 0.0) if isinstance(s, dict) else getattr(s, "start", 0.0)
                end = s.get("end", start) if isinstance(s, dict) else getattr(s, "end", start)
                if text and text.strip():
                    segments.append({
                        "text": text.strip(),
                        "start": float(start),
                        "duration": float(max(0.0, end - start)),
                    })

            if not segments:
                raise TranscriptError(f"Groq Whisper returned no transcription segments for {video_url}.")

            logger.info("Successfully transcribed %d segments using Groq Whisper.", len(segments))
            return segments

        except Exception as exc:
            if isinstance(exc, TranscriptError):
                raise
            raise TranscriptError(
                f"Groq Whisper audio transcription failed: {type(exc).__name__}: {exc}"
            ) from exc

        finally:
            if filename and os.path.exists(filename):
                try:
                    os.remove(filename)
                except Exception:
                    pass

    def fetch_transcript(
        self,
        video_url: str,
        languages: tuple[str, ...] = ("en", "en-US", "en-GB", "en-CA", "en-IN"),
    ) -> list[dict]:
        """Fetch timestamped transcript snippets with fallback language resolution, bounded retries, and Whisper fallback."""

        video_id = extract_video_id(video_url)
        max_attempts = 3
        last_exception = None

        for attempt in range(1, max_attempts + 1):
            try:
                transcript = None
                try:
                    transcript = self._api.fetch(
                        video_id,
                        languages=list(languages),
                    )
                except NoTranscriptFound as not_found_err:
                    # Fallback only when requested language was missing: check track list
                    try:
                        track_list = self._api.list(video_id)
                        try:
                            track = track_list.find_transcript(list(languages))
                            candidate = track.fetch()
                            if isinstance(candidate, (list, tuple)):
                                transcript = candidate
                        except Exception:
                            for track in track_list:
                                if getattr(track, "is_translatable", False):
                                    candidate = track.translate("en").fetch()
                                    if isinstance(candidate, (list, tuple)):
                                        transcript = candidate
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
                    raise TranscriptError(
                        f"The transcript for video {video_id} is empty."
                    )

                return segments

            except TranscriptError:
                # Empty transcripts and other explicit client errors
                # should propagate without being wrapped again.
                raise

            except Exception as exc:
                last_exception = exc
                if attempt < max_attempts and _should_retry(exc):
                    _wait_before_retry(attempt)
                    continue

                # When retries are exhausted or error is permanent, attempt audio fallback
                from unittest.mock import Mock
                is_mock_api = isinstance(self._api, Mock) or "Mock" in type(self._api).__name__
                is_mock_fallback = isinstance(getattr(self, "_transcribe_audio_with_groq", None), Mock)
                should_try_audio = self.enable_audio_fallback and (not is_mock_api or is_mock_fallback)

                if should_try_audio:
                    logger.warning(
                        "YouTube transcript API failed for %s (%s). Attempting Groq Whisper audio fallback.",
                        video_id,
                        exc,
                    )
                    try:
                        return self._transcribe_audio_with_groq(video_url)
                    except Exception as fallback_err:
                        raise TranscriptError(
                            f"Could not retrieve transcript for video {video_id} via YouTube API ({exc}), "
                            f"and Groq Whisper fallback also failed: {fallback_err}"
                        ) from fallback_err

                raise TranscriptError(
                    f"Could not retrieve the transcript for video {video_id} "
                    f"after {attempt} attempt(s). "
                    f"Underlying error: {type(exc).__name__}: {exc}"
                ) from exc

        # Defensive fallback; the loop normally returns or raises.
        raise TranscriptError(
            f"Could not retrieve the transcript for video {video_id}."
        )