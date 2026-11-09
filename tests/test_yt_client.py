from unittest.mock import Mock, patch

from requests.exceptions import RetryError
from youtube_transcript_api._errors import IpBlocked, TranscriptsDisabled, YouTubeRequestFailed

from ingestion.yt_client import (
    TranscriptError,
    YouTubeTranscriptClient,
    _should_retry,
)


VIDEO_URL = "https://youtu.be/xhTXq26dwoA"


def reset_circuit():
    YouTubeTranscriptClient._blocked_until = 0.0
    YouTubeTranscriptClient._block_reason = ""


def test_retries_transient_connection_then_succeeds():
    reset_circuit()
    client = YouTubeTranscriptClient(enable_audio_fallback=False)
    client._api = Mock()
    client._api.fetch.side_effect = [
        ConnectionError("temporary"),
        [Mock(text="Hello world", start=0.0, duration=2.0)],
    ]

    with patch("ingestion.yt_client._wait_before_retry") as wait_mock:
        segments = client.fetch_transcript(VIDEO_URL)

    assert segments[0]["text"] == "Hello world"
    assert client._api.fetch.call_count == 2
    wait_mock.assert_called_once()


def test_http_503_is_retryable():
    exc = YouTubeRequestFailed("xhTXq26dwoA", Mock())
    exc.reason = "503 Server Error: Service Unavailable"
    assert _should_retry(exc) is True


def test_http_403_is_not_retryable():
    exc = YouTubeRequestFailed("xhTXq26dwoA", Mock())
    exc.reason = "403 Client Error: Forbidden"
    assert _should_retry(exc) is False


def test_ip_block_is_not_retryable():
    assert _should_retry(IpBlocked("xhTXq26dwoA")) is False


def test_retry_error_with_429_is_not_retryable():
    exc = RetryError("HTTPSConnectionPool: too many 429 error responses")
    assert _should_retry(exc) is False


def test_429_opens_circuit_and_does_not_retry():
    reset_circuit()
    client = YouTubeTranscriptClient(enable_audio_fallback=False, block_cooldown_seconds=120)
    client._api = Mock()
    client._api.fetch.side_effect = RetryError("too many 429 error responses")

    with patch("ingestion.yt_client._wait_before_retry") as wait_mock:
        try:
            client.fetch_transcript(VIDEO_URL)
        except TranscriptError as exc:
            assert "HTTP 429" in str(exc)
            assert "cooldown" in str(exc).lower()
        else:
            raise AssertionError("Expected TranscriptError")

    assert client._api.fetch.call_count == 1
    wait_mock.assert_not_called()

    # A second video is rejected before another upstream request.
    try:
        client.fetch_transcript("https://youtu.be/abcdefghijk")
    except TranscriptError as exc:
        assert "cooling down" in str(exc).lower()
    else:
        raise AssertionError("Expected circuit-open TranscriptError")
    assert client._api.fetch.call_count == 1
    reset_circuit()


def test_transcripts_disabled_fails_without_audio_fallback():
    reset_circuit()
    client = YouTubeTranscriptClient(enable_audio_fallback=False)
    client._api = Mock()
    client._api.fetch.side_effect = TranscriptsDisabled("xhTXq26dwoA")
    client._transcribe_audio_with_groq = Mock()

    try:
        client.fetch_transcript(VIDEO_URL)
    except TranscriptError:
        pass
    else:
        raise AssertionError("Expected TranscriptError")

    client._transcribe_audio_with_groq.assert_not_called()
    client._api.fetch.assert_called_once()


def test_invalid_url_does_not_call_api():
    client = YouTubeTranscriptClient(enable_audio_fallback=False)
    client._api = Mock()

    try:
        client.fetch_transcript("https://example.com/not-youtube")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    client._api.fetch.assert_not_called()


def test_empty_transcript_fails_immediately():
    reset_circuit()
    client = YouTubeTranscriptClient(enable_audio_fallback=False)
    client._api = Mock()
    client._api.fetch.return_value = []

    try:
        client.fetch_transcript(VIDEO_URL)
    except TranscriptError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("Expected TranscriptError")

    client._api.fetch.assert_called_once()


def test_webshare_proxy_initialization():
    client = YouTubeTranscriptClient(
        proxy_username="test_user",
        proxy_password="test_pass",
        enable_audio_fallback=False,
    )
    assert client.proxy_username == "test_user"
    assert client.proxy_password == "test_pass"
    assert client._effective_proxy_url is not None
    assert "test_user" in client._effective_proxy_url
    assert "p.webshare.io" in client._effective_proxy_url


def test_generic_proxy_initialization():
    client = YouTubeTranscriptClient(
        proxy_url="http://proxy.example.com:8080",
        proxy_username="",
        proxy_password="",
        enable_audio_fallback=False,
    )
    assert client._effective_proxy_url == "http://proxy.example.com:8080"
