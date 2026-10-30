from unittest.mock import Mock, patch

from ingestion.yt_client import YouTubeTranscriptClient, TranscriptError
from youtube_transcript_api._errors import (
    YouTubeRequestFailed,
    IpBlocked,
)

from youtube_transcript_api._errors import (
    TranscriptsDisabled,
)

# Specific Tets Cases for Diff HTTP Errors and their Retry Logic : 

def test_retries_then_succeeds():
    client = YouTubeTranscriptClient()

    # Simulate two temporary failures followed by success.
    client._api = Mock()
    client._api.fetch.side_effect = [
        ConnectionError("Temporary failure 1"),
        ConnectionError("Temporary failure 2"),
        [
            Mock(text="Hello world", start=0.0, duration=2.0)
        ],
    ]

    with patch("ingestion.yt_client._wait_before_retry") as wait_mock:
        segments = client.fetch_transcript(
            "https://youtu.be/xhTXq26dwoA"
        )

    assert len(segments) == 1
    assert segments[0]["text"] == "Hello world"
    assert segments[0]["start"] == 0.0

    assert client._api.fetch.call_count == 3
    assert wait_mock.call_count == 2

    print("PASS: Retry then succeed")
    print("Attempts:", client._api.fetch.call_count)
    print("Waits:", wait_mock.call_count)


def test_stops_after_three_attempts():
    client = YouTubeTranscriptClient()
    client._api = Mock()
    client._api.fetch.side_effect = ConnectionError(
        "Persistent network failure"
    )

    with patch("ingestion.yt_client._wait_before_retry"):
        try:
            client.fetch_transcript(
                "https://youtu.be/xhTXq26dwoA"
            )
        except TranscriptError as exc:
            print("PASS: Final failure handled")
            print("Attempts:", client._api.fetch.call_count)
            print("Error:", exc)
        else:
            raise AssertionError("Expected TranscriptError")

    assert client._api.fetch.call_count == 3

def test_retries_server_error():
    exc = YouTubeRequestFailed(
        "xhTXq26dwoA",
        Mock(status=503, __str__=lambda self: "503 Server Error"),
    )

    exc.reason = "503 Server Error: Service Unavailable"

    from ingestion.yt_client import _should_retry

    assert _should_retry(exc) is True
    print("PASS: HTTP 503 is retryable")


def test_does_not_retry_forbidden_error():
    exc = YouTubeRequestFailed(
        "xhTXq26dwoA",
        Mock(status=403, __str__=lambda self: "403 Forbidden"),
    )

    exc.reason = "403 Client Error: Forbidden"

    from ingestion.yt_client import _should_retry

    assert _should_retry(exc) is False
    print("PASS: HTTP 403 is not retryable")


def test_does_not_retry_ip_block():  # Test for IP Block Case.
    exc = IpBlocked("xhTXq26dwoA")

    from ingestion.yt_client import _should_retry

    assert _should_retry(exc) is False
    print("PASS: IP blocking is not retryable")


def test_fetch_retries_http_503_then_succeeds():
    from ingestion.yt_client import _wait_before_retry
    from youtube_transcript_api._errors import YouTubeRequestFailed

    client = YouTubeTranscriptClient()
    client._api = Mock()

    error = YouTubeRequestFailed(
        "xhTXq26dwoA",
        Mock(),
    )
    error.reason = "503 Server Error: Service Unavailable"

    client._api.fetch.side_effect = [
        error,
        error,
        [Mock(text="Hello world", start=0.0, duration=2.0)],
    ]

    with patch("ingestion.yt_client._wait_before_retry") as wait_mock:
        segments = client.fetch_transcript(
            "https://youtu.be/xhTXq26dwoA"
        )

    assert len(segments) == 1
    assert segments[0]["text"] == "Hello world"
    assert client._api.fetch.call_count == 3
    assert wait_mock.call_count == 2

    print("PASS: HTTP 503 retries twice, then succeeds")


def test_fetch_does_not_retry_http_403():
    from youtube_transcript_api._errors import YouTubeRequestFailed

    client = YouTubeTranscriptClient()
    client._api = Mock()

    error = YouTubeRequestFailed(
        "xhTXq26dwoA",
        Mock(),
    )
    error.reason = "403 Client Error: Forbidden"

    client._api.fetch.side_effect = error

    with patch("ingestion.yt_client._wait_before_retry") as wait_mock:
        try:
            client.fetch_transcript(
                "https://youtu.be/xhTXq26dwoA"
            )
        except TranscriptError:
            pass
        else:
            raise AssertionError("Expected TranscriptError")

    assert client._api.fetch.call_count == 1
    assert wait_mock.call_count == 0

    print("PASS: HTTP 403 fails immediately without retrying")


def test_invalid_url_does_not_call_api():  # Before external API call, YT URL should be Valid.
    client = YouTubeTranscriptClient()
    client._api = Mock()

    try:
        client.fetch_transcript("https://example.com/not-youtube")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid URL")

    client._api.fetch.assert_not_called()
    print("PASS: Invalid URL rejected before API call")


def test_empty_transcript_fails_immediately():  # No point of retrying empty transcript.
    client = YouTubeTranscriptClient()
    client._api = Mock()
    client._api.fetch.return_value = []

    with patch("ingestion.yt_client._wait_before_retry") as wait_mock:
        try:
            client.fetch_transcript("https://youtu.be/xhTXq26dwoA")
        except TranscriptError as exc:
            assert "empty" in str(exc).lower()
        else:
            raise AssertionError("Expected TranscriptError for empty transcript")

    assert client._api.fetch.call_count == 1
    assert wait_mock.call_count == 0
    print("PASS: Empty transcript fails without retrying")


def test_permanent_transcript_failure_is_not_retried():  # Perm failure like maybe IPBlocked or stuff ain't supp to be retried.
    client = YouTubeTranscriptClient()
    client._api = Mock()

    error = TranscriptsDisabled("xhTXq26dwoA")
    client._api.fetch.side_effect = error

    with patch("ingestion.yt_client._wait_before_retry") as wait_mock:
        try:
            client.fetch_transcript("https://youtu.be/xhTXq26dwoA")
        except TranscriptError:
            pass
        else:
            raise AssertionError("Expected TranscriptError")

    assert client._api.fetch.call_count == 1
    assert wait_mock.call_count == 0
    print("PASS: Disabled transcript fails without retrying")


if __name__ == "__main__":
    test_retries_then_succeeds()
    test_stops_after_three_attempts()
    test_retries_server_error()
    test_does_not_retry_forbidden_error()
    test_does_not_retry_ip_block()
    test_fetch_retries_http_503_then_succeeds()
    test_fetch_does_not_retry_http_403()
    test_invalid_url_does_not_call_api()
    test_empty_transcript_fails_immediately()
    test_permanent_transcript_failure_is_not_retried()