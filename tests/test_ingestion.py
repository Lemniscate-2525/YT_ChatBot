import pytest
from unittest.mock import Mock

import ingestion.ingestion as ingestion
from langchain_core.documents import Document

from ingestion.yt_client import TranscriptError

# --------------------------------------------------
# 1. TEXT CLEANING
# --------------------------------------------------

def test_clean_text_normalizes_whitespace():
    text = "  Hello   world!\n\nHow\tare you?  "

    result = ingestion.clean_text(text)

    assert result == "Hello world! How are you?"

def test_clean_text_handles_non_string_input():
    assert ingestion.clean_text(None) == ""
    assert ingestion.clean_text(123) == ""


# --------------------------------------------------
# 2. TRANSCRIPT PREPARATION
# --------------------------------------------------

def test_prepare_transcript_preserves_timestamps():
    segments = [
        {"text": "Hello world.", "start": 0.0, "duration": 2.0},
        {"text": "Welcome back.", "start": 2.0, "duration": 3.0},
        {"text": "Let's begin.", "start": 5.0, "duration": 2.0},
    ]

    transcript, offsets, timestamps = ingestion.prepare_transcript(
        segments
    )

    assert transcript == "Hello world.\nWelcome back.\nLet's begin."
    assert offsets == [0, 13, 27]
    assert timestamps == [0.0, 2.0, 5.0]


def test_prepare_transcript_ignores_empty_segments():
    segments = [
        {"text": "Hello.", "start": 0.0, "duration": 1.0},
        {"text": "   ", "start": 1.0, "duration": 1.0},
        {"text": "Goodbye.", "start": 2.0, "duration": 1.0},
    ]

    transcript, offsets, timestamps = ingestion.prepare_transcript(
        segments
    )

    assert transcript == "Hello.\nGoodbye."
    assert offsets == [0, 7]
    assert timestamps == [0.0, 2.0]


def test_prepare_transcript_rejects_empty_transcript():
    segments = [
        {"text": "  ", "start": 0.0, "duration": 1.0},
        {"text": "", "start": 1.0, "duration": 1.0},
    ]

    with pytest.raises(ValueError, match="no usable text"):
        ingestion.prepare_transcript(segments)


# --------------------------------------------------
# 3. CHUNK CREATION
# --------------------------------------------------

def test_create_chunks_preserves_metadata():
    segments = [
        {
            "text": "Transformers use attention mechanisms.",
            "start": 10.0,
            "duration": 4.0,
        },
        {
            "text": "Attention helps process relationships between tokens.",
            "start": 14.0,
            "duration": 5.0,
        },
    ]

    chunks = ingestion.create_chunks(
        segments=segments,
        video_id="abcdefghijk",
        video_url="https://www.youtube.com/watch?v=abcdefghijk",
    )

    assert len(chunks) >= 1

    for chunk in chunks:
        assert isinstance(chunk, Document)
        assert chunk.page_content.strip()
        assert chunk.metadata["video_id"] == "abcdefghijk"
        assert (
            chunk.metadata["video_url"]
            == "https://www.youtube.com/watch?v=abcdefghijk"
        )
        assert chunk.metadata["source"] == "youtube"
        assert isinstance(chunk.metadata["start_seconds"], float)
        assert chunk.metadata["start_seconds"] >= 0


# --------------------------------------------------
# 4. DUPLICATE DETECTION
# --------------------------------------------------

def test_video_already_ingested_returns_true():
    vector_store = Mock()

    vector_store.get.return_value = {
        "ids": ["chunk_1", "chunk_2"],
        "metadatas": [{}, {}],
    }

    result = ingestion.video_already_ingested(
        vector_store,
        "abcdefghijk",
    )

    assert result is True

    vector_store.get.assert_called_once_with(
        where={"video_id": "abcdefghijk"},
        include=["metadatas"],
    )


def test_video_already_ingested_returns_false():
    vector_store = Mock()

    vector_store.get.return_value = {
        "ids": [],
        "metadatas": [],
    }

    result = ingestion.video_already_ingested(
        vector_store,
        "abcdefghijk",
    )

    assert result is False


# --------------------------------------------------
# 5. SUCCESSFUL INGESTION
# --------------------------------------------------

def test_ingest_video_success(monkeypatch):
    video_id = "abcdefghijk"
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    vector_store = Mock()

    # Pretend the video has not been ingested before.
    vector_store.get.return_value = {
        "ids": [],
        "metadatas": [],
    }

    transcript_segments = [
        {
            "text": "This is a test transcript.",
            "start": 0.0,
            "duration": 3.0,
        },
        {
            "text": "We are testing the ingestion pipeline.",
            "start": 3.0,
            "duration": 4.0,
        },
    ]

    transcript_client = Mock()
    transcript_client.fetch_transcript.return_value = transcript_segments

    monkeypatch.setattr(
        ingestion,
        "get_vector_store",
        lambda **kwargs: vector_store,
    )

    monkeypatch.setattr(
        ingestion,
        "transcript_client",
        transcript_client,
    )

    result = ingestion.ingest_video(video_url)

    assert result["status"] == "success"
    assert result["video_id"] == video_id
    assert result["chunks_added"] >= 1

    transcript_client.fetch_transcript.assert_called_once_with(
        video_url
    )

    vector_store.add_documents.assert_called_once()

    call_kwargs = vector_store.add_documents.call_args.kwargs

    documents = call_kwargs["documents"]
    chunk_ids = call_kwargs["ids"]

    assert len(documents) == len(chunk_ids)

    assert all(
        document.metadata["video_id"] == video_id
        for document in documents
    )

    assert len(chunk_ids) == len(set(chunk_ids))


# --------------------------------------------------
# 6. DUPLICATE INGESTION
# --------------------------------------------------

def test_ingest_video_skips_existing_video(monkeypatch):
    video_id = "abcdefghijk"
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    vector_store = Mock()

    vector_store.get.return_value = {
        "ids": ["existing_chunk_1"],
        "metadatas": [{"video_id": video_id}],
    }

    transcript_client = Mock()

    monkeypatch.setattr(
        ingestion,
        "get_vector_store",
        lambda **kwargs: vector_store,
    )

    monkeypatch.setattr(
        ingestion,
        "transcript_client",
        transcript_client,
    )

    result = ingestion.ingest_video(video_url)

    assert result["status"] == "skipped"
    assert result["chunks_added"] == 0

    transcript_client.fetch_transcript.assert_not_called()
    vector_store.add_documents.assert_not_called()


# --------------------------------------------------
# 7. FAILURE HANDLING
# --------------------------------------------------

def test_invalid_url_fails_before_database_access(monkeypatch):
    vector_store_factory = Mock()

    monkeypatch.setattr(
        ingestion,
        "get_vector_store",
        vector_store_factory,
    )

    with pytest.raises(ValueError):
        ingestion.ingest_video("https://example.com/video")

    vector_store_factory.assert_not_called()


def test_empty_transcript_does_not_store_documents(monkeypatch):
    video_id = "abcdefghijk"
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    vector_store = Mock()

    vector_store.get.return_value = {
        "ids": [],
        "metadatas": [],
    }

    transcript_client = Mock()
    transcript_client.fetch_transcript.return_value = []

    monkeypatch.setattr(
        ingestion,
        "get_vector_store",
        lambda **kwargs: vector_store,
    )

    monkeypatch.setattr(
        ingestion,
        "transcript_client",
        transcript_client,
    )

    with pytest.raises(ValueError, match="no usable text"):
        ingestion.ingest_video(video_url)

    vector_store.add_documents.assert_not_called()


def test_transcript_failure_does_not_store_documents(monkeypatch):
    video_id = "abcdefghijk"
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    vector_store = Mock()

    vector_store.get.return_value = {
        "ids": [],
        "metadatas": [],
    }

    transcript_client = Mock()
    transcript_client.fetch_transcript.side_effect = (
        TranscriptError("Transcript unavailable")
    )

    monkeypatch.setattr(
        ingestion,
        "get_vector_store",
        lambda **kwargs: vector_store,
    )

    monkeypatch.setattr(
        ingestion,
        "transcript_client",
        transcript_client,
    )

    with pytest.raises(ingestion.TranscriptError):
        ingestion.ingest_video(video_url)

    vector_store.add_documents.assert_not_called()

def test_refresh_skips_fetch_when_already_ingested(monkeypatch):
    video_id = "abcdefghijk"
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    vector_store = Mock()
    vector_store.get.return_value = {
        "ids": ["existing_chunk"],
        "metadatas": [{"video_id": video_id}],
    }

    transcript_client = Mock()

    monkeypatch.setattr(
        ingestion,
        "get_vector_store",
        lambda **kwargs: vector_store,
    )
    monkeypatch.setattr(
        ingestion,
        "transcript_client",
        transcript_client,
    )

    result = ingestion.ingest_video(video_url)

    assert result["status"] == "skipped"
    transcript_client.fetch_transcript.assert_not_called()
    vector_store.add_documents.assert_not_called()


def test_refresh_fetch_failure_preserves_existing_chunks(monkeypatch):
    video_id = "abcdefghijk"
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    vector_store = Mock()
    vector_store.get.return_value = {
        "ids": ["existing_chunk"],
        "metadatas": [{"video_id": video_id}],
    }

    transcript_client = Mock()
    transcript_client.fetch_transcript.side_effect = TranscriptError(
        "Transcript unavailable"
    )

    monkeypatch.setattr(
        ingestion,
        "get_vector_store",
        lambda **kwargs: vector_store,
    )
    monkeypatch.setattr(
        ingestion,
        "transcript_client",
        transcript_client,
    )

    with pytest.raises(TranscriptError):
        ingestion.ingest_video(video_url, refresh=True)

    vector_store._collection.delete.assert_not_called()
    vector_store.add_documents.assert_not_called()


def test_generate_chunk_ids_are_deterministic():
    chunks = [
        Document(
            page_content="First chunk",
            metadata={"video_id": "abcdefghijk"},
        ),
        Document(
            page_content="Second chunk",
            metadata={"video_id": "abcdefghijk"},
        ),
    ]

    first = ingestion.generate_chunk_ids(chunks, "abcdefghijk")
    second = ingestion.generate_chunk_ids(chunks, "abcdefghijk")

    assert first == second
    assert len(first) == len(set(first))