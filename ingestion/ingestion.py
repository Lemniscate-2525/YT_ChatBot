import hashlib
import re
from bisect import bisect_right

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.yt_client import YouTubeTranscriptClient, extract_video_id
from retrieval.vec_store import get_vector_store

from ingestion.yt_client import TranscriptError


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


transcript_client = YouTubeTranscriptClient()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = CHUNK_SIZE,
    chunk_overlap = CHUNK_OVERLAP,
    length_function = len,
    add_start_index = True, # Gives starting char position. 
)

# --------------------------------------------------
# 2. TEXT CLEANING
# --------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize whitespace and remove empty text."""

    if not isinstance(text, str):
        return ""

    text = re.sub(r"\s+", " ", text) # Whitespace removal/norm; to give chunking algo a cleaner input.
    return text.strip()


# --------------------------------------------------
# 3. TRANSCRIPT PREPARATION
# --------------------------------------------------

def prepare_transcript(segments: list[dict]) -> tuple[str, list[int], list[float]]:
    """
    Combine transcript segments into one text string.

    Track each segment's character offset and starting timestamp
    so individual chunks can retain their approximate start time.
    """

    transcript_parts = []
    segment_offsets = []
    segment_timestamps = []

    current_length = 0

    for segment in segments:
        text = clean_text(segment.get("text", ""))

        if not text:
            continue

        start_seconds = float(segment["start"])

        # Account for the newline inserted between transcript segments.
        if transcript_parts:
            current_length += 1

        segment_offsets.append(current_length)
        segment_timestamps.append(start_seconds)

        transcript_parts.append(text)
        current_length += len(text)

    transcript = "\n".join(transcript_parts)

    if not transcript:
        raise ValueError("The transcript contains no usable text.")

    return transcript, segment_offsets, segment_timestamps


# --------------------------------------------------
# 4. CHUNKING WITH TIMESTAMP PRESERVATION
# --------------------------------------------------

def create_chunks(
    segments: list[dict],
    video_id: str,
    video_url: str,
) -> list[Document]:
    """Create timestamp-aware LangChain documents."""

    transcript, segment_offsets, segment_timestamps = (
        prepare_transcript(segments)
    )

    source_document = Document(
        page_content = transcript,
        metadata = {
            "video_id": video_id,
            "video_url": video_url,
            "source": "youtube",
        },
    )

    chunks = text_splitter.split_documents([source_document])

    timestamped_chunks = []

    for chunk in chunks:
        start_index = chunk.metadata.pop("start_index")

        # Find the transcript segment at or immediately before
        # this chunk's starting character position.
        segment_index = bisect_right(
            segment_offsets,
            start_index,
        ) - 1

        if segment_index < 0:
            start_seconds = segment_timestamps[0]
        else:
            start_seconds = segment_timestamps[segment_index]

        chunk.metadata["start_seconds"] = start_seconds

        timestamped_chunks.append(chunk)

    return timestamped_chunks


# --------------------------------------------------
# 5. DUPLICATE DETECTION
# --------------------------------------------------

def video_already_ingested(vector_store, video_id: str) -> bool:
    """Check whether this video already has stored chunks."""

    existing = vector_store.get(
        where = {"video_id": video_id},
        include = ["metadatas"],
    )

    return bool(existing.get("ids")) 

# Generate Chunk IDs : 

def generate_chunk_ids(chunks: list[Document], video_id: str) -> list[str]:
    """Generate deterministic IDs for transcript chunks."""

    chunk_ids = []

    for index, chunk in enumerate(chunks):
        content_hash = hashlib.sha256(
            chunk.page_content.encode("utf-8")
        ).hexdigest()[:12]

        chunk_ids.append(f"{video_id}_{index}_{content_hash}")

    return chunk_ids

# --------------------------------------------------
# 6. INGESTION PIPELINE
# --------------------------------------------------

def ingest_video(video_url: str, refresh: bool = False) -> dict:
    """
    Fetch, clean, chunk, embed, and store a YouTube transcript.

    By default, skip videos that have already been ingested.
    Set refresh=True to replace their stored transcript chunks.
    """

    video_id = extract_video_id(video_url)

    canonical_url = (
        f"https://www.youtube.com/watch?v={video_id}"
    )

    vector_store = get_vector_store(require_non_empty = False)

    existing = video_already_ingested(vector_store, video_id)

    if existing and not refresh:
        return {
            "status": "skipped",
            "video_id": video_id,
            "video_url": canonical_url,
            "message": "This video has already been ingested.",
            "chunks_added": 0,
        }

    # Fetch and prepare the new transcript before modifying stored data : 

    segments = transcript_client.fetch_transcript(canonical_url)

    chunks = create_chunks(
        segments = segments,
        video_id = video_id,
        video_url = canonical_url,
    )

    if not chunks:
        raise ValueError("No chunks were produced from the transcript.")

    chunk_ids = generate_chunk_ids(chunks, video_id)

    if existing and refresh:
        # Remove only this video's chunks. 
        vector_store._collection.delete(where = {"video_id": video_id})

    vector_store.add_documents(
        documents = chunks,
        ids = chunk_ids,
    )

    return {
        "status": "refreshed" if existing and refresh else "success",
        "video_id": video_id,
        "video_url": canonical_url,
        "chunks_added": len(chunks),
        "message": (
            "Video refreshed successfully."
            if existing and refresh
            else "Video ingested successfully."
        ),
    }
# --------------------------------------------------
# 7. COMMAND-LINE ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    url = input("Enter a YouTube video URL: ").strip()

    try:
        result = ingest_video(url)

        print("\nIngestion result:")

        for key, value in result.items():
            print(f"{key}: {value}")

    except Exception as exc:
        print(f"\nIngestion failed: {type(exc).__name__}: {exc}")