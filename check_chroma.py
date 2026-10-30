from retrieval.vec_store import get_vector_store

vector_store = get_vector_store()

# Retrieve all stored document IDs and metadata.
result = vector_store.get(
    include=["metadatas", "documents"]
)

print(f"Total chunks in Chroma: {len(result['ids'])}")

print("\nMetadata for the latest video's chunks:\n")

target_video_id = "WZPLz3TfxLc"

for chunk_id, metadata, document in zip(
    result["ids"],
    result["metadatas"],
    result["documents"],
):
    if metadata.get("video_id") == target_video_id:
        print(f"Chunk ID: {chunk_id}")
        print(f"Metadata: {metadata}")
        print(f"Text preview: {document[:150]}")
        print("-" * 60)