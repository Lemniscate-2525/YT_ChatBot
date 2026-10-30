import pytest

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma


class FakeEmbeddings(Embeddings):
    """Generate deterministic embeddings without an API call."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        # Simple deterministic vector for testing only.
        return [
            float(len(text)),
            float(sum(ord(char) for char in text) % 1000),
            float(text.count(" ") + 1),
        ]


def test_chroma_stores_and_retrieves_documents(tmp_path):
    embeddings = FakeEmbeddings()

    vector_store = Chroma(
        collection_name="integration_test",
        embedding_function=embeddings,
        persist_directory=str(tmp_path / "chroma_test"),
    )

    documents = [
        Document(
            page_content="Transformers use attention mechanisms.",
            metadata={
                "video_id": "abcdefghijk",
                "video_url": "https://www.youtube.com/watch?v=abcdefghijk",
                "start_seconds": 12.0,
                "source": "youtube",
            },
        ),
        Document(
            page_content="Convolutional neural networks process images.",
            metadata={
                "video_id": "lmnopqrstuv",
                "video_url": "https://www.youtube.com/watch?v=lmnopqrstuv",
                "start_seconds": 25.0,
                "source": "youtube",
            },
        ),
    ]

    vector_store.add_documents(
        documents=documents,
        ids=["chunk_1", "chunk_2"],
    )

    results = vector_store.get(
        where={"video_id": "abcdefghijk"},
        include=["documents", "metadatas"],
    )

    assert len(results["ids"]) == 1
    assert results["ids"][0] == "chunk_1"
    assert (
        results["documents"][0]
        == "Transformers use attention mechanisms."
    )
    assert results["metadatas"][0]["start_seconds"] == 12.0
    assert results["metadatas"][0]["source"] == "youtube"

    retrieved_docs = vector_store.similarity_search(
        "Transformers and attention",
        k=1,
    )

    assert len(retrieved_docs) == 1
    assert isinstance(retrieved_docs[0], Document)
    assert retrieved_docs[0].page_content