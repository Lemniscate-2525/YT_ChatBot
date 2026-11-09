from unittest.mock import patch
from langchain_core.documents import Document
import rag


def test_exact_answer_cache_hit_skips_runtime():
    payload = {"answer": "A cached, validated answer.", "sources": [{
        "page_content": "Transcript evidence",
        "metadata": {"video_id": "abcdefghijk", "video_url": "https://youtube.com/watch?v=abcdefghijk", "start_seconds": 12.0},
    }]}
    with patch.object(rag, "check_rate_limit"), patch.object(rag, "get_json", return_value=payload), patch.object(rag, "_get_runtime") as runtime:
        answer, docs = rag.ask_question("What happened?", user_id="user-a")
    assert answer == payload["answer"]
    assert isinstance(docs[0], Document)
    runtime.assert_not_called()


def test_cache_serialization_preserves_source_documents():
    doc = Document(page_content="Evidence", metadata={"video_id": "abcdefghijk"})
    payload = rag._serialize_result("Answer", [doc])
    answer, docs = rag._deserialize_result(payload)
    assert answer == "Answer"
    assert docs[0].page_content == "Evidence"
    assert docs[0].metadata["video_id"] == "abcdefghijk"
