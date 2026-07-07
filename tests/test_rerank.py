import os, sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _reranker_available():
    try:
        from sentence_transformers import CrossEncoder  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _reranker_available(), reason="sentence-transformers not installed")
def test_rerank_orders_relevant_passage_first():
    from src.rag.rerank import Reranker

    passages = [
        {"id": "irrelevant", "text": "The Eiffel Tower is in Paris."},
        {"id": "relevant", "text": "Photosynthesis converts sunlight into chemical energy in plants."},
        {"id": "somewhat", "text": "Plants grow in soil."},
    ]
    r = Reranker()
    out = r.rerank("How do plants make food from sunlight?", passages, top_k=3)
    assert out[0]["id"] == "relevant"
    assert "rerank_score" in out[0]


@pytest.mark.skipif(not _reranker_available(), reason="sentence-transformers not installed")
def test_rerank_empty_input():
    from src.rag.rerank import Reranker
    r = Reranker()
    assert r.rerank("anything", [], top_k=5) == []


@pytest.mark.skipif(not _reranker_available(), reason="sentence-transformers not installed")
def test_rerank_respects_top_k():
    from src.rag.rerank import Reranker
    passages = [{"id": str(i), "text": f"Document {i} content."} for i in range(10)]
    r = Reranker()
    out = r.rerank("query text", passages, top_k=3)
    assert len(out) == 3