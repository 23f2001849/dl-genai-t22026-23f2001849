import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.index import HybridIndex, _tokenize


def _corpus():
    return [
        {"id": "a", "text": "The mitochondrion is the powerhouse of the cell."},
        {"id": "b", "text": "Photosynthesis converts sunlight into chemical energy in plants."},
        {"id": "c", "text": "Newton's second law relates force, mass and acceleration."},
        {"id": "d", "text": "Water freezes at zero degrees Celsius at standard pressure."},
        {"id": "e", "text": "DNA carries genetic instructions in living organisms."},
    ]


def test_tokenize_lowercases_and_splits():
    assert _tokenize("Hello World!") == ["hello", "world!"]


def test_bm25_finds_relevant_doc():
    idx = HybridIndex(_corpus(), embedder=None)
    hits = idx.search_bm25("what is the powerhouse of the cell", top_k=1)
    assert idx.get(hits[0][0])["id"] == "a"


def test_bm25_returns_requested_topk():
    idx = HybridIndex(_corpus(), embedder=None)
    hits = idx.search_bm25("cell", top_k=3)
    assert len(hits) == 3


def test_dense_search_requires_embedder():
    idx = HybridIndex(_corpus(), embedder=None)
    try:
        idx.search_dense("anything")
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError when embedder is None")


def test_hybrid_falls_back_to_bm25_without_embedder():
    idx = HybridIndex(_corpus(), embedder=None)
    hits = idx.search_hybrid("Newton force acceleration", top_k=2)
    top_id = idx.get(hits[0][0])["id"]
    assert top_id == "c"