import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.index import HybridIndex
from src.rag.score import score_options_with_rag, score_to_probs


def _corpus():
    return [
        {"id": "photo", "text": "Photosynthesis converts sunlight into glucose in plants."},
        {"id": "resp", "text": "Cellular respiration breaks down glucose to release energy."},
        {"id": "newton", "text": "Newton's laws describe the relationship between force and motion."},
        {"id": "water", "text": "Water is composed of hydrogen and oxygen atoms."},
    ]


def test_score_options_returns_one_per_letter():
    idx = HybridIndex(_corpus(), embedder=None)
    prompt = "How do plants make food?"
    options = {
        "A": "photosynthesis using sunlight",
        "B": "digesting soil directly",
        "C": "consuming other plants",
        "D": "absorbing rainwater only",
        "E": "converting nitrogen",
    }
    scores = score_options_with_rag(prompt, options, idx, reranker=None, top_k=2)
    assert set(scores.keys()) == {"A", "B", "C", "D", "E"}
    for v in scores.values():
        assert isinstance(v, float)


def test_score_options_prefers_supported_option():
    idx = HybridIndex(_corpus(), embedder=None)
    prompt = "How do plants make food?"
    options = {
        "A": "photosynthesis using sunlight and glucose",
        "B": "unrelated random content xyz",
        "C": "another unrelated content",
        "D": "yet another unrelated",
        "E": "one more unrelated",
    }
    scores = score_options_with_rag(prompt, options, idx, reranker=None, top_k=2)
    assert scores["A"] == max(scores.values())


def test_score_to_probs_sums_to_one():
    scores = {"A": 0.1, "B": 0.9, "C": 0.2, "D": 0.3, "E": 0.05}
    probs = score_to_probs(scores)
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert probs["B"] == max(probs.values())


def test_score_to_probs_temperature_sharpens():
    scores = {"A": 1.0, "B": 2.0, "C": 0.5, "D": 0.5, "E": 0.5}
    soft = score_to_probs(scores, temperature=2.0)
    sharp = score_to_probs(scores, temperature=0.5)
    assert sharp["B"] > soft["B"]