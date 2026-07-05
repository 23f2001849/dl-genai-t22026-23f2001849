import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.chunk import split_sentences, chunk_passage


def test_split_sentences_basic():
    assert split_sentences("A. B. C.") == ["A.", "B.", "C."]


def test_split_sentences_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_chunk_respects_word_cap():
    long_text = ". ".join(["Word " * 30] * 6).strip() + "."
    chunks = chunk_passage(long_text, max_words=50, overlap_sentences=0)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.split()) <= 60  # some slack for the sentence boundary


def test_chunk_single_short_passage():
    text = "Short passage. Two sentences."
    chunks = chunk_passage(text, max_words=100)
    assert chunks == ["Short passage. Two sentences."]


def test_chunk_overlap():
    long_text = ". ".join([f"Sentence {i}" for i in range(20)]) + "."
    chunks = chunk_passage(long_text, max_words=30, overlap_sentences=1)
    # overlap means consecutive chunks share at least one sentence prefix
    if len(chunks) >= 2:
        first_end = chunks[0].split(". ")[-1]
        second_start = chunks[1].split(". ")[0]
        assert first_end == second_start or first_end.rstrip(".") == second_start.rstrip(".")