import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.corpus import load_corpus_from_json, save_corpus_to_json


def test_save_and_load_roundtrip():
    passages = [
        {"id": "a", "title": "A", "text": "First passage."},
        {"id": "b", "title": "B", "text": "Second passage."},
    ]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "corpus.json")
        save_corpus_to_json(passages, path)
        loaded = load_corpus_from_json(path)
        assert loaded == passages


def test_save_creates_missing_directory():
    passages = [{"id": "x", "title": "X", "text": "Content."}]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "nested", "dir", "corpus.json")
        save_corpus_to_json(passages, path)
        assert os.path.exists(path)
        with open(path) as f:
            assert json.load(f) == passages