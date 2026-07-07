# STEM corpus loader for the RAG pipeline
import json
import os

_DEFAULT_TOPICS = [
    "Photosynthesis", "Cellular respiration", "DNA", "Mitochondrion",
    "Newton's laws of motion", "Thermodynamics", "Quantum mechanics",
    "Superconductor", "Electromagnetism", "Special relativity",
    "Atomic theory", "Periodic table", "Chemical bond", "Acid-base reaction",
    "Evolution", "Natural selection", "Genetics", "Ecosystem",
    "Calculus", "Linear algebra", "Probability theory", "Set theory",
    "Phenomenology", "Existentialism", "Epistemology", "Metaphysics",
    "Logic", "Ethics", "Philosophy of mind", "Ontology",
]


def fetch_wikipedia_passages(topics=None, cache_path=None):
    """
    Fetch first-section text from Wikipedia for each topic via the REST API.
    Returns list of {'id': str, 'title': str, 'text': str}.
    Caches to cache_path if provided.
    """
    import requests

    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    topics = topics or _DEFAULT_TOPICS
    passages = []
    for t in topics:
        slug = t.replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "mcq-rag/1.0"})
            if r.status_code == 200:
                data = r.json()
                extract = data.get("extract", "").strip()
                if extract:
                    passages.append({
                        "id": slug,
                        "title": data.get("title", t),
                        "text": extract,
                    })
        except Exception:
            continue

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(passages, f, ensure_ascii=False, indent=2)

    return passages


def load_corpus_from_json(path):
    """Load a pre-built corpus from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_corpus_to_json(passages, path):
    """Persist a corpus to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False, indent=2)