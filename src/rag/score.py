# per-option retrieval-augmented scoring for MCQA
import numpy as np


def score_options_with_rag(prompt, options, index, reranker=None, top_k=5, alpha=0.5):
    """
    For a single MCQ row, produce a score per option using RAG.

    prompt: str, the question stem
    options: dict {'A': text, 'B': text, ...}
    index: HybridIndex instance
    reranker: optional Reranker instance
    top_k: passages to retrieve per option-augmented query
    alpha: BM25/dense fusion weight in the index

    Returns dict {letter: score} where higher = more supported by retrieved passages.
    """
    scores = {}
    for letter, opt_text in options.items():
        query = f"{prompt} {opt_text}"
        hits = index.search_hybrid(query, top_k=top_k, alpha=alpha)

        if reranker is not None and hits:
            passages = [{"id": i, "text": index.get(i)["text"]} for i, _ in hits]
            reranked = reranker.rerank(query, passages, top_k=top_k)
            scores[letter] = float(np.mean([p["rerank_score"] for p in reranked]))
        else:
            scores[letter] = float(np.mean([s for _, s in hits])) if hits else 0.0

    return scores


def score_to_probs(scores, temperature=1.0):
    """Softmax over the 5 option scores."""
    letters = list(scores.keys())
    vals = np.array([scores[L] for L in letters]) / temperature
    vals = vals - vals.max()
    exp = np.exp(vals)
    probs = exp / exp.sum()
    return dict(zip(letters, probs.tolist()))