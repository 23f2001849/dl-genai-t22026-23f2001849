# hybrid retrieval: BM25 sparse + MiniLM dense, with score fusion
import numpy as np
from rank_bm25 import BM25Okapi


def _tokenize(text):
    return text.lower().split()


class HybridIndex:
    def __init__(self, passages, embedder=None):
        """
        passages: list of dicts with keys 'id' and 'text'
        embedder: sentence-transformers model instance (optional; enables dense search)
        """
        self.passages = passages
        self.tokenized = [_tokenize(p["text"]) for p in passages]
        self.bm25 = BM25Okapi(self.tokenized)
        self.embedder = embedder
        self.dense_matrix = None
        if embedder is not None:
            self.dense_matrix = embedder.encode(
                [p["text"] for p in passages],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

    def search_bm25(self, query, top_k=20):
        scores = self.bm25.get_scores(_tokenize(query))
        idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in idx]

    def search_dense(self, query, top_k=20):
        if self.dense_matrix is None:
            raise RuntimeError("no embedder configured")
        q = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = self.dense_matrix @ q
        idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in idx]

    def search_hybrid(self, query, top_k=20, alpha=0.5):
        """
        Reciprocal rank fusion of BM25 and dense results.
        alpha in [0,1]: weight on dense score.
        """
        bm25_hits = self.search_bm25(query, top_k=top_k * 2)
        if self.dense_matrix is None:
            return bm25_hits[:top_k]
        dense_hits = self.search_dense(query, top_k=top_k * 2)

        # min-max normalize each score list, then blend
        def _norm(hits):
            if not hits:
                return {}
            vals = np.array([s for _, s in hits])
            lo, hi = vals.min(), vals.max()
            rng = hi - lo if hi > lo else 1.0
            return {i: (s - lo) / rng for i, s in hits}

        n_bm25 = _norm(bm25_hits)
        n_dense = _norm(dense_hits)
        keys = set(n_bm25) | set(n_dense)
        fused = {
            k: (1 - alpha) * n_bm25.get(k, 0.0) + alpha * n_dense.get(k, 0.0)
            for k in keys
        }
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(k, float(v)) for k, v in ranked]

    def get(self, idx):
        return self.passages[idx]