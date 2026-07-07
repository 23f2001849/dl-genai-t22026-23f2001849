# cross-encoder reranking for RAG retrieval hits
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-12-v2", device=None):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query, passages, top_k=5):
        """
        query: str
        passages: list of dicts with 'id' and 'text'
        Returns top_k passages sorted by cross-encoder score, each with 'rerank_score' added.
        """
        if not passages:
            return []
        pairs = [(query, p["text"]) for p in passages]
        scores = self.model.predict(pairs, show_progress_bar=False)
        scored = [
            {**p, "rerank_score": float(s)} for p, s in zip(passages, scores)
        ]
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]