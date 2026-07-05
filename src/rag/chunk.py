# sentence-boundary chunking with a token-count cap
import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text):
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def chunk_passage(text, max_words=120, overlap_sentences=1):
    """Group sentences into chunks capped at max_words, with overlap between chunks."""
    sents = split_sentences(text)
    if not sents:
        return []

    chunks = []
    cur, cur_words = [], 0
    for s in sents:
        w = len(s.split())
        if cur and cur_words + w > max_words:
            chunks.append(" ".join(cur))
            if overlap_sentences and len(cur) >= overlap_sentences:
                cur = cur[-overlap_sentences:]
                cur_words = sum(len(x.split()) for x in cur)
            else:
                cur, cur_words = [], 0
        cur.append(s)
        cur_words += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks