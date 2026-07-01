# 4-tier lookup for the smart mcq solver competition
from collections import Counter


def option_set(row):
    """Sorted tuple of the five option strings — the T1/T2 key."""
    return tuple(sorted(str(row[c]) for c in ("A", "B", "C", "D", "E")))


def build_lookup_tables(train):
    """Build the four lookup tables from the training dataframe.

    Returns a dict with:
        T1 — {(core_q, opt_set): correct_text}
        T2 — {opt_set: correct_text}
        T3 — set of option texts that are correct in train and never wrong
        T4 — {core_q: set of known correct answer texts}
    """
    T1, T2 = {}, {}
    core_to_correct = {}
    text_correct, text_wrong = Counter(), Counter()

    for _, r in train.iterrows():
        os_ = option_set(r)
        correct_text = str(r[r["answer"]])
        T1[(r["core_q"], os_)] = correct_text
        T2[os_] = correct_text
        core_to_correct.setdefault(r["core_q"], set()).add(correct_text)
        for L in "ABCDE":
            txt = str(r[L])
            if L == r["answer"]:
                text_correct[txt] += 1
            else:
                text_wrong[txt] += 1

    T3 = {t for t in text_correct if text_wrong.get(t, 0) == 0}
    return {"T1": T1, "T2": T2, "T3": T3, "T4": core_to_correct}


def lookup(row, tables):
    """Resolve a row through the four tiers.

    Returns (letter, tier). Both None if no tier fires unambiguously.
    """
    cq = row["core_q"]
    os_ = option_set(row)
    cand = {L: str(row[L]) for L in "ABCDE"}

    t = tables["T1"].get((cq, os_))
    if t is not None:
        hits = [L for L, v in cand.items() if v == t]
        if len(hits) == 1:
            return hits[0], "T1"

    t = tables["T2"].get(os_)
    if t is not None:
        hits = [L for L, v in cand.items() if v == t]
        if len(hits) == 1:
            return hits[0], "T2"

    hits = [L for L, v in cand.items() if v in tables["T3"]]
    if len(hits) == 1:
        return hits[0], "T3"

    known = tables["T4"].get(cq, set())
    hits = [L for L, v in cand.items() if v in known]
    if len(hits) == 1:
        return hits[0], "T4"

    return None, None
