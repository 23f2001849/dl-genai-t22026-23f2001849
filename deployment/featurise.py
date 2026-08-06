import re
OPT = ["A", "B", "C", "D", "E"]

def pair_features(prompt, option):
    """Everything the classical model needs for one (question, option) pair."""
    p  = str(prompt)
    o  = str(option)
    pw = set(re.findall(r"\\w+", p.lower()))
    ow = set(re.findall(r"\\w+", o.lower()))
    ov = len(pw & ow)
    return {
        "text":      f"{p} [SEP] {o}",
        "overlap":   ov,
        "jaccard":   ov / max(len(pw | ow), 1),
        "len_ratio": len(o) / max(len(p), 1),
    }

def expand(prompt, options):
    """One question with five options -> five feature dicts, in A..E order."""
    return [pair_features(prompt, options[L]) for L in OPT]
