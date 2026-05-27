# wrapper stem normalization for the smart mcq solver competition
import re

_PREFIXES = [
    "Pick the best possible answer:",
    "Select the most accurate option:",
    "Determine the correct option:",
    "Identify the correct statement:",
    "Choose the correct answer:",
    "Which of the following is correct?",
]

_SUFFIXES = [
    "among the listed options.",
    "based on the given context.",
    "from the following choices.",
    "carefully.",
]

_PREFIX_RE = re.compile(r"^\s*(?:" + "|".join(re.escape(p) for p in _PREFIXES) + r")\s*")
_SUFFIX_RE = re.compile(r"\s*(?:" + "|".join(re.escape(s) for s in _SUFFIXES) + r")\s*$")
_WS_RE = re.compile(r"\s+")


def normalize_stem(text):
    if not isinstance(text, str):
        return text
    s = text
    while True:
        before = s
        s = _PREFIX_RE.sub("", s)
        s = _SUFFIX_RE.sub("", s)
        if s == before:
            break
    s = _WS_RE.sub(" ", s).strip()
    return s