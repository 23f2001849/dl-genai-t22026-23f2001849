"""
download_models.py — fetch model weights from the Hugging Face Hub into ./artifacts/.

The weights live in a Hub *model* repo rather than in git, because GitHub's free LFS quota is
1 GB of storage and 1 GB of bandwidth per month, which a 418 MB BERT checkpoint exhausts almost
immediately. The Hub has no such limit for this use.

app.py calls ensure_artifacts() at import. If artifacts/ is already populated (local development,
or a Space that has them committed directly) nothing is downloaded.

Set the repo id with the MODEL_REPO environment variable, or edit the default below.
"""

import os
from pathlib import Path

MODEL_REPO = os.environ.get("MODEL_REPO", "Krishna-11Analyst/smart-mcq-solver-models")
ART = Path("artifacts")

# filename in the Hub repo -> path it must land at locally
REQUIRED = {"lr_model.joblib": ART / "lr_model.joblib"}
OPTIONAL = {
    "textcnn.pt": ART / "textcnn.pt",
    # tokenizer.json is the fast-tokenizer format and carries the vocabulary inside it,
    # so vocab.txt and special_tokens_map.json are not required.
    "bert_ranker/config.json":           ART / "bert_ranker/config.json",
    "bert_ranker/model.safetensors":     ART / "bert_ranker/model.safetensors",
    "bert_ranker/tokenizer.json":        ART / "bert_ranker/tokenizer.json",
    "bert_ranker/tokenizer_config.json": ART / "bert_ranker/tokenizer_config.json",
}


def ensure_artifacts(verbose=True):
    """Populate ./artifacts/ from the Hub. Returns the list of files available locally."""
    ART.mkdir(exist_ok=True)

    missing = [k for k, v in {**REQUIRED, **OPTIONAL}.items() if not v.exists()]
    if not missing:
        if verbose:
            print("artifacts/ already populated, skipping download")
        return sorted(p.name for p in ART.rglob("*") if p.is_file())

    from huggingface_hub import hf_hub_download

    def grab(name, dest, required):
        if dest.exists():
            return True
        try:
            src = hf_hub_download(repo_id=MODEL_REPO, filename=name, repo_type="model")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                os.symlink(src, dest)
            if verbose:
                print(f"  fetched {name}")
            return True
        except Exception as exc:
            if required:
                raise RuntimeError(
                    f"Could not fetch required file '{name}' from '{MODEL_REPO}'. "
                    f"Check MODEL_REPO and that the repo is public. Original error: {exc}"
                ) from exc
            if verbose:
                print(f"  skipped {name} (not in the repo)")
            return False

    if verbose:
        print(f"downloading weights from {MODEL_REPO}")
    for name, dest in REQUIRED.items():
        grab(name, dest, required=True)
    for name, dest in OPTIONAL.items():
        grab(name, dest, required=False)

    return sorted(p.name for p in ART.rglob("*") if p.is_file())


if __name__ == "__main__":
    for f in ensure_artifacts():
        print(f)
