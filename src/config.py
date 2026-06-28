# Central configuration for the Smart MCQ Solver project (T2-2026).
# Imported by notebooks/ and src/* modules so seed, label ordering,
# paths, and W&B constants live in exactly one place.

import os

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── Label space ──────────────────────────────────────────────────────────────
# Single sorted source so LABEL2IDX never drifts across modules.
LABELS = sorted(["A", "B", "C", "D", "E"])
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}
IDX2LABEL = {i: l for l, i in LABEL2IDX.items()}
NUM_CLASSES = len(LABELS)
NUM_OPTIONS = 5

# ── Tokenization & sequence ──────────────────────────────────────────────────
MAX_LEN = 256

# ── Cross-validation ─────────────────────────────────────────────────────────
N_FOLDS = 5
GROUP_COL = "core_q"  # stem-grouped GroupKFold — prevents core-question leakage

# ── Paths (Kaggle vs local auto-detect) ──────────────────────────────────────
IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    DATA_DIR = "/kaggle/input/smart-mcq-solver-challenge"
    OUT_DIR = "/kaggle/working"
else:
    DATA_DIR = os.environ.get("MCQ_DATA_DIR", "./data")
    OUT_DIR = os.environ.get("MCQ_OUT_DIR", "./outputs")

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

# Artifacts directory — cached embeddings, lookup tables, OOF probs.
# Gitignored. Created on demand by the pipeline.
ARTIFACTS_DIR = os.path.join(OUT_DIR, "artifacts")

# ── Weights & Biases ─────────────────────────────────────────────────────────
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "23f2001849")
WANDB_PROJECT = "23f2001849-t22026"

# ── Submission ───────────────────────────────────────────────────────────────
SUBMISSION_NAME = "submission.csv"
TOP_K = 3  # competition metric is MAP@3
