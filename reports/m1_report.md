# Milestone 1 Report — Smart MCQ Solver Challenge

**Author:** Krishna J (23f2001849)
**Submission date:** 29 May 2026
**Kaggle competition:** Smart MCQ Solver Challenge
**Public LB score (M1):** 0.74147 (rank 56 / 112 registered at submission time)

---

## 1. Scope

Milestone 1 deliverables:

- Reproducible wrapper-stripping regex with unit tests (`src/normalize.py`, `tests/test_normalize.py`).
- EDA notebook documenting the closed-question-bank finding (`notebooks/m1_eda.ipynb`).
- Workstream A v1 lookup pipeline with PH-A placeholder for the 45 paraphrase rows (`notebooks/m1_workstream_a_v1.ipynb`).
- Kaggle inference notebook port with diagnostic cells (`notebooks/kaggle_inference_v1.ipynb`).
- First competition submission to establish the LB floor.
- Revised dataset findings reflecting the actual LB ceiling (`reports/dataset_findings_revised.md`).

Workstream B (trained models) was not in M1 scope.

## 2. Dataset analysis (reproduced on the official data)

- Train: 2000 rows over 252 unique normalized stems (~8× replication per stem with re-randomized wrappers).
- Test: 500 rows over 228 unique normalized stems. All 228 stems appear in train.
- Identity-option match: 455 / 500 test rows (91%).
- Paraphrased-option residual: 45 / 500 (9%) share a stem with train but have rephrased option text.
- Answer-label consistency: all 222 multi-occurrence train stems have one consistent answer label across duplicates.
- Train label distribution: B 24.50%, C 22.95%, A 18.45%, D 17.90%, E 16.20%.
- Wrapper inventory: 6 prefixes, 4 suffixes; iterative stripping required for compound cases. After full stripping every test stem matches a train stem.

The structural facts reproduce both locally and on the Kaggle runtime — diagnostic cells in `notebooks/kaggle_inference_v1.ipynb` confirm 2000/500 shapes, 252/228 unique stems, 500/500 stem coverage, and 455 identity-match rows.

## 3. Workstream A v1 pipeline

Deterministic. No training.

1. Apply `normalize_stem` (iterative regex strip) to both train and test prompts.
2. Build two structures from train:
   - `train_lookup`: `(stem, frozenset(options))` → correct option text. Handles the 455 identity-match rows.
   - `stem_to_answer`: `stem` → answer label. PH-A fallback for the 45 paraphrase rows.
3. For each test row: if its signature `(stem, frozenset(options))` appears in `train_lookup`, locate the correct option text inside the test row's options and predict its label. Otherwise (paraphrase residual), predict the train stem's answer label directly (PH-A).
4. Top-3 construction: predicted label first, then two filler labels from the global label prior (B, C, A, D, E in that order, skipping the predicted label).
5. Self-check: applying the same lookup to train recovers all 2000 train answer labels (100%).

PH-A justification: on the 45 paraphrase rows, fuzzy text similarity between each test option and the train correct option text agrees with the train stem's answer label in 43 / 45 cases (95.56%). Above the 0.80 threshold to prefer PH-A over PH-B (global prior baseline).

## 4. M1 submission result

- Public LB score: 0.74147.
- Submission file: `submission.csv` written to `/kaggle/working/` by the inference notebook.
- Diagnostic confirmation on Kaggle: shapes, unique-stem counts, identity-match count, and top-1 prediction distribution all match the local analysis. Pipeline correctness verified end-to-end.

## 5. Strategic finding — corrected lookup ceiling

The preliminary findings document predicted that naive lookup with wrapper stripping would score 0.90+ MAP@3 with a competitive ceiling near 0.97–0.99. The LB contradicts this:

- Ranks 1, 2, 3 are tied at 0.76475 — three independent submitters arriving at the same score indicates this is the deterministic lookup ceiling.
- Rank 4: 0.76226. Rank 5: 0.75935.
- Our submission at 0.74147 is approximately 97% of the way to that ceiling.

Most plausible explanation: test answer keys were generated independently of train. For roughly 23–25% of test rows the scorer's correct answer disagrees with the train label even when stem and options match byte-exactly. Pure lookup cannot recover these rows. Full revision in `reports/dataset_findings_revised.md`.

## 6. Pivot for M2

Workstream A is near-optimal at 0.74. Further lookup polish (TTA option permutation, fuzzy paraphrase matching, smarter filler) has bounded upside (~0.02 LB) and is deferred to a later opportunistic pass.

Workstream B is promoted from "course-grade obligation" to "primary LB lever". A trained model that reasons over each question independently can in principle answer correctly on the ~25% rows where the train label is wrong from the scorer's perspective.

M2 plan:

- First Workstream B model: DeBERTa-v3-base with per-option encoding and shared backbone, softmax over 5 option scores.
- Stem-grouped GroupKFold validation (no train/val stem overlap) — val MAP@3 reflects pure inferential ability.
- W&B logging: loss curves, val MAP@3 per fold, learning-rate schedule, hyperparameters, gradient norms.
- The originally planned M2 (paraphrase fuzzy matcher) is demoted to opportunistic polish, executed only if individual Workstream B models converge above 0.74 and an ensemble could plausibly lift further.

## 7. Artifacts

| Artifact | Path |
|----------|------|
| Wrapper-stripping module | `src/normalize.py` |
| Normalize unit tests | `tests/test_normalize.py` |
| EDA notebook | `notebooks/m1_eda.ipynb` |
| Workstream A v1 local notebook | `notebooks/m1_workstream_a_v1.ipynb` |
| Kaggle inference notebook (with diagnostics) | `notebooks/kaggle_inference_v1.ipynb` |
| Revised findings | `reports/dataset_findings_revised.md` |
| This report | `reports/m1_report.md` |

GitHub: `https://github.com/23f2001849/dl-genai-t22026-23f2001849` (main branch, milestone work committed inline).

Kaggle submissions: `https://www.kaggle.com/competitions/smart-mcq-solver-challenge/submissions` (filter user `krishnaanalyst`).

W&B: not used in M1 (Workstream A is deterministic). W&B activation begins in M2 with the first Workstream B training run.

## 8. Open items carried into M2

- Set up the M2 training notebook scaffold (Kaggle GPU or Colab).
- Confirm DeBERTa-v3-base weights source (HuggingFace).
- Decide tokenization strategy under the 512-token window given max option length 662 chars (per-option encoding handles this naturally).
- Establish W&B run naming convention and project structure for Workstream B.
- Stem-grouped GroupKFold split: 5 folds, deterministic seed, fold assignment cached so subsequent models train on identical splits.

None of the above blocks closing M1.