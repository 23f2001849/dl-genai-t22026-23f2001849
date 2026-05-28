# Dataset Findings — Revised (post-M1 LB result)

This document supersedes the preliminary findings (`dataset_findings_preliminary.md`, in the project context) where the two disagree. The structural facts are reproduced; the strategic claim about the lookup ceiling has been corrected based on direct Kaggle LB evidence from the M1 submission.

## Structural facts (reproduced)

- Train rows: 2000. Test rows: 500. Schema as documented.
- Train answer label distribution: B 24.50%, C 22.95%, A 18.45%, D 17.90%, E 16.20%.
- Prompt avg 117.67 chars; single-option mean 164.95 chars, max 662.
- 252 unique normalized stems in train, 228 in test, all 228 present in train.
- 455 / 500 (91%) test rows have at least one train row with identical option text.
- 45 / 500 (9%) test rows share a stem with train but have paraphrased option text.
- All 222 multi-occurrence train stems have 100% consistent answer labels across duplicates.
- Wrapper inventory: 6 prefixes, 4 suffixes (per `src/normalize.py`). Iterative stripping is required to handle compound wrappers. After full stripping every test stem matches a train stem.

All of the above reproduced on Kaggle (see diagnostic cells in `notebooks/kaggle_inference_v1.ipynb`).

## Corrected claim: lookup ceiling is ~0.76, not 0.90+

The preliminary document stated: *"A naive (stem → answer) lookup with simple wrapper stripping should score in the 0.90+ MAP@3 range on both public and private LB. The competitive ceiling is closer to 0.97–0.99..."*

Kaggle public LB as of the M1 submission shows:
- Ranks 1, 2, 3 tied at 0.76475 (deterministic submissions, separate users).
- Rank 4: 0.76226. Rank 5: 0.75935.
- Our M1 lookup pipeline (PH-A on 45 paraphrase rows + label-prior filler): 0.74147, rank 56 of 112 registered.

Three independent submitters tied at exactly 0.76475 indicates this is the maximum achievable with deterministic lookup-style strategies. The gap between 0.74147 and 0.76475 is residual lookup polish (TTA, fuzzy paraphrase, smarter filler). To exceed 0.76 a submission must reason about each question rather than look it up.

## Most likely explanation for the cap

The competition's test answer keys were generated independently of train. Even when a test row's stem and option text are byte-identical to a train row, the labeled correct answer sometimes disagrees with train.

Rough decomposition from the 0.76 ceiling:
- ~75–77% of test rows have a correct answer that matches the train label for the same (stem, options).
- ~23–25% of test rows have a correct answer that disagrees with the train label, regardless of how cleanly the lookup hits.

Consistent with the answer keys being produced by a separate annotation pass (e.g., a different LLM rerun) over the same question pool. The questions and options are shared between train and test; the labeling decision is not.

## Strategic consequences

Original two-workstream framing assumed:
- Workstream A (lookup): ceiling 0.97–0.99. Trivially wins the LB.
- Workstream B (trained models): mandatory for the course grade. Not competitive on LB.

Revised framing:
- Workstream A: ceiling ~0.76. Already near-optimal at 0.74. Marginal returns from further work.
- Workstream B: trained models that genuinely reason about each question have a structural path to exceed 0.76, because they are not bound to the train label and can output the test-correct answer even where train disagrees. Workstream B is now the primary LB lever, not just a course-grade obligation.

The paraphrase matcher originally scheduled as M2 addresses 45 rows where PH-A is already 95.6% accurate. Best-case lift ~0.005 LB. Demoted from M2 centerpiece to opportunistic polish.

## Updated milestone priorities

- M1: done. Workstream A v1 submitted at 0.74147.
- M2 (pivoted): first Workstream B model — fine-tuned DeBERTa-v3-base with per-option encoding head and shared backbone, stem-grouped GroupKFold validation, W&B logged. Originally scheduled for M3; promoted because of the corrected ceiling.
- M3+: remaining Workstream B models (custom DL, RNN) per Report Guidelines.
- Lookup polish (paraphrase fuzzy matcher, TTA, per-row filler): opportunistic. Ensemble into the inference notebook only if it lifts above the best Workstream B model.

## Verification

- Structural claims: reproducible via `notebooks/m1_eda.ipynb` and diagnostic cells in `notebooks/kaggle_inference_v1.ipynb`.
- LB scores: visible in the Kaggle submission history for `krishnaanalyst` and on the public leaderboard for the competition.
- The 0.76475 ceiling is observable at the top of the leaderboard.