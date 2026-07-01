# Smart MCQ Solver — DL & GenAI Project (T2-2026)

Part of the **May 2026 Deep Learning & Generative AI Project** at IIT Madras.

- **Name:** Krishna
- **Roll No:** 23f2001849
- **Term:** May–Sept 2026
- **Kaggle Competition:** [Smart MCQ Solver Challenge](https://www.kaggle.com/competitions/smart-mcq-solver-challenge)
- **Kaggle Notebook:** `DL-23f2001849-notebook-t22026`
- **W&B Project:** `23f2001849-t22026`

---

## Task

Multiple-choice question answering. 5 options per question (A–E), one correct.

Evaluation metric: MAP@3. Train: 2000 rows. Test: 500 rows.

---

## Repo Structure

```
dl-genai-t22026-23f2001849/
├── notebooks/      # milestone notebooks + final Kaggle inference
├── src/            # modular code: dataset, training, inference, models
├── deployment/     # Streamlit/Gradio app for HF Spaces
├── reports/        # final report and figures
└── data/           # gitignored
```

---

## Models

| # | Type | Architecture | Status |
|---|------|--------------|--------|
| 1 | From scratch | TBD | Planned |
| 2 | Pretrained fine-tune | TBD (HF) | Planned |
| 3 | Of choice | TBD | Planned |

---

## Results

| Approach | LB (MAP@3) |
|----------|------------|
| 5-model ensemble + 4-tier lookup + rank-2 paraphrase rule | 0.76517 |

---

## Setup

```bash
git clone https://github.com/23f2001849/dl-genai-t22026-23f2001849.git
cd dl-genai-t22026-23f2001849
pip install -r requirements.txt
```

---

## Tools

- **Frameworks:** PyTorch, HuggingFace Transformers
- **Tracking:** [Weights & Biases](https://wandb.ai/23f2001849/23f2001849-t22026)
- **Deployment:** Streamlit on HuggingFace Spaces (planned)
- **Training:** Kaggle (T4/P100), Colab

---

## Milestones

- [x] M0 — Setup
- [x] M1 — EDA & baseline
- [ ] M2 — Classical ML baseline
- [ ] M3 — Neural network from scratch
- [ ] M4 — Sequential model
- [ ] M5 — Pretrained fine-tune
- [ ] Final submission
