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

(Filled after EDA — Milestone 1.)

Evaluation metric: TBD from competition page.

---

## Repo Structure

\`\`\`
dl-genai-t22026-23f2001849/
├── notebooks/      # milestone notebooks + final Kaggle inference
├── src/            # modular code: dataset, training, inference, models
├── deployment/     # Streamlit/Gradio app for HF Spaces
├── reports/        # final report and figures
└── data/           # gitignored
\`\`\`

---

## Models

| # | Type | Architecture | Status |
|---|------|--------------|--------|
| 1 | From scratch | TBD | Planned |
| 2 | Pretrained fine-tune | TBD (HF) | Planned |
| 3 | Of choice | TBD | Planned |

---

## Results

(Updated after each milestone.)

| Experiment | Model | Val Score | LB Score |
|-----------|-------|-----------|----------|
| — | — | — | — |

---

## Setup

```bash
git clone https://github.com/<your-username>/dl-genai-t22026-23f2001849.git
cd dl-genai-t22026-23f2001849
pip install -r requirements.txt
```

---

## Tools

- **Frameworks:** PyTorch, HuggingFace Transformers
- **Tracking:** [Weights & Biases](https://wandb.ai/<your-wandb-username>/23f2001849-t22026)
- **Deployment:** Streamlit on HuggingFace Spaces (planned)
- **Training:** Kaggle (T4/P100), Colab

---

## Milestones

- [ ] M0 — Setup (Jun 10)
- [ ] M1 — EDA & baseline (Jun 17)
- [ ] M2 — Classical ML baseline (Jun 24)
- [ ] M3 — Neural network from scratch (Jul 1)
- [ ] M4 — Sequential model (Jul 8)
- [ ] M5 — Pretrained fine-tune (Jul 15)
- [ ] Final submission (Jul 19)
