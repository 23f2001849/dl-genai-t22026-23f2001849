---
title: Smart MCQ Solver
emoji: 🔤
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# Smart MCQ Solver

Enter a question and five candidate answers. The model scores each option independently and
returns them ranked, with a confidence for each and the top-3 in the competition's submission
format.

Three models are available:

| Model | Parameters | Pretrained |
|---|---|---|
| TF-IDF + Logistic Regression | 53,432 features | No |
| TextCNN | 2,002,945 | No, trained from scratch |
| BERT-base-uncased | 109,483,009 | Yes |

Weights are pulled from a Hugging Face model repository at startup. Set `MODEL_REPO` in the Space
settings if you fork this.

**A note on what the numbers mean.** In the source competition most test questions also appear in
the training data in paraphrased form, so leaderboard performance partly reflects recognition of
previously seen answer text. On a genuinely novel question typed here, expect weaker performance
than the competition score suggests. BERT is the only one of the three carrying pretrained world
knowledge, so it generalises best to unseen questions.
