"""
app.py — Smart MCQ Solver demo for Hugging Face Spaces.

Enter a question and five options; the app ranks them and shows a confidence for each,
plus the top-3 string in the competition's submission format.

Models are loaded lazily, so the Space starts instantly and only pays the BERT load cost
if someone actually selects BERT.
"""

import os, json
import numpy as np
import gradio as gr
import joblib
import spaces
from scipy.sparse import hstack, csr_matrix

from featurise import pair_features, OPT

# NEW block
@spaces.GPU
def _zero_gpu_stub():
    return None

# Pull weights from the Hub if they are not already on disk. No-op when artifacts/ exists.
try:
    from download_models import ensure_artifacts
    ensure_artifacts()
except Exception as exc:  # noqa: BLE001
    print(f"weight download skipped: {exc}")

ART = "artifacts"
HAS_TEXTCNN = os.path.exists(f"{ART}/textcnn.pt")
HAS_BERT    = os.path.exists(f"{ART}/bert_ranker/model.safetensors")

_cache = {}

# --------------------------------------------------------------------- loaders
def load_lr():
    if "lr" not in _cache:
        _cache["lr"] = joblib.load(f"{ART}/lr_model.joblib")
    return _cache["lr"]

def _torch():
    import torch
    return torch

def load_textcnn():
    if "cnn" not in _cache:
        import torch, torch.nn as nn
        from transformers import AutoTokenizer

        class TextCNN(nn.Module):
            def __init__(self, vocab_size, embed_dim, n_filters, kernels, dropout, pad_id):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
                self.convs = nn.ModuleList([nn.Conv1d(embed_dim, n_filters, k, padding=k // 2)
                                            for k in kernels])
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(n_filters * len(kernels), 1)
            def forward(self, input_ids, attention_mask=None):
                emb = self.embedding(input_ids).transpose(1, 2)
                pooled = [torch.relu(c(emb)).max(dim=2).values for c in self.convs]
                return self.fc(self.dropout(torch.cat(pooled, dim=1))).squeeze(1)

        ckpt = torch.load(f"{ART}/textcnn.pt", map_location="cpu")
        model = TextCNN(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"]); model.eval()
        tok = AutoTokenizer.from_pretrained(ckpt["tokenizer"])
        _cache["cnn"] = (model, tok, ckpt["max_len"])
    return _cache["cnn"]

def load_bert():
    if "bert" not in _cache:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(f"{ART}/bert_ranker")
        model = AutoModelForSequenceClassification.from_pretrained(f"{ART}/bert_ranker")
        model.eval()
        # dynamic int8 quantisation: roughly 4x smaller in memory, 2-3x faster on CPU
        try:
            qd = getattr(torch.ao.quantization, "quantize_dynamic", None) \
                 or torch.quantization.quantize_dynamic
            model = qd(model, {torch.nn.Linear}, dtype=torch.qint8)
        except Exception as exc:  # quantisation is an optimisation, not a requirement
            print(f"int8 quantisation unavailable, using float32: {exc}")
        _cache["bert"] = (model, tok, 128)
    return _cache["bert"]

# --------------------------------------------------------------------- scoring
def score_lr(prompt, options):
    art = load_lr()
    feats = [pair_features(prompt, options[L]) for L in OPT]
    texts = [f["text"] for f in feats]
    nums  = np.array([[f[c] for c in art["num_cols"]] for f in feats], dtype=np.float32)
    X = hstack([art["word"].transform(texts), art["char"].transform(texts),
                csr_matrix(nums)]).tocsr()
    return art["clf"].predict_proba(X)[:, 1]

def _score_neural(model, tok, max_len, prompt, options):
    torch = _torch()
    texts = [pair_features(prompt, options[L])["text"] for L in OPT]
    enc = tok(texts, truncation=True, padding="max_length",
              max_length=max_len, return_tensors="pt")
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        logits = out.logits.squeeze(-1) if hasattr(out, "logits") else out
        return torch.sigmoid(logits).numpy()

def score_textcnn(prompt, options):
    return _score_neural(*load_textcnn(), prompt, options)

def score_bert(prompt, options):
    return _score_neural(*load_bert(), prompt, options)

SCORERS = {"TF-IDF + Logistic Regression": score_lr}
if HAS_TEXTCNN: SCORERS["TextCNN (from scratch)"] = score_textcnn
if HAS_BERT:    SCORERS["BERT-base-uncased"]      = score_bert

# --------------------------------------------------------------------- app
def predict(model_name, prompt, a, b, c, d, e):
    options = dict(zip(OPT, [a, b, c, d, e]))
    if not prompt.strip() or any(not v.strip() for v in options.values()):
        return "", "Enter a question and all five options.", None

    raw = np.asarray(SCORERS[model_name](prompt, options), dtype=float)
    # normalise across the five siblings so the numbers read as a distribution over options
    conf = raw / raw.sum() if raw.sum() > 0 else np.full(5, 0.2)
    order = np.argsort(-raw)

    top3 = " ".join(OPT[i] for i in order[:3])
    verdict = (f"### Predicted answer: **{OPT[order[0]]}**\n"
               f"Confidence {conf[order[0]]:.1%} &nbsp;•&nbsp; "
               f"submission format: `{top3}`")

    rows = []
    for rank, i in enumerate(order, start=1):
        marker = {1: "1st", 2: "2nd", 3: "3rd"}.get(rank, f"{rank}th")
        text = options[OPT[i]]
        rows.append([marker, OPT[i], f"{conf[i]:.1%}", f"{raw[i]:.4f}",
                     text if len(text) <= 90 else text[:87] + "..."])
    return verdict, "", rows

EXAMPLE = dict(
    prompt="Why is it nearly impossible to see light emitted at the Lyman-alpha transition "
           "wavelength from a star more than a few hundred light years from Earth?",
    A="Far ultraviolet light is scattered by dust grains in the interstellar medium, which have a "
      "typical scattering wavelength of about 121.5 nanometers.",
    B="Far ultraviolet light is absorbed effectively by the neutral components of the ISM, "
      "including atomic hydrogen, which has a typical absorption wavelength of about 121.5 "
      "nanometers, the Lyman-alpha transition.",
    C="Far ultraviolet light is absorbed effectively by the charged components of the ISM, "
      "including atomic hydrogen, which has a typical absorption wavelength of about 121.5 "
      "nanometers, the Lyman-alpha transition.",
    D="Far ultraviolet light is absorbed effectively by the neutral components of the ISM, "
      "including atomic helium, which has a typical absorption wavelength of about 121.5 "
      "nanometers, the Lyman-alpha transition.",
    E="Far ultraviolet light is absorbed effectively by the neutral components of the ISM, "
      "including atomic hydrogen, which has a typical absorption wavelength of about 212.5 "
      "nanometers, the Lyman-alpha transition.",
)

CSS = """
.gradio-container {max-width: 1000px !important}
#verdict {font-size: 1.05rem}
"""

with gr.Blocks(title="Smart MCQ Solver", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown(
        "# Smart MCQ Solver\n"
        "Enter a question and five candidate answers. The model scores each option "
        "independently and returns them ranked, with a confidence for each.\n\n"
        "*Three models are available: a TF-IDF and Logistic Regression baseline, a TextCNN "
        "trained from scratch, and a fine-tuned BERT ranker. The example below is a real "
        "question whose distractors differ from the correct answer by a single word or digit.*"
    )

    model_name = gr.Dropdown(list(SCORERS), value=list(SCORERS)[0], label="Model")
    prompt = gr.Textbox(label="Question", lines=3, placeholder="Type the question...")
    with gr.Row():
        a = gr.Textbox(label="Option A", lines=2)
        b = gr.Textbox(label="Option B", lines=2)
    with gr.Row():
        c = gr.Textbox(label="Option C", lines=2)
        d = gr.Textbox(label="Option D", lines=2)
    e = gr.Textbox(label="Option E", lines=2)

    with gr.Row():
        go = gr.Button("Rank the options", variant="primary")
        fill = gr.Button("Load example")

    verdict = gr.Markdown(elem_id="verdict")
    warn = gr.Markdown()
    table = gr.Dataframe(
        headers=["Rank", "Option", "Confidence", "Raw score", "Answer text"],
        datatype=["str", "str", "str", "str", "str"],
        label="Ranking", wrap=True, interactive=False,
    )

    go.click(predict, [model_name, prompt, a, b, c, d, e], [verdict, warn, table])
    fill.click(lambda: [EXAMPLE["prompt"], EXAMPLE["A"], EXAMPLE["B"],
                        EXAMPLE["C"], EXAMPLE["D"], EXAMPLE["E"]],
               outputs=[prompt, a, b, c, d, e])

if __name__ == "__main__":
    demo.launch()
