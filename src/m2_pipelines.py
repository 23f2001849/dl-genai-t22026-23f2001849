from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import torch

OPTS = ("A", "B", "C", "D", "E")


def load_train_dataset(train_csv: str):
    from datasets import load_dataset
    ds = load_dataset("csv", data_files=train_csv, split="train")
    return ds.map(lambda x: {"combined_text": x["prompt"] + " " + x["A"]})


def get_bert_tokenizer(name: str = "bert-base-uncased"):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def encode_prompts_batch(prompts: list[str], tokenizer, max_length: int = 128):
    return tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def get_bert_model(name: str = "bert-base-uncased", output_attentions: bool = False):
    from transformers import AutoModel
    m = AutoModel.from_pretrained(name, output_attentions=output_attentions)
    m.eval()
    return m


def get_sentence_transformer(name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def minilm_rank_top3(df, st_model, batch_size: int = 64) -> list[list[str]]:
    from sentence_transformers import util
    prompt_embs = st_model.encode(df["prompt"].tolist(), batch_size=batch_size,
                                  convert_to_tensor=True, show_progress_bar=True)
    option_embs = {c: st_model.encode(df[c].tolist(), batch_size=batch_size,
                                      convert_to_tensor=True, show_progress_bar=True)
                   for c in OPTS}
    preds = []
    for i in range(len(df)):
        sims = np.array([util.cos_sim(prompt_embs[i], option_embs[c][i]).item()
                         for c in OPTS])
        order = np.argsort(-sims)[:3]
        preds.append([OPTS[j] for j in order])
    return preds


def tfidf_rank_top3(df) -> list[list[str]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    preds = []
    for i in range(len(df)):
        row = df.iloc[i]
        texts = [row["prompt"]] + [row[c] for c in OPTS]
        vec = TfidfVectorizer().fit(texts)
        mat = vec.transform(texts)
        sims = cosine_similarity(mat[0:1], mat[1:]).ravel()
        order = np.argsort(-sims)[:3]
        preds.append([OPTS[j] for j in order])
    return preds


def map_at_k(preds: list[list[str]], truths: Iterable[str], k: int = 3) -> float:
    score = 0.0
    truths = list(truths)
    for p, t in zip(preds, truths):
        for r, pred in enumerate(p[:k]):
            if pred == t:
                score += 1.0 / (r + 1)
                break
    return score / len(preds)


def zero_shot_scores(prompt: str, candidates: list[str], multi_label: bool = False):
    from transformers import pipeline
    zsc = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return zsc(prompt, candidate_labels=candidates, multi_label=multi_label)


def flan_t5_letter_qa(prompt: str, opt_a: str, opt_b: str,
                      model_name: str = "google/flan-t5-small",
                      max_new_tokens: int = 5) -> str:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained(model_name)
    m = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    m.eval()
    formatted = (
        f"Question: {prompt}. Is the correct answer "
        f"A: {opt_a} or B: {opt_b}? "
        f"Answer with just the letter A or B."
    )
    inputs = tok(formatted, return_tensors="pt", truncation=True)
    with torch.no_grad():
        out_ids = m.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, num_beams=1)
    return tok.decode(out_ids[0], skip_special_tokens=True)
