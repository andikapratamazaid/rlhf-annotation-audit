#!/usr/bin/env python3
"""Step 04: reward-model (preference classifier) training and evaluation.

For each split, a logistic-regression preference model is trained on that
split's official train file and evaluated on its official test file. The
model scores a pair from the *difference* of per-response features
(Bradley-Terry style): P(chosen preferred) = sigmoid(w . (f_chosen - f_rejected)).

Feature blocks:
  A. Engineered features per response: length (words, chars, log words),
     lexical diversity, sentence / question / list structure, refusal and
     apology markers, first-person usage, and lexicon hit counts from
     step 03 (data/curated/lexicon.csv categories).
  B. TF-IDF difference features over the response text (max 3,000 features).

Reported per split on the held-out test set:
  - full model accuracy (A + B)
  - engineered-only model accuracy (A)
  - the trivial length-only rule from step 02, recomputed on the test set

A small transformer fine-tune (e.g. DistilBERT) was considered and NOT run:
this environment provides ~1.5 shared CPUs and 6 GB RAM, and fine-tuning on
tens of thousands of pairs would take many hours. The decision and its
rationale are recorded in data/results/summary.json (step 05) and the README
rather than replaced with an invented number.

Outputs:
  data/results/model_metrics.csv
  data/processed/test_predictions.csv.gz  (per test pair: model correctness)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROCESSED, RESULTS, SPLITS, ensure_dirs, load_pairs  # noqa: E402

REFUSAL_PHRASES = [
    "i can't", "i cannot", "i'm not able", "i am not able", "i won't",
    "i will not", "i'm sorry", "sorry, but", "i apologize", "i apologise",
    "can't help", "cannot help", "not able to help", "i must decline",
    "i don't feel comfortable", "i do not feel comfortable",
]
LEX_CATS = ["profanity", "slur", "sexual_explicit", "sexual_violence",
            "self_harm", "violence", "any"]
_LIST_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s", re.M)
_FIRST_PERSON = re.compile(r"\b(?:i|i'm|i've|i'll|i'd|my|me)\b")
_SENTENCE = re.compile(r"[.!?]+")


def engineered(resp: str, lex_counts: dict[str, int]) -> list[float]:
    low = resp.lower()
    words = resp.split()
    n_words = len(words)
    uniq = len(set(w.lower() for w in words)) / n_words if n_words else 0.0
    return [
        n_words,
        len(resp),
        np.log1p(n_words),
        (sum(len(w) for w in words) / n_words) if n_words else 0.0,
        uniq,
        len(_SENTENCE.findall(resp)),
        resp.count("?"),
        resp.count("!"),
        len(_LIST_LINE.findall(resp)),
        float(any(p in low for p in REFUSAL_PHRASES)),
        float("sorry" in low),
        len(_FIRST_PERSON.findall(low)),
        *[float(lex_counts.get(c, 0)) for c in LEX_CATS],
    ]


def load_flags() -> pd.DataFrame:
    flags = pd.read_csv(PROCESSED / "pair_flags.csv.gz")
    return flags.set_index(["split", "kind", "idx"])


def pair_arrays(split: str, kind: str, flags: pd.DataFrame):
    texts_c, texts_r, dense_c, dense_r, idxs = [], [], [], [], []
    for p in load_pairs(split, kind):
        row = flags.loc[(split, kind, p["idx"])]
        lc = {c: int(row[f"chosen_{c}"]) for c in LEX_CATS}
        lr = {c: int(row[f"rejected_{c}"]) for c in LEX_CATS}
        texts_c.append(p["chosen"])
        texts_r.append(p["rejected"])
        dense_c.append(engineered(p["chosen"], lc))
        dense_r.append(engineered(p["rejected"], lr))
        idxs.append(p["idx"])
    return texts_c, texts_r, np.array(dense_c), np.array(dense_r), idxs


def fit_eval(split: str, flags: pd.DataFrame, use_tfidf: bool):
    tc_tr, tr_tr, dc_tr, dr_tr, _ = pair_arrays(split, "train", flags)
    tc_te, tr_te, dc_te, dr_te, idx_te = pair_arrays(split, "test", flags)

    # NOTE: difference features must be scaled but NOT centred. Centring a
    # difference feature removes exactly the mean shift that carries the
    # Bradley-Terry signal, and symmetrised logistic regression then
    # collapses to the constant 0.5 model (verified empirically in this
    # project: centred features yielded all-zero coefficients).
    scaler = StandardScaler(with_mean=False).fit(dc_tr - dr_tr)
    d_tr = scaler.transform(dc_tr - dr_tr)
    d_te = scaler.transform(dc_te - dr_te)

    if use_tfidf:
        vec = TfidfVectorizer(max_features=3000, sublinear_tf=True, min_df=3)
        all_train = tc_tr + tr_tr
        vec.fit(all_train)
        x_tr = sparse.hstack([sparse.csr_matrix(d_tr),
                              vec.transform(tc_tr) - vec.transform(tr_tr)]).tocsr()
        x_te = sparse.hstack([sparse.csr_matrix(d_te),
                              vec.transform(tc_te) - vec.transform(tr_te)]).tocsr()
    else:
        x_tr, x_te = d_tr, d_te

    # Symmetrised training set: (diff, 1) and (-diff, 0).
    X = sparse.vstack([x_tr, -x_tr]) if sparse.issparse(x_tr) else np.vstack([x_tr, -x_tr])
    y = np.array([1] * x_tr.shape[0] + [0] * x_tr.shape[0])
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(X, y)
    prob = clf.predict_proba(x_te)[:, 1]
    # Chosen is always the true label. An exact 0.5 probability (a model that
    # learned nothing) must score as half correct, never as a free win.
    acc = float((prob > 0.5).mean() + 0.5 * (prob == 0.5).mean())
    return acc, idx_te, prob


def main() -> int:
    ensure_dirs()
    flags = load_flags()
    feats = pd.read_csv(PROCESSED / "pairs_features.csv.gz")
    rows = []
    pred_frames = []
    for split in SPLITS:
        acc_full, idx_te, prob = fit_eval(split, flags, use_tfidf=True)
        acc_eng, _, _ = fit_eval(split, flags, use_tfidf=False)
        te = feats[(feats["split"] == split) & (feats["kind"] == "test")]
        length_acc = (te["chosen_longer"].sum() + 0.5 * te["equal_length"].sum()) / len(te) * 100
        rows.append(
            {
                "split": split,
                "test_pairs": len(te),
                "model_full_accuracy_pct": acc_full * 100,
                "model_engineered_accuracy_pct": acc_eng * 100,
                "length_only_accuracy_pct": length_acc,
            }
        )
        pred_frames.append(
            pd.DataFrame(
                {
                    "split": split,
                    "kind": "test",
                    "idx": idx_te,
                    "model_prob_chosen": prob,
                    "model_correct": prob > 0.5,
                }
            )
        )
        print(f"{split}: full={acc_full*100:.2f}%  engineered={acc_eng*100:.2f}%  "
              f"length-only={length_acc:.2f}%  (n={len(te):,})")
    pd.DataFrame(rows).to_csv(RESULTS / "model_metrics.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(
        PROCESSED / "test_predictions.csv.gz", index=False, compression="gzip"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
