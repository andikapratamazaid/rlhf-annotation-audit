#!/usr/bin/env python3
"""Step 05: adjudicated gold set, error taxonomy, and project summary.

A stratified sample of ~200 pairs (50 per split) is drawn across splits and
signal strata (random, model disagreements, length extremes, flagged chosen
responses, identical / tiny pairs). Each sampled pair is adjudicated by the
analyst against ANNOTATION_GUIDELINE.md. For reproducibility the adjudication
is operationalised as the documented decision rules in `adjudicate()` below,
applied to measured signals (length difference, lexicon hits, refusal
markers, model probability, text similarity). The rationale stored for every
pair cites that pair's actual measured signals.

These are SINGLE-ANALYST adjudication labels. The public HH-RLHF release has
no annotator IDs, so true inter-annotator agreement cannot be computed; what
is reported is (a) agreement between the original labels and this audit
sample, and (b) agreement between the original labels and the step-04 model.

Outputs:
  data/results/gold_set.csv
  data/results/error_taxonomy_counts.csv
  data/results/summary.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    PROCESSED,
    RESULTS,
    SPLITS,
    load_pairs,
)

SEED = 42
REFUSAL = re.compile(
    r"\b(i can't|i cannot|i'm not able|i am not able|i won't|i will not|"
    r"i'm sorry|i apologize|i apologise|can't help|cannot help)\b"
)
SEVERE_CATS = ["slur", "sexual_violence", "self_harm"]


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def adjudicate(row: pd.Series) -> tuple[str, str, str]:
    """Return (auditor_label, error_category, rationale) for one pair."""
    sim = row["similarity"]
    if row["chosen"] == row["rejected"]:
        return ("invalid", "identical_pair",
                "Chosen and rejected responses are textually identical; the "
                "pair carries no preference information (guideline section 1.4).")
    if sim >= 0.95:
        return ("invalid", "near_identical_pair",
                f"Word-set similarity is {sim:.2f}; the responses differ only "
                "trivially, so any preference between them is noise.")
    # Severe = slur + sexual-violence terms. Self-harm terms are excluded
    # here on purpose: in a chosen response they can belong to a supportive
    # reply, which the lexicon cannot distinguish from encouragement.
    severe_chosen = sum(row[f"chosen_{c}"] for c in ["slur", "sexual_violence"])
    severe_rejected = sum(row[f"rejected_{c}"] for c in ["slur", "sexual_violence"])
    if (row["split"] == "harmless-base" and severe_chosen > 0
            and severe_rejected == 0):
        return ("disagree", "harmful_content_in_chosen",
                f"Chosen response contains {severe_chosen} severe flagged "
                "term(s) (slur / sexual-violence lexicon) while the rejected "
                "response contains none; in a harmlessness task the response "
                "carrying slurs or sexual-violence terms should not win "
                "(guideline sections 3.1 and 3.5). Lexicon proxy caveat: "
                "quoted or analytical use cannot be ruled out automatically.")
    if (row["split"] != "harmless-base" and row["chosen_refusal"]
            and row["rejected_words"] >= 30 and row["rejected_any"] == 0):
        return ("disagree", "refusal_over_helpful",
                "Chosen response is a refusal while the rejected response is "
                "a substantive answer (>=30 words) with no flagged content; "
                "guideline section 3.3 counts over-refusal as an error.")
    if (row["split"] != "harmless-base" and row["chosen_words"] <= 5
            and row["rejected_words"] >= 40 and row["rejected_any"] == 0):
        return ("disagree", "under_answer_chosen",
                f"Chosen response has only {int(row['chosen_words'])} words "
                f"against a substantive {int(row['rejected_words'])}-word "
                "alternative with no flagged content; guideline section 4 "
                "counts this as under-answering outside safety tasks.")
    prob = row.get("model_prob_chosen")
    if (row["len_diff_words"] >= 50 and row["chosen_any"] == 0
            and row["rejected_any"] == 0 and row["rejected_words"] >= 40):
        # Two-tier flag: the pair is length-bias-suspect on its face. The
        # label is overturned only when the audit model independently
        # prefers the shorter rejected response; otherwise the label is
        # retained and the pair is flagged for calibration review. (The
        # model itself has learned some length preference, so its agreement
        # cannot exonerate a pair on its own.)
        if pd.notna(prob) and prob < 0.45:
            return ("disagree", "length_bias_suspect",
                    f"Chosen response is {int(row['len_diff_words'])} words "
                    f"longer, neither side has flagged content, and the audit "
                    f"model prefers the rejected response (p(chosen)={prob:.2f}); "
                    "consistent with length being rewarded over substance "
                    "(guideline section 2.4).")
        return ("agree", "length_bias_suspect",
                f"Chosen response is {int(row['len_diff_words'])} words "
                "longer than a substantive rejected response with no flagged "
                "content on either side; the label is retained, but the pair "
                "is flagged as length-bias-suspect for calibration review "
                "(guideline section 2.4).")
    if pd.notna(prob) and 0.45 <= prob <= 0.55 and abs(row["len_diff_words"]) <= 5:
        return ("agree", "ambiguous_pair",
                "Near-tie pair: model probability is close to 0.5 and the "
                "length difference is minimal; the original preference is "
                "retained but the pair is a hard, low-margin example.")
    return ("agree", "none",
            "No audit rule fires: no severe flagged content asymmetry, no "
            "over-refusal pattern, and no extreme length gap.")


def main() -> int:
    rng = np.random.default_rng(SEED)
    feats = pd.read_csv(PROCESSED / "pairs_features.csv.gz")
    flags = pd.read_csv(PROCESSED / "pair_flags.csv.gz")
    preds = pd.read_csv(PROCESSED / "test_predictions.csv.gz")
    df = feats.merge(flags, on=["split", "kind", "idx"], how="left")
    # Predictions exist only for test pairs; merge on split+idx and keep the
    # values only on test rows so train rows (same idx range) stay empty.
    df = df.merge(
        preds.rename(columns={"idx": "idx_t"})[
            ["split", "idx_t", "model_prob_chosen", "model_correct"]
        ],
        left_on=["split", "idx"], right_on=["split", "idx_t"], how="left",
    )
    df.loc[df["kind"] == "train", ["model_prob_chosen", "model_correct"]] = np.nan
    df = df.drop(columns=["idx_t"])

    sampled_keys: list[tuple[str, str, int, str]] = []

    def take(frame: pd.DataFrame, n: int, stratum: str, split: str) -> None:
        nonlocal sampled_keys
        have = {(s, k, i) for s, k, i, _ in sampled_keys if s == split}
        cand = frame[~frame.apply(lambda r: (r["split"], r["kind"], int(r["idx"])) in have, axis=1)]
        if len(cand) == 0 or n <= 0:
            return
        pick = cand.sample(n=min(n, len(cand)), random_state=int(rng.integers(1 << 31)))
        for _, r in pick.iterrows():
            sampled_keys.append((split, r["kind"], int(r["idx"]), stratum))

    for split in SPLITS:
        sdf = df[df["split"] == split]
        test = sdf[sdf["kind"] == "test"]
        train = sdf[sdf["kind"] == "train"]
        take(test, 15, "random_test", split)
        take(train, 10, "random_train", split)
        disagree = test[test["model_correct"] == False].sort_values("model_prob_chosen")  # noqa: E712
        take(disagree, 8, "model_disagreement", split)
        extreme = sdf.reindex(sdf["len_diff_words"].abs().sort_values(ascending=False).index)
        take(extreme, 7, "length_extreme", split)
        flagged = sdf[sdf["chosen_any"] > 0].sort_values("chosen_any", ascending=False)
        take(flagged, 5, "flagged_chosen", split)
        tiny = sdf[(sdf["chosen_words"] <= 2) | (sdf["rejected_words"] <= 2)]
        take(tiny, 5, "tiny_response", split)
        # top up to 50 with uniform random pairs
        have = {(s, k, i) for s, k, i, _ in sampled_keys if s == split}
        if len(have) < 50:
            take(sdf, 50 - len(have), "random_topup", split)

    # Fetch texts for sampled pairs.
    wanted: dict[tuple[str, str], set[int]] = {}
    for s, k, i, _ in sampled_keys:
        wanted.setdefault((s, k), set()).add(i)
    texts: dict[tuple[str, str, int], dict] = {}
    for (s, k), idxs in wanted.items():
        for p in load_pairs(s, k):
            if p["idx"] in idxs:
                texts[(s, k, p["idx"])] = p

    base = df.set_index(["split", "kind", "idx"])
    gold_rows = []
    for s, k, i, stratum in sampled_keys:
        r = base.loc[(s, k, i)]
        p = texts[(s, k, i)]
        row = {
            "chosen": p["chosen"], "rejected": p["rejected"],
            "chosen_words": r["chosen_words"], "rejected_words": r["rejected_words"],
            "len_diff_words": r["len_diff_words"],
            "chosen_any": r["chosen_any"], "rejected_any": r["rejected_any"],
            "model_prob_chosen": r["model_prob_chosen"],
            "split": s,
            "chosen_refusal": bool(REFUSAL.search(p["chosen"].lower())),
            "rejected_refusal": bool(REFUSAL.search(p["rejected"].lower())),
            "similarity": jaccard(set(p["chosen"].split()), set(p["rejected"].split())),
            **{f"chosen_{c}": r[f"chosen_{c}"] for c in SEVERE_CATS},
            **{f"rejected_{c}": r[f"rejected_{c}"] for c in SEVERE_CATS},
        }
        label, category, rationale = adjudicate(pd.Series(row))
        gold_rows.append(
            {
                "split": s, "kind": k, "idx": i, "stratum": stratum,
                "prompt_excerpt": p["prompt"][-300:].replace("\n", " "),
                "chosen_excerpt": p["chosen"][:300].replace("\n", " "),
                "rejected_excerpt": p["rejected"][:300].replace("\n", " "),
                "chosen_words": int(r["chosen_words"]),
                "rejected_words": int(r["rejected_words"]),
                "len_diff_words": int(r["len_diff_words"]),
                "chosen_flagged_terms": int(r["chosen_any"]),
                "rejected_flagged_terms": int(r["rejected_any"]),
                "model_prob_chosen": (None if pd.isna(r["model_prob_chosen"])
                                      else round(float(r["model_prob_chosen"]), 3)),
                "auditor_label": label,
                "error_category": category,
                "rationale": rationale,
            }
        )
    gold = pd.DataFrame(gold_rows)
    gold.to_csv(RESULTS / "gold_set.csv", index=False)

    tax = (gold.groupby("error_category")
           .size().reset_index(name="pairs")
           .sort_values("pairs", ascending=False))
    tax["pct_of_gold_set"] = tax["pairs"] / len(gold) * 100
    tax.to_csv(RESULTS / "error_taxonomy_counts.csv", index=False)

    decisive = gold[gold["auditor_label"].isin(["agree", "disagree"])]
    audit_agreement = (decisive["auditor_label"] == "agree").mean() * 100
    metrics = pd.read_csv(RESULTS / "model_metrics.csv")
    model_agreement = float(metrics["model_full_accuracy_pct"].mean())

    profile = pd.read_csv(RESULTS / "split_profile.csv")
    summary = {
        "dataset": "Anthropic HH-RLHF (MIT license), official GitHub release",
        "total_pairs": int(profile["pairs"].sum()),
        "pairs_per_split": {
            s: int(profile[profile["split"] == s]["pairs"].sum()) for s in SPLITS
        },
        "gold_set_pairs": int(len(gold)),
        "audit_agreement_decisive_pct": round(float(audit_agreement), 2),
        "audit_invalid_pairs": int((gold["auditor_label"] == "invalid").sum()),
        "model_mean_test_accuracy_pct": round(model_agreement, 2),
        "transformer_reward_model": {
            "status": "not_run",
            "reason": ("Environment provides ~1.5 shared CPUs and 6 GB RAM; "
                       "installing a transformer stack (~2 GB) and fine-tuning "
                       "on tens of thousands of pairs would take many hours. "
                       "The guaranteed deliverable (logistic preference model "
                       "on engineered + TF-IDF features) was run instead; no "
                       "transformer number is reported or implied."),
        },
        "limitations": [
            "No annotator IDs in the public release: true inter-annotator "
            "agreement cannot be computed.",
            "Gold-set labels are single-analyst adjudications operationalised "
            "through documented rules, not independent human re-annotation.",
            "The lexicon is a surface proxy for harmful content, not ground "
            "truth about harm.",
            "Word counts use whitespace tokenisation, not a model tokenizer.",
        ],
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Gold set: {len(gold)} pairs")
    print(gold["auditor_label"].value_counts().to_string())
    print(tax.to_string(index=False))
    print(f"Audit agreement on decisive pairs: {audit_agreement:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
