#!/usr/bin/env python3
"""Step 06: charts for the README and dashboard. All charts are drawn from
the result tables produced by steps 02-05; nothing is hand-entered."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROCESSED, RESULTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CHARTS = ROOT / "charts"
SHORT = {
    "helpful-base": "helpful\nbase",
    "helpful-online": "helpful\nonline",
    "helpful-rejection-sampled": "helpful\nrejection-sampled",
    "harmless-base": "harmless\nbase",
}
COLORS = ["#2563eb", "#059669", "#d97706", "#dc2626"]


def main() -> int:
    CHARTS.mkdir(parents=True, exist_ok=True)
    bias = pd.read_csv(RESULTS / "length_bias.csv")
    leak = pd.read_csv(RESULTS / "leakage_table.csv")
    metrics = pd.read_csv(RESULTS / "model_metrics.csv")
    tax = pd.read_csv(RESULTS / "error_taxonomy_counts.csv")
    feats = pd.read_csv(PROCESSED / "pairs_features.csv.gz")

    # 1. Chosen-longer rate by split.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([SHORT[s] for s in bias["split"]], bias["pct_chosen_longer"],
           color=COLORS)
    ax.axhline(50, color="grey", linestyle="--", linewidth=1)
    ax.text(3.45, 51, "50% = no length preference", ha="right", fontsize=8, color="grey")
    for i, v in enumerate(bias["pct_chosen_longer"]):
        ax.text(i, v + 1.2, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("% of pairs where chosen response is longer")
    ax.set_title("Length bias: annotators prefer the longer response\nin helpfulness tranches, the shorter one in harmlessness")
    ax.set_ylim(0, 68)
    fig.tight_layout(); fig.savefig(CHARTS / "chosen_longer_by_split.png", dpi=150); plt.close(fig)

    # 2. Length-difference distribution per split.
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
    for ax, (split, color) in zip(axes.flat, zip(bias["split"], COLORS)):
        d = feats[feats["split"] == split]["len_diff_words"].clip(-200, 200)
        ax.hist(d, bins=40, color=color, alpha=0.85)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(split, fontsize=10)
        ax.set_xlabel("chosen words − rejected words")
    fig.suptitle("Distribution of response-length differences per pair")
    fig.tight_layout(); fig.savefig(CHARTS / "length_diff_distribution.png", dpi=150); plt.close(fig)

    # 3. Flagged-term rate, chosen vs rejected (any category).
    any_leak = leak[leak["category"] == "any"].set_index("split").loc[bias["split"]]
    x = np.arange(len(any_leak)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - w / 2, any_leak["chosen_flagged_pct"], w, label="chosen", color="#2563eb")
    ax.bar(x + w / 2, any_leak["rejected_flagged_pct"], w, label="rejected", color="#9ca3af")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[s] for s in any_leak.index])
    ax.set_ylabel("% of responses containing ≥1 lexicon-flagged term")
    ax.set_title("Label leakage: flagged terms survive in chosen responses\n(lexicon proxy — not ground truth about harm)")
    ax.legend()
    fig.tight_layout(); fig.savefig(CHARTS / "flagged_rate_chosen_vs_rejected.png", dpi=150); plt.close(fig)

    # 4. Harmless-base flagged rate by category.
    hb = leak[(leak["split"] == "harmless-base") & (leak["category"] != "any")]
    x = np.arange(len(hb)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, hb["chosen_flagged_pct"], w, label="chosen", color="#dc2626")
    ax.bar(x + w / 2, hb["rejected_flagged_pct"], w, label="rejected", color="#9ca3af")
    ax.set_xticks(x); ax.set_xticklabels(hb["category"], rotation=20, ha="right")
    ax.set_ylabel("% of responses flagged")
    ax.set_title("Harmlessness split: flagged-term rate by lexicon category")
    ax.legend()
    fig.tight_layout(); fig.savefig(CHARTS / "flagged_by_category_harmless.png", dpi=150); plt.close(fig)

    # 5. Model accuracy by split.
    m = metrics.set_index("split").loc[bias["split"]]
    x = np.arange(len(m)); w = 0.27
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w, m["model_full_accuracy_pct"], w, label="reward model (engineered + TF-IDF)", color="#2563eb")
    ax.bar(x, m["model_engineered_accuracy_pct"], w, label="engineered features only", color="#059669")
    ax.bar(x + w, m["length_only_accuracy_pct"], w, label="'always pick longer' rule", color="#d97706")
    ax.axhline(50, color="grey", linestyle="--", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[s] for s in m.index])
    ax.set_ylabel("held-out test accuracy (%)")
    ax.set_ylim(0, 78)
    ax.set_title("How learnable are the labels? Held-out accuracy per split")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(CHARTS / "model_accuracy_by_split.png", dpi=150); plt.close(fig)

    # 6. Error taxonomy counts.
    t = tax.sort_values("pairs")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh(t["error_category"], t["pairs"],
            color=["#9ca3af" if c == "none" else "#2563eb" for c in t["error_category"]])
    for i, v in enumerate(t["pairs"]):
        ax.text(v + 2, i, str(v), va="center", fontsize=9)
    ax.set_xlabel("pairs in 200-pair gold set")
    ax.set_title("Gold-set error taxonomy (single-analyst adjudication)")
    ax.set_xlim(0, t["pairs"].max() * 1.15)
    fig.tight_layout(); fig.savefig(CHARTS / "taxonomy_counts.png", dpi=150); plt.close(fig)

    print("Charts written:", sorted(p.name for p in CHARTS.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
