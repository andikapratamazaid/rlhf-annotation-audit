#!/usr/bin/env python3
"""Step 02: dataset profile and length-bias audit.

Outputs:
  data/results/split_profile.csv   - pair counts and length distributions
  data/results/length_bias.csv     - chosen-longer rates and the accuracy of
                                     the trivial "always pick the longer
                                     response" rule, per split
  data/processed/pairs_features.csv.gz - one row per pair with length features
                                     (input for steps 03 and 06)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KINDS, SPLITS, ensure_dirs, load_pairs, word_count  # noqa: E402


def main() -> int:
    ensure_dirs()
    rows = []
    profile_rows = []
    for split in SPLITS:
        for kind in KINDS:
            feats = []
            for p in load_pairs(split, kind):
                cw, rw = word_count(p["chosen"]), word_count(p["rejected"])
                feats.append(
                    {
                        "split": split,
                        "kind": kind,
                        "idx": p["idx"],
                        "prompt_words": word_count(p["prompt"]),
                        "prompt_mismatch": p["prompt_mismatch"],
                        "chosen_words": cw,
                        "rejected_words": rw,
                        "len_diff_words": cw - rw,
                        "len_ratio": cw / rw if rw else np.nan,
                        "chosen_longer": cw > rw,
                        "equal_length": cw == rw,
                    }
                )
            df = pd.DataFrame(feats)
            rows.append(df)
            profile_rows.append(
                {
                    "split": split,
                    "kind": kind,
                    "pairs": len(df),
                    "prompt_mismatch_pairs": int(df["prompt_mismatch"].sum()),
                    "prompt_words_mean": df["prompt_words"].mean(),
                    "prompt_words_median": df["prompt_words"].median(),
                    "chosen_words_mean": df["chosen_words"].mean(),
                    "chosen_words_median": df["chosen_words"].median(),
                    "chosen_words_p10": df["chosen_words"].quantile(0.10),
                    "chosen_words_p90": df["chosen_words"].quantile(0.90),
                    "rejected_words_mean": df["rejected_words"].mean(),
                    "rejected_words_median": df["rejected_words"].median(),
                    "rejected_words_p10": df["rejected_words"].quantile(0.10),
                    "rejected_words_p90": df["rejected_words"].quantile(0.90),
                }
            )
            print(f"{split}/{kind}: {len(df):,} pairs")
    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(
        Path(__file__).resolve().parent.parent
        / "data" / "processed" / "pairs_features.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(profile_rows).to_csv(
        Path(__file__).resolve().parent.parent / "data" / "results" / "split_profile.csv",
        index=False,
    )

    # Length bias per split (train + test combined, as published splits).
    bias_rows = []
    for split in SPLITS:
        df = all_df[all_df["split"] == split]
        pct_longer = df["chosen_longer"].mean() * 100
        pct_equal = df["equal_length"].mean() * 100
        # "Always pick the longer response" rule: correct when chosen is the
        # longer one; an exact tie is scored as half a correct guess.
        baseline_acc = (df["chosen_longer"].sum() + 0.5 * df["equal_length"].sum()) / len(df) * 100
        bias_rows.append(
            {
                "split": split,
                "pairs": len(df),
                "pct_chosen_longer": pct_longer,
                "pct_equal_length": pct_equal,
                "pct_chosen_shorter": 100 - pct_longer - pct_equal,
                "mean_len_diff_words": df["len_diff_words"].mean(),
                "median_len_diff_words": df["len_diff_words"].median(),
                "len_diff_p10": df["len_diff_words"].quantile(0.10),
                "len_diff_p90": df["len_diff_words"].quantile(0.90),
                "length_only_baseline_accuracy_pct": baseline_acc,
            }
        )
    bias = pd.DataFrame(bias_rows)
    bias.to_csv(
        Path(__file__).resolve().parent.parent / "data" / "results" / "length_bias.csv",
        index=False,
    )
    print("\nLength bias summary:")
    print(bias.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
