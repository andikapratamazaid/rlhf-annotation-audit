#!/usr/bin/env python3
"""Step 03: annotation-noise signals and harmlessness label-leakage audit.

Noise signals (per split):
  identical_pair        chosen and rejected final responses are identical
  near_identical_pair   word-set Jaccard similarity >= 0.95 (not identical)
  duplicate_pair        the same (prompt, chosen, rejected) row appears more
                        than once in the published split
  role_bleed            final response contains a further Human:/Assistant:
                        turn marker, i.e. the transcript did not end cleanly
  tiny_response         chosen or rejected final response has <= 2 words

Leakage audit: the lexicon in data/curated/lexicon.csv is a transparent proxy
for profane / harmful surface content. It is NOT ground truth about harm:
many flagged terms appear in benign, medical, news or support contexts, and
much genuinely harmful content contains no flagged term at all. The audit
question is narrower: how often does the response the annotator *chose*
still carry flagged surface terms, compared with the rejected response?

Outputs:
  data/results/leakage_table.csv  (long format, per split x category)
  data/results/noise_counts.csv
  data/processed/pair_flags.csv.gz (per-pair lexicon counts for step 04/05)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    KINDS,
    PROCESSED,
    RESULTS,
    SPLITS,
    ensure_dirs,
    load_pairs,
    normalise,
)

ROOT = Path(__file__).resolve().parent.parent


def build_patterns() -> dict[str, re.Pattern]:
    lex = pd.read_csv(ROOT / "data" / "curated" / "lexicon.csv")
    patterns: dict[str, re.Pattern] = {}
    for cat, grp in lex.groupby("category"):
        terms = [re.escape(t.strip().lower()) for t in grp["term"]]
        patterns[cat] = re.compile(r"\b(?:" + "|".join(terms) + r")\b")
    all_terms = [re.escape(t.strip().lower()) for t in lex["term"]]
    patterns["any"] = re.compile(r"\b(?:" + "|".join(all_terms) + r")\b")
    return patterns


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def main() -> int:
    ensure_dirs()
    patterns = build_patterns()
    categories = list(patterns.keys())

    leakage_rows = []
    noise_rows = []
    flag_rows = []

    for split in SPLITS:
        seen_pairs: set[tuple[str, str, str]] = set()
        stats = {c: {"c_flag": 0, "r_flag": 0, "c_terms": 0, "r_terms": 0} for c in categories}
        noise = {
            "identical_pair": 0,
            "near_identical_pair": 0,
            "duplicate_pair": 0,
            "role_bleed": 0,
            "tiny_response": 0,
        }
        n = 0
        for kind in KINDS:
            for p in load_pairs(split, kind):
                n += 1
                chosen_low = p["chosen"].lower()
                rejected_low = p["rejected"].lower()

                counts = {}
                for cat, pat in patterns.items():
                    c_hits = len(pat.findall(chosen_low))
                    r_hits = len(pat.findall(rejected_low))
                    counts[cat] = (c_hits, r_hits)
                    s = stats[cat]
                    s["c_flag"] += c_hits > 0
                    s["r_flag"] += r_hits > 0
                    s["c_terms"] += c_hits
                    s["r_terms"] += r_hits
                flag_rows.append(
                    {
                        "split": split,
                        "kind": kind,
                        "idx": p["idx"],
                        **{f"chosen_{c}": counts[c][0] for c in categories},
                        **{f"rejected_{c}": counts[c][1] for c in categories},
                    }
                )

                # --- noise signals ---
                if p["chosen"] == p["rejected"]:
                    noise["identical_pair"] += 1
                elif jaccard(set(p["chosen"].split()), set(p["rejected"].split())) >= 0.95:
                    noise["near_identical_pair"] += 1
                key = (normalise(p["prompt"]), normalise(p["chosen"]), normalise(p["rejected"]))
                if key in seen_pairs:
                    noise["duplicate_pair"] += 1
                else:
                    seen_pairs.add(key)
                if "\n\nHuman:" in p["chosen"] or "\n\nHuman:" in p["rejected"]:
                    noise["role_bleed"] += 1
                if len(p["chosen"].split()) <= 2 or len(p["rejected"].split()) <= 2:
                    noise["tiny_response"] += 1

        for cat in categories:
            s = stats[cat]
            leakage_rows.append(
                {
                    "split": split,
                    "category": cat,
                    "pairs": n,
                    "chosen_flagged_pairs": s["c_flag"],
                    "chosen_flagged_pct": s["c_flag"] / n * 100,
                    "rejected_flagged_pairs": s["r_flag"],
                    "rejected_flagged_pct": s["r_flag"] / n * 100,
                    "chosen_mean_terms": s["c_terms"] / n,
                    "rejected_mean_terms": s["r_terms"] / n,
                }
            )
        for signal, count in noise.items():
            noise_rows.append(
                {"split": split, "signal": signal, "pairs": n,
                 "count": count, "pct": count / n * 100}
            )
        print(f"{split}: {n:,} pairs audited")

    pd.DataFrame(leakage_rows).to_csv(RESULTS / "leakage_table.csv", index=False)
    pd.DataFrame(noise_rows).to_csv(RESULTS / "noise_counts.csv", index=False)
    pd.DataFrame(flag_rows).to_csv(
        PROCESSED / "pair_flags.csv.gz", index=False, compression="gzip"
    )

    leak = pd.DataFrame(leakage_rows)
    print("\nFlagged-term rate ('any' category), chosen vs rejected:")
    print(
        leak[leak["category"] == "any"][
            ["split", "chosen_flagged_pct", "rejected_flagged_pct"]
        ].to_string(index=False)
    )
    print("\nNoise signals:")
    print(pd.DataFrame(noise_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
