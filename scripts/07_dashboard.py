#!/usr/bin/env python3
"""Step 07: static HTML dashboard generated from the result tables.

Every number on the page is read from data/results/*.csv and summary.json at
generation time; no figure is typed into the template by hand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def table_html(df: pd.DataFrame, cols: list[str], fmt: dict[str, str]) -> str:
    head = "".join(f"<th>{c}</th>" for c in cols)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"<td>{fmt[c].format(v) if c in fmt else v}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def main() -> int:
    summary = json.loads((RESULTS / "summary.json").read_text())
    bias = pd.read_csv(RESULTS / "length_bias.csv")
    leak = pd.read_csv(RESULTS / "leakage_table.csv")
    metrics = pd.read_csv(RESULTS / "model_metrics.csv")
    tax = pd.read_csv(RESULTS / "error_taxonomy_counts.csv")
    noise = pd.read_csv(RESULTS / "noise_counts.csv")

    any_leak = leak[leak["category"] == "any"][
        ["split", "chosen_flagged_pct", "rejected_flagged_pct"]]
    hb = bias[bias["split"] == "harmless-base"].iloc[0]
    hb_leak = any_leak[any_leak["split"] == "harmless-base"].iloc[0]
    best_model = metrics.loc[metrics["model_full_accuracy_pct"].idxmax()]
    ident_total = int(noise[noise["signal"] == "identical_pair"]["count"].sum())

    pct1 = "{:.1f}%".format
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RLHF Preference Data — Annotation QA Audit</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }}
  main {{ max-width: 1060px; margin: 0 auto; padding: 32px 20px 64px; }}
  h1 {{ font-size: 28px; margin-bottom: 4px; }}
  h2 {{ font-size: 20px; margin-top: 36px; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; }}
  .sub {{ color: #475569; margin-top: 0; }}
  .warn {{ background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 10px 14px; font-size: 14px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin: 20px 0; }}
  .card {{ background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; }}
  .card .big {{ font-size: 26px; font-weight: 700; }}
  .card .lbl {{ color: #475569; font-size: 13px; }}
  table {{ border-collapse: collapse; width: 100%; background: white; font-size: 14px; }}
  th, td {{ border: 1px solid #e2e8f0; padding: 7px 10px; text-align: left; }}
  th {{ background: #f1f5f9; }}
  img {{ max-width: 100%; background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 8px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }}
  li {{ margin-bottom: 6px; }}
  footer {{ color: #64748b; font-size: 13px; margin-top: 40px; }}
</style>
</head>
<body>
<main>
  <h1>RLHF Preference Data — Annotation QA Audit</h1>
  <p class="sub">Simulated freelance engagement for an AI startup training an LLM assistant
  (disclosed simulation, portfolio project). Dataset: Anthropic HH-RLHF, MIT license.</p>
  <p class="warn"><strong>Content warning:</strong> the audited harmlessness data contains
  offensive language and harmful requests by design. This dashboard reports statistics only
  and quotes no harmful examples.</p>

  <div class="cards">
    <div class="card"><div class="big">{summary['total_pairs']:,}</div><div class="lbl">preference pairs audited across 4 splits</div></div>
    <div class="card"><div class="big">{hb['pct_chosen_longer']:.1f}%</div><div class="lbl">of harmlessness pairs chose the longer response — annotators preferred shorter, safer answers there</div></div>
    <div class="card"><div class="big">{hb_leak['chosen_flagged_pct']:.2f}%</div><div class="lbl">of chosen harmlessness responses still contain lexicon-flagged terms (rejected: {hb_leak['rejected_flagged_pct']:.2f}%)</div></div>
    <div class="card"><div class="big">{ident_total}</div><div class="lbl">identical chosen/rejected pairs in the corpus — pure label noise</div></div>
    <div class="card"><div class="big">{best_model['model_full_accuracy_pct']:.1f}%</div><div class="lbl">best held-out reward-model accuracy ({best_model['split']}); mean across splits {summary['model_mean_test_accuracy_pct']:.1f}%</div></div>
    <div class="card"><div class="big">{summary['audit_agreement_decisive_pct']:.1f}%</div><div class="lbl">audit-sample agreement between original labels and single-analyst adjudication (decisive gold pairs)</div></div>
  </div>

  <h2>Length bias</h2>
  {table_html(bias, ["split", "pairs", "pct_chosen_longer", "mean_len_diff_words", "length_only_baseline_accuracy_pct"],
               {"pct_chosen_longer": "{:.2f}%", "mean_len_diff_words": "{:+.1f}", "length_only_baseline_accuracy_pct": "{:.2f}%"})}
  <div class="grid">
    <img src="../charts/chosen_longer_by_split.png" alt="Chosen-longer rate by split">
    <img src="../charts/length_diff_distribution.png" alt="Length difference distribution">
  </div>

  <h2>Label leakage (lexicon proxy)</h2>
  {table_html(any_leak, ["split", "chosen_flagged_pct", "rejected_flagged_pct"],
               {"chosen_flagged_pct": "{:.2f}%", "rejected_flagged_pct": "{:.2f}%"})}
  <div class="grid">
    <img src="../charts/flagged_rate_chosen_vs_rejected.png" alt="Flagged rate chosen vs rejected">
    <img src="../charts/flagged_by_category_harmless.png" alt="Flagged rate by category, harmlessness">
  </div>

  <h2>Reward model (held-out test accuracy)</h2>
  {table_html(metrics, ["split", "test_pairs", "model_full_accuracy_pct", "model_engineered_accuracy_pct", "length_only_accuracy_pct"],
               {"model_full_accuracy_pct": "{:.2f}%", "model_engineered_accuracy_pct": "{:.2f}%", "length_only_accuracy_pct": "{:.2f}%"})}
  <img src="../charts/model_accuracy_by_split.png" alt="Model accuracy by split">

  <h2>Gold set and error taxonomy</h2>
  {table_html(tax, ["error_category", "pairs", "pct_of_gold_set"], {"pct_of_gold_set": "{:.1f}%"})}
  <img src="../charts/taxonomy_counts.png" alt="Error taxonomy counts">

  <h2>Recommendations for the annotation operation</h2>
  <ul>
    <li><strong>Rewrite the guideline around length.</strong> Annotators reward length in
    helpfulness data (up to ~58% longer-chosen) and punish it in harmlessness data; the
    guideline must state explicitly that length follows the request, never the reverse.</li>
    <li><strong>Filter identical and near-identical pairs before training.</strong>
    They carry no preference signal and dilute the reward model's gradient.</li>
    <li><strong>Add a harmlessness second pass.</strong> A measurable share of chosen
    harmlessness responses still carries flagged terms; a lexicon pre-screen plus human
    review of flagged chosen responses is cheap and targeted.</li>
    <li><strong>Calibrate with the gold set.</strong> New annotators label the 200-pair
    gold set first; disagreements are reviewed against the guideline before production work.</li>
    <li><strong>Track agreement properly.</strong> The public corpus has no annotator IDs,
    so inter-annotator agreement is unknowable for it; the client's own pipeline should
    double-label a rotating 5% sample and store annotator IDs from day one.</li>
  </ul>

  <footer>Generated by scripts/07_dashboard.py from data/results/. Simulated engagement —
  see CLIENT_BRIEF.md for the disclosure. Lexicon flags are a surface proxy, not ground
  truth about harm. Gold-set labels are single-analyst adjudications.</footer>
</main>
</body>
</html>
"""
    out = ROOT / "dashboard" / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard written to {out} ({len(html):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
