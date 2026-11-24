"""Consistency tests for the RLHF annotation audit deliverables.

These tests validate the produced result tables against each other and
against the download manifest; they do not re-run the full pipeline.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results"
SPLITS = ["helpful-base", "helpful-online", "helpful-rejection-sampled", "harmless-base"]


@pytest.fixture(scope="module")
def profile():
    return pd.read_csv(RESULTS / "split_profile.csv")


@pytest.fixture(scope="module")
def manifest():
    return json.loads((ROOT / "data" / "raw" / "download_manifest.json").read_text())


def test_profile_matches_manifest(profile, manifest):
    expected = {(m["split"], m["kind"]): m["pairs"] for m in manifest}
    assert len(expected) == 8
    for _, r in profile.iterrows():
        assert r["pairs"] == expected[(r["split"], r["kind"])]


def test_total_pairs(profile):
    assert profile["pairs"].sum() == 169_352


def test_length_bias_columns_sum():
    bias = pd.read_csv(RESULTS / "length_bias.csv")
    assert set(bias["split"]) == set(SPLITS)
    total = (bias["pct_chosen_longer"] + bias["pct_equal_length"]
             + bias["pct_chosen_shorter"])
    assert (total - 100).abs().max() < 0.01


def test_length_bias_direction():
    bias = pd.read_csv(RESULTS / "length_bias.csv").set_index("split")
    # Helpfulness tranches reward length; harmlessness prefers shorter.
    assert bias.loc["helpful-base", "pct_chosen_longer"] > 55
    assert bias.loc["harmless-base", "pct_chosen_longer"] < 45
    # The trivial rule must beat chance where bias exists and fail on harmless.
    assert bias.loc["helpful-base", "length_only_baseline_accuracy_pct"] > 55
    assert bias.loc["harmless-base", "length_only_baseline_accuracy_pct"] < 50


def test_leakage_harmless_ordering():
    leak = pd.read_csv(RESULTS / "leakage_table.csv")
    any_rows = leak[leak["category"] == "any"].set_index("split")
    # Annotators reduced flagged content in chosen responses on harmlessness.
    assert (any_rows.loc["harmless-base", "chosen_flagged_pct"]
            < any_rows.loc["harmless-base", "rejected_flagged_pct"])
    # ...but did not eliminate it.
    assert any_rows.loc["harmless-base", "chosen_flagged_pct"] > 3.0


def test_noise_identical_pairs():
    noise = pd.read_csv(RESULTS / "noise_counts.csv")
    ident = noise[noise["signal"] == "identical_pair"].set_index("split")
    assert ident.loc["helpful-rejection-sampled", "count"] == 667
    assert ident["count"].sum() == 776


def test_model_metrics():
    m = pd.read_csv(RESULTS / "model_metrics.csv")
    assert set(m["split"]) == set(SPLITS)
    # The full model must beat the trivial length rule on every split and
    # stay in the plausible band for a linear preference model.
    assert (m["model_full_accuracy_pct"] > m["length_only_accuracy_pct"]).all()
    assert m["model_full_accuracy_pct"].between(52, 75).all()
    assert (m["model_full_accuracy_pct"] >= m["model_engineered_accuracy_pct"] - 0.01).all()


def test_gold_set_shape_and_labels():
    gold = pd.read_csv(RESULTS / "gold_set.csv")
    assert len(gold) == 200
    assert gold.groupby("split").size().eq(50).all()
    assert set(gold["auditor_label"]) <= {"agree", "disagree", "invalid"}
    assert set(gold["error_category"]) >= {
        "none", "identical_pair", "harmful_content_in_chosen",
        "refusal_over_helpful", "length_bias_suspect", "ambiguous_pair",
    }
    assert gold["rationale"].str.len().min() > 20


def test_taxonomy_sums_to_gold_set():
    tax = pd.read_csv(RESULTS / "error_taxonomy_counts.csv")
    assert tax["pairs"].sum() == 200


def test_lexicon_structure():
    lex = pd.read_csv(ROOT / "data" / "curated" / "lexicon.csv")
    assert {"term", "category", "notes"} <= set(lex.columns)
    assert len(lex) >= 50
    assert {"profanity", "slur", "self_harm", "violence"} <= set(lex["category"])


def test_summary_consistency(profile):
    summary = json.loads((RESULTS / "summary.json").read_text())
    assert summary["total_pairs"] == int(profile["pairs"].sum())
    assert summary["gold_set_pairs"] == 200
    assert summary["transformer_reward_model"]["status"] == "not_run"
    assert 90 <= summary["audit_agreement_decisive_pct"] <= 100


def test_dashboard_exists_and_warns():
    html = (ROOT / "dashboard" / "index.html").read_text()
    assert "Content warning" in html
    assert "simulated" in html.lower()
