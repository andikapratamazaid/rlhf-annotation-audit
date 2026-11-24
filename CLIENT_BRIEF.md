# Client Brief — Preference Data Annotation QA Audit

> **Disclosure.** This is a **simulated client engagement created for portfolio
> purposes**. There is no real client, no real contract, and no confidential
> data involved. The "client" below is a fictionalised composite of the kind
> of AI startup that commissions this work. All analysis is performed on the
> public Anthropic HH-RLHF dataset (MIT license), and every number in the
> deliverables is reproducible from the code in this repository.

## The client (simulated)

An early-stage AI startup is training a general-purpose LLM assistant with
RLHF (reinforcement learning from human feedback). It has collected a large
corpus of human preference pairs: for the same conversation, an annotator
chose one of two assistant responses as the better one. Before the startup
spends compute on reward-model training, it wants an independent QA audit of
that preference data, because a reward model can only learn what the labels
actually say — including their biases and their mistakes.

## Engagement

**Role:** Freelance AI Data Annotation QA Analyst
**Deliverable window:** one audit cycle, reproducible pipeline included

### Questions the client asked

1. **Profile.** What exactly is in the preference corpus — how many pairs per
   split and tranche, and what do the responses look like in length and shape?
2. **Length bias.** Do annotators systematically prefer longer responses? If
   a trivial rule ("always pick the longer answer") performs well, the reward
   model will learn verbosity instead of quality.
3. **Noise.** How much of the corpus carries no signal at all — identical or
   near-identical pairs, duplicates, malformed transcripts?
4. **Harmlessness leakage.** In the harmlessness data, how often does the
   *chosen* response still contain profane or harmful surface content? And
   does the same happen in the helpfulness data?
5. **Learnability.** How well can a transparent baseline model reproduce the
   annotators' preferences on held-out data, per split? Where does it
   disagree, and what does that say about label consistency?
6. **Gold set.** Produce a small, carefully adjudicated gold set (≈200 pairs)
   with an explicit error taxonomy, so the client has a reference sample for
   annotator calibration and future audits.
7. **Guideline.** Turn the findings into a practical annotation guideline the
   client's annotators can actually follow.

### Scope decisions

- Both the **helpfulness** tranches (base, rejection-sampled, online) and the
  **harmlessness** tranche (base) are audited. The harmlessness data contains
  offensive content by design; the audit covers it because that is where
  annotation quality matters most. Statistical detail is reported in full;
  quoted examples are kept to the minimum needed.
- Label quality is assessed with transparent proxies (length statistics, a
  documented lexicon, duplicate detection, a baseline preference model, and
  a single-analyst adjudicated sample). The public dataset contains **no
  annotator IDs**, so true inter-annotator agreement cannot be computed —
  this limitation is stated wherever agreement numbers appear.

## Deliverables produced

| # | Deliverable | Location |
|---|---|---|
| 1 | Dataset profile and length-bias audit | `data/results/split_profile.csv`, `data/results/length_bias.csv` |
| 2 | Noise and leakage audit with documented lexicon | `data/results/noise_counts.csv`, `data/results/leakage_table.csv`, `data/curated/lexicon.csv` |
| 3 | Baseline reward model with held-out accuracy per split | `data/results/model_metrics.csv`, `scripts/04_reward_model.py` |
| 4 | Adjudicated gold set (≈200 pairs) with error taxonomy | `data/results/gold_set.csv`, `data/results/error_taxonomy_counts.csv` |
| 5 | Annotation guideline for helpfulness and harmlessness | `ANNOTATION_GUIDELINE.md` |
| 6 | Executive summary, findings and recommendations | `README.md`, `dashboard/index.html` |

## Out of scope

- No re-annotation of the full corpus; adjudication is a sample-based audit.
- No production reward model; the classifier is a diagnostic baseline.
- No claims about individual annotators (impossible without annotator IDs).
