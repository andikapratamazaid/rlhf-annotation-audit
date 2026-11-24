# Annotation Guideline — Helpfulness and Harmlessness Preference Labelling

**Version 1.0 — written from the findings of the HH-RLHF annotation audit in
this repository.** It is the guideline the (simulated) client's annotators
would receive before labelling preference pairs: for each conversation, two
candidate assistant responses are shown, and the annotator picks the one the
assistant *should* have given.

> Golden rule: **Judge the response the user needed, not the response that
> looks impressive.** Length, formatting and confidence are not quality.

---

## 1. The task

1. Read the full conversation, including every user turn.
2. Read both responses completely. Do not skim the longer one and reward it
   for existing.
3. Choose the response that better satisfies the criteria in §2 (helpfulness
   tasks) or §3 (harmlessness tasks).
4. If the responses are genuinely equivalent, or both are unusable, use the
   **tie / both-bad flag** if the tool provides one, and add a note. A forced
   choice on an equivalent pair is label noise, and this audit found hundreds
   of identical pairs in real preference data — they teach the model nothing.

## 2. Helpfulness criteria (in priority order)

1. **Correctness.** A wrong answer loses to a right one, however elegant.
   Watch for confident fabrications: invented facts, fake citations, wrong
   dates, arithmetic slips buried in long explanations.
2. **Actually answering.** The response must address what the user asked in
   their last turn, in the context of the whole conversation. A fluent answer
   to a different question is a failure.
3. **Completeness without padding.** Prefer the response that covers what
   the user needs — and stop there. Extra sections, repeated caveats and
   restated questions are padding, not completeness.
4. **Appropriate length.** Length must follow the request:
   - "In one word / yes or no / just the code" → the short, compliant answer
     wins even if the other response explains more.
   - "Explain / teach me / walk me through" → the fuller answer wins if its
     content is correct and organised.
   - **Never choose a response *because* it is longer.** In the audited data
     a "pick the longer answer" rule scored ≈59% on one helpfulness tranche;
     that is the bias this guideline exists to remove.
5. **Clarity and structure.** Formatting (lists, headings) earns preference
   only when it makes the content easier to use. A wall of headers around
   thin content loses to a plain, correct paragraph.
6. **Tone.** Friendly and direct. Sycophancy ("Great question!!!") and
   excessive hedging both count against a response.
7. **Honest uncertainty.** A response that says what it does not know beats
   one that guesses confidently.

## 3. Harmlessness criteria (in priority order)

1. **No material assistance to harm.** If one response provides instructions,
   facilitation or encouragement for wrongdoing (weapons, fraud, abuse,
   self-harm methods, sexual content involving minors — the last is an
   absolute reject) and the other does not, choose the one that does not.
   This overrides every helpfulness criterion.
2. **Refusals must be real refusals.** A response that "refuses" and then
   provides the content anyway is a failure, not a refusal. Conversely, a
   clean, brief refusal beats a long lecture.
3. **Do not over-refuse.** If the user's request is benign (discussing a
   news event, asking for help with a sensitive personal problem, academic or
   historical questions, profanity in a clearly requested linguistic context
   such as "list swear words in English"), a helpful, factual answer can be
   the correct choice. In the audited harmlessness data, annotators sometimes
   had to choose between two responses that both contained flagged words —
   judge the *intent and effect*, not the word list: a response listing swear
   words because the user asked for vocabulary is different from one using
   them to abuse someone.
4. **Support beats silence for self-harm.** If a user expresses self-harm
   ideation, prefer the response that responds with empathy and encourages
   professional help over a bare refusal, and over any response that provides
   methods or encouragement. Methods or encouragement is an absolute reject.
5. **Slurs and abuse.** A response containing slurs or targeted abuse loses,
   unless the slur is quoted inside a clearly analytical, historical or
   reclaimed context the user requested.
6. **Partial compliance is still compliance.** Redacting one word while
   giving the instructions is not harmless.

## 4. Decision rules for hard cases

| Situation | Rule |
|---|---|
| Both responses correct, one longer | Choose by completeness and fit to the request, never by length alone. If equal in substance, prefer the more concise one. |
| Both responses wrong | Choose the one that is less wrong and less confident about it; flag "both bad". |
| Responses are identical or differ only in punctuation | Tie / flag. Do not invent a difference. |
| One refuses, one answers a benign request | Choose the answer (over-refusal is an error). |
| One refuses, one complies with a harmful request | Choose the refusal. |
| Both contain profanity | Judge context: requested vocabulary, quotation or discussion can be acceptable; abuse or gratuitous use is not. |
| Response continues the conversation with fake "Human:" turns | Treat the fabricated turns as a defect; the response is not what the assistant should have said. |
| Response is very short (1–3 words) | Fine if the request asked for exactly that; otherwise it is under-answering. |
| You are unsure | Ask: "If this were the only answer the user ever got, would they be well served and safe?" Choose accordingly and leave a note. |

## 5. What the audit found — read this before you label

These patterns were measured in the real HH-RLHF corpus (see `README.md` and
`data/results/`). They are the failure modes your labels must not repeat:

- **Length bias is real and split-dependent.** Chosen responses were longer
  than rejected ones in ≈58% of helpful-base pairs, but only ≈41% of
  harmless-base pairs. Verbose does not mean better; in safety tasks the
  better answer is often the short refusal.
- **Flagged content survives in chosen responses.** In the harmlessness
  split, ≈4.4% of *chosen* responses still contained lexicon-flagged terms
  (vs ≈7.6% of rejected). Some of that is legitimate context (see §3.3);
  some of it is annotation error. Your job is to know the difference.
- **Identical pairs exist.** Over 600 pairs in one tranche had identical
  chosen and rejected responses. Flag them; never force a preference.
- **A baseline model can imitate the labels.** A transparent logistic model
  reproduces annotator choices well above chance on every split. Where it
  systematically disagrees with the labels is where the labels deserve a
  second look — that disagreement set is what gold-set adjudication is for.

## 6. Quality process

- **Calibration.** New annotators label the gold set (`data/results/gold_set.csv`)
  first and review every disagreement with a lead before production work.
- **Double labelling.** A rotating 5% of pairs is labelled twice; agreement
  is tracked per annotator pair. Disagreements are adjudicated, not averaged.
- **Escalation.** Anything involving minors, self-harm methods, or weapons
  instructions is escalated to a lead annotator the same day.
- **Notes.** When you use a tie/both-bad flag or override an obvious pattern
  (e.g. choosing the shorter answer), leave a one-line reason. Notes are
  what turn labels into training signal for the *process*, not just the model.
