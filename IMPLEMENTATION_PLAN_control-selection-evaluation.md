# Implementation Plan: Control Selection and Paper-Facing Evaluation

**Recommended branch:** `research/control-selection-evaluation`
**Motivation:** May 28 meeting with Joe Near
**Status:** planned

## Goal

Move from open-ended model improvement to paper-facing evaluation and control
selection. Joe's May 28 guidance was that the system is basically working; the
next work should justify which controls are defensible, repair only high-value
controls such as gender, and simplify the path toward the paper.

## Scope

In scope:

- Training-data separability audit for emotion/style labels.
- Evidence-backed selection of top emotion/style controls for the paper/demo.
- Gender-focused CommonVoice follow-up using many more gender-known rows.
- Optional broad-bucket age probe if cheap.
- Explicit accent exclusion for the current OpenVoice speaker-embedding path.
- Paper-methods updates that explain control selection and limitations.

Out of scope:

- More `disgust` repair pressure without stronger perceptual examples.
- More style-strength escalation as a default repair strategy.
- Accent control in the current architecture.
- Long age-control rabbit holes.
- Architecture changes not tied to paper-facing evaluation.

## Workstream 1: Training-Data Separability Audit

Question:

- Which emotion/style labels are separable enough in the training data to be
  fair headline controls?

Tasks:

1. Identify the labeled training clips used for CREMA-D and Expresso style
   supervision.
2. Run the current emotion classifier/evaluator on the original training audio,
   not generated outputs.
3. Compute per-label support, precision, recall, F1, and confusion matrix.
4. Rank labels by separability and support.
5. Produce a Markdown summary that recommends headline controls and labels to
   demote to limitations/future work.

Validation:

- `results/training_style_separability_*.csv` exists with per-row predictions.
- `results/training_style_separability_summary.md` exists with per-label F1 and
  confusion summary.
- The summary explicitly justifies any top-3/top-k style choice.
- `disgust` and `anger` are discussed directly, not buried in aggregate recall.

## Workstream 2: Gender-Focused CommonVoice Follow-Up

Question:

- Can gender control become perceptually clear enough to support the paper/demo?

Tasks:

1. Extract a larger local CommonVoice subset with known gender metadata.
2. Report gender-known row counts before training.
3. Train/evaluate gender-only control first, without emotion interaction.
4. Test gender plus only the top separable emotion/style controls from
   Workstream 1.
5. If gender remains unclear, run one bounded latent-dimension comparison to
   check whether the current low-dimensional VAE is capacity-limited.
6. Build a small listening panel for gender-only and gender-plus-style outputs.

Validation:

- Gender-known CommonVoice counts are written to a metadata report.
- A gender-only listening panel exists.
- A gender-plus-top-style listening panel exists if gender-only is plausible.
- Metadata separability or a lightweight classifier result is reported.
- The failure mode is categorized as data shortage, control interaction, or
  latent-capacity limitation if gender still fails.

## Workstream 3: Age and Accent Scope Cleanup

Question:

- How do we prevent weak metadata controls from muddying the paper story?

Tasks:

1. Mark accent out of scope for the current OpenVoice speaker-embedding VAE,
   because accent is likely carried by content representation rather than
   speaker embedding.
2. Reframe age as optional broad-bucket classification rather than continuous
   scalar control.
3. If time permits, test broad age buckets only after gender and top-style
   selection are complete.

Validation:

- Docs say accent is out of scope and explain why.
- Docs say age is low priority and broad-bucket only.
- No headline claim depends on age or accent unless a later validated result
  supports it.

## Workstream 4: Paper Simplification

Question:

- What do we need to write the paper cleanly from current evidence?

Tasks:

1. Update paper-methods docs around the current reference checkpoint.
2. Add a control-selection rationale section.
3. Add a limitations section for weak labels, accent, age, formal DP accounting,
   and final EER/privacy validation.
4. Keep the narrative simple: OpenVoice embeddings, controllable VAE, selected
   controls, objective metrics, human listening, and privacy/identity-shift
   evidence.

Validation:

- The paper outline has a concise claim and evaluation table plan.
- Every headline control has an evidence-backed justification.
- Every excluded/deprioritized control has a defensible reason.
- The worklog points to the updated paper/evaluation artifacts.

## Acceptance

- The next two-week work produces either a defensible top-control set or a clear
  reason why additional controls should stay out of the paper.
- Gender has a bounded, evidence-backed status: working, not working because of
  data, not working because of control interaction, or not working because of
  capacity.
- `disgust` is no longer pursued as a repair target unless the training-data
  audit contradicts Joe's perception and shows strong separability.
- The repo docs explain why accent is out of scope and age is low priority.
- The paper path is simpler after the work, not more complicated.

## Future Upgrades To Preserve

- Add a larger human perceptual study only after the headline controls are
  selected.
- Add repeated-seed confidence intervals before freezing final tables.
- Add independent labeled speaker-verification trials for final EER/privacy
  claims.
- Add formal DP accounting and privacy-utility curves before submission.
