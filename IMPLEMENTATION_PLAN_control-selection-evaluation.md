# Implementation Plan: Control Selection and Paper-Facing Evaluation

**Recommended branch:** `research/control-selection-evaluation`
**Motivation:** May 28 meeting with Joe Near
**Status:** in progress — Workstream 1 implemented on `research/control-selection-evaluation`; Workstream 2 implemented on `research/control-shortlist`

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

- [x] Identify the labeled training clips used for CREMA-D and Expresso style
   supervision.
- [x] Run the current emotion classifier/evaluator on the original training audio,
   not generated outputs.
- [x] Compute per-label support, precision, recall, F1, and confusion matrix.
- [x] Rank labels by separability and support.
- [x] Produce a Markdown summary that recommends headline controls and labels to
   demote to limitations/future work.

Validation:

- [x] `results/training_style_separability_*.csv` exists with per-row predictions.
- [x] `results/training_style_separability_summary.md` exists with per-label F1 and
  confusion summary.
- [x] The summary explicitly justifies separable, quality-sensitive, and weak
  source-label groups.
- [x] `disgust` and `anger` are discussed directly, not buried in aggregate recall.

Result:

- CREMA-D emotion labels are source-separable (`0.7692-0.9451` direct recall).
- `confused` is weak (`0.2703` embedding F1).
- `enunciated` and `whisper` are supported but quality-sensitive (`0.5000` and
  `0.5161` embedding F1).
- Source separability does not promote a control by itself; the next workstream
  must intersect these results with generated-output metrics and listening
  evidence.

## Workstream 2: Control Shortlist and Paper Claim Selection

Question:

- Which controls belong in the headline paper/demo story?

Tasks:

- [x] Combine source separability with the current reference guard's generated
  emotion, WER, MOS, external-novelty, and collapse metrics.
- [x] Add an explicit perceptual-evidence ledger so Joe/Stephen listening notes
  are machine-readable input to the recommendation.
- [x] Categorize controls into headline, candidate-headline-pending-listening,
  supported-but-quality-sensitive, and diagnostic/limitation buckets.
- [x] Create a paper-facing control-selection table and Markdown
  recommendation.

Validation:

- [x] `results/control_selection_recommendation.csv` exists with one row per
  style and the source/generated/perceptual gates.
- [x] `results/control_selection_recommendation.md` exists with bucket counts,
  interpretation, and the next listening queue.
- [x] No style is promoted as fully paper-ready without positive focused
  listening evidence.
- [x] Every excluded/deprioritized control has a defensible reason.

Result:

- `neutral` and `sad` are candidate headline controls pending focused
  listening.
- `happy`, `enunciated`, and `whisper` are supported but quality-sensitive.
- `anger`, `confused`, `disgust`, and `fear` are diagnostic or limitation
  controls under current evidence.
- The recommendation reduces the active paper claim set and blocks more
  training for weak hard-style labels until listening evidence changes.

## Workstream 3: Gender-Focused CommonVoice Follow-Up

Question:

- Can gender control become perceptually clear enough to support the paper/demo?

Tasks:

1. Extract a larger local CommonVoice subset with known gender metadata.
2. Report gender-known row counts before training.
3. Train/evaluate gender-only control first, without emotion interaction.
4. Test gender plus only the shortlisted controls from Workstream 2.
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

## Workstream 4: Age and Accent Scope Cleanup

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

## Workstream 5: Paper Simplification

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
- `disgust` is no longer pursued as a repair target unless generated-output
  listening evidence contradicts Joe's current neutral-perception review; source
  separability alone is not enough.
- The repo docs explain why accent is out of scope and age is low priority.
- The paper path is simpler after the work, not more complicated.

## Future Upgrades To Preserve

- Add a larger human perceptual study only after the headline controls are
  selected.
- Add repeated-seed confidence intervals before freezing final tables.
- Add independent labeled speaker-verification trials for final EER/privacy
  claims.
- Add formal DP accounting and privacy-utility curves before submission.
- Add a quieter shared emotion2vec runner for long generated-output and
  source-audit jobs; the current FunASR progress output is reproducible but too
  noisy for collaborator-facing logs.
