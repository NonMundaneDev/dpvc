# Research Plan: Control Selection and Paper-Facing Evaluation

**Recommended branch:** `research/control-selection-evaluation`
**Prepared after:** May 28, 2026 Joe Near meeting
**Purpose:** Align before implementation

## 1. Research Objective

The current system appears strong enough to stop open-ended model improvement
and move toward paper-facing evaluation. The next research step is to identify
which controls are defensible headline claims, which controls should become
limitations/future work, and whether gender control can be made clear enough to
include in the paper/demo.

The practical objective is to produce a simpler paper story:

> A controllable VAE over OpenVoice speaker embeddings can produce speaker
> identity shift while preserving selected controllable attributes, but only
> attributes with clear perceptual/training-data signal should be claimed.

## 2. Alignment Decisions Needed Before Implementation

We should align on these before starting:

1. Should `research/control-selection-evaluation` be the next branch?
2. Should training-data separability be the first experiment?
3. Should gender be the only metadata control we actively repair now?
4. Should accent be documented as out of scope immediately?
5. Should age be kept as optional broad-bucket/future work unless cheap?
6. Should paper-methods documentation proceed in parallel with the audit?

My recommendation is yes to all six.

## 3. Core Hypotheses

### H1: Emotion/style failures are partly training-label separability failures

Some labels, especially `disgust`, may not be perceptually separable even in the
original training data. If emotion2vec cannot classify the original labeled
clips reliably, then it is not reasonable to expect the VAE to produce a strong,
classifier-visible version of that label.

Evidence needed:

- Per-label precision, recall, F1, support, and confusion matrix on original
  CREMA-D/Expresso training clips.

Decision rule:

- High-separability labels can be headline controls.
- Low-separability labels become limitations/future work, not repair targets.

### H2: The paper should focus on selected controls, not all labels

The paper does not need every CREMA-D/Expresso label to work. It needs a
defensible selection rule and clear evidence for the selected controls.

Evidence needed:

- A ranked control table that combines training-data separability, generated
  output metrics, and listening evidence.

Decision rule:

- Select a small top-k control set for the headline story.
- Explicitly explain why excluded labels are not central claims.

### H3: Gender should be repairable with better data and narrower setup

Joe believes gender is perceptually important and previously heard it work in a
simpler CommonVoice-only setup. Our weak first metadata-control result may be
caused by too little gender-known data, interference from emotion controls, or
latent capacity limits.

Evidence needed:

- Larger gender-known CommonVoice extraction counts.
- Gender-only control listening panel.
- Gender-plus-selected-style listening panel if gender-only works.
- Optional bounded latent-dimension comparison if gender remains unclear.

Decision rule:

- If gender works alone and survives selected-style combination, include it.
- If gender works alone but fails with emotion/style, report interaction as a
  limitation and decide whether to include gender-only.
- If gender fails alone, diagnose data/capacity and decide whether to demote it.

### H4: Accent and fine-grained age should not drive the next cycle

Accent likely lives in the content representation for OpenVoice-style systems,
not in the speaker embedding. Fine-grained age is not perceptually stable enough
for scalar control.

Evidence needed:

- Documentation update, not a large experiment.
- Optional cheap age-bucket separability probe only if time remains.

Decision rule:

- Accent is out of scope for this paper setup.
- Age is future work unless broad buckets show clear separability cheaply.

## 4. Workstream A: Training-Data Style Separability Audit

Question:

- Which emotion/style labels are separable enough in the source training data?

Inputs:

- Original CREMA-D labeled training audio.
- Original Expresso labeled training audio.
- Existing label mappings used by the OpenVoice controllable VAE pipeline.

Method:

1. Build or adapt a script to evaluate original labeled training clips with the
   same emotion2vec pipeline used for generated-output evaluation.
2. Normalize label names consistently across CREMA-D, Expresso, and emotion2vec.
3. Compute per-row predictions and confidence scores.
4. Compute per-label support, precision, recall, F1, top confusions, and macro /
   weighted averages.
5. Produce a ranked control shortlist.

Expected artifacts:

- `results/training_style_separability_rows.csv`
- `results/training_style_separability_by_label.csv`
- `results/training_style_separability_confusion.csv`
- `results/training_style_separability_summary.md`

Validation:

- The script runs from documented commands with no hardcoded local paths.
- The summary explicitly discusses `disgust`, `anger`, `sad`, `whisper`, and
  any other candidate headline controls.
- The shortlist explains why each selected control is defensible.
- Weak labels are preserved as limitations, not silently discarded.

Stop/continue rule:

- If only a few labels are separable, the paper story becomes selected-control
  generation rather than full emotion conversion.
- If most labels are separable but generated outputs fail, then the bottleneck
  is model/control transfer and a follow-up repair may be justified.

## 5. Workstream B: Control Shortlist and Paper Claim Selection

Question:

- Which controls belong in the headline paper/demo story?

Method:

1. Combine training-data separability with existing generated-output metrics.
2. Cross-check against human listening notes from Stephen and Joe.
3. Categorize controls into four buckets:
   - headline controls;
   - supported but quality-sensitive controls;
   - diagnostic/limitation controls;
   - out-of-scope controls.
4. Create a paper-facing control-selection table.

Expected artifacts:

- `results/control_selection_recommendation.md`
- `results/control_selection_recommendation.csv`

Validation:

- Every headline control has objective and perceptual justification.
- Every excluded control has a defensible reason.
- The recommendation reduces, not expands, the number of active claims.

Decision gate:

- Do not start more model training for a weak label until this shortlist exists.

## 6. Workstream C: Gender-Focused CommonVoice Follow-Up

Question:

- Can gender control be made perceptually clear enough for the paper/demo?

Inputs:

- Local CommonVoice corpus with `validated.tsv` and `clips/`.
- Known gender metadata rows.
- Existing OpenVoice embedding extraction pipeline.

Method:

1. Extract a larger gender-known CommonVoice subset.
2. Report metadata counts by gender and by speaker.
3. Train/evaluate a gender-only control setup first.
4. Build a small listening panel with source, baseline, male-target, and
   female-target examples.
5. If gender-only works, test gender plus top selected style controls.
6. If gender remains weak, run one bounded latent-dimension comparison only if
   the data counts are adequate.

Expected artifacts:

- `results/commonvoice_gender_metadata_report.md`
- `embeddings/openvoice_commonvoice_gender_*.pt`
- `results/listening_gender_control_*.html`
- `results/gender_control_summary.md`

Validation:

- Gender-known row counts are materially larger than the first metadata smoke
  test.
- Listening examples are easy for Stephen and Joe to inspect.
- The result clearly says whether gender is working, blocked by data,
  interacting poorly with style, or likely capacity-limited.

Decision gate:

- If gender works, include it as a major non-emotion controllable attribute.
- If gender does not work after the bounded follow-up, record it as a limitation
  and prioritize paper writing over more metadata repair.

## 7. Workstream D: Age and Accent Scope Cleanup

Question:

- How do we prevent weak metadata controls from confusing the paper story?

Method:

1. Update docs to explain why accent is out of scope for the current speaker
   embedding path.
2. Reframe age as optional broad buckets, not continuous scalar control.
3. Only run an age-bucket probe if Workstreams A-C finish cleanly.

Expected artifacts:

- Updated `FINDINGS.md`, `WORKLOG.md`, and paper-methods docs.
- Optional `results/age_bucket_separability_summary.md` if the probe is run.

Validation:

- The paper story no longer implies we can or should control accent.
- Age is not included as a headline claim without evidence.

## 8. Workstream E: Paper-Methods Scaffold

Question:

- What needs to be written so the current evidence becomes a paper?

Method:

1. Update `PAPER_METHODS_AND_EVIDENCE.md` around the current reference guard.
2. Add a control-selection rationale subsection.
3. Add a limitations subsection.
4. Add a table plan for the final paper.
5. Add anticipated reviewer questions and short answers.

Expected artifacts:

- Updated `PAPER_METHODS_AND_EVIDENCE.md`
- Optional `PAPER_OUTLINE_control_selection.md`

Validation:

- A collaborator can understand the claim without branch archaeology.
- The selected controls and excluded controls are both justified.
- The plan makes the paper simpler than the current branch history.

## 9. Proposed Execution Order

### Step 1: Create branch and freeze scope

- Branch: `research/control-selection-evaluation`.
- Do not train new models yet.
- Add a short scope note to `WORKLOG.md`.

### Step 2: Run training-data separability audit

- This is the first real experiment because it decides what should be claimed.

### Step 3: Build control-selection recommendation

- Convert audit results plus existing generated-output evidence into a top-k
  recommendation.

### Step 4: Run gender-focused follow-up

- Only after the top-style shortlist exists, so gender-plus-style experiments do
  not combine gender with weak/noisy style labels.

### Step 5: Update paper-methods docs

- Fold the selected controls, exclusions, and limitations into the paper story.

### Step 6: Closeout and review with Joe

- Prepare a Joe-facing summary with anticipated questions and concise answers.
- Include listening links for any gender examples.

## 10. Acceptance Criteria

The plan is successful if it produces:

- A defensible top-control set for the paper/demo.
- A clear status for `disgust`, `anger`, and other hard labels.
- A clear gender-control result or limitation.
- An explicit accent exclusion rationale.
- A low-priority age framing that does not distract from the paper.
- Updated paper-methods documentation that reduces branch/story complexity.

## 11. Risks and Mitigations

Risk:

- emotion2vec may be an imperfect judge of training-data separability.

Mitigation:

- Treat classifier F1 as one selection signal and preserve human listening notes
  as a second signal.

Risk:

- Gender works only in isolation and fails when mixed with emotion/style.

Mitigation:

- Report gender-only as a validated subcase and treat simultaneous control as a
  limitation/follow-up.

Risk:

- The plan expands into another large model-search branch.

Mitigation:

- No hard-style repair or broad metadata sweep until the control-selection table
  exists.

Risk:

- Age and accent consume time.

Mitigation:

- Accent is docs-only/out-of-scope. Age is optional, broad-bucket, and lower
  priority than gender.

## 12. Alignment Recommendation

I recommend we align on this plan and then start with Step 1 and Step 2 only:

1. Create `research/control-selection-evaluation`.
2. Implement the training-data separability audit.
3. Review the shortlist before running gender training.

This keeps us aligned with Joe's instruction: get to the paper by simplifying
and justifying the evaluation, not by adding more model complexity.
