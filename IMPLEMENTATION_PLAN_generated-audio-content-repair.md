# Implementation Plan: Generated-Audio Content-Repair Gate

**Branch:** `research/generated-audio-content-repair`
**Started:** 2026-05-28
**Base:** `research/metadata-separability-probe`

## Goal

Turn the existing hard-style strength grid into a conservative content-repair
gate so candidate presets cannot be promoted from classifier gains alone.

## Scope

In scope:

- add a reusable selector for hard-style generated-audio candidates;
- combine objective target gain with content, naturalness, novelty, and
  perceptual-listening gates;
- fold Joe's five-row A/B review into the decision logic;
- produce CSV, Markdown, and machine-readable JSON outputs;
- record whether any `anger`, `disgust`, or `fear` candidate is safe to promote.

Out of scope:

- retraining the model;
- generating a new strength grid;
- claiming that the current grid fixes hard styles;
- replacing listening review with classifier scores.

## Implementation Tasks

1. Add `scripts/select_generated_audio_content_repairs.py`.
   - Inputs: A/B priority CSV and optional perceptual ratings CSV.
   - Gates: target-label gain, WER safety, MOS safety, novelty safety, and
     perceptual preference when ratings exist.
   - Outputs: decision CSV, Markdown summary, and JSON profile payload.

2. Add unit tests.
   - strength-token parsing;
   - objective-pass rows without ratings require listening;
   - perceptual ties block promotion;
   - candidate preference promotes only if objective gates pass;
   - quality-risk rows reject before perceptual promotion;
   - no-target-gain rows reject;
   - style summaries separate diagnostic rows from true repairs.

3. Run the gate on the checked-in hard-style strength-grid review.
   - Priority CSV:
     `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority.csv`
   - Joe ratings:
     `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority_ratings_joe_2026-05-19.csv`
   - Outputs:
     `results/generated_audio_content_repair_gate.csv`
     `results/generated_audio_content_repair_gate.md`
     `results/generated_audio_content_repair_gate.json`

4. Update durable docs and ledgers.
   - `FINDINGS.md`: add the no-promote generated-audio repair gate result.
   - `WORKLOG.md`: mark this branch complete and add next work.
   - Public docs: update current next queue and results index.

## Acceptance / Validation

- Unit tests pass with standard-library `unittest`.
- `py_compile` passes for the new selector.
- The selector writes CSV, Markdown, and JSON outputs from checked-in artifacts.
- The report explicitly blocks preset promotion when objective gains fail
  perceptual review.

## Future Upgrades

- Build a true generated-audio-calibrated training objective for hard styles,
  because strength-grid reranking alone does not produce a promoted repair.
- Add a larger blind listening panel only after a candidate beats this gate on
  objective content/style criteria.
- Add row-level audio-feature diagnostics for blocked candidates, especially
  the `fear_s7p5` pitch-change case Joe heard as unnatural.
