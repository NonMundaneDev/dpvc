# Generated-Audio-Calibrated Objective Implementation Plan

**Branch:** `research/generated-audio-calibrated-objective`
**Started:** 2026-05-28
**Base:** `research/generated-audio-content-repair`

## Goal

Turn the generated-audio repair gate into a trainer-ready hard-style objective
without promoting any style-strength preset.

## Rationale

The content-repair gate showed that no current `anger`, `disgust`, or `fear`
strength-grid candidate should become a preset. The next useful move is not
another stronger inference setting; it is a training-side objective that uses
the generated-audio evidence to decide which styles receive repair pressure.

## Scope

In scope:

- build a plan from `generated_audio_content_repair_gate.json` and
  `eval_mixed_teacher_failure_conditioned_targets.json`;
- select only styles with clean generated-audio failure targets;
- block `fear` until clean target supply or a pitch-artifact diagnostic exists;
- add a mixed-trainer hook that applies the generated-audio objective plan;
- add decoder-prototype style row weights so non-target styles do not receive
  repair loss pressure;
- add a reproducible inference condition name for the future checkpoint.

Out of scope:

- claiming a new model result before full training and generated-audio
  evaluation;
- committing new checkpoint weights;
- promoting `anger_s10`, `disgust_s10`, or `fear_s7p5` as presets.

## Implementation Tasks

1. Add `scripts/plan_generated_audio_calibrated_objective.py`.
   - Inputs: generated-audio content-repair gate JSON and failure-conditioned
     target JSON.
   - Outputs: JSON, CSV, and Markdown objective-plan artifacts.
   - Default selected styles: `anger`, `disgust`.
   - Default blocked style: `fear`.

2. Extend `examples/openvoice_train_vae_mixed.py`.
   - Add `--generated-audio-objective-plan`.
   - Add `--generated-audio-objective-report`.
   - Add `--decoder-prototype-style-weights`.
   - Apply plan overrides before trainer validation so existing flags can use
     the evidence-derived style weights and strengths.

3. Add the future checkpoint alias to `scripts/run_ablation_inference.py`.
   - Condition: `mixed_teacher_cvrare_audio_calibrated_labeled_warmup`.
   - Checkpoint:
     `embeddings/openvoice_vae_mixed_teacher_cvrare_audio_calibrated_labeled_warmup.pt`.

4. Update durable docs and ledgers.
   - `WORKLOG.md`: record this branch as an implementation step, not a model
     finding.
   - Public docs: document the new planner, trainer flags, and result
     artifacts.
   - `FINDINGS.md`: no update unless a fully trained/evaluated checkpoint
     produces a verified result.

## Acceptance / Validation

- Unit tests prove the planner selects `anger`/`disgust`, blocks `fear`, and
  applies trainer overrides.
- The planner regenerates JSON/CSV/Markdown outputs from checked-in evidence.
- `py_compile` passes for the planner, trainer, and inference alias.
- A one-epoch CommonVoice-only smoke run exercises nonzero style-teacher,
  decoder-prototype, and anti-neutral repair losses using the generated-audio
  plan.

## Future Upgrades

- Train the full
  `mixed_teacher_cvrare_audio_calibrated_labeled_warmup` checkpoint from the
  recommended command in `results/generated_audio_calibrated_objective_plan.md`.
- Generate audio, run emotion/WER/MOS/novelty/external-speaker verification,
  and build a listening panel before calling the objective successful.
- Add a fear-specific pitch/artifact diagnostic before allowing `fear` into the
  generated-audio repair objective.
