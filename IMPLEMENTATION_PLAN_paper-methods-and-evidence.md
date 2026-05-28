# Implementation Plan: Paper Methods and Evidence Consolidation

**Branch:** `docs/paper-methods-and-evidence`
**Started:** 2026-05-28
**Base:** latest accepted research line after CommonVoice metadata-control
diagnostic outcome

## Goal

Stop exploratory training temporarily and make the current substantial result
paper-readable:

- document the model/data/training/evaluation method;
- map claims to evidence and listening artifacts;
- preserve the negative perceptual gates without overclaiming them;
- keep the next research queue targeted and recoverable after context
compaction.

## Scope

In scope:

- root paper/evidence packet;
- evidence/demo packet refresh;
- metric/collapse guide refresh;
- README pointer updates;
- WORKLOG closeout and next-task ledger.

Out of scope:

- new model training;
- WER/MOS/novelty evaluation for the failed metadata-control smoke panel;
- new `FINDINGS.md` entries without verified experimental evidence.

## Implementation Tasks

1. Create `PAPER_METHODS_AND_EVIDENCE.md`.
   - Include problem framing, method outline, active model path, data mixture,
     supervision strategy, evaluation stack, claim-to-evidence table, current
     non-claims, Joe-facing Q&A, and next research queue.

2. Refresh `EVIDENCE_DEMO_PACKET.md`.
   - Promote the current quality-balanced reference:
     `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard`.
   - Keep the generated-audio strength grid diagnostic after Joe's `0/5`
     candidate-win review.
   - Mark CommonVoice age/gender controls as implemented but not perceptually
     validated.

3. Refresh `docs/metric_collapse_guide.md`.
   - Add the paper-readiness rule: metrics nominate candidates, listening
     decides paper/demo claims.
   - Record the status of the style guard, strength grid, and metadata-control
     panel.

4. Update `README.md`.
   - Point collaborators to the new paper-method packet.
   - Replace stale "next: age/gender controls" wording with the current
     diagnostic outcome and next targeted research queue.

5. Update `WORKLOG.md`.
   - Mark paper-method documentation as complete for this branch.
   - Mark the failed metadata-control metric run as intentionally skipped.
   - Add next tasks with enough context to survive compaction.

## Acceptance / Validation

- `PAPER_METHODS_AND_EVIDENCE.md` exists and includes a claim-to-evidence map.
- The docs do not claim age/gender control is perceptually solved.
- The docs do not promote `anger_s10` or `fear_s7p5` as presets.
- The canonical listening entrypoint is still
  `results/listening_evidence_demo_index.html`.
- `FINDINGS.md` remains unchanged unless a verified finding is added later.
- `git diff --check` passes.

## Future Upgrades

- Add an external speaker-verifier / EER-style novelty check so identity-shift
  evidence does not depend only on native OpenVoice embedding space.
- Run a metadata separability probe before more age/gender training; if
  OpenVoice embeddings do not encode recoverable age/gender signal, direct
  scalar controls are likely to remain generic timbre knobs.
- Build a generated-audio/content-repair loop for `anger`, `disgust`, and
  `fear`, because latent/embedding proxy objectives did not beat the current
  guard.
- Add repeated-seed confidence intervals before freezing final paper tables.
- Add formal DP accounting and privacy-utility curves before submission.
