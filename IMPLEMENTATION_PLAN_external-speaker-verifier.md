# Implementation Plan: External Speaker-Verifier Novelty Validation

**Branch:** `research/external-speaker-verifier`
**Started:** 2026-05-28
**Base:** latest accepted research/docs line after paper-method consolidation

## Goal

Add an external speaker-verifier check so the paper's identity-shift evidence
does not rely only on OpenVoice's native embedding space.

## Scope

In scope:

- add a reusable ECAPA-TDNN speaker-verifier evaluation script;
- support manifest-driven generated-output evaluation;
- support EER thresholding from either a real trial CSV or a clearly labeled
  proxy trial set derived from source/baseline outputs;
- run the current quality-balanced reference panel;
- record paper-facing interpretation with caveats.

Out of scope:

- replacing native OpenVoice novelty metrics;
- claiming formal speaker-verification EER without an independent labeled trial
  CSV;
- retraining the voice model.

## Implementation Tasks

1. Add `scripts/eval_external_speaker_verifier.py`.
   - Backend: SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`).
   - Inputs: `generation_manifest.jsonl` or single source/generated pair.
   - Outputs: per-row CSV plus optional Markdown summary.
   - Metrics: external source similarity, distance, baseline-relative novelty
     gain, optional accept-as-source decision at an EER threshold.

2. Add tests for reusable verifier behavior.
   - cosine edge cases;
   - EER threshold calculation;
   - proxy trial construction;
   - summary aggregation;
   - soundfile-based audio loading to avoid torchaudio/torchcodec drift.

3. Add optional dependency metadata.
   - `pyproject.toml` gets `speaker-verifier = ["speechbrain"]`.

4. Run the current quality-balanced reference.
   - Manifest:
     `output/mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard_eval/generation_manifest.jsonl`
   - Outputs:
     `results/eval_external_speaker_verifier_cvrare_sad_enunc_guard.csv`
     `results/eval_external_speaker_verifier_cvrare_sad_enunc_guard.md`

5. Update docs and ledgers.
   - `FINDINGS.md` gets a paper-facing finding only if the run completes.
   - `WORKLOG.md`, `README.md`, `results/README.md`, and
     `PAPER_METHODS_AND_EVIDENCE.md` point to the new metric.

## Acceptance / Validation

- Unit tests pass with standard-library `unittest`.
- `py_compile` passes for the new script.
- The ECAPA run writes CSV and Markdown artifacts for the current reference.
- The finding explicitly labels derived baseline trials as proxy calibration,
  not formal independent speaker-verification EER.

## Future Upgrades

- Build an independent labeled trial CSV from multiple source utterances per
  speaker and rerun EER without proxy baselines.
- Compare ECAPA results against another external verifier, such as WavLM or
  Resemblyzer, to check model-family sensitivity.
- Add privacy-utility curves that combine external accept-as-source rates with
  WER/MOS/style recall.
