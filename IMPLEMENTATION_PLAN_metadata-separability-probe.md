# Implementation Plan: CommonVoice Metadata Separability Probe

**Branch:** `research/metadata-separability-probe`
**Started:** 2026-05-28
**Base:** `research/external-speaker-verifier`

## Goal

Before spending more training cycles on CommonVoice age/gender controls, test
whether the current OpenVoice embedding artifacts and metadata-control VAE
latents contain recoverable metadata structure.

## Scope

In scope:

- add a reusable metadata separability probe script;
- support raw embedding-space and optional VAE encoder-latent probes;
- evaluate `gender`, `age`, and `accent` with deterministic splits;
- compare against majority and permutation baselines;
- run the expanded CommonVoice artifact and the actual mixed metadata-training
  artifact;
- record the result as a diagnostic paper-facing finding.

Out of scope:

- claiming perceptual age/gender control;
- retraining the metadata-control checkpoint;
- replacing the human listening gate.

## Implementation Tasks

1. Add `scripts/probe_commonvoice_metadata_separability.py`.
   - Inputs: embedding artifact, optional VAE checkpoint, fields, class caps.
   - Metrics: nearest-centroid accuracy, macro-F1, majority baseline,
     permutation baseline, centroid separation ratio.
   - Outputs: CSV and Markdown summary.

2. Add unit tests.
   - metadata normalization;
   - sparse-class filtering;
   - stratified split behavior;
   - clear synthetic separability detection;
   - skipped-field behavior;
   - conservative verdict labels.

3. Run the expanded CommonVoice probe.
   - Artifact: `embeddings/openvoice_commonvoice_cvrare_expanded_emb.pt`
   - VAE checkpoint:
     `embeddings/openvoice_vae_mixed_teacher_cvrare_metadata_w010_labeled_warmup.pt`
   - Outputs:
     `results/commonvoice_metadata_separability_cvrare_expanded.csv`
     `results/commonvoice_metadata_separability_cvrare_expanded.md`

4. Run the mixed metadata-training-base probe.
   - Artifact: `embeddings/openvoice_mixed_teacher_cvrare_hybrid_extra_metadata_base.pt`
   - Same VAE checkpoint.
   - Outputs:
     `results/commonvoice_metadata_separability_mixed_metadata_base.csv`
     `results/commonvoice_metadata_separability_mixed_metadata_base.md`

5. Update durable docs and ledgers.
   - `FINDINGS.md`: add a verified diagnostic finding.
   - `WORKLOG.md`: mark the probe complete and add future upgrades.
   - Public docs: list the script and result artifacts where relevant.

## Acceptance / Validation

- Unit tests pass with standard-library `unittest`.
- `py_compile` passes for the probe script.
- Both probe commands write CSV and Markdown artifacts.
- The interpretation explicitly separates objective separability from
  perceptual controllability.

## Future Upgrades

- Run a gender-focused balanced metadata-control follow-up only if the paper
  needs a metadata-control appendix; gender is objectively separable, but the
  first listening panel did not prove perceptual control.
- Do not retry age/accent scalar controls without better labels, balanced
  classes, or a stronger perceptual/acoustic target because current probes show
  only weak structure in the metadata-control latents.
- Add per-dimension latent diagnostics for metadata checkpoints so we can see
  whether dims `9-10` actually carry the supervised scalar targets or leak into
  free speaker dimensions.
