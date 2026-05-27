# Implementation Plan: CommonVoice Metadata Controls

**Branch:** `research/commonvoice-metadata-controls`
**Started:** 2026-05-27
**Base:** latest accepted research/evidence packet line

## Goal

Add CommonVoice age/gender as controllable speaker attributes so the project is
not framed as only emotion conversion. This tests Joe's broader claim: a single
controllable VAE should support multiple labeled speaker attributes when labels
exist.

## Current Audit Result

Audited local CommonVoice English corpus:

- Corpus: `/Users/steve/datasets/cv-corpus-21.0-2025-03-14/en`
- Local clips: `40000`
- Local validated rows: `40000`
- Unique local speakers: `20537`
- Age-control rows: `5504` (`13.8%`)
- Gender-control rows: `5291` (`13.2%`)
- Rows with both age and gender: `5258` (`13.1%`)

Audited extracted OpenVoice artifact:

- Artifact: `embeddings/openvoice_commonvoice_cvrare_expanded_emb.pt`
- Rows: `25910`
- Unique speakers: `13308`
- Age-control rows: `3566` (`13.8%`)
- Gender-control rows: `3429` (`13.2%`)
- Rows with both age and gender: `3403` (`13.1%`)

The metadata is usable, but imbalanced:

- Gender is mostly `male_masculine` with fewer `female_feminine` rows.
- Age is mostly `twenties` / `thirties`; older buckets are much smaller.

## Design Decision

Use a conservative first-pass scalar control design instead of a large one-hot
demographic taxonomy:

- style dims: `0-8`
- `dim_9`: gender binary scalar, `female_feminine/female=-1`, `male_masculine/male=1`
- `dim_10`: age ordinal scalar, youngest bucket `-1`, oldest bucket `1`
- free dims: `11-14`
- latent dims remain `15`

Why:

- It preserves the existing 15-dim model family.
- It leaves some free speaker/identity capacity.
- It avoids overfitting sparse age/gender categories.
- It creates direct controls that can be manipulated at inference time.

## Implementation Tasks

1. Preserve per-row metadata in mixed artifacts. **Status: implemented and smoke-validated.**
   - Extend `scripts/build_mixed_training_set.py` so output artifacts contain:
     - `metadata_gender_scalar`
     - `metadata_gender_mask`
     - `metadata_age_ordinal_scalar`
     - `metadata_age_mask`
     - metadata label reports and class/count summaries.
   - Preserve missing metadata as masked labels, not zeros.

2. Add masked metadata-control loss to mixed VAE training. **Status: implemented and smoke-validated.**
   - Extend `dpvc.utils.train_mixed_autoencoder` or add a small wrapper that can
     supervise selected latent dims with per-row masks.
   - Do not reuse the existing CommonVoice pretraining metadata-classification
     head as the main control mechanism; that head is useful as an auxiliary
     baseline, but it does not make age/gender directly controllable at
     inference time.

3. Extend `examples/openvoice_train_vae_mixed.py` with metadata-control flags. **Status: implemented and smoke-validated.**
   - Suggested flags:
     - `--metadata-control-weight`
     - `--metadata-gender-dim 9`
     - `--metadata-age-dim 10`
     - `--metadata-control-report`
   - Print labeled-row counts and warn if either mask is empty.

4. Add inference support for metadata controls. **Status: implemented at CLI/manifest level; needs a real metadata-control checkpoint before generated-audio validation.**
   - Extend the controllable inference path with explicit age/gender controls.
   - Keep this separate from emotion/style controls in the metadata output.
   - Ensure every generated output records style, age, gender, seed, source,
     checkpoint, and latent-dim settings in the manifest.

5. Evaluate first checkpoints. **Status: first checkpoint trained and listening smoke panel generated; perceptual review next.**
   - Run a metadata-only or metadata-light mixed checkpoint first.
   - Generate a small panel with gender/age controls while holding content and
     style fixed.
   - Evaluate WER, MOS, novelty, and a simple latent/proxy metadata-readout.
   - Build a listening report for perceptual review.

## Acceptance / Validation

- `scripts/audit_commonvoice_metadata_controls.py` runs on the local corpus and
  writes Markdown/JSON reports.
- `scripts/build_mixed_training_set.py` emits per-row metadata scalars and masks.
- Trainer can run when metadata masks are sparse and does not treat missing
  metadata as a target value.
- A smoke training run completes with `--metadata-control-weight > 0`.
- `examples/openvoice_infer_controllable.py` exposes deterministic age/gender
  controls and records them in the generation manifest.
- Listening artifacts include enough metadata for Joe or another collaborator
  to inspect the outputs.

## Validation Log

- 2026-05-27: `py_compile` passed for the mixed builder, mixed trainer,
  VAE model, controllable inference CLI, and metadata audit script.
- 2026-05-27: training CLI help exposes `--metadata-control-weight`,
  `--metadata-gender-dim`, `--metadata-age-dim`, and
  `--metadata-control-report`.
- 2026-05-27: inference CLI help exposes `--gender-control`,
  `--gender-control-dim`, `--age-control`, and `--age-control-dim`.
- 2026-05-27: `/private/tmp/openvoice_mixed_metadata_smoke.pt` built
  successfully and contains metadata scalar/mask tensors.
- 2026-05-27: one-epoch smoke training with `--metadata-control-weight 0.1`
  completed successfully and wrote `/private/tmp/openvoice_vae_metadata_smoke.pt`.
- 2026-05-27: full metadata-ready mixed artifact
  `embeddings/openvoice_mixed_teacher_cvrare_hybrid_extra_metadata_base.pt`
  built successfully with `1889` age-control rows and `1795` gender-control
  rows.
- 2026-05-27: first 1000-epoch checkpoint
  `embeddings/openvoice_vae_mixed_teacher_cvrare_metadata_w010_labeled_warmup.pt`
  trained successfully with metadata-control weight `0.1`.
- 2026-05-27: generated
  `results/listening_metadata_w010_labeled_warmup.html` and
  `results/listening_metadata_w010_labeled_warmup_ratings.csv` for local
  perceptual review.

## Risks

- CommonVoice age/gender labels are self-reported and imbalanced.
- Age/gender may be weakly represented in OpenVoice speaker embeddings.
- Gender and age controls can raise fairness/ethics concerns; report them as
  speaker-attribute controls from metadata, not as reliable demographic
  classifiers.
- Supervising too many metadata dimensions could reduce free speaker capacity;
  the first pass should stay scalar and conservative.

## Future Upgrades

- Add a gender-balanced local CommonVoice subset if the current shard is too
  male-heavy for perceptual control.
- Compare direct scalar latent supervision with auxiliary metadata-head
  supervision from `openvoice_pretrain_vae_commonvoice.py`.
- Add an external age/gender/speaker-attribute verifier only after first
  generated audio sounds plausible; do not pick a verifier before the control
  path exists.
- Add fairness/ethics wording before any paper/demo claims about age/gender.
