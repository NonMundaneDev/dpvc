# CommonVoice Gender Follow-Up Listening Outcome

Generated: 2026-07-09

Branch: `research/commonvoice-gender-followup`

Checkpoint:

- `embeddings/openvoice_vae_mixed_gender_followup_available_w005_labeled_warmup.pt`

Training artifact:

- `embeddings/openvoice_mixed_gender_followup_available_base.pt`

Listening report:

- `results/listening_gender_followup_available_w005_labeled_warmup.html`
- `results/listening_gender_followup_available_w005_labeled_warmup_ratings.csv`

Review bundle:

- `results/gender_followup_available_w005_review_bundle_2026-06-13.zip`
- `results/gender_followup_available_w005_review_bundle_2026-06-13/`

## What This Tests

This is the first gender-only CommonVoice follow-up after the neutral/sad
control-selection gate. It tests whether a direct scalar gender control on the
OpenVoice speaker-embedding VAE produces listener-clear `female` and `male`
outputs while preserving intelligibility and naturalness.

The run is deliberately limited:

- gender-only metadata control, no age/accent claim;
- available-subset CommonVoice artifact, not the full preflight plan;
- `1181/1788` preflight-selected clips matched from the already-extracted
  expanded CommonVoice artifact.

## Perceptual Outcome

Local review on 2026-07-09 found that the `female` / `male` controls were not
reliably perceptible as gender controls. The result sounded closer to subtle,
inconsistent, or generic speaker/timbre movement than to a stable controllable
gender attribute.

This fails the perceptual gate for a paper-facing gender-control claim.

## Interpretation

- Do not promote this checkpoint as a paper/demo gender-control result.
- Do not treat objective gender or speaker-verifier diagnostics on this
  checkpoint as claim-establishing evidence; they would mainly characterize
  generic speaker movement after a failed perceptual gate.
- Keep the paper story anchored on the current reference guard plus the two
  perceptually confirmed headline style controls: `neutral` and `sad`.
- Preserve the result as a diagnostic limitation: gender is objectively
  recoverable in OpenVoice embeddings and metadata-control latents, but this
  scalar speaker-embedding VAE knob did not produce reliable perceptual gender
  control in the available-subset follow-up.

## Next

- Prioritize paper/evaluation cleanup over more scalar metadata-control tuning.
- Revisit gender only with a materially stronger setup, such as a full
  extraction of the preflight-selected clips plus a better-balanced or more
  perceptually grounded objective, or a bounded latent-capacity comparison.
- Keep age and accent out of headline scope unless a later result has both
  objective structure and listener-clear perceptual evidence.

## Acoustic Sanity Check

After the listening gate failed, a lightweight acoustic diagnostic checked
whether the generated `female` / `male` controls moved simple pitch-like cues in
a consistent direction.

Artifacts:

- `results/gender_followup_available_w005_acoustic_diagnostic.md`
- `results/gender_followup_available_w005_acoustic_diagnostic.csv`
- `results/gender_followup_available_w005_acoustic_pairs.csv`

Result:

- `female`-control median F0 was higher than `male`-control median F0 in only
  `1/4` source pairs.
- Median `female` minus `male` F0 delta was `-5.69 Hz`.
- Median `female` minus `male` spectral-centroid delta was `-27.59 Hz`.

Interpretation:

- This is an acoustic proxy, not a perceptual gender classifier.
- It supports the same conservative conclusion as listening: the current
  scalar gender knob does not produce a consistent, listener-clear gender
  control.

## Validation

- `Validation`: Stephen listened to the four-source review bundle and
  classified the outcome as not reliably perceptible gender control.
- `Validation`: no numeric ratings were fabricated from the aggregate review;
  this Markdown file records the qualitative gate outcome.
- `Validation`: the acoustic sanity check ran locally on the existing
  `output/gender_followup_available_w005_2026-06-13/generation_manifest.jsonl`
  review audio; it is diagnostic only and does not override the listening gate.
