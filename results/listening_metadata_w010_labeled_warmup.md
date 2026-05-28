# CommonVoice Metadata Control Smoke Panel

Generated: 2026-05-27

Branch: `research/commonvoice-metadata-controls`

Checkpoint:

- `embeddings/openvoice_vae_mixed_teacher_cvrare_metadata_w010_labeled_warmup.pt`

Training artifact:

- `embeddings/openvoice_mixed_teacher_cvrare_hybrid_extra_metadata_base.pt`

Listening report:

- `results/listening_metadata_w010_labeled_warmup.html`
- `results/listening_metadata_w010_labeled_warmup_ratings.csv`

Generated audio directory:

- `output/metadata_control_panel_w010/`

## What This Tests

This is a smoke panel for the first direct CommonVoice metadata-control
checkpoint. It holds the style fixed at `happy` and varies the metadata
controls:

- reference: no age/gender metadata control
- `female` + `twenties`
- `male` + `twenties`
- `female` + `fifties`
- `male` + `fifties`

Two source speakers are included:

- `examples/source_speakers/female_1_cremad_1002.wav`
- `examples/source_speakers/male_1_cremad_1003.wav`

## How To Listen

Run:

```bash
cd /Users/steve/UVM-plaid/dp-vc
python3 -m http.server 8000
```

Open:

```text
http://localhost:8000/results/listening_metadata_w010_labeled_warmup.html
```

For each source, listen in this order:

1. Source audio.
2. `happy` reference with no metadata control.
3. Female/male variants at the same age bucket.
4. Twenties/fifties variants with the same gender control.

Score the rows in:

```text
results/listening_metadata_w010_labeled_warmup_ratings.csv
```

## Interpretation Rules

- Do not treat the requested age/gender labels as achieved unless they are
  perceptible.
- The desired win is not just a change in timbre; it must preserve
  intelligibility and naturalness.
- If all variants sound identical, this is a useful negative/diagnostic result:
  the direct metadata loss trained without producing audible control.
- If variants change identity but not age/gender, that suggests the metadata
  dims are acting as generic speaker-shift knobs rather than interpretable
  controls.

## Perceptual Outcome

Local review on 2026-05-28 found that the variants sounded identical or mostly
like generic speaker/timbre shifts. The first direct scalar metadata-control
checkpoint therefore should be treated as a negative diagnostic, not as evidence
that age/gender controls are perceptually working.

Action:

- Do not promote this checkpoint as a paper/demo result.
- Do not run the full WER/MOS/novelty stack for this checkpoint unless a later
  documentation need requires it; the perceptual gate failed.
- Before retrying metadata controls, first test whether OpenVoice speaker
  embeddings contain recoverable age/gender signal and use a more balanced or
  stronger metadata objective.

## Validation

- `Validation`: generated `10` audio rows from `2` source speakers.
- `Validation`: `results/listening_metadata_w010_labeled_warmup.html` contains
  `12` valid audio references, including source clips.
- `Validation`: the rating CSV includes `gender_control` and `age_control`
  columns so perceptual review can separate style quality from metadata-control
  behavior.
- `Validation`: local listener classified the panel as effectively identical
  or generic timbre/identity movement rather than perceptible age/gender
  control.
