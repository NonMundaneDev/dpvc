# CommonVoice Pseudo-Label Supply Audit

This report tracks rare-style supply before running more mixed-data weighting or curriculum experiments.

## `embeddings/openvoice_commonvoice_cv500_pseudo_scored.pt`

- type: `commonvoice`
- rows: `1202`
- speakers: `500`

| style | raw pseudo | selected | mixed CV | mixed total | score>=0.60 | score>=0.90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `anger` | 15 | 0 | 0 | 0 | 11 | 6 |
| `confused` | 0 | 0 | 0 | 0 | 0 | 0 |
| `disgust` | 57 | 0 | 0 | 0 | 46 | 36 |
| `enunciated` | 0 | 0 | 0 | 0 | 0 | 0 |
| `fear` | 9 | 0 | 0 | 0 | 5 | 4 |
| `happy` | 59 | 0 | 0 | 0 | 51 | 45 |
| `neutral` | 680 | 0 | 0 | 0 | 640 | 541 |
| `sad` | 359 | 0 | 0 | 0 | 336 | 268 |
| `whisper` | 0 | 0 | 0 | 0 | 0 | 0 |

## `embeddings/openvoice_commonvoice_cv500_pseudo_filtered.pt`

- type: `commonvoice`
- rows: `1202`
- speakers: `500`

| style | raw pseudo | selected | mixed CV | mixed total | score>=0.60 | score>=0.90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `anger` | 15 | 6 | 0 | 0 | 11 | 6 |
| `confused` | 0 | 0 | 0 | 0 | 0 | 0 |
| `disgust` | 57 | 36 | 0 | 0 | 46 | 36 |
| `enunciated` | 0 | 0 | 0 | 0 | 0 | 0 |
| `fear` | 9 | 4 | 0 | 0 | 5 | 4 |
| `happy` | 59 | 42 | 0 | 0 | 51 | 45 |
| `neutral` | 680 | 120 | 0 | 0 | 640 | 541 |
| `sad` | 359 | 110 | 0 | 0 | 336 | 268 |
| `whisper` | 0 | 0 | 0 | 0 | 0 | 0 |

## `embeddings/openvoice_commonvoice_cv500_pseudo_hybrid_extra_priority.pt`

- type: `commonvoice`
- rows: `1202`
- speakers: `500`

| style | raw pseudo | selected | mixed CV | mixed total | score>=0.60 | score>=0.90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `anger` | 5 | 5 | 0 | 0 | 5 | 5 |
| `confused` | 40 | 40 | 0 | 0 | 40 | 7 |
| `disgust` | 32 | 32 | 0 | 0 | 32 | 32 |
| `enunciated` | 40 | 40 | 0 | 0 | 40 | 13 |
| `fear` | 4 | 4 | 0 | 0 | 4 | 4 |
| `happy` | 37 | 37 | 0 | 0 | 37 | 37 |
| `neutral` | 109 | 109 | 0 | 0 | 109 | 109 |
| `sad` | 106 | 106 | 0 | 0 | 106 | 106 |
| `whisper` | 12 | 12 | 0 | 0 | 0 | 0 |

## `embeddings/openvoice_mixed_teacher_hybrid_extra_base.pt`

- type: `mixed`
- rows: `1325`
- speakers: `594`

| style | raw pseudo | selected | mixed CV | mixed total | score>=0.60 | score>=0.90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `anger` | 0 | 0 | 4 | 95 | 0 | 0 |
| `confused` | 0 | 0 | 22 | 112 | 0 | 0 |
| `disgust` | 0 | 0 | 27 | 118 | 0 | 0 |
| `enunciated` | 0 | 0 | 27 | 117 | 0 | 0 |
| `fear` | 0 | 0 | 4 | 95 | 0 | 0 |
| `happy` | 0 | 0 | 30 | 124 | 0 | 0 |
| `neutral` | 0 | 0 | 66 | 160 | 0 | 0 |
| `sad` | 0 | 0 | 78 | 172 | 0 | 0 |
| `whisper` | 0 | 0 | 9 | 99 | 0 | 0 |

## Readout

- `openvoice_commonvoice_cv500_pseudo_filtered.pt` has `anger=6` selected rows.
- `openvoice_commonvoice_cv500_pseudo_filtered.pt` has `fear=4` selected rows.
- `openvoice_commonvoice_cv500_pseudo_hybrid_extra_priority.pt` has `anger=5` selected rows.
- `openvoice_commonvoice_cv500_pseudo_hybrid_extra_priority.pt` has `fear=4` selected rows.
- `openvoice_mixed_teacher_hybrid_extra_base.pt` has `anger=4` selected rows.
- `openvoice_mixed_teacher_hybrid_extra_base.pt` has `fear=4` selected rows.
- If rare canonical styles remain below roughly tens of rows, prefer extracting/scoring more CommonVoice data over another loss-weight tweak.
- If raw score counts are high but selected counts are low, adjust filters; if raw counts are low, the local corpus is the bottleneck.
