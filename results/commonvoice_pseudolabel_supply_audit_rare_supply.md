# CommonVoice Pseudo-Label Supply Audit

This report tracks rare-style supply before running more mixed-data weighting or curriculum experiments.

## `embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_scored.pt`

- type: `commonvoice`
- rows: `25910`
- speakers: `13308`

| style | raw pseudo | selected | mixed CV | mixed total | score>=0.60 | score>=0.90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `anger` | 151 | 0 | 0 | 0 | 126 | 83 |
| `confused` | 0 | 0 | 0 | 0 | 0 | 0 |
| `disgust` | 280 | 0 | 0 | 0 | 237 | 164 |
| `enunciated` | 0 | 0 | 0 | 0 | 0 | 0 |
| `fear` | 65 | 0 | 0 | 0 | 50 | 25 |
| `happy` | 362 | 0 | 0 | 0 | 316 | 232 |
| `neutral` | 3258 | 0 | 0 | 0 | 3077 | 2575 |
| `sad` | 2065 | 0 | 0 | 0 | 1904 | 1475 |
| `whisper` | 0 | 0 | 0 | 0 | 0 | 0 |

## `embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_filtered.pt`

- type: `commonvoice`
- rows: `25910`
- speakers: `13308`

| style | raw pseudo | selected | mixed CV | mixed total | score>=0.60 | score>=0.90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `anger` | 151 | 50 | 0 | 0 | 126 | 83 |
| `confused` | 0 | 0 | 0 | 0 | 0 | 0 |
| `disgust` | 280 | 80 | 0 | 0 | 237 | 164 |
| `enunciated` | 0 | 0 | 0 | 0 | 0 | 0 |
| `fear` | 65 | 50 | 0 | 0 | 50 | 25 |
| `happy` | 362 | 80 | 0 | 0 | 316 | 232 |
| `neutral` | 3258 | 120 | 0 | 0 | 3077 | 2575 |
| `sad` | 2065 | 120 | 0 | 0 | 1904 | 1475 |
| `whisper` | 0 | 0 | 0 | 0 | 0 | 0 |

## `embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_prototype_filtered.pt`

- type: `commonvoice`
- rows: `25910`
- speakers: `13308`

| style | raw pseudo | selected | mixed CV | mixed total | score>=0.60 | score>=0.90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `anger` | 2251 | 1956 | 0 | 0 | 970 | 179 |
| `confused` | 5147 | 50 | 0 | 0 | 2072 | 189 |
| `disgust` | 3540 | 3107 | 0 | 0 | 1625 | 261 |
| `enunciated` | 3526 | 50 | 0 | 0 | 1831 | 407 |
| `fear` | 1008 | 761 | 0 | 0 | 247 | 23 |
| `happy` | 2561 | 2244 | 0 | 0 | 1102 | 144 |
| `neutral` | 2410 | 2083 | 0 | 0 | 1042 | 186 |
| `sad` | 4833 | 4321 | 0 | 0 | 2487 | 602 |
| `whisper` | 634 | 50 | 0 | 0 | 49 | 0 |

## `embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_hybrid_extra_priority.pt`

- type: `commonvoice`
- rows: `25910`
- speakers: `13308`

| style | raw pseudo | selected | mixed CV | mixed total | score>=0.60 | score>=0.90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `anger` | 50 | 50 | 0 | 0 | 50 | 50 |
| `confused` | 50 | 50 | 0 | 0 | 50 | 50 |
| `disgust` | 79 | 79 | 0 | 0 | 79 | 79 |
| `enunciated` | 50 | 50 | 0 | 0 | 50 | 50 |
| `fear` | 50 | 50 | 0 | 0 | 50 | 25 |
| `happy` | 80 | 80 | 0 | 0 | 80 | 80 |
| `neutral` | 116 | 116 | 0 | 0 | 116 | 116 |
| `sad` | 120 | 120 | 0 | 0 | 120 | 120 |
| `whisper` | 50 | 50 | 0 | 0 | 49 | 0 |

## `embeddings/openvoice_mixed_teacher_cvrare_hybrid_extra_base.pt`

- type: `mixed`
- rows: `14195`
- speakers: `13402`

| style | raw pseudo | selected | mixed CV | mixed total | score>=0.60 | score>=0.90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `anger` | 0 | 0 | 50 | 141 | 0 | 0 |
| `confused` | 0 | 0 | 50 | 140 | 0 | 0 |
| `disgust` | 0 | 0 | 79 | 170 | 0 | 0 |
| `enunciated` | 0 | 0 | 50 | 140 | 0 | 0 |
| `fear` | 0 | 0 | 50 | 141 | 0 | 0 |
| `happy` | 0 | 0 | 80 | 174 | 0 | 0 |
| `neutral` | 0 | 0 | 116 | 210 | 0 | 0 |
| `sad` | 0 | 0 | 120 | 214 | 0 | 0 |
| `whisper` | 0 | 0 | 50 | 140 | 0 | 0 |

## Readout

- If rare canonical styles remain below roughly tens of rows, prefer extracting/scoring more CommonVoice data over another loss-weight tweak.
- If raw score counts are high but selected counts are low, adjust filters; if raw counts are low, the local corpus is the bottleneck.
