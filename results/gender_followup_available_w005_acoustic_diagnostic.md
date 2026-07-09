# Gender Follow-Up Acoustic Diagnostic

- Manifest: `/Users/steve/UVM-plaid/dp-vc/output/gender_followup_available_w005_2026-06-13/generation_manifest.jsonl`
- Audio rows: `12`
- Source groups: `4`
- Complete female/male F0 pairs: `4`
- Pairs where female-control median F0 > male-control median F0: `1/4`
- Median female-minus-male F0 delta: `-5.69 Hz`
- Median female-minus-male spectral-centroid delta: `-27.59 Hz`

## Source-Level F0 Pairs

| Source | Baseline F0 | Female-control F0 | Male-control F0 | Female - male |
| --- | ---: | ---: | ---: | ---: |
| `female_1_cremad_1002` | `202.2936` | `200.4711` | `213.0485` | `-12.5774` |
| `female_2_cremad_1012` | `210.0000` | `210.0000` | `220.5000` | `-10.5000` |
| `male_1_cremad_1003` | `157.7011` | `187.6630` | `182.2812` | `5.3818` |
| `male_2_cremad_1051` | `98.4375` | `98.4375` | `99.3243` | `-0.8868` |

## Interpretation

- This is a diagnostic acoustic proxy, not a perceptual gender-control result.
- A listener-clear gender control would usually be expected to move simple cues such as F0 in a consistent direction, but F0 alone is not sufficient for gender perception.
- Use this only to characterize the failed listening gate, not to promote the gender checkpoint.
