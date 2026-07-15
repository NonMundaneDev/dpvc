# Control Recovery: Legacy Combined vs Current Reference Guard

This study compares two already-generated, fully matched evaluation corpora.
It is a recovery/regression diagnostic, not a substitute for listening.

## Scope

- Legacy condition: `legacy_combined`
- Current condition: `current_reference_guard`
- Matched sources: `11`
- Styled outputs per condition: `99`
- Inference seed/noise: matched (`42`, `0.0`)
- Legacy style strength: `5.0` for every style
- Current guard uses its checked-in per-style strength profile

## Per-Style Comparison

| Style | Legacy recall | Current recall | Legacy WER | Current WER | Legacy MOS delta | Current MOS delta | Legacy novelty | Current novelty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `anger` | 0.2727 | 0.0909 | 0.1594 | 0.1740 | -0.2519 | -0.0532 | 0.1506 | 0.2962 |
| `confused` | n/a | n/a | 0.3584 | 0.2152 | -0.2972 | -0.4156 | 0.5589 | 0.2764 |
| `disgust` | 0.0909 | 0.1818 | 0.1854 | 0.1529 | -0.0272 | -0.2315 | 0.1256 | 0.2296 |
| `enunciated` | n/a | n/a | 0.3976 | 0.1542 | -0.0179 | -0.5339 | 0.4495 | 0.1686 |
| `fear` | 0.0000 | 0.2727 | 0.1740 | 0.3071 | 0.0218 | -0.3903 | 0.1028 | 0.3485 |
| `happy` | 0.0000 | 0.4545 | 0.1153 | 0.3981 | -0.0313 | -0.0026 | 0.0795 | 0.1687 |
| `neutral` | 0.8182 | 0.9091 | 0.1201 | 0.1269 | -0.0048 | -0.1472 | 0.0995 | 0.2351 |
| `sad` | 0.3636 | 0.9091 | 0.1594 | 0.2955 | -0.0028 | -0.0318 | 0.1163 | 0.0877 |
| `whisper` | n/a | n/a | 0.4478 | 0.2889 | -0.1013 | -0.0668 | 0.6561 | 0.6430 |

## Interpretation Rules

- Recall applies only to the six emotion2vec-aligned emotions.
- Whisper, confused, and enunciated require listening; their recall is `n/a`.
- Lower WER is better. A higher MOS delta is better. Higher novelty means more identity movement, not necessarily better control.
- A control is recovered only if the intended attribute is audible and intelligibility/naturalness remain acceptable.
- The matched listening dashboard is the decision gate for `anger`, `happy`, `neutral`, `sad`, and `whisper`.
