# Style-Strength Grid Ranking

Condition prefix: `mixed_teacher_cvrare_strength_grid`

## Best Per Style

| Style | Strength | Recall | Novelty | WER | MOS delta | Style-to-neutral | Any collapse | Condition |
|-------|----------|--------|---------|-----|-----------|------------------|--------------|-----------|
| `anger` | `10` | `0.2727` | `0.3302` | `0.2081` | `-0.0408` | `7` | `7` | `mixed_teacher_cvrare_strength_grid_anger_s10` |
| `disgust` | `10` | `0.1818` | `0.3124` | `0.2831` | `-0.6039` | `8` | `8` | `mixed_teacher_cvrare_strength_grid_disgust_s10` |
| `fear` | `7.5` | `0.5455` | `0.4136` | `0.5231` | `-0.2972` | `1` | `3` | `mixed_teacher_cvrare_strength_grid_fear_s7p5` |

## Full Ranking

| Style | Rank | Strength | Recall | Novelty | WER | MOS delta | Style-to-neutral | Any collapse |
|-------|------|----------|--------|---------|-----|-----------|------------------|--------------|
| `anger` | `1` | `10` | `0.2727` | `0.3302` | `0.2081` | `-0.0408` | `7` | `7` |
| `anger` | `2` | `7.5` | `0.1818` | `0.3306` | `0.2563` | `-0.0016` | `7` | `7` |
| `anger` | `3` | `5` | `0.0909` | `0.2962` | `0.1740` | `-0.0532` | `7` | `7` |
| `anger` | `4` | `3` | `0.0909` | `0.1994` | `0.1843` | `-0.1862` | `10` | `10` |
| `disgust` | `1` | `10` | `0.1818` | `0.3124` | `0.2831` | `-0.6039` | `8` | `8` |
| `disgust` | `2` | `5` | `0.1818` | `0.2296` | `0.1529` | `-0.2315` | `9` | `9` |
| `disgust` | `3` | `7.5` | `0.1818` | `0.2859` | `0.3364` | `-0.4440` | `9` | `9` |
| `disgust` | `4` | `3` | `0.0000` | `0.1498` | `0.1607` | `-0.0042` | `11` | `11` |
| `fear` | `1` | `7.5` | `0.5455` | `0.4136` | `0.5231` | `-0.2972` | `1` | `3` |
| `fear` | `2` | `5` | `0.2727` | `0.3485` | `0.3331` | `-0.3903` | `0` | `0` |
| `fear` | `3` | `10` | `0.2727` | `0.4496` | `0.4873` | `-0.3532` | `2` | `3` |
| `fear` | `4` | `3` | `0.0909` | `0.2579` | `0.1935` | `-0.4154` | `5` | `5` |

## Ranking Rule

Rows are ranked within each style by: higher emotion recall, fewer files with any collapse, fewer style-to-neutral collapses, lower WER, higher MOS delta, then higher novelty.

Use this as an audio-calibrated candidate selector. A strength should still be checked perceptually before becoming a default.
