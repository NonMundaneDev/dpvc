# Style-Strength A/B Review Priority

This is an objective-assisted triage sheet for the listening dashboard.
It is not a substitute for perceptual review.

Priority rows: `33`
Ratings file: `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority_ratings_joe_2026-05-19.csv`

## Bucket Counts

| Bucket | Count |
|--------|-------|
| `clean_target_gain` | `2` |
| `metric_trap` | `8` |
| `novelty_gain_no_recall_gain` | `11` |
| `target_gain_quality_risk` | `3` |
| `tie_or_minor_change` | `9` |

## Counts By Style

### `anger`

| Bucket | Count |
|--------|-------|
| `clean_target_gain` | `1` |
| `novelty_gain_no_recall_gain` | `4` |
| `target_gain_quality_risk` | `1` |
| `tie_or_minor_change` | `5` |

### `disgust`

| Bucket | Count |
|--------|-------|
| `metric_trap` | `5` |
| `novelty_gain_no_recall_gain` | `5` |
| `tie_or_minor_change` | `1` |

### `fear`

| Bucket | Count |
|--------|-------|
| `clean_target_gain` | `1` |
| `metric_trap` | `3` |
| `novelty_gain_no_recall_gain` | `2` |
| `target_gain_quality_risk` | `2` |
| `tie_or_minor_change` | `3` |

## Listen First

| Priority | Style | Source | Bucket | Ref match | Cand match | Cand WER | Cand MOS delta | Recommendation |
|----------|-------|--------|--------|-----------|------------|----------|----------------|----------------|
| `1` | `anger` | `cremad_1006` | `clean_target_gain` | `0` | `1` | `0.0000` | `-0.5494` | listen first; candidate may be preset-worthy if it sounds natural |
| `1` | `fear` | `male_1_cremad_1003` | `clean_target_gain` | `0` | `1` | `0.0000` | `-0.6636` | listen first; candidate may be preset-worthy if it sounds natural |
| `2` | `anger` | `female_1_cremad_1002` | `target_gain_quality_risk` | `0` | `1` | `0.7500` | `-0.0111` | listen early; candidate gains target label but quality/content may block preset |
| `2` | `fear` | `cremad_1003` | `target_gain_quality_risk` | `0` | `1` | `1.1429` | `-0.3900` | listen early; candidate gains target label but quality/content may block preset |
| `2` | `fear` | `female_1_cremad_1002` | `target_gain_quality_risk` | `0` | `1` | `0.7500` | `-0.0063` | listen early; candidate gains target label but quality/content may block preset |

## Likely Reject / Metric-Trap Rows

| Style | Source | Bucket | Ref match | Cand match | WER delta | MOS delta delta |
|-------|--------|--------|-----------|------------|-----------|-----------------|
| `disgust` | `cremad_1003` | `metric_trap` | `1` | `1` | `0.0000` | `-1.6204` |
| `disgust` | `cremad_1004` | `metric_trap` | `1` | `1` | `0.0000` | `-1.0516` |
| `disgust` | `cremad_1023` | `metric_trap` | `0` | `0` | `0.4286` | `-0.0244` |
| `disgust` | `female_2_cremad_1012` | `metric_trap` | `0` | `0` | `0.2000` | `-0.1361` |
| `disgust` | `male_2_cremad_1051` | `metric_trap` | `0` | `0` | `0.0000` | `-0.5079` |
| `fear` | `cremad_1004` | `metric_trap` | `1` | `1` | `0.4285` | `0.0823` |
| `fear` | `cremad_1076` | `metric_trap` | `1` | `1` | `1.0000` | `-0.0800` |
| `fear` | `female_2_cremad_1012` | `metric_trap` | `0` | `0` | `0.0000` | `0.0086` |

## Human Ratings Summary

Filled rows: `5`

| Preference | Count |
|------------|-------|
| `reference` | `1` |
| `tie` | `4` |

### Preference By Style

- `anger`: {'tie': 2}
- `fear`: {'reference': 1, 'tie': 2}

## Recommended Use

1. Start with priority `1` and `2` rows in the A/B dashboard.
2. Prefer a candidate only if human listening agrees with the objective gain.
3. Do not promote `disgust_s10` unless listening contradicts the MOS warning.
