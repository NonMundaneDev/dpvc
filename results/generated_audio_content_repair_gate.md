# Generated-Audio Content-Repair Gate

This report applies a conservative gate to hard-style strength-grid candidates.
A candidate needs objective target gain plus content/naturalness/novelty safety, and if human ratings are present it must win perceptually before promotion.

Ratings: `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority_ratings_joe_2026-05-19.csv`

## Thresholds

- maximum candidate WER: `0.3`
- maximum WER increase over reference: `0.15`
- minimum candidate MOS delta: `-0.75`
- maximum MOS-delta drop: `0.25`
- minimum novelty delta: `-0.02`

## Decision Counts

| Decision | Count |
|----------|------:|
| `blocked_by_perceptual_tie` | 1 |
| `blocked_by_reference_preference` | 1 |
| `reject_no_target_gain` | 28 |
| `reject_quality` | 3 |

## Style Decisions

| Style | Rows | Objective-pass rows | Needs listening | Perceptually blocked | Promoted | Style decision |
|-------|-----:|--------------------:|----------------:|---------------------:|---------:|----------------|
| `anger` | 11 | 1 | 0 | 1 | 0 | `diagnostic_only` |
| `disgust` | 11 | 0 | 0 | 0 | 0 | `needs_content_repair` |
| `fear` | 11 | 1 | 0 | 1 | 0 | `diagnostic_only` |

## Promotion Result

No style-strength candidate is promoted by the current gate.

This means the grid remains diagnostic: it finds classifier gains, but those gains have not passed the combined content and perceptual standard.

## Needs More Listening

| Style | Source | Strength | Candidate WER | Candidate MOS delta | Novelty delta | Candidate |
|-------|--------|----------|--------------:|--------------------:|--------------:|-----------|
| - | - | - | - | - | - | - |

## Perceptually Blocked Objective-Pass Rows

| Style | Source | Strength | Human preference | Notes |
|-------|--------|----------|------------------|-------|
| `anger` | `cremad_1006` | `10` | `tie` | Joe: reference and candidate sound identical; no perceptual candidate advantage. |
| `fear` | `male_1_cremad_1003` | `7.5` | `reference` | Joe: reference preferred; candidate had an unnatural pitch change; difference was small. |

## Interpretation

- `anger_s10` and `fear_s7p5` still have useful diagnostic rows, but Joe's review blocks promotion because the perceived result was tie/reference, not candidate.
- `disgust` has no safe promoted repair under this gate; it needs a generated-audio/content objective rather than a stronger strength preset.
- The next training-side repair should optimize for generated-audio behavior directly instead of relying on embedding-space or classifier-only gains.
