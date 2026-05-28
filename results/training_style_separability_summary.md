# Training Style Separability Audit

This audit evaluates source training clips, not generated outputs. It is a
control-selection diagnostic: labels that are weak in the source data should
not become open-ended repair targets without stronger data.

## Configuration

- Datasets: `cremad,expresso`
- CREMA-D mode: `speaker_emotion`
- Expresso mode: `unified_balanced`
- Expresso-only cap: `90`
- Max per label: `none`
- Max total: `none`
- Minimum classifier class count: `5`
- Seed: `42`
- Progress print interval: `25`

## Artifacts

- Rows: `results/training_style_separability_rows.csv`
- Per-label summary: `results/training_style_separability_by_label.csv`
- Confusion tables: `results/training_style_separability_confusion.csv`

## Per-Style Results

| Style | Support | Datasets | emotion2vec target | Direct recall | Embedding F1 | Verdict |
| --- | ---: | --- | --- | ---: | ---: | --- |
| anger | 91 | CREMA-D | angry | 0.9451 | 0.9545 | headline candidate |
| confused | 90 | Expresso | n/a | n/a | 0.2703 | diagnostic / weak |
| disgust | 91 | CREMA-D | disgusted | 0.9341 | 0.9091 | headline candidate |
| enunciated | 90 | Expresso | n/a | n/a | 0.5000 | supported but quality-sensitive |
| fear | 91 | CREMA-D | fearful | 0.7692 | 0.7027 | headline candidate |
| happy | 95 | CREMA-D,Expresso | happy | 0.9053 | 0.8889 | headline candidate |
| neutral | 95 | CREMA-D,Expresso | neutral | 0.9263 | 0.9091 | headline candidate |
| sad | 95 | CREMA-D,Expresso | sad | 0.8842 | 0.7925 | headline candidate |
| whisper | 90 | Expresso | n/a | n/a | 0.5161 | supported but quality-sensitive |

## Initial Recommendation

Headline candidates from this audit:

- `anger` (headline candidate)
- `disgust` (headline candidate)
- `fear` (headline candidate)
- `happy` (headline candidate)
- `neutral` (headline candidate)
- `sad` (headline candidate)

## Interpretation Notes

- `Direct recall` only applies to styles with an emotion2vec output label.
- `Embedding F1` is a held-out nearest-centroid classifier over emotion2vec embeddings.
- A high embedding F1 for styles such as `whisper` means the style is separable in emotion2vec space even without a direct emotion2vec class.
- This audit does not prove generated control quality; it decides which source labels are fair to claim or repair.
