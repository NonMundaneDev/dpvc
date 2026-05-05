# Mixed-Teacher Style Diagnostics: `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup`

This report joins training-artifact label supply, teacher/student latent geometry, and generated-output metrics by style.

## Diagnostic Table

| style | teacher rows | pseudo rows | teacher top1 | student top1 | recall | neutral pred | novelty | WER | MOS delta | any collapse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anger | 50 | 50 | 0.2600 | 0.3600 | 0.0909 | 0.6364 | 0.2962 | 0.1740 | -0.0532 | 7 |
| confused | 50 | 50 | 1.0000 | 0.8400 |  | 0.5455 | 0.3199 | 0.1556 | -0.4950 | 0 |
| disgust | 79 | 79 | 0.1899 | 0.1519 | 0.1818 | 0.8182 | 0.2296 | 0.1529 | -0.2315 | 9 |
| enunciated | 50 | 50 | 1.0000 | 0.9000 |  | 0.9091 | 0.3392 | 0.2484 | -0.9581 | 0 |
| fear | 50 | 50 | 0.1400 | 0.2000 | 0.2727 | 0.0000 | 0.3485 | 0.3461 | -0.3903 | 1 |
| happy | 80 | 80 | 0.2375 | 0.0625 | 0.4545 | 0.0909 | 0.1687 | 0.3981 | -0.0026 | 3 |
| neutral | 116 | 116 | 0.0948 | 0.0603 | 0.9091 | 0.9091 | 0.2351 | 0.1269 | -0.1472 | 0 |
| sad | 120 | 120 | 0.3167 | 0.1917 | 0.9091 | 0.0909 | 0.1158 | 0.5847 | -0.0311 | 5 |
| whisper | 50 | 50 | 1.0000 | 0.7200 |  | 0.0000 | 0.6430 | 0.2889 | -0.0668 | 0 |

## Automatic Readout

- Active teacher supply is not extremely sparse for any style under the current threshold.
- No emotion-scored style has both zero recall and at least 80% neutral predictions.
- Teacher latent target is often not the top style dim for: `anger` teacher-top1=0.2600, `disgust` teacher-top1=0.1899, `fear` teacher-top1=0.1400, `happy` teacher-top1=0.2375, `neutral` teacher-top1=0.0948, `sad` teacher-top1=0.3167.
- Student target-dim alignment trails the teacher strongly for: `whisper` teacher=1.0000 student=0.7200.

## Recommendation

- Do not spend the next turn on another latent-only scalar/mask variant without first addressing the diagnostic failure mode above.
- Labeled-first curriculum has already been tested for this condition; prefer decoder-aware/generated-audio style supervision or stronger rare-class supply next.
- Treat rare canonical classes as a data-supply issue, not merely a weighting issue.
