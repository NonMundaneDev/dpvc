# Mixed-Teacher Style Diagnostics: `mixed_teacher_hybrid_style_distill_labeled_warmup`

This report joins training-artifact label supply, teacher/student latent geometry, and generated-output metrics by style.

## Diagnostic Table

| style | teacher rows | pseudo rows | teacher top1 | student top1 | recall | neutral pred | novelty | WER | MOS delta | any collapse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anger | 4 | 4 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0051 | 0.0325 | 0.0034 | 11 |
| confused | 22 | 22 | 1.0000 | 0.6818 |  | 0.7273 | 0.2167 | 0.0766 | -0.2704 | 0 |
| disgust | 27 | 27 | 0.2222 | 0.1111 | 0.0000 | 1.0000 | 0.0274 | 0.0507 | 0.0166 | 11 |
| enunciated | 27 | 27 | 1.0000 | 0.7037 |  | 1.0000 | 0.0683 | 0.0571 | -0.1194 | 3 |
| fear | 4 | 4 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0137 | 0.1688 | -0.0239 | 11 |
| happy | 30 | 30 | 0.1000 | 0.2333 | 0.0000 | 1.0000 | -0.0048 | 0.1656 | -0.0259 | 11 |
| neutral | 66 | 66 | 0.0606 | 0.0000 | 1.0000 | 1.0000 | 0.0571 | 0.0507 | -0.0352 | 3 |
| sad | 78 | 78 | 0.2692 | 0.7692 | 0.0000 | 1.0000 | 0.0352 | 0.0455 | -0.0124 | 11 |
| whisper | 9 | 9 | 1.0000 | 0.7778 |  | 0.0909 | 0.4184 | 0.1838 | -0.5169 | 0 |

## Automatic Readout

- Rare active teacher supply: `anger=4`, `fear=4`, `whisper=9`. Weighting these rows cannot substitute for more examples.
- Emotion-scored styles still collapsing to neutral: `anger`, `disgust`, `fear`, `happy`, `sad`.
- Teacher latent target is often not the top style dim for: `anger` teacher-top1=0.0000, `disgust` teacher-top1=0.2222, `fear` teacher-top1=0.0000, `happy` teacher-top1=0.1000, `neutral` teacher-top1=0.0606, `sad` teacher-top1=0.2692.
- Student target-dim alignment trails the teacher strongly for: `confused` teacher=1.0000 student=0.6818, `enunciated` teacher=1.0000 student=0.7037.

## Recommendation

- Do not spend the next turn on another latent-only scalar/mask variant without first addressing the diagnostic failure mode above.
- Labeled-first curriculum has already been tested for this condition; prefer decoder-aware/generated-audio style supervision or stronger rare-class supply next.
- Treat rare canonical classes as a data-supply issue, not merely a weighting issue.
