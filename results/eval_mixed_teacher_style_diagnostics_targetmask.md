# Mixed-Teacher Style Diagnostics: `mixed_teacher_hybrid_style_distill_targetmask_balanced`

This report joins training-artifact label supply, teacher/student latent geometry, and generated-output metrics by style.

## Diagnostic Table

| style | teacher rows | pseudo rows | teacher top1 | student top1 | recall | neutral pred | novelty | WER | MOS delta | any collapse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anger | 4 | 4 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.0020 | 0.0325 | -0.0011 | 11 |
| confused | 22 | 22 | 1.0000 | 0.6818 |  | 0.8182 | 0.2411 | 0.0831 | -0.3444 | 0 |
| disgust | 27 | 27 | 0.2222 | 0.0741 | 0.0000 | 1.0000 | 0.0251 | 0.0507 | 0.0110 | 11 |
| enunciated | 27 | 27 | 1.0000 | 0.8519 |  | 1.0000 | 0.0704 | 0.1318 | -0.1399 | 2 |
| fear | 4 | 4 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.0036 | 0.1071 | -0.0320 | 11 |
| happy | 30 | 30 | 0.1000 | 0.2333 | 0.0000 | 0.9091 | -0.0011 | 0.2089 | -0.0210 | 10 |
| neutral | 66 | 66 | 0.0606 | 0.0000 | 1.0000 | 1.0000 | 0.0420 | 0.0961 | -0.0062 | 7 |
| sad | 78 | 78 | 0.2692 | 0.7308 | 0.0000 | 1.0000 | 0.0232 | 0.0617 | -0.0148 | 11 |
| whisper | 9 | 9 | 1.0000 | 0.7778 |  | 0.3636 | 0.3714 | 0.1838 | -0.5144 | 0 |

## Automatic Readout

- Rare active teacher supply: `anger=4`, `fear=4`, `whisper=9`. Weighting these rows cannot substitute for more examples.
- Emotion-scored styles still collapsing to neutral: `anger`, `disgust`, `fear`, `happy`, `sad`.
- Teacher latent target is often not the top style dim for: `anger` teacher-top1=0.0000, `disgust` teacher-top1=0.2222, `fear` teacher-top1=0.0000, `happy` teacher-top1=0.1000, `neutral` teacher-top1=0.0606, `sad` teacher-top1=0.2692.
- Student target-dim alignment trails the teacher strongly for: `confused` teacher=1.0000 student=0.6818.

## Recommendation

- Do not spend the next turn on another latent-only scalar/mask variant without first addressing the diagnostic failure mode above.
- Prefer a labeled-first curriculum if student latent alignment trails the teacher, or a decoder-aware/generated-audio style objective if latent alignment exists but emotion recall remains neutral.
- Treat rare canonical classes as a data-supply issue, not merely a weighting issue.
