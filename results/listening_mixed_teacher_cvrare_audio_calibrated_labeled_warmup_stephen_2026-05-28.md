# Stephen Perceptual Review — Audio-Calibrated Checkpoint

Date: 2026-05-28

Panel:

- `results/listening_mixed_teacher_cvrare_audio_calibrated_labeled_warmup.html`

Checkpoint:

- `mixed_teacher_cvrare_audio_calibrated_labeled_warmup`

Scope:

- Informal first-listener review of the `anger` and `disgust` styles.
- This is not yet a multi-listener perceptual result; use it to decide the next review queue.

Feedback:

| Style | Perceptual read | Intelligibility | Interpretation |
|-------|-----------------|-----------------|----------------|
| `disgust` | Sounds like the speaker is disgusted | Good / intelligible | Promising perceptual success despite emotion2vec reporting `0/11` recall |
| `anger` | Some anger/style change is audible | Weaker; speech is distorted and less intelligible | Emotion/style pressure is present, but content preservation is the bottleneck |

Takeaway:

- Do not treat the `disgust` classifier failure as a complete perceptual failure until Joe or another listener reviews the same panel.
- The next perceptual-review queue should ask Joe to compare `disgust` rows first, then `anger` rows.
- The next training/selection repair should focus on preserving intelligibility for `anger`, while validating whether `disgust` needs metric calibration more than stronger style pressure.
