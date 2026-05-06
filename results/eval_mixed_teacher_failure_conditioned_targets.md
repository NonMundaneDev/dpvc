# Failure-Conditioned Target Selection

Failure CSV: `results/eval_mixed_teacher_generated_audio_failure_mining.csv`

Reference condition: `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard`

## Selection Rule

- Target styles: `anger, disgust, fear`
- Required modes: `emotion_miss, style_to_neutral`
- Exclude modes: `content_collapse, high_wer, low_mos_delta, identity_collapse, mixed_collapse, low_novelty`
- Ready style threshold: at least `3` clean selected rows in the reference condition

A selected row is a clean style-control failure: the generated audio missed the target emotion by collapsing to neutral, but it does not also have high WER, low MOS, low novelty, content collapse, identity collapse, or mixed collapse.

## Reference Readout

| Style | Target rows | Selected clean targets | Status |
|-------|-------------|------------------------|--------|
| `anger` | `11` | `5` | `ready` |
| `disgust` | `11` | `6` | `ready` |
| `fear` | `11` | `0` | `blocked` |

## Condition Counts

| Condition | Target rows | Selected | Missing required modes | Confounded failure |
|-----------|-------------|----------|------------------------|--------------------|
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `33` | `11` | `11` | `11` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup_sad_enunc_guard` | `33` | `11` | `11` | `11` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `33` | `12` | `11` | `10` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `33` | `11` | `17` | `5` |

## Selected Reference Rows

| Speaker | Style | Predicted | Target | WER | MOS delta | Novelty | File |
|---------|-------|-----------|--------|-----|-----------|---------|------|
| `cremad_1023` | `anger` | `neutral` | `angry` | `0.0714` | `+0.0151` | `0.2830` | `cremad_1023_anger.wav` |
| `cremad_1045` | `anger` | `neutral` | `angry` | `0.1429` | `+0.0316` | `0.3009` | `cremad_1045_anger.wav` |
| `cremad_1076` | `anger` | `neutral` | `angry` | `0.0000` | `+0.0197` | `0.2971` | `cremad_1076_anger.wav` |
| `female_2_cremad_1012` | `anger` | `neutral` | `angry` | `0.2000` | `+0.0660` | `0.2978` | `female_2_cremad_1012_anger.wav` |
| `male_1_cremad_1003` | `anger` | `neutral` | `angry` | `0.0000` | `-0.0055` | `0.2287` | `male_1_cremad_1003_anger.wav` |
| `cremad_1006` | `disgust` | `neutral` | `disgusted` | `0.0000` | `+0.0288` | `0.2396` | `cremad_1006_disgust.wav` |
| `cremad_1023` | `disgust` | `neutral` | `disgusted` | `0.2143` | `+0.0150` | `0.1672` | `cremad_1023_disgust.wav` |
| `cremad_1045` | `disgust` | `neutral` | `disgusted` | `0.1429` | `+0.0398` | `0.1245` | `cremad_1045_disgust.wav` |
| `cremad_1076` | `disgust` | `neutral` | `disgusted` | `0.0000` | `+0.0497` | `0.2780` | `cremad_1076_disgust.wav` |
| `female_1_cremad_1002` | `disgust` | `neutral` | `disgusted` | `0.0000` | `+0.0112` | `0.2858` | `female_1_cremad_1002_disgust.wav` |
| `male_1_cremad_1003` | `disgust` | `neutral` | `disgusted` | `0.0000` | `-0.4472` | `0.2799` | `male_1_cremad_1003_disgust.wav` |

## Recommended Training Focus

- Ready styles: `anger`, `disgust`
- Blocked styles: `fear` (`0/11` selected)

Suggested existing-trainer argument for a target-dim style-teacher follow-up:

```bash
--style-teacher-target-mode target_dim \
--style-teacher-require-label \
--style-teacher-style-weights anger=3,confused=0,disgust=3,enunciated=0,fear=0,happy=0,neutral=0,sad=0,whisper=0
```

If the next run uses decoder-prototype controls, the analogous ready-style strength map is:

```bash
--decoder-prototype-style-strengths anger=5.0,confused=0.0,disgust=5.0,enunciated=0.0,fear=0.0,happy=0.0,neutral=0.0,sad=0.0,whisper=0.0
```

## Interpretation

- `anger` and `disgust` are ready for a conservative positive target objective under the current reference condition.
- `fear` remains blocked under the current reference because its failures are usually confounded with high WER, low MOS, or non-neutral emotion confusions.
- The next experiment should be targeted and conservative: improve clean neutral-collapse failures without turning content-damaged examples into positive style targets.
