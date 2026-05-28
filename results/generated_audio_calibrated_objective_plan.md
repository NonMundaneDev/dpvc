# Generated-Audio-Calibrated Objective Plan

This plan converts generated-audio evidence into trainer overrides.

Reference condition: `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard`

## Selected Styles

`anger`, `disgust`

## Style Plan

| Style | Selected | Action | Weight | Strength | Reason |
|-------|----------|--------|-------:|---------:|--------|
| `anger` | `True` | `train_conservative_repair` | `2` | `5` | gate keeps preset diagnostic, but clean generated-audio failures exist |
| `disgust` | `True` | `train_content_repair` | `3` | `5` | gate found no safe preset, but clean generated-audio failures exist |
| `fear` | `False` | `blocked_no_clean_targets` | `0` | `0` | failure target selector found 0/11 clean rows |

## Trainer Overrides

- `style_teacher_target_mode`: `target_dim`
- `style_teacher_require_label`: `True`
- `style_teacher_style_weights`: `anger=2,confused=0,disgust=3,enunciated=0,fear=0,happy=0,neutral=0,sad=0,whisper=0`
- `decoder_prototype_style_weights`: `anger=2,confused=0,disgust=3,enunciated=0,fear=0,happy=0,neutral=0,sad=0,whisper=0`
- `decoder_prototype_style_strengths`: `anger=5,confused=0,disgust=5,enunciated=0,fear=0,happy=0,neutral=0,sad=0,whisper=0`
- `anti_neutral_styles`: `anger,disgust`
- `anti_neutral_style_weights`: `anger=2,confused=0,disgust=3,enunciated=0,fear=0,happy=0,neutral=0,sad=0,whisper=0`
- `anti_neutral_style_strengths`: `anger=5,confused=0,disgust=5,enunciated=0,fear=0,happy=0,neutral=0,sad=0,whisper=0`

## Recommended Command

```bash
python examples/openvoice_train_vae_mixed.py \
  --embeddings embeddings/openvoice_mixed_teacher_cvrare_hybrid_extra_base.pt \
  --output embeddings/openvoice_vae_mixed_teacher_cvrare_audio_calibrated_labeled_warmup.pt \
  --schedule labeled_warmup \
  --schedule-epochs 1500 \
  --epochs 3000 \
  --lr 1e-6 \
  --style-teacher-checkpoint embeddings/openvoice_vae_combined.pt \
  --style-teacher-weight 0.0 \
  --style-teacher-weight-final 0.25 \
  --decoder-prototype-weight 0.0 \
  --decoder-prototype-weight-final 0.005 \
  --decoder-prototype-datasets CommonVoice \
  --decoder-prototype-source true \
  --anti-neutral-weight 0.0 \
  --anti-neutral-weight-final 0.005 \
  --anti-neutral-mode prototype_margin \
  --anti-neutral-datasets CommonVoice \
  --generated-audio-objective-plan results/generated_audio_calibrated_objective_plan.json \
  --generated-audio-objective-report results/generated_audio_calibrated_objective_training_report.json
```

## Interpretation

- `anger` receives conservative repair pressure because it has clean failures, but the strength preset did not win perceptually.
- `disgust` receives content-repair pressure because no safe strength-grid row passed the gate.
- `fear` remains blocked from this training objective because its clean-failure target supply is not established and Joe heard an unnatural pitch change in the best metric row.
