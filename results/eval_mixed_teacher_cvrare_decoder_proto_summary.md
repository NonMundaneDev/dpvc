# Expanded Rare-Supply Decoder-Prototype Pilot Summary

Date: 2026-05-05

Branch: `research/controllable-vae`

## Purpose

Test whether a decoder-aware training objective can learn the quality-balanced
repair that the hand-authored `sad/enunciated` inference guard found for the
expanded rare-supply checkpoint.

## Compared Conditions

| Condition | Recall | Novelty gain | Mean styled WER | MOS delta | Content collapse | Style-to-neutral collapse | Identity collapse | Mixed collapse | Files with any collapse |
|-----------|--------|--------------|-----------------|-----------|------------------|---------------------------|-------------------|----------------|-------------------------|
| `combined` | `25.8%` | `0.2599` | `0.2353` | `-0.0792` | `6` | `37` | `7` | `4` | `46` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup` | `47.0%` | `0.2995` | `0.2751` | `-0.2640` | `7` | `18` | `1` | `1` | `25` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `47.0%` | `0.2726` | `0.2348` | `-0.2081` | `2` | `18` | `1` | `1` | `20` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `42.4%` | `0.3008` | `0.2863` | `-0.2148` | `4` | `22` | `1` | `0` | `27` |

## Decoder-Prototype Command

```bash
.venv/bin/python examples/openvoice_train_vae_mixed.py \
    --embeddings embeddings/openvoice_mixed_teacher_cvrare_hybrid_extra_base.pt \
    --output embeddings/openvoice_vae_mixed_teacher_cvrare_decoder_proto_labeled_warmup.pt \
    --init-checkpoint embeddings/openvoice_vae_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.pt \
    --epochs 1000 \
    --schedule labeled_warmup \
    --schedule-epochs 1000 \
    --style-teacher-checkpoint embeddings/openvoice_vae_combined.pt \
    --style-teacher-weight 0.0 \
    --style-teacher-weight-final 0.25 \
    --style-teacher-datasets CommonVoice \
    --style-teacher-dims 0-8 \
    --decoder-prototype-weight 0.0 \
    --decoder-prototype-weight-final 0.02 \
    --decoder-prototype-datasets CommonVoice \
    --decoder-prototype-source true \
    --decoder-prototype-strength 5.0 \
    --decoder-prototype-style-strengths sad=3.5,enunciated=2.5,confused=4.0 \
    --decoder-prototype-control-mode target_only
```

## Listening

Open the checked-in listening report:

- `results/listening_mixed_teacher_cvrare_decoder_proto_labeled_warmup.html`

For comparison, the current quality-balanced reference remains:

- `results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.html`

## Readout

The decoder-prototype objective is implemented and reproducible, but the first
pilot is not a new best model. It preserves high novelty and improves recall
over the original `combined` baseline, but it loses recall and worsens WER /
collapse versus the current `sad/enunciated` guard. The next training-side
objective should be safer and more directly calibrated against generated audio,
not only decoded embedding prototypes.
