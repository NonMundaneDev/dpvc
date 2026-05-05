# Expanded Rare-Supply Strength Profile Summary

Date: 2026-05-05
Branch: `research/controllable-vae`

This compares the unguarded expanded rare-supply checkpoint against two
per-style inference strength profiles.

| Condition | Recall | Novelty gain | Mean styled WER | MOS delta | Content collapse | Style-to-neutral collapse | Identity collapse | Mixed collapse | Files with any collapse | Readout |
|-----------|--------|--------------|-----------------|-----------|------------------|---------------------------|-------------------|----------------|-------------------------|---------|
| `combined` | `25.8%` | `0.2599` | `0.2353` | `-0.0792` | `6` | `37` | `7` | `4` | `46` | cleanest original quality baseline |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup` | `47.0%` | `0.2995` | `0.2751` | `-0.2640` | `7` | `18` | `1` | `1` | `25` | strongest novelty, but quality/content cost |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_content_guard` | `40.9%` | `0.2653` | `0.2133` | `-0.1989` | `1` | `24` | `1` | `2` | `24` | repairs quality most, but loses recall |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `47.0%` | `0.2726` | `0.2348` | `-0.2081` | `2` | `18` | `1` | `1` | `20` | best current recall/quality Pareto point |

Recommended listening report:

- `results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.html`

Profile command:

```bash
python scripts/run_ablation_inference.py \
  --source-dir examples/source_speakers/ \
  --condition mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup \
  --out output/mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard_eval \
  --style-strength 5.0 \
  --style-strength-map configs/style_strength_profiles/cvrare_sad_enunc_guard.json \
  --noise-level 0.0 \
  --seed 42
```

Evaluation command:

```bash
python scripts/run_generated_audio_eval_suite.py \
  --input output/mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard_eval \
  --result-tag mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard \
  --input-tag mixed_teacher
```
