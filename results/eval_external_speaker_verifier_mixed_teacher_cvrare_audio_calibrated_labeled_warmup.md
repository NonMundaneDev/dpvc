# External Speaker-Verifier Novelty Summary

- Backend: `speechbrain-ecapa`
- Manifest: `/Users/steve/UVM-plaid/dp-vc/output/mixed_teacher_cvrare_audio_calibrated_labeled_warmup_eval/generation_manifest.jsonl`
- Output CSV: `/Users/steve/UVM-plaid/dp-vc/results/eval_external_speaker_verifier_mixed_teacher_cvrare_audio_calibrated_labeled_warmup.csv`
- Rows: `110`
- Styled rows: `99`
- Trial source: `derived_proxy_trials`
- Trial count: `121`
- EER: `0.0000`
- Threshold: `0.4473`
- FAR at threshold: `0.0000`
- FRR at threshold: `0.0000`
- Positive trials: `11`
- Negative trials: `110`
- Mean styled novelty gain vs baseline: `0.3336`
- Styled accept-as-source rate: `0.0909`

## Per-Style Summary

| Style | Rows | Mean external similarity | Mean novelty gain vs baseline | Accept-as-source rate |
|-------|------|--------------------------|-------------------------------|-----------------------|
| `anger` | `11` | `0.3209` | `0.3016` | `0.0909` |
| `confused` | `11` | `0.2982` | `0.3243` | `0.0909` |
| `disgust` | `11` | `0.3550` | `0.2675` | `0.1818` |
| `enunciated` | `11` | `0.2836` | `0.3390` | `0.0000` |
| `fear` | `11` | `0.2135` | `0.4090` | `0.0000` |
| `happy` | `11` | `0.2313` | `0.3913` | `0.0000` |
| `neutral` | `11` | `0.4362` | `0.1864` | `0.4545` |
| `sad` | `11` | `0.2711` | `0.3514` | `0.0000` |
| `whisper` | `11` | `0.1905` | `0.4321` | `0.0000` |

## Interpretation Rules

- Lower `external_similarity` means the verifier sees the generated voice as
  farther from the source speaker.
- Positive `external_novelty_gain_vs_baseline` means the style output moved
  farther from the source than the same-source baseline conversion did.
- `accepted_as_source_at_threshold` is only populated when an EER threshold
  is available from `--trial-csv` or `--derive-proxy-trials`.
- Proxy trials are diagnostic. Paper-facing EER should use an independent
  labeled trial CSV.
