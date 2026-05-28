# External Speaker-Verifier Novelty Summary

- Backend: `speechbrain-ecapa`
- Manifest: `/Users/steve/UVM-plaid/dp-vc/output/mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard_eval/generation_manifest.jsonl`
- Output CSV: `/Users/steve/UVM-plaid/dp-vc/results/eval_external_speaker_verifier_cvrare_sad_enunc_guard.csv`
- Rows: `110`
- Styled rows: `99`
- Trial source: `derived_proxy_trials`
- Trial count: `121`
- EER: `0.0000`
- Threshold: `0.4647`
- FAR at threshold: `0.0000`
- FRR at threshold: `0.0000`
- Positive trials: `11`
- Negative trials: `110`
- Mean styled novelty gain vs baseline: `0.3594`
- Styled accept-as-source rate: `0.0606`

## Per-Style Summary

| Style | Rows | Mean external similarity | Mean novelty gain vs baseline | Accept-as-source rate |
|-------|------|--------------------------|-------------------------------|-----------------------|
| `anger` | `11` | `0.2633` | `0.3563` | `0.0000` |
| `confused` | `11` | `0.3075` | `0.3121` | `0.0909` |
| `disgust` | `11` | `0.2584` | `0.3612` | `0.0909` |
| `enunciated` | `11` | `0.3095` | `0.3100` | `0.0909` |
| `fear` | `11` | `0.1742` | `0.4454` | `0.0000` |
| `happy` | `11` | `0.1688` | `0.4508` | `0.0000` |
| `neutral` | `11` | `0.3480` | `0.2716` | `0.1818` |
| `sad` | `11` | `0.3362` | `0.2834` | `0.0909` |
| `whisper` | `11` | `0.1762` | `0.4434` | `0.0000` |

## Interpretation Rules

- Lower `external_similarity` means the verifier sees the generated voice as
  farther from the source speaker.
- Positive `external_novelty_gain_vs_baseline` means the style output moved
  farther from the source than the same-source baseline conversion did.
- `accepted_as_source_at_threshold` is only populated when an EER threshold
  is available from `--trial-csv` or `--derive-proxy-trials`.
- Proxy trials are diagnostic. Paper-facing EER should use an independent
  labeled trial CSV.
