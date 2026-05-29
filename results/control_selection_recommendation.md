# Control Selection Recommendation

This recommendation intersects source training-label separability, generated-output metrics for the current `sad/enunciated` reference guard, and available human listening evidence.
It is deliberately conservative: a source-separable label is not promoted unless generated audio and listening evidence also support the claim.

## Inputs

- Source separability: `results/training_style_separability_by_label.csv`
- Generated emotion metrics: `results/eval_emotion_mixed_teacher_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.csv`
- Generated WER metrics: `results/eval_wer_mixed_teacher_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.csv`
- Generated MOS metrics: `results/eval_mos_mixed_teacher_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.csv`
- External speaker novelty: `results/eval_external_speaker_verifier_cvrare_sad_enunc_guard.csv`
- Collapse diagnostics: `results/eval_mixed_teacher_collapse.csv` (`mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard`)
- Perceptual evidence ledger: `results/control_selection_perceptual_evidence.csv`

## Bucket Counts

| Bucket | Count |
| --- | ---: |
| `candidate_headline_pending_listening` | `2` |
| `supported_but_quality_sensitive` | `3` |
| `diagnostic_or_limitation` | `4` |

## Recommendation Table

| Style | Bucket | Source gate | Generated gate | Generated recall | WER | MOS delta | External novelty | Perceptual status | Claim status |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `neutral` | `candidate_headline_pending_listening` | `source_separable` | `generated_strong` | 0.9091 | 0.1269 | -0.1472 | 0.2716 | `needs_review` | candidate headline control; needs focused listening before claim |
| `sad` | `candidate_headline_pending_listening` | `source_separable` | `generated_strong` | 0.9091 | 0.2955 | -0.0318 | 0.2834 | `needs_review` | candidate headline control; needs focused listening before claim |
| `enunciated` | `supported_but_quality_sensitive` | `source_supported_quality_sensitive` | `generated_nonemotion_quality_sensitive` | n/a | 0.1542 | -0.5339 | 0.3100 | `needs_review` | secondary or demo-only candidate; not headline yet |
| `happy` | `supported_but_quality_sensitive` | `source_separable` | `generated_mixed` | 0.4545 | 0.3981 | -0.0026 | 0.4508 | `needs_review` | secondary or demo-only candidate; not headline yet |
| `whisper` | `supported_but_quality_sensitive` | `source_supported_quality_sensitive` | `generated_nonemotion_supported` | n/a | 0.2889 | -0.0668 | 0.4434 | `needs_review` | secondary or demo-only candidate; not headline yet |
| `anger` | `diagnostic_or_limitation` | `source_separable` | `generated_diagnostic` | 0.0909 | 0.1740 | -0.0532 | 0.3563 | `blocked_or_subtle` | do not use as a headline claim |
| `confused` | `diagnostic_or_limitation` | `source_weak` | `generated_nonemotion_quality_sensitive` | n/a | 0.2152 | -0.4156 | 0.3121 | `needs_review` | do not use as a headline claim |
| `disgust` | `diagnostic_or_limitation` | `source_separable` | `generated_diagnostic` | 0.1818 | 0.1529 | -0.2315 | 0.3612 | `blocked` | do not use as a headline claim |
| `fear` | `diagnostic_or_limitation` | `source_separable` | `generated_diagnostic` | 0.2727 | 0.3071 | -0.3903 | 0.4454 | `blocked` | do not use as a headline claim |

## Interpretation

No style control is promoted as fully paper-ready by this conservative gate yet, because no current style has both strong objective evidence and positive focused listening evidence recorded.

Candidate headline controls needing focused listening:

- `neutral`: No focused human review is recorded; generated metrics are strong, so this should be listened to before a headline control claim.
- `sad`: No focused human review is recorded; generated metrics are strong, so this should be listened to before a headline control claim.

Supported but quality-sensitive controls:

- `enunciated`: source supported quality sensitive; generated nonemotion quality sensitive; perceptual status: needs_review
- `happy`: source separable; generated mixed; perceptual status: needs_review
- `whisper`: source supported quality sensitive; generated nonemotion supported; perceptual status: needs_review

Diagnostic / limitation controls:

- `anger`: Joe heard objective-pass anger strength candidates as ties, and later audio-calibrated anger as only subtle/source-dependent.
- `confused`: No focused human review is recorded; source separability is weak, so keep diagnostic unless a perceptual demo story emerges.
- `disgust`: Joe heard generated disgust as neutral across the focused panel; Stephen heard it as promising, but the result is not confirmed.
- `fear`: Joe preferred the reference for the objective-pass fear row because the candidate had an unnatural pitch change; other reviewed rows tied.

## Next Listening Queue

Use the existing browser report for the current reference guard:

```bash
python3 -m http.server 8000
# open http://localhost:8000/results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.html
```

Listen first to the `candidate_headline_pending_listening` rows, then the quality-sensitive rows. A control should become a headline claim only if a listener can hear the intended control without major intelligibility or naturalness loss.
