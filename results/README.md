# Evaluation Results

Raw per-file evaluation outputs backing the numbers in [`FINDINGS.md`](../FINDINGS.md).
These are the full 258-file sweep over 27 speaker/variant configurations run on
2026-04-17 using the combined CREMA-D + Expresso VAE (`embeddings/openvoice_vae_combined.pt`).

| File | Rows | Script | Backs |
|------|------|--------|-------|
| `eval_emotion_full.csv` | 258 | [`examples/eval_emotion.py`](../examples/eval_emotion.py) | Finding 7 (Recall Rate, emo_sim) |
| `eval_wer_full.csv`     | 258 | [`examples/eval_wer.py`](../examples/eval_wer.py)         | Finding 8 (drift-from-baseline WER) |
| `eval_mos_full.csv`     | 258 | [`examples/eval_mos.py`](../examples/eval_mos.py)         | Finding 9 (SQUIM_SUBJECTIVE predicted MOS) |

CommonVoice metadata-control preflight artifacts from 2026-05-27:

Local perceptual review on 2026-05-28 found that the first metadata-control
panel sounded identical or mostly like generic speaker/timbre shifts. These
artifacts are therefore diagnostic engineering evidence, not paper-facing proof
of perceptible age/gender control.

| File | Rows | Script | Backs |
|------|------|--------|-------|
| `commonvoice_metadata_controls_audit.md` | local corpus + extracted artifact summary | [`scripts/audit_commonvoice_metadata_controls.py`](../scripts/audit_commonvoice_metadata_controls.py) | Worklog-only preflight for `research/commonvoice-metadata-controls` |
| `commonvoice_metadata_controls_audit.json` | local corpus + extracted artifact summary | [`scripts/audit_commonvoice_metadata_controls.py`](../scripts/audit_commonvoice_metadata_controls.py) | Machine-readable audit for metadata-control training setup |
| `openvoice_vae_mixed_teacher_cvrare_metadata_w010_labeled_warmup_report.json` | metadata-control training config | [`examples/openvoice_train_vae_mixed.py`](../examples/openvoice_train_vae_mixed.py) | Worklog-only first metadata-control checkpoint record |
| `listening_metadata_w010_labeled_warmup.html` | 10-row browser listening panel | [`scripts/build_listening_report.py`](../scripts/build_listening_report.py) | Diagnostic negative perceptual smoke review for age/gender controls |
| `listening_metadata_w010_labeled_warmup_ratings.csv` | subjective rating template | [`scripts/build_listening_report.py`](../scripts/build_listening_report.py) | Rating sheet for the metadata-control smoke panel |
| `listening_metadata_w010_labeled_warmup.md` | listening instructions, outcome, and interpretation rules | manual summary | Worklog-only perceptual smoke outcome |

CommonVoice gender-only available-subset follow-up artifacts from 2026-06-13:

This is the first gender-only candidate after the neutral/sad control gate.
Local listening on 2026-07-09 found that the `female` / `male` controls were
not reliably perceptible as gender controls and sounded closer to subtle or
generic speaker/timbre movement. These artifacts are therefore diagnostic
engineering evidence, not paper-facing proof of perceptible gender control. The
post-listening acoustic sanity check reached the same conservative read: the
`female` control had higher median F0 than the `male` control in only `1/4`
source pairs. The run also reuses the already-extracted expanded CommonVoice
artifact and therefore matches `1181/1788` preflight manifest clips, not the
full preflight-selected plan.

| File | Rows | Script | Backs |
|------|------|--------|-------|
| `commonvoice_gender_followup_artifact.md` | summary | [`scripts/build_commonvoice_gender_followup_artifact.py`](../scripts/build_commonvoice_gender_followup_artifact.py) | Worklog-only available-subset artifact report |
| `commonvoice_gender_followup_artifact.json` | summary | [`scripts/build_commonvoice_gender_followup_artifact.py`](../scripts/build_commonvoice_gender_followup_artifact.py) | Machine-readable available-subset coverage report |
| `openvoice_vae_mixed_gender_followup_available_w005_labeled_warmup_report.json` | training config | [`examples/openvoice_train_vae_mixed.py`](../examples/openvoice_train_vae_mixed.py) | Gender-only candidate checkpoint record |
| `listening_gender_followup_available_w005_labeled_warmup.html` | 4-source browser listening panel | [`scripts/build_listening_report.py`](../scripts/build_listening_report.py) | Perceptual gate for the gender-only candidate |
| `listening_gender_followup_available_w005_labeled_warmup_ratings.csv` | 8 scoreable rows | [`scripts/build_listening_report.py`](../scripts/build_listening_report.py) | Rating sheet for female/male controls |
| `listening_gender_followup_available_w005_labeled_warmup_stephen_2026-07-09.md` | local listening outcome | manual perceptual review | FINDINGS Finding 43 |
| `gender_followup_available_w005_acoustic_diagnostic.md` | 4-source summary | [`scripts/analyze_gender_followup_acoustics.py`](../scripts/analyze_gender_followup_acoustics.py) | Diagnostic acoustic sanity check after failed listening gate |
| `gender_followup_available_w005_acoustic_diagnostic.csv` | 12 audio rows | [`scripts/analyze_gender_followup_acoustics.py`](../scripts/analyze_gender_followup_acoustics.py) | Per-output F0 / centroid diagnostic |
| `gender_followup_available_w005_acoustic_pairs.csv` | 4 source pairs | [`scripts/analyze_gender_followup_acoustics.py`](../scripts/analyze_gender_followup_acoustics.py) | Source-level female-vs-male F0 / centroid deltas |
| `gender_followup_available_w005_review_bundle_2026-06-13.zip` | 28 files | manual bundle | Self-contained local/Joe review bundle |

External speaker-verifier artifacts from 2026-05-28:

| File | Rows | Script | Backs |
|------|------|--------|-------|
| `eval_external_speaker_verifier_cvrare_sad_enunc_guard.csv` | 110 | [`scripts/eval_external_speaker_verifier.py`](../scripts/eval_external_speaker_verifier.py) | Finding 36 external ECAPA novelty validation |
| `eval_external_speaker_verifier_cvrare_sad_enunc_guard.md` | summary | [`scripts/eval_external_speaker_verifier.py`](../scripts/eval_external_speaker_verifier.py) | Finding 36 summary and caveats |

CommonVoice metadata separability artifacts from 2026-05-28:

These probe whether metadata labels are recoverable in raw OpenVoice embeddings
and metadata-control VAE latents. They are diagnostic evidence only; they do
not prove perceptual age/gender control.

| File | Rows | Script | Backs |
|------|------|--------|-------|
| `commonvoice_metadata_separability_cvrare_expanded.csv` | 6 | [`scripts/probe_commonvoice_metadata_separability.py`](../scripts/probe_commonvoice_metadata_separability.py) | Finding 37 expanded CommonVoice metadata separability |
| `commonvoice_metadata_separability_cvrare_expanded.md` | summary | [`scripts/probe_commonvoice_metadata_separability.py`](../scripts/probe_commonvoice_metadata_separability.py) | Finding 37 expanded CommonVoice summary |
| `commonvoice_metadata_separability_mixed_metadata_base.csv` | 6 | [`scripts/probe_commonvoice_metadata_separability.py`](../scripts/probe_commonvoice_metadata_separability.py) | Finding 37 mixed metadata-training artifact separability |
| `commonvoice_metadata_separability_mixed_metadata_base.md` | summary | [`scripts/probe_commonvoice_metadata_separability.py`](../scripts/probe_commonvoice_metadata_separability.py) | Finding 37 mixed metadata-training artifact summary |

Validation-scale CommonVoice pretraining comparison artifacts from 2026-04-28:

| File | Rows | Script | Backs |
|------|------|--------|-------|
| `eval_emotion_pass2_combined.csv` | 110 | [`examples/eval_emotion.py`](../examples/eval_emotion.py) | CommonVoice pretraining pipeline combined-only baseline for Finding 10 |
| `eval_wer_pass2_combined.csv`     | 110 | [`examples/eval_wer.py`](../examples/eval_wer.py)         | CommonVoice pretraining pipeline combined-only baseline for Finding 10 |
| `eval_emotion_pass2_cv500.csv`    | 110 | [`examples/eval_emotion.py`](../examples/eval_emotion.py) | Finding 10 (`cv500` CommonVoice init candidate) |
| `eval_wer_pass2_cv500.csv`        | 110 | [`examples/eval_wer.py`](../examples/eval_wer.py)         | Finding 10 (`cv500` CommonVoice init candidate) |
| `eval_novelty_pass2_combined.csv` | 110 | [`examples/eval_novelty.py`](../examples/eval_novelty.py) | speaker novelty metric work novelty baseline for Finding 11 |
| `eval_novelty_pass2_cv500.csv`    | 110 | [`examples/eval_novelty.py`](../examples/eval_novelty.py) | Finding 11 (`cv500` novelty candidate) |

Evaluation ablation matrix artifacts from 2026-04-28:

| Bundle | Rows | Scripts | Backs |
|--------|------|---------|-------|
| `pass4_combined` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 12 combined reference condition |
| `pass4_commonvoice_cv500_init` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 12 CommonVoice-init condition |
| `pass4_cremad_only` | 77 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 12 CREMA-D-only condition |
| `pass4_expresso_only` | 77 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 12 Expresso-only condition |
| `pass4_naive_noise_baseline` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 12 naive unlabeled-latent baseline |
| `eval_ablation_summary_pass4.csv` | 5 conditions | [`scripts/summarize_ablation_results.py`](../scripts/summarize_ablation_results.py) | Finding 12 top-line matrix |
| `eval_ablation_collapse_pass4.csv` | per generated file | [`scripts/summarize_ablation_results.py`](../scripts/summarize_ablation_results.py) | Finding 12 collapse taxonomy |

CommonVoice finetune ablation artifacts from 2026-04-28:

| Bundle | Rows | Scripts | Backs |
|--------|------|---------|-------|
| `pass5_combined` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 13 combined reference condition |
| `pass5_commonvoice_cv500_init` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 13 raw `cv500` reference condition |
| `pass5_cv500_ft_short` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 13 short-finetune condition |
| `pass5_cv500_ft_low_lr` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 13 low-LR condition |
| `pass5_cv500_ft_short_low_lr` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 13 best partial-recovery condition |
| `pass5_cv500_ft_freeze_decoder` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 13 decoder-freeze negative result |
| `pass5_cv500_ft_freeze_encoder` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 13 encoder-freeze condition |
| `eval_commonvoice_finetune_summary_pass5.csv` | 7 conditions | [`scripts/summarize_commonvoice_finetune_ablation.py`](../scripts/summarize_commonvoice_finetune_ablation.py) | Finding 13 top-line matrix |
| `eval_commonvoice_finetune_collapse_pass5.csv` | per generated file | [`scripts/summarize_commonvoice_finetune_ablation.py`](../scripts/summarize_commonvoice_finetune_ablation.py) | Finding 13 collapse taxonomy |

CommonVoice objective ablation artifacts from 2026-04-28:

| Bundle | Rows | Scripts | Backs |
|--------|------|---------|-------|
| `pass6_combined` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 14 combined reference condition |
| `pass6_commonvoice_cv500_init` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 14 raw `cv500` reference condition |
| `pass6_cv500_ft_short_low_lr` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 14 best CommonVoice finetune reference |
| `pass6_cv500_obj_label2` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 14 label-upweight condition |
| `pass6_cv500_obj_label4` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 14 high-label-weight condition |
| `pass6_cv500_obj_label_ramp` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 14 label-ramp condition |
| `pass6_cv500_obj_recon_half_label2` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 14 reduced-reconstruction condition |
| `eval_commonvoice_objective_summary_pass6.csv` | 7 conditions | [`scripts/summarize_commonvoice_objective_ablation.py`](../scripts/summarize_commonvoice_objective_ablation.py) | Finding 14 top-line matrix |
| `eval_commonvoice_objective_collapse_pass6.csv` | per generated file | [`scripts/summarize_commonvoice_objective_ablation.py`](../scripts/summarize_commonvoice_objective_ablation.py) | Finding 14 collapse taxonomy |

CommonVoice rich-objective ablation artifacts from 2026-04-29:

| Bundle | Rows | Scripts | Backs |
|--------|------|---------|-------|
| `pass7_combined` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 15 combined reference condition |
| `pass7_commonvoice_cv500_init` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 15 raw `cv500` reference condition |
| `pass7_cv500_ft_short_low_lr` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 15 best CommonVoice finetune reference |
| `pass7_cv500_rich_teacher_style` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 15 style-teacher distillation condition |
| `pass7_cv500_rich_free_anchor` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 15 free-dim anchor condition |
| `pass7_cv500_rich_teacher_plus_anchor` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 15 combined teacher+anchor condition |
| `eval_commonvoice_rich_objectives_summary_pass7.csv` | 6 conditions | [`scripts/summarize_commonvoice_rich_objectives.py`](../scripts/summarize_commonvoice_rich_objectives.py) | Finding 15 top-line matrix |
| `eval_commonvoice_rich_objectives_collapse_pass7.csv` | per generated file | [`scripts/summarize_commonvoice_rich_objectives.py`](../scripts/summarize_commonvoice_rich_objectives.py) | Finding 15 collapse taxonomy |

CommonVoice partial-label pretraining artifacts from 2026-04-29:

| Bundle | Rows | Scripts | Backs |
|--------|------|---------|-------|
| `pass8_combined` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 16 combined reference condition |
| `pass8_commonvoice_cv500_init` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 16 raw `cv500` reference condition |
| `pass8_cv500_ft_short_low_lr` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 16 best CommonVoice finetune reference |
| `pass8_cv500_rich_free_anchor` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 16 best CommonVoice rich-objective reference |
| `pass8_cv500_pl_meta` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 16 metadata-only weak-label condition |
| `pass8_cv500_pl_pseudo_style` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 16 pseudo-style weak-label condition |
| `pass8_cv500_pl_meta_plus_pseudo` | 110 | `eval_emotion.py`, `eval_novelty.py`, `eval_wer.py`, `eval_mos.py` | Finding 16 hybrid weak-label condition |
| `eval_commonvoice_partial_label_summary_pass8.csv` | 7 conditions | [`scripts/summarize_commonvoice_partial_label.py`](../scripts/summarize_commonvoice_partial_label.py) | Finding 16 top-line matrix |
| `eval_commonvoice_partial_label_collapse_pass8.csv` | per generated file | [`scripts/summarize_commonvoice_partial_label.py`](../scripts/summarize_commonvoice_partial_label.py) | Finding 16 collapse taxonomy |

Mixed-data pseudolabel mix schedule matrix from 2026-04-30:

- The full mixed-data result bundle is now checked in for the three new schedule conditions:
  - `pass9_mixed_static_balanced`
  - `pass9_mixed_cv_warmup`
  - `pass9_mixed_labeled_finish`
- The mixed-data summary also reuses copied reference CSVs for:
  - `combined`
  - `commonvoice_cv500_init`
  - `cv500_ft_short_low_lr`
  - `cv500_rich_free_anchor`
  - `cv500_pl_meta`
- Top-line matrix:
  - `mixed_static_balanced`: recall `16.7%`, novelty `0.0865`, mean WER `0.0825`, MOS delta `-0.1316`
  - `mixed_cv_warmup`: recall `16.7%`, novelty `0.0738`, mean WER `0.0700`, MOS delta `-0.1341`
  - `mixed_labeled_finish`: recall `16.7%`, novelty `0.0828`, mean WER `0.0606`, MOS delta `-0.1425`
- Current conclusion:
  - schedule choice changes the WER/novelty tradeoff modestly
  - none of the three schedules improves recall beyond `16.7%`
  - mixed-data training helps the CommonVoice line more on intelligibility than on controllability

Mixed-data pseudolabel quality follow-up from 2026-05-03:

- The full quality-follow-up result bundle is now checked in for:
  - `mixed_quality_static_balanced`
  - `mixed_quality_labeled_finish`
  - `mixed_quality_labeled_guarded`
- The quality summary also reuses copied reference CSVs for:
  - `combined`
  - `commonvoice_cv500_init`
  - `cv500_ft_short_low_lr`
  - `cv500_rich_free_anchor`
  - `cv500_pl_meta`
  - `mixed_static_balanced`
  - `mixed_cv_warmup`
  - `mixed_labeled_finish`
- Top-line matrix:
  - `mixed_quality_static_balanced`: recall `16.7%`, novelty `0.0770`, mean WER `0.0911`, MOS delta `-0.1130`
  - `mixed_quality_labeled_finish`: recall `16.7%`, novelty `0.0763`, mean WER `0.0649`, MOS delta `-0.1232`
  - `mixed_quality_labeled_guarded`: recall `18.2%`, novelty `0.0764`, mean WER `0.0978`, MOS delta `-0.1234`
- Current conclusion:
  - stricter pseudo-label filtering plus stronger labeled-data protection can move recall a little
  - `mixed_quality_labeled_guarded` is the first mixed-data condition to improve recall above `16.7%`
  - the gain is not a clean win, because it gives back WER versus `mixed_labeled_finish` and novelty versus `mixed_static_balanced`
  - `mixed_quality_labeled_guarded` was the checkpoint carried into the first non-Trump style-strength sweep before the teacher-family follow-up

Mixed-data pseudolabel teacher follow-up from 2026-05-04/2026-05-05:

- The full teacher-family result bundle is now checked in for:
  - `mixed_teacher_threshold_balanced`
  - `mixed_teacher_labeled_finish`
  - `mixed_teacher_labeled_guarded`
  - `mixed_teacher_mapped015_balanced`
  - `mixed_teacher_prototype_balanced`
  - `mixed_teacher_prototype_guarded`
  - `mixed_teacher_hybrid_extra_balanced`
  - `mixed_teacher_hybrid_style_distill_balanced`
  - `mixed_teacher_hybrid_style_distill_targetmask_balanced`
  - `mixed_teacher_hybrid_style_distill_labeled_warmup`
  - `mixed_teacher_hybrid_style_distill_w010_balanced`
  - `mixed_teacher_hybrid_style_distill_w050_balanced`
- The teacher summary reuses copied reference CSVs for:
  - `combined`
  - `commonvoice_cv500_init`
  - `cv500_ft_short_low_lr`
  - `cv500_rich_free_anchor`
  - `cv500_pl_meta`
  - `mixed_static_balanced`
  - `mixed_cv_warmup`
  - `mixed_labeled_finish`
  - `mixed_quality_static_balanced`
  - `mixed_quality_labeled_finish`
  - `mixed_quality_labeled_guarded`
- Top-line matrix:
  - `mixed_teacher_threshold_balanced`: recall `18.2%`, novelty `0.0785`, mean WER `0.0829`, MOS delta `-0.1012`
  - `mixed_teacher_labeled_finish`: recall `16.7%`, novelty `0.0763`, mean WER `0.0727`, MOS delta `-0.1150`
  - `mixed_teacher_labeled_guarded`: recall `18.2%`, novelty `0.0760`, mean WER `0.1095`, MOS delta `-0.1173`
  - `mixed_teacher_mapped015_balanced`: recall `16.7%`, novelty `0.0818`, mean WER `0.1090`, MOS delta `-0.1115`
  - `mixed_teacher_prototype_balanced`: recall `18.2%`, novelty `0.0854`, mean WER `0.1009`, MOS delta `-0.1086`
  - `mixed_teacher_prototype_guarded`: recall `18.2%`, novelty `0.0761`, mean WER `0.0920`, MOS delta `-0.1081`
  - `mixed_teacher_hybrid_extra_balanced`: recall `16.7%`, novelty `0.0860`, mean WER `0.0931`, MOS delta `-0.1190`
  - `mixed_teacher_hybrid_style_distill_balanced`: recall `16.7%`, novelty `0.0861`, mean WER `0.0938`, MOS delta `-0.1072`
  - `mixed_teacher_hybrid_style_distill_targetmask_balanced`: recall `16.7%`, novelty `0.0852`, mean WER `0.1062`, MOS delta `-0.1181`
  - `mixed_teacher_hybrid_style_distill_labeled_warmup`: recall `16.7%`, novelty `0.0930`, mean WER `0.0924`, MOS delta `-0.1093`
  - `mixed_teacher_hybrid_style_distill_w010_balanced`: recall `16.7%`, novelty `0.0854`, mean WER `0.0924`, MOS delta `-0.1161`
  - `mixed_teacher_hybrid_style_distill_w050_balanced`: recall `16.7%`, novelty `0.0840`, mean WER `0.0821`, MOS delta `-0.1196`
- Current conclusion:
  - the first teacher-family run does not improve mixed-data recall beyond `18.2%`
  - `mixed_teacher_threshold_balanced` is still a useful improvement because it matches `mixed_quality_labeled_guarded` on recall while improving novelty, WER, MOS, and identity collapse
  - the guarded teacher schedule is not the answer: it preserves the recall bump but gives back too much WER and identity stability
  - `mixed_teacher_mapped015_balanced` shows that a softer same-teacher agreement rule only buys a small novelty gain while losing recall and WER/MOS
  - `mixed_teacher_prototype_balanced` shows that a genuinely different latent-prototype teacher improves coverage, novelty, and collapse counts, but still gives back WER/MOS versus `mixed_teacher_threshold_balanced`
  - `mixed_teacher_prototype_guarded` shows that strong guardrails improve WER versus the unguarded prototype but erase the prototype novelty/collapse advantage
  - `mixed_teacher_hybrid_extra_balanced` shows that prototype+emotion2vec hard-label mixing produces the best mixed-teacher novelty so far, but loses recall and MOS
  - `mixed_teacher_hybrid_style_distill_balanced` shows that continuous style-space distillation preserves the hybrid novelty gain and improves MOS/collapse versus hard hybrid labels, but recall remains fixed at `16.7%`
  - the global style-teacher weight sweep shows that weights `0.10`, `0.25`, and `0.50` all remain fixed at `16.7%` recall; `0.50` improves WER and identity/mixed collapse, while `0.25` remains the better novelty/MOS tradeoff
  - `mixed_teacher_hybrid_style_distill_targetmask_balanced` shows that target-dimension masks, per-style row weights, and confidence scaling also remain fixed at `16.7%` recall while slightly worsening WER/MOS versus global `0.25` style distillation
  - `results/eval_mixed_teacher_style_diagnostics_targetmask.md` localizes the target-mask failure: canonical emotion pseudo-labels often lack teacher target-dim dominance, `anger` and `fear` have only `4` active rows each, and `sad` can align latently while still decoding to neutral-classified audio
  - `mixed_teacher_hybrid_style_distill_labeled_warmup` shows that a labeled-first curriculum improves novelty and collapse counts but remains fixed at `16.7%` recall
  - `results/eval_mixed_teacher_style_diagnostics_labeled_warmup.md` confirms the curriculum still leaves canonical emotions in the neutral recall basin while `confused` and `whisper` carry most of the novelty signal
  - `results/listening_mixed_teacher_hybrid_style_distill_labeled_warmup.html` provides a browser-playable listening report for the newest corpus, and `results/listening_mixed_teacher_hybrid_style_distill_labeled_warmup_ratings.csv` provides the subjective-rating template
  - `results/commonvoice_pseudolabel_supply_audit.md` confirms the current local CommonVoice subset is rare-class limited: the hybrid artifact has only `anger=5` and `fear=4` selected rows before mixed-data sampling
  - `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup` shows that expanded rare-class supply changes the result materially: recall jumps to `47.0%` and novelty to `0.2995`, but mean styled WER rises to `0.2751` and MOS delta falls to `-0.2640`
  - `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` is the best current quality-balanced profile for that checkpoint: it keeps `47.0%` recall while improving mean styled WER to `0.2348`, MOS delta to `-0.2081`, and files with any collapse to `20`
  - `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_content_guard` is useful diagnostically because it repairs WER/MOS further (`0.2133` WER, `-0.1989` MOS delta) but drops recall to `40.9%`
  - `results/eval_mixed_teacher_style_diagnostics_cvrare_labeled_warmup.md` confirms the new bottleneck is calibration/output alignment, not just raw rare-class row scarcity
  - `results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.html` provides the browser-playable listening report for perceptual review, and `results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_ratings.csv` provides the subjective-rating template
  - `results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.html` is the recommended quality-balanced listening report for the expanded checkpoint
  - the next mixed-data branch should focus on decoder-aware/generated-audio style objectives, not more schedule variants, hard pseudo-label arbitration, scalar teacher-weight sweeps, schedule-only curricula, or another latent-only mask/weight variant

CommonVoice rare-class supply preflight from 2026-05-05:

| File | Rows / decision | Script | Backs |
|------|-----------------|--------|-------|
| `commonvoice_rare_supply_expansion_preflight.json` | `GO` | [`scripts/plan_commonvoice_rare_supply_expansion.py`](../scripts/plan_commonvoice_rare_supply_expansion.py) | WORKLOG section 0.30 |
| `commonvoice_rare_supply_expansion_preflight.md` | `GO` | [`scripts/plan_commonvoice_rare_supply_expansion.py`](../scripts/plan_commonvoice_rare_supply_expansion.py) | WORKLOG section 0.30 |

- Literal `/data/cv-corpus-21.0-2025-03-14/en` was not creatable in this macOS session because `/` is read-only.
- The expanded local corpus at `/Users/steve/datasets/cv-corpus-21.0-2025-03-14/en` has `40000` usable rows and `20537` usable speakers.
- Based on checked-in selected `anger` / `fear` rates, the next rare-supply extraction should target at least `22538` usable rows before another model run.
- This is an engineering/reproducibility gate, not a new paper-facing finding by itself.

Expanded rare-class extraction / teacher-scoring status from 2026-05-05:

- `embeddings/openvoice_commonvoice_cvrare_expanded_emb.pt` validates with
  `25910` OpenVoice embeddings, `13308` unique speakers, and zero
  missing/unreadable clips.
- The expanded emotion2vec scorer is intentionally resumable and target-seeking:

```bash
python scripts/annotate_commonvoice_pseudolabels.py \
    --embeddings embeddings/openvoice_commonvoice_cvrare_expanded_emb.pt \
    --output embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_scored.pt \
    --save-style-score-map \
    --report-threshold 0.60 \
    --batch-size 4 \
    --checkpoint-every 500 \
    --stop-when-accepted-targets anger=50,fear=50
```

- Do not interpret this as a paper-facing finding until the resulting scored,
  filtered, and audited artifacts show whether selected `anger` and `fear`
  supply actually reaches target.

Expanded rare-class supply audit from 2026-05-05:

| File | Rows / decision | Script | Backs |
|------|-----------------|--------|-------|
| `commonvoice_pseudolabel_supply_audit_rare_supply.csv` | expanded supply cleared | [`scripts/audit_commonvoice_pseudolabel_supply.py`](../scripts/audit_commonvoice_pseudolabel_supply.py) | WORKLOG section 0.32 |
| `commonvoice_pseudolabel_supply_audit_rare_supply.md` | expanded supply cleared | [`scripts/audit_commonvoice_pseudolabel_supply.py`](../scripts/audit_commonvoice_pseudolabel_supply.py) | WORKLOG section 0.32 |

- The target-seeking emotion2vec scorer annotated `6380/25910` expanded
  CommonVoice rows and reached the rare-class stop condition.
- The emotion2vec filtered artifact selected `anger=50` and `fear=50`; the
  prototype side added `confused=50`, `enunciated=50`, and `whisper=50`.
- The hybrid selected artifact contains `645` selected pseudo rows across the
  nine unified styles.
- `scripts/build_mixed_training_set.py --commonvoice-preserve-selected-pseudo`
  keeps the one-clip-per-speaker speaker-breadth baseline while adding the `62`
  selected pseudo rows that the speaker cap would otherwise drop.
- The final mixed artifact
  `embeddings/openvoice_mixed_teacher_cvrare_hybrid_extra_base.pt` has `14195`
  rows, `13308` CommonVoice speakers, `645` CommonVoice pseudo-labeled rows,
  and mixed CommonVoice counts `anger=50` / `fear=50`.
- This cleared the data-readiness gate and now has a generated-audio result.

Expanded rare-supply generated-audio evaluation from 2026-05-05:

| File | Rows / result | Script | Backs |
|------|---------------|--------|-------|
| `eval_emotion_mixed_teacher_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.csv` | `47.0%` recall | [`examples/eval_emotion.py`](../examples/eval_emotion.py) | FINDINGS Finding 30 |
| `eval_novelty_mixed_teacher_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.csv` | `0.2995` novelty gain | [`examples/eval_novelty.py`](../examples/eval_novelty.py) | FINDINGS Finding 30 |
| `eval_wer_mixed_teacher_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.csv` | `0.2751` mean styled WER | [`examples/eval_wer.py`](../examples/eval_wer.py) | FINDINGS Finding 30 |
| `eval_mos_mixed_teacher_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.csv` | `-0.2640` MOS delta | [`examples/eval_mos.py`](../examples/eval_mos.py) | FINDINGS Finding 30 |
| `eval_mixed_teacher_style_diagnostics_cvrare_labeled_warmup.md` | calibrated limitation | [`scripts/analyze_mixed_teacher_style_diagnostics.py`](../scripts/analyze_mixed_teacher_style_diagnostics.py) | FINDINGS Finding 30 |
| `listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.html` | browser listening review | [`scripts/build_listening_report.py`](../scripts/build_listening_report.py) | FINDINGS Finding 30 |
| `eval_emotion_mixed_teacher_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_content_guard.csv` | `40.9%` recall | [`examples/eval_emotion.py`](../examples/eval_emotion.py) | FINDINGS Finding 31 |
| `eval_novelty_mixed_teacher_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_content_guard.csv` | `0.2653` novelty gain | [`examples/eval_novelty.py`](../examples/eval_novelty.py) | FINDINGS Finding 31 |
| `eval_wer_mixed_teacher_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_content_guard.csv` | `0.2133` mean styled WER | [`examples/eval_wer.py`](../examples/eval_wer.py) | FINDINGS Finding 31 |
| `eval_mos_mixed_teacher_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_content_guard.csv` | `-0.1989` MOS delta | [`examples/eval_mos.py`](../examples/eval_mos.py) | FINDINGS Finding 31 |
| `listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_content_guard.html` | browser listening review | [`scripts/build_listening_report.py`](../scripts/build_listening_report.py) | FINDINGS Finding 31 |
| `eval_emotion_mixed_teacher_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.csv` | `47.0%` recall | [`examples/eval_emotion.py`](../examples/eval_emotion.py) | FINDINGS Finding 31 |
| `eval_novelty_mixed_teacher_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.csv` | `0.2726` novelty gain | [`examples/eval_novelty.py`](../examples/eval_novelty.py) | FINDINGS Finding 31 |
| `eval_wer_mixed_teacher_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.csv` | `0.2348` mean styled WER | [`examples/eval_wer.py`](../examples/eval_wer.py) | FINDINGS Finding 31 |
| `eval_mos_mixed_teacher_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.csv` | `-0.2081` MOS delta | [`examples/eval_mos.py`](../examples/eval_mos.py) | FINDINGS Finding 31 |
| `listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.html` | browser listening review | [`scripts/build_listening_report.py`](../scripts/build_listening_report.py) | FINDINGS Finding 31 |
| `eval_mixed_teacher_cvrare_strength_profiles_summary.md` | profile comparison summary | manual summary from checked-in CSVs | FINDINGS Finding 31 |
| `eval_emotion_mixed_teacher_mixed_teacher_cvrare_decoder_proto_labeled_warmup.csv` | `42.4%` recall | [`examples/eval_emotion.py`](../examples/eval_emotion.py) | FINDINGS Finding 32 |
| `eval_novelty_mixed_teacher_mixed_teacher_cvrare_decoder_proto_labeled_warmup.csv` | `0.3008` novelty gain | [`examples/eval_novelty.py`](../examples/eval_novelty.py) | FINDINGS Finding 32 |
| `eval_wer_mixed_teacher_mixed_teacher_cvrare_decoder_proto_labeled_warmup.csv` | `0.2863` mean styled WER | [`examples/eval_wer.py`](../examples/eval_wer.py) | FINDINGS Finding 32 |
| `eval_mos_mixed_teacher_mixed_teacher_cvrare_decoder_proto_labeled_warmup.csv` | `-0.2148` MOS delta | [`examples/eval_mos.py`](../examples/eval_mos.py) | FINDINGS Finding 32 |
| `listening_mixed_teacher_cvrare_decoder_proto_labeled_warmup.html` | browser listening review | [`scripts/build_listening_report.py`](../scripts/build_listening_report.py) | FINDINGS Finding 32 |
| `eval_emotion_mixed_teacher_mixed_teacher_cvrare_decoder_proto_labeled_warmup_sad_enunc_guard.csv` | `42.4%` recall | [`examples/eval_emotion.py`](../examples/eval_emotion.py) | FINDINGS Finding 32 |
| `eval_novelty_mixed_teacher_mixed_teacher_cvrare_decoder_proto_labeled_warmup_sad_enunc_guard.csv` | `0.2718` novelty gain | [`examples/eval_novelty.py`](../examples/eval_novelty.py) | FINDINGS Finding 32 |
| `eval_wer_mixed_teacher_mixed_teacher_cvrare_decoder_proto_labeled_warmup_sad_enunc_guard.csv` | `0.2592` mean styled WER | [`examples/eval_wer.py`](../examples/eval_wer.py) | FINDINGS Finding 32 |
| `eval_mos_mixed_teacher_mixed_teacher_cvrare_decoder_proto_labeled_warmup_sad_enunc_guard.csv` | `-0.1787` MOS delta | [`examples/eval_mos.py`](../examples/eval_mos.py) | FINDINGS Finding 32 |
| `listening_mixed_teacher_cvrare_decoder_proto_labeled_warmup_sad_enunc_guard.html` | browser listening review | [`scripts/build_listening_report.py`](../scripts/build_listening_report.py) | FINDINGS Finding 32 |
| `eval_emotion_mixed_teacher_mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup.csv` | `42.4%` recall | [`examples/eval_emotion.py`](../examples/eval_emotion.py) | FINDINGS Finding 32 |
| `eval_novelty_mixed_teacher_mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup.csv` | `0.3032` novelty gain | [`examples/eval_novelty.py`](../examples/eval_novelty.py) | FINDINGS Finding 32 |
| `eval_wer_mixed_teacher_mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup.csv` | `0.2782` mean styled WER | [`examples/eval_wer.py`](../examples/eval_wer.py) | FINDINGS Finding 32 |
| `eval_mos_mixed_teacher_mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup.csv` | `-0.2122` MOS delta | [`examples/eval_mos.py`](../examples/eval_mos.py) | FINDINGS Finding 32 |
| `listening_mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup.html` | browser listening review | [`scripts/build_listening_report.py`](../scripts/build_listening_report.py) | FINDINGS Finding 32 |
| `eval_mixed_teacher_cvrare_decoder_proto_summary.md` | decoder-prototype pilot comparison | manual summary from checked-in CSVs | FINDINGS Finding 32 |
| `eval_emotion_mixed_teacher_mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup.csv` | `39.4%` recall | [`examples/eval_emotion.py`](../examples/eval_emotion.py) | FINDINGS Finding 33 |
| `eval_novelty_mixed_teacher_mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup.csv` | `0.2960` novelty gain | [`examples/eval_novelty.py`](../examples/eval_novelty.py) | FINDINGS Finding 33 |
| `eval_wer_mixed_teacher_mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup.csv` | `0.2651` mean styled WER | [`examples/eval_wer.py`](../examples/eval_wer.py) | FINDINGS Finding 33 |
| `eval_mos_mixed_teacher_mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup.csv` | `-0.2072` MOS delta | [`examples/eval_mos.py`](../examples/eval_mos.py) | FINDINGS Finding 33 |
| `listening_mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup.html` | browser listening review | [`scripts/build_listening_report.py`](../scripts/build_listening_report.py) | FINDINGS Finding 33 |
| `eval_emotion_mixed_teacher_mixed_teacher_cvrare_antineutral_labeled_warmup.csv` | `40.9%` recall | [`examples/eval_emotion.py`](../examples/eval_emotion.py) | FINDINGS Finding 34 |
| `eval_novelty_mixed_teacher_mixed_teacher_cvrare_antineutral_labeled_warmup.csv` | `0.2962` novelty gain | [`examples/eval_novelty.py`](../examples/eval_novelty.py) | FINDINGS Finding 34 |
| `eval_wer_mixed_teacher_mixed_teacher_cvrare_antineutral_labeled_warmup.csv` | `0.2609` mean styled WER | [`examples/eval_wer.py`](../examples/eval_wer.py) | FINDINGS Finding 34 |
| `eval_mos_mixed_teacher_mixed_teacher_cvrare_antineutral_labeled_warmup.csv` | `-0.2065` MOS delta | [`examples/eval_mos.py`](../examples/eval_mos.py) | FINDINGS Finding 34 |
| `listening_mixed_teacher_cvrare_antineutral_labeled_warmup.html` | browser listening review | [`scripts/build_listening_report.py`](../scripts/build_listening_report.py) | FINDINGS Finding 34 |
| `eval_mixed_teacher_generated_audio_failure_mining.csv` | `495` styled rows joined across five conditions, now including the failure-targeted and anti-neutral follow-ups | [`scripts/analyze_generated_audio_failures.py`](../scripts/analyze_generated_audio_failures.py) | FINDINGS Findings 32-34 |
| `eval_mixed_teacher_generated_audio_failure_mining.md` | row-level failure readout | [`scripts/analyze_generated_audio_failures.py`](../scripts/analyze_generated_audio_failures.py) | FINDINGS Findings 32-34 |
| `eval_mixed_teacher_failure_conditioned_targets.csv` | target decision rows for `anger`/`disgust`/`fear` | [`scripts/select_failure_conditioned_targets.py`](../scripts/select_failure_conditioned_targets.py) | FINDINGS Finding 32 |
| `eval_mixed_teacher_failure_conditioned_targets.json` | trainer-ready target-plan config | [`scripts/select_failure_conditioned_targets.py`](../scripts/select_failure_conditioned_targets.py) | FINDINGS Finding 32 |
| `eval_mixed_teacher_failure_conditioned_targets.md` | target-selection readout | [`scripts/select_failure_conditioned_targets.py`](../scripts/select_failure_conditioned_targets.py) | FINDINGS Finding 32 |
| `eval_mixed_teacher_strength_grid_summary.csv` | `12` generated-audio grid conditions | [`scripts/run_style_strength_grid.py`](../scripts/run_style_strength_grid.py) | FINDINGS Finding 35 |
| `eval_mixed_teacher_strength_grid_collapse.csv` | per-row collapse labels for the grid | [`scripts/run_style_strength_grid.py`](../scripts/run_style_strength_grid.py) | FINDINGS Finding 35 |
| `eval_mixed_teacher_cvrare_strength_grid_ranking.csv` | per-style ranked grid cells | [`scripts/summarize_style_strength_grid.py`](../scripts/summarize_style_strength_grid.py) | FINDINGS Finding 35 |
| `eval_mixed_teacher_cvrare_strength_grid_ranking.md` | grid readout and best-cell table | [`scripts/summarize_style_strength_grid.py`](../scripts/summarize_style_strength_grid.py) | FINDINGS Finding 35 |
| `listening_evidence_demo_index.html` | canonical listening index for current evidence/demo review | hand-authored evidence packet | evidence/demo packet for Findings 30-35 |
| `commonvoice_metadata_controls_audit.md` | local CommonVoice age/gender coverage audit | [`scripts/audit_commonvoice_metadata_controls.py`](../scripts/audit_commonvoice_metadata_controls.py) | metadata-control branch preflight |
| `commonvoice_metadata_controls_audit.json` | machine-readable CommonVoice age/gender coverage audit | [`scripts/audit_commonvoice_metadata_controls.py`](../scripts/audit_commonvoice_metadata_controls.py) | metadata-control branch preflight |
| `listening_mixed_teacher_cvrare_strength_grid_anger_s10.html` | best ranked `anger` grid cell | [`scripts/build_listening_report.py`](../scripts/build_listening_report.py) | FINDINGS Finding 35 |
| `listening_mixed_teacher_cvrare_strength_grid_disgust_s10.html` | best ranked `disgust` grid cell | [`scripts/build_listening_report.py`](../scripts/build_listening_report.py) | FINDINGS Finding 35 |
| `listening_mixed_teacher_cvrare_strength_grid_fear_s7p5.html` | best ranked `fear` grid cell | [`scripts/build_listening_report.py`](../scripts/build_listening_report.py) | FINDINGS Finding 35 |
| `listening_mixed_teacher_cvrare_strength_grid_ab_review.html` | guard-vs-candidate A/B perceptual review dashboard | [`scripts/build_style_grid_review.py`](../scripts/build_style_grid_review.py) | perceptual review queue for Finding 35 |
| `listening_mixed_teacher_cvrare_strength_grid_ab_review_ratings.csv` | 33-row subjective rating template | [`scripts/build_style_grid_review.py`](../scripts/build_style_grid_review.py) | perceptual review queue for Finding 35 |
| `listening_mixed_teacher_cvrare_strength_grid_ab_review_priority.csv` | objective-assisted A/B listening triage | [`scripts/summarize_style_grid_review.py`](../scripts/summarize_style_grid_review.py) | perceptual review queue for Finding 35 |
| `listening_mixed_teacher_cvrare_strength_grid_ab_review_priority.md` | five-row listen-first summary plus metric traps | [`scripts/summarize_style_grid_review.py`](../scripts/summarize_style_grid_review.py) | perceptual review queue for Finding 35 |
| `listening_mixed_teacher_cvrare_strength_grid_ab_review_priority.html` | five-row priority-only A/B perceptual review dashboard | [`scripts/build_style_grid_review.py`](../scripts/build_style_grid_review.py) | perceptual review queue for Finding 35 |
| `listening_mixed_teacher_cvrare_strength_grid_ab_review_priority_ratings.csv` | five-row subjective rating template | [`scripts/build_style_grid_review.py`](../scripts/build_style_grid_review.py) | perceptual review queue for Finding 35 |
| `listening_mixed_teacher_cvrare_strength_grid_ab_review_priority_ratings_joe_2026-05-19.csv` | Joe's encoded five-row subjective ratings from Teams | manual transcription from Joe review | FINDINGS Finding 35 |
| `listening_mixed_teacher_cvrare_strength_grid_ab_review_priority_joe_2026-05-19.csv` | objective-assisted A/B triage with Joe ratings attached | [`scripts/summarize_style_grid_review.py`](../scripts/summarize_style_grid_review.py) | FINDINGS Finding 35 |
| `listening_mixed_teacher_cvrare_strength_grid_ab_review_priority_joe_2026-05-19.md` | Joe review summary: `4` ties, `1` reference preference, `0` candidate wins | [`scripts/summarize_style_grid_review.py`](../scripts/summarize_style_grid_review.py) | FINDINGS Finding 35 |
| `generated_audio_content_repair_gate.csv` | `33` hard-style strength rows with objective and perceptual gate decisions | [`scripts/select_generated_audio_content_repairs.py`](../scripts/select_generated_audio_content_repairs.py) | FINDINGS Finding 38 |
| `generated_audio_content_repair_gate.md` | gate summary: `0` promoted candidates, `2` perceptually blocked objective-pass rows | [`scripts/select_generated_audio_content_repairs.py`](../scripts/select_generated_audio_content_repairs.py) | FINDINGS Finding 38 |
| `generated_audio_content_repair_gate.json` | machine-readable promoted-profile payload; currently empty because no preset passes | [`scripts/select_generated_audio_content_repairs.py`](../scripts/select_generated_audio_content_repairs.py) | FINDINGS Finding 38 |
| `generated_audio_calibrated_objective_plan.csv` | style-level training plan selecting `anger`/`disgust` and blocking `fear` | [`scripts/plan_generated_audio_calibrated_objective.py`](../scripts/plan_generated_audio_calibrated_objective.py) | Trainer-ready follow-up to Finding 38 |
| `generated_audio_calibrated_objective_plan.md` | recommended command and interpretation for the next hard-style repair checkpoint | [`scripts/plan_generated_audio_calibrated_objective.py`](../scripts/plan_generated_audio_calibrated_objective.py) | Trainer-ready follow-up to Finding 38 |
| `generated_audio_calibrated_objective_plan.json` | machine-readable generated-audio objective-plan payload for `openvoice_train_vae_mixed.py --generated-audio-objective-plan` | [`scripts/plan_generated_audio_calibrated_objective.py`](../scripts/plan_generated_audio_calibrated_objective.py) | Trainer-ready follow-up to Finding 38 |
| `generated_audio_calibrated_objective_training_report.json` | trainer-applied objective plan for `mixed_teacher_cvrare_audio_calibrated_labeled_warmup` | [`examples/openvoice_train_vae_mixed.py`](../examples/openvoice_train_vae_mixed.py) | FINDINGS Finding 39 |
| `eval_emotion_mixed_teacher_mixed_teacher_cvrare_audio_calibrated_labeled_warmup.csv` | `110` generated rows | [`examples/eval_emotion.py`](../examples/eval_emotion.py) | FINDINGS Finding 39 |
| `eval_novelty_mixed_teacher_mixed_teacher_cvrare_audio_calibrated_labeled_warmup.csv` | `110` generated rows | [`examples/eval_novelty.py`](../examples/eval_novelty.py) | FINDINGS Finding 39 |
| `eval_wer_mixed_teacher_mixed_teacher_cvrare_audio_calibrated_labeled_warmup.csv` | `110` generated rows | [`examples/eval_wer.py`](../examples/eval_wer.py) | FINDINGS Finding 39 |
| `eval_mos_mixed_teacher_mixed_teacher_cvrare_audio_calibrated_labeled_warmup.csv` | `110` generated rows | [`examples/eval_mos.py`](../examples/eval_mos.py) | FINDINGS Finding 39 |
| `eval_external_speaker_verifier_mixed_teacher_cvrare_audio_calibrated_labeled_warmup.csv` | `110` generated rows plus proxy-trial thresholding | [`scripts/eval_external_speaker_verifier.py`](../scripts/eval_external_speaker_verifier.py) | FINDINGS Finding 39 |
| `eval_external_speaker_verifier_mixed_teacher_cvrare_audio_calibrated_labeled_warmup.md` | external ECAPA summary | [`scripts/eval_external_speaker_verifier.py`](../scripts/eval_external_speaker_verifier.py) | FINDINGS Finding 39 |
| `listening_mixed_teacher_cvrare_audio_calibrated_labeled_warmup.html` | browser-playable listening panel for the trained audio-calibrated checkpoint | [`scripts/build_listening_report.py`](../scripts/build_listening_report.py) | perceptual follow-up for Finding 39 |
| `listening_mixed_teacher_cvrare_audio_calibrated_labeled_warmup_ratings.csv` | subjective rating template | [`scripts/build_listening_report.py`](../scripts/build_listening_report.py) | perceptual follow-up for Finding 39 |
| `listening_mixed_teacher_cvrare_audio_calibrated_labeled_warmup_stephen_2026-05-28.md` | Stephen's informal first-listener review of `anger` and `disgust` | manual perceptual review | perceptual follow-up for Finding 39 |
| `listening_mixed_teacher_cvrare_audio_calibrated_labeled_warmup_joe_2026-05-28.md` | Joe's focused review: `disgust` sounds neutral; early `anger` rows are slightly angry | manual perceptual review | FINDINGS Finding 39 |
| `training_style_separability_rows.csv` | `828` source training clips scored by emotion2vec | [`scripts/audit_training_style_separability.py`](../scripts/audit_training_style_separability.py) | FINDINGS Finding 40 |
| `training_style_separability_by_label.csv` | source-label direct recall / embedding F1 control-selection table | [`scripts/audit_training_style_separability.py`](../scripts/audit_training_style_separability.py) | FINDINGS Finding 40 |
| `training_style_separability_confusion.csv` | direct emotion2vec and held-out embedding-classifier confusion counts | [`scripts/audit_training_style_separability.py`](../scripts/audit_training_style_separability.py) | FINDINGS Finding 40 |
| `training_style_separability_summary.md` | Markdown source-label separability interpretation | [`scripts/audit_training_style_separability.py`](../scripts/audit_training_style_separability.py) | FINDINGS Finding 40 |
| `listening_control_shortlist_neutral_sad_joe_2026-06-08.md` | Joe's focused review: neutral sounds neutral; sad is perceptible but subtle | manual perceptual review | FINDINGS Finding 42 |
| `control_selection_perceptual_evidence.csv` | structured Joe/Stephen listening evidence used by the shortlist gate | manual evidence ledger | FINDINGS Findings 41-42 |
| `control_selection_recommendation.csv` | paper-facing per-style shortlist table | [`scripts/build_control_selection_recommendation.py`](../scripts/build_control_selection_recommendation.py) | FINDINGS Findings 41-42 |
| `control_selection_recommendation.md` | Markdown control-selection recommendation and next listening queue | [`scripts/build_control_selection_recommendation.py`](../scripts/build_control_selection_recommendation.py) | FINDINGS Findings 41-42 |
| `commonvoice_gender_followup_preflight.md` | local CommonVoice gender-readiness summary | [`scripts/preflight_commonvoice_gender_followup.py`](../scripts/preflight_commonvoice_gender_followup.py) | Worklog-only preflight for the future gender-only follow-up |
| `commonvoice_gender_followup_preflight.json` | machine-readable gender-readiness summary | [`scripts/preflight_commonvoice_gender_followup.py`](../scripts/preflight_commonvoice_gender_followup.py) | Worklog-only preflight for the future gender-only follow-up |
| `commonvoice_gender_followup_speakers.csv` | deterministic gender-balanced speaker manifest | [`scripts/preflight_commonvoice_gender_followup.py`](../scripts/preflight_commonvoice_gender_followup.py) | Input contract for a future gender-only branch |
| `listening_gender_followup_available_w005_labeled_warmup_stephen_2026-07-09.md` | local review: gender controls are not reliably perceptible and sound more like generic timbre/speaker movement | manual perceptual review | FINDINGS Finding 43 |
| `gender_followup_available_w005_acoustic_diagnostic.md` | acoustic proxy review: female-control median F0 exceeds male-control median F0 in only `1/4` source pairs | [`scripts/analyze_gender_followup_acoustics.py`](../scripts/analyze_gender_followup_acoustics.py) | Diagnostic support for Finding 43 |

- The expanded rare-supply run is now the strongest checked-in controllability
  / novelty result, but not the cleanest quality result.
- Collapse improves sharply versus earlier mixed-teacher rows: content collapse
  `7`, style-to-neutral collapse `18`, identity collapse `1`, mixed collapse
  `1`, files with any collapse `25`.
- The result should be read as a paper-facing positive finding plus a clear
  objective: preserve the recall/novelty gain while reducing WER/MOS damage.
  The first decoder-prototype training pilot implements that direction, and
  its own `sad/enunciated` guarded readout improves WER/MOS, but neither beats
  the expanded rare-supply `sad/enunciated` inference guard. Treat both
  decoder-prototype readouts as cautionary baselines for generated-audio
  failure mining / teacher-calibrated objectives rather than as new
  references. The lower-weight `0.005` decoder-prototype variant keeps recall
  at `42.4%` and WER at `0.2782`, so weight-only prototype calibration is not
  the next best move.
- The generated-audio failure-mining artifact confirms the current
  `sad/enunciated` guard has the lowest row-level failure score (`1.9899`) and
  localizes persistent failures to `disgust`, `fear`, and `anger`.
- The failure-conditioned target selector narrows the next positive target
  objective to `anger` and `disgust`; `fear` is blocked because the current
  reference has `0/11` clean fear targets after content/naturalness exclusions.
- The first failure-conditioned target-dim style-teacher follow-up is now a
  validated negative result: it keeps novelty high (`0.2960`) and reduces
  content collapse to `1`, but recall drops to `39.4%` and style-to-neutral
  collapse rises to `26`. This ruled out stronger target-dim teacher pressure
  as the next main path.
- The first anti-neutral prototype-margin follow-up is also a validated
  negative result: it improves slightly over the target-dim follow-up
  (`40.9%` recall, `0.2962` novelty, `0.2609` WER), but still loses to the
  current guard on recall/WER/collapse and leaves `25` style-to-neutral
  collapses. The next calibration signal should come from actual generated
  audio, such as a reproducible style-strength grid or reranking artifact.
- The generated-audio style-strength grid is now the first audio-calibrated
  reranking artifact for the hard styles. Best cells are `anger_s10`
  (`3/11` recall, `0.2081` WER), `disgust_s10` (`2/11` recall, `0.2831` WER,
  `-0.6039` MOS delta), and `fear_s7p5` (`6/11` recall, `0.5231` WER). This
  supports perceptual review and style-specific preset design, not a global
  strength increase.
- The A/B perceptual review dashboard puts the current guard and those top
  grid cells side by side for the same `33` source/style pairs. Use it before
  converting any grid cell into a checked-in style-strength profile.
- The A/B priority sheet narrows first listening to five rows: two clean target
  gains and three target gains with quality risk. The priority-only HTML
  dashboard and rating template expose only those five rows for the first
  perceptual pass. The triage also confirms `disgust_s10` has no target-gain
  rows in this panel.
- Joe's first five-row listening review found no candidate wins: `4/5` rows
  sounded identical to the current guard and `1/5` preferred the guard because
  the candidate had an unnatural pitch change. Keep `anger_s10` and
  `fear_s7p5` diagnostic for now rather than promoting them as presets.
- The trained audio-calibrated checkpoint is also diagnostic rather than a new
  reference: it verifies the trainer hook and keeps strong external speaker
  novelty (`0.3336` ECAPA gain), but falls below the current guard on recall
  (`40.91%` vs `46.97%`), WER (`0.2465` vs `0.2348`), OpenVoice novelty
  (`0.2351` vs `0.2726`), and collapse count (`37` vs `20` files with any
  collapse). `disgust` remains at `0/11` recall.
- Stephen's first listening pass initially complicated the `disgust`
  interpretation: `disgust` sounded convincingly disgusted and intelligible to
  him despite the emotion2vec miss, while `anger` carried some style change but
  distorted intelligibility.
- Joe's focused review supersedes that as the current working interpretation:
  he heard `disgust` as neutral across the focused panel and only weak,
  source-dependent `anger` signal in early CREMA-D rows. He also found many
  CREMA-D `disgust` training examples sound neutral, so future work should not
  keep forcing `disgust` without stronger perceptual training data.
- The source training-data separability audit is now the first control-selection
  artifact after Joe's May 28 pivot. It shows that the six CREMA-D emotion
  labels are separable before conversion, while `confused` is weak and
  `enunciated` / `whisper` need non-emotion perceptual framing. This should
  drive the paper/demo shortlist before more model repair.
- Joe's focused neutral/sad review resolves the first control-selection gate:
  `neutral` and `sad` are now headline controls under the conservative
  recommendation. The `sad` claim should carry the caveat that the sadness is
  perceptible but subtle and source-dependent.
- `listening_evidence_demo_index.html` is the first page to open for local
  perceptual review. It links to the current guard, the high-novelty checkpoint,
  Joe's priority A/B gate, and embeds a small quick-listen panel.

Non-Trump style-strength sweep from 2026-05-03:

- The full strength-sweep result bundle is now checked in for:
  - `5p0`
  - `7p5`
  - `10p0`
  - `12p5`
- Top-line overall matrix:
  - `5.0`: recall `20.8%`, novelty `0.0789`, mean WER `0.1472`, MOS delta `-0.1312`
  - `7.5`: recall `16.7%`, novelty `0.1246`, mean WER `0.1681`, MOS delta `-0.1701`
  - `10.0`: recall `16.7%`, novelty `0.1570`, mean WER `0.2028`, MOS delta `-0.2287`
  - `12.5`: recall `16.7%`, novelty `0.1761`, mean WER `0.2444`, MOS delta `-0.2119`
- Focus conclusion:
  - `5.0` remains the safest default
  - `7.5` is the best stronger-than-default compromise on the checked-in non-Trump panel
  - `whisper` and `confused` gain the most novelty from higher strength
  - `10.0-12.5` are better treated as higher-risk, higher-novelty style-specific settings than as new defaults

## Schema

### `eval_emotion_full.csv`
`speaker, style, predicted, target, match, score, emo_sim, file`

- `predicted`: emotion2vec_plus_large argmax label (9 classes)
- `target`: what the generated file is supposed to be
- `match`: 1 if `predicted == mapped(target)`, 0 otherwise, empty for non-emotional styles (confused/enunciated/whisper) and for baseline rows
- `score`: softmax probability of the predicted label
- `emo_sim`: cosine similarity of emotion2vec embedding to same-speaker baseline (empty on baseline rows)

### `eval_wer_full.csv`
`speaker, style, wer, ref_source, reference, hypothesis, file`

- `wer`: word error rate between hypothesis and reference (lower = better, 0 = identical after normalization)
- `ref_source`: `baseline` (same-speaker baseline transcript), `self` (baseline file compared against itself), or `fixed` (when run with `--reference-text`)

### `eval_mos_full.csv`
`speaker, style, mos, delta_vs_baseline, ref_source, file`

- `mos`: SQUIM_SUBJECTIVE predicted MOS, 1–5 scale
- `delta_vs_baseline`: `mos(row) − mos(same-speaker baseline row)`, empty for baseline rows
- `ref_source`: what SQUIM used as the non-matching reference (`baseline` or `self`)

### `eval_novelty_pass2_combined.csv` / `eval_novelty_pass2_cv500.csv`
`speaker, style, source_file, generated_file, baseline_file, similarity, distance, baseline_similarity, baseline_distance, similarity_delta_vs_baseline, distance_delta_vs_baseline, novelty_gain_vs_baseline, noise_level, style_strength, seed, vae_checkpoint, latent_dims`

- `similarity`: cosine similarity between the source speaker embedding and the generated output embedding in OpenVoice's native embedding space
- `distance`: `1 - similarity`
- `baseline_similarity`: cosine similarity between the source speaker embedding and the same-speaker baseline conversion
- `novelty_gain_vs_baseline`: `baseline_similarity - similarity`; positive means the style output is farther from the source than baseline conversion already was

### `eval_ablation_summary_pass4.csv`
`condition, styles_present, styles_count, sources_count, rows_total, emotion_rows_scored, emotion_recall, mean_emo_sim, mean_wer, mean_mos, mean_mos_delta_vs_baseline, mean_novelty_gain_vs_baseline, content_collapse_count, style_collapse_to_neutral_count, identity_collapse_to_baseline_count, mixed_collapse_count, files_with_any_collapse`

- `emotion_recall`: fraction of emotional rows whose predicted label matches target
- `mean_novelty_gain_vs_baseline`: average `baseline_similarity - similarity`
- `mean_mos_delta_vs_baseline`: average style-row MOS minus same-speaker baseline MOS
- collapse counts use the evaluation ablation matrix taxonomy implemented in `scripts/summarize_ablation_results.py`

### `eval_ablation_collapse_pass4.csv`
`condition, file, speaker, style, content_collapse, style_collapse_to_neutral, identity_collapse_to_baseline, mixed_collapse`

- `content_collapse`: `WER >= 0.8`
- `style_collapse_to_neutral`: emotional target predicted as `neutral`
- `identity_collapse_to_baseline`: novelty gain vs baseline `<= 0.05`
- `mixed_collapse`: at least two collapse axes on the same row

### `eval_commonvoice_finetune_summary_pass5.csv`
`condition, styles_present, styles_count, sources_count, rows_total, emotion_rows_scored, emotion_recall, mean_emo_sim, mean_wer, mean_mos, mean_mos_delta_vs_baseline, mean_novelty_gain_vs_baseline, content_collapse_count, style_collapse_to_neutral_count, identity_collapse_to_baseline_count, mixed_collapse_count, files_with_any_collapse, delta_recall_vs_cv500, delta_novelty_vs_cv500, delta_wer_vs_cv500, delta_mos_delta_vs_cv500, delta_recall_vs_combined, delta_novelty_vs_combined`

- same core fields as `eval_ablation_summary_pass4.csv`
- `delta_*_vs_cv500`: direct comparison against the original `commonvoice_cv500_init` condition
- `delta_*_vs_combined`: direct comparison against the main paper checkpoint

### `eval_commonvoice_finetune_collapse_pass5.csv`
`condition, file, speaker, style, content_collapse, style_collapse_to_neutral, identity_collapse_to_baseline, mixed_collapse`

- same taxonomy as evaluation ablation matrix, but applied to the CommonVoice finetune matrix

### `eval_commonvoice_objective_summary_pass6.csv`
`condition, styles_present, styles_count, sources_count, rows_total, emotion_rows_scored, emotion_recall, mean_emo_sim, mean_wer, mean_mos, mean_mos_delta_vs_baseline, mean_novelty_gain_vs_baseline, content_collapse_count, style_collapse_to_neutral_count, identity_collapse_to_baseline_count, mixed_collapse_count, files_with_any_collapse, delta_recall_vs_cv500, delta_novelty_vs_cv500, delta_wer_vs_cv500, delta_mos_delta_vs_cv500, delta_recall_vs_best_ft, delta_novelty_vs_best_ft, delta_wer_vs_best_ft, delta_mos_delta_vs_best_ft, delta_recall_vs_combined, delta_novelty_vs_combined`

- same core fields as `eval_ablation_summary_pass4.csv`
- `delta_*_vs_cv500`: direct comparison against the original `commonvoice_cv500_init` condition
- `delta_*_vs_best_ft`: direct comparison against `cv500_ft_short_low_lr`, the best CommonVoice finetune recipe
- `delta_*_vs_combined`: direct comparison against the main paper checkpoint

### `eval_commonvoice_objective_collapse_pass6.csv`
`condition, file, speaker, style, content_collapse, style_collapse_to_neutral, identity_collapse_to_baseline, mixed_collapse`

- same taxonomy as Passes 4-5, but applied to the CommonVoice objective matrix

### `eval_commonvoice_rich_objectives_summary_pass7.csv`
`condition, styles_present, styles_count, sources_count, rows_total, emotion_rows_scored, emotion_recall, mean_emo_sim, mean_wer, mean_mos, mean_mos_delta_vs_baseline, mean_novelty_gain_vs_baseline, content_collapse_count, style_collapse_to_neutral_count, identity_collapse_to_baseline_count, mixed_collapse_count, files_with_any_collapse, delta_recall_vs_cv500, delta_novelty_vs_cv500, delta_wer_vs_cv500, delta_mos_delta_vs_cv500, delta_recall_vs_best_ft, delta_novelty_vs_best_ft, delta_wer_vs_best_ft, delta_mos_delta_vs_best_ft, delta_recall_vs_combined, delta_novelty_vs_combined`

- same core fields as `eval_ablation_summary_pass4.csv`
- `delta_*_vs_cv500`: direct comparison against the original `commonvoice_cv500_init` condition
- `delta_*_vs_best_ft`: direct comparison against `cv500_ft_short_low_lr`, the best CommonVoice finetune recipe
- `delta_*_vs_combined`: direct comparison against the main paper checkpoint

### `eval_commonvoice_rich_objectives_collapse_pass7.csv`
`condition, file, speaker, style, content_collapse, style_collapse_to_neutral, identity_collapse_to_baseline, mixed_collapse`

- same taxonomy as Passes 4-6, but applied to the CommonVoice rich-objective matrix

### `eval_commonvoice_partial_label_summary_pass8.csv`
`condition, styles_present, styles_count, sources_count, rows_total, emotion_rows_scored, emotion_recall, mean_emo_sim, mean_wer, mean_mos, mean_mos_delta_vs_baseline, mean_novelty_gain_vs_baseline, content_collapse_count, style_collapse_to_neutral_count, identity_collapse_to_baseline_count, mixed_collapse_count, files_with_any_collapse, delta_recall_vs_cv500, delta_novelty_vs_cv500, delta_wer_vs_cv500, delta_mos_delta_vs_cv500, delta_recall_vs_best_ft, delta_novelty_vs_best_ft, delta_wer_vs_best_ft, delta_mos_delta_vs_best_ft, delta_recall_vs_best_rich, delta_novelty_vs_best_rich, delta_wer_vs_best_rich, delta_mos_delta_vs_best_rich, delta_recall_vs_combined, delta_novelty_vs_combined, delta_wer_vs_combined, delta_mos_delta_vs_combined`

- same core fields as the CommonVoice finetune, objective, and rich-objective ablation result bundles
- `best_ft` is `cv500_ft_short_low_lr`
- `best_rich` is `cv500_rich_free_anchor`
- deltas make the CommonVoice partial-label pretraining weak-label conditions directly comparable to the strongest earlier CommonVoice baselines

### `eval_commonvoice_partial_label_collapse_pass8.csv`
`condition, file, speaker, style, content_collapse, style_collapse_to_neutral, identity_collapse_to_baseline, mixed_collapse`

- same taxonomy as Passes 4-7, but applied to the CommonVoice weak-label matrix

## Reproducing

From a clone with the model checkpoint built (see [`examples/README.md`](../examples/README.md) steps 1–4):

```bash
# Generate the 258-file corpus (steps 1-5 of examples/README.md)
python examples/openvoice_infer_controllable.py \
    --source-dir examples/source_speakers/ \
    --out output/diverse_speakers/ \
    --vae-checkpoint embeddings/openvoice_vae_combined.pt \
    --all-styles

# Install eval deps (one-time)
pip install -e ".[eval]"

# Re-run the three evaluations
python examples/eval_emotion.py --input output/diverse_speakers/ --out results/eval_emotion_full.csv
python examples/eval_wer.py     --input output/diverse_speakers/ --out results/eval_wer_full.csv
python examples/eval_mos.py     --input output/diverse_speakers/ --out results/eval_mos_full.csv
```

The generation step also writes `output/diverse_speakers/generation_manifest.jsonl`.
That manifest records the exact source file, output file, style, noise level,
style strength, seed, and checkpoint used for each row in the evaluation
corpus.

Numbers should reproduce within rounding.

## CommonVoice pretraining pipeline Reproduction (`cv500`)

This is the validation-scale CommonVoice pretraining comparison used in
Finding 10. It compares the current combined-only checkpoint against a
CommonVoice-initialized checkpoint trained on a local `cv500` subset
(`500` speakers / `1,202` clips).

```bash
# Combined-only baseline corpus
python examples/openvoice_infer_controllable.py \
    --source-dir examples/source_speakers/ \
    --out output/pass2_combined_eval/ \
    --vae-checkpoint embeddings/openvoice_vae_combined.pt \
    --all-styles \
    --style-strength 5.0 \
    --noise-level 0.0 \
    --seed 42

python examples/eval_emotion.py --input output/pass2_combined_eval/ --out results/eval_emotion_pass2_combined.csv
python examples/eval_novelty.py --manifest output/pass2_combined_eval/generation_manifest.jsonl --out results/eval_novelty_pass2_combined.csv
python examples/eval_wer.py     --input output/pass2_combined_eval/ --out results/eval_wer_pass2_combined.csv

# CommonVoice-pretrained candidate corpus
python examples/openvoice_infer_controllable.py \
    --source-dir examples/source_speakers/ \
    --out output/pass2_cv500_eval/ \
    --vae-checkpoint embeddings/openvoice_vae_combined_cv500.pt \
    --all-styles \
    --style-strength 5.0 \
    --noise-level 0.0 \
    --seed 42

python examples/eval_emotion.py --input output/pass2_cv500_eval/ --out results/eval_emotion_pass2_cv500.csv
python examples/eval_novelty.py --manifest output/pass2_cv500_eval/generation_manifest.jsonl --out results/eval_novelty_pass2_cv500.csv
python examples/eval_wer.py     --input output/pass2_cv500_eval/ --out results/eval_wer_pass2_cv500.csv

# Emotion recall delta
python scripts/compare_emotion_eval.py \
    --baseline results/eval_emotion_pass2_combined.csv \
    --candidate results/eval_emotion_pass2_cv500.csv
```

Expected qualitative outcome from the checked-in CSVs:

- emotion recall gets worse (`25.8%` -> `16.7%`)
- mean novelty gain vs baseline collapses (`0.2599` -> `0.0369`)
- predicted labels collapse almost entirely to `neutral` (`70/110` -> `109/110`)
- WER improves substantially (`0.235` -> `0.084` mean across all non-baseline style rows)

That makes the `cv500` run a useful negative result: better content
preservation, weaker style controllability.

## evaluation ablation matrix Reproduction (ablation matrix)

This is the paper-strengthening ablation pass used in Finding 12. The matrix
compares:

- `combined`
- `commonvoice_cv500_init`
- `cremad_only`
- `expresso_only`
- `naive_noise_baseline`

Prepare and train the two single-dataset ablation conditions:

```bash
python scripts/prepare_ablation_embeddings.py --condition cremad_only
python scripts/prepare_ablation_embeddings.py --condition expresso_only \
    --parquet-dir ~/.cache/huggingface/hub/datasets--ylacombe--expresso/snapshots/*/read

python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_cremad_only_ablation_emb.pt \
    --output embeddings/openvoice_vae_cremad_ablation.pt

python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_expresso_only_ablation_emb.pt \
    --output embeddings/openvoice_vae_expresso_ablation.pt
```

Generate the three new evaluation corpora:

```bash
python scripts/run_ablation_inference.py \
    --source-dir examples/source_speakers/ \
    --condition cremad_only \
    --out output/pass4_cremad_only_eval/ \
    --style-strength 5.0 \
    --noise-level 0.0 \
    --seed 42

python scripts/run_ablation_inference.py \
    --source-dir examples/source_speakers/ \
    --condition expresso_only \
    --out output/pass4_expresso_only_eval/ \
    --style-strength 5.0 \
    --noise-level 0.0 \
    --seed 42

python scripts/run_ablation_inference.py \
    --source-dir examples/source_speakers/ \
    --condition naive_noise_baseline \
    --out output/pass4_naive_noise_baseline_eval/ \
    --style-strength 5.0 \
    --noise-level 0.0 \
    --seed 42
```

The unchanged `combined` and `commonvoice_cv500_init` conditions reuse the
already generated CommonVoice pretraining pipeline corpora:

- `output/pass2_combined_eval/`
- `output/pass2_cv500_eval/`

Run the metrics:

```bash
python examples/eval_emotion.py --input output/pass4_cremad_only_eval --out results/eval_emotion_pass4_cremad_only.csv
python examples/eval_novelty.py --manifest output/pass4_cremad_only_eval/generation_manifest.jsonl --out results/eval_novelty_pass4_cremad_only.csv
python examples/eval_wer.py     --input output/pass4_cremad_only_eval --out results/eval_wer_pass4_cremad_only.csv
python examples/eval_mos.py     --input output/pass4_cremad_only_eval --out results/eval_mos_pass4_cremad_only.csv

python examples/eval_emotion.py --input output/pass4_expresso_only_eval --out results/eval_emotion_pass4_expresso_only.csv
python examples/eval_novelty.py --manifest output/pass4_expresso_only_eval/generation_manifest.jsonl --out results/eval_novelty_pass4_expresso_only.csv
python examples/eval_wer.py     --input output/pass4_expresso_only_eval --out results/eval_wer_pass4_expresso_only.csv
python examples/eval_mos.py     --input output/pass4_expresso_only_eval --out results/eval_mos_pass4_expresso_only.csv

python examples/eval_emotion.py --input output/pass4_naive_noise_baseline_eval --out results/eval_emotion_pass4_naive_noise_baseline.csv
python examples/eval_novelty.py --manifest output/pass4_naive_noise_baseline_eval/generation_manifest.jsonl --out results/eval_novelty_pass4_naive_noise_baseline.csv
python examples/eval_wer.py     --input output/pass4_naive_noise_baseline_eval --out results/eval_wer_pass4_naive_noise_baseline.csv
python examples/eval_mos.py     --input output/pass4_naive_noise_baseline_eval --out results/eval_mos_pass4_naive_noise_baseline.csv
```

Then aggregate:

```bash
python scripts/summarize_ablation_results.py
```

Expected qualitative outcome from the checked-in CSVs:

- `combined` is still the best overall tradeoff
- `cv500`, `cremad_only`, and `expresso_only` are all stability-biased failures that collapse back toward neutral emotion and/or baseline identity
- the naive baseline produces **more novelty** than the combined model, but with worse recall and a much worse MOS delta

## CommonVoice finetune ablation Reproduction (CommonVoice finetune ablation)

This is the narrow follow-up to Finding 10. It keeps the CommonVoice-pretrained
checkpoint fixed and varies only the finetuning recipe.

Train the five new finetune variants:

```bash
python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500_ft_short.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt \
    --epochs 1000 --lr 1e-6

python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500_ft_low_lr.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt \
    --epochs 3000 --lr 3e-7

python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500_ft_short_low_lr.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt \
    --epochs 1000 --lr 3e-7

python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500_ft_freeze_decoder.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt \
    --epochs 3000 --lr 1e-6 --freeze-decoder

python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500_ft_freeze_encoder.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt \
    --epochs 3000 --lr 1e-6 --freeze-encoder
```

Generate the five evaluation corpora:

```bash
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_ft_short --out output/pass5_cv500_ft_short_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_ft_low_lr --out output/pass5_cv500_ft_low_lr_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_ft_short_low_lr --out output/pass5_cv500_ft_short_low_lr_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_ft_freeze_decoder --out output/pass5_cv500_ft_freeze_decoder_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_ft_freeze_encoder --out output/pass5_cv500_ft_freeze_encoder_eval --style-strength 5.0 --noise-level 0.0 --seed 42
```

Run the metrics for each corpus:

```bash
python examples/eval_emotion.py --input output/pass5_cv500_ft_short_eval --out results/eval_emotion_pass5_cv500_ft_short.csv
python examples/eval_novelty.py --manifest output/pass5_cv500_ft_short_eval/generation_manifest.jsonl --out results/eval_novelty_pass5_cv500_ft_short.csv
python examples/eval_wer.py     --input output/pass5_cv500_ft_short_eval --out results/eval_wer_pass5_cv500_ft_short.csv
python examples/eval_mos.py     --input output/pass5_cv500_ft_short_eval --out results/eval_mos_pass5_cv500_ft_short.csv
```

Repeat that four-metric block for:

- `cv500_ft_low_lr`
- `cv500_ft_short_low_lr`
- `cv500_ft_freeze_decoder`
- `cv500_ft_freeze_encoder`

Reuse the unchanged `combined` and `commonvoice_cv500_init` references by
copying their evaluation ablation matrix CSVs into the CommonVoice finetune ablation naming scheme, then summarize:

```bash
python scripts/summarize_commonvoice_finetune_ablation.py
```

Expected qualitative outcome from the checked-in CSVs:

- none of the simple gentler CommonVoice finetune variants recovers the **combined** model's controllability/novelty tradeoff
- `cv500_ft_short_low_lr` is the best partial-recovery condition, mainly by reducing identity collapse rather than improving recall
- `cv500_ft_freeze_encoder` gives the only recall bump, but it loses too much naturalness
- `cv500_ft_freeze_decoder` is a strong negative result and should not be treated as the path forward

## CommonVoice objective ablation Reproduction (CommonVoice objective ablation)

This is the objective-design follow-up to CommonVoice finetune ablation. It keeps the CommonVoice
pretrained checkpoint, the combined embeddings, and the evaluation corpus fixed,
and changes only the loss weighting during finetuning.

Train the four new objective variants:

```bash
python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500_obj_label2.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt \
    --epochs 1000 --lr 3e-7 \
    --label-weight 2.0

python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500_obj_label4.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt \
    --epochs 1000 --lr 3e-7 \
    --label-weight 4.0

python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500_obj_label_ramp.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt \
    --epochs 1000 --lr 3e-7 \
    --label-weight 1.0 \
    --label-weight-final 4.0 \
    --schedule-epochs 1000

python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500_obj_recon_half_label2.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt \
    --epochs 1000 --lr 3e-7 \
    --recon-weight 0.5 \
    --label-weight 2.0
```

Generate the four evaluation corpora:

```bash
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_obj_label2 --out output/pass6_cv500_obj_label2_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_obj_label4 --out output/pass6_cv500_obj_label4_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_obj_label_ramp --out output/pass6_cv500_obj_label_ramp_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_obj_recon_half_label2 --out output/pass6_cv500_obj_recon_half_label2_eval --style-strength 5.0 --noise-level 0.0 --seed 42
```

Run the metrics for each corpus:

```bash
python examples/eval_emotion.py --input output/pass6_cv500_obj_label2_eval --out results/eval_emotion_pass6_cv500_obj_label2.csv
python examples/eval_novelty.py --manifest output/pass6_cv500_obj_label2_eval/generation_manifest.jsonl --out results/eval_novelty_pass6_cv500_obj_label2.csv
python examples/eval_wer.py     --input output/pass6_cv500_obj_label2_eval --out results/eval_wer_pass6_cv500_obj_label2.csv
python examples/eval_mos.py     --input output/pass6_cv500_obj_label2_eval --out results/eval_mos_pass6_cv500_obj_label2.csv
```

Repeat that four-metric block for:

- `cv500_obj_label4`
- `cv500_obj_label_ramp`
- `cv500_obj_recon_half_label2`

Reuse the unchanged `combined`, `commonvoice_cv500_init`, and
`cv500_ft_short_low_lr` references by copying their existing metric CSVs into
the CommonVoice objective ablation naming scheme, then summarize:

```bash
python scripts/summarize_commonvoice_objective_ablation.py
```

Expected qualitative outcome from the checked-in CSVs:

- none of the simple scalar objective variants recovers the **combined** model's controllability/novelty tradeoff
- none of the four new variants improves recall beyond `16.7%`
- `cv500_obj_label2` is the strongest of the new objective variants, but it still trails `cv500_ft_short_low_lr` on novelty and identity collapse
- `cv500_obj_label_ramp` preserves MOS closest to the raw `cv500` init, but only by staying near the same conservative collapse basin
- the next CommonVoice experiments should focus on richer supervision or larger-scale training, not more scalar loss-weight sweeps

## CommonVoice rich-objective ablation Reproduction (CommonVoice rich objectives)

This is the richer-supervision follow-up to CommonVoice objective ablation. It keeps the CommonVoice
init checkpoint, the combined embeddings, and the evaluation corpus fixed, and
changes only the finetune-time supervision by adding style-teacher and
free-anchor losses in latent space.

Train the three new rich-objective variants:

```bash
python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500_rich_teacher_style.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt \
    --style-teacher-checkpoint embeddings/openvoice_vae_combined.pt \
    --style-teacher-weight 2.0 \
    --epochs 1000 --lr 3e-7

python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500_rich_free_anchor.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt \
    --free-anchor-weight 2.0 \
    --epochs 1000 --lr 3e-7

python examples/openvoice_train_vae_combined.py \
    --embeddings embeddings/openvoice_combined_emb.pt \
    --output embeddings/openvoice_vae_combined_cv500_rich_teacher_plus_anchor.pt \
    --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt \
    --style-teacher-checkpoint embeddings/openvoice_vae_combined.pt \
    --style-teacher-weight 2.0 \
    --free-anchor-weight 1.0 \
    --epochs 1000 --lr 3e-7
```

Generate the three evaluation corpora:

```bash
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_rich_teacher_style --out output/pass7_cv500_rich_teacher_style_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_rich_free_anchor --out output/pass7_cv500_rich_free_anchor_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_rich_teacher_plus_anchor --out output/pass7_cv500_rich_teacher_plus_anchor_eval --style-strength 5.0 --noise-level 0.0 --seed 42
```

Run the metrics for each corpus:

```bash
python examples/eval_emotion.py --input output/pass7_cv500_rich_teacher_style_eval --out results/eval_emotion_pass7_cv500_rich_teacher_style.csv
python examples/eval_novelty.py --manifest output/pass7_cv500_rich_teacher_style_eval/generation_manifest.jsonl --out results/eval_novelty_pass7_cv500_rich_teacher_style.csv
python examples/eval_wer.py     --input output/pass7_cv500_rich_teacher_style_eval --out results/eval_wer_pass7_cv500_rich_teacher_style.csv
python examples/eval_mos.py     --input output/pass7_cv500_rich_teacher_style_eval --out results/eval_mos_pass7_cv500_rich_teacher_style.csv
```

Repeat that four-metric block for:

- `cv500_rich_free_anchor`
- `cv500_rich_teacher_plus_anchor`

Reuse the unchanged `combined`, `commonvoice_cv500_init`, and
`cv500_ft_short_low_lr` references by copying their existing metric CSVs into
the CommonVoice rich-objective ablation naming scheme, then summarize:

```bash
python scripts/summarize_commonvoice_rich_objectives.py
```

Expected qualitative outcome from the checked-in CSVs:

- none of the richer teacher/anchor variants recovers the **combined** model's controllability/novelty tradeoff
- none of the three new variants improves recall beyond `16.7%`
- `cv500_rich_free_anchor` is the strongest CommonVoice rich-objective ablation variant, mainly by improving WER and MOS while still trailing `cv500_ft_short_low_lr` on novelty and identity collapse
- `cv500_rich_teacher_style` and `cv500_rich_teacher_plus_anchor` stay in the same conservative neutral-collapse basin
- the next CommonVoice experiments should focus on richer supervision earlier in the pipeline, partial-label/pseudo-label CommonVoice objectives, or larger-scale training once a stronger objective survives on the validation corpus

---

## CommonVoice partial-label pretraining Reproduction (CommonVoice partial-label / pseudo-label pretraining)

CommonVoice partial-label pretraining tests whether adding weak supervision during the CommonVoice stage
itself helps before combined finetuning begins.

First annotate the CommonVoice embedding artifact with pseudo labels:

```bash
python scripts/annotate_commonvoice_pseudolabels.py \
    --embeddings embeddings/openvoice_commonvoice_cv500_emb.pt \
    --output embeddings/openvoice_commonvoice_cv500_pseudo.pt \
    --report-threshold 0.6
```

Train the three weak-supervision CommonVoice pretraining variants:

```bash
python examples/openvoice_pretrain_vae_commonvoice.py \
    --embeddings embeddings/openvoice_commonvoice_cv500_pseudo.pt \
    --output embeddings/openvoice_vae_commonvoice_cv500_pl_meta.pt \
    --epochs 3000 \
    --metadata-targets gender,age_bucket \
    --metadata-weight 0.5

python examples/openvoice_pretrain_vae_commonvoice.py \
    --embeddings embeddings/openvoice_commonvoice_cv500_pseudo.pt \
    --output embeddings/openvoice_vae_commonvoice_cv500_pl_pseudo_style.pt \
    --epochs 3000 \
    --pseudo-style-weight 1.0 \
    --pseudo-style-threshold 0.6

python examples/openvoice_pretrain_vae_commonvoice.py \
    --embeddings embeddings/openvoice_commonvoice_cv500_pseudo.pt \
    --output embeddings/openvoice_vae_commonvoice_cv500_pl_meta_plus_pseudo.pt \
    --epochs 3000 \
    --metadata-targets gender,age_bucket \
    --metadata-weight 0.5 \
    --pseudo-style-weight 1.0 \
    --pseudo-style-threshold 0.6
```

Finetune on the combined labeled embeddings:

```bash
python examples/openvoice_train_vae_combined.py --embeddings embeddings/openvoice_combined_emb.pt --output embeddings/openvoice_vae_combined_cv500_pl_meta.pt --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500_pl_meta.pt --epochs 1000 --lr 3e-7
python examples/openvoice_train_vae_combined.py --embeddings embeddings/openvoice_combined_emb.pt --output embeddings/openvoice_vae_combined_cv500_pl_pseudo_style.pt --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500_pl_pseudo_style.pt --epochs 1000 --lr 3e-7
python examples/openvoice_train_vae_combined.py --embeddings embeddings/openvoice_combined_emb.pt --output embeddings/openvoice_vae_combined_cv500_pl_meta_plus_pseudo.pt --init-checkpoint embeddings/openvoice_vae_commonvoice_cv500_pl_meta_plus_pseudo.pt --epochs 1000 --lr 3e-7
```

Generate the three matched corpora:

```bash
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_pl_meta --out output/pass8_cv500_pl_meta_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_pl_pseudo_style --out output/pass8_cv500_pl_pseudo_style_eval --style-strength 5.0 --noise-level 0.0 --seed 42
python scripts/run_ablation_inference.py --source-dir examples/source_speakers/ --condition cv500_pl_meta_plus_pseudo --out output/pass8_cv500_pl_meta_plus_pseudo_eval --style-strength 5.0 --noise-level 0.0 --seed 42
```

Run the four metrics on each corpus, reuse the unchanged reference CSVs in the
CommonVoice partial-label pretraining naming scheme, then summarize:

```bash
python examples/eval_emotion.py --input output/pass8_cv500_pl_meta_eval --out results/eval_emotion_pass8_cv500_pl_meta.csv
python examples/eval_novelty.py --manifest output/pass8_cv500_pl_meta_eval/generation_manifest.jsonl --out results/eval_novelty_pass8_cv500_pl_meta.csv
python examples/eval_wer.py     --input output/pass8_cv500_pl_meta_eval --out results/eval_wer_pass8_cv500_pl_meta.csv
python examples/eval_mos.py     --input output/pass8_cv500_pl_meta_eval --out results/eval_mos_pass8_cv500_pl_meta.csv
python scripts/summarize_commonvoice_partial_label.py
```

Expected qualitative outcome from the checked-in CSVs:

- none of the weak-label variants improves recall beyond `16.7%`
- `cv500_pl_meta` is the best novelty result of the new variants (`0.0570`), but it still trails `cv500_ft_short_low_lr` (`0.0692`) and `cv500_rich_free_anchor` (`0.0646`)
- `cv500_pl_pseudo_style` and `cv500_pl_meta_plus_pseudo` produce the best WER of any CommonVoice variants tested so far (`0.0263` and `0.0285`), but they do so by collapsing toward baseline identity (`85-90` identity-collapse rows)
- the next CommonVoice work should focus on better pseudo-label quality, stronger teacher/prototype targets during CommonVoice pretraining, or curriculum strategies rather than simply adding these weak labels at the current validation scale
