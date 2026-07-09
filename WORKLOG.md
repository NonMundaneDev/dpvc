# Controllable DP Voice Conversion — Work Log

**Last updated:** 2026-06-13
**Branches:** `feat/controlvc`, `feat/openvoice-expresso`, `feat/f0-style-control`, `feat/cremad-experiments`, `feat/openvoice-pipeline-stabilization`, `feat/commonvoice-pretrain`, `feat/speaker-novelty-metric`, `research/eval-ablations`, `research/commonvoice-finetune-ablation`, `research/commonvoice-objective-ablation`, `research/commonvoice-rich-objectives`, `research/commonvoice-partial-label-pretrain`, `research/combined-data-pseudolabel-mix`, `research/mixed-data-pseudolabel-quality`, `research/nontrump-style-strength-sweep`, `integration/research-rollup`, `research/controllable-vae`, `research/commonvoice-metadata-controls`, `docs/paper-methods-and-evidence`, `research/external-speaker-verifier`, `research/metadata-separability-probe`, `research/generated-audio-content-repair`, `research/generated-audio-calibrated-objective`, `research/generated-audio-calibrated-training`, `research/control-selection-evaluation`, `research/control-shortlist`, `research/control-feedback-gender-preflight`, `research/commonvoice-gender-followup`
**Author:** Stephen Oladele (with Claude, and Joe Near's upstream work)

---

## 0. Roadmap

Priority tags:
- `NOW` = immediate next work; unblocks reproducibility or addresses the main research bottleneck
- `SOON` = important after the `NOW` items are stable
- `LATER` = downstream, stretch, or productization work

### Phase 0.5: Consolidate Active Path
- [x] `[NOW]` Consolidate the active controllable pipeline around OpenVoice; treat ControlVC as a useful DP baseline and negative result for style control

### Phase 0.6: Repository Consolidation
- [x] `[DONE]` Revert the research rollup from upstream `main` after Joe clarified that `uvm-plaid/dpvc/main` should stay a stable reflection of the published work
- [x] `[DONE]` Move ongoing controllable-VAE research to the fork `NonMundaneDev/dpvc`, with canonical active branch `research/controllable-vae`
- [ ] `[SOON]` Keep future research pushes on the fork and avoid opening upstream-main PRs until Joe explicitly wants a paper/research integration surface

### Phase 1: Core Controllability (prove it works)
- [x] End-to-end pipeline: extract → train VAE → infer → intelligible speech
- [x] Demonstrate style control with OpenVoice (not ControlVC — Joe confirmed, April 9)
- [x] Speaker diversity: train on 91+ speakers (CREMA-D) so VAE learns style, not speaker
- [x] Combined dataset (CREMA-D + Expresso): 9 styles, 94 speakers, all perceptually distinct
- [x] Acoustic analysis confirming style differences (F0, energy, spectral centroid)
- [x] Test style control survives DP noise (noise_level 0–1.0, styles persist at 0.1, degrade gracefully)
- [x] Test on diverse source speakers (5 speakers: brightness consistent 7/9 styles, F0 inconsistent, 9% collapse rate)
- [x] Per-speaker style evaluation: brightness is the reliable cross-speaker metric, not F0

### Phase 1.5: Scale Up Speaker Diversity (Joe's suggestion, April 16)
- [x] `[NOW]` Extract OpenVoice embeddings from CommonVoice (20k+ speakers, no style labels)
- [x] `[NOW]` Pre-train VAE on CommonVoice (reconstruction loss only)
- [x] `[NOW]` Finetune on CREMA-D + Expresso (style labels, label loss)
- [x] `[NOW]` Compare with current combined-only VAE: does pre-training reduce collapse rate?
- [ ] `[SOON]` Scale the validated `cv500` CommonVoice recipe to a much larger local slice or a full English mirror before drawing a final conclusion about pre-training
- [x] `[SOON]` Test gentler finetuning from the CommonVoice init checkpoint (fewer finetune epochs, lower LR, or partial-freeze) to see whether the `cv500` neutral-collapse failure is an over-regularization problem rather than a dead-end. **CommonVoice finetune ablation result:** simple recipe changes recover only small novelty gains (`0.0369 -> 0.0692` best case) and do not recover the combined model's recall/novelty tradeoff
- [ ] `[SOON]` Scale the best CommonVoice finetune ablation recipe (`cv500_ft_short_low_lr`) to a larger local CommonVoice slice and test whether its partial novelty recovery survives at higher speaker counts
- [x] `[SOON]` Test layer-granular freezing or loss-weight schedules instead of full encoder/decoder freezes; CommonVoice objective ablation result: simple CommonVoice loss reweighting (`label2`, `label4`, label-ramp, reduced-reconstruction) still leaves recall flat at `16.7%` and does not beat `cv500_ft_short_low_lr`
- [ ] `[SOON]` Compare reconstruction-only CommonVoice pretraining against partially labeled or multi-objective pretraining, because Passes 6-7 suggest finetune-time reweighting and teacher/anchor finetuning are still too weak when the CommonVoice stage itself remains purely reconstruction-driven
- [x] `[SOON]` Test richer CommonVoice adaptation objectives (teacher or latent anchoring, curriculum label emphasis, or partial-label pretraining) because CommonVoice objective ablation shows simple loss-weight ramps do not restore recall or beat the best CommonVoice finetune ablation novelty recovery. **CommonVoice rich-objective ablation result:** teacher-style distillation and free-dim anchoring preserve WER/MOS better than some earlier variants, but recall stays flat at `16.7%` and none beats `cv500_ft_short_low_lr` on novelty or identity collapse
- [x] `[SOON]` Test richer CommonVoice adaptation objectives that use partial labels, pseudo-labels, or a better pretraining objective on CommonVoice itself; CommonVoice partial-label pretraining result: metadata-only weak supervision gives modest novelty recovery (`0.0570`) but no recall gain, while pseudo-style supervision drives WER down sharply (`0.0263-0.0285`) at the cost of even lower novelty (`0.0190-0.0181`) and much higher identity collapse (`85-90`)
- [ ] `[SOON]` Compare teacher-style distillation against prototype-style or curriculum supervision in the labeled style subspace, because CommonVoice rich-objective ablation's latent teacher/anchor losses improved stability more than control
- [ ] `[SOON]` Improve CommonVoice pseudo-label quality and calibration (better teacher, confidence filtering, or class-balanced acceptance), because CommonVoice partial-label pretraining suggests the current pseudo-style labels over-regularize the model into a conservative neutral / baseline-identity basin
- [ ] `[SOON]` Test prototype-style or teacher-embedding targets during CommonVoice pretraining itself, not just combined finetuning, because CommonVoice partial-label pretraining's label-space pseudo supervision preserved intelligibility much more than controllability
- [ ] `[SOON]` Compare metadata-only weak supervision against stronger free-dim supervision (for example: age/gender/accent + auxiliary speaker-structure constraints), because CommonVoice partial-label pretraining suggests metadata shapes novelty a little but does not recover recall
- [ ] `[SOON]` Add per-style recovery plots for the CommonVoice finetune and objective variants; the finetune, objective, and rich-objective ablations suggest novelty returns first for a few conservative styles, not as broad emotion recovery
- [x] `[DONE]` Add age/gender control dims using CommonVoice metadata (dims 9-10) to the mixed-artifact/training/inference plumbing; next step is a real checkpoint/evaluation panel, not another interface change
- [ ] `[SOON]` Test orthogonality: does pushing emotion dims shift perceived age/gender?
- [ ] `[SOON]` Test whether CommonVoice-broad pretraining preserves age/gender control more easily than emotion control; CommonVoice finetune ablation suggests different attribute families may survive broad speaker priors differently
- [ ] `[SOON]` **Open question (Joe, April 16):** Can we train all knobs at once when labels come from different datasets? CommonVoice has age/gender, CREMA-D has emotion — each stage only trains a subset of latent dims

### Phase 1.6: Mixed-Data Bootstrap (Joe's April 30 call)
- [x] `[DONE]` Run the first sampled mixed-data experiment that trains on **CommonVoice + CREMA-D + Expresso together**; completed on branch `research/combined-data-pseudolabel-mix`, with all three schedule variants improving WER more than controllability and remaining fixed at `16.7%` recall
- [x] `[DONE]` Build the mixed-data corpus around **speaker breadth first**, not raw CommonVoice clip count; completed in `embeddings/openvoice_mixed_base.pt`, which keeps `500` CommonVoice speakers with one clip per speaker
- [x] `[DONE]` Compare at least three mixture schedules for the first mixed-data run; completed with `mixed_static_balanced`, `mixed_cv_warmup`, and `mixed_labeled_finish`
- [x] `[DONE]` Protect the small labeled datasets inside the first mixed-data run; completed with schedule-aware dataset masses, but the result still suggests stronger labeled-data protection is the next higher-value change
- [x] `[DONE]` Start `research/mixed-data-pseudolabel-quality`: improved CommonVoice pseudo-label filtering inside the mixed-data builder with per-class thresholds/caps, accepted-vs-rejected reporting, and cap-aware fallback selection; the improved artifact keeps `500` CommonVoice speakers but reduces pseudo-labeled CommonVoice rows from `462` to `300`
- [x] `[DONE]` Add configurable dataset-mass controls and test a more aggressively labeled-protected mixed schedule; completed with `mixed_quality_static_balanced`, `mixed_quality_labeled_finish`, and `mixed_quality_labeled_guarded`
- [x] `[DONE]` Re-run the mixed-data evaluation matrix on the improved artifact and compare it against the original mixed schedules; result: `mixed_quality_labeled_guarded` becomes the first mixed-data condition to move recall above `16.7%`, reaching `18.2%`, but the gain comes with worse WER (`0.0978`) and weaker novelty (`0.0764`) than the best original mixed schedules
- [ ] `[SOON]` Clean up and enrich the Expresso label mapping inside the mixed-data follow-up, because Joe agreed the richer Expresso label space is still one of the best ways to inject emotion structure into the broader CommonVoice speaker prior
- [ ] `[SOON]` Compare one-clip-per-speaker versus two-clips-per-speaker CommonVoice sampling inside the mixed-data setup, because the current real `openvoice_mixed_base.pt` artifact uses one clip per speaker and Joe's speaker-breadth heuristic still needs a direct empirical check
- [x] `[DONE]` Tighten the mixed-data pseudo-label teacher and per-class acceptance rules further; the expanded rare-supply teacher run reached `47.0%` recall and `0.2995` novelty gain, though the next work must repair WER/MOS
- [x] `[DONE]` Add per-class CommonVoice pseudo-label thresholds or caps to the mixed-data builder; the quality artifact now records thresholds, threshold-rejected counts, cap-skipped counts, and final CommonVoice pseudo-style counts in `mixture_report`
- [x] `[DONE]` Save one artifact-level `mixture_report` snapshot per mixed-data pseudolabel mix condition alongside the result bundle; the quality artifact now preserves threshold and row-weight config directly inside `embeddings/openvoice_mixed_quality_base.pt`
- [x] `[DONE]` Run style-strength sweeps above `5.0` on representative non-Trump speakers on a dedicated follow-up branch `research/nontrump-style-strength-sweep`; result: `5.0` remains the safest default, `7.5` is a defensible stronger setting for whisper/confused when higher novelty matters, and `10.0-12.5` behave more like high-novelty demo settings with clearly worse overall WER/MOS
- [ ] `[SOON]` Add a concise "how to read the metrics" guide for Joe covering emotion recall / emo_sim, novelty, WER, and MOS, because he explicitly said the branch and metric layout is hard to interpret quickly
- [ ] `[SOON]` Extend the non-Trump strength sweep to a larger panel and compare `combined` against `mixed_quality_labeled_guarded`, because the first 4-speaker sweep shows that higher strengths are usable for whisper/confused but not yet broad enough to freeze a universal style-strength policy
- [ ] `[SOON]` Add style-specific inference guidance or presets (`default`, `strong-whisper`, `strong-confused`), because the non-Trump sweep shows that a single global `style_strength` default hides meaningful style-dependent tradeoffs
- [x] `[DONE]` Move the controllable-VAE line back out of upstream `main` and into the research fork `NonMundaneDev/dpvc`; upstream `main` now tracks the stable published-work state again, while the canonical research branch is `research/controllable-vae`
- [x] `[DONE]` Train the first teacher-focused mixed-data checkpoint family on canonical branch `research/controllable-vae`; `examples/openvoice_train_vae_mixed.py` now produced `embeddings/openvoice_vae_mixed_teacher_threshold_balanced.pt`, `embeddings/openvoice_vae_mixed_teacher_labeled_finish.pt`, and `embeddings/openvoice_vae_mixed_teacher_labeled_guarded.pt` from `embeddings/openvoice_mixed_teacher_base.pt`, and `scripts/run_ablation_inference.py` now exposes the matching condition names for the evaluation step
- [x] `[DONE]` Score the first teacher-focused mixed-data checkpoint family on canonical branch `research/controllable-vae`; generated `output/mixed_teacher_*_eval/` corpora, wrote the full `eval_*_mixed_teacher_*` CSV bundle plus `results/eval_mixed_teacher_summary.csv`, and verified that `mixed_teacher_threshold_balanced` matches the `18.2%` mixed-data recall bump while improving WER/MOS/novelty versus `mixed_quality_labeled_guarded`
- [x] `[DONE]` Add stronger teacher diagnostics to the mixed-data artifact flow; the teacher branch now records top-k teacher labels/scores, optional per-style score maps, row-level filter decisions, and preserves pseudo-label report/filter metadata inside the mixed artifact `mixture_report`
- [x] `[DONE]` Preserve a reusable score -> filter -> build flow for CommonVoice pseudo labels; `scripts/annotate_commonvoice_pseudolabels.py`, `scripts/filter_commonvoice_pseudolabels.py`, and `scripts/build_mixed_training_set.py --acceptance-policy artifact_selected` now let future teacher comparisons reuse one scored artifact across multiple acceptance policies
- [x] `[DONE]` Re-score the full `cv500` CommonVoice artifact with the updated annotate script before the first teacher matrix training run; `embeddings/openvoice_commonvoice_cv500_pseudo_scored.pt` now provides branch-native teacher metadata, top-k scores, and mapped style-score totals, and `embeddings/openvoice_mixed_teacher_base.pt` is rebuilt from the scored -> filtered artifact path
- [x] `[DONE]` Address rare-class supply limits inside the teacher branch after full rescoring; the expanded CommonVoice rare-supply artifact now selects `anger=50` / `fear=50`, and the final mixed artifact preserves those counts while keeping `13308` CommonVoice speakers
- [x] `[DONE]` Test a softer mapped-score teacher-agreement rule inside the mixed-data teacher branch; `scripts/annotate_commonvoice_pseudolabels.py` now records a second center-crop teacher view plus score maps, `scripts/filter_commonvoice_pseudolabels.py` now supports `--secondary-agreement-mode mapped_score`, and the full `mixed_teacher_mapped015_balanced` checkpoint/eval bundle proved that the looser agreement rule raises novelty slightly (`0.0818`) but still loses to `mixed_teacher_threshold_balanced` on recall (`16.7%` vs `18.2%`), WER (`0.1090` vs `0.0829`), and MOS delta (`-0.1115` vs `-0.1012`)
- [x] `[DONE]` Compare a genuinely different pseudo-label teacher inside the mixed-data teacher branch; `scripts/annotate_commonvoice_latent_prototypes.py` now uses combined-VAE latent style prototypes as an alternate teacher, and `mixed_teacher_prototype_balanced` improves novelty (`0.0854`) and identity/mixed collapse while tying the best mixed-data recall (`18.2%`) but giving back WER/MOS versus `mixed_teacher_threshold_balanced`
- [x] `[DONE]` Test a guarded prototype-teacher variant with pseudo-confidence scaling, lower pseudo row weight, and stronger true-label protection; `mixed_teacher_prototype_guarded` ties the `18.2%` recall ceiling and improves WER versus the unguarded prototype (`0.0920` vs `0.1009`), but gives back the prototype novelty advantage (`0.0761` vs `0.0854`) and worsens identity/mixed collapse, so it does not replace `mixed_teacher_threshold_balanced`
- [x] `[DONE]` Compare a prototype+emotion2vec multi-teacher rule; `mixed_teacher_hybrid_extra_balanced` produced the best mixed-teacher novelty so far (`0.0860`) and slightly reduced files with any collapse, but dropped recall to `16.7%` and worsened MOS delta, so it is a tradeoff result rather than the new reference
- [x] `[DONE]` Move from hard row-label teacher mixing to richer style-space supervision, prototype distillation, or a per-style curriculum; the first continuous style-space distillation run preserved the hybrid novelty gain and improved MOS/collapse modestly, but recall stayed fixed at `16.7%`
- [x] `[DONE]` Calibrate the style-space distillation objective with a first teacher-loss weight sweep; global weights `0.10`, `0.25`, and `0.50` all stayed at `16.7%` recall, so the next move is class-specific masks/curriculum rather than another scalar weight tweak
- [x] `[DONE]` Add per-style teacher masks, confidence weighting, and a first labeled-first curriculum for the style-space loss; target masking and labeled warmup both stayed at `16.7%` recall, so curriculum timing with the current teacher is not enough to recover target emotion recall
- [x] `[DONE]` Design and test a first decoder-aware style objective for canonical emotions; the decoder-prototype pilot is implemented and evaluated, but it is a cautionary baseline rather than the new reference because it preserves novelty while lowering recall and worsening WER/collapse versus the `sad/enunciated` inference guard
- [x] `[DONE]` Add a CommonVoice rare-supply preflight gate before more weighting experiments; it scans local `validated.tsv` + `clips/`, estimates the needed row count from the checked-in pseudo-label artifacts, and now returns `GO` on the expanded local corpus at `/Users/steve/datasets/cv-corpus-21.0-2025-03-14/en` (`40000` usable rows / `20537` speakers)
- [x] `[DONE]` Extract OpenVoice embeddings from the expanded local CommonVoice corpus, then score/filter/audit pseudo labels before training; `embeddings/openvoice_mixed_teacher_cvrare_hybrid_extra_base.pt` now passes the rare-label supply gate with `anger=50` / `fear=50`
- [x] `[DONE]` Evaluate the expanded rare-supply checkpoint and make a listening report; `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup` reached `46.97%` emotion recall and `0.2995` novelty gain, but exposed a content/naturalness tradeoff (`0.2751` mean styled WER, `-0.2640` MOS delta)
- [x] `[DONE]` Add inference-side per-style strength profiles and a one-command generated-audio eval runner; the narrow `sad/enunciated` guard keeps the expanded checkpoint at `47.0%` recall while improving WER to `0.2348`, MOS delta to `-0.2081`, and files with any collapse to `20`
- [x] `[DONE]` Add a decoder-prototype training objective and first pilot; `mixed_teacher_cvrare_decoder_proto_labeled_warmup` reaches `42.4%` recall and `0.3008` novelty gain, but does not beat the `sad/enunciated` guard on WER, MOS, or collapse
- [x] `[DONE]` Evaluate the decoder-prototype checkpoint with the existing `cvrare_sad_enunc_guard` style-strength map; the guard improves WER/MOS (`0.2592`, `-0.1787`) but does not recover recall (`42.4%`) or collapse (`28` files), so the decoder-prototype checkpoint remains diagnostic rather than a reference
- [x] `[DONE]` Try a lower-weight decoder-prototype variant before abandoning the simple weight family; `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` keeps recall at `42.4%`, improves novelty to `0.3032`, and lowers any-collapse to `26`, but still misses the `sad/enunciated` guard on recall/WER/MOS/collapse
- [x] `[DONE]` Build a generated-audio calibration / failure-mining artifact that scores pilot generations with emotion2vec, WER, MOS, novelty, and collapse labels; it confirms the current `sad/enunciated` guard has the lowest row-level failure score and localizes persistent failures to `disgust`, `fear`, and `anger`
- [x] `[DONE]` Build the failure-conditioned target selector; it emits clean target decisions from the generated-audio failure table and marks `anger` / `disgust` as ready while blocking `fear` because the current guard has `0/11` clean fear targets
- [x] `[DONE]` Run one conservative failure-conditioned style-teacher follow-up using only ready styles (`anger`, `disgust`) with `target_dim` teacher supervision; it is a useful negative result because recall drops to `39.4%`, style-to-neutral collapse rises to `26`, and the current `sad/enunciated` guard remains the reference
- [x] `[DONE]` Test an explicit anti-neutral prototype-margin objective for `anger`/`disgust`; result: the embedding-space proxy is a useful negative result because recall reaches only `40.9%`, style-to-neutral collapse remains `25`, and the current `sad/enunciated` guard remains the quality-balanced reference
- [x] `[DONE]` Move beyond embedding-space proxies toward a true generated-audio-calibrated intervention; the first generated-audio style-strength grid over `anger`, `disgust`, and `fear` is now checked in, with ranking/listening artifacts showing style-specific candidates but no safe universal default
- [x] `[DONE]` Build a single A/B perceptual-review dashboard for the best grid cells (`anger_s10`, `disgust_s10`, `fear_s7p5`) against the current `sad/enunciated` guard
- [x] `[DONE]` Add an objective-assisted A/B triage sheet so perceptual review starts with the most informative rows instead of all 33 pairs
- [x] `[DONE]` Build a priority-only A/B listening dashboard from the five triaged rows so perceptual review can start with the cleanest target-gain candidates
- [x] `[DONE]` Complete Joe's first five-row human/perceptual review from the A/B dashboard; result was `4` ties/indistinguishable, `1` reference preference, and `0` candidate wins, so no style-specific preset is promoted yet
- [x] `[DONE]` Create the canonical evidence/demo packet and listening index so the substantial current result can be reviewed without branch archaeology
- [x] `[DONE]` Add a Joe-facing metric and collapse taxonomy guide, especially clarifying that identity collapse is low novelty gain vs baseline, not WER
- [x] `[DONE]` Audit CommonVoice metadata coverage for age/gender controls; the local 40k-clip subset has `5504` age-control rows, `5291` binary gender-control rows, and `5258` rows with both labels
- [x] `[DONE]` Implement masked direct metadata-control supervision for age/gender on `research/commonvoice-metadata-controls`, because Joe's May 14 feedback reframed emotion as one controllable speaker attribute rather than the only target
- [x] `[DONE]` Rebuild the full `openvoice_mixed_teacher_cvrare_hybrid_extra_base` artifact with the new metadata tensors, train the first metadata-control checkpoint, and generate a small age/gender listening panel before claiming perceptual control
- [x] `[DONE]` Listen to `results/listening_metadata_w010_labeled_warmup.html`; local perceptual review found the variants sounded identical or like generic speaker/timbre shifts, so this first metadata-control checkpoint is diagnostic rather than a perceptual age/gender-control win
- [x] `[DONE]` Start paper-method documentation for architecture, data mixture, training schedule, and evaluation justification; see `PAPER_METHODS_AND_EVIDENCE.md` and `IMPLEMENTATION_PLAN_paper-methods-and-evidence.md`
- [x] `[DONE]` Do not spend WER/MOS/novelty compute on the first metadata-control checkpoint unless needed for documentation; the perceptual gate failed, so metrics would likely characterize generic speaker shift rather than useful age/gender control
- [x] `[DONE]` Add an external speaker-verifier / EER-style novelty validation branch; SpeechBrain ECAPA corroborates the current guard's identity shift with mean styled external novelty gain `0.3594` and only `6/99` styled rows accepted as source at the derived proxy threshold
- [x] `[DONE]` Add a metadata separability probe before more age/gender training; result: gender is strongly separable in raw embeddings and VAE latents, but age is weak and accent mostly washes out in the metadata-control latent space, so more scalar age/accent training should wait for better labels or a stronger perceptual target
- [x] `[DONE]` Add a generated-audio content-repair gate for the hard-style strength grid; result: no `anger`, `disgust`, or `fear` candidate is promoted because the only objective-pass rows were blocked by Joe's perceptual review, and `disgust` has no objective-safe repair row
- [x] `[DONE]` Build a generated-audio-calibrated objective plan and trainer hook for hard styles; the branch selects `anger`/`disgust`, blocks `fear`, and adds decoder-prototype style weights so non-target styles do not receive repair pressure
- [x] `[DONE]` Train and evaluate `mixed_teacher_cvrare_audio_calibrated_labeled_warmup` from `results/generated_audio_calibrated_objective_plan.md`; result is a useful negative/diagnostic finding because it confirms the trainer path but loses recall and does not repair `anger`/`disgust` targeting versus the current `sad/enunciated` guard
- [x] `[DONE]` Listen to `results/listening_mixed_teacher_cvrare_audio_calibrated_labeled_warmup.html`; Stephen's first perceptual review said `disgust` sounded convincingly disgusted and intelligible, but Joe's focused review did not confirm it
- [x] `[DONE]` Ask Joe for a focused perceptual confirmation on the audio-calibrated `disgust` and `anger` rows; Joe heard `disgust` as neutral across rows and `anger` as only slightly/source-dependently angry in early CREMA-D rows
- [x] `[DONE]` Record the May 28 Joe meeting direction update; Joe said the system is basically working and the next work should focus on paper-facing evaluation, control selection, and simplification rather than more open-ended model improvement
- [x] `[DONE]` Run the first source training-data style separability audit on branch `research/control-selection-evaluation`; CREMA-D emotion labels are strongly separable by emotion2vec direct recall, while Expresso-only `confused` is weak and `enunciated` / `whisper` are quality-sensitive embedding-space controls
- [x] `[DONE]` Convert the separability audit into a paper/demo control shortlist on branch `research/control-shortlist`; current result promotes no fully paper-ready style claim yet, makes `neutral` and `sad` candidate headline controls pending focused listening, keeps `happy` / `enunciated` / `whisper` quality-sensitive, and keeps `anger` / `disgust` / `fear` / `confused` diagnostic or limitation controls
- [x] `[DONE]` Ingest Joe's focused `neutral`/`sad` listening feedback with `scripts/ingest_control_selection_feedback.py`; the shortlist now promotes `neutral` and `sad` to headline controls, with `sad` explicitly caveated as perceptible but subtle
- [x] `[DONE]` Preflight the local CommonVoice gender follow-up before training; `/Users/steve/datasets/cv-corpus-21.0-2025-03-14/en` has enough balanced gender-known speaker coverage for a gender-only follow-up manifest, but this is data-readiness only
- [x] `[DONE]` Start the gender-focused follow-up from `results/commonvoice_gender_followup_speakers.csv`; branch `research/commonvoice-gender-followup` builds an available-subset artifact, trains the first gender-only checkpoint, and creates a review bundle for perceptual gating
- [x] `[DONE]` Listen to `results/gender_followup_available_w005_review_bundle_2026-06-13.zip` locally before sending broader claims to Joe; local review found the female/male controls were not reliably perceptible as gender controls and sounded more like subtle/generic speaker-timbre movement, so gender is not promoted
- [x] `[DONE]` Start paper-facing simplification around the current reference guard by adding a conservative shortlist gate: source separability decides which labels are fair to evaluate, generated-output metrics decide which labels are plausible, and listening evidence decides which labels can become headline claims
- [ ] `[SOON]` Mark accent explicitly out of scope for the current OpenVoice speaker-embedding VAE path, because Joe expects accent information to live in the content representation rather than the speaker embedding
- [ ] `[SOON]` Reframe age as optional broad-bucket classification only, not continuous scalar control; keep it lower priority than gender and top-style selection
- [ ] `[SOON]` Add an eval-suite preflight for `ffmpeg` / `torchcodec`, because WER evaluation required `PATH=/opt/homebrew/bin:$PATH` on this macOS machine even though the repo virtualenv was otherwise ready
- [ ] `[LATER]` Only design another hard-style repair if the separability audit shows the target label is perceptually/classifier separable in the training data; otherwise weak labels such as `disgust` remain limitations/future work
- [ ] `[SOON]` Add a fear-specific diagnostic for the pitch-change artifact Joe heard in `fear_s7p5`, because fear can gain target recall but remains content/naturalness fragile
- [ ] `[SOON]` Do not use the decoded-teacher `teacher_margin` anti-neutral proxy without calibration; smoke diagnostics showed zero loss on the selected `anger`/`disgust` rows even though generated audio still collapsed toward neutral
- [ ] `[SOON]` Revisit agreement-style filtering with class-specific secondary support only after richer style-space supervision is planned, because the current single-teacher and hybrid row-label paths improve novelty slightly but stay in the same neutral / baseline-identity basin
- [ ] `[SOON]` Compare strict pseudo-label filtering against looser confidence-only or minimally filtered CommonVoice pseudo labels, because Joe's May 14 question raised a valid possibility that filtering may discard useful breadth once all CommonVoice rows have weak labels
- [ ] `[SOON]` If metadata controls stay in scope, run a narrow gender-focused balanced-control follow-up rather than another broad age/gender/accent sweep; the separability probe shows gender has objective structure, while the first listening panel still says perceptual controllability is unproven
- [ ] `[SOON]` Add per-dimension metadata latent diagnostics for dims `9-10` and free dims `11-14`, because the separability probe confirms gender survives in `vae_mu` but does not prove the intended scalar control dim is the one carrying the signal
- [ ] `[SOON]` Extract the 607 missing clips from `results/commonvoice_gender_followup_speakers.csv` only if gender is revisited with a materially stronger objective; the current available-subset artifact matched `1181/1788` selected clips and failed the perceptual gate, so scaling the same checkpoint is not the next move
- [ ] `[SOON]` Do not retry age/accent scalar controls without better labels, class balancing, or explicit perceptual/acoustic targets; the current probe finds weak age structure and weak-to-moderate accent structure that does not survive strongly in metadata-control latents
- [ ] `[SOON]` Build an independent labeled speaker-verification trial CSV for final EER, because the current ECAPA threshold is derived from source-vs-baseline proxy trials
- [ ] `[SOON]` Add repeated-seed confidence intervals for the current reference tables before freezing final paper claims, because most ablations so far use deterministic single-seed comparisons
- [ ] `[SOON]` Add formal DP accounting and privacy-utility curves before submission; the current strongest evidence is controllability/quality, while privacy accounting remains an explicit paper task
- [x] `[DONE]` Convert the hand-authored per-style strength profiles into a small reproducible grid/optimizer over style strengths; the first grid is intentionally narrow and should be expanded only after perceptual review confirms the ranked cells sound useful
- [ ] `[SOON]` Add a browser index for grid listening reports, because the grid now produces many per-cell HTML/rating files and manual link-hunting is error-prone
- [ ] `[SOON]` Include anticipated questions and concise answers in every future Joe-facing meeting brief, because the May 14 meeting showed predictable questions about pseudo-labeling, filtering, collapse, DP, and evaluation should be pre-answered
- [x] `[DONE]` Persist teacher-branch evaluation corpora and summary artifacts under stable `mixed_teacher_*` names; the branch now has `output/mixed_teacher_threshold_balanced_eval/`, `output/mixed_teacher_labeled_finish_eval/`, `output/mixed_teacher_labeled_guarded_eval/`, and the checked-in `results/eval_mixed_teacher_summary.csv` / `results/eval_mixed_teacher_collapse.csv` bundle

### Phase 2: Evaluation (Joe: emotion eval is #1 priority)
- [x] **Research TTS controllability evaluation metrics** — settled on EmoVoice pipeline (arxiv 2504.12867, Joe's suggestion): emotion2vec Recall Rate + emo_sim (primary), UTMOS (naturalness), WER (intelligibility)
- [x] Build emotion evaluation pipeline: `examples/eval_emotion.py` runs emotion2vec_plus_large on a directory of generated audio and writes a CSV with per-file Recall Rate and emo_sim vs. same-speaker baseline (April 17)
- [x] Run emotion eval on full 258-file diverse-speaker corpus: **36/162 = 22.2% overall recall** at strength=5.0; per-style neutral 67%, anger 30%, sad 26%, disgust 11%, fear 0%, happy 0%. Whisper has the largest emo_sim deviation (0.875 mean, 0.616 min) — emo_sim validated as a secondary signal for styles with no emotion2vec counterpart
- [x] Word error rate via Whisper: `examples/eval_wer.py` runs Whisper `base` on a directory and computes per-file WER against the same-speaker baseline (April 17). **Result: 6 of 9 styles have median WER ≤ 0.20; whisper is the only style with systemic loss (mean 0.356). Style control is orthogonal to the content channel.**
- [x] Predicted MOS via torchaudio SQUIM_SUBJECTIVE (UTMOS substitute — the `utmos` pip package conflicts with our fairseq monkey-patches). `examples/eval_mos.py`, runs on directory, outputs MOS + delta-vs-baseline per file. **Result: baseline MOS 4.05; 6 of 9 styles stay within 0.12 MOS of baseline; whisper/confused/anger degrade most. emo_sim + WER + MOS converge on the same three hardest styles — cross-metric triangulation validates the evaluation pipeline.** (April 17)
- [x] Speaker novelty metric — `examples/eval_novelty.py` computes source-vs-generated cosine similarity in native OpenVoice speaker-embedding space, with delta-vs-baseline conversion. **speaker novelty metric work result (11-speaker validation corpus):** combined-only model mean novelty gain vs baseline = `0.2599`; `cv500` CommonVoice checkpoint mean novelty gain vs baseline = `0.0369`, confirming the CommonVoice neutral-collapse result on a second axis
- [ ] `[SOON]` Speaker verification / privacy metric (secondary — Joe: "not sure we want to focus on privacy as the main thing")
- [ ] `[SOON]` Calibrate novelty thresholds from source-vs-baseline and source-vs-style distributions, so the metric can support a more explicit "novel enough" claim instead of raw cosine values alone
- [ ] `[SOON]` Cross-check the novelty metric with an independent speaker encoder / EER pipeline, not just OpenVoice's native embedding space
- [ ] `[SOON]` Run novelty-vs-noise and novelty-vs-style-strength sweeps once the speaker novelty metric is stable, to add a privacy/utility-style novelty curve
- [x] Ablation study: CREMA-D only vs. Expresso only vs. combined, extended with the validation-scale CommonVoice `cv500` init and a naive baseline. **evaluation ablation matrix result:** the combined model remains the best tradeoff across controllability, novelty, intelligibility, and naturalness (`results/eval_ablation_summary_pass4.csv`)
- [x] Compare with naive baseline: random unlabeled latent control without style supervision. **evaluation ablation matrix result:** it creates more novelty than the combined model (`0.4708` vs. `0.2599`) but with worse recall (`18.2%`) and much worse MOS delta (`-0.6453`)
- [x] Write up negative result: ControlVC D_VECTOR doesn't encode style (separability ratio 0.88) — now explicitly carried by Finding 1 and the evaluation ablation matrix paper-strengthening pass
- [x] Collapse taxonomy across ablations: content collapse (`WER >= 0.8`), style collapse to neutral, identity collapse to baseline, and mixed collapse. **evaluation ablation matrix result:** `cv500`, `cremad_only`, and `expresso_only` are dominated by identity/style collapse; the combined model has fewer but more diverse failures
- [ ] `[LATER]` Investigate F0-based re-identification attack: can F0 alone re-identify speakers after embedding anonymization?
- [ ] `[SOON]` Address collapse issue: 9% of speaker-style combinations produce unintelligible output (Joe says expected, no perfect fix needed, but worth tracking)
- [ ] `[SOON]` Add bootstrap confidence intervals or repeated-seed uncertainty to the evaluation ablation matrix ablation matrix before freezing paper tables
- [ ] `[SOON]` Add condition-by-style plots for emotion recall, novelty gain, WER, and MOS so the paper can show where each condition fails, not just overall means
- [ ] `[SOON]` Add a manual collapse-audit sheet for rows where metrics disagree (for example: high novelty but low emotional alignment, or good WER but strong neutral collapse)
- [ ] `[SOON]` Cross-check the ablation matrix with an independent speaker verifier / EER pipeline, not just OpenVoice's native embedding space

### Phase 2.5: Framing-driven tasks (from Joe's April 16 evening message)

Joe clarified that our problem is **controllable speaker generation for voice-to-voice** — a more general problem than VoicePrivacy (preserves emotion) or TTS (generates from text). The VAE enables multiple use cases; we've been showcasing only one.

- [ ] `[SOON]` **Demo use case #2: emotion change without identity change** — modify style latent dims while keeping the rest of the embedding fixed. Same-sounding person, different mood. Needs a new inference mode in `openvoice_infer_controllable.py` (or a new script) that takes a *source* speaker and only perturbs style dims instead of re-sampling the whole latent.
- [ ] `[SOON]` **Demo use case #3: fully random speaker with style control** — sample the VAE from prior (no source speaker reference) and apply style. Produces a brand-new speaker targeting a specific emotion.
- [ ] `[SOON]` **Demo use case #4: random speaker near an existing one** — sample in a neighborhood of the source speaker's latent code (small sigma) rather than from the full prior.
- [ ] `[LATER]` **Literature search for SOTA comparison** — what systems claim controllable speaker generation for voice-to-voice? Need to know what we're being compared against for Joe's "might be SOTA" claim. *Blocked: waiting for Joe's reply on April 17 message.*

### Phase 3: Reproducibility & Collaboration (for Joe)
- [x] Commit training script + inference CLI + README (done April 16, during call)
- [x] Add evaluation script (`eval_emotion.py`) with README docs — reproducible emotion2vec + emo_sim pipeline (April 17)
- [x] Add WER evaluation script (`eval_wer.py`) — Whisper + jiwer, drift-from-baseline mode by default, fixed-reference mode via `--reference-text` (April 17)
- [x] Add MOS evaluation script (`eval_mos.py`) — torchaudio SQUIM_SUBJECTIVE, baseline-as-reference mode by default, fixed-reference mode via `--reference` (April 17)
- [ ] `[NOW]` Share Expresso download instructions with Joe
- [ ] `[NOW]` Ensure Joe can run extraction + training + inference from scratch
- [ ] `[NOW]` Pin dependencies (fairseq compat, OpenVoice install steps)
- [ ] `[NOW]` Keep `FINDINGS.md` as the single async review document for Joe and point branch-heavy result dumps back to it, because the April 30 meeting confirmed that direct branch-by-branch review is slowing interpretation
- [ ] `[SOON]` Add a short reproducibility checklist for Joe (install -> dataset prep -> extraction -> train -> inference -> eval), because the rollup PR solves branch sprawl but Joe still needs one minimal checklist he can follow without reading the full branch history
- [ ] `[SOON]` Add a manifest-driven multi-metric eval helper so emotion, WER, novelty, and future privacy metrics can be rerun together on the same corpus without ad hoc command reconstruction
- [ ] `[SOON]` Add a reusable experiment runner that records checkpoint -> corpus -> metrics -> summary for CommonVoice follow-up experiments, so future finetune and objective sweeps are less manual than the finetune, objective, and rich-objective ablations
- [ ] `[SOON]` Add a checked-in OpenVoice constraints file or lockfile matching the tested `.venv` stack, so setup is copy-paste reproducible beyond the README version notes
- [ ] `[SOON]` Add an automated smoke test for `openvoice_infer_controllable.py --source-dir` + manifest generation against cached local checkpoints

### Phase 4: Application / Demo
- [ ] `[LATER]` Design interactive demo: upload voice → choose style → choose privacy level → download anonymized output
- [ ] `[LATER]` Web UI or CLI tool that non-researchers can use
- [ ] `[LATER]` Real-time voice conversion mode (stretch goal)
- [ ] `[LATER]` Package as installable tool (pip install dpvc or similar)

### Key Insight from Joe (April 9 call)
> "If I want to invent a completely hypothetical speaker who has some properties, everybody's bad at that. And in some ways, that's what we're trying to do. If we can make this work, that's a big deal."

The paper contribution is **controllable speaker profile synthesis with formal privacy guarantees** — something the speech community has struggled with. Even without the privacy angle, demonstrating controllability over speaker profiles is itself a significant result.

### 0.6 OpenVoice stabilization and reproducibility Closeout (April 28, branch `feat/openvoice-pipeline-stabilization`)

- Reframed the public docs so **OpenVoice is the active controllable pipeline** and **ControlVC is the DP baseline / negative-result path for style control**
- Added a temporary `vae_checkpoint_path` compatibility alias back to `dpvc.Anonymizer`, while keeping `vae_config` as the canonical interface for new docs and scripts
- Extended `examples/openvoice_infer_controllable.py` with:
  - `--source-dir` batch generation
  - default JSONL manifest output (`generation_manifest.jsonl` for batch runs)
  - manifest rows containing source path, output path, style, style strength, noise level, seed, checkpoint, and latent dims
- Updated `README.md`, `examples/README.md`, `results/README.md`, `docs/using.md`, `docs/training.md`, and ControlVC docs to match the real interfaces and current project framing
- Fixed packaging drift for the active OpenVoice path: `requests` is now a base dependency, `pandas` is included in the `expresso` extra

**Verification**
- `./.venv/bin/python examples/openvoice_infer_controllable.py --help` shows the new `--source-dir` and `--manifest` interface
- `./.venv/bin/python` compatibility smoke test confirmed both `Anonymizer(..., vae_checkpoint_path=...)` and `Anonymizer(..., vae_config=...)` load the same VAE class successfully
- Real OpenVoice run completed on a one-file source directory: baseline + 9 style outputs written to `/tmp/dpvc_pass1_out/`, with a 10-row manifest at `/tmp/dpvc_pass1_out/generation_manifest.jsonl`

**FINDINGS.md review**
- Reviewed after OpenVoice stabilization and reproducibility. No new paper-facing scientific finding was added because this pass stabilized interfaces and reproducibility rather than producing new experimental evidence.

### 0.7 CommonVoice pretraining pipeline Closeout (April 28, branch `feat/commonvoice-pretrain`)

- Added a local-corpus Common Voice extraction CLI: `examples/openvoice_extract_commonvoice.py`
  - expects `<corpus-path>/validated.tsv` + `clips/`
  - deterministic seed `42`
  - optional speaker and clip caps
  - recursive clip discovery, missing/unreadable-file accounting, and checkpointed extraction
- Added `scripts/prepare_commonvoice_subset.py` so downloaded Common Voice shards can be turned into a local subset without hardcoded paths
- Added reconstruction-only Common Voice pretraining via `examples/openvoice_pretrain_vae_commonvoice.py`
- Extended `examples/openvoice_train_vae_combined.py` with `--init-checkpoint` so the combined controllable VAE can finetune from Common Voice pretrained weights
- Added `scripts/compare_emotion_eval.py` to compare baseline vs candidate emotion-eval CSVs directly
- Built and validated a local English Common Voice subset at `/Users/steve/datasets/cv-corpus-21.0-2025-03-14-subset/en`
  - local clips matched: `28,116`
  - local speakers available: `6,795`
  - validation-scale training subset used for this pass: `500` speakers / `1,202` clips (`cv500`)
- Ran the full `cv500` comparison end to end:
  - extracted `embeddings/openvoice_commonvoice_cv500_emb.pt`
  - pretrained `embeddings/openvoice_vae_commonvoice_cv500.pt`
  - finetuned `embeddings/openvoice_vae_combined_cv500.pt`
  - generated `output/pass2_cv500_eval/` + manifest
  - evaluated `results/eval_emotion_pass2_cv500.csv` and `results/eval_wer_pass2_cv500.csv`

**Validation**
- [x] A small local CommonVoice subset can be extracted and pretrained without hardcoded paths
  - validated with the local `cv500` subset: `500` speakers, `1,202` embeddings, `0` missing files, `0` unreadable files
- [x] Finetuning from the pretrained checkpoint is supported by the existing combined-training path
  - validated by training `embeddings/openvoice_vae_combined_cv500.pt` from `--init-checkpoint embeddings/openvoice_vae_commonvoice_cv500.pt`
- [x] The comparison explicitly answers whether pre-training reduces collapse rate and improves recall/generalization
  - answer on the validation-scale `cv500` run: **no**
  - emotion recall dropped from `25.8%` to `16.7%` (`-9.1 pts`)
  - neutral predictions rose from `70/110` files to `109/110` files
  - WER improved from `0.235` mean to `0.084` mean across all `99` scored style rows
  - interpretation: reconstruction-only Common Voice pretraining improved intelligibility/content preservation but over-regularized the model toward baseline/neutral speech and weakened controllable style expression

**FINDINGS.md review**
- Updated after CommonVoice pretraining pipeline with a new paper-facing result: validation-scale Common Voice pretraining (`cv500`) improved WER substantially but caused near-total neutral collapse in emotion classification, so the current naive pretraining recipe is not yet a win for controllable style generalization.

### 0.8 speaker novelty metric work Closeout (April 28, branch `feat/speaker-novelty-metric`)

- Added `examples/eval_novelty.py`
  - manifest-driven corpus evaluation via `generation_manifest.jsonl`
  - one-off `--source` / `--generated` mode for spot checks
  - native OpenVoice speaker-embedding cosine similarity + cosine distance
  - same-speaker baseline conversion delta when a baseline row exists in the manifest
- Ran novelty evaluation on both validation corpora:
  - `results/eval_novelty_pass2_combined.csv`
  - `results/eval_novelty_pass2_cv500.csv`
- Updated docs so novelty is part of the reproducible evaluation stack alongside emotion, WER, and MOS

**Validation**
- [x] The novelty CLI runs on manifest-driven corpora without manual file mapping
  - validated on `output/pass2_combined_eval/generation_manifest.jsonl` and `output/pass2_cv500_eval/generation_manifest.jsonl`
- [x] The novelty CSV is reproducible and includes source, generated, style, and similarity fields needed for analysis
  - validated by writing both checked-in CSV artifacts plus a one-off smoke CSV from explicit `--source` / `--generated`
- [x] The summary clearly answers whether generated outputs differ from the source speaker, and whether that differs across styles or checkpoints
  - combined-only checkpoint: mean novelty gain vs baseline = `0.2599`
  - `cv500` CommonVoice checkpoint: mean novelty gain vs baseline = `0.0369`
  - strongest novelty on combined-only: `whisper` (`0.6561`) and `confused` (`0.5589`)
  - interpretation: the native novelty metric confirms that the `cv500` checkpoint's style control largely collapses back toward baseline voice identity

**FINDINGS.md review**
- Updated after speaker novelty metric work with a new paper-facing result: the novelty metric confirms that the combined-only model creates genuinely shifted speakers relative to the source, while the `cv500` CommonVoice checkpoint suppresses that shift and aligns with the neutral-collapse story from CommonVoice pretraining pipeline.

### 0.9 evaluation ablation matrix Closeout (April 28, branch `research/eval-ablations`)

- Added `scripts/prepare_ablation_embeddings.py`
  - prepares `cremad_only` and `expresso_only` ablation datasets directly in the unified `label_*` format expected by the OpenVoice trainer
  - uses saved Expresso row ids to realign parquet metadata with the extracted embedding file, so the ablation dataset is reproducible even when extraction skipped rows
- Added `scripts/run_ablation_inference.py`
  - generates condition-specific evaluation corpora without overloading the main public inference CLI
  - supports the single-dataset style maps and the naive free-dimension baseline
- Added `scripts/summarize_ablation_results.py`
  - reads `results/eval_*_pass4_*.csv`
  - writes `results/eval_ablation_summary_pass4.csv`
  - writes `results/eval_ablation_collapse_pass4.csv`
- Trained the two missing single-dataset ablation checkpoints:
  - `embeddings/openvoice_vae_cremad_ablation.pt`
  - `embeddings/openvoice_vae_expresso_ablation.pt`
- Generated the three new evaluation ablation matrix evaluation corpora:
  - `output/pass4_cremad_only_eval/`
  - `output/pass4_expresso_only_eval/`
  - `output/pass4_naive_noise_baseline_eval/`
- Reused the already validated CommonVoice pretraining pipeline generation corpora for the unchanged `combined` and `commonvoice_cv500_init` conditions, then copied their existing emotion/WER/novelty CSVs into the evaluation ablation matrix naming scheme and added the missing MOS layer

**evaluation ablation matrix condition matrix**
- `combined`
- `commonvoice_cv500_init`
- `cremad_only`
- `expresso_only`
- `naive_noise_baseline`

**Top-line result (`results/eval_ablation_summary_pass4.csv`)**
- `combined`: recall `25.8%`, novelty gain `0.2599`, mean WER `0.2353`, mean MOS delta `-0.0792`
- `commonvoice_cv500_init`: recall `16.7%`, novelty gain `0.0369`, mean WER `0.0844`, mean MOS delta `-0.0123`
- `cremad_only`: recall `16.7%`, novelty gain `0.0158`, mean WER `0.0534`, mean MOS delta `+0.0059`
- `expresso_only`: recall `36.4%` on `33` emotional rows only, novelty gain `-0.0008`, mean WER `0.0549`, mean MOS delta `-0.0120`
- `naive_noise_baseline`: recall `18.2%`, novelty gain `0.4708`, mean WER `0.1579`, mean MOS delta `-0.6453`

**Interpretation**
- The combined checkpoint remains the best overall tradeoff. It is the only condition with both non-trivial controllability and non-trivial identity shift while keeping WER and MOS within a survivable range.
- The `cv500` CommonVoice init, `cremad_only`, and `expresso_only` conditions all expose the same research trap from different angles: they can sound stable and transcribe well while still collapsing back toward baseline identity and/or neutral emotion.
- The naive baseline proves that **novelty alone is not the paper objective**. Random unlabeled latent pushes can create larger identity shifts than the combined model, but they do not create clean target emotion control and they degrade naturalness badly.

**Validation**
- [x] Every ablation condition has a reproducible command path and named artifacts
  - generation: `scripts/run_ablation_inference.py`
  - aggregation: `scripts/summarize_ablation_results.py`
  - artifact bundle: `results/eval_*_pass4_*.csv`
- [x] The summary table clearly shows which condition best balances controllability, novelty, intelligibility, and naturalness
  - `results/eval_ablation_summary_pass4.csv`
- [x] The naive baseline is concrete, reproducible, and fair
  - deterministic random control vectors are applied only to the six free latent dims (`9-14`) and are L2-matched to the style-control strength
- [x] The collapse summary distinguishes failure modes rather than flattening them into one bucket
  - `results/eval_ablation_collapse_pass4.csv`
- [x] The ControlVC negative-result writeup is backed by concrete evidence already present in the repo
  - Finding 1 plus the explicit evaluation ablation matrix matrix framing

**FINDINGS.md review**
- Updated after evaluation ablation matrix with a new paper-facing result: the combined model remains the best overall tradeoff in the ablation matrix, and the naive baseline shows that raw novelty is not enough without target alignment and naturalness.

### 0.10 CommonVoice finetune ablation Closeout (April 28, branch `research/commonvoice-finetune-ablation`)

- Extended `examples/openvoice_train_vae_combined.py` with coarse finetune controls:
  - `--freeze-encoder`
  - `--freeze-decoder`
  - explicit reporting of trainable vs total parameter counts
- Extended `scripts/run_ablation_inference.py` with the new CommonVoice finetune conditions:
  - `cv500_ft_short`
  - `cv500_ft_low_lr`
  - `cv500_ft_short_low_lr`
  - `cv500_ft_freeze_decoder`
  - `cv500_ft_freeze_encoder`
- Added `scripts/summarize_commonvoice_finetune_ablation.py`
  - reads `results/eval_*_pass5_*.csv`
  - writes `results/eval_commonvoice_finetune_summary_pass5.csv`
  - writes `results/eval_commonvoice_finetune_collapse_pass5.csv`
- Trained the five new finetune variants from `embeddings/openvoice_vae_commonvoice_cv500.pt` against the unchanged combined embedding set:
  - `embeddings/openvoice_vae_combined_cv500_ft_short.pt` (`1000` epochs, `1e-6`)
  - `embeddings/openvoice_vae_combined_cv500_ft_low_lr.pt` (`3000` epochs, `3e-7`)
  - `embeddings/openvoice_vae_combined_cv500_ft_short_low_lr.pt` (`1000` epochs, `3e-7`)
  - `embeddings/openvoice_vae_combined_cv500_ft_freeze_decoder.pt` (`3000` epochs, `1e-6`, decoder frozen)
  - `embeddings/openvoice_vae_combined_cv500_ft_freeze_encoder.pt` (`3000` epochs, `1e-6`, encoder frozen)
- Generated the five matched CommonVoice finetune ablation evaluation corpora:
  - `output/pass5_cv500_ft_short_eval/`
  - `output/pass5_cv500_ft_low_lr_eval/`
  - `output/pass5_cv500_ft_short_low_lr_eval/`
  - `output/pass5_cv500_ft_freeze_decoder_eval/`
  - `output/pass5_cv500_ft_freeze_encoder_eval/`
- Reused the already validated `combined` and `commonvoice_cv500_init` result CSVs from evaluation ablation matrix by copying them into the CommonVoice finetune ablation naming scheme, so the changed variable stayed strictly on finetuning policy

**CommonVoice finetune ablation condition matrix**
- `combined`
- `commonvoice_cv500_init`
- `cv500_ft_short`
- `cv500_ft_low_lr`
- `cv500_ft_short_low_lr`
- `cv500_ft_freeze_decoder`
- `cv500_ft_freeze_encoder`

**Top-line result (`results/eval_commonvoice_finetune_summary_pass5.csv`)**
- `combined`: recall `25.8%`, novelty gain `0.2599`, mean WER `0.2353`, mean MOS delta `-0.0792`
- `commonvoice_cv500_init`: recall `16.7%`, novelty gain `0.0369`, mean WER `0.0844`, mean MOS delta `-0.0123`
- `cv500_ft_short`: recall `16.7%`, novelty gain `0.0558`, mean WER `0.0390`, mean MOS delta `-0.0108`
- `cv500_ft_low_lr`: recall `16.7%`, novelty gain `0.0495`, mean WER `0.0340`, mean MOS delta `-0.0142`
- `cv500_ft_short_low_lr`: recall `16.7%`, novelty gain `0.0692`, mean WER `0.0791`, mean MOS delta `-0.0441`
- `cv500_ft_freeze_decoder`: recall `16.7%`, novelty gain `0.0192`, mean WER `0.0330`, mean MOS delta `-0.0154`
- `cv500_ft_freeze_encoder`: recall `18.2%`, novelty gain `0.0594`, mean WER `0.0905`, mean MOS delta `-0.1261`

**Interpretation**
- None of the simple finetune-policy changes restored the combined model's tradeoff. The best novelty recovery came from `cv500_ft_short_low_lr`, but it still stayed far below the combined model on both novelty (`0.0692` vs. `0.2599`) and recall (`16.7%` vs. `25.8%`).
- `cv500_ft_short_low_lr` was the strongest partial recovery recipe:
  - best novelty among the CommonVoice finetune variants
  - lowest identity-collapse count among the variants (`44`, down from `72` on the original `cv500`)
  - but no recall improvement over the original `cv500`
- `cv500_ft_freeze_encoder` was the only variant to improve recall at all (`18.2%` vs. `16.7%`), but it gave back most of the CommonVoice stability story by worsening MOS delta to `-0.1261`
- `cv500_ft_freeze_decoder` was the clearest negative result in the sweep: it reduced novelty below the original `cv500` (`0.0192`) and increased identity-collapse count to `88`
- All five variants preserved the same basic CommonVoice pretraining pipeline failure shape: very low content collapse, but dominant collapse back toward `neutral` emotion and/or baseline identity

**Validation**
- [x] New finetuning controls are reproducible from the checked-in training interface
  - validated via `examples/openvoice_train_vae_combined.py --freeze-encoder/--freeze-decoder`
  - each run logged its checkpoint source, freeze settings, and trainable parameter count
- [x] Every finetuning variant has a named checkpoint and matched evaluation corpus
  - five checkpoints under `embeddings/`
  - five 110-row corpora + manifests under `output/pass5_*`
- [x] The comparison explicitly answers whether gentler finetuning preserves CommonVoice's WER/MOS gains while recovering novelty and emotion control
  - answer: **not with these simple recipe changes**
  - novelty improves modestly for `cv500_ft_short`, `cv500_ft_low_lr`, `cv500_ft_short_low_lr`, and `cv500_ft_freeze_encoder`
  - recall remains flat at `16.7%` for four of five variants and only rises to `18.2%` once, still well below the combined model's `25.8%`
- [x] The pass isolates finetuning strategy as the changed variable rather than mixing in new datasets or new metrics
  - same CommonVoice pretrained init checkpoint
  - same combined fine-tuning embeddings
  - same 11-speaker evaluation corpus format
  - same four-metric evaluation stack

**FINDINGS.md review**
- Updated after CommonVoice finetune ablation with a new paper-facing result: simple gentler finetuning helps only marginally and does not fix the CommonVoice collapse, which narrows the next research step to better objectives or larger-scale training rather than just lighter fine-tuning.

### 0.11 CommonVoice objective ablation Closeout (April 28, branch `research/commonvoice-objective-ablation`)

- Extended `dpvc/utils.py::train_autoencoder` with explicit objective controls:
  - `recon_weight`
  - `kl_weight`
  - `label_weight`
  - optional `*_final` targets plus `schedule_epochs` for linear ramps
- Extended `examples/openvoice_train_vae_combined.py` with the matching CLI flags:
  - `--recon-weight`
  - `--kl-weight`
  - `--label-weight`
  - `--recon-weight-final`
  - `--kl-weight-final`
  - `--label-weight-final`
  - `--schedule-epochs`
- Extended `scripts/run_ablation_inference.py` with four CommonVoice objective ablation objective conditions:
  - `cv500_obj_label2`
  - `cv500_obj_label4`
  - `cv500_obj_label_ramp`
  - `cv500_obj_recon_half_label2`
- Added `scripts/summarize_commonvoice_objective_ablation.py`
  - reads `results/eval_*_pass6_*.csv`
  - writes `results/eval_commonvoice_objective_summary_pass6.csv`
  - writes `results/eval_commonvoice_objective_collapse_pass6.csv`
- Trained the four new objective variants from the unchanged CommonVoice init checkpoint `embeddings/openvoice_vae_commonvoice_cv500.pt`:
  - `embeddings/openvoice_vae_combined_cv500_obj_label2.pt`
  - `embeddings/openvoice_vae_combined_cv500_obj_label4.pt`
  - `embeddings/openvoice_vae_combined_cv500_obj_label_ramp.pt`
  - `embeddings/openvoice_vae_combined_cv500_obj_recon_half_label2.pt`
- Generated the four matched CommonVoice objective ablation evaluation corpora:
  - `output/pass6_cv500_obj_label2_eval/`
  - `output/pass6_cv500_obj_label4_eval/`
  - `output/pass6_cv500_obj_label_ramp_eval/`
  - `output/pass6_cv500_obj_recon_half_label2_eval/`
- Reused the already validated `combined`, `commonvoice_cv500_init`, and `cv500_ft_short_low_lr` CSVs by copying them into the CommonVoice objective ablation naming scheme, so the changed variable stayed strictly on objective design

**CommonVoice objective ablation condition matrix**
- `combined`
- `commonvoice_cv500_init`
- `cv500_ft_short_low_lr`
- `cv500_obj_label2`
- `cv500_obj_label4`
- `cv500_obj_label_ramp`
- `cv500_obj_recon_half_label2`

**Top-line result (`results/eval_commonvoice_objective_summary_pass6.csv`)**
- `combined`: recall `25.8%`, novelty gain `0.2599`, mean WER `0.2353`, mean MOS delta `-0.0792`
- `commonvoice_cv500_init`: recall `16.7%`, novelty gain `0.0369`, mean WER `0.0844`, mean MOS delta `-0.0123`
- `cv500_ft_short_low_lr`: recall `16.7%`, novelty gain `0.0692`, mean WER `0.0791`, mean MOS delta `-0.0441`
- `cv500_obj_label2`: recall `16.7%`, novelty gain `0.0589`, mean WER `0.0862`, mean MOS delta `-0.0300`
- `cv500_obj_label4`: recall `16.7%`, novelty gain `0.0496`, mean WER `0.1222`, mean MOS delta `-0.0228`
- `cv500_obj_label_ramp`: recall `16.7%`, novelty gain `0.0442`, mean WER `0.0832`, mean MOS delta `-0.0126`
- `cv500_obj_recon_half_label2`: recall `16.7%`, novelty gain `0.0500`, mean WER `0.1146`, mean MOS delta `-0.0226`

**Interpretation**
- None of the simple objective variants beat the best CommonVoice finetune ablation recipe (`cv500_ft_short_low_lr`) on novelty, recall, or collapse counts.
- `cv500_obj_label2` was the strongest of the new objective variants, but it still trailed `cv500_ft_short_low_lr` on novelty (`0.0589` vs. `0.0692`) and identity collapse (`53` vs. `44`).
- `cv500_obj_label_ramp` preserved MOS closest to the raw `cv500` init, but it did so by staying close to the same conservative failure shape rather than recovering controllability.
- The CommonVoice bottleneck now looks deeper than both simple finetune-policy changes (CommonVoice finetune ablation) and simple scalar objective reweighting (CommonVoice objective ablation).

**Validation**
- [x] New objective controls are reproducible from the checked-in training interface
  - validated via `examples/openvoice_train_vae_combined.py --recon-weight/--label-weight/...`
  - smoke-tested with short scheduled-weight runs and then used for the four full CommonVoice objective ablation checkpoints
- [x] Every objective variant has a named checkpoint and matched evaluation corpus
  - four checkpoints under `embeddings/`
  - four 110-row corpora + manifests under `output/pass6_*`
- [x] The comparison explicitly answers whether objective changes recover controllability and novelty without surrendering the CommonVoice WER/MOS gains
  - answer: **not with simple scalar reweighting or label ramps**
  - all four objective variants stay flat at `16.7%` recall and none beats `cv500_ft_short_low_lr` on novelty
- [x] The pass isolates objective design as the changed variable rather than mixing in new datasets or new metrics
  - same CommonVoice init checkpoint
  - same combined fine-tuning embeddings
  - same 11-speaker evaluation corpus format
  - same four-metric evaluation stack

**FINDINGS.md review**
- Updated after CommonVoice objective ablation with a new paper-facing result: simple loss reweighting and label-weight schedules do not fix the CommonVoice collapse either, which narrows the next research step further toward richer objectives or larger-scale supervision rather than more scalar tuning.

### 0.12 CommonVoice rich-objective ablation Closeout (April 29, branch `research/commonvoice-rich-objectives`)

- Extended `dpvc/model_embedding_vae.py` so the VAE caches `last_mu` and `last_logvar` during `forward`, which lets training-time auxiliary losses supervise the latent geometry directly instead of only the decoded embedding or sampled latent
- Extended `dpvc/utils.py::train_autoencoder` with richer CommonVoice adaptation controls:
  - frozen style-teacher loss on style dims (`0-8`)
  - frozen free-anchor loss on non-style dims (`9-14`)
  - scheduled weights for both auxiliary terms
- Extended `examples/openvoice_train_vae_combined.py` with the matching CLI flags:
  - `--style-teacher-checkpoint`
  - `--style-teacher-weight`
  - `--style-teacher-weight-final`
  - `--free-anchor-checkpoint`
  - `--free-anchor-weight`
  - `--free-anchor-weight-final`
- Extended `scripts/run_ablation_inference.py` with three CommonVoice rich-objective conditions:
  - `cv500_rich_teacher_style`
  - `cv500_rich_free_anchor`
  - `cv500_rich_teacher_plus_anchor`
- Added `scripts/summarize_commonvoice_rich_objectives.py`
  - reads `results/eval_*_pass7_*.csv`
  - writes `results/eval_commonvoice_rich_objectives_summary_pass7.csv`
  - writes `results/eval_commonvoice_rich_objectives_collapse_pass7.csv`
- Trained the three new rich-objective variants from the unchanged CommonVoice init checkpoint `embeddings/openvoice_vae_commonvoice_cv500.pt`:
  - `embeddings/openvoice_vae_combined_cv500_rich_teacher_style.pt`
  - `embeddings/openvoice_vae_combined_cv500_rich_free_anchor.pt`
  - `embeddings/openvoice_vae_combined_cv500_rich_teacher_plus_anchor.pt`
- Generated the three matched CommonVoice rich-objective ablation evaluation corpora:
  - `output/pass7_cv500_rich_teacher_style_eval/`
  - `output/pass7_cv500_rich_free_anchor_eval/`
  - `output/pass7_cv500_rich_teacher_plus_anchor_eval/`
- Reused the already validated `combined`, `commonvoice_cv500_init`, and `cv500_ft_short_low_lr` CSVs by copying them into the CommonVoice rich-objective ablation naming scheme, so the changed variable stayed strictly on rich supervision during CommonVoice adaptation

**CommonVoice rich-objective ablation condition matrix**
- `combined`
- `commonvoice_cv500_init`
- `cv500_ft_short_low_lr`
- `cv500_rich_teacher_style`
- `cv500_rich_free_anchor`
- `cv500_rich_teacher_plus_anchor`

**Top-line result (`results/eval_commonvoice_rich_objectives_summary_pass7.csv`)**
- `combined`: recall `25.8%`, novelty gain `0.2599`, mean WER `0.2353`, mean MOS delta `-0.0792`
- `commonvoice_cv500_init`: recall `16.7%`, novelty gain `0.0369`, mean WER `0.0844`, mean MOS delta `-0.0123`
- `cv500_ft_short_low_lr`: recall `16.7%`, novelty gain `0.0692`, mean WER `0.0791`, mean MOS delta `-0.0441`
- `cv500_rich_teacher_style`: recall `16.7%`, novelty gain `0.0574`, mean WER `0.0851`, mean MOS delta `-0.0200`
- `cv500_rich_free_anchor`: recall `16.7%`, novelty gain `0.0646`, mean WER `0.0724`, mean MOS delta `-0.0079`
- `cv500_rich_teacher_plus_anchor`: recall `16.7%`, novelty gain `0.0566`, mean WER `0.0809`, mean MOS delta `-0.0161`

**Interpretation**
- None of the three rich-objective variants improved recall beyond `16.7%`.
- None beat the best CommonVoice finetune ablation recipe (`cv500_ft_short_low_lr`) on novelty or identity collapse.
- `cv500_rich_free_anchor` was the strongest of the new variants: it improved WER and MOS beyond `cv500_ft_short_low_lr`, but novelty still trailed slightly (`0.0646` vs. `0.0692`) and identity collapse stayed worse (`50` vs. `44`).
- The style-teacher and teacher-plus-anchor variants also stayed in the same conservative failure shape: recall flat, style collapse near `54-55`, and identity collapse still well above the combined model.
- The CommonVoice bottleneck now looks deeper than simple finetune-policy changes (CommonVoice finetune ablation), scalar loss reweighting (CommonVoice objective ablation), and the first richer-teacher/anchor objective family (CommonVoice rich-objective ablation).

**Validation**
- [x] Rich-objective controls are reproducible from the checked-in training interface
  - validated via `examples/openvoice_train_vae_combined.py --style-teacher-* / --free-anchor-*`
  - smoke-tested with short runs and then used for the three full CommonVoice rich-objective ablation checkpoints
- [x] Every rich-objective variant has a named checkpoint and matched evaluation corpus
  - three checkpoints under `embeddings/`
  - three 110-row corpora + manifests under `output/pass7_*`
- [x] The comparison explicitly answers whether richer supervision beats raw `cv500` and the best CommonVoice finetune ablation recipe
  - answer: **not with these teacher/anchor losses**
  - all three variants stay flat at `16.7%` recall and none beats `cv500_ft_short_low_lr` on novelty or identity collapse
- [x] The pass isolates supervision/objective design as the changed variable
  - same CommonVoice init checkpoint
  - same combined fine-tuning embeddings
  - same 11-speaker evaluation corpus format
  - same four-metric evaluation stack
- [x] The pass yields a clear next-step decision
  - the next CommonVoice work should move either to richer pretraining supervision on CommonVoice itself, partial-label / pseudo-label objectives, or a larger-scale follow-up once a stronger objective survives on this validation-scale setup

**FINDINGS.md review**
- Updated after CommonVoice rich-objective ablation with a new paper-facing result: richer teacher/anchor supervision during combined finetuning still does not recover recall after CommonVoice pretraining, which suggests the remaining bottleneck is deeper than finetune-time scalar tuning or the first richer-objective family tried here.

### 0.13 CommonVoice partial-label pretraining Closeout (April 29, branch `research/commonvoice-partial-label-pretrain`)

- Extended `examples/openvoice_extract_commonvoice.py` so saved CommonVoice embedding artifacts now include a `metadata_report` summarizing field coverage and categorical distributions
- Extended `dpvc/utils.py` with CommonVoice weak-supervision helpers:
  - metadata coverage summarization utilities
  - `train_commonvoice_pretrain(...)` for reconstruction + masked metadata + pseudo-style losses
  - inverse-frequency balancing for metadata classes and pseudo-style rows
- Extended `examples/openvoice_pretrain_vae_commonvoice.py` with weak-label CLI support:
  - `--metadata-targets`
  - `--metadata-weight`
  - `--metadata-min-count`
  - `--pseudo-style-weight`
  - `--pseudo-style-threshold`
  - `--style-dims`
  - `--free-dims`
- Added `scripts/annotate_commonvoice_pseudolabels.py`
  - reads a CommonVoice embedding artifact
  - adds `pseudo_style`, `pseudo_style_confidence`, `pseudo_style_raw_label`, `pseudo_style_report`
  - keeps all pseudo labels in the artifact and uses thresholds only at training/report time so later threshold sweeps do not require relabeling
- Extended `scripts/run_ablation_inference.py` with three CommonVoice partial-label pretraining weak-supervision conditions:
  - `cv500_pl_meta`
  - `cv500_pl_pseudo_style`
  - `cv500_pl_meta_plus_pseudo`
- Added `scripts/summarize_commonvoice_partial_label.py`
  - reads `results/eval_*_pass8_*.csv`
  - writes `results/eval_commonvoice_partial_label_summary_pass8.csv`
  - writes `results/eval_commonvoice_partial_label_collapse_pass8.csv`
- Recorded CommonVoice metadata sparsity on the `cv500` subset artifact:
  - age known on `193/1202` clips
  - gender known on `184/1202`
  - accent known on `190/1202`
  - only two age buckets (`young`, `adult`) had enough support for the metadata-weighted run at the default minimum-count filter
- Generated a pseudo-labeled CommonVoice artifact:
  - `embeddings/openvoice_commonvoice_cv500_pseudo.pt`
  - accepted pseudo-style counts at threshold `0.60`: `neutral=640`, `sad=336`, `happy=51`, `disgust=46`, `anger=11`, `fear=5`
  - this class imbalance is why CommonVoice partial-label pretraining used inverse-frequency row weighting for pseudo-style supervision
- Trained the three new CommonVoice pretraining variants:
  - `embeddings/openvoice_vae_commonvoice_cv500_pl_meta.pt`
  - `embeddings/openvoice_vae_commonvoice_cv500_pl_pseudo_style.pt`
  - `embeddings/openvoice_vae_commonvoice_cv500_pl_meta_plus_pseudo.pt`
- Finetuned each one on the unchanged combined labeled embeddings:
  - `embeddings/openvoice_vae_combined_cv500_pl_meta.pt`
  - `embeddings/openvoice_vae_combined_cv500_pl_pseudo_style.pt`
  - `embeddings/openvoice_vae_combined_cv500_pl_meta_plus_pseudo.pt`
- Generated the three matched CommonVoice partial-label pretraining evaluation corpora:
  - `output/pass8_cv500_pl_meta_eval/`
  - `output/pass8_cv500_pl_pseudo_style_eval/`
  - `output/pass8_cv500_pl_meta_plus_pseudo_eval/`
- Reused the already validated `combined`, `commonvoice_cv500_init`, `cv500_ft_short_low_lr`, and `cv500_rich_free_anchor` CSVs by copying them into the CommonVoice partial-label pretraining naming scheme, so the changed variable stayed strictly on weak supervision during CommonVoice pretraining

**CommonVoice partial-label pretraining condition matrix**
- `combined`
- `commonvoice_cv500_init`
- `cv500_ft_short_low_lr`
- `cv500_rich_free_anchor`
- `cv500_pl_meta`
- `cv500_pl_pseudo_style`
- `cv500_pl_meta_plus_pseudo`

**Top-line result (`results/eval_commonvoice_partial_label_summary_pass8.csv`)**
- `combined`: recall `25.8%`, novelty gain `0.2599`, mean WER `0.2353`, mean MOS delta `-0.0792`
- `commonvoice_cv500_init`: recall `16.7%`, novelty gain `0.0369`, mean WER `0.0844`, mean MOS delta `-0.0123`
- `cv500_ft_short_low_lr`: recall `16.7%`, novelty gain `0.0692`, mean WER `0.0791`, mean MOS delta `-0.0441`
- `cv500_rich_free_anchor`: recall `16.7%`, novelty gain `0.0646`, mean WER `0.0724`, mean MOS delta `-0.0079`
- `cv500_pl_meta`: recall `16.7%`, novelty gain `0.0570`, mean WER `0.0918`, mean MOS delta `-0.0505`
- `cv500_pl_pseudo_style`: recall `16.7%`, novelty gain `0.0190`, mean WER `0.0263`, mean MOS delta `-0.0229`
- `cv500_pl_meta_plus_pseudo`: recall `16.7%`, novelty gain `0.0181`, mean WER `0.0285`, mean MOS delta `-0.0148`

**Interpretation**
- None of the three weak-label variants improved recall beyond `16.7%`.
- `cv500_pl_meta` was the best novelty result of the new variants, but it still trailed both `cv500_ft_short_low_lr` and `cv500_rich_free_anchor`, and it did so with worse WER and MOS than those stronger earlier CommonVoice baselines.
- `cv500_pl_pseudo_style` and `cv500_pl_meta_plus_pseudo` dramatically improved WER, but only by collapsing much harder toward baseline speaker identity: identity-collapse rows jumped to `85` and `90`.
- The pseudo-label path therefore looks conservative rather than expressive on this validation-scale setup: it preserves content and naturalness, but not controllability or speaker shift.
- The CommonVoice bottleneck now looks deeper than finetune-policy tuning (CommonVoice finetune ablation), scalar weight schedules (CommonVoice objective ablation), finetune-time teacher/anchor supervision (CommonVoice rich-objective ablation), and this first weak-label pretraining family (CommonVoice partial-label pretraining).

**Validation**
- [x] CommonVoice metadata coverage is measured and reported from the checked-in extraction interface
  - validated through the `metadata_report` saved by `examples/openvoice_extract_commonvoice.py`
  - confirmed sparse coverage on the checked-in `cv500` subset artifact, which now informs how we interpret the metadata-only run
- [x] Partial-label CommonVoice pretraining is reproducible from the checked-in training interface
  - validated through `examples/openvoice_pretrain_vae_commonvoice.py --metadata-* / --pseudo-style-*`
  - smoke-tested with short runs and then used for the three full CommonVoice partial-label pretraining pretraining checkpoints
- [x] Every CommonVoice partial-label pretraining condition has a named pretrain checkpoint, finetuned checkpoint, and matched evaluation corpus
  - three CommonVoice pretrain checkpoints under `embeddings/`
  - three combined finetune checkpoints under `embeddings/`
  - three 110-row corpora + manifests under `output/pass8_*`
- [x] The comparison explicitly answers whether weak supervision during CommonVoice pretraining beats raw `cv500`, the best CommonVoice finetune ablation recipe, and the best CommonVoice rich-objective ablation recipe
  - answer: **not on this validation-scale setup**
  - metadata-only supervision improves novelty over raw `cv500`, but not enough to beat the best earlier CommonVoice variants
  - pseudo-style supervision improves WER sharply, but collapses identity shift even harder
- [x] The pass isolates pretraining supervision as the changed variable
  - same `cv500` CommonVoice subset scale
  - same combined finetune data
  - same 11-speaker evaluation corpus format
  - same four-metric evaluation stack
- [x] The pass yields a clear next-step decision
  - the next CommonVoice work should focus on pseudo-label quality, prototype/teacher-space targets during CommonVoice pretraining, or stronger curricula rather than simply adding the current weak labels at this scale

**FINDINGS.md review**
- Updated after CommonVoice partial-label pretraining with a new paper-facing result: validation-scale weak supervision during CommonVoice pretraining still does not recover recall, and the pseudo-style variants appear to trade controllability/novelty for stronger intelligibility rather than solving the underlying collapse.

---

### 0.14 April 30 Meeting with Joe — Direction Update

- Joe confirmed that the **biggest untried experiment** is still the first real sampled mixed-data run that combines CommonVoice, CREMA-D, and Expresso in one training setup
- Joe explicitly endorsed **pseudo-labeled CommonVoice + mixed-data training** as the most promising current path, more than another isolated CommonVoice-only refinement
- Joe warned that a naive combined run may just behave like CommonVoice unless we manage the mixture carefully; he called out the training schedule itself as an open empirical question
- Joe's current sampling intuition is that **speaker breadth may matter more than raw clip count** on CommonVoice, so the next mixed-data corpus should prioritize at least one clip per speaker before adding more clips
- Joe agrees that architecture may matter, but he put **data mixing and supervision quality ahead of architecture changes** for the immediate next step
- Joe qualitatively found that style strengths **above `5.0` can still work well**, especially for whisper on non-Trump examples, so our docs should not treat `5.0` as a hard ceiling
- Joe said the branch stack is getting hard to interpret directly and asked for a **single document** that summarizes findings and how to read the metrics, which reinforces keeping `FINDINGS.md` as the async review hub
- Joe confirmed the current **12GB GPU machine is sufficient for these VAE experiments**, while larger cluster access may be possible later but should not be assumed for the next branch
- **Completed branch from Joe's April 30 recommendation:** `research/combined-data-pseudolabel-mix`
- **Completed follow-up branch:** `research/mixed-data-pseudolabel-quality`
- **Completed follow-up branch:** `research/nontrump-style-strength-sweep`
- **Current canonical research branch:** `research/controllable-vae`
- **Current experiment focus:** mixed-data pseudo-label teacher / acceptance logic

### 0.15 Mixed-Data Pseudolabel Mix Bootstrap Implementation (April 30, branch `research/combined-data-pseudolabel-mix`)

- Added `scripts/build_mixed_training_set.py`
  - builds the first sampled **CommonVoice + CREMA-D + Expresso** training artifact
  - preserves `source_dataset`, `style_label_mask`, `style_label_row_weight`, pseudo-label confidence, and a saved `mixture_report`
  - prioritizes CommonVoice **speaker breadth first**, with optional pseudo-label preference and per-style caps
- Added `examples/openvoice_train_vae_mixed.py`
  - trains directly from the mixed artifact
  - supports the first three schedule families Joe called out:
    - `static_balanced`
    - `cv_warmup`
    - `labeled_finish`
- Added `dpvc.utils.train_mixed_autoencoder()`
  - samples rows each epoch according to dataset-mass schedules rather than treating the merged artifact as a flat dataset
  - keeps the labeled-row loss masked so unlabeled CommonVoice rows do not act like fake negatives
- Added `scripts/summarize_mixed_data_results.py`
  - added the checked-in summary format used by the mixed-data schedule matrix and collapse taxonomy

**Smoke validation**
- [x] Mixed-data artifact builder compiles and runs
  - validated with a small smoke artifact at `/private/tmp/openvoice_mixed_smoke.pt`
  - smoke artifact summary: `627` total rows = `36` CommonVoice + `546` CREMA-D + `45` Expresso
  - labeled rows: `622/627`
- [x] First real mixed-data artifact built for the branch
  - wrote `embeddings/openvoice_mixed_base.pt`
  - current branch artifact summary: `1,325` total rows = `500` CommonVoice + `546` CREMA-D + `279` Expresso
  - labeled rows: `1,287/1,325`
  - current CommonVoice pseudo-label mix in the artifact: `neutral=207`, `sad=170`, `happy=37`, `disgust=34`, `anger=10`, `fear=4`
- [x] CommonVoice speaker-first sampling, pseudo-label filtering, and mixture reporting execute end to end
  - validated with `20` CommonVoice speakers, `max 2` clips per speaker, pseudo-label preference on, and explicit style caps
- [x] All three mixed-data schedules execute end to end on the smoke artifact
  - `static_balanced`
  - `cv_warmup`
  - `labeled_finish`
- [x] The shared evaluation-corpus generator now accepts the first mixed-data conditions
  - smoke-validated with `mixed_static_balanced` plus a one-file source run that wrote 10 outputs and a manifest to `/private/tmp/pass9_mixed_infer_smoke/`
- [x] New scripts pass syntax validation
  - `py_compile` passed for `dpvc/utils.py`, `examples/openvoice_train_vae_mixed.py`, `scripts/build_mixed_training_set.py`, `scripts/run_ablation_inference.py`, and `scripts/summarize_mixed_data_results.py`

**Status**
- This bootstrap implementation fed directly into the mixed-data result closeout below.

### 0.16 Mixed-Data Pseudolabel Mix Results Closeout (April 30, branch `research/combined-data-pseudolabel-mix`)

- Trained the three first real mixed-data checkpoints from `embeddings/openvoice_mixed_base.pt`:
  - `embeddings/openvoice_vae_mixed_static_balanced.pt`
  - `embeddings/openvoice_vae_mixed_cv_warmup.pt`
  - `embeddings/openvoice_vae_mixed_labeled_finish.pt`
- Generated the three matched evaluation corpora:
  - `output/pass9_mixed_static_balanced_eval/`
  - `output/pass9_mixed_cv_warmup_eval/`
  - `output/pass9_mixed_labeled_finish_eval/`
- Ran the full metric stack for all three:
  - `eval_emotion.py`
  - `eval_novelty.py`
  - `eval_wer.py`
  - `eval_mos.py`
- Copied the already validated reference CSVs for:
  - `combined`
  - `commonvoice_cv500_init`
  - `cv500_ft_short_low_lr`
  - `cv500_rich_free_anchor`
  - `cv500_pl_meta`
  into the mixed-data summary naming scheme so the schedule matrix could report deltas against the strongest earlier baselines.
- Wrote:
  - `results/eval_mixed_data_summary_pass9.csv`
  - `results/eval_mixed_data_collapse_pass9.csv`

**Mixed-data condition matrix**

| Condition | Recall | Novelty gain | Mean WER | Mean MOS delta | Identity collapse | Takeaway |
|-----------|--------|--------------|----------|----------------|-------------------|----------|
| `mixed_static_balanced` | `16.7%` | `0.0865` | `0.0825` | `-0.1316` | `58` | Best novelty of the mixed schedules, but no recall gain |
| `mixed_cv_warmup` | `16.7%` | `0.0738` | `0.0700` | `-0.1341` | `60` | Better WER than static, weaker novelty, still no recall gain |
| `mixed_labeled_finish` | `16.7%` | `0.0828` | `0.0606` | `-0.1425` | `64` | Best WER of the mixed schedules, but highest identity collapse |

**Interpretation**
- The first real mixed-data run did **not** recover recall above the CommonVoice line: all three schedules stayed fixed at `16.7%`.
- Schedule choice changed WER and novelty modestly, but not the core controllability outcome.
- `mixed_labeled_finish` gave the strongest intelligibility result (`0.0606` mean WER), while `mixed_static_balanced` gave the strongest novelty result (`0.0865`).
- All three mixed-data schedules beat the raw `cv500` model on novelty and at least matched or beat the stronger CommonVoice-only references on WER, but all remained far below the `combined` model on controllability and speaker shift.
- The schedule question is now narrower: better pseudo-label quality, per-class filtering/caps, or stronger labeled-data protection are higher-value next steps than more simple schedule sweeps.

**Validation**
- [x] The first mixed-data training artifact is reproducible and documents dataset composition, speaker counts, and pseudo-label coverage
- [x] The branch compares at least three explicit mixture strategies, not just one naive combined run
- [x] The three real mixed-data checkpoints train successfully from `embeddings/openvoice_mixed_base.pt`
- [x] Each mixed-data checkpoint has a matched evaluation corpus and complete metric bundle (`emotion`, `novelty`, `WER`, `MOS`)
- [x] The CommonVoice sampling policy is speaker-breadth aware and documented
- [x] The mixed-data result explicitly answers whether combining CommonVoice with CREMA-D and Expresso improves controllability beyond the CommonVoice-only line
- [x] The branch preserves comparison against the established references (`combined`, `cv500`, best CommonVoice finetune ablation / rich-objective / partial-label references)
- [ ] A non-Trump style-strength sweep above `5.0` is still pending on the best mixed-data checkpoint

**Finding update**
- Updated after the mixed-data pseudolabel mix result with a new paper-facing result: the first real CommonVoice + CREMA-D + Expresso run improves WER more than it improves control. None of the three schedule variants recovers recall above `16.7%`, although `mixed_labeled_finish` yields the best WER and `mixed_static_balanced` yields the best novelty among the mixed schedules.

### 0.17 Mixed-Data Pseudolabel Quality Closeout (May 3, branch `research/mixed-data-pseudolabel-quality`)

What we changed:
- Added per-class CommonVoice pseudo-label thresholds to `scripts/build_mixed_training_set.py`
- Added threshold-rejected, cap-skipped, and final accepted pseudo-label counts to `mixture_report`
- Fixed fallback selection so class caps stay enforced even when preserving speaker breadth
- Added explicit dataset-mass controls to `examples/openvoice_train_vae_mixed.py` and `dpvc.utils.train_mixed_autoencoder()`
- Added three new mixed-data conditions to `scripts/run_ablation_inference.py`:
  - `mixed_quality_static_balanced`
  - `mixed_quality_labeled_finish`
  - `mixed_quality_labeled_guarded`
- Generalized `scripts/summarize_mixed_data_results.py` so it can summarize arbitrary tagged mixed-data result families

Improved artifact recipe:
- `embeddings/openvoice_mixed_quality_base.pt`
- `500` CommonVoice speakers, one clip per speaker
- CommonVoice pseudo-label thresholds:
  - `neutral=0.995`
  - `sad=0.98`
  - `happy=0.92`
  - `disgust=0.92`
  - `anger=0.90`
  - `fear=0.90`
- CommonVoice pseudo-style caps:
  - `neutral=120`
  - `sad=110`
  - `happy=60`
  - `disgust=50`
  - `anger=20`
  - `fear=20`
- Row weighting:
  - `pseudo_row_weight=0.75`
  - `true_row_weight=1.25`

Improved artifact composition:
- total rows: `1325`
- `CommonVoice=500`, `CREMA-D=546`, `Expresso=279`
- labeled CommonVoice rows: `300`
- unlabeled CommonVoice fallback rows: `200`
- final selected CommonVoice pseudo-style counts:
  - `neutral=120`
  - `sad=110`
  - `happy=32`
  - `disgust=29`
  - `anger=5`
  - `fear=4`

Validation:
- `Validation`: The mixed-data quality branch preserves a reproducible artifact-level `mixture_report` with per-style pseudo-label acceptance and rejection counts
- `Validation`: The branch makes labeled-data protection explicit in the training interface rather than leaving it hard-coded
- `Validation`: Every new condition has a named checkpoint and matched evaluation corpus
- `Validation`: The comparison explicitly answers whether better pseudo-label filtering plus stronger labeled-data protection improve recall beyond the original mixed-data line

New checkpoints:
- `embeddings/openvoice_vae_mixed_quality_static_balanced.pt`
- `embeddings/openvoice_vae_mixed_quality_labeled_finish.pt`
- `embeddings/openvoice_vae_mixed_quality_labeled_guarded.pt`

New corpora:
- `output/mixed_quality_static_balanced_eval/`
- `output/mixed_quality_labeled_finish_eval/`
- `output/mixed_quality_labeled_guarded_eval/`

Result artifacts:
- `results/eval_mixed_quality_summary.csv`
- `results/eval_mixed_quality_collapse.csv`

Top-line comparison:

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Identity collapse | Takeaway |
|-----------|--------|--------------------------|----------|----------------|-------------------|----------|
| `mixed_quality_static_balanced` | `16.7%` | `0.0770` | `0.0911` | `-0.1130` | `64` | Better reporting and cleaner class balance, but no control gain |
| `mixed_quality_labeled_finish` | `16.7%` | `0.0763` | `0.0649` | `-0.1232` | `65` | Keeps most of the mixed-data WER story, but no recall gain |
| `mixed_quality_labeled_guarded` | `18.2%` | `0.0764` | `0.0978` | `-0.1234` | `69` | First mixed-data recall bump, but not a clean overall win |

Interpretation:
- Better pseudo-label filtering plus stronger labeled-data protection produced a **small but real recall gain** in `mixed_quality_labeled_guarded`
- That gain is narrow and costly: the best new condition still loses to `mixed_labeled_finish` on WER and to `mixed_static_balanced` on novelty
- The branch therefore does **not** overturn Finding 17; it narrows it by showing that better mixed-data supervision can move recall a little, but not yet enough to beat the `combined` model or produce a clean mixed-data win
- `mixed_quality_labeled_guarded` is still the right checkpoint to carry into the non-Trump style-strength sweep, because it is the most control-capable mixed-data condition so far

### 0.18 Non-Trump Style-Strength Sweep Closeout (May 3, branch `research/nontrump-style-strength-sweep`)

What we changed:
- Added a fixed non-Trump panel at `examples/nontrump_strength_panel/`
- Added `IMPLEMENTATION_PLAN_nontrump-style-strength-sweep.md`
- Added `scripts/summarize_nontrump_strength_sweep.py`
- Generated four full evaluation corpora from `embeddings/openvoice_vae_mixed_quality_labeled_guarded.pt` at strengths:
  - `5.0`
  - `7.5`
  - `10.0`
  - `12.5`

Fixed panel:
- `examples/nontrump_strength_panel/female_1_cremad_1002.wav`
- `examples/nontrump_strength_panel/female_2_cremad_1012.wav`
- `examples/nontrump_strength_panel/male_1_cremad_1003.wav`
- `examples/nontrump_strength_panel/male_2_cremad_1051.wav`

Validation:
- `Validation`: The sweep uses a fixed non-Trump source panel checked into the repo
- `Validation`: The results compare `5.0`, `7.5`, `10.0`, and `12.5` on the same checkpoint and panel
- `Validation`: Any updated strength guidance is backed by checked-in metrics and an explicit panel definition

Result artifacts:
- `results/eval_nontrump_strength_5p0_emotion.csv`
- `results/eval_nontrump_strength_5p0_novelty.csv`
- `results/eval_nontrump_strength_5p0_wer.csv`
- `results/eval_nontrump_strength_5p0_mos.csv`
- same four metric CSVs for `7p5`, `10p0`, and `12p5`
- `results/eval_nontrump_strength_sweep.csv`
- `results/eval_nontrump_strength_sweep_summary.md`

Top-line overall sweep:

| Strength | Recall | Mean emo_sim | Mean novelty gain | Mean WER | Mean MOS delta | Takeaway |
|----------|--------|--------------|-------------------|----------|----------------|----------|
| `5.0` | `20.8%` | `0.9874` | `0.0789` | `0.1472` | `-0.1312` | Safest overall setting on this panel |
| `7.5` | `16.7%` | `0.9773` | `0.1246` | `0.1681` | `-0.1701` | Best stronger-than-default compromise |
| `10.0` | `16.7%` | `0.9726` | `0.1570` | `0.2028` | `-0.2287` | Higher novelty, noticeably worse overall quality |
| `12.5` | `16.7%` | `0.9728` | `0.1761` | `0.2444` | `-0.2119` | Highest novelty, but clearly no longer a balanced default |

Focus-style takeaways:
- `whisper` novelty gain rises steadily:
  - `0.3651` at `5.0`
  - `0.5123` at `7.5`
  - `0.6042` at `10.0`
  - `0.6521` at `12.5`
- `whisper` WER stays flat at `0.3000`, while MOS delta worsens from `-0.6416` at `5.0` to `-0.7248` at `7.5`, `-0.7918` at `10.0`, then partially recovers to `-0.7141` at `12.5`
- `confused` also becomes much more novel:
  - `0.2871` at `5.0`
  - `0.4549` at `7.5`
  - `0.5341` at `10.0`
  - `0.5694` at `12.5`
- But `confused` pays a steep naturalness cost: MOS delta drops from `-0.5253` at `5.0` to `-1.0095` at `12.5`
- `sad` is the opposite pattern: `5.0` is the only setting with non-zero recall (`25%`), while higher strengths increase novelty slightly but do not recover emotional classification
- `happy` does not benefit from higher strength on this panel: novelty gain stays negative and WER worsens above `5.0`

Interpretation:
- Joe was right that `5.0` is **not** a hard ceiling
- Higher strengths are useful for **specific styles**, especially `whisper` and `confused`, when novelty is more important than overall quality
- `7.5` is the best stronger-than-default compromise on this panel
- `10.0-12.5` are better treated as style-specific or demo-specific settings, not as new global defaults

---

### 0.19 Mixed-Data Pseudo-Label Teacher Closeout (May 4, branch `research/controllable-vae`)

What we changed:
- Scored the first three teacher-family checkpoints with full matched corpora:
  - `mixed_teacher_threshold_balanced`
  - `mixed_teacher_labeled_finish`
  - `mixed_teacher_labeled_guarded`
- Extended `scripts/run_ablation_inference.py` with the new condition aliases
- Added the checked-in result bundle:
  - `results/eval_emotion_mixed_teacher_mixed_teacher_threshold_balanced.csv`
  - `results/eval_novelty_mixed_teacher_mixed_teacher_threshold_balanced.csv`
  - `results/eval_wer_mixed_teacher_mixed_teacher_threshold_balanced.csv`
  - `results/eval_mos_mixed_teacher_mixed_teacher_threshold_balanced.csv`
  - same four metric CSVs for `mixed_teacher_labeled_finish` and `mixed_teacher_labeled_guarded`
  - `results/eval_mixed_teacher_summary.csv`
  - `results/eval_mixed_teacher_collapse.csv`

Validation:
- `Validation`: Every teacher-family condition has a named checkpoint, matched evaluation corpus, and result bundle
- `Validation`: The comparison explicitly answers whether a cleaner single-teacher path beats the current mixed-data quality baseline
- `Validation`: The branch isolates pseudo-label teacher / acceptance changes rather than architecture changes

New corpora:
- `output/mixed_teacher_threshold_balanced_eval/`
- `output/mixed_teacher_labeled_finish_eval/`
- `output/mixed_teacher_labeled_guarded_eval/`

Top-line comparison:

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Identity collapse | Takeaway |
|-----------|--------|--------------------------|----------|----------------|-------------------|----------|
| `mixed_quality_labeled_guarded` | `18.2%` | `0.0764` | `0.0978` | `-0.1234` | `69` | Previous best mixed-data recall condition |
| `mixed_teacher_threshold_balanced` | `18.2%` | `0.0785` | `0.0829` | `-0.1012` | `62` | Matches best mixed-data recall while improving novelty, WER, MOS, and identity collapse |
| `mixed_teacher_labeled_finish` | `16.7%` | `0.0763` | `0.0727` | `-0.1150` | `68` | Better WER, but recall falls back to the conservative basin |
| `mixed_teacher_labeled_guarded` | `18.2%` | `0.0760` | `0.1095` | `-0.1173` | `70` | Preserves the recall bump, but gives back too much WER and identity stability |

Interpretation:
- The first teacher-family run does **not** move mixed-data recall above `18.2%`
- `mixed_teacher_threshold_balanced` is still the best teacher-family result, because it matches the best mixed-data recall while improving WER, MOS, novelty, and identity collapse versus `mixed_quality_labeled_guarded`
- The guarded teacher schedule is not the right next direction inside the current single-teacher family
- The next branch should compare an alternative pseudo-label teacher or a multi-teacher / agreement rule rather than repeating more schedule variants on the same teacher

---

### 0.20 Mapped-Score Teacher-Agreement Follow-Up (May 4, branch `research/controllable-vae`)

What we changed:
- Extended `scripts/annotate_commonvoice_pseudolabels.py` with an optional second teacher view (`--consistency-view center_crop`) so each CommonVoice row can preserve:
  - top-k teacher predictions
  - mapped per-style score maps
  - row-level agreement metadata between the full clip and the center crop
- Extended `scripts/filter_commonvoice_pseudolabels.py` with `--secondary-agreement-mode mapped_score`, so the secondary view can support the primary style by score rather than strict top-1 equality
- Rebuilt the agreement-filtered CommonVoice artifact:
  - `embeddings/openvoice_commonvoice_cv500_pseudo_agreement_mapped015_filtered.pt`
- Rebuilt the mixed artifact:
  - `embeddings/openvoice_mixed_teacher_mapped015_base.pt`
- Trained and evaluated the new checkpoint:
  - `embeddings/openvoice_vae_mixed_teacher_mapped015_balanced.pt`
  - `output/mixed_teacher_mapped015_balanced_eval/`
  - `results/eval_emotion_mixed_teacher_mixed_teacher_mapped015_balanced.csv`
  - `results/eval_novelty_mixed_teacher_mixed_teacher_mapped015_balanced.csv`
  - `results/eval_wer_mixed_teacher_mixed_teacher_mapped015_balanced.csv`
  - `results/eval_mos_mixed_teacher_mixed_teacher_mapped015_balanced.csv`

Validation:
- `Validation`: The same-teacher agreement rule is now reproducible from checked-in score -> filter -> build -> train scripts
- `Validation`: The new agreement condition has a named checkpoint, corpus, and full metric bundle
- `Validation`: The comparison explicitly answers whether a softer same-teacher agreement rule beats `mixed_teacher_threshold_balanced`

Agreement-filter selection summary:
- Selected CommonVoice counts before speaker-first mixing:
  - `anger=6`
  - `disgust=26`
  - `fear=4`
  - `happy=37`
  - `neutral=120`
  - `sad=110`
- Selected pseudo-style counts inside the real mixed artifact:
  - `anger=5`
  - `disgust=22`
  - `fear=4`
  - `happy=32`
  - `neutral=79`
  - `sad=82`

Top-line comparison:

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Identity collapse | Takeaway |
|-----------|--------|--------------------------|----------|----------------|-------------------|----------|
| `mixed_teacher_threshold_balanced` | `18.2%` | `0.0785` | `0.0829` | `-0.1012` | `62` | Current best teacher-family result |
| `mixed_teacher_mapped015_balanced` | `16.7%` | `0.0818` | `0.1090` | `-0.1115` | `63` | Slight novelty gain, but loses recall and gives back WER/MOS |

Interpretation:
- The softer mapped-score agreement rule does **not** beat `mixed_teacher_threshold_balanced`
- Its only clear win is a small novelty bump (`0.0818` vs `0.0785`)
- That novelty bump is not enough to justify the drop back to `16.7%` recall or the worse WER/MOS profile
- The branch now has a stronger negative result: cleaner reuse of the same teacher, even with a softer second-view agreement rule, is still not enough
- The next useful comparison should change the teacher itself or use a richer multi-teacher rule, not keep polishing agreement heuristics on the same model

---

### 0.21 Latent Prototype Pseudo-Label Teacher Follow-Up (May 4, branch `research/controllable-vae`)

What we changed:
- Added `scripts/annotate_commonvoice_latent_prototypes.py` as a genuinely different CommonVoice pseudo-label teacher
- Used `embeddings/openvoice_vae_combined.pt` to encode the labeled combined corpus and build one prototype per style from VAE style dims `0-8`
- Scored `embeddings/openvoice_commonvoice_cv500_emb.pt` against those style prototypes instead of using emotion2vec labels
- Filtered the prototype pseudo-labels with the same balanced-target acceptance path and rebuilt the mixed artifact:
  - `embeddings/openvoice_commonvoice_cv500_pseudo_prototype.pt`
  - `embeddings/openvoice_commonvoice_cv500_pseudo_prototype_filtered.pt`
  - `embeddings/openvoice_mixed_teacher_prototype_base.pt`
- Trained and evaluated the prototype checkpoint:
  - `embeddings/openvoice_vae_mixed_teacher_prototype_balanced.pt`
  - `output/mixed_teacher_prototype_balanced_eval/`
  - `results/eval_emotion_mixed_teacher_mixed_teacher_prototype_balanced.csv`
  - `results/eval_novelty_mixed_teacher_mixed_teacher_prototype_balanced.csv`
  - `results/eval_wer_mixed_teacher_mixed_teacher_prototype_balanced.csv`
  - `results/eval_mos_mixed_teacher_mixed_teacher_prototype_balanced.csv`

Validation:
- `Validation`: The alternate-teacher path is reproducible from checked-in scripts with local artifacts and no hardcoded dataset path
- `Validation`: The prototype condition has a named scored artifact, filtered artifact, mixed artifact, checkpoint, corpus, and full metric bundle
- `Validation`: The comparison explicitly answers whether a genuinely different teacher beats the current teacher-family reference
- `Validation`: The branch isolates pseudo-label teacher changes rather than changing the VAE architecture or evaluation corpus

Prototype pseudo-label selection:
- Accepted counts before speaker-first mixing:
  - `anger=40`
  - `confused=40`
  - `disgust=40`
  - `enunciated=40`
  - `fear=36`
  - `happy=40`
  - `neutral=40`
  - `sad=40`
  - `whisper=12`
- Selected pseudo-style counts inside the real mixed artifact:
  - `anger=25`
  - `confused=25`
  - `disgust=33`
  - `enunciated=29`
  - `fear=21`
  - `happy=27`
  - `neutral=23`
  - `sad=34`
  - `whisper=10`

Top-line comparison:

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Identity collapse | Mixed collapse | Takeaway |
|-----------|--------|--------------------------|----------|----------------|-------------------|----------------|----------|
| `mixed_teacher_threshold_balanced` | `18.2%` | `0.0785` | `0.0829` | `-0.1012` | `62` | `49` | Best overall mixed-data teacher reference |
| `mixed_teacher_mapped015_balanced` | `16.7%` | `0.0818` | `0.1090` | `-0.1115` | `63` | `51` | Softer same-teacher agreement raises novelty but loses recall/WER/MOS |
| `mixed_teacher_prototype_balanced` | `18.2%` | `0.0854` | `0.1009` | `-0.1086` | `60` | `48` | Best mixed-teacher novelty and slightly lower collapse, but not the best overall tradeoff |

Interpretation:
- The prototype teacher is a real alternate-teacher result, not another same-teacher filtering tweak
- It gives much broader CommonVoice pseudo-style coverage than emotion2vec, including `confused`, `enunciated`, and `whisper`
- It ties the best mixed-data recall at `18.2%` and produces the best mixed-teacher novelty so far (`0.0854`)
- It slightly reduces identity and mixed collapse versus `mixed_teacher_threshold_balanced`
- It does not replace `mixed_teacher_threshold_balanced` as the overall reference because WER and MOS are worse
- The next practical move at this point was a guarded prototype variant or prototype+emotion2vec multi-teacher rule; both have now been tested in sections 0.22 and 0.23

Future upgrade to preserve:
- `[DONE]` Build `mixed_teacher_prototype_guarded` with pseudo-confidence scaling, lower CommonVoice pseudo row weight, and stronger true-label protection; result preserved WER somewhat but erased the prototype novelty advantage
- `[DONE]` Compare prototype+emotion2vec union/intersection-style policy; the tested hybrid extra-priority route improved novelty but lost recall/MOS, so richer style-space supervision is now higher priority than more hard-label arbitration
- `[SOON]` Audit low-confidence `whisper` prototype rows manually or with an acoustic whisper proxy before scaling, because prototype confidence for `whisper` is much weaker than the other styles

### 0.22 Guarded Prototype Pseudo-Label Teacher Follow-Up (May 4, branch `research/controllable-vae`)

What we changed:
- Built `mixed_teacher_prototype_guarded` from the filtered prototype pseudo-label artifact using pseudo-confidence scaling, lower CommonVoice pseudo row weight, and stronger true-label protection
- Added `mixed_teacher_prototype_guarded` to `scripts/run_ablation_inference.py` so it can be regenerated through the same ablation CLI as the other teacher conditions
- Trained and evaluated the guarded prototype checkpoint:
  - `embeddings/openvoice_mixed_teacher_prototype_guarded_base.pt`
  - `embeddings/openvoice_vae_mixed_teacher_prototype_guarded.pt`
  - `output/mixed_teacher_prototype_guarded_eval/`
  - `results/eval_emotion_mixed_teacher_mixed_teacher_prototype_guarded.csv`
  - `results/eval_novelty_mixed_teacher_mixed_teacher_prototype_guarded.csv`
  - `results/eval_wer_mixed_teacher_mixed_teacher_prototype_guarded.csv`
  - `results/eval_mos_mixed_teacher_mixed_teacher_prototype_guarded.csv`
  - `results/eval_mixed_teacher_summary.csv`
  - `results/eval_mixed_teacher_collapse.csv`

Validation:
- `Validation`: The guarded prototype condition has a named mixed artifact, checkpoint, evaluation corpus, and full metric bundle
- `Validation`: The ablation inference CLI exposes the guarded condition with deterministic `style_strength=5.0`, `noise_level=0.0`, and `seed=42` reproduction settings
- `Validation`: The comparison explicitly answers whether strong guardrails preserve the prototype teacher's novelty/coverage gain while improving WER/MOS
- `Validation`: The branch isolates pseudo-label weighting and schedule guardrails rather than changing the VAE architecture or evaluation corpus

Guarded prototype pseudo-label selection:
- Selected CommonVoice pseudo-style counts inside the real mixed artifact:
  - `anger=25`
  - `confused=25`
  - `disgust=33`
  - `enunciated=29`
  - `fear=21`
  - `happy=27`
  - `neutral=23`
  - `sad=34`
  - `whisper=10`
- The selected style coverage stayed the same as the unguarded prototype artifact; the experiment changed row weighting and training schedule, not pseudo-label coverage

Top-line comparison:

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Identity collapse | Mixed collapse | Takeaway |
|-----------|--------|--------------------------|----------|----------------|-------------------|----------------|----------|
| `mixed_teacher_threshold_balanced` | `18.2%` | `0.0785` | `0.0829` | `-0.1012` | `62` | `49` | Best overall mixed-data teacher reference |
| `mixed_teacher_prototype_balanced` | `18.2%` | `0.0854` | `0.1009` | `-0.1086` | `60` | `48` | Best mixed-teacher novelty and slightly lower collapse, but worse WER/MOS |
| `mixed_teacher_prototype_guarded` | `18.2%` | `0.0761` | `0.0920` | `-0.1081` | `71` | `51` | Strong guardrails improve WER versus the unguarded prototype but erase the prototype novelty/collapse advantage |

Interpretation:
- The strong guard did not break recall, but it also did not break the `18.2%` mixed-data recall ceiling
- It partially repaired prototype WER (`0.0920` vs `0.1009`) while leaving MOS essentially unchanged
- It gave back the prototype teacher's main benefit: novelty fell from `0.0854` to `0.0761`
- Collapse behavior worsened, especially identity collapse (`71` vs `60` for the unguarded prototype and `62` for the threshold teacher)
- The result argues against simply making prototype supervision weaker and more labeled-heavy; the next useful move should either use an intermediate guard or combine prototype coverage with emotion2vec precision through a multi-teacher rule

Future upgrade to preserve:
- `[DONE]` Compare a prototype+emotion2vec multi-teacher policy that keeps prototype labels for `confused`, `enunciated`, and `whisper` while using emotion2vec for canonical emotions; result logged in section 0.23
- `[NOW]` Move from hard row-label teacher mixing to richer style-space supervision, prototype distillation, or per-style curriculum because the multi-teacher hard-label route improved novelty but did not improve recall or MOS
- `[SOON]` Try an intermediate guard (`pseudo_row_weight=0.75`, `true_row_weight=1.25`) only if we need one more scalar ablation for completeness; the hybrid result makes a structural style-space objective higher priority
- `[SOON]` Add per-style collapse diagnostics for the prototype and hybrid teacher family so we can see whether the failures are mostly rare styles, canonical emotion classes, or specific source speakers

### 0.23 Hybrid Emotion2Vec + Prototype Extra-Style Teacher Follow-Up (May 4, branch `research/controllable-vae`)

What changed:
- Added `scripts/combine_commonvoice_pseudolabel_teachers.py`, a reusable CommonVoice artifact combiner that merges filtered emotion2vec pseudo labels with filtered combined-VAE latent prototype labels
- Used the `prototype_extra_priority` policy:
  - emotion2vec supplies canonical emotion labels (`anger`, `disgust`, `fear`, `happy`, `neutral`, `sad`)
  - the latent-prototype teacher supplies extra controllable styles that emotion2vec cannot label directly (`confused`, `enunciated`, `whisper`)
- Built, trained, generated, and evaluated the hybrid condition:
  - `embeddings/openvoice_commonvoice_cv500_pseudo_hybrid_extra_priority.pt`
  - `embeddings/openvoice_mixed_teacher_hybrid_extra_base.pt`
  - `embeddings/openvoice_vae_mixed_teacher_hybrid_extra_balanced.pt`
  - `output/mixed_teacher_hybrid_extra_balanced_eval/`
  - `results/eval_emotion_mixed_teacher_mixed_teacher_hybrid_extra_balanced.csv`
  - `results/eval_novelty_mixed_teacher_mixed_teacher_hybrid_extra_balanced.csv`
  - `results/eval_wer_mixed_teacher_mixed_teacher_hybrid_extra_balanced.csv`
  - `results/eval_mos_mixed_teacher_mixed_teacher_hybrid_extra_balanced.csv`
  - `results/eval_mixed_teacher_summary.csv`
  - `results/eval_mixed_teacher_collapse.csv`

Validation:
- `Validation`: The hybrid combiner compiles and preserves row-level pseudo-label metadata, component source, selected reason, teacher metadata, and filter reports
- `Validation`: The hybrid CommonVoice artifact, mixed artifact, trained checkpoint, deterministic evaluation corpus, and full metric bundle were produced under stable names
- `Validation`: The inference CLI exposes `mixed_teacher_hybrid_extra_balanced` with deterministic `style_strength=5.0`, `noise_level=0.0`, and `seed=42` reproduction settings
- `Validation`: The comparison explicitly answers whether emotion2vec canonical labels plus prototype extra-style labels improves the overall mixed-data teacher tradeoff

Hybrid pseudo-label selection:
- Source-level selected rows before one-clip-per-speaker mixing:
  - `emotion2vec=293`
  - `prototype=92`
  - `unselected=817`
- Selected style counts before one-clip-per-speaker mixing:
  - `anger=5`
  - `confused=40`
  - `disgust=32`
  - `enunciated=40`
  - `fear=4`
  - `happy=37`
  - `neutral=109`
  - `sad=106`
  - `whisper=12`
- Selected CommonVoice pseudo-style counts inside the real mixed artifact:
  - `anger=4`
  - `confused=22`
  - `disgust=27`
  - `enunciated=27`
  - `fear=4`
  - `happy=30`
  - `neutral=66`
  - `sad=78`
  - `whisper=9`

Top-line comparison:

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Identity collapse | Style collapse | Mixed collapse | Files with any collapse | Takeaway |
|-----------|--------|--------------------------|----------|----------------|-------------------|----------------|----------------|-------------------------|----------|
| `mixed_teacher_threshold_balanced` | `18.2%` | `0.0785` | `0.0829` | `-0.1012` | `62` | `53` | `49` | `66` | Best overall mixed-data teacher reference |
| `mixed_teacher_mapped015_balanced` | `16.7%` | `0.0818` | `0.1090` | `-0.1115` | `63` | `54` | `51` | `66` | Softer same-teacher agreement raises novelty but loses recall/WER/MOS |
| `mixed_teacher_prototype_balanced` | `18.2%` | `0.0854` | `0.1009` | `-0.1086` | `60` | `54` | `48` | `66` | Best prototype novelty/coverage before hybrid, but worse WER/MOS |
| `mixed_teacher_prototype_guarded` | `18.2%` | `0.0761` | `0.0920` | `-0.1081` | `71` | `52` | `51` | `72` | Strong guard improves WER versus prototype but erases novelty/collapse advantage |
| `mixed_teacher_hybrid_extra_balanced` | `16.7%` | `0.0860` | `0.0931` | `-0.1190` | `61` | `55` | `51` | `65` | Best mixed-teacher novelty and slightly fewer files with any collapse, but recall/MOS are worse |

Interpretation:
- The hybrid teacher did not become the new overall reference: recall fell back to `16.7%`, WER remained worse than `mixed_teacher_threshold_balanced`, and MOS delta worsened
- It did produce the best mixed-teacher novelty so far (`0.0860`) and the lowest files-with-any-collapse count in the teacher matrix (`65`)
- The result supports the narrow conclusion that prototype labels add useful novelty/coverage, especially for `confused`, `enunciated`, and `whisper`
- The result argues against another hard row-label mixing tweak as the main next move; the higher-value next step is to use prototype/style teacher information as an auxiliary style-space target, distillation loss, or per-style curriculum rather than as only selected hard pseudo labels

Future upgrade to preserve:
- `[DONE]` Design and test a style-space auxiliary/prototype-distillation objective for mixed-data training so CommonVoice rows can carry continuous teacher geometry without forcing noisy hard style labels; first result logged in section 0.24
- `[SOON]` Add a per-style curriculum for `confused`, `enunciated`, and `whisper`, because prototype extra-style labels consistently buy novelty but need better protection from WER/MOS degradation
- `[SOON]` Diagnose whether hybrid teacher failures come from tiny rare canonical counts (`anger=4`, `fear=4` after speaker-first mixing), pseudo-label noise, or row-label supervision being too weak to steer the VAE decoder
- `[SOON]` Add per-style collapse diagnostics to the teacher summary so the next objective can target the classes and source speakers that actually fail

---

### 0.24 Mixed-Data Style-Space Distillation Follow-Up (May 5, branch `research/controllable-vae`)

What changed:
- Extended `dpvc/utils.py` so mixed-data training can add a frozen teacher-style MSE loss on selected VAE encoder mean dimensions
- Extended `examples/openvoice_train_vae_mixed.py` with:
  - `--style-teacher-checkpoint`
  - `--style-teacher-weight`
  - `--style-teacher-dims`
  - `--style-teacher-datasets`
- Added the deterministic evaluation condition `mixed_teacher_hybrid_style_distill_balanced` to `scripts/run_ablation_inference.py`
- Trained the first style-space distillation checkpoint from the hybrid teacher artifact:
  - `embeddings/openvoice_mixed_teacher_hybrid_extra_base.pt`
  - `embeddings/openvoice_vae_mixed_teacher_hybrid_style_distill_balanced.pt`
- Generated and evaluated the matched corpus:
  - `output/mixed_teacher_hybrid_style_distill_balanced_eval/`
  - `results/eval_emotion_mixed_teacher_mixed_teacher_hybrid_style_distill_balanced.csv`
  - `results/eval_novelty_mixed_teacher_mixed_teacher_hybrid_style_distill_balanced.csv`
  - `results/eval_wer_mixed_teacher_mixed_teacher_hybrid_style_distill_balanced.csv`
  - `results/eval_mos_mixed_teacher_mixed_teacher_hybrid_style_distill_balanced.csv`
  - `results/eval_mixed_teacher_summary.csv`
  - `results/eval_mixed_teacher_collapse.csv`

Validation:
- `Validation`: The new mixed-training CLI flags compile and appear in `examples/openvoice_train_vae_mixed.py --help`
- `Validation`: A two-epoch smoke run verified teacher masking and scale on the real hybrid artifact (`500/1325` CommonVoice teacher-style rows, style dims `0-8`, teacher weight `0.25`)
- `Validation`: The full checkpoint trained from `embeddings/openvoice_mixed_teacher_hybrid_extra_base.pt` with deterministic seed-controlled generation
- `Validation`: The condition has a named checkpoint, evaluation corpus, four metric CSVs, and regenerated summary/collapse tables
- `Validation`: The comparison explicitly answers whether continuous style-space teacher supervision improves over hard hybrid pseudo-label mixing

Top-line comparison:

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Identity collapse | Style collapse | Mixed collapse | Files with any collapse | Takeaway |
|-----------|--------|--------------------------|----------|----------------|-------------------|----------------|----------------|-------------------------|----------|
| `mixed_teacher_threshold_balanced` | `18.2%` | `0.0785` | `0.0829` | `-0.1012` | `62` | `53` | `49` | `66` | Best overall mixed-data teacher reference |
| `mixed_teacher_prototype_balanced` | `18.2%` | `0.0854` | `0.1009` | `-0.1086` | `60` | `54` | `48` | `66` | Best prototype novelty/coverage before hybrid, but worse WER/MOS |
| `mixed_teacher_hybrid_extra_balanced` | `16.7%` | `0.0860` | `0.0931` | `-0.1190` | `61` | `55` | `51` | `65` | Best hard-label hybrid novelty, but recall/MOS are worse |
| `mixed_teacher_hybrid_style_distill_balanced` | `16.7%` | `0.0861` | `0.0938` | `-0.1072` | `58` | `55` | `50` | `63` | Preserves hybrid novelty and improves MOS/collapse, but recall remains stuck |

Interpretation:
- Continuous style-space supervision is a better use of hybrid teacher geometry than hard row-label mixing on naturalness/collapse: MOS delta improves from `-0.1190` to `-0.1072`, identity collapse falls from `61` to `58`, mixed collapse falls from `51` to `50`, and files with any collapse fall from `65` to `63`
- It preserves the hybrid teacher's novelty advantage (`0.0861`, narrowly above `0.0860`)
- It still does **not** recover emotion recall; recall remains `16.7%`, with the same neutral-basin failure pattern
- It also does not replace `mixed_teacher_threshold_balanced` as the best overall mixed-data teacher reference because threshold remains better on recall, WER, and MOS delta
- The next useful move is not another hard pseudo-label arbitration rule; it is a calibrated style-space objective, likely with per-style weighting/curriculum and teacher-confidence masks

Future upgrade to preserve:
- `[DONE]` Sweep style-teacher weights (`0.10`, `0.50`, and optionally a ramp) to test whether the current `0.25` weight is underpowered for recall or already at the naturalness/novelty sweet spot; first scalar sweep result logged in section 0.25
- `[NOW]` Add per-style teacher masks/curricula so canonical emotion styles with weak CommonVoice support are not dominated by neutral/sad teacher geometry
- `[SOON]` Compare teacher targets from the prototype-only teacher versus the hybrid teacher inside the same continuous style-space loss, because hard-label prototype supervision produced broader coverage but worse WER/MOS
- `[SOON]` Add teacher-confidence weighting to the style-space loss instead of treating all selected CommonVoice teacher rows equally
- `[SOON]` Inspect per-style failures for the style-distillation model, especially whether `confused`, `enunciated`, and `whisper` preserve novelty while canonical emotions collapse to neutral

---

### 0.25 Style-Space Distillation Weight Sweep (May 5, branch `research/controllable-vae`)

What changed:
- Added deterministic inference conditions for two calibrated style-teacher weights:
  - `mixed_teacher_hybrid_style_distill_w010_balanced`
  - `mixed_teacher_hybrid_style_distill_w050_balanced`
- Trained both checkpoints from the same hybrid mixed artifact and schedule as the original `0.25` run:
  - `embeddings/openvoice_mixed_teacher_hybrid_extra_base.pt`
  - `embeddings/openvoice_vae_mixed_teacher_hybrid_style_distill_w010_balanced.pt`
  - `embeddings/openvoice_vae_mixed_teacher_hybrid_style_distill_w050_balanced.pt`
- Generated matched deterministic corpora:
  - `output/mixed_teacher_hybrid_style_distill_w010_balanced_eval/`
  - `output/mixed_teacher_hybrid_style_distill_w050_balanced_eval/`
- Evaluated both with the full metric stack:
  - `results/eval_emotion_mixed_teacher_mixed_teacher_hybrid_style_distill_w010_balanced.csv`
  - `results/eval_novelty_mixed_teacher_mixed_teacher_hybrid_style_distill_w010_balanced.csv`
  - `results/eval_wer_mixed_teacher_mixed_teacher_hybrid_style_distill_w010_balanced.csv`
  - `results/eval_mos_mixed_teacher_mixed_teacher_hybrid_style_distill_w010_balanced.csv`
  - `results/eval_emotion_mixed_teacher_mixed_teacher_hybrid_style_distill_w050_balanced.csv`
  - `results/eval_novelty_mixed_teacher_mixed_teacher_hybrid_style_distill_w050_balanced.csv`
  - `results/eval_wer_mixed_teacher_mixed_teacher_hybrid_style_distill_w050_balanced.csv`
  - `results/eval_mos_mixed_teacher_mixed_teacher_hybrid_style_distill_w050_balanced.csv`
  - `results/eval_mixed_teacher_summary.csv`
  - `results/eval_mixed_teacher_collapse.csv`

Validation:
- `Validation`: Both new checkpoints trained from the fixed hybrid artifact with only `--style-teacher-weight` changed (`0.10` and `0.50`)
- `Validation`: Both new inference conditions generate the same 110-row, 11-speaker evaluation corpus shape as the existing mixed-teacher conditions
- `Validation`: Both new conditions have a four-metric CSV bundle and regenerated summary/collapse rows
- `Validation`: The comparison explicitly answers whether a global style-teacher weight change recovers recall

Top-line comparison:

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Identity collapse | Style collapse | Mixed collapse | Files with any collapse | Takeaway |
|-----------|--------|--------------------------|----------|----------------|-------------------|----------------|----------------|-------------------------|----------|
| `mixed_teacher_threshold_balanced` | `18.2%` | `0.0785` | `0.0829` | `-0.1012` | `62` | `53` | `49` | `66` | Best overall mixed-data teacher reference |
| `mixed_teacher_hybrid_style_distill_w010_balanced` | `16.7%` | `0.0854` | `0.0924` | `-0.1161` | `62` | `54` | `51` | `65` | Lower teacher weight preserves novelty but gives back MOS/collapse |
| `mixed_teacher_hybrid_style_distill_balanced` | `16.7%` | `0.0861` | `0.0938` | `-0.1072` | `58` | `55` | `50` | `63` | Best style-distillation novelty/MOS tradeoff so far |
| `mixed_teacher_hybrid_style_distill_w050_balanced` | `16.7%` | `0.0840` | `0.0821` | `-0.1196` | `56` | `55` | `48` | `63` | Higher teacher weight improves WER and identity/mixed collapse, but not recall or MOS |

Interpretation:
- A global teacher-weight sweep does **not** recover target-emotion recall; all three tested weights stay at `16.7%`
- Weight `0.50` improves mean WER (`0.0821`) and identity/mixed collapse (`56` / `48`), but loses novelty and naturalness relative to the `0.25` run
- Weight `0.25` remains the best style-distillation tradeoff for novelty and MOS, but it still does not replace `mixed_teacher_threshold_balanced` because recall remains lower
- The bottleneck is probably not "teacher loss too weak" globally; it is more likely per-style imbalance, noisy teacher geometry for canonical emotions, or a decoder/latent mismatch that needs class-specific masks, confidence weighting, or curriculum

Future upgrade to preserve:
- `[DONE]` Add per-style teacher masks or style-specific loss weights so rare canonical emotions (`anger`, `fear`, `happy`, `sad`) are not trained with the same scalar pressure as high-supply or extra-style rows; the target-masked follow-up is logged in section 0.26 and did not recover recall
- `[DONE]` Add teacher-confidence weighting to the continuous style loss so low-confidence CommonVoice pseudo-style rows contribute less than high-confidence rows; the first confidence-weighted target-mask run is logged in section 0.26
- `[SOON]` Test a curriculum that starts with protected true-label reconstruction/style loss, then introduces continuous CommonVoice teacher geometry after the labeled style axes are stable
- `[SOON]` Compare prototype-only continuous targets against hybrid continuous targets, because the scalar hybrid sweep shows teacher geometry is useful for novelty but not sufficient for recall

---

### 0.26 Target-Dimension Style-Teacher Mask Follow-Up (2026-05-05, branch `research/controllable-vae`)

What changed:

- Added target-dimension style-teacher loss support to mixed-data training:
  - `--style-teacher-target-mode {all_dims,target_dim}`
  - `--style-teacher-require-label`
  - `--style-teacher-style-weights`
  - `--style-teacher-confidence-power`
- Extended `dpvc/utils.py` so the frozen teacher loss can be applied only to
  the accepted style dimension for each row, with optional row weights and
  pseudo-label confidence scaling.
- Added the deterministic inference condition
  `mixed_teacher_hybrid_style_distill_targetmask_balanced`.
- Trained the first target-masked / per-style-weighted / confidence-weighted
  style-distillation checkpoint from the fixed hybrid artifact:
  - `embeddings/openvoice_mixed_teacher_hybrid_extra_base.pt`
  - `embeddings/openvoice_vae_mixed_teacher_hybrid_style_distill_targetmask_balanced.pt`
- Evaluated the matched 110-row, 11-speaker corpus:
  - `output/mixed_teacher_hybrid_style_distill_targetmask_balanced_eval/`
  - `results/eval_emotion_mixed_teacher_mixed_teacher_hybrid_style_distill_targetmask_balanced.csv`
  - `results/eval_novelty_mixed_teacher_mixed_teacher_hybrid_style_distill_targetmask_balanced.csv`
  - `results/eval_wer_mixed_teacher_mixed_teacher_hybrid_style_distill_targetmask_balanced.csv`
  - `results/eval_mos_mixed_teacher_mixed_teacher_hybrid_style_distill_targetmask_balanced.csv`
  - regenerated `results/eval_mixed_teacher_summary.csv`
  - regenerated `results/eval_mixed_teacher_collapse.csv`

Training command:

```bash
python examples/openvoice_train_vae_mixed.py \
    --embeddings embeddings/openvoice_mixed_teacher_hybrid_extra_base.pt \
    --output embeddings/openvoice_vae_mixed_teacher_hybrid_style_distill_targetmask_balanced.pt \
    --schedule static_balanced \
    --style-teacher-checkpoint embeddings/openvoice_vae_combined.pt \
    --style-teacher-weight 0.25 \
    --style-teacher-datasets CommonVoice \
    --style-teacher-dims 0-8 \
    --style-teacher-target-mode target_dim \
    --style-teacher-require-label \
    --style-teacher-style-weights anger=4.0,fear=4.0,happy=2.0,disgust=2.0,sad=1.5,neutral=0.25,confused=1.0,enunciated=1.0,whisper=1.0 \
    --style-teacher-confidence-power 0.5
```

Validation:

- `Validation`: Python compile checks passed for the changed training,
  utility, and inference scripts.
- `Validation`: The new training flags appear in
  `examples/openvoice_train_vae_mixed.py --help`.
- `Validation`: A two-epoch smoke run verified positive teacher loss with
  `target_dim`, accepted-label masking, style row weights, and confidence
  scaling on the real hybrid artifact.
- `Validation`: The full checkpoint trained with `267/1325` active teacher
  rows, target mode `target_dim`, row-weight mean `1.705`, min `0.250`, max
  `4.000`.
- `Validation`: Deterministic inference wrote the expected 110-row manifest;
  the only warning was the known short-audio watermark warning for
  `male_2_cremad_1051`.
- `Validation`: Emotion, novelty, WER, and MOS CSVs were regenerated and
  summarized into the mixed-teacher summary/collapse tables.
- `Validation`: The comparison explicitly answers whether per-style target
  masks and confidence weighting recover recall.

Top-line comparison:

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Identity collapse | Style collapse | Mixed collapse | Files with any collapse | Takeaway |
|-----------|--------|--------------------------|----------|----------------|-------------------|----------------|----------------|-------------------------|----------|
| `mixed_teacher_threshold_balanced` | `18.2%` | `0.0785` | `0.0829` | `-0.1012` | `62` | `53` | `49` | `66` | Best overall mixed-data teacher reference |
| `mixed_teacher_hybrid_style_distill_balanced` | `16.7%` | `0.0861` | `0.0938` | `-0.1072` | `58` | `55` | `50` | `63` | Best global style-distillation novelty/MOS tradeoff |
| `mixed_teacher_hybrid_style_distill_targetmask_balanced` | `16.7%` | `0.0852` | `0.1062` | `-0.1181` | `60` | `54` | `51` | `63` | Target masks and confidence weighting do not recover recall and slightly worsen WER/MOS |
| `mixed_teacher_hybrid_style_distill_w050_balanced` | `16.7%` | `0.0840` | `0.0821` | `-0.1196` | `56` | `55` | `48` | `63` | Higher global teacher weight improves WER/collapse but not recall or MOS |

Interpretation:

- Target-dimension teacher masking does **not** break the neutral recall basin;
  recall remains `16.7%`, again entirely from neutral.
- The run preserves much of the style-distillation novelty signal (`0.0852`),
  especially for `confused` and `whisper`, but it does not improve classifier
  target alignment.
- Compared with global `0.25` style distillation, target masking slightly
  lowers novelty (`0.0852` vs `0.0861`) and worsens mean WER / MOS delta
  (`0.1062` / `-0.1181` vs `0.0938` / `-0.1072`).
- The failure is now narrower: the issue is not simply global teacher-loss
  strength, and it is not solved by only supervising the accepted style
  dimension. The likely gap is decoder-aware style alignment, pseudo-label
  quality, rare-class supply, or curriculum timing.

Future upgrade to preserve:

- `[DONE]` Add a per-style diagnostic/probe report before the next training run:
  compare teacher mean targets, student encoder means, generated emotion
  predictions, novelty gain, and collapse flags by style and speaker so the
  next objective is aimed at the actual failing axes rather than guessed; first
  report logged in section 0.27.
- `[NOW]` Test a curriculum that first protects labeled CREMA-D/Expresso style
  axes, then introduces CommonVoice teacher geometry after the decoder has a
  stable target-emotion map.
- `[SOON]` Add a decoder-aware style objective or generated-audio
  teacher/proxy signal if latent teacher alignment keeps moving novelty without
  moving emotion2vec recall.
- `[SOON]` Compare prototype-only continuous targets against hybrid continuous
  targets only after the diagnostic confirms whether the current hybrid teacher
  geometry is the source of the neutral-basin failure.

---

### 0.27 Per-Style Teacher Diagnostic Report (2026-05-05, branch `research/controllable-vae`)

What changed:

- Added `scripts/analyze_mixed_teacher_style_diagnostics.py`, a reusable
  diagnostic tool that joins:
  - mixed-artifact label supply by style
  - frozen teacher encoder means
  - trained student encoder means
  - generated-output emotion / novelty / WER / MOS metrics
  - collapse taxonomy rows
- Ran the diagnostic on:
  - mixed artifact:
    `embeddings/openvoice_mixed_teacher_hybrid_extra_base.pt`
  - frozen teacher:
    `embeddings/openvoice_vae_combined.pt`
  - student:
    `embeddings/openvoice_vae_mixed_teacher_hybrid_style_distill_targetmask_balanced.pt`
  - evaluated condition:
    `mixed_teacher_hybrid_style_distill_targetmask_balanced`
- New diagnostic artifacts:
  - `results/eval_mixed_teacher_style_diagnostics_targetmask.csv`
  - `results/eval_mixed_teacher_style_diagnostics_targetmask.md`

Validation:

- `Validation`: The diagnostic script compiles and exposes a documented CLI.
- `Validation`: The script successfully encoded all `1325` mixed-artifact rows
  through both teacher and student VAEs.
- `Validation`: The report joins the existing four-metric target-mask CSVs and
  collapse taxonomy without regenerating audio.
- `Validation`: The diagnostic explains why the target-mask run failed before
  we spend another turn on a guessed objective.

Key diagnostic table:

| Style | Active teacher rows | Teacher target top1 | Student target top1 | Recall | Neutral prediction rate | Novelty | WER | MOS delta | Takeaway |
|-------|---------------------|---------------------|---------------------|--------|-------------------------|---------|-----|-----------|----------|
| `anger` | `4` | `0.0000` | `0.0000` | `0.0000` | `1.0000` | `-0.0020` | `0.0325` | `-0.0011` | Rare rows and teacher geometry do not support the target axis |
| `disgust` | `27` | `0.2222` | `0.0741` | `0.0000` | `1.0000` | `0.0251` | `0.0507` | `+0.0110` | Moderate supply, but teacher target is usually not dominant |
| `fear` | `4` | `0.0000` | `0.0000` | `0.0000` | `1.0000` | `-0.0036` | `0.1071` | `-0.0320` | Same failure as anger: too few rows and no teacher target dominance |
| `happy` | `30` | `0.1000` | `0.2333` | `0.0000` | `0.9091` | `-0.0011` | `0.2089` | `-0.0210` | More rows, but teacher geometry still does not point cleanly at happy |
| `sad` | `78` | `0.2692` | `0.7308` | `0.0000` | `1.0000` | `0.0232` | `0.0617` | `-0.0148` | Student can make sad latent top-ranked, but generated audio still reads neutral |
| `confused` | `22` | `1.0000` | `0.6818` | n/a | `0.8182` | `0.2411` | `0.0831` | `-0.3444` | Extra-style geometry moves novelty, but hurts MOS and has no emotion2vec recall label |
| `whisper` | `9` | `1.0000` | `0.7778` | n/a | `0.3636` | `0.3714` | `0.1838` | `-0.5144` | Strong latent novelty, but quality/intelligibility cost remains high |

Interpretation:

- The target-mask failure is now localized. For canonical emotions that should
  count in emotion2vec recall, the frozen teacher's accepted CommonVoice rows
  usually do **not** rank the intended style dimension first.
- `anger` and `fear` have only `4` active CommonVoice teacher rows each, so row
  weighting cannot substitute for real data supply.
- `happy` has more rows (`30`) but still has weak teacher-target dominance
  (`0.1000`), which explains why per-style weighting did not help.
- `sad` is especially informative: the student makes the sad dim top-ranked for
  many active rows (`0.7308`), but generated audio is still classified as
  neutral. That points to a decoder/output-level mismatch, not just latent
  underfitting.
- `confused` and `whisper` show the opposite pattern: latent teacher geometry
  is coherent and novelty moves strongly, but these are not emotion2vec-scored
  recall classes and they carry WER/MOS costs.

Future upgrade to preserve:

- `[NOW]` Build a labeled-first curriculum condition that keeps CREMA-D/Expresso
  style axes stable before adding CommonVoice teacher geometry.
- `[NOW]` Add an output-level or decoder-aware style objective/proxy for
  canonical emotions, because latent alignment alone can still generate
  neutral-classified audio.
- `[SOON]` Rebuild CommonVoice pseudo-label supply for rare canonical classes
  before more weighting experiments; `anger=4` and `fear=4` are not enough.
- `[SOON]` Add this diagnostic to future mixed-teacher reports so every new
  condition reports both latent target dominance and generated metric behavior.

---

### 0.28 Labeled-First Style-Teacher Curriculum (2026-05-05, branch `research/controllable-vae`)

What changed:

- Added a `labeled_warmup` mixed-data schedule to `dpvc/utils.py`:
  - start masses normalize to `CommonVoice=0.00`, `CREMA-D=0.50`,
    `Expresso=0.50`
  - end masses normalize to `CommonVoice=0.33`, `CREMA-D=0.33`,
    `Expresso=0.33`
- Added `--style-teacher-weight-final` to
  `examples/openvoice_train_vae_mixed.py` so teacher-style pressure can ramp
  over the schedule instead of being static from epoch 1.
- Added deterministic inference support for
  `mixed_teacher_hybrid_style_distill_labeled_warmup`.
- Trained the curriculum checkpoint from the fixed hybrid teacher artifact:
  - `embeddings/openvoice_mixed_teacher_hybrid_extra_base.pt`
  - `embeddings/openvoice_vae_mixed_teacher_hybrid_style_distill_labeled_warmup.pt`
- Evaluated the matched 110-row, 11-speaker corpus:
  - `output/mixed_teacher_hybrid_style_distill_labeled_warmup_eval/`
  - `results/eval_emotion_mixed_teacher_mixed_teacher_hybrid_style_distill_labeled_warmup.csv`
  - `results/eval_novelty_mixed_teacher_mixed_teacher_hybrid_style_distill_labeled_warmup.csv`
  - `results/eval_wer_mixed_teacher_mixed_teacher_hybrid_style_distill_labeled_warmup.csv`
  - `results/eval_mos_mixed_teacher_mixed_teacher_hybrid_style_distill_labeled_warmup.csv`
  - regenerated `results/eval_mixed_teacher_summary.csv`
  - regenerated `results/eval_mixed_teacher_collapse.csv`
- Ran the per-style diagnostic on the new curriculum condition:
  - `results/eval_mixed_teacher_style_diagnostics_labeled_warmup.csv`
  - `results/eval_mixed_teacher_style_diagnostics_labeled_warmup.md`

Training command:

```bash
python examples/openvoice_train_vae_mixed.py \
    --embeddings embeddings/openvoice_mixed_teacher_hybrid_extra_base.pt \
    --output embeddings/openvoice_vae_mixed_teacher_hybrid_style_distill_labeled_warmup.pt \
    --schedule labeled_warmup \
    --schedule-epochs 1000 \
    --style-teacher-checkpoint embeddings/openvoice_vae_combined.pt \
    --style-teacher-weight 0.0 \
    --style-teacher-weight-final 0.25 \
    --style-teacher-datasets CommonVoice \
    --style-teacher-dims 0-8
```

Validation:

- `Validation`: Python compile checks passed for the changed training,
  utility, inference, summary, and diagnostic scripts.
- `Validation`: `examples/openvoice_train_vae_mixed.py --help` exposes
  `labeled_warmup` and `--style-teacher-weight-final`.
- `Validation`: `scripts/run_ablation_inference.py --help` exposes
  `mixed_teacher_hybrid_style_distill_labeled_warmup`.
- `Validation`: A two-epoch smoke run verified the curriculum starts with no
  CommonVoice mass and no teacher loss, then ramps toward balanced data and
  teacher weight `0.25`.
- `Validation`: The full checkpoint trained from
  `embeddings/openvoice_mixed_teacher_hybrid_extra_base.pt` with
  `schedule_epochs=1000`; by the end, the run used balanced
  CommonVoice / CREMA-D / Expresso masses and teacher weight `0.25`.
- `Validation`: Deterministic inference wrote the expected 110-row manifest;
  the only warning was the known short-audio watermark warning for
  `male_2_cremad_1051`.
- `Validation`: Emotion, novelty, WER, and MOS CSVs were regenerated and
  summarized into the mixed-teacher summary/collapse tables.
- `Validation`: The diagnostic report joins label supply, teacher/student
  latent geometry, generated metrics, and collapse rows for the curriculum
  condition.

Top-line comparison:

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Identity collapse | Style collapse | Mixed collapse | Files with any collapse | Takeaway |
|-----------|--------|--------------------------|----------|----------------|-------------------|----------------|----------------|-------------------------|----------|
| `mixed_teacher_threshold_balanced` | `18.2%` | `0.0785` | `0.0829` | `-0.1012` | `62` | `53` | `49` | `66` | Best overall mixed-data teacher reference |
| `mixed_teacher_hybrid_style_distill_balanced` | `16.7%` | `0.0861` | `0.0938` | `-0.1072` | `58` | `55` | `50` | `63` | Best previous style-distillation novelty/MOS tradeoff |
| `mixed_teacher_hybrid_style_distill_labeled_warmup` | `16.7%` | `0.0930` | `0.0924` | `-0.1093` | `54` | `55` | `48` | `61` | Labeled-first curriculum improves novelty/collapse but not emotion recall |
| `mixed_teacher_hybrid_style_distill_targetmask_balanced` | `16.7%` | `0.0852` | `0.1062` | `-0.1181` | `60` | `54` | `51` | `63` | Target masks and confidence weighting do not recover recall |

Interpretation:

- The labeled-first curriculum does **not** break the neutral recall basin;
  recall remains `16.7%`, again coming only from neutral.
- It does improve some secondary axes: novelty rises from `0.0861` to
  `0.0930` versus global `0.25` style distillation, identity collapse falls
  from `58` to `54`, mixed collapse falls from `50` to `48`, and files with
  any collapse fall from `63` to `61`.
- That means protecting labeled CREMA-D/Expresso axes before adding
  CommonVoice teacher geometry can help the latent/identity tradeoff, but it
  still does not make generated audio classifier-visible as target emotion.
- The next useful move is no longer another schedule-only curriculum with the
  same teacher. The next move should be decoder-aware / generated-audio style
  supervision or better rare-class CommonVoice supply.

Future upgrade to preserve:

- `[NOW]` Build a decoder-aware generated-audio style objective for canonical
  emotions, likely by scoring generated samples or decoder outputs with an
  emotion proxy instead of relying only on frozen latent teacher agreement.
- `[NOW]` Rebuild rare canonical pseudo-label supply before more weighting:
  `anger=4` and `fear=4` active teacher rows are too small for a stable class
  manifold.
- `[SOON]` Compare one-clip-per-speaker versus two-clips-per-speaker
  CommonVoice sampling under the same teacher, because rare-class supply may be
  constrained by the current speaker-first artifact.
- `[SOON]` Add per-condition diagnostic generation to the summary workflow so
  every future mixed-teacher condition automatically reports target top1,
  generated recall, novelty, WER, MOS, and collapse by style.

---

### 0.29 Listening Report and Rare-Class Supply Audit (2026-05-05, branch `research/controllable-vae`)

What changed:

- Added `scripts/build_listening_report.py`, a reusable local HTML report
  builder for generated corpora:
  - groups audio by source speaker
  - embeds source, baseline, and styled output audio controls
  - joins emotion, novelty, WER, and MOS metric rows when available
  - emits a companion subjective-rating CSV template
- Generated the first listening bundle for the labeled-warmup condition:
  - `results/listening_mixed_teacher_hybrid_style_distill_labeled_warmup.html`
  - `results/listening_mixed_teacher_hybrid_style_distill_labeled_warmup_ratings.csv`
- Added `scripts/audit_commonvoice_pseudolabel_supply.py`, a reusable audit for
  CommonVoice pseudo-label supply across scored, filtered, and mixed artifacts.
- Generated the current supply audit:
  - `results/commonvoice_pseudolabel_supply_audit.csv`
  - `results/commonvoice_pseudolabel_supply_audit.md`

Validation:

- `Validation`: The listening report builds from the latest
  `generation_manifest.jsonl` and detects the matching `mixed_teacher` metric
  CSVs.
- `Validation`: The listening report includes `110` generated rows across `11`
  source speakers and the rating template includes `99` styled rows plus a
  header.
- `Validation`: The CommonVoice supply audit reads the raw scored,
  emotion-filtered, hybrid-filtered, and mixed artifacts without rerunning any
  teacher model.
- `Validation`: Python compile checks passed for both new scripts.

Readout:

- Local CommonVoice data currently found at
  `/Users/steve/datasets/cv-corpus-21.0-2025-03-14-subset/en`.
- That subset has `1202` rows in `validated.tsv`, matching the current
  extracted/scored CommonVoice artifacts.
- The raw emotion2vec-scored artifact has only `anger=15` and `fear=9` top
  pseudo labels.
- The filtered emotion artifact selects only `anger=6` and `fear=4`.
- The hybrid extra-style artifact selects only `anger=5` and `fear=4`.
- The final mixed teacher artifact includes only `anger=4` and `fear=4`
  CommonVoice pseudo rows after speaker-first sampling.

Interpretation:

- The current local CommonVoice subset is too rare-class limited to justify
  another loss-weight or schedule-only experiment as the next main move.
- The best next model-side move is either:
  - extract/score a larger local CommonVoice subset to get real rare canonical
    emotion supply, or
  - move to decoder-aware/generated-audio style supervision that does not rely
    on the current tiny `anger` / `fear` pseudo-label pool.

Future upgrade to preserve:

- `[NOW]` Make every future generated corpus produce a listening HTML report and
  subjective-rating CSV before closeout.
- `[DONE]` Download/build an expanded local English CommonVoice corpus at
  `/Users/steve/datasets/cv-corpus-21.0-2025-03-14/en`; `/data` is not
  creatable in this macOS session because the root filesystem is read-only.
- `[DONE]` Extract OpenVoice embeddings from the expanded corpus into
  `embeddings/openvoice_commonvoice_cvrare_expanded_emb.pt`; validated shape is
  `25910 x 256 x 1`, with `13308` unique speakers and zero missing/unreadable
  clips.
- `[DONE]` Score/filter the expanded corpus and stop at the supply audit before
  training; the selected-label-preserving mixed artifact now keeps
  `anger=50` / `fear=50` CommonVoice pseudo rows.
- `[SOON]` Add optional sampled listening panels to compare multiple conditions
  side-by-side for the same speaker/style, so Joe can evaluate differences
  without opening several output folders.

---

### 0.30 CommonVoice Rare-Class Supply Preflight and Corpus Expansion (2026-05-05, branch `research/controllable-vae`)

What changed:

- Added `scripts/plan_commonvoice_rare_supply_expansion.py`, a reusable
  preflight gate for data-first rare-class expansion.
- Generated the current preflight reports:
  - `results/commonvoice_rare_supply_expansion_preflight.json`
  - `results/commonvoice_rare_supply_expansion_preflight.md`
- Downloaded/build an expanded local English CommonVoice corpus from
  Hugging Face `mteb/common_voice_21_0` validated assets:
  - `transcript/en/validated.tsv`
  - `audio/en/validated/en_validated_0.tar`
  - local corpus path:
    `/Users/steve/datasets/cv-corpus-21.0-2025-03-14/en`
- The script checks candidate local CommonVoice language directories for:
  - `validated.tsv`
  - `clips/`
  - usable validated rows with matching local clip files
  - usable speaker count
  - age / gender / accent metadata coverage
- The script uses the checked-in pseudo-label artifacts to estimate how many
  usable CommonVoice rows are needed before another rare-style run is credible.

Validation:

- `Validation`: `.venv/bin/python -m py_compile scripts/plan_commonvoice_rare_supply_expansion.py`
- `Validation`: `.venv/bin/python scripts/plan_commonvoice_rare_supply_expansion.py`
- `Validation`: the generated report scanned the intended `/data` path, the
  expanded local corpus path
  `/Users/steve/datasets/cv-corpus-21.0-2025-03-14/en`, and the older local
  subset
  `/Users/steve/datasets/cv-corpus-21.0-2025-03-14-subset/en`.
- `Validation`: the generated report loaded the checked-in pseudo-label
  artifacts and estimated the rare-row need from actual selected counts rather
  than a hand-waved target.
- `Validation`: the expanded corpus contains `40000` extracted MP3 clips and
  the preflight counts `40000` usable validated rows / `20537` usable speakers.

Readout:

- Literal `/data/cv-corpus-21.0-2025-03-14/en` is not creatable in this live
  macOS session because `/` is read-only.
- Stable usable corpus for this machine is now:
  `/Users/steve/datasets/cv-corpus-21.0-2025-03-14/en`.
- The expanded corpus preserves the full English metadata as
  `validated_full.tsv` (`1845370` lines including header), and its active
  `validated.tsv` is filtered to the first validated audio shard (`40000` MP3
  clips / `40001` TSV lines including header).
- The preflight counts `40000` usable validated rows and `20537` usable
  speakers, clearing the current row and speaker gate.
- The older local subset remains available with `1202` usable validated rows,
  `500` usable speakers, and `0` missing clip files.
- Based on selected rare-label rates in the current artifacts:
  - `anger` needs about `12020` rows before safety to reach `50` selected rows
  - `fear` needs about `15025` rows before safety to reach `50` selected rows
  - with the `1.5x` safety multiplier, the preflight recommends at least
    `22538` usable rows before extracting the next rare-supply artifact
- Current decision is `GO` for
  `/Users/steve/datasets/cv-corpus-21.0-2025-03-14/en`.

Interpretation:

- The next data-first model run can now start with OpenVoice extraction from
  the expanded local corpus.
- This is still not a license to train blindly: after extraction, stop at the
  pseudo-label supply audit unless selected `anger` and `fear` rows reach the
  target selected-row count.
- The older `1202`-row subset should no longer be used for rare-class
  expansion experiments.

Future upgrade to preserve:

- `[DONE]` Run the gated extraction -> scoring -> filtering -> hybrid-combine ->
  supply-audit commands from
  `results/commonvoice_rare_supply_expansion_preflight.md`; the final mixed
  artifact now clears the selected `anger` / `fear` gate.
- `[SOON]` If collaborators need a literal `/data/...` path on macOS, create a
  synthetic mount/symlink outside this live session; the current research-safe
  path is the user-space corpus above.
- `[SOON]` Add an optional `--min-selected-rare-rows` check to
  `scripts/audit_commonvoice_pseudolabel_supply.py` so the audit itself can
  fail CI/automation when rare labels remain undersupplied.

---

### 0.31 Expanded CommonVoice Extraction and Resumable Teacher Scoring (2026-05-05, branch `research/controllable-vae`)

What changed:

- Extracted OpenVoice embeddings from the expanded local English CommonVoice
  corpus:
  - corpus: `/Users/steve/datasets/cv-corpus-21.0-2025-03-14/en`
  - output: `embeddings/openvoice_commonvoice_cvrare_expanded_emb.pt`
  - command used the preflight-recommended speaker/clip caps:
    `--max-speakers 13308 --max-clips-per-speaker 3 --seed 42`
- Hardened `scripts/annotate_commonvoice_pseudolabels.py` so expanded
  emotion2vec scoring is practical:
  - added `--batch-size`
  - added resumable checkpointing via `--checkpoint-every`,
    `--checkpoint-path`, and automatic checkpoint resume
  - added `--no-resume` for clean smoke tests
  - added `--fail-on-error` plus per-row `pseudo_style_error` recording
  - added `--stop-when-accepted-targets`, e.g. `anger=50,fear=50`, so a
    rare-supply run can stop once enough high-confidence target rows exist

Validation:

- `Validation`: the expanded extractor completed and saved `25910` embeddings.
- `Validation`: extracted artifact loads with keys for `data`, `speaker_ids`,
  `clip_paths`, `age`, `gender`, `accent`, corpus metadata, and metadata
  coverage report.
- `Validation`: tensor shape is `(25910, 256, 1)` with dtype `float32`.
- `Validation`: sidecar list lengths match the tensor row count:
  `speaker_ids=25910`, `clip_paths=25910`, `age=25910`, `gender=25910`,
  `accent=25910`.
- `Validation`: extracted unique speaker count is `13308`.
- `Validation`: extraction skipped `0` missing clip files and `0` unreadable
  clip files.
- `Validation`: `.venv/bin/python -m py_compile scripts/annotate_commonvoice_pseudolabels.py`
- `Validation`: a 16-row scoring smoke test wrote checkpoint and final
  artifacts, annotated `16/16` rows, recorded `0` failures, and produced
  accepted pseudo-style counts.
- `Validation`: rerunning the same 16-row smoke command resumed from the
  checkpoint with `0` pending rows and did not recompute already-scored rows.
- `Validation`: a 64-row timing run with `--batch-size 16` completed and
  confirmed the full emotion2vec path remains multi-hour, which justifies
  checkpoint/resume and target-seeking scoring.

Readout:

- The expanded extraction step is now complete and reproducible from a local
  CommonVoice layout; this satisfies the data-side precondition for the
  rare-class supply experiment.
- Full emotion2vec scoring is compute-bound and should be treated as a
  resumable teacher job rather than an interactive smoke command.
- The active teacher command is target-seeking:

```bash
.venv/bin/python scripts/annotate_commonvoice_pseudolabels.py \
  --embeddings embeddings/openvoice_commonvoice_cvrare_expanded_emb.pt \
  --output embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_scored.pt \
  --save-style-score-map \
  --report-threshold 0.60 \
  --batch-size 4 \
  --checkpoint-every 500 \
  --stop-when-accepted-targets anger=50,fear=50
```

Interpretation:

- This is engineering/reproducibility evidence, not a new paper-facing result
  yet.
- Do **not** update `FINDINGS.md` until the scored/filter/audit chain verifies
  whether selected `anger` and `fear` rows actually reach the target.
- If the rare targets are met, proceed to filtering, hybrid teacher combining,
  supply audit, mixed-artifact construction, and only then a new model run.
- If the rare targets are not met, the next evidence-bearing move is either a
  larger CommonVoice audio shard or a different teacher/calibration strategy,
  not another loss-weight experiment on undersupplied rare rows.

Future upgrade to preserve:

- `[DONE]` Finish the target-seeking emotion2vec scoring job and inspect
  `embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_scored.pt`.
- `[DONE]` Run class-balanced filtering and the pseudo-label supply audit only
  after the scorer has enough annotated rows.
- `[SOON]` Add a documented resume/monitor command for long teacher-scoring
  jobs, since backgrounding is reaped in this Codex execution environment.
- `[SOON]` Consider another CommonVoice audio shard only if the expanded
  first-shard scorer still cannot reach `anger=50` and `fear=50` at the
  selected threshold.

---

### 0.32 Expanded CommonVoice Rare-Supply Filter, Hybrid Teacher, and Mixed Artifact (2026-05-05, branch `research/controllable-vae`)

What changed:

- Finished the target-seeking expanded CommonVoice emotion2vec scoring job:
  - input: `embeddings/openvoice_commonvoice_cvrare_expanded_emb.pt`
  - output: `embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_scored.pt`
  - checkpoint:
    `embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_scored.pt.checkpoint.pt`
  - annotated `6380/25910` rows before the configured rare-class stop condition
    was satisfied.
- Filtered the expanded emotion2vec teacher artifact with balanced canonical
  targets:
  - output: `embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_filtered.pt`
  - selected counts: `anger=50`, `disgust=80`, `fear=50`, `happy=80`,
    `neutral=120`, `sad=120`.
- Ran the latent-prototype teacher on all expanded CommonVoice embeddings and
  filtered the extra OpenVoice styles:
  - scored output:
    `embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_prototype.pt`
  - filtered output:
    `embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_prototype_filtered.pt`
  - selected extra-style counts: `confused=50`, `enunciated=50`,
    `whisper=50`.
- Combined emotion2vec canonical labels and prototype extra-style labels into
  the hybrid teacher artifact:
  - output:
    `embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_hybrid_extra_priority.pt`
  - selected counts: `anger=50`, `confused=50`, `disgust=79`,
    `enunciated=50`, `fear=50`, `happy=80`, `neutral=116`, `sad=120`,
    `whisper=50`.
- Added `--commonvoice-preserve-selected-pseudo` to
  `scripts/build_mixed_training_set.py` so speaker-first sampling can keep one
  clip per CommonVoice speaker while adding selected pseudo rows that the
  per-speaker cap would otherwise drop.
- Built the expanded rare-supply mixed artifact:
  - output: `embeddings/openvoice_mixed_teacher_cvrare_hybrid_extra_base.pt`
  - rows: `14195`
  - CommonVoice rows: `13370`
  - CommonVoice speakers: `13308`
  - CommonVoice pseudo-labeled rows: `645`
  - selected CommonVoice pseudo counts: `anger=50`, `confused=50`,
    `disgust=79`, `enunciated=50`, `fear=50`, `happy=80`, `neutral=116`,
    `sad=120`, `whisper=50`.
- Added deterministic inference support for the first expanded rare-supply
  checkpoint condition:
  - `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup`
  - checkpoint:
    `embeddings/openvoice_vae_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.pt`
- Regenerated the rare-supply audit:
  - `results/commonvoice_pseudolabel_supply_audit_rare_supply.csv`
  - `results/commonvoice_pseudolabel_supply_audit_rare_supply.md`

Validation:

- `Validation`: `.venv/bin/python -m py_compile scripts/build_mixed_training_set.py`
- `Validation`: the one-clip-per-speaker dry run without selected preservation
  dropped selected CommonVoice pseudo counts below target (`anger=49`,
  `fear=44`), confirming the need for an explicit preservation mode.
- `Validation`: the selected-preserving dry run kept `anger=50` and `fear=50`
  while adding only `62` rows beyond the compact one-clip speaker-breadth
  baseline.
- `Validation`: the canonical mixed artifact loads with shape `(14195, 256)`,
  `source_dataset`, `style_label_mask`, `style_label_confidence`,
  `style_label_row_weight`, and a populated `mixture_report`.
- `Validation`: the canonical mixed artifact reports
  `CommonVoice=13370`, `CREMA-D=546`, `Expresso=279`; label sources are
  `none=12725`, `pseudo=645`, `true=825`.
- `Validation`: the canonical mixed artifact preserves the rare-label gate:
  `commonvoice_selected_pseudo_style_counts['anger']=50` and
  `commonvoice_selected_pseudo_style_counts['fear']=50`.
- `Validation`: `results/commonvoice_pseudolabel_supply_audit_rare_supply.md`
  includes the scored, filtered, prototype-filtered, hybrid, and final mixed
  artifacts in one reproducible supply table.

Readout:

- The data-side bottleneck has moved: the project is no longer blocked by
  having only `4-5` active `anger` / `fear` CommonVoice rows in the mixed
  teacher artifact.
- Speaker breadth is still protected. We did not solve rare supply by simply
  moving to three clips per speaker and bloating the CommonVoice reconstruction
  pool; the selected-pseudo preservation mode adds only the selected rows lost
  to the one-clip cap.
- `disgust` and `neutral` are slightly below their original filtered targets in
  the hybrid artifact because prototype extra-style rows take priority on a few
  overlapping rows. That is acceptable for this run because the gate condition
  was rare canonical `anger` / `fear` supply plus extra-style coverage, not
  exact neutral/disgust caps.

Interpretation:

- This clears the training gate for an expanded rare-supply model, but it is
  still engineering/data-readiness evidence rather than a paper-facing finding.
- The paper-facing question is now testable: does fixing rare pseudo-label
  supply improve generated-audio emotion recall/generalization, or does the
  neutral decoding basin remain because the objective is still too latent-only?
- Do not update `FINDINGS.md` until the new checkpoint has generated audio,
  emotion recall, novelty, WER, MOS, collapse summaries, and a listening report.

Training command:

```bash
.venv/bin/python examples/openvoice_train_vae_mixed.py \
  --embeddings embeddings/openvoice_mixed_teacher_cvrare_hybrid_extra_base.pt \
  --output embeddings/openvoice_vae_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.pt \
  --schedule labeled_warmup \
  --schedule-epochs 1000 \
  --style-teacher-checkpoint embeddings/openvoice_vae_combined.pt \
  --style-teacher-weight 0.0 \
  --style-teacher-weight-final 0.25 \
  --style-teacher-datasets CommonVoice \
  --style-teacher-dims 0-8
```

Future upgrade to preserve:

- `[DONE]` Finish training
  `embeddings/openvoice_vae_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.pt`,
  run deterministic inference/evaluation, and produce a browser-playable
  listening report for perceptual review.
- `[SOON]` Since expanded rare supply improved recall but worsened WER/MOS,
  prioritize a decoder-aware/generated-audio style objective over another
  latent-only teacher-weight or schedule variant.
- `[SOON]` Add a selected-preserving speaker-sampling unit test or smoke test
  fixture so future builder changes cannot silently drop scarce selected pseudo
  rows behind the per-speaker cap.

---

### 0.33 Expanded Rare-Supply Teacher Checkpoint Evaluation (2026-05-05, branch `research/controllable-vae`)

What changed:

- Trained the first expanded rare-supply teacher checkpoint from
  `embeddings/openvoice_mixed_teacher_cvrare_hybrid_extra_base.pt`:
  - checkpoint:
    `embeddings/openvoice_vae_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.pt`
  - schedule: `labeled_warmup`
  - teacher: frozen combined VAE
    `embeddings/openvoice_vae_combined.pt`
  - teacher weight ramp: `0.0 -> 0.25`
  - teacher rows: `CommonVoice`
  - teacher dims: `0-8`
- Generated the standard deterministic 110-row evaluation corpus:
  - output:
    `output/mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_eval/`
  - manifest:
    `output/mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_eval/generation_manifest.jsonl`
  - source panel: `11` checked-in source speakers
  - style strength: `5.0`
  - noise level: `0.0`
  - seed: `42`
- Ran the full metric stack:
  - `results/eval_emotion_mixed_teacher_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.csv`
  - `results/eval_novelty_mixed_teacher_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.csv`
  - `results/eval_wer_mixed_teacher_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.csv`
  - `results/eval_mos_mixed_teacher_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.csv`
  - regenerated `results/eval_mixed_teacher_summary.csv`
  - regenerated `results/eval_mixed_teacher_collapse.csv`
- Built the browser-playable perceptual review bundle:
  - `results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.html`
  - `results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_ratings.csv`
- Ran style diagnostics for the expanded condition:
  - `results/eval_mixed_teacher_style_diagnostics_cvrare_labeled_warmup.csv`
  - `results/eval_mixed_teacher_style_diagnostics_cvrare_labeled_warmup.md`
- Fixed `examples/eval_wer.py` summary wording so the CLI reports all styled
  WER rows separately from the non-zero-only subset. The CSV schema is
  unchanged.

Validation:

- `Validation`: `scripts/run_ablation_inference.py` exposes
  `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup` as a deterministic
  condition mapped to the expanded rare-supply checkpoint.
- `Validation`: generation produced `110` manifest rows and the complete
  baseline + 9-style output family for each of the `11` source speakers.
- `Validation`: emotion evaluation scored `66` canonical emotion rows and
  reached `31/66 = 46.97%` recall.
- `Validation`: per-style canonical recall was `anger=1/11`,
  `disgust=2/11`, `fear=3/11`, `happy=5/11`, `neutral=10/11`,
  `sad=10/11`.
- `Validation`: novelty evaluation wrote `110` rows with mean novelty gain
  `0.2995`, above both the combined baseline (`0.2599`) and the previous
  style-distillation high (`0.0930`).
- `Validation`: WER evaluation wrote `110` rows; mean styled WER across all
  non-baseline style rows is `0.2751`, while the non-zero-only subset is
  `0.4778`.
- `Validation`: MOS evaluation wrote `110` rows; mean non-baseline MOS is
  `3.6771` and mean MOS delta vs same-speaker baseline is `-0.2640`.
- `Validation`: collapse summary improved sharply on style/identity collapse:
  content collapse `7`, style-to-neutral collapse `18`, identity collapse `1`,
  mixed collapse `1`, files with any collapse `25`.
- `Validation`: listening report and subjective-rating template were generated
  for perceptual review before closeout.

Readout:

- Fixing rare CommonVoice pseudo-label supply is not merely an engineering
  improvement. It produces the first large mixed-teacher recall jump:
  `18.2% -> 47.0%` versus the prior mixed-teacher ceiling.
- The recall improvement is broad enough to matter but uneven: `neutral` and
  `sad` are strong, `happy` improves materially, `fear` moves off zero, while
  `anger` and `disgust` remain weak and often still classify as neutral.
- Novelty also becomes paper-strong: `0.2995` exceeds the combined-only model's
  `0.2599`, meaning the expanded mixed teacher now moves speaker embeddings
  more than the original combined reference.
- The tradeoff is real. Intelligibility and naturalness degrade relative to the
  best mixed-teacher rows, especially for `sad`, `happy`, `fear`, and
  `enunciated`.
- The diagnostic report shows why this is still not solved: rare canonical
  classes now have enough rows, but teacher target-dim alignment remains weak
  for several emotion2vec classes. Expanded data fixes supply; it does not
  fully calibrate the teacher.

Paper story:

- This is a new paper-facing finding and has been added to `FINDINGS.md`.
- The positive claim is now stronger: broad speaker coverage plus selected
  rare pseudo-label preservation can push controllable emotion recall close to
  the `>50%` target Joe described.
- The limitation is also clearer: the method trades content/naturalness for
  stronger style movement, so the next research move should be decoder-aware
  style supervision or generated-audio feedback, not another latent-only
  scalar/mask/schedule tweak.

Future upgrade to preserve:

- `[NOW]` Design and test a decoder-aware or generated-audio style objective
  that keeps the expanded rare-supply recall gain while reducing WER/MOS
  damage, especially for `sad`, `happy`, `fear`, and `enunciated`.
- `[SOON]` Add per-style acceptance calibration for canonical pseudo labels,
  because expanded supply solved row scarcity but diagnostics still show weak
  teacher target-dim dominance for `anger`, `disgust`, `fear`, `happy`,
  `neutral`, and `sad`.
- `[SOON]` Add a manifest-driven all-metrics runner so the emotion, novelty,
  WER, MOS, summary, collapse, diagnostics, and listening report commands can
  be reproduced from one experiment spec.

---

### 0.34 Per-Style Strength Profiles for the Expanded Rare-Supply Checkpoint (2026-05-05, branch `research/controllable-vae`)

What changed:

- Added `--style-strength-map` to `scripts/run_ablation_inference.py` so
  deterministic ablation generation can use a JSON or inline per-style strength
  profile instead of one global strength for every style.
- Added profile configs:
  - `configs/style_strength_profiles/cvrare_content_guard.json`
  - `configs/style_strength_profiles/cvrare_sad_enunc_guard.json`
- Added `scripts/run_generated_audio_eval_suite.py`, a manifest-driven runner
  that executes emotion, novelty, WER, MOS, mixed-teacher summary/collapse, and
  listening-report generation for one generated corpus.
- Generated and evaluated two inference-calibrated corpora for
  `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup`:
  - `output/mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_content_guard_eval/`
  - `output/mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard_eval/`
- Built browser-playable listening reports:
  - `results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_content_guard.html`
  - `results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.html`

Validation:

- `Validation`: `.venv/bin/python -m py_compile scripts/run_ablation_inference.py scripts/run_generated_audio_eval_suite.py`
- `Validation`: `scripts/run_ablation_inference.py --help` exposes
  `--style-strength-map` and still lists the expanded rare-supply condition.
- `Validation`: `scripts/run_generated_audio_eval_suite.py --help` documents
  the generated-audio evaluation wrapper.
- `Validation`: `cvrare_content_guard` generated `110` manifest rows with
  `sad=3.5`, `enunciated=2.5`, `fear=4.0`, `happy=4.0`, `confused=4.0`, and
  other listed strengths at `5.0`.
- `Validation`: `cvrare_sad_enunc_guard` generated `110` manifest rows with
  `sad=3.5`, `enunciated=2.5`, `confused=4.0`, and the canonical emotions
  otherwise at `5.0`.
- `Validation`: the generated-audio eval suite wrote emotion, novelty, WER,
  MOS, summary/collapse, listening HTML, and subjective-rating CSV artifacts
  for both profiles.

Result matrix:

| Condition | Recall | Novelty gain | Mean styled WER | MOS delta | Content collapse | Style-to-neutral collapse | Identity collapse | Mixed collapse | Files with any collapse |
|-----------|--------|--------------|-----------------|-----------|------------------|---------------------------|-------------------|----------------|-------------------------|
| `combined` | `25.8%` | `0.2599` | `0.2353` | `-0.0792` | `6` | `37` | `7` | `4` | `46` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup` | `47.0%` | `0.2995` | `0.2751` | `-0.2640` | `7` | `18` | `1` | `1` | `25` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_content_guard` | `40.9%` | `0.2653` | `0.2133` | `-0.1989` | `1` | `24` | `1` | `2` | `24` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `47.0%` | `0.2726` | `0.2348` | `-0.2081` | `2` | `18` | `1` | `1` | `20` |

Readout:

- The broad `content_guard` profile repairs WER/MOS the most, but it lowers
  canonical recall from `47.0%` to `40.9%` by weakening `fear` and `happy`.
- The narrower `sad/enunciated` guard is the better current Pareto point: it
  preserves `47.0%` recall, keeps novelty above the combined baseline
  (`0.2726` vs `0.2599`), improves mean WER to roughly the combined baseline
  level (`0.2348` vs `0.2353`), improves MOS delta relative to the unguarded
  expanded run, and reduces files with any collapse from `25` to `20`.
- This is an inference-side calibration result, not a replacement for the
  decoder-aware/generated-audio training objective.

Paper story:

- Added as a paper-facing finding because the result is generated-audio
  verified and changes the interpretation of the expanded rare-supply
  checkpoint: some of the quality cost can be reduced without giving up recall.
- The recommended listening artifact for the expanded rare-supply model is now
  `results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.html`.
- The unguarded expanded run remains the strongest novelty setting; the
  `sad/enunciated` guard is the stronger quality-balanced setting.

Future upgrade to preserve:

- `[NOW]` Design and test the decoder-aware/generated-audio style objective;
  the per-style profile is a useful stopgap, but a paper-ready training method
  should learn this tradeoff rather than depend on manual inference calibration.
- `[SOON]` Add a small per-style strength grid/optimizer over the expanded
  checkpoint and log the search policy, so style profiles are reproducible
  experiment outputs instead of hand-authored presets.
- `[SOON]` Run subjective listening review on the unguarded and
  `sad/enunciated` guard reports before freezing which profile appears in the
  paper demo table.

---

### 0.35 Decoder-Prototype Training Pilot for Expanded Rare-Supply Checkpoint (2026-05-05, branch `research/controllable-vae`)

What changed:

- Added a decoder-prototype loss to `dpvc.utils.train_mixed_autoencoder`.
  During training, the model encodes a row, applies an inference-like style
  control in latent space, decodes that controlled latent, and matches the
  decoded embedding to the real labeled style prototype for the target style.
- Extended `examples/openvoice_train_vae_mixed.py` with decoder-prototype CLI
  flags:
  - `--decoder-prototype-weight`
  - `--decoder-prototype-weight-final`
  - `--decoder-prototype-datasets`
  - `--decoder-prototype-source`
  - `--decoder-prototype-min-count`
  - `--decoder-prototype-strength`
  - `--decoder-prototype-style-strengths`
  - `--decoder-prototype-control-mode`
- Added the deterministic inference condition
  `mixed_teacher_cvrare_decoder_proto_labeled_warmup` to
  `scripts/run_ablation_inference.py`.
- Trained the first pilot from the current best expanded checkpoint:
  `embeddings/openvoice_vae_mixed_teacher_cvrare_decoder_proto_labeled_warmup.pt`.
- Generated and evaluated:
  - `output/mixed_teacher_cvrare_decoder_proto_labeled_warmup_eval/`
  - `results/eval_emotion_mixed_teacher_mixed_teacher_cvrare_decoder_proto_labeled_warmup.csv`
  - `results/eval_novelty_mixed_teacher_mixed_teacher_cvrare_decoder_proto_labeled_warmup.csv`
  - `results/eval_wer_mixed_teacher_mixed_teacher_cvrare_decoder_proto_labeled_warmup.csv`
  - `results/eval_mos_mixed_teacher_mixed_teacher_cvrare_decoder_proto_labeled_warmup.csv`
  - `results/listening_mixed_teacher_cvrare_decoder_proto_labeled_warmup.html`

Training command:

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

Evaluation command:

```bash
.venv/bin/python scripts/run_ablation_inference.py \
    --source-dir examples/source_speakers/ \
    --condition mixed_teacher_cvrare_decoder_proto_labeled_warmup \
    --out output/mixed_teacher_cvrare_decoder_proto_labeled_warmup_eval \
    --style-strength 5.0 \
    --noise-level 0.0 \
    --seed 42

.venv/bin/python scripts/run_generated_audio_eval_suite.py \
    --input output/mixed_teacher_cvrare_decoder_proto_labeled_warmup_eval \
    --result-tag mixed_teacher_cvrare_decoder_proto_labeled_warmup \
    --input-tag mixed_teacher
```

Validation:

- `Validation`: `.venv/bin/python -m py_compile examples/openvoice_train_vae_mixed.py dpvc/utils.py scripts/run_ablation_inference.py scripts/run_generated_audio_eval_suite.py`
- `Validation`: `examples/openvoice_train_vae_mixed.py --help` exposes the
  decoder-prototype flags.
- `Validation`: a 2-epoch smoke run built true-label prototypes for all nine
  styles and applied decoder-prototype loss to CommonVoice rows.
- `Validation`: a 12-epoch scale smoke from the expanded checkpoint showed the
  ramped teacher and decoder-prototype loss terms active without dominating the
  training loss.
- `Validation`: the 1000-epoch pilot completed, saved the checkpoint, produced
  a 110-row generated-audio manifest, and the generated-audio eval suite wrote
  emotion, novelty, WER, MOS, summary/collapse, listening HTML, and
  subjective-rating CSV artifacts.

Result matrix:

| Condition | Recall | Novelty gain | Mean styled WER | MOS delta | Content collapse | Style-to-neutral collapse | Identity collapse | Mixed collapse | Files with any collapse |
|-----------|--------|--------------|-----------------|-----------|------------------|---------------------------|-------------------|----------------|-------------------------|
| `combined` | `25.8%` | `0.2599` | `0.2353` | `-0.0792` | `6` | `37` | `7` | `4` | `46` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup` | `47.0%` | `0.2995` | `0.2751` | `-0.2640` | `7` | `18` | `1` | `1` | `25` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `47.0%` | `0.2726` | `0.2348` | `-0.2081` | `2` | `18` | `1` | `1` | `20` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `42.4%` | `0.3008` | `0.2863` | `-0.2148` | `4` | `22` | `1` | `0` | `27` |

Readout:

- The decoder-prototype path is implemented and reproducible, so the training
  side of the decoder-aware hypothesis is no longer just a plan.
- The first naive version is not a new reference. It keeps novelty high
  (`0.3008`) and improves recall over `combined` (`42.4%` vs `25.8%`), but it
  loses recall versus the expanded rare-supply checkpoint and gives back WER /
  collapse versus the `sad/enunciated` guard.
- Per-style recall suggests the objective helps `happy` (`6/11`) and keeps
  `sad` strong (`10/11`), but hurts `fear` (`1/11`) and `disgust` (`0/11`).
- Paper interpretation: decoded-embedding prototype matching is a useful
  cautionary baseline. It is not enough by itself to learn the manual
  quality/recall repair found by the inference-side guard.

Future upgrade to preserve:

- `[DONE]` Try a lower-weight decoder-prototype variant before abandoning the
  simple weight family; the `0.005` run improves novelty slightly but still
  misses the current guard on recall/WER/collapse.
- `[NOW]` Move to generated-audio failure mining before the next
  decoder-aware training run, then decide whether true-labeled-only
  application or a canonical-emotion subset is warranted.
- `[SOON]` Add an offline generated-audio teacher loop: generate a small corpus,
  score it with emotion2vec/WER/MOS, and use those failures to calibrate target
  rows or style strengths rather than matching only decoded embedding
  prototypes.
- `[DONE]` Evaluate the decoder-prototype checkpoint with
  `configs/style_strength_profiles/cvrare_sad_enunc_guard.json`; the guard
  repairs WER/MOS but not recall/collapse, so the next work should change the
  training objective rather than add more inference variants on this checkpoint.

---

### 0.36 Guarded Decoder-Prototype Readout (2026-05-05, branch `research/controllable-vae`)

What changed:

- Reused the first decoder-prototype checkpoint and applied the existing
  `configs/style_strength_profiles/cvrare_sad_enunc_guard.json` inference
  profile.
- Generated:
  `output/mixed_teacher_cvrare_decoder_proto_labeled_warmup_sad_enunc_guard_eval/`
  with `110` manifest rows.
- Evaluated the full generated-audio metric stack and built:
  - `results/eval_emotion_mixed_teacher_mixed_teacher_cvrare_decoder_proto_labeled_warmup_sad_enunc_guard.csv`
  - `results/eval_novelty_mixed_teacher_mixed_teacher_cvrare_decoder_proto_labeled_warmup_sad_enunc_guard.csv`
  - `results/eval_wer_mixed_teacher_mixed_teacher_cvrare_decoder_proto_labeled_warmup_sad_enunc_guard.csv`
  - `results/eval_mos_mixed_teacher_mixed_teacher_cvrare_decoder_proto_labeled_warmup_sad_enunc_guard.csv`
  - `results/listening_mixed_teacher_cvrare_decoder_proto_labeled_warmup_sad_enunc_guard.html`

Commands:

```bash
.venv/bin/python scripts/run_ablation_inference.py \
    --source-dir examples/source_speakers/ \
    --condition mixed_teacher_cvrare_decoder_proto_labeled_warmup \
    --out output/mixed_teacher_cvrare_decoder_proto_labeled_warmup_sad_enunc_guard_eval \
    --style-strength 5.0 \
    --style-strength-map configs/style_strength_profiles/cvrare_sad_enunc_guard.json \
    --noise-level 0.0 \
    --seed 42

.venv/bin/python scripts/run_generated_audio_eval_suite.py \
    --input output/mixed_teacher_cvrare_decoder_proto_labeled_warmup_sad_enunc_guard_eval \
    --result-tag mixed_teacher_cvrare_decoder_proto_labeled_warmup_sad_enunc_guard \
    --input-tag mixed_teacher
```

Validation:

- `Validation`: deterministic generation completed with `110` manifest rows.
- `Validation`: the generated-audio eval suite wrote emotion, novelty, WER,
  MOS, summary/collapse, listening HTML, and subjective-rating CSV artifacts.
- `Validation`: the guarded decoder-prototype summary row is present in
  `results/eval_mixed_teacher_summary.csv`.

Result matrix:

| Condition | Recall | Novelty gain | Mean styled WER | MOS delta | Content collapse | Style-to-neutral collapse | Identity collapse | Mixed collapse | Files with any collapse |
|-----------|--------|--------------|-----------------|-----------|------------------|---------------------------|-------------------|----------------|-------------------------|
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `47.0%` | `0.2726` | `0.2348` | `-0.2081` | `2` | `18` | `1` | `1` | `20` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `42.4%` | `0.3008` | `0.2863` | `-0.2148` | `4` | `22` | `1` | `0` | `27` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup_sad_enunc_guard` | `42.4%` | `0.2718` | `0.2592` | `-0.1787` | `4` | `23` | `2` | `1` | `28` |

Readout:

- The inference guard improves the decoder-prototype checkpoint's WER
  (`0.2863 -> 0.2592`) and MOS delta (`-0.2148 -> -0.1787`).
- The same guard does not recover emotion recall (`42.4%`) and slightly
  worsens collapse files (`27 -> 28`).
- Compared with the current reference
  `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard`,
  the guarded decoder-prototype checkpoint has lower recall, lower novelty,
  worse WER, and more collapse files, though it has a better predicted-MOS
  delta.
- This separates the failure modes cleanly: inference calibration can repair
  some naturalness/content damage, but the decoder-prototype training objective
  still does not learn the recall/quality Pareto point.

Future upgrade to preserve:

- `[NOW]` Move to a safer generated-audio-calibrated objective rather than
  more inference variants on this checkpoint.
- `[SOON]` Keep the guarded decoder-prototype listening report for perceptual
  review, because predicted MOS improves even though aggregate recall/WER do
  not beat the current reference.

---

### 0.37 Low-Weight Decoder-Prototype Pilot (2026-05-05, branch `research/controllable-vae`)

What changed:

- Added the deterministic inference condition
  `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` to
  `scripts/run_ablation_inference.py`.
- Trained a lower-risk decoder-prototype variant from the same expanded
  rare-supply checkpoint, reducing the final decoder-prototype weight from
  `0.02` to `0.005`.
- Generated and evaluated:
  - `output/mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup_eval/`
  - `results/eval_emotion_mixed_teacher_mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup.csv`
  - `results/eval_novelty_mixed_teacher_mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup.csv`
  - `results/eval_wer_mixed_teacher_mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup.csv`
  - `results/eval_mos_mixed_teacher_mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup.csv`
  - `results/listening_mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup.html`

Training command:

```bash
.venv/bin/python examples/openvoice_train_vae_mixed.py \
    --embeddings embeddings/openvoice_mixed_teacher_cvrare_hybrid_extra_base.pt \
    --output embeddings/openvoice_vae_mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup.pt \
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
    --decoder-prototype-weight-final 0.005 \
    --decoder-prototype-datasets CommonVoice \
    --decoder-prototype-source true \
    --decoder-prototype-strength 5.0 \
    --decoder-prototype-style-strengths sad=3.5,enunciated=2.5,confused=4.0 \
    --decoder-prototype-control-mode target_only
```

Evaluation command:

```bash
.venv/bin/python scripts/run_ablation_inference.py \
    --source-dir examples/source_speakers/ \
    --condition mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup \
    --out output/mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup_eval \
    --style-strength 5.0 \
    --noise-level 0.0 \
    --seed 42

.venv/bin/python scripts/run_generated_audio_eval_suite.py \
    --input output/mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup_eval \
    --result-tag mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup \
    --input-tag mixed_teacher
```

Validation:

- `Validation`: `.venv/bin/python -m py_compile examples/openvoice_train_vae_mixed.py dpvc/utils.py scripts/run_ablation_inference.py scripts/run_generated_audio_eval_suite.py`
- `Validation`: `.venv/bin/python scripts/run_ablation_inference.py --help | rg "mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup|style-strength-map"` confirms the condition and style-strength-map option are exposed.
- `Validation`: the 1000-epoch lower-weight pilot completed and saved
  `embeddings/openvoice_vae_mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup.pt`.
- `Validation`: deterministic generation completed with `110` manifest rows.
- `Validation`: the generated-audio eval suite wrote emotion, novelty, WER,
  MOS, summary/collapse, listening HTML, and subjective-rating CSV artifacts.
- `Validation`: `results/eval_mixed_teacher_summary.csv` now contains the
  lower-weight decoder-prototype row.
- `Validation`: `git diff --check`

Result matrix:

| Condition | Recall | Novelty gain | Mean styled WER | MOS delta | Content collapse | Style-to-neutral collapse | Identity collapse | Mixed collapse | Files with any collapse |
|-----------|--------|--------------|-----------------|-----------|------------------|---------------------------|-------------------|----------------|-------------------------|
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `47.0%` | `0.2726` | `0.2348` | `-0.2081` | `2` | `18` | `1` | `1` | `20` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `42.4%` | `0.3008` | `0.2863` | `-0.2148` | `4` | `22` | `1` | `0` | `27` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup_sad_enunc_guard` | `42.4%` | `0.2718` | `0.2592` | `-0.1787` | `4` | `23` | `2` | `1` | `28` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `42.4%` | `0.3032` | `0.2782` | `-0.2122` | `3` | `22` | `1` | `0` | `26` |

Readout:

- Lowering the decoder-prototype weight from `0.02` to `0.005` improves
  novelty slightly and reduces content / any-collapse a little.
- It does not recover the missing recall: the condition remains at `42.4%`,
  below the current `47.0%` quality-balanced reference.
- It also does not repair WER/MOS enough: WER remains `0.2782`, worse than the
  current guard's `0.2348`, and MOS delta remains slightly worse than the
  guard's `-0.2081`.
- Paper interpretation: the simple decoder-prototype family is now a stronger
  negative result. The next high-value work is generated-audio failure mining,
  not another scalar prototype-weight sweep.

Future upgrade to preserve:

- `[NOW]` Build a generated-audio calibration artifact that joins manifest
  rows with emotion2vec predictions, WER, MOS, novelty, and collapse labels,
  then identifies target-style failures for the next objective.
- `[SOON]` Use that failure-mining artifact to decide whether the next
  decoder-aware run should be true-labeled-only, style-specific, or filtered
  by generated-audio success/failure rather than by decoded embedding distance
  alone.

---

### 0.38 Generated-Audio Failure Mining Artifact (2026-05-05, branch `research/controllable-vae`)

What changed:

- Added `scripts/analyze_generated_audio_failures.py`.
- Joined the current best expanded rare-supply guard and the decoder-prototype
  family across emotion, novelty, WER, MOS, and collapse taxonomy outputs.
- Wrote:
  - `results/eval_mixed_teacher_generated_audio_failure_mining.csv`
  - `results/eval_mixed_teacher_generated_audio_failure_mining.md`

Command:

```bash
.venv/bin/python scripts/analyze_generated_audio_failures.py
```

Validation:

- `Validation`: the script joined all four compared conditions without missing
  metric CSVs.
- `Validation`: the output CSV contains row-level failure labels for `396`
  styled rows (`4` conditions x `99` styled generated files).
- `Validation`: the Markdown readout reports condition-level, style-level,
  failure-mode, and highest-priority-row tables.

Compared conditions:

| Condition | Recall | Novelty gain | Mean styled WER | MOS delta | Files with any collapse | Any-failure rows | Mean failure score |
|-----------|--------|--------------|-----------------|-----------|-------------------------|------------------|--------------------|
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `47.0%` | `0.2726` | `0.2348` | `-0.2081` | `20` | `58/99` | `1.9899` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `42.4%` | `0.3008` | `0.2863` | `-0.2148` | `27` | `65/99` | `2.2929` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup_sad_enunc_guard` | `42.4%` | `0.2718` | `0.2592` | `-0.1787` | `28` | `61/99` | `2.2929` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `42.4%` | `0.3032` | `0.2782` | `-0.2122` | `26` | `66/99` | `2.2525` |

Readout:

- The current `sad/enunciated` inference guard remains the best row-level
  reference: it has the lowest any-failure count and mean failure score.
- Decoder-prototype variants do not change the hard failure pattern enough:
  their emotion misses remain at `38`, versus `35` for the current guard.
- Persistent failures concentrate in:
  - `disgust`: `43/44` rows fail across compared conditions, mean emotion
    recall `0.0455`
  - `fear`: `39/44` rows fail, mean emotion recall `0.1364`
  - `anger`: `37/44` rows fail, mean emotion recall `0.1591`
  - `enunciated`: `29/44` rows fail, mostly quality/MOS rather than
    emotion-recall failure

Training implication:

- The next objective should not be another global decoder-prototype weight
  sweep.
- Best next design: focus on generated-audio rows with `emotion_miss` plus
  `style_to_neutral` for `anger` / `disgust` / `fear`, while excluding
  high-WER or very-low-MOS rows from direct positive style targets unless the
  objective explicitly repairs content.

Future upgrade to preserve:

- `[NOW]` Add a failure-conditioned training-target selector that can emit a
  small target file from `results/eval_mixed_teacher_generated_audio_failure_mining.csv`.
- `[DONE]` Use that selector to run one targeted objective against the clean
  `anger`/`disgust` rows, then evaluate against the current guard rather than
  against the weaker decoder-prototype rows.
- `[SOON]` Convert the next target objective into an anti-neutral or
  generated-audio-calibrated loss; the selector found clean rows, but the
  first target-dim teacher follow-up still increased neutral collapse.

---

### 0.39 Failure-Conditioned Target Selector (2026-05-06, branch `research/controllable-vae`)

What changed:

- Added `scripts/select_failure_conditioned_targets.py`.
- Converted the generated-audio failure-mining CSV into conservative target
  decisions for the current hard canonical styles.
- Wrote:
  - `results/eval_mixed_teacher_failure_conditioned_targets.csv`
  - `results/eval_mixed_teacher_failure_conditioned_targets.json`
  - `results/eval_mixed_teacher_failure_conditioned_targets.md`

Command:

```bash
.venv/bin/python scripts/select_failure_conditioned_targets.py
```

Selection rule:

- Target styles: `anger`, `disgust`, `fear`
- Required modes: `emotion_miss`, `style_to_neutral`
- Exclude modes: `content_collapse`, `high_wer`, `low_mos_delta`,
  `identity_collapse`, `mixed_collapse`, `low_novelty`
- Ready threshold: at least `3` clean selected rows in the reference condition
  `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard`

Validation:

- `Validation`: `.venv/bin/python -m py_compile scripts/select_failure_conditioned_targets.py scripts/analyze_generated_audio_failures.py examples/openvoice_train_vae_mixed.py dpvc/utils.py`
- `Validation`: `.venv/bin/python scripts/select_failure_conditioned_targets.py --help`
- `Validation`: selector ran against
  `results/eval_mixed_teacher_generated_audio_failure_mining.csv` and wrote
  CSV/JSON/Markdown outputs.
- `Validation`: reference readout has `33` target rows and `11` selected clean
  targets.
- `Validation`: rerun to `/private/tmp` produced `132` target-style decision
  rows, `45` selected rows across all compared conditions, ready styles
  `anger`/`disgust`, and blocked style `fear`.
- `Validation`: generated JSON includes trainer-ready argument strings for
  style-teacher and decoder-prototype follow-ups.
- `Validation`: `git diff --check`

Reference readout:

| Style | Target rows | Selected clean targets | Status |
|-------|-------------|------------------------|--------|
| `anger` | `11` | `5` | ready |
| `disgust` | `11` | `6` | ready |
| `fear` | `11` | `0` | blocked |

Recommended first follow-up:

```bash
--style-teacher-target-mode target_dim \
--style-teacher-require-label \
--style-teacher-style-weights anger=3,confused=0,disgust=3,enunciated=0,fear=0,happy=0,neutral=0,sad=0,whisper=0
```

Readout:

- `anger` and `disgust` have enough clean style-control failures to justify a
  conservative positive target objective.
- `fear` is still a hard style, but it is blocked for this objective because
  the current reference has no clean fear rows after excluding content,
  naturalness, identity, mixed-collapse, and low-novelty confounds.
- The next experiment should not globally increase all style losses. It should
  apply target-dim teacher supervision only to the ready styles and compare
  directly against the current `sad/enunciated` guard.

Future upgrade to preserve:

- `[DONE]` Train
  `mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` from
  the current expanded rare-supply checkpoint using the selector's
  `anger`/`disgust` target-dim style-teacher weights.
- `[SOON]` Add a separate fear diagnostic that distinguishes fear-as-style
  failure from content/naturalness failure before using fear rows as positive
  style targets.
- `[SOON]` Test an explicit anti-neutral generated-audio objective for
  `anger`/`disgust`; the target selector alone is not sufficient if the
  training loss can satisfy style-teacher pressure while still decoding to
  neutral-classified audio.

---

### 0.40 Failure-Conditioned Style-Teacher Follow-Up (2026-05-06, branch `research/controllable-vae`)

What changed:

- Added the ablation condition
  `mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` to
  `scripts/run_ablation_inference.py`.
- Trained a conservative target-dim style-teacher follow-up from the expanded
  rare-supply labeled-warmup checkpoint.
- Used only selector-ready styles as positive targets:
  `anger=3`, `disgust=3`, every other style weight `0`, and `fear=0` because
  the selector blocked it.
- Generated the deterministic 110-row audio panel and evaluated emotion,
  novelty, WER, MOS proxy, collapse taxonomy, and listening artifacts.
- Re-ran generated-audio failure mining with the targeted condition included.

Training command:

```bash
.venv/bin/python examples/openvoice_train_vae_mixed.py \
  --embeddings embeddings/openvoice_mixed_teacher_cvrare_hybrid_extra_base.pt \
  --output embeddings/openvoice_vae_mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup.pt \
  --init-checkpoint embeddings/openvoice_vae_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.pt \
  --epochs 1000 \
  --schedule labeled_warmup \
  --schedule-epochs 1000 \
  --style-teacher-checkpoint embeddings/openvoice_vae_combined.pt \
  --style-teacher-weight 0.0 \
  --style-teacher-weight-final 0.25 \
  --style-teacher-datasets CommonVoice \
  --style-teacher-dims 0-8 \
  --style-teacher-target-mode target_dim \
  --style-teacher-require-label \
  --style-teacher-style-weights anger=3,confused=0,disgust=3,enunciated=0,fear=0,happy=0,neutral=0,sad=0,whisper=0
```

Generation and evaluation commands:

```bash
.venv/bin/python scripts/run_ablation_inference.py \
  --source-dir examples/source_speakers/ \
  --condition mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup \
  --out output/mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup_eval \
  --style-strength 5.0 \
  --noise-level 0.0 \
  --seed 42

.venv/bin/python scripts/run_generated_audio_eval_suite.py \
  --input output/mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup_eval \
  --result-tag mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup \
  --input-tag mixed_teacher

.venv/bin/python scripts/analyze_generated_audio_failures.py \
  --conditions \
    mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard \
    mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup \
    mixed_teacher_cvrare_decoder_proto_labeled_warmup \
    mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup
```

Validation:

- `Validation`: training loaded `14195 x 256` embeddings, including `13370`
  CommonVoice, `546` CREMA-D, and `279` Expresso rows.
- `Validation`: training loaded
  `embeddings/openvoice_vae_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.pt`
  as the init checkpoint and
  `embeddings/openvoice_vae_combined.pt` as the style teacher.
- `Validation`: the style-teacher target mode was `target_dim`,
  `--style-teacher-require-label` was active, and only `129/14195` rows were
  used for the weighted teacher-style term.
- `Validation`: generated audio wrote a `110`-row manifest to
  `output/mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup_eval/generation_manifest.jsonl`.
- `Validation`: the generated-audio eval suite wrote emotion, novelty, WER,
  MOS, collapse, summary, listening HTML, and rating-template outputs.
- `Validation`: failure mining was re-run and now compares the current guard,
  the failure-targeted follow-up, and the two decoder-prototype baselines.
- `Validation`: `.venv/bin/python -m py_compile examples/openvoice_train_vae_mixed.py dpvc/utils.py scripts/run_ablation_inference.py scripts/run_generated_audio_eval_suite.py scripts/analyze_generated_audio_failures.py scripts/select_failure_conditioned_targets.py`
- `Validation`: `.venv/bin/python scripts/run_ablation_inference.py --help | rg 'mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup|style-strength-map'`
- `Validation`: `git diff --check`

Comparison:

| Condition | Recall | Novelty gain | Mean styled WER | MOS delta | Content collapse | Style-to-neutral collapse | Identity collapse | Mixed collapse | Files with any collapse |
|-----------|--------|--------------|-----------------|-----------|------------------|---------------------------|-------------------|----------------|-------------------------|
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `47.0%` | `0.2726` | `0.2348` | `-0.2081` | `2` | `18` | `1` | `1` | `20` |
| `mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` | `39.4%` | `0.2960` | `0.2651` | `-0.2072` | `1` | `26` | `1` | `0` | `28` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `42.4%` | `0.3008` | `0.2863` | `-0.2148` | `4` | `22` | `1` | `0` | `27` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `42.4%` | `0.3032` | `0.2782` | `-0.2122` | `3` | `22` | `1` | `0` | `26` |

Per-style recall for the targeted follow-up:

| Style | Recall |
|-------|--------|
| `anger` | `2/11` |
| `disgust` | `0/11` |
| `fear` | `1/11` |
| `happy` | `5/11` |
| `neutral` | `9/11` |
| `sad` | `9/11` |

Readout:

- This is a validated negative result. The failure-conditioned target-dim
  teacher objective does not replace the current `sad/enunciated` guard.
- It slightly reduces content collapse versus the current guard (`1` vs `2`)
  and keeps MOS delta nearly identical (`-0.2072` vs `-0.2081`), but that is
  not enough because recall drops from `47.0%` to `39.4%`.
- The main regression is style-to-neutral collapse: `26` files versus `18` for
  the guard. The selected `anger`/`disgust` target rows were clean, but the
  target-dim teacher loss still did not force decoded audio out of the neutral
  attractor.
- The next objective should not be another scalar teacher-weight sweep. It
  needs an explicit anti-neutral, generated-audio-calibrated, or
  contrastive-output signal tied to observed decoder behavior.

Listening artifacts:

- `results/listening_mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup.html`
- `results/listening_mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup_ratings.csv`

Future upgrade to preserve:

- `[DONE]` Test a first anti-neutral objective for `anger`/`disgust`; the
  prototype-margin run is logged in section 0.41 and shows that
  embedding-space anti-neutral margins are still too indirect.
- `[SOON]` Run a small style-strength grid for the expanded rare-supply
  checkpoint so the current hand-authored `sad/enunciated` guard becomes a
  reproducible calibration result rather than a hidden manual profile.
- `[SOON]` Keep fear separate until a diagnostic can distinguish fear-style
  failure from content/MOS failure; the selector blocked fear correctly.

---

### 0.41 Anti-Neutral Prototype-Margin Objective (2026-05-06, branch `research/controllable-vae`)

What changed:

- Added anti-neutral training support to `dpvc.utils.train_mixed_autoencoder`.
- Added anti-neutral CLI controls to
  `examples/openvoice_train_vae_mixed.py`, including:
  `--anti-neutral-weight`, `--anti-neutral-weight-final`,
  `--anti-neutral-mode`, `--anti-neutral-datasets`,
  `--anti-neutral-styles`, `--anti-neutral-style-weights`,
  `--anti-neutral-margin`, `--anti-neutral-strength`,
  `--anti-neutral-style-strengths`, and `--anti-neutral-control-mode`.
- Added the deterministic ablation condition
  `mixed_teacher_cvrare_antineutral_labeled_warmup` to
  `scripts/run_ablation_inference.py`.
- Tested two anti-neutral proxies:
  - `teacher_margin`: decoded controlled embeddings are re-encoded by the
    frozen combined VAE and penalized if target style score does not beat
    neutral.
  - `prototype_margin`: decoded controlled embeddings are penalized if they
    are not closer to the target style prototype than to the neutral prototype.
- Rejected `teacher_margin` after smoke diagnostics because it produced zero
  loss on the selected CommonVoice `anger`/`disgust` rows even though
  generated audio still collapsed toward neutral.
- Trained and evaluated the stronger `prototype_margin` variant from the
  expanded rare-supply labeled-warmup checkpoint.

Training command:

```bash
.venv/bin/python examples/openvoice_train_vae_mixed.py \
  --embeddings embeddings/openvoice_mixed_teacher_cvrare_hybrid_extra_base.pt \
  --output embeddings/openvoice_vae_mixed_teacher_cvrare_antineutral_labeled_warmup.pt \
  --init-checkpoint embeddings/openvoice_vae_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.pt \
  --epochs 1000 \
  --schedule labeled_warmup \
  --schedule-epochs 1000 \
  --anti-neutral-weight 0.0 \
  --anti-neutral-weight-final 0.02 \
  --anti-neutral-mode prototype_margin \
  --anti-neutral-datasets CommonVoice \
  --anti-neutral-styles anger,disgust \
  --anti-neutral-style-weights anger=3,disgust=3 \
  --anti-neutral-margin 10.0 \
  --anti-neutral-strength 5.0 \
  --decoder-prototype-source true \
  --decoder-prototype-min-count 5
```

Generation and evaluation commands:

```bash
.venv/bin/python scripts/run_ablation_inference.py \
  --source-dir examples/source_speakers/ \
  --condition mixed_teacher_cvrare_antineutral_labeled_warmup \
  --out output/mixed_teacher_cvrare_antineutral_labeled_warmup_eval \
  --style-strength 5.0 \
  --noise-level 0.0 \
  --seed 42

.venv/bin/python scripts/run_generated_audio_eval_suite.py \
  --input output/mixed_teacher_cvrare_antineutral_labeled_warmup_eval \
  --result-tag mixed_teacher_cvrare_antineutral_labeled_warmup \
  --input-tag mixed_teacher

.venv/bin/python scripts/analyze_generated_audio_failures.py \
  --conditions \
    mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard \
    mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup \
    mixed_teacher_cvrare_antineutral_labeled_warmup \
    mixed_teacher_cvrare_decoder_proto_labeled_warmup \
    mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup
```

Validation:

- `Validation`: `py_compile` passed for
  `examples/openvoice_train_vae_mixed.py`, `dpvc/utils.py`, and
  `scripts/run_ablation_inference.py` before training.
- `Validation`: the training CLI help exposes `--anti-neutral-mode`,
  `teacher_margin`, and `prototype_margin`.
- `Validation`: the ablation inference CLI exposes
  `mixed_teacher_cvrare_antineutral_labeled_warmup`.
- `Validation`: `teacher_margin` smoke diagnostics activated the intended
  selected rows but produced `0.00` anti-neutral loss, so it was rejected as an
  uncalibrated proxy.
- `Validation`: the `prototype_margin` preflight found `129` selected
  CommonVoice `anger`/`disgust` rows and showed margin `10.0` produced
  `65/129` violations before training, making it a real optimization signal.
- `Validation`: training loaded `14195 x 256` embeddings, including `13370`
  CommonVoice, `546` CREMA-D, and `279` Expresso rows, then saved
  `embeddings/openvoice_vae_mixed_teacher_cvrare_antineutral_labeled_warmup.pt`.
- `Validation`: generated audio wrote the expected `110`-row manifest to
  `output/mixed_teacher_cvrare_antineutral_labeled_warmup_eval/generation_manifest.jsonl`.
- `Validation`: the generated-audio eval suite wrote emotion, novelty, WER,
  MOS, summary/collapse, listening HTML, and rating-template outputs.
- `Validation`: generated-audio failure mining was rerun across five
  conditions, now including the anti-neutral prototype-margin follow-up.

Comparison:

| Condition | Recall | Novelty gain | Mean styled WER | MOS delta | Content collapse | Style-to-neutral collapse | Identity collapse | Mixed collapse | Files with any collapse |
|-----------|--------|--------------|-----------------|-----------|------------------|---------------------------|-------------------|----------------|-------------------------|
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `47.0%` | `0.2726` | `0.2348` | `-0.2081` | `2` | `18` | `1` | `1` | `20` |
| `mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` | `39.4%` | `0.2960` | `0.2651` | `-0.2072` | `1` | `26` | `1` | `0` | `28` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `40.9%` | `0.2962` | `0.2609` | `-0.2065` | `1` | `25` | `2` | `1` | `27` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `42.4%` | `0.3008` | `0.2863` | `-0.2148` | `4` | `22` | `1` | `0` | `27` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `42.4%` | `0.3032` | `0.2782` | `-0.2122` | `3` | `22` | `1` | `0` | `26` |

Per-style recall for the anti-neutral prototype-margin follow-up:

| Style | Recall |
|-------|--------|
| `anger` | `2/11` |
| `disgust` | `0/11` |
| `fear` | `0/11` |
| `happy` | `5/11` |
| `neutral` | `10/11` |
| `sad` | `10/11` |

Readout:

- This is a validated negative result. The anti-neutral prototype-margin proxy
  does not replace the current `sad/enunciated` guard.
- It slightly improves over the failure-conditioned target-dim run, but still
  loses to the guard on recall (`40.9%` vs `47.0%`), WER (`0.2609` vs
  `0.2348`), style-to-neutral collapse (`25` vs `18`), and any-collapse files
  (`27` vs `20`).
- The result is important because it rules out another tempting
  embedding-space shortcut: decoded embeddings can be closer to target
  prototypes than neutral prototypes without producing generated audio that
  emotion2vec recognizes as the target style.
- The next intervention should use actual generated-audio behavior as the
  calibration signal, starting with a reproducible style-strength grid or
  generated-audio reranking loop before turning that signal into a training
  objective.

Listening artifacts:

- `results/listening_mixed_teacher_cvrare_antineutral_labeled_warmup.html`
- `results/listening_mixed_teacher_cvrare_antineutral_labeled_warmup_ratings.csv`

Future upgrade to preserve:

- `[DONE]` Build a true generated-audio-calibrated grid/reranking artifact over
  hard styles (`anger`, `disgust`, `fear`) and current reference checkpoints;
  completed in section 0.42 with actual generated-audio emotion recall, WER,
  MOS, novelty, collapse labels, and listening reports.
- `[SOON]` If the grid finds a consistently better audio-level choice, convert
  it into a training-time objective or selection rule; do not promote another
  embedding-only proxy unless it predicts generated-audio metrics.
- `[SOON]` Keep the failed `teacher_margin` diagnostic in mind: frozen-teacher
  margin satisfaction in decoded embedding space was not calibrated to
  generated-audio emotion recognition.

---

### 0.42 Generated-Audio Style-Strength Grid / Reranking (2026-05-06, branch `research/controllable-vae`)

What changed:

- Added `--styles` to `scripts/run_ablation_inference.py` so a deterministic
  corpus can generate only the target style under test while still keeping the
  matched baseline rows.
- Fixed `scripts/run_generated_audio_eval_suite.py` so non-default
  `--input-tag` runs write their own summary/collapse files instead of
  overwriting `results/eval_mixed_teacher_summary.csv`.
- Added `scripts/run_style_strength_grid.py`, a one-command generated-audio
  grid runner for style subsets and strength values.
- Added `scripts/summarize_style_strength_grid.py`, which ranks each style by
  generated-audio metrics: recall first, then fewer collapse rows, lower WER,
  higher MOS delta, and novelty.
- Ran the first narrow grid over the hard styles identified by failure mining:
  `anger`, `disgust`, and `fear` at strengths `3.0`, `5.0`, `7.5`, and `10.0`.

Command:

```bash
.venv/bin/python scripts/run_style_strength_grid.py \
  --source-dir examples/source_speakers/ \
  --styles anger,disgust,fear \
  --strengths 3.0,5.0,7.5,10.0
```

Validation:

- `Validation`: `py_compile` passed for
  `scripts/run_ablation_inference.py`,
  `scripts/run_generated_audio_eval_suite.py`,
  `scripts/run_style_strength_grid.py`, and
  `scripts/summarize_style_strength_grid.py`.
- `Validation`: CLI help checks passed for the new `--styles`,
  `--style-strength-map`, grid-runner, and grid-summarizer interfaces.
- `Validation`: a smoke generation run with `--styles anger` wrote the
  expected 2-row manifest for one source file.
- `Validation`: the full grid produced `12` corpora, each with a `22`-row
  manifest (`11` baselines + `11` styled outputs).
- `Validation`: every grid cell wrote emotion, novelty, WER, MOS, listening
  HTML, and rating-template artifacts.
- `Validation`: the summary/ranking files were generated:
  `results/eval_mixed_teacher_strength_grid_summary.csv`,
  `results/eval_mixed_teacher_strength_grid_collapse.csv`,
  `results/eval_mixed_teacher_cvrare_strength_grid_ranking.csv`, and
  `results/eval_mixed_teacher_cvrare_strength_grid_ranking.md`.
- `Validation`: `git diff --check` passed before commit.

Best per-style grid cells:

| Style | Best strength | Recall | Novelty gain | Mean styled WER | MOS delta | Style-to-neutral | Any collapse | Readout |
|-------|---------------|--------|--------------|-----------------|-----------|------------------|--------------|---------|
| `anger` | `10.0` | `3/11` | `0.3302` | `0.2081` | `-0.0408` | `7` | `7` | Improves target recall over the guard's `1/11` anger row with modest WER cost |
| `disgust` | `10.0` | `2/11` | `0.3124` | `0.2831` | `-0.6039` | `8` | `8` | Does not improve recall over the guard and badly hurts MOS; not a safe preset |
| `fear` | `7.5` | `6/11` | `0.4136` | `0.5231` | `-0.2972` | `1` | `3` | Doubles target recall over the guard's `3/11` fear row, but WER is high |

Important comparison note:

- The grid cells are **single-style targeted corpora**, so their recall values
  should not be compared directly to the 110-row overall recall of the current
  `sad/enunciated` guard.
- Against the guard on the same style rows:
  - `anger_s10` improves recall (`1/11 -> 3/11`) and novelty
    (`0.2962 -> 0.3302`), while WER rises moderately (`0.1740 -> 0.2081`).
  - `disgust_s10` ties recall (`2/11 -> 2/11`) but worsens WER
    (`0.1529 -> 0.2831`) and MOS delta (`-0.2315 -> -0.6039`).
  - `fear_s7p5` improves recall (`3/11 -> 6/11`) and novelty
    (`0.3485 -> 0.4136`), but WER worsens sharply (`0.3071 -> 0.5231`).

Readout:

- This is the first true generated-audio-calibrated reranking artifact after
  the failure-conditioned and anti-neutral proxy negative results.
- It confirms that style strength can recover some hard-style target labels,
  especially `anger` and `fear`, but the gains are not uniformly quality-safe.
- `disgust` remains a hard failure: the ranked high-strength cell improves
  novelty but not target recall, and it damages MOS enough that it should not
  become a default.
- The next move should be perceptual review of the best cells before adding
  style-specific presets or turning the grid into training supervision.

Listening artifacts:

- `results/listening_mixed_teacher_cvrare_strength_grid_anger_s10.html`
- `results/listening_mixed_teacher_cvrare_strength_grid_disgust_s10.html`
- `results/listening_mixed_teacher_cvrare_strength_grid_fear_s7p5.html`
- reference: `results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.html`

Future upgrade to preserve:

- `[NOW]` Perceptually review the top-ranked grid cells against the current
  guard before promoting any style-specific inference preset.
- `[DONE]` Add a combined listening dashboard for the top grid candidates; see
  section 0.43 and
  `results/listening_mixed_teacher_cvrare_strength_grid_ab_review.html`.
- `[SOON]` Expand the grid only where perceptual review supports it; candidate
  next axes are speaker-specific reranking and style-specific strength maps,
  not another global strength sweep.
- `[SOON]` If a grid cell is perceptually strong and metric-stable, convert it
  into a checked-in style-strength profile or training target; otherwise keep
  it as diagnostic evidence.

---

### 0.43 Style-Strength A/B Perceptual Review Dashboard (2026-05-06, branch `research/controllable-vae`)

What changed:

- Added `scripts/build_style_grid_review.py`, which builds a single A/B
  listening dashboard from a reference manifest plus one grid candidate per
  style.
- Generated a matched perceptual-review artifact for the current guard versus
  the top-ranked grid candidates:
  - reference: `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard`
  - candidate `anger`: `mixed_teacher_cvrare_strength_grid_anger_s10`
  - candidate `disgust`: `mixed_teacher_cvrare_strength_grid_disgust_s10`
  - candidate `fear`: `mixed_teacher_cvrare_strength_grid_fear_s7p5`
- The dashboard places source, baseline, reference guard, candidate, and the
  objective metrics side by side for the same source/style row.

Command:

```bash
.venv/bin/python scripts/build_style_grid_review.py
```

Validation:

- `Validation`: `scripts/build_style_grid_review.py` compiled with
  `py_compile`.
- `Validation`: CLI help exposes `--reference-manifest`, `--reference-tag`,
  `--candidate`, `--candidate-input-tag`, and `--rating-template`.
- `Validation`: the default run produced `33` matched A/B pairs:
  `11` sources x `3` candidate styles.
- `Validation`: the HTML uses repo-relative audio links when served from the
  repo root, so it works with `python -m http.server 8000`.
- `Validation`: the rating template has one row per A/B pair and columns for
  preference, target match, intelligibility, naturalness, identity shift, and
  notes.
- `Validation`: `git diff --check` passed before commit.

Artifacts:

- `results/listening_mixed_teacher_cvrare_strength_grid_ab_review.html`
- `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_ratings.csv`

FINDINGS.md review:

- Reviewed after building the perceptual dashboard. No new paper-facing
  finding was added because this is a review interface, not a completed human
  perceptual result. Finding 35 remains the current evidence statement until
  the rating template is filled and summarized.

Future upgrade to preserve:

- `[NOW]` Fill or collect ratings in
  `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_ratings.csv`
  before promoting any candidate preset.
- `[DONE]` Add a small summarizer/triage script for the A/B ratings CSV; see
  section 0.44 and
  `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority.md`.
- `[SOON]` If `anger_s10` or `fear_s7p5` wins perceptually, create a
  candidate style-strength profile; keep `disgust_s10` diagnostic unless
  listening contradicts the MOS warning.

---

### 0.44 Objective-Assisted A/B Review Triage (2026-05-06, branch `research/controllable-vae`)

What changed:

- Added `scripts/summarize_style_grid_review.py`, which summarizes the A/B
  review dashboard before and after human ratings are filled.
- Generated an objective-assisted priority sheet that classifies each matched
  guard-vs-candidate pair as:
  - `clean_target_gain`
  - `target_gain_quality_risk`
  - `novelty_gain_no_recall_gain`
  - `metric_trap`
  - `candidate_regression`
  - `tie_or_minor_change`
- The script also reads
  `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_ratings.csv`
  and will summarize human preferences once ratings are entered.

Command:

```bash
.venv/bin/python scripts/summarize_style_grid_review.py
```

Validation:

- `Validation`: `scripts/summarize_style_grid_review.py` compiled with
  `py_compile`.
- `Validation`: CLI help exposes `--ratings`, `--out-csv`, `--candidate`, and
  the reference/candidate input controls.
- `Validation`: default run summarized `33` matched A/B pairs and wrote both
  CSV and Markdown outputs.
- `Validation`: the script correctly reports that no filled human ratings are
  present yet, so this is triage rather than a perceptual result.
- `Validation`: `git diff --check` passed before commit.

Artifacts:

- `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority.csv`
- `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority.md`

Triage result:

- Priority rows to listen first: `5`.
- Clean target gains: `2` rows (`anger` / `cremad_1006`,
  `fear` / `male_1_cremad_1003`).
- Target gains with quality risk: `3` rows (`anger` /
  `female_1_cremad_1002`, `fear` / `cremad_1003`, and `fear` /
  `female_1_cremad_1002`).
- `disgust_s10` has no target-gain rows; it is mostly novelty gain without
  recall gain or metric-trap behavior.

FINDINGS.md review:

- Reviewed after triage generation. No new finding was added because this is
  objective-assisted review prioritization. It supports Finding 35, but does
  not replace perceptual scoring.

Future upgrade to preserve:

- `[NOW]` Listen to the five priority rows first in
  `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority.html`.
- `[NOW]` Fill the ratings CSV for those rows, then rerun
  `scripts/summarize_style_grid_review.py` to generate a human-preference
  summary.
- `[SOON]` If the filled ratings support `anger_s10` or `fear_s7p5`, add a
  candidate style-strength profile and rerun the generated-audio eval suite.

---

### 0.45 Priority-Only Style-Strength A/B Review Dashboard (2026-05-08, branch `research/controllable-vae`)

What changed:

- Extended `scripts/build_style_grid_review.py` with optional
  `--priority-csv` and `--max-priority` filters.
- Preserved the default 33-row dashboard behavior; priority filtering is
  opt-in and keyed by `(style, source_stem)` from the triage CSV.
- Generated a five-row dashboard and rating template for the first perceptual
  review pass:
  - two clean target-gain rows
  - three target-gain rows with quality risk

Command:

```bash
.venv/bin/python scripts/build_style_grid_review.py \
  --priority-csv results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority.csv \
  --max-priority 2 \
  --out results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority.html \
  --rating-template results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority_ratings.csv \
  --title "Generated-Audio Strength Grid Priority A/B Review"
```

Validation:

- `Validation`: the priority dashboard build produced exactly `5` matched A/B
  pairs from the `priority <= 2` rows.
- `Validation`: the default dashboard build still produced `33` matched A/B
  pairs in a `/tmp` smoke run, so full-review behavior was preserved.
- `Validation`: `scripts/build_style_grid_review.py` compiled with
  `py_compile`.
- `Validation`: CLI help exposes `--priority-csv`, `--max-priority`, and
  `--rating-template`.
- `Validation`: `scripts/summarize_style_grid_review.py --ratings` accepts
  the five-row priority ratings template in a `/tmp` smoke run.
- `Validation`: `git diff --check` passed before commit.

Artifacts:

- `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority.html`
- `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority_ratings.csv`

FINDINGS.md review:

- Updated Finding 35 after Joe's first priority A/B listening review. The
  perceptual result is negative for preset promotion: `0/5` candidate wins,
  `4/5` ties/indistinguishable, and `1/5` reference preference.

Future upgrade to preserve:

- `[DONE]` Joe listened to the priority-only dashboard and provided qualitative
  ratings in Teams. Encoded results are in
  `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority_ratings_joe_2026-05-19.csv`.
- `[DONE]` Reran `scripts/summarize_style_grid_review.py` with Joe's filled
  ratings. Summary is in
  `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority_joe_2026-05-19.md`.
- `[NOW]` Because none of the five priority rows won perceptually, keep the
  style-strength grid as a diagnostic artifact and do not promote `anger_s10`
  or `fear_s7p5` as checked-in style profiles yet.
- `[SOON]` If we return to generated-audio calibration later, focus on
  content/naturalness repair or broader listener panels rather than simply
  increasing style strength.

---

### 0.46 May 14 Joe Meeting Debrief and Direction Update (2026-05-14, branch `research/controllable-vae`)

Context:

- Reviewed the current mixed-data / generated-audio calibration status with
  Joe and Ivoline.
- A durable post-meeting debrief is saved at
  `MEETING_DEBRIEF_JOE_2026-05-14.md`.

What Joe clarified:

- Emotion control does not need to be perfect or world-class; the broader
  claim is controllable speaker generation / anonymization with multiple
  controllable attributes.
- Strong, perceptually obvious styles such as `whisper` and possibly `anger`
  are enough to carry the demonstration if the outputs sound convincing.
- CommonVoice age/gender metadata should be incorporated next, because age and
  gender are additional speaker attributes with labels already present in the
  broad corpus.
- The phrase `identity collapse` needs to be explained more carefully; Joe
  reasonably heard `collapse` as possibly meaning garbage audio, while our
  current metric means low OpenVoice novelty gain versus baseline.
- If the five-row listening review and age/gender controls look acceptable,
  the project may be close to paper-writing mode: architecture, training data,
  and evaluation strategy should be documented clearly.

Important correction preserved for future meetings:

- `content_collapse`: high WER / poor intelligibility.
- `style_collapse_to_neutral`: non-neutral target predicted as neutral by the
  emotion model.
- `identity_collapse_to_baseline`: low novelty gain versus baseline in
  OpenVoice speaker-embedding space; this is not WER.
- `mixed_collapse`: multiple collapse axes on the same generated file.

Communication feedback preserved:

- Start future updates with the top-line result before experiment details.
- Avoid saying pseudo-label filtering is `cherry picking` or `hope for the
  best`; call it a confidence-gated weak-label selection step with auditable
  accepted/rejected counts.
- Say `emotion2vec_plus_large` for the pseudo-label teacher/evaluator, not
  `EmoVoice classifier`.
- Separate content preservation, style control, identity/novelty, and
  anonymization modes explicitly.
- Every future Joe-facing meeting brief should include anticipated questions
  and concise answers.

Updated near-term task sequence:

- `[DONE]` Joe completed the five-row priority listening review in Teams; no
  CSV transfer was needed after encoding his message into the ratings artifact.
- `[NOW]` Add a plain-English metric/collapse guide for Joe and future paper
  readers.
- `[NOW]` Design and implement the first CommonVoice metadata-controls
  experiment for age and gender.
- `[SOON]` Add a filtered-vs-looser-pseudo-label ablation so Joe's question
  about whether filtering is actually necessary becomes an empirical result.
- `[SOON]` Start paper-method documentation covering architecture, data
  mixture, training schedule, and evaluation justification.
- `[SOON]` Package generated-audio review artifacts so collaborators do not
  depend on Steve-local output directories.

FINDINGS.md review:

- No new paper-facing finding was added from the meeting alone. The next
  FINDINGS update should wait for either filled perceptual ratings or an
  evaluated age/gender-control result.

---

### 0.47 Joe Priority A/B Listening Review Result (2026-05-24, branch `research/controllable-vae`)

Context:

- Joe could not send back the CSV file through Teams, but he returned the five
  row judgments in message form.
- I encoded his response into a separate ratings file so the blank template
  remains reusable.

Artifacts:

- `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority_ratings_joe_2026-05-19.csv`
- `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority_joe_2026-05-19.csv`
- `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority_joe_2026-05-19.md`

Perceptual result:

- Row 1, `anger / cremad_1006`, `anger_s10`: tie; Joe said the reference and
  candidate sounded identical.
- Row 2, `fear / male_1_cremad_1003`, `fear_s7p5`: reference preferred; Joe
  heard an unnatural pitch change in the candidate, though the difference was
  small.
- Row 3, `anger / female_1_cremad_1002`, `anger_s10`: tie.
- Row 4, `fear / cremad_1003`, `fear_s7p5`: tie.
- Row 5, `fear / female_1_cremad_1002`, `fear_s7p5`: tie.

Interpretation:

- Objective metric gains did not translate into a perceptual win in Joe's
  first five-row review.
- Do not promote `anger_s10` or `fear_s7p5` as checked-in presets yet.
- Treat the grid as a useful diagnostic for where objective metrics and human
  perception diverge.
- At that time, the next practical research step was CommonVoice age/gender
  controls and the metric/collapse guide, rather than another style-strength
  increase. That follow-up is now complete through the first diagnostic
  metadata-control panel; see sections 0.49-0.51.

Validation:

- `Validation`: Joe's five Teams judgments were mapped to the exact priority
  dashboard row order.
- `Validation`: the Joe-specific ratings CSV has the expected `16` columns on
  all rows.
- `Validation`: `scripts/summarize_style_grid_review.py` produced a human
  ratings summary with `5` filled rows, `4` ties, `1` reference preference, and
  `0` candidate wins.

---

### 0.48 Canonical Evidence/Demo Packet (2026-05-26, branch `research/evidence-demo-packet`)

Goal:

- Consolidate the substantial current result into one reviewable packet before
  starting CommonVoice age/gender controls.
- Make it easy to explain that Joe's latest priority A/B review rejected the
  stronger strength-grid candidates, not the whole controllable speaker system.

Artifacts:

- `EVIDENCE_DEMO_PACKET.md`
- `docs/metric_collapse_guide.md`
- `results/listening_evidence_demo_index.html`

What changed:

- Added a root-level evidence packet that separates:
  - current quality-balanced demo evidence,
  - high-recall/high-novelty scientific tradeoff evidence,
  - Joe's negative perceptual gate on the metric-selected strength candidates,
  - historical evidence that OpenVoice + CREMA-D/Expresso produced perceptibly
    distinct controls.
- Added a browser-playable listening index with a quick-listen panel for two
  source speakers and useful current controls (`anger`, `happy`, `sad`,
  `whisper`, plus one cleaner `fear` row).
- Added a plain-English metric/collapse guide that defines style recall,
  WER/content collapse, MOS/naturalness, novelty/identity collapse, and mixed
  collapse.

Interpretation:

- The current project has a substantial result: controllable speaker
  generation/anonymization with audible style controls and a strong mixed-data
  quantitative result around `47%` emotion recall.
- The newest high-strength `anger_s10` / `fear_s7p5` candidates should not be
  promoted as presets because Joe did not hear a perceptual candidate win.
- The next research branch at that point was CommonVoice age/gender controls.
  That branch is now implemented through a first diagnostic panel; it did not
  yet produce perceptible age/gender control, so the current next move is paper
  methods/evidence consolidation.

Validation:

- `Validation`: the listening index references existing local source/audio
  files and existing result dashboards.
- `Validation`: the metric guide preserves the corrected collapse taxonomy
  from the May 14 Joe debrief.
- `Validation`: no new paper-facing finding was added because this packet
  consolidates existing evidence rather than producing a new experiment.

Next:

- `[DONE]` Start `research/commonvoice-metadata-controls` from the canonical
  research line and audit CommonVoice age/gender metadata coverage before
  training.

---

### 0.49 CommonVoice Metadata Controls Audit (2026-05-27, branch `research/commonvoice-metadata-controls`)

Goal:

- Start the age/gender-control branch by verifying whether the local
  CommonVoice corpus and extracted OpenVoice artifact actually contain enough
  metadata for supervised controls.

Artifacts:

- `scripts/audit_commonvoice_metadata_controls.py`
- `results/commonvoice_metadata_controls_audit.md`
- `results/commonvoice_metadata_controls_audit.json`
- `IMPLEMENTATION_PLAN_commonvoice-metadata-controls.md`

Audit result:

- Local corpus: `/Users/steve/datasets/cv-corpus-21.0-2025-03-14/en`
- Local clips: `40000`
- Local validated rows: `40000`
- Unique local speakers: `20537`
- Age-control rows: `5504` (`13.8%`)
- Gender-control rows: `5291` (`13.2%`)
- Rows with both age and gender controls: `5258` (`13.1%`)
- Extracted OpenVoice artifact rows: `25910`
- Extracted age-control rows: `3566` (`13.8%`)
- Extracted gender-control rows: `3429` (`13.2%`)
- Extracted rows with both age and gender controls: `3403` (`13.1%`)

Design implication:

- The metadata is usable but sparse and imbalanced.
- First-pass controls should be conservative scalar controls:
  - style dims `0-8`
  - `dim_9`: binary gender scalar (`female_feminine/female=-1`,
    `male_masculine/male=1`)
  - `dim_10`: ordinal age scalar
  - free dims `11-14`
- Missing metadata must be handled with masks; missing age/gender rows should
  still contribute reconstruction and style supervision, not false metadata
  targets.

Validation:

- `Validation`: the audit script ran against the local CommonVoice corpus and
  existing `embeddings/openvoice_commonvoice_cvrare_expanded_emb.pt` artifact.
- `Validation`: the script normalizes CommonVoice gender values
  `male_masculine` / `female_feminine` into binary scalar controls.
- `Validation`: audit reports were written to Markdown and JSON for later
  reproducibility.

Next:

- `[DONE]` Extend `scripts/build_mixed_training_set.py` to preserve per-row
  metadata scalars/masks in mixed artifacts.
- `[DONE]` Extend mixed VAE training with masked direct metadata-control loss.
- `[NOW]` Generate a small age/gender listening/evaluation panel after the
  first smoke checkpoint.

---

### 0.50 CommonVoice Metadata-Control Plumbing (2026-05-27, branch `research/commonvoice-metadata-controls`)

Goal:

- Turn the age/gender audit into runnable metadata-control infrastructure
  without claiming that age/gender control works perceptually yet.

Implementation:

- Extended `scripts/build_mixed_training_set.py` so mixed artifacts now preserve
  per-row CommonVoice metadata controls:
  - `metadata_gender_scalar`
  - `metadata_gender_mask`
  - `metadata_age_ordinal_scalar`
  - `metadata_age_mask`
  - raw metadata lists and `metadata_control_report`
- Extended `dpvc.utils.train_mixed_autoencoder` with a masked direct
  metadata-control loss over selected latent dimensions.
- Extended `examples/openvoice_train_vae_mixed.py` with:
  - `--metadata-control-weight`
  - `--metadata-gender-dim`
  - `--metadata-age-dim`
  - `--metadata-control-report`
- Extended `examples/openvoice_infer_controllable.py` with:
  - `--gender-control female|male`
  - `--gender-control-dim`
  - `--age-control teens|twenties|...|nineties`
  - `--age-control-dim`
- Added `latent_dims` metadata to `dpvc.VariationalAutoencoder` so utility
  validation can reject out-of-range metadata-control dims cleanly.
- Built the full cvrare metadata-ready mixed artifact:
  - `embeddings/openvoice_mixed_teacher_cvrare_hybrid_extra_metadata_base.pt`
  - rows: `14195`
  - age-control rows: `1889`
  - gender-control rows: `1795`
  - rows with both controls: `1780`
- Trained the first conservative metadata-control checkpoint:
  - `embeddings/openvoice_vae_mixed_teacher_cvrare_metadata_w010_labeled_warmup.pt`
  - initialized from `embeddings/openvoice_vae_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.pt`
  - metadata weight: `0.1`
  - style-teacher weight: `0.25`
  - dims: gender `9`, age `10`
  - report: `results/openvoice_vae_mixed_teacher_cvrare_metadata_w010_labeled_warmup_report.json`
- Generated the first metadata-control perceptual smoke panel:
  - `output/metadata_control_panel_w010/`
  - `results/listening_metadata_w010_labeled_warmup.html`
  - `results/listening_metadata_w010_labeled_warmup_ratings.csv`
  - `results/listening_metadata_w010_labeled_warmup.md`

Validation:

- `Validation`: `.venv/bin/python -m py_compile scripts/build_mixed_training_set.py dpvc/utils.py dpvc/model_embedding_vae.py examples/openvoice_train_vae_mixed.py examples/openvoice_infer_controllable.py scripts/audit_commonvoice_metadata_controls.py`
- `Validation`: `.venv/bin/python examples/openvoice_train_vae_mixed.py --help | rg "metadata-control|metadata-gender|metadata-age"` exposes the training flags.
- `Validation`: `.venv/bin/python examples/openvoice_infer_controllable.py --help | rg "gender-control|age-control"` exposes the inference flags.
- `Validation`: a small `/private/tmp/openvoice_mixed_metadata_smoke.pt`
  artifact built successfully from the cvrare CommonVoice artifact plus
  CREMA-D and Expresso, and contained metadata scalar/mask tensors.
- `Validation`: a one-epoch smoke training run with
  `--metadata-control-weight 0.1` completed and wrote
  `/private/tmp/openvoice_vae_metadata_smoke.pt` plus
  `/private/tmp/openvoice_metadata_train_smoke.json`.
- `Validation`: full metadata-ready artifact rebuild completed with
  `14195` rows and non-empty metadata masks (`1889` age rows, `1795` gender
  rows).
- `Validation`: first 1000-epoch metadata-control checkpoint training completed
  and wrote the checkpoint plus JSON training report.
- `Validation`: metadata-control inference smoke generated a single
  `happy/male/forties` file successfully.
- `Validation`: the 10-row listening panel generated successfully and the HTML
  contains `12` valid audio references including source clips.
- `Validation`: local perceptual review on 2026-05-28 found that the metadata
  variants sounded identical or mostly like generic speaker/timbre shifts, not
  interpretable age/gender control.

Interpretation:

- This is an engineering/reproducibility result plus a useful negative
  diagnostic, not a paper-facing positive finding. `FINDINGS.md` stays
  unchanged.
- The first checkpoint proves the end-to-end age/gender-control path can train,
  generate audio, and expose review artifacts. It does not yet show perceptible
  age/gender control.
- Because the perceptual gate failed, WER/MOS/novelty should not be the next
  move for this checkpoint. Those metrics would mainly characterize a generic
  timbre/identity shift rather than validate useful metadata control.

Next:

- `[DONE]` Shift back to paper-method documentation for the substantial current
  style-control result and describe metadata control as future work / diagnostic
  infrastructure.
- `[SOON]` Probe whether OpenVoice speaker embeddings encode recoverable
  age/gender before spending more training on metadata controls.
- `[SOON]` Retry metadata control only with a balanced subset, stronger
  prototype/classifier supervision, or a verified metadata-readout objective.
- `[SOON]` Add a fairness/ethics note before any external-facing age/gender
  claims; CommonVoice labels are self-reported, sparse, and imbalanced.

---

### 0.51 Paper Methods and Evidence Consolidation (2026-05-28, branch `docs/paper-methods-and-evidence`)

Goal:

- Pause exploratory model training long enough to make the current substantial
  result paper-readable and collaborator-readable.

Artifacts:

- `PAPER_METHODS_AND_EVIDENCE.md`
- `IMPLEMENTATION_PLAN_paper-methods-and-evidence.md`
- `EVIDENCE_DEMO_PACKET.md`
- `docs/metric_collapse_guide.md`
- `README.md`

What changed:

- Added a paper-facing methods and evidence packet covering:
  - problem framing as controllable speaker generation / anonymization;
  - OpenVoice as the active controllable path;
  - latent layout and style dimensions;
  - CREMA-D / Expresso / CommonVoice data mixture;
  - pseudo-label and teacher-supervision strategy;
  - current evaluation stack;
  - claim-to-evidence mapping;
  - current non-claims;
  - Joe-facing anticipated Q&A.
- Refreshed the evidence/demo packet so it no longer says the next step is
  first-pass age/gender controls. That branch has now been implemented and
  perceptually failed its first gate.
- Updated the metric/collapse guide with a paper-readiness rule: metrics can
  nominate candidates, but listening decides whether a candidate becomes a
  demo or paper claim.
- Updated the README to point collaborators to the new paper-method packet and
  to mark CommonVoice age/gender controls as diagnostic infrastructure, not a
  current positive claim.

Interpretation:

- The current paper backbone is the expanded rare-supply mixed teacher plus
  the `cvrare_sad_enunc_guard` inference profile.
- The generated-audio strength grid is diagnostic because Joe heard `0/5`
  candidate wins in the priority A/B review.
- The first CommonVoice age/gender control path is diagnostic because local
  listening heard identical outputs or generic speaker/timbre shifts.
- No `FINDINGS.md` update was made because this branch consolidates existing
  evidence rather than producing a new verified experiment.

Validation:

- `Validation`: `PAPER_METHODS_AND_EVIDENCE.md` includes a claim-to-evidence
  table and the required Joe-facing Q&A.
- `Validation`: docs preserve the canonical listening entrypoint
  `results/listening_evidence_demo_index.html`.
- `Validation`: docs do not promote `anger_s10`, `fear_s7p5`, or first-pass
  age/gender controls as paper/demo wins.
- `Validation`: the docs-only 0.51 commit left `FINDINGS.md` unchanged; the
  later 0.52 external-verifier run adds Finding 36.

Next:

- `[DONE]` Add an external speaker-verifier / EER-style novelty validation
  branch so identity-shift evidence does not depend only on native OpenVoice
  embedding-space novelty.
- `[DONE]` Add a metadata separability probe before more age/gender training.
- `[SOON]` Build a generated-audio/content-repair loop for hard styles
  (`anger`, `disgust`, `fear`).
- `[SOON]` Add repeated-seed confidence intervals before final tables.
- `[SOON]` Add formal DP accounting and privacy-utility curves before paper
  submission.

---

### 0.52 External Speaker-Verifier Novelty Validation (2026-05-28, branch `research/external-speaker-verifier`)

Goal:

- Add an external speaker-verifier check so identity-shift evidence does not
  depend only on OpenVoice's native embedding space.

Artifacts:

- `scripts/eval_external_speaker_verifier.py`
- `tests/test_external_speaker_verifier.py`
- `results/eval_external_speaker_verifier_cvrare_sad_enunc_guard.csv`
- `results/eval_external_speaker_verifier_cvrare_sad_enunc_guard.md`
- `IMPLEMENTATION_PLAN_external-speaker-verifier.md`

Implementation:

- Added a manifest-driven external verifier script using SpeechBrain
  ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`).
- Added optional dependency group `speaker-verifier = ["speechbrain"]`.
- Added support for:
  - per-row source/generated ECAPA similarity;
  - baseline-relative external novelty gain;
  - EER-style thresholding from a real `--trial-csv`;
  - derived proxy trials from source-vs-baseline and source-vs-other-baseline
    pairs when a real trial CSV is not available.
- Used `soundfile` for audio loading to avoid torchaudio/torchcodec decoder
  drift in the local environment.

Result:

- Condition:
  `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard`
- Rows: `110`
- Styled rows: `99`
- Backend: SpeechBrain ECAPA
- Derived proxy trials: `121`
- Proxy EER: `0.0000`
- Proxy threshold: `0.4647`
- Mean styled external novelty gain vs baseline: `0.3594`
- Styled accept-as-source rate at proxy threshold: `0.0606` (`6/99`)

Interpretation:

- This is a paper-facing corroboration that the current quality-balanced guard
  moves speaker identity away from the source under an external verifier, not
  only under OpenVoice's native embedding metric.
- The EER threshold is proxy-calibrated from the current generated panel, so it
  should not be presented as a final formal speaker-verification benchmark.
- A final privacy/security claim still needs an independent labeled trial CSV
  and privacy-utility curves.

Validation:

- `Validation`: `.venv/bin/python -m unittest tests.test_external_speaker_verifier`
- `Validation`: `.venv/bin/python scripts/eval_external_speaker_verifier.py --manifest output/mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard_eval/generation_manifest.jsonl --backend speechbrain-ecapa --derive-proxy-trials --out results/eval_external_speaker_verifier_cvrare_sad_enunc_guard.csv --summary-out results/eval_external_speaker_verifier_cvrare_sad_enunc_guard.md`
- `Validation`: `FINDINGS.md` Finding 36 records the result with the proxy
  threshold caveat.

Next:

- `[DONE]` Add a metadata separability probe before more age/gender training.
- `[SOON]` Build an independent labeled speaker-verification trial CSV for
  final EER.
- `[SOON]` Combine external accept-as-source rates with WER/MOS/style recall
  into privacy-utility curves.

---

### 0.53 CommonVoice Metadata Separability Probe (2026-05-28, branch `research/metadata-separability-probe`)

Goal:

- Test whether the current CommonVoice metadata labels have recoverable
  structure in raw OpenVoice embeddings and metadata-control VAE latents before
  spending more compute on age/gender/accent scalar controls.

Artifacts:

- `scripts/probe_commonvoice_metadata_separability.py`
- `tests/test_probe_commonvoice_metadata_separability.py`
- `results/commonvoice_metadata_separability_cvrare_expanded.csv`
- `results/commonvoice_metadata_separability_cvrare_expanded.md`
- `results/commonvoice_metadata_separability_mixed_metadata_base.csv`
- `results/commonvoice_metadata_separability_mixed_metadata_base.md`
- `IMPLEMENTATION_PLAN_metadata-separability-probe.md`

Implementation:

- Added a deterministic nearest-centroid metadata probe with majority and
  train-label permutation baselines.
- Added optional VAE encoder probing, using checkpoint shape inference to
  encode artifacts into `vae_mu` latents without requiring a separate config
  file.
- Probed `gender`, `age`, and `accent` for both the expanded CommonVoice
  artifact and the actual mixed metadata-training base artifact.

Result:

- Expanded CommonVoice artifact:
  - `gender`: strongly separable in embeddings (`macro_f1=0.9077`) and VAE
    latents (`0.9005`)
  - `age`: weak / diagnostic only in embeddings (`0.1901`) and VAE latents
    (`0.1343`)
  - `accent`: moderately separable in embeddings (`0.1902`) but weak in VAE
    latents (`0.1446`)
- Mixed metadata-training base:
  - `gender`: strongly separable in embeddings (`0.8797`) and VAE latents
    (`0.8670`)
  - `age`: weak / diagnostic only (`0.1499` embedding, `0.1412` VAE latent)
  - `accent`: weak in embeddings (`0.1501`) and not meaningfully separable in
    VAE latents (`0.1016`)

Interpretation:

- The failed first listening panel should not be read as "CommonVoice metadata
  has no signal." Gender has strong objective structure.
- It should be read as "direct scalar age/gender controls are not yet
  perceptually validated." A generic timbre/identity shift can preserve
  classifier-separable gender structure without sounding like a clear,
  controllable age/gender change.
- Age and accent are not good next scalar-control targets without better
  labels, class balancing, or stronger perceptual/acoustic targets.

Validation:

- `Validation`: `.venv/bin/python -m unittest tests.test_probe_commonvoice_metadata_separability`
- `Validation`: `.venv/bin/python -m py_compile scripts/probe_commonvoice_metadata_separability.py`
- `Validation`: `.venv/bin/python scripts/probe_commonvoice_metadata_separability.py --artifact embeddings/openvoice_commonvoice_cvrare_expanded_emb.pt --vae-checkpoint embeddings/openvoice_vae_mixed_teacher_cvrare_metadata_w010_labeled_warmup.pt --min-class-count 20 --top-k-classes 8 --permutations 100 --seed 42 --out-csv results/commonvoice_metadata_separability_cvrare_expanded.csv --out-md results/commonvoice_metadata_separability_cvrare_expanded.md`
- `Validation`: `.venv/bin/python scripts/probe_commonvoice_metadata_separability.py --artifact embeddings/openvoice_mixed_teacher_cvrare_hybrid_extra_metadata_base.pt --vae-checkpoint embeddings/openvoice_vae_mixed_teacher_cvrare_metadata_w010_labeled_warmup.pt --min-class-count 20 --top-k-classes 8 --permutations 100 --seed 42 --out-csv results/commonvoice_metadata_separability_mixed_metadata_base.csv --out-md results/commonvoice_metadata_separability_mixed_metadata_base.md`
- `Validation`: `FINDINGS.md` Finding 37 records the result as diagnostic,
  not as a perceptual age/gender-control win.

Next:

- `[SOON]` If metadata controls remain in scope, run a narrow gender-focused
  balanced-control follow-up with a listening-first gate.
- `[SOON]` Add per-dimension latent diagnostics for metadata dims `9-10` and
  free dims `11-14`.
- `[DONE]` Prioritize generated-audio/content repair for hard styles before
  another broad age/gender/accent control branch.

---

### 0.54 Generated-Audio Content-Repair Gate (2026-05-28, branch `research/generated-audio-content-repair`)

Goal:

- Convert the hard-style strength grid into a conservative content-repair gate
  so classifier gains cannot become presets unless they also preserve content,
  naturalness, novelty, and human preference.

Artifacts:

- `scripts/select_generated_audio_content_repairs.py`
- `tests/test_select_generated_audio_content_repairs.py`
- `results/generated_audio_content_repair_gate.csv`
- `results/generated_audio_content_repair_gate.md`
- `results/generated_audio_content_repair_gate.json`
- `IMPLEMENTATION_PLAN_generated-audio-content-repair.md`

Implementation:

- Added a reusable gate over
  `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority.csv`.
- Folded in Joe's five-row ratings from
  `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority_ratings_joe_2026-05-19.csv`.
- Required target-label gain plus content/naturalness/novelty safety before a
  row can pass the objective gate.
- Required perceptual candidate preference before an objective-pass row can be
  promoted.

Result:

- Total hard-style A/B candidate rows: `33`
- `promote_candidate`: `0`
- `blocked_by_perceptual_tie`: `1`
- `blocked_by_reference_preference`: `1`
- `reject_quality`: `3`
- `reject_no_target_gain`: `28`
- Style decisions:
  - `anger`: `diagnostic_only`; one objective-pass row was a perceptual tie
  - `fear`: `diagnostic_only`; one objective-pass row preferred the reference
  - `disgust`: `needs_content_repair`; no objective-pass repair row

Interpretation:

- The strength grid remains useful diagnostics, not a promoted repair path.
- `anger_s10` and `fear_s7p5` can produce classifier gains on individual rows,
  but those rows did not win the perceptual gate.
- `disgust` needs a training-side or generated-audio-calibrated objective; a
  stronger strength preset is not enough.

Validation:

- `Validation`: `.venv/bin/python -m unittest tests.test_select_generated_audio_content_repairs`
- `Validation`: `.venv/bin/python -m py_compile scripts/select_generated_audio_content_repairs.py`
- `Validation`: `.venv/bin/python scripts/select_generated_audio_content_repairs.py --priority-csv results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority.csv --ratings results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority_ratings_joe_2026-05-19.csv --out-csv results/generated_audio_content_repair_gate.csv --out-md results/generated_audio_content_repair_gate.md --out-json results/generated_audio_content_repair_gate.json`
- `Validation`: `FINDINGS.md` Finding 38 records the no-promote result as a
  paper-facing guardrail, not a negative result against the whole system.

Next:

- `[SOON]` Build a generated-audio-calibrated training objective for hard
  styles, with `disgust` as the clearest repair target.
- `[SOON]` Add row-level audio-feature diagnostics for the `fear_s7p5`
  pitch-change artifact Joe heard.
- `[SOON]` Use the content-repair gate before any future style-strength preset
  is promoted.

---

### 0.55 Generated-Audio-Calibrated Objective Plan (2026-05-28, branch `research/generated-audio-calibrated-objective`)

Goal:

- Convert the generated-audio content-repair gate into a trainer-ready
  hard-style repair objective without promoting any current strength preset.

Artifacts:

- `IMPLEMENTATION_PLAN_generated-audio-calibrated-objective.md`
- `scripts/plan_generated_audio_calibrated_objective.py`
- `tests/test_plan_generated_audio_calibrated_objective.py`
- `results/generated_audio_calibrated_objective_plan.csv`
- `results/generated_audio_calibrated_objective_plan.md`
- `results/generated_audio_calibrated_objective_plan.json`

Implementation:

- Added a planner that combines
  `results/generated_audio_content_repair_gate.json` with
  `results/eval_mixed_teacher_failure_conditioned_targets.json`.
- Selected `anger` for conservative repair pressure because it has clean
  generated-audio style-to-neutral failures but no perceptual preset win.
- Selected `disgust` for content-repair pressure because the gate found no
  safe strength-grid repair row.
- Blocked `fear` because the target selector still has `0/11` clean reference
  rows and Joe heard an unnatural pitch shift in the best metric row.
- Added `--generated-audio-objective-plan` and
  `--generated-audio-objective-report` to the mixed trainer.
- Added `--decoder-prototype-style-weights` so decoder-prototype loss can be
  restricted to the evidence-selected hard styles instead of leaking to every
  pseudo-labeled CommonVoice style.
- Added inference alias `mixed_teacher_cvrare_audio_calibrated_labeled_warmup`
  for the future checkpoint.

Result:

- This branch is an implementation/reproducibility step, not a new model
  result.
- Trainer-ready selected styles: `anger`, `disgust`
- Blocked style: `fear`
- Plan overrides:
  - style-teacher target mode: `target_dim`
  - style-teacher row weights:
    `anger=2,confused=0,disgust=3,enunciated=0,fear=0,happy=0,neutral=0,sad=0,whisper=0`
  - decoder-prototype style weights:
    `anger=2,confused=0,disgust=3,enunciated=0,fear=0,happy=0,neutral=0,sad=0,whisper=0`
  - decoder/anti-neutral strengths:
    `anger=5,confused=0,disgust=5,enunciated=0,fear=0,happy=0,neutral=0,sad=0,whisper=0`

Validation:

- `Validation`: `.venv/bin/python -m unittest tests.test_plan_generated_audio_calibrated_objective`
- `Validation`: `.venv/bin/python scripts/plan_generated_audio_calibrated_objective.py --gate-json results/generated_audio_content_repair_gate.json --failure-target-json results/eval_mixed_teacher_failure_conditioned_targets.json --out-json results/generated_audio_calibrated_objective_plan.json --out-md results/generated_audio_calibrated_objective_plan.md --out-csv results/generated_audio_calibrated_objective_plan.csv`
- `Validation`: one-epoch trainer smoke on the real mixed artifact with the
  generated-audio plan loaded, writing only temporary outputs under
  `/private/tmp`.
- `Validation`: one-epoch CommonVoice-only smoke exercised nonzero
  style-teacher, decoder-prototype, and anti-neutral repair losses with the
  plan-applied `anger`/`disgust` rows.

Next:

- `[SOON]` Train the full
  `mixed_teacher_cvrare_audio_calibrated_labeled_warmup` checkpoint from the
  recommended command in `results/generated_audio_calibrated_objective_plan.md`.
- `[SOON]` Run generated-audio evaluation, external speaker verification, and a
  listening panel before adding any new paper-facing finding.
- `[SOON]` Add a fear-specific pitch/artifact diagnostic before allowing `fear`
  into this objective.

---

### 0.56 Generated-Audio-Calibrated Training Evaluation (2026-05-28, branch `research/generated-audio-calibrated-training`)

Goal:

- Train and evaluate the first full checkpoint produced by the
  generated-audio-calibrated objective plan from section 0.55.

Artifacts:

- local checkpoint:
  `embeddings/openvoice_vae_mixed_teacher_cvrare_audio_calibrated_labeled_warmup.pt`
- `results/generated_audio_calibrated_objective_training_report.json`
- `output/mixed_teacher_cvrare_audio_calibrated_labeled_warmup_eval/generation_manifest.jsonl`
- `results/eval_emotion_mixed_teacher_mixed_teacher_cvrare_audio_calibrated_labeled_warmup.csv`
- `results/eval_novelty_mixed_teacher_mixed_teacher_cvrare_audio_calibrated_labeled_warmup.csv`
- `results/eval_wer_mixed_teacher_mixed_teacher_cvrare_audio_calibrated_labeled_warmup.csv`
- `results/eval_mos_mixed_teacher_mixed_teacher_cvrare_audio_calibrated_labeled_warmup.csv`
- `results/eval_external_speaker_verifier_mixed_teacher_cvrare_audio_calibrated_labeled_warmup.csv`
- `results/eval_external_speaker_verifier_mixed_teacher_cvrare_audio_calibrated_labeled_warmup.md`
- `results/listening_mixed_teacher_cvrare_audio_calibrated_labeled_warmup.html`
- `results/listening_mixed_teacher_cvrare_audio_calibrated_labeled_warmup_ratings.csv`
- `results/listening_mixed_teacher_cvrare_audio_calibrated_labeled_warmup_stephen_2026-05-28.md`
- `results/listening_mixed_teacher_cvrare_audio_calibrated_labeled_warmup_joe_2026-05-28.md`
- `ASYNC_FEEDBACK_JOE_2026-05-28.md`

Implementation:

- Trained `mixed_teacher_cvrare_audio_calibrated_labeled_warmup` from
  `embeddings/openvoice_mixed_teacher_cvrare_hybrid_extra_base.pt` with the
  generated-audio objective plan loaded.
- The objective report confirms selected styles `anger` and `disgust`, blocked
  style `fear`, and plan-applied style/decoder/anti-neutral weights only for
  the selected hard styles.
- Generated a deterministic 110-row OpenVoice corpus with `style_strength=5.0`,
  `noise_level=0.0`, and `seed=42`.
- Ran the full generated-audio evaluation stack plus external SpeechBrain
  ECAPA speaker-verifier novelty validation.

Result:

| Condition | Recall | Mean WER | MOS delta | Novelty gain | Style-to-neutral | Any collapse |
|-----------|--------|----------|-----------|--------------|------------------|--------------|
| current `sad/enunciated` guard | `46.97%` | `0.2348` | `-0.2081` | `0.2726` | `18` | `20` |
| audio-calibrated training | `40.91%` | `0.2465` | `-0.1236` | `0.2351` | `29` | `37` |

- Targeted hard styles did not improve:
  - `anger`: `3/11` recall
  - `disgust`: `0/11` recall
- Stephen's first perceptual review complicates the metric story:
  - `disgust` sounds convincingly disgusted and remains intelligible
  - `anger` has some style change, but speech is distorted and less
    intelligible
- Joe's pre-meeting focused Teams review is the stronger current perceptual
  read:
  - all rows are subtle enough that he might not identify the intended emotion
    if asked cold
  - early `anger` rows up to `cremad_1076` sound slightly more angry, while
    later rows sound more neutral
  - `disgust` sounds neutral across the focused panel
  - many CREMA-D `disgust` training examples also sound neutral to him, so
    forcing a strong `disgust` signal is likely the wrong objective without
    stronger data
- External ECAPA still sees strong identity movement:
  - mean styled external novelty gain vs baseline: `0.3336`
  - styled accept-as-source rate: `0.0909`
- This is not a new reference checkpoint. It verifies the new trainer path but
  shows that the current generated-audio-calibrated loss mainly preserves
  speaker movement. Joe's review suggests `disgust` is a weak-training-signal
  limitation rather than a simple emotion2vec calibration miss.

Validation:

- `Validation`: full 3000-epoch training completed and wrote the local
  checkpoint plus objective-training report.
- `Validation`: deterministic inference wrote `110` generated rows and a
  manifest at
  `output/mixed_teacher_cvrare_audio_calibrated_labeled_warmup_eval/generation_manifest.jsonl`.
- `Validation`: generated-audio eval suite wrote emotion, novelty, WER, MOS,
  summary, collapse-taxonomy, listening HTML, and rating-template artifacts.
- `Validation`: external ECAPA verifier ran with `--derive-proxy-trials` and
  wrote CSV/Markdown summary artifacts.
- `Validation`: WER evaluation required `PATH=/opt/homebrew/bin:$PATH` so
  local `ffmpeg` was visible; add a preflight check before asking Joe or a new
  collaborator to rerun the suite.
- `Validation`: Stephen performed a first perceptual review of the `anger` and
  `disgust` styles and recorded the result in
  `results/listening_mixed_teacher_cvrare_audio_calibrated_labeled_warmup_stephen_2026-05-28.md`.
- `Validation`: Joe performed the focused review and the result is recorded in
  `results/listening_mixed_teacher_cvrare_audio_calibrated_labeled_warmup_joe_2026-05-28.md`.

FINDINGS.md review:

- Updated with Finding 39 as a paper-facing diagnostic result: this
  training-side generated-audio-calibrated objective does not beat the existing
  quality guard on aggregate metrics, and Joe's focused review suggests
  `disgust` should be treated as a weak/neutral training-label limitation
  rather than a current perceptual success.

Next:

- `[NOW]` Add a training-data perceptual audit / style-priority note before the
  next model run; include Joe's observation that many CREMA-D `disgust`
  examples sound neutral.
- `[SOON]` Add eval-suite dependency preflight for `ffmpeg` / `torchcodec`.
- `[SOON]` Move future demos/paper claims toward controls with clear perceptual
  signal, and treat subtle hard emotion labels as limitations or future work
  unless stronger labeled data is added.

### 0.57 May 28 Meeting with Joe — Paper-Facing Evaluation Pivot (2026-05-28, branch `research/generated-audio-calibrated-training`)

Source artifacts:

- `MEETING_DEBRIEF_JOE_2026-05-28.md`
- `ASYNC_FEEDBACK_JOE_2026-05-28.md`
- `IMPLEMENTATION_PLAN_control-selection-evaluation.md`

Joe's guidance:

- The system is basically working; do not spend the next cycle trying to make
  every weak emotion control stronger.
- The next work should focus on writing the paper, simplifying the evaluation,
  and justifying which controls we choose to highlight.
- For emotion/style controls, run the classifier on the original training data
  and use per-label F1/confusion to select the top separable controls. This is
  the defensible way to pick a top-3/top-k story instead of hand-selecting
  labels.
- `disgust` should not remain a hard repair target unless the training-data
  audit shows stronger separability than Joe heard perceptually.
- Gender is important and should be repaired if unclear. Joe previously heard
  CommonVoice-only gender control work, so the likely failure causes are
  insufficient gender-known rows, interaction with emotion control, or latent
  capacity.
- Accent should not be pursued in the current setup because OpenVoice likely
  carries accent in the content representation rather than the speaker
  embedding; our speaker-embedding VAE should not be expected to control it.
- Age is lowest priority. If tested, it should be broad-bucket classification,
  not fine-grained scalar regression.

Task-sequence impact:

- Next recommended branch: `research/control-selection-evaluation`.
- Primary next tasks: training-data style separability audit, gender-focused
  CommonVoice follow-up, and paper-methods simplification.
- Deprioritized tasks: `disgust` repair, accent control, broad metadata sweeps,
  and additional model complexity that does not directly answer a paper-facing
  evaluation question.

FINDINGS.md review:

- No new empirical finding was added from the meeting alone. The meeting is
  recorded as direction-setting guidance; the next empirical finding should
  come from the training-data separability audit and/or gender-focused
  follow-up.

---

### 0.58 Training-Data Style Separability Audit (2026-05-28, branch `research/control-selection-evaluation`)

Source artifacts:

- `scripts/audit_training_style_separability.py`
- `tests/test_audit_training_style_separability.py`
- `results/training_style_separability_rows.csv`
- `results/training_style_separability_by_label.csv`
- `results/training_style_separability_confusion.csv`
- `results/training_style_separability_summary.md`

Goal:

- Answer Joe's May 28 control-selection question with source-data evidence:
  which labeled controls are separable in the original training clips, and
  which labels should stay diagnostic rather than becoming open-ended repair
  targets?

Implementation:

- Added a wrapper-style audit CLI that loads cached local CREMA-D and Expresso
  datasets, mirrors the current training row policies, evaluates each source
  clip with `iic/emotion2vec_plus_large`, and writes rows, per-label summary,
  confusion counts, and a Markdown interpretation.
- Decoded dataset audio with `soundfile` and passed NumPy waveforms directly
  into FunASR, avoiding a hard runtime dependency on machine-level `ffmpeg` or
  `torchcodec`.
- Added a held-out nearest-centroid classifier over emotion2vec embeddings so
  Expresso-only labels without direct emotion2vec classes (`confused`,
  `enunciated`, `whisper`) can still be evaluated as embedding-space controls.

Result:

| Style | Support | Direct recall | Embedding F1 | Working interpretation |
| --- | ---: | ---: | ---: | --- |
| `anger` | 91 | `0.9451` | `0.9545` | source-label headline candidate |
| `disgust` | 91 | `0.9341` | `0.9091` | source-label headline candidate, but generated-audio perceptual evidence remains weak |
| `fear` | 91 | `0.7692` | `0.7027` | source-label headline candidate, generated-output quality still constrains claims |
| `happy` | 95 | `0.9053` | `0.8889` | source-label headline candidate |
| `neutral` | 95 | `0.9263` | `0.9091` | source-label headline candidate |
| `sad` | 95 | `0.8842` | `0.7925` | source-label headline candidate |
| `confused` | 90 | n/a | `0.2703` | weak / diagnostic |
| `enunciated` | 90 | n/a | `0.5000` | supported but quality-sensitive |
| `whisper` | 90 | n/a | `0.5161` | supported but quality-sensitive |

Validation:

- `Validation`: `python3 -m unittest tests/test_audit_training_style_separability.py`
  passed (`6` tests).
- `Validation`: `PYTHONPYCACHEPREFIX=/private/tmp/dpvc_pycache python3 -m py_compile scripts/audit_training_style_separability.py tests/test_audit_training_style_separability.py`
  passed.
- `Validation`: `.venv/bin/python scripts/audit_training_style_separability.py --offline --max-per-label 2 --min-class-count 2 --out-prefix results/training_style_separability_smoke`
  passed and wrote smoke rows/by-label/confusion/summary artifacts.
- `Validation`: `.venv/bin/python scripts/audit_training_style_separability.py --offline --out-prefix results/training_style_separability`
  passed on `828` source clips and wrote the checked-in full audit artifacts.

FINDINGS.md review:

- Added Finding 40 because this is verified paper-facing evidence for control
  selection. The finding is deliberately framed as source-label evidence, not
  generated-output success.

Next:

- `[NOW]` Build a control shortlist that intersects source separability,
  generated-output recall/WER/MOS/novelty, and perceptual review. This should
  prevent the paper from overclaiming controls that are measurable in source
  clips but subtle after conversion.
- `[SOON]` Add a quieter/batched path to the generated-output emotion evaluator
  or shared emotion2vec wrapper, because FunASR progress output is too noisy
  for long collaborator-facing runs.
- `[SOON]` Continue with the gender-focused follow-up only after the style
  shortlist is explicit.

---

### 0.59 Control Shortlist and Paper Claim Selection (2026-05-29, branch `research/control-shortlist`)

Source artifacts:

- `scripts/build_control_selection_recommendation.py`
- `tests/test_build_control_selection_recommendation.py`
- `results/control_selection_perceptual_evidence.csv`
- `results/control_selection_recommendation.csv`
- `results/control_selection_recommendation.md`

Goal:

- Convert source-label separability into a conservative paper/demo shortlist by
  intersecting source evidence, generated-output metrics, collapse diagnostics,
  external speaker novelty, and available perceptual evidence.

Implementation:

- Added a reproducible recommendation CLI that reads the checked-in current
  guard metrics, source separability table, generated-audio collapse table,
  ECAPA external-novelty table, and an explicit perceptual-evidence ledger.
- Added a small perceptual-evidence CSV so Joe/Stephen listening evidence is
  structured input to the recommendation, not hidden in narrative text.
- Implemented conservative claim gating: source separability alone cannot
  promote a control; generated-output metrics and positive focused listening
  are required before a style becomes a headline paper claim.

Result:

| Bucket | Styles | Interpretation |
| --- | --- | --- |
| `candidate_headline_pending_listening` | `neutral`, `sad` | strong source and generated metrics, but no focused listening confirmation recorded yet |
| `supported_but_quality_sensitive` | `enunciated`, `happy`, `whisper` | plausible secondary/demo controls, but quality or source-strength caveats prevent headline status |
| `diagnostic_or_limitation` | `anger`, `confused`, `disgust`, `fear` | do not use as headline claims under current evidence |

Key interpretation:

- No style control is promoted as fully paper-ready yet because the conservative
  gate requires positive focused listening evidence and none is recorded for
  the candidate headline controls.
- `neutral` and `sad` are the cleanest next listening targets.
- `disgust` remains source-separable but generated-perception-unconfirmed; Joe
  heard generated `disgust` as neutral, so it stays diagnostic unless stronger
  listening evidence changes the gate.

Validation:

- `Validation`: `python3 scripts/build_control_selection_recommendation.py`
  passed and wrote `results/control_selection_recommendation.csv` /
  `results/control_selection_recommendation.md`.
- `Validation`: `python3 -m unittest tests/test_build_control_selection_recommendation.py`
  passed (`5` tests).

FINDINGS.md review:

- Added Finding 41 because this is verified paper-facing evidence: it narrows
  the claim set and defines the next listening gate before any more training.

Next:

- `[NOW]` Focused listening on `neutral` and `sad` in the current reference
  guard; if they are perceptually clear, they can become the first headline
  style controls.
- `[SOON]` Listen to `happy`, `whisper`, and `enunciated` only as
  quality-sensitive secondary/demo controls.
- `[SOON]` Start the gender-focused CommonVoice follow-up after the style
  shortlist listening gate is resolved.

---

### 0.60 Listening Feedback Ingestion and Gender Follow-Up Preflight (2026-05-29, branch `research/control-feedback-gender-preflight`)

Source artifacts:

- `scripts/ingest_control_selection_feedback.py`
- `tests/test_ingest_control_selection_feedback.py`
- `scripts/preflight_commonvoice_gender_followup.py`
- `tests/test_preflight_commonvoice_gender_followup.py`
- `results/commonvoice_gender_followup_preflight.md`
- `results/commonvoice_gender_followup_preflight.json`
- `results/commonvoice_gender_followup_speakers.csv`
- `IMPLEMENTATION_PLAN_control-feedback-gender-preflight.md`

Goal:

- Keep working while Joe reviews the `neutral`/`sad` listening bundle by
  making his eventual feedback ingestible and by preparing, but not yet
  training, the narrow gender-only CommonVoice follow-up Joe identified as the
  most plausible metadata-control repair path.

Implementation:

- Added a feedback-ingestion CLI that turns a filled focused-listening ratings
  CSV or plain-text Joe response into structured rows for
  `results/control_selection_perceptual_evidence.csv`.
- The ingestion tool supports manual `STYLE=STATUS` overrides for Teams-style
  text replies, preserves canonical style order, and can optionally rerun the
  conservative control-selection recommendation.
- Added a local CommonVoice gender preflight that reads `validated.tsv` plus
  `clips/`, normalizes binary gender metadata, verifies local clip
  availability, and emits a deterministic speaker manifest for a future
  gender-only extraction/training branch.
- The gender script explicitly treats the result as metadata/data-readiness,
  not as a perceptual gender-control finding.

Result:

- Feedback ingestion smoke-tested against the empty Joe bundle ratings sheet
  and correctly left `neutral` / `sad` as `needs_review`.
- Local CommonVoice gender preflight result:
  - corpus: `/Users/steve/datasets/cv-corpus-21.0-2025-03-14/en`
  - local clips / validated rows: `40000` / `40000`
  - gender-known local clips: `5291`
  - gender-known local speakers: `2775`
  - female clips / speakers: `907` / `494`
  - male clips / speakers: `4384` / `2281`
  - deterministic selected plan: `994` speakers and `1788` clips
  - recommendation: `GO` for a gender-only follow-up preflight

Key interpretation:

- This unblocks the next practical step without short-cutting the science:
  when Joe replies, we can update the paper-control shortlist through a
  reproducible ledger rather than hand-editing interpretation text.
- The CommonVoice corpus is sufficient for a narrow gender-only follow-up, but
  the first metadata listening panel already failed perceptually. Any future
  gender claim still needs a new checkpoint, a listening panel, and objective
  evaluation before it enters `FINDINGS.md` as a control result.

Validation:

- `Validation`: `python3 -m unittest tests.test_ingest_control_selection_feedback tests.test_preflight_commonvoice_gender_followup` passed (`10` tests).
- `Validation`: `env PYTHONPYCACHEPREFIX=/private/tmp/dpvc_pycache python3 -m py_compile scripts/ingest_control_selection_feedback.py scripts/preflight_commonvoice_gender_followup.py tests/test_ingest_control_selection_feedback.py tests/test_preflight_commonvoice_gender_followup.py` passed.
- `Validation`: `python3 scripts/preflight_commonvoice_gender_followup.py` wrote the JSON/Markdown/CSV artifacts and returned `GO`.
- `Validation`: `python3 scripts/ingest_control_selection_feedback.py --ratings-csv results/joe_control_shortlist_neutral_sad_review_bundle_2026-05-29/ratings_neutral_sad.csv --out-ledger /private/tmp/control_selection_perceptual_evidence_smoke.csv` wrote a temp ledger and kept both styles at `needs_review`.

FINDINGS.md review:

- Reviewed after this branch. No new `FINDINGS.md` entry was added because the
  branch produced reproducibility/data-readiness evidence and an ingestion
  path, not a new verified generated-audio or perceptual control finding.

Next:

- `[NOW]` When Joe replies, save or transcribe his feedback, run
  `scripts/ingest_control_selection_feedback.py` against the real perceptual
  ledger, and rerun `scripts/build_control_selection_recommendation.py`.
- `[SOON]` If the `neutral`/`sad` listening gate is positive or clearly
  bounded, start the gender-only CommonVoice extraction/training branch from
  `results/commonvoice_gender_followup_speakers.csv`.
- `[SOON]` Keep age and accent out of the next training branch unless there is
  a separate, explicit data-readiness and perceptual target rationale.

---

### 0.61 Neutral/Sad Focused Listening Gate Resolved (2026-06-13, branch `research/control-feedback-gender-preflight`)

Source artifacts:

- `results/listening_control_shortlist_neutral_sad_joe_2026-06-08.md`
- `results/control_selection_perceptual_evidence.csv`
- `results/control_selection_recommendation.csv`
- `results/control_selection_recommendation.md`
- `scripts/build_control_selection_recommendation.py`
- `tests/test_build_control_selection_recommendation.py`

Goal:

- Convert Joe's focused Teams feedback on the neutral/sad review bundle into
  structured paper-facing evidence, then rerun the conservative shortlist gate.

Implementation:

- Recorded Joe's review as a Markdown evidence artifact.
- Ingested the feedback through `scripts/ingest_control_selection_feedback.py`
  instead of hand-editing the recommendation.
- Classified `neutral=supported` because Joe heard all neutral rows as neutral
  and all reviewed outputs as intelligible / reasonably natural.
- Classified `sad=supported` because Joe heard sad rows as sad for the most
  part, while preserving the caveat that row 4 was less obvious and the effect
  is perceptible but subtle.
- Patched `scripts/build_control_selection_recommendation.py` so the Markdown
  footer no longer tells us to listen to pending headline rows when no pending
  headline rows remain.

Result:

- The control-selection bucket counts are now:
  - `headline_control`: `2` (`neutral`, `sad`)
  - `supported_but_quality_sensitive`: `3` (`enunciated`, `happy`, `whisper`)
  - `diagnostic_or_limitation`: `4` (`anger`, `confused`, `disgust`, `fear`)
- `neutral` is now a paper/demo headline control.
- `sad` is now a paper/demo headline control with a clear wording caveat:
  perceptible, intelligible, and natural, but somewhat subtle / source-dependent.

Key interpretation:

- The paper story can now say the current system has two defensible style
  controls, not merely identity shift plus pending style evidence.
- The stronger claim is still narrow: do not say all nine controls work.
- The next scientific step is the gender-only CommonVoice follow-up, not more
  hard-emotion repair.

Validation:

- `Validation`: `python3 scripts/ingest_control_selection_feedback.py --feedback-text results/listening_control_shortlist_neutral_sad_joe_2026-06-08.md --evidence-path results/listening_control_shortlist_neutral_sad_joe_2026-06-08.md --style-status neutral=supported --style-status sad=supported --style-summary neutral="Joe confirmed all neutral outputs sound neutral; all reviewed outputs were intelligible and reasonably natural." --style-summary sad="Joe heard sad outputs as sad for the most part; row 4 was less obvious, and the sadness is perceptible but subtle." --rerun-recommendation` wrote the updated ledger and recommendation.
- `Validation`: `python3 scripts/build_control_selection_recommendation.py`
  reran after the Markdown footer patch and produced `headline_control: 2`.
- `Validation`: `python3 -m unittest tests.test_ingest_control_selection_feedback tests.test_preflight_commonvoice_gender_followup tests.test_build_control_selection_recommendation` passed (`16` tests).
- `Validation`: `env PYTHONPYCACHEPREFIX=/private/tmp/dpvc_pycache python3 -m py_compile scripts/ingest_control_selection_feedback.py scripts/preflight_commonvoice_gender_followup.py scripts/build_control_selection_recommendation.py tests/test_ingest_control_selection_feedback.py tests/test_preflight_commonvoice_gender_followup.py tests/test_build_control_selection_recommendation.py` passed.
- `Validation`: `git diff --check` passed.

FINDINGS.md review:

- Added Finding 42 because this is verified paper-facing perceptual evidence
  that changes the claim status for `neutral` and `sad`.

Next:

- `[NOW]` Begin the gender-only CommonVoice follow-up from
  `results/commonvoice_gender_followup_speakers.csv`.
- `[SOON]` Update the next Joe meeting brief around the new concise claim:
  current reference supports identity shift plus two perceptually confirmed
  headline style controls (`neutral`, `sad`), while harder emotions remain
  limitations.

---

### 0.62 CommonVoice Gender-Only Follow-Up Candidate (2026-06-13, branch `research/commonvoice-gender-followup`)

Source artifacts:

- `results/commonvoice_gender_followup_speakers.csv`
- `embeddings/openvoice_commonvoice_cvrare_expanded_emb.pt`
- `embeddings/openvoice_commonvoice_gender_followup_available_emb.pt`
- `embeddings/openvoice_mixed_gender_followup_available_base.pt`
- `embeddings/openvoice_vae_mixed_gender_followup_available_w005_labeled_warmup.pt`
- `results/commonvoice_gender_followup_artifact.md`
- `results/commonvoice_gender_followup_artifact.json`
- `results/openvoice_vae_mixed_gender_followup_available_w005_labeled_warmup_report.json`
- `results/listening_gender_followup_available_w005_labeled_warmup.html`
- `results/listening_gender_followup_available_w005_labeled_warmup_ratings.csv`
- `results/gender_followup_available_w005_review_bundle_2026-06-13.zip`
- `scripts/build_commonvoice_gender_followup_artifact.py`
- `tests/test_build_commonvoice_gender_followup_artifact.py`
- `tests/test_build_mixed_training_set_commonvoice.py`
- `tests/test_openvoice_train_vae_mixed_metadata_controls.py`
- `tests/test_openvoice_infer_controllable_baseline_only.py`
- `tests/test_build_listening_report_metadata_template.py`

Goal:

- Turn the CommonVoice gender preflight into the first reproducible
  gender-only candidate checkpoint and listening gate, without reviving
  age/accent as claims.

Implementation:

- Added `scripts/build_commonvoice_gender_followup_artifact.py` to subset an
  existing CommonVoice embedding artifact from the deterministic gender
  speaker manifest.
- The new artifact builder fails by default if manifest clips are missing, so
  partial coverage cannot be mistaken for the full plan.
- Built an explicit available-subset artifact with `--allow-missing`:
  - matched clips: `1181/1788`
  - missing clips: `607`
  - matched speakers: `656`
  - matched clips by gender: `female=493`, `male=688`
  - matched speakers by gender: `female=312`, `male=344`
- Patched `scripts/build_mixed_training_set.py` so unlabeled CommonVoice rows
  without pseudo-style fields are safe. This matters because the gender-only
  artifact intentionally has metadata labels, not pseudo emotion labels.
- Extended `examples/openvoice_train_vae_mixed.py` with
  `--metadata-control-targets`, defaulting to `gender,age` for compatibility
  but allowing `--metadata-control-targets gender` for a true gender-only run.
- Built `embeddings/openvoice_mixed_gender_followup_available_base.pt`:
  - total rows: `2006`
  - CommonVoice rows: `1181`, all gender-labeled
  - CREMA-D rows: `546`, true style-labeled
  - Expresso rows: `279`, true style-labeled
- Trained `embeddings/openvoice_vae_mixed_gender_followup_available_w005_labeled_warmup.pt`
  from the current style-distilled reference checkpoint using:
  - `--metadata-control-weight 0.05`
  - `--metadata-control-targets gender`
  - `--metadata-gender-dim 9`
  - `--style-teacher-weight 0.05`
  - `--style-teacher-datasets CommonVoice`
  - `--schedule labeled_warmup`
  - `--epochs 1000`
- Added `--baseline-only` to `examples/openvoice_infer_controllable.py` so
  metadata-only listening panels do not need to generate all style variants.
- Patched `scripts/build_listening_report.py` so baseline rows with
  `gender_control` / `age_control` remain scoreable in rating CSVs.
- Generated a compact review panel with four source speakers:
  source, no-metadata baseline, baseline+female, and baseline+male.

Result:

- The end-to-end gender-only candidate path now exists and trains.
- The current candidate is deliberately labeled an available-subset run, not
  the full preflight-selected CommonVoice plan.
- No paper-facing gender-control finding is recorded yet. The next gate is
  perceptual: if the female/male rows sound identical or only like generic
  speaker/timbre shifts, the result remains diagnostic.

How to listen:

```bash
cd /Users/steve/UVM-plaid/dp-vc/results/gender_followup_available_w005_review_bundle_2026-06-13
python3 -m http.server 8000
```

Open:

```text
http://localhost:8000/results/listening_gender_followup_available_w005_labeled_warmup.html
```

Listen per source in this order:

1. Source
2. Baseline
3. Baseline + female control
4. Baseline + male control

Key interpretation:

- This branch tests Joe's recommended metadata-control direction in the most
  conservative way: gender-only first, no age/accent claims, and perceptual
  review before metrics or paper language.
- If the panel fails perceptually, the useful result is that gender is
  objectively present in embeddings but still not controllable through this
  scalar speaker-embedding VAE knob.
- If the panel passes perceptually, the next step is objective gender/verifier
  evaluation plus a fuller extraction of the missing `607` manifest clips.

Validation:

- `Validation`: `.venv/bin/python -m unittest tests.test_build_commonvoice_gender_followup_artifact tests.test_openvoice_train_vae_mixed_metadata_controls` first failed before implementation because the artifact script was missing and metadata controls always returned both `gender` and `age`.
- `Validation`: `.venv/bin/python -m unittest tests.test_build_mixed_training_set_commonvoice` first failed before implementation because unlabeled CommonVoice artifacts without pseudo-style fields raised an `IndexError`.
- `Validation`: `.venv/bin/python -m unittest tests.test_openvoice_infer_controllable_baseline_only` first failed before implementation because `--baseline-only` did not exist.
- `Validation`: `.venv/bin/python -m unittest tests.test_build_listening_report_metadata_template` first failed before implementation because scoreable metadata-control baseline rows were omitted from the rating CSV.
- `Validation`: `.venv/bin/python -m unittest tests.test_build_listening_report_metadata_template tests.test_openvoice_infer_controllable_baseline_only tests.test_build_commonvoice_gender_followup_artifact tests.test_openvoice_train_vae_mixed_metadata_controls tests.test_build_mixed_training_set_commonvoice` passed (`8` tests).
- `Validation`: `.venv/bin/python scripts/build_commonvoice_gender_followup_artifact.py --commonvoice-embeddings embeddings/openvoice_commonvoice_cvrare_expanded_emb.pt --speaker-manifest results/commonvoice_gender_followup_speakers.csv --output embeddings/openvoice_commonvoice_gender_followup_available_emb.pt --report-json results/commonvoice_gender_followup_artifact.json --report-md results/commonvoice_gender_followup_artifact.md` failed as designed with `607` missing manifest clips.
- `Validation`: `.venv/bin/python scripts/build_commonvoice_gender_followup_artifact.py --commonvoice-embeddings embeddings/openvoice_commonvoice_cvrare_expanded_emb.pt --speaker-manifest results/commonvoice_gender_followup_speakers.csv --output embeddings/openvoice_commonvoice_gender_followup_available_emb.pt --report-json results/commonvoice_gender_followup_artifact.json --report-md results/commonvoice_gender_followup_artifact.md --allow-missing` wrote the explicit available-subset artifact.
- `Validation`: `.venv/bin/python scripts/build_mixed_training_set.py --commonvoice embeddings/openvoice_commonvoice_gender_followup_available_emb.pt --cremad embeddings/openvoice_cremad_emb.pt --expresso embeddings/openvoice_expresso_emb.pt --output embeddings/openvoice_mixed_gender_followup_available_base.pt --commonvoice-min-clips-per-speaker 1 --commonvoice-max-clips-per-speaker 2 --acceptance-policy confidence_only` wrote the mixed training artifact with `1181` gender-control rows.
- `Validation`: `.venv/bin/python examples/openvoice_train_vae_mixed.py --embeddings embeddings/openvoice_mixed_gender_followup_available_base.pt --output embeddings/openvoice_vae_mixed_gender_followup_available_w005_labeled_warmup.pt --epochs 1000 --schedule labeled_warmup --init-checkpoint embeddings/openvoice_vae_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.pt --metadata-control-weight 0.05 --metadata-control-targets gender --metadata-gender-dim 9 --metadata-control-report results/openvoice_vae_mixed_gender_followup_available_w005_labeled_warmup_report.json --style-teacher-checkpoint embeddings/openvoice_vae_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.pt --style-teacher-weight 0.05 --style-teacher-dims 0-8 --style-teacher-datasets CommonVoice` completed and wrote the checkpoint/report.
- `Validation`: baseline-only OpenVoice generation completed for no-metadata,
  female-control, and male-control batches, writing `12` generated audio rows.
- `Validation`: listening HTML validation found `16` audio refs, `0` missing
  refs, and `8` scoreable rating rows (`female`, `male`).
- `Validation`: zip validation found `28` files, `16` audio refs, and `0`
  missing audio refs inside
  `results/gender_followup_available_w005_review_bundle_2026-06-13.zip`.

FINDINGS.md review:

- Reviewed and intentionally not updated. This branch has a trained candidate
  and listening gate, but no verified perceptual or objective gender-control
  finding yet.

Next:

- `[DONE]` Stephen listened to the four-row panel; see section 0.63.
- `[DONE]` The panel sounded subtle/generic rather than reliably gender
  controlled, so the result is documented as a limitation.
- `[NOW]` Prioritize paper/evaluation cleanup over more scalar metadata-control
  tuning.
- `[SOON]` Extract the missing `607` manifest clips only if this path is
  revisited with a materially stronger objective, not to scale the current
  failed-gate checkpoint unchanged.

---

### 0.63 Gender Follow-Up Listening Gate Outcome (2026-07-09, branch `research/commonvoice-gender-followup`)

Source artifacts:

- `results/gender_followup_available_w005_review_bundle_2026-06-13.zip`
- `results/gender_followup_available_w005_review_bundle_2026-06-13/`
- `results/listening_gender_followup_available_w005_labeled_warmup.html`
- `results/listening_gender_followup_available_w005_labeled_warmup_ratings.csv`
- `results/listening_gender_followup_available_w005_labeled_warmup_stephen_2026-07-09.md`
- `results/commonvoice_gender_followup_artifact.md`
- `results/openvoice_vae_mixed_gender_followup_available_w005_labeled_warmup_report.json`

Goal:

- Resolve the perceptual gate for the first gender-only available-subset
  checkpoint before sending any broader gender claim to Joe or running
  objective verifier diagnostics.

Result:

- Stephen listened to the four-source panel.
- The `female` / `male` controls were not reliably perceptible as gender
  controls.
- The differences sounded closer to subtle, inconsistent, or generic
  speaker/timbre movement than to a stable controllable gender attribute.
- No paper-facing gender-control finding is promoted from this checkpoint.

Key interpretation:

- This is a diagnostic limitation, not a new headline claim.
- The result is consistent with Finding 37: gender is objectively recoverable in
  OpenVoice embeddings and metadata-control latents, but direct scalar
  metadata control has not produced listener-clear gender control.
- Do not run objective gender/speaker-verifier diagnostics as claim evidence for
  this checkpoint after the perceptual gate failed.
- Keep the paper story anchored on the current reference guard plus the two
  perceptually confirmed headline style controls: `neutral` and `sad`.

FINDINGS.md review:

- Added Finding 43 because this is a verified local perceptual gate outcome
  that fixes the gender-control status for the available-subset follow-up.

Next:

- `[NOW]` Prioritize paper/evaluation cleanup around the current reference
  guard and two confirmed style controls.
- `[SOON]` Revisit gender only with a materially stronger setup, such as full
  extraction of the preflight-selected clips plus a better-balanced or more
  perceptually grounded objective, or a bounded latent-capacity comparison.

---

### 0.64 Paper Evidence Cleanup (2026-07-09, branch `research/commonvoice-gender-followup`)

Source artifacts:

- `PAPER_METHODS_AND_EVIDENCE.md`
- `PAPER_EVIDENCE_CHECKLIST.md`
- `MEETING_BRIEF_JOE_2026-07-09.md`
- `FINDINGS.md`
- `results/control_selection_recommendation.md`
- `results/listening_gender_followup_available_w005_labeled_warmup_stephen_2026-07-09.md`

Goal:

- Make the paper-facing source of truth match the current evidence after the
  neutral/sad listening gate and the failed gender-only perceptual gate.

Implementation:

- Updated `PAPER_METHODS_AND_EVIDENCE.md` so `neutral` and `sad` are no longer
  described as pending-listening controls.
- Updated the metadata-control language so gender is no longer framed as an
  active near-term paper claim after the gender-only panel failed perceptually.
- Added `PAPER_EVIDENCE_CHECKLIST.md` as the short paper-readiness tracker.
- Linked the checklist from `README.md`.

Current claim boundary:

- Current reference guard is the paper/evaluation anchor.
- Headline style controls: `neutral`, `sad`.
- Secondary or demo-only candidates: `happy`, `enunciated`, `whisper`.
- Diagnostic/limitation controls: `anger`, `confused`, `disgust`, `fear`,
  gender, age, accent.

Remaining paper blockers:

- Formal DP accounting and privacy-utility curves.
- Independent labeled speaker-verification / EER trial.
- Repeated-seed confidence intervals for final tables.
- Decision on whether Joe's existing `neutral` / `sad` review is enough or
  whether a small broader listener panel is needed.

Next:

- `[NOW]` Ask Joe which blocker matters most before writing.
- `[SOON]` Implement the smallest validation task that answers that blocker.

---

## 1. Project Overview

**dpvc** is a Python library for **differentially private voice conversion** — it anonymizes a speaker's identity by passing their voice embedding through a VAE with calibrated DP noise, then reconstructs audio with a modified (anonymized) speaker embedding.

The new work on `feat/controlvc` extends this with **controllable anonymization**: instead of just adding noise, we can steer specific perceptual attributes (happy, sad, whisper, etc.) in the anonymized output by manipulating labeled latent dimensions of the VAE.

### Core Pipeline

```
Source Audio
    │
    ├─► HuBERT (content codes)  ─────────────────────────┐
    ├─► D_VECTOR (256-dim speaker embedding) ─► VAE ─► modified embedding
    ├─► YAAPT (F0 pitch contour)  ────────────────────────┤
    │                                                      │
    └──────────────────────────────────────────────────────┴─► CodeGenerator (vocoder) ─► Output Audio
```

**Key insight:** The VAE sits on the speaker embedding only. Content (HuBERT codes) and prosody (F0) pass through unchanged. So style control operates on *who it sounds like*, not *what they say*.

---

## 2. What Joe Near Built (Pre-April 2026)

Joe's work existed on two branches:

### `feat/controlvc` (merged ~Feb 2026)
- Added `ControlVCWrapper` in `dpvc/controlvc.py` (~500 lines) wrapping the [control-vc](https://github.com/auspicious3000/control-vc) system
- Integrated it alongside the existing `OpenVoiceWrapper`
- **Issues found:** hardcoded device `"cuda"` (broke on Mac), hardcoded paths to Joe's machine, fixed `INPUT_DIM=256` in VAE

### `controllable_vae` branch (never merged)
- Upgraded `model_embedding_vae.py` with a more sophisticated architecture: GELU activations, LayerNorm, proper reparameterization trick, `control_features` dict for overriding specific latent dims at inference
- Updated `utils.py` `train_autoencoder()` to support label-aware training (MSE loss on first K latent dims)
- Updated `anonymizer.py` to use a `vae_config` dict pattern
- All of this was OpenVoice-specific; needed porting to work with ControlVC

---

## 3. What We Built (April 8-9, 2026)

### 3.1 Portability Fixes (commit `139d0e5`)
- Changed ControlVC device default from `"cuda"` to auto-detect (`cuda` if available, else `cpu`)
- Added `get_vae_config()` to `ControlVCWrapper` and abstract `Wrapper` base class
- Fixed hardcoded paths in example scripts with argparse

### 3.2 Controllable VAE Port (commit `dfcbf9a`)
- Ported the full controllable VAE architecture from `origin/controllable_vae` into the main codebase
- Made it work with ControlVC (was OpenVoice-only)
- **Files replaced/updated:**
  - `dpvc/model_embedding_vae.py` — full rewrite with GELU/LayerNorm encoder, reparameterization trick, control_features support
  - `dpvc/anonymizer.py` — rewritten with vae_config dict pattern
  - `dpvc/utils.py` — `train_autoencoder()` now supports `labels` dict and label MSE loss
  - `dpvc/wrapper.py` — added `get_vae_config()` to abstract interface
  - `dpvc/openvoice.py` — added `get_vae_config()` for consistency

### 3.3 Expresso Pipeline (commit `dfcbf9a`)
Three new example scripts for end-to-end controllable training:

1. **`examples/controlvc_extract_expresso.py`** — Extracts ControlVC speaker embeddings from HuggingFace's `ylacombe/expresso` dataset
   - Handles 11 styles: default, confused, enunciated, happy, laughing, sad, whisper, emphasis, essentials, longform, singing
   - One-hot style encoding: active style = +1, all others = -1
   - Workarounds for: xet storage stalls (`HF_HUB_DISABLE_XET=1`), torchcodec incompatibility (uses `soundfile` instead), dataset download failures (supports `--parquet-dir` for local cache)

2. **`examples/controlvc_train_vae_expresso.py`** — Trains controllable VAE with style labels
   - Default: latent_dims=16 (11 style + 5 free), lr=1e-6
   - Label loss forces first 11 latent dims to match one-hot style encoding via MSE

3. **`examples/controlvc_infer_controllable.py`** — CLI for controllable voice anonymization
   - Maps style name to latent dim index
   - Sets all 11 style dims at inference (active = style_value, others = -1) to match training encoding

### 3.4 Fairseq/HuBERT Fix (April 9, uncommitted)
The HuBERT content encoder is critical — without it, the vocoder produces unintelligible noise. Fairseq 0.12.2 is incompatible with Python 3.11 due to mutable dataclass defaults.

**Fix applied in `dpvc/controlvc.py`:**
- Before importing fairseq, monkey-patches `dataclasses._get_field` to convert mutable defaults to `default_factory` calls
- Adds the control-vc repo to `sys.path` so `fairseq_feature_reader` is importable
- Also patched `fairseq/dataclass/configs.py` (manual `field(default_factory=...)` edits) and `fairseq/dataclass/initialize.py` (handle MISSING defaults) and `fairseq/checkpoint_utils.py` (`weights_only=False`) — these are in the venv, not version-controlled

**Result:** HuBERT now loads and produces real content codes, generating intelligible speech output.

### 3.6 F0 Prosody-Based Style Control (April 9, branch `feat/f0-style-control`)

After confirming that D_VECTOR embeddings don't carry style information (Section 5), pivoted to controlling style via F0 (pitch) manipulation.

**Changes:**
- `dpvc/controlvc.py` — Added `f0_transform` parameter to `inference()`. Supports three operations applied to the F0 contour before it hits the vocoder:
  - `pitch_shift`: multiply voiced F0 values (e.g., 1.25 = 25% higher pitch)
  - `range_scale`: expand/compress variation around mean (e.g., 0.4 = flatter, 2.0 = more expressive)
  - `flatten`: interpolate toward mean (0.0 = no change, 1.0 = completely monotone)
- `dpvc/anonymizer.py` — Pass-through `f0_transform` to wrapper
- `dpvc/wrapper.py` — Added `f0_transform` to abstract base class signature
- `examples/controlvc_infer_controllable.py` — Rewritten with F0 preset system and custom CLI args

**F0 Presets (tuned for audible distinction):**
| Style | pitch_shift | range_scale | flatten |
|-------|------------|-------------|---------|
| happy | 1.25 | 1.6 | — |
| sad | 0.80 | 0.4 | 0.3 |
| whisper | 0.90 | 0.2 | 0.8 |
| confused | 1.10 | 2.0 | — |
| laughing | 1.35 | 2.2 | — |
| enunciated | 1.00 | 1.8 | — |

**Result:** Audible style differences confirmed. Tested with trump_0.wav as source. Happy and laughing are clearly higher/more animated, sad is lower/flatter, whisper is near-monotone. The VC pipeline itself changes voice identity (expected — HuBERT quantization is lossy), but prosody differences are clearly distinguishable.

### 3.5 Inference Style Encoding Fix (April 9, uncommitted)
Original inference script only set 1 latent dim (the active style). But training used a full one-hot encoding (+1 active, -1 all others across all 11 dims). Fixed to set all 11 style dims at inference.

---

## 4. Training Runs

### Extraction
- **4,840 samples** extracted from Expresso dataset (5 of 6 shards — shard 5 consistently failed to download)
- 256-dim speaker embeddings + 11-dim one-hot style labels
- Saved to `embeddings/controlvc_expresso_emb.pt`
- Took ~9.5 minutes on CPU

### VAE Training

| Version | System | Epochs | LR   | Final Label Loss | Final Recon | Notes |
|---------|--------|--------|------|-----------------|-------------|-------|
| v1      | ControlVC | 2000   | 1e-6 | ~590            | 0.20        | LR too low — label dims barely learned |
| v2      | ControlVC | 2000   | 1e-4 | ~55             | 0.20        | 10x improvement, but still high |
| v3      | ControlVC | 5000   | 1e-4 | ~15             | 0.18        | Label loss improved but no audible style diff — even at style_value=5.0 |
| **v4**  | **OpenVoice** | **2000** | **1e-4** | **~1.5** | **~67** | **Label loss 10x lower than ControlVC v3! OpenVoice embeddings encode style.** |
| **v5**  | **OpenVoice+CREMA-D** | **2000** | **1e-4** | **~18** | **~200** | **91 speakers, 6 emotions. Higher label loss but all emotions perceptually distinct.** |

Checkpoints saved in `embeddings/`:
- `controlvc_vae_expresso.pt` (v1, ControlVC)
- `controlvc_vae_expresso_v2.pt` (v2, ControlVC)
- `controlvc_vae_expresso_v3.pt` (v3, ControlVC)
- `openvoice_vae_expresso.pt` (v4, OpenVoice+Expresso)
- `openvoice_vae_cremad.pt` (v5, OpenVoice+CREMA-D — best diversity)

---

## 5. Current Status & Open Problems

### Working
- End-to-end pipeline: extract → train → infer produces intelligible speech
- HuBERT content encoder loads and produces real content codes
- VAE reconstructs speaker embeddings with low reconstruction loss
- Style control infrastructure is in place

### OpenVoice Style Control Results (April 9)

Trained controllable VAE on 8,712 OpenVoice embeddings (v4, label loss ~1.5 — 10x better than ControlVC). Tested with `trump_0.wav` as source.

**Perceptual evaluation (amplified style values):**

| Style | x1 | x3 | x5 |
|-------|-----|-----|-----|
| whisper | Subtle | **Sounds like a real whisper** | Over-exaggerated, background noise |
| sad | Subtle shift | **Sounds like an actual sad person** (some distortion) | Sad but more distortion |
| happy | Barely perceptible | Jovial but close to original, not pronounced | Distorted, not clearly happy |
| laughing | No difference | No meaningful change | Minimal change |

**Key findings:**
- **x3 is the sweet spot** — x5 overdrives the decoder and introduces artifacts
- **Whisper and sad work well** via embedding alone — these styles map to strong tonal shifts (pitch reduction, energy reduction, flattening)
- **Happy and laughing need F0 augmentation** — their perceptual qualities (pitch variation, rhythmic changes) aren't fully captured in the 256-dim embedding
- **Massive improvement over ControlVC** — went from zero audible difference to clearly recognizable style shifts

**Numerical differences from baseline (mean absolute):**
| Style | x1 | x3 | x5 |
|-------|-----|-----|-----|
| whisper | 0.039 | 0.062 | **0.099** |
| sad | 0.036 | 0.038 | 0.056 |
| happy | 0.035 | 0.037 | 0.037 |
| laughing | 0.034 | 0.030 | 0.037 |

**F0 post-processing attempt (failed):**
Tried combining embedding control (x3) with F0 pitch shifting on the output audio for happy (+20%), laughing (+30%), confused (+10%). Result: voices became thin and high-pitched, not perceptibly "happy" or "laughing." Pitch shifting output audio is too crude — happiness and laughter are speech *behaviors* (varied intonation patterns, rhythm, breath) not achievable by shifting a single signal. Reverted F0 post-processing from OpenVoice wrapper.

**Conclusion:** Embedding-based style control with OpenVoice at x3 works for **tonal/energy styles** (whisper, sad) but not for **behavioral styles** (happy, laughing, confused). This is a meaningful finding — it reveals which style dimensions are capturable in a 256-dim speaker embedding vs. which require fundamentally different representations.

### CREMA-D Results (April 10, branch `feat/cremad-experiments`)

**Motivation:** Expresso has only 3 speakers — within-style variance is dominated by speaker identity, not emotion. Joe recommended maximizing speaker diversity. CREMA-D has 91 speakers × 6 emotions × ~13 sentences = 7,442 clips.

**Dataset:** [AbstractTTS/CREMA-D on HuggingFace](https://huggingface.co/datasets/AbstractTTS/CREMA-D). 6 emotions: anger, disgust, fear, happy, neutral, sad. Extracted 1 sample per speaker per emotion = 546 samples (per Joe's recommendation).

**Embedding separability (CREMA-D, 91 speakers):**

| Emotion | Dist from global | Within var | Ratio |
|---------|-----------------|------------|-------|
| anger | 0.2712 | 0.4875 | 0.56 |
| sad | 0.2006 | 0.4875 | 0.41 |
| happy | 0.1612 | 0.4978 | 0.32 |
| neutral | 0.1430 | 0.4768 | 0.30 |
| disgust | 0.1483 | 0.4891 | 0.30 |
| fear | 0.1321 | 0.5180 | 0.26 |

Overall ratio: 0.54. Speaker separability: 1.43 (speakers are well-separated, emotions are not). The embedding fundamentally prioritizes speaker identity over emotion — but the VAE can still learn nonlinear separations.

**Perceptual evaluation (x3 amplification):**

| Emotion | Listener assessment |
|---------|-------------------|
| anger | Subtly different, noticeable |
| disgust | Similar to anger |
| fear | Distinct but not "fearful" — sounds timid |
| happy | Does sound happy, especially toward end of speech |
| neutral | Neutral sounding |
| sad | Sounds sad |

**Acoustic analysis (librosa F0/RMS/spectral centroid):**

| Style | dF0 Mean | dF0 Std | dF0 Range | dEnergy | dBrightness |
|-------|----------|---------|-----------|---------|-------------|
| anger | +12.4 | +2.1 | -12.7 | -0.001 | +123 |
| disgust | -38.9 | +3.8 | -36.3 | +0.001 | -27 |
| fear | -5.4 | +7.6 | -6.4 | +0.017 | -26 |
| happy | -14.4 | +12.1 | +12.9 | +0.001 | -49 |
| neutral | -53.4 | +0.1 | -43.7 | +0.007 | -178 |
| sad | -12.9 | +2.7 | -8.0 | +0.008 | -223 |

**Key findings:**
- **Happy has highest F0 variance** (+12.1 over baseline) — exactly matches how happy speech sounds
- **Anger is brightest** (+123) — sharper, edgier tone
- **Sad is darkest** (-223 brightness) — muffled, lower quality
- **Neutral is flattest** (F0 std +0.1, smallest range) — monotone
- **All 6 emotions are acoustically distinct** — major improvement over Expresso where only whisper/sad worked
- **Speaker diversity matters more than label separability** — despite lower raw separability (0.54 vs 0.62), the VAE learned better generalizable patterns from 91 speakers vs 3

**Why CREMA-D works better than Expresso for emotion control:**
1. 91 speakers forces the VAE to find emotion patterns that generalize across voices
2. 1 sample per speaker per emotion prevents speaker memorization
3. CREMA-D emotions are acted with clear intent (professional actors), while Expresso styles are more subtle reading variations

### Combined CREMA-D + Expresso Results (April 10, branch `feat/cremad-experiments`)

**Motivation:** CREMA-D provides speaker diversity (91 speakers, 6 emotions) but lacks Expresso's unique styles (whisper, confused, enunciated). Combining both gives the best of both worlds: 9 unified labels with strong speaker diversity.

**Unified label scheme (825 samples, ~90-94 per label):**
- From CREMA-D (91 speakers): anger, disgust, fear, happy, neutral, sad
- From Expresso (3 speakers, capped at 90): confused, enunciated, whisper
- Shared (CREMA-D + 1-per-speaker Expresso): happy, neutral, sad

**VAE v6:** 3000 epochs, lr=1e-4, latent_dims=15 (9 labels + 6 free). Final label loss ~30.

**Acoustic analysis (all deltas from baseline):**

| Style | dF0 Mean | dF0 Std | dF0 Range | dBrightness | Signature |
|-------|----------|---------|-----------|-------------|-----------|
| anger | +13.2 | -3.9 | +4.4 | +369 | Sharp, edgy |
| confused | +2.5 | -3.8 | -29.5 | +111 | Hesitant, narrow range |
| disgust | -25.6 | +6.1 | +24.3 | -156 | Low, withdrawn |
| enunciated | -10.5 | -3.9 | +5.2 | +274 | Crisp, bright |
| fear | +17.4 | +4.8 | +80.8 | -155 | Tense, variable |
| **happy** | **+26.8** | **+33.2** | **+240.7** | +84 | **Most expressive — 2x F0 variation** |
| neutral | -46.0 | +6.7 | +38.8 | -533 | Flat, subdued |
| sad | -0.3 | +11.2 | +44.3 | -272 | Darker tone |
| **whisper** | **-128.6** | **-30.8** | **-143.6** | **+945** | **F0 near zero, breathy/airy** |

**Perceptual evaluation:** All 9 styles perceptually distinct and matching acoustic expectations. This is the first time happy has produced a convincingly animated output.

**Why the combined model works:**
1. **Speaker diversity from CREMA-D** (91 speakers) prevents speaker memorization
2. **Style richness from Expresso** adds whisper/confused/enunciated which CREMA-D lacks
3. **Balanced sampling** (~90 per label) prevents any label from dominating
4. **Cross-dataset generalization** — the model learns emotion patterns that hold across two completely different recording conditions

**Progression of results:**
| Model | Dataset | Speakers | Working styles | Happy? |
|-------|---------|----------|---------------|--------|
| v1-v3 | Expresso+ControlVC | 3 | 0 | No |
| v4 | Expresso+OpenVoice | 3 | 2 (whisper, sad) | No |
| v5 | CREMA-D+OpenVoice | 91 | 6 (all emotions) | Yes |
| **v6** | **Combined** | **94** | **9 (all labels)** | **Yes** |

---

### ControlVC Style Differentiation (April 9 — superseded by OpenVoice)
**The core problem with ControlVC:** All style-controlled outputs sounded identical. Setting different style dims at inference produced no audible differences.

**Root cause identified (April 9):** Empirical analysis of the raw 256-dim D_VECTOR embeddings shows that **styles are not separable in embedding space:**

```
Embedding analysis (4,840 samples, unit-normalized):
  Between-style mean centroid distance: 0.0298
  Within-style mean distance to centroid: 0.0338
  Separability ratio: 0.88 (needs >>1)

  Only whisper shows any separation (~0.06 from others)
  All other styles overlap almost completely
```

**What this means:** The D_VECTOR speaker embedding encodes speaker identity, not speaking style. Within any given style, samples from different speakers vary MORE than the style signal itself. The VAE cannot learn to control what the input representation doesn't encode.

**Style distribution in training data:**
| Style | Samples | Notes |
|-------|---------|-------|
| default | 759 | |
| confused | 760 | |
| enunciated | 760 | |
| happy | 760 | |
| laughing | 557 | |
| sad | 379 | |
| whisper | 379 | Only style with measurable embedding separation |
| emphasis | 400 | |
| essentials | 80 | Too few samples |
| longform | 3 | Effectively zero |
| singing | 3 | Effectively zero |

**Training code observation:** `beta = 1` and losses use `.sum()` not `.mean()`. Label loss sums over batch×11 dims = 2,816 terms. Recon loss sums over batch×256 dims = 65,536 terms. Label loss is inherently ~23x smaller in gradient magnitude — the VAE heavily prioritizes reconstruction over label matching.

**Confirmed April 9:** Tested v3 checkpoint (label loss ~15) with style_value=5.0 (5x normal). Zero audible difference from baseline. The decoder is simply not sensitive to these latent dims because the input representation doesn't carry style.

**Possible paths forward:**

1. **Control F0 instead of (or in addition to) speaker embedding:** Style differences primarily manifest in prosody (F0 contour), not speaker identity. Applying the controllable VAE to F0 features might yield audible style differences. The ControlVC pipeline already extracts F0 via YAAPT — we just don't route it through the VAE.

2. **Use a style-aware embedding model:** Replace D_VECTOR with an embedding model that jointly encodes identity and style (e.g., emotion-aware speaker encoder, or a multi-task model trained on both speaker ID and emotion).

3. **Switch to per-speaker style control:** Within a single speaker, style differences may be more detectable (no cross-speaker variance to drown the signal). Train speaker-specific VAEs or condition on speaker ID.

4. **Operate on a concatenated representation:** Instead of just the 256-dim speaker embedding, pass [speaker_emb; F0_stats; energy_stats] through the VAE. This gives the model more style-relevant information to work with.

5. **~~Ask Joe~~** ✓ **ANSWERED (April 9):** Joe confirmed that OpenVoice embeds F0 in the speaker embedding, so style changes ARE audible with OpenVoice. He noted that some VC systems (like ControlVC) treat F0 separately, so randomizing the speaker embedding alone doesn't produce audible style changes. He also reminded us that the original plan was to use Expresso with **OpenVoice, not ControlVC**, since OpenVoice has been easier to work with.

**Conclusion: Pivot to OpenVoice for controllable style work.** The ControlVC pipeline is still valuable for DP anonymization (speaker embedding noise), but style control requires a system where F0 is part of the embedding — which OpenVoice provides.

---

## 6. Architecture Details

### VAE (model_embedding_vae.py)

```
Encoder: Linear(256→512) → GELU → LayerNorm(512) → Linear(512→256) → GELU → Linear(256→64) → GELU
         → to_mu(64→latent_dim), to_logvar(64→latent_dim)

Decoder: Linear(latent_dim→64) → GELU → Linear(64→256) → GELU → Linear(256→512) → GELU → Linear(512→256)
```

Reparameterization: `z = mu + eps * exp(0.5 * logvar)`

DP noise (at inference): L2 clip → Gaussian noise → post-clip → clamp

Control features: After computing z, override specific dims: `z[:, idx] = value`

### Training Loss
```
total_loss = recon_loss + kl_loss + beta * label_mse_loss
```
Where:
- `recon_loss`: MSE between input and reconstructed embedding
- `kl_loss`: KL divergence (standard VAE)
- `label_mse_loss`: MSE between first K latent dims and target labels (one-hot style encoding)
- `beta`: weighting factor (needs investigation — see `dpvc/utils.py`)

### Anonymizer Flow
```python
source_embedding = wrapper.extract_embedding(source_file)   # 256-dim
target_embedding = VAE(source_embedding, noise, control_features)  # 256-dim (modified)
wrapper.inference(source_file, output_file, source_embedding, target_embedding)
```

The `inference()` call uses the source audio's HuBERT codes and F0, but replaces the speaker embedding with the VAE-modified one.

---

## 7. File Inventory

### Core Library (`dpvc/`)
| File | Purpose | Status |
|------|---------|--------|
| `controlvc.py` | ControlVC wrapper (HuBERT, D_VECTOR, F0, vocoder) | Updated: fairseq compat patch, device auto-detect |
| `anonymizer.py` | DP pipeline orchestrator | Rewritten: vae_config dict pattern |
| `model_embedding_vae.py` | VAE model (GELU/LayerNorm, control_features) | Rewritten from controllable_vae branch |
| `utils.py` | Training utilities (train_autoencoder with labels) | Updated: label-aware training |
| `wrapper.py` | Abstract base class | Updated: get_vae_config() |
| `openvoice.py` | OpenVoice wrapper | Updated: get_vae_config() |
| `__init__.py` | Package exports | Unchanged |

### Examples (`examples/`)
| File | Purpose | Status |
|------|---------|--------|
| `controlvc_extract_expresso.py` | Extract embeddings from Expresso | New |
| `controlvc_train_vae_expresso.py` | Train controllable VAE | New |
| `controlvc_infer_controllable.py` | Controllable inference CLI | New, updated (full style encoding) |
| `controlvc_extract_commonvoice.py` | CommonVoice extraction | Updated (argparse) |
| `openvoice_inference.py` | OpenVoice demo | Updated (vae_config API) |

### Generated Artifacts (not committed)
| Path | Description |
|------|-------------|
| `embeddings/controlvc_expresso_emb.pt` | 4,840 extracted embeddings + style labels |
| `embeddings/controlvc_vae_expresso.pt` | VAE v1 (lr=1e-6, 2000 epochs) |
| `embeddings/controlvc_vae_expresso_v2.pt` | VAE v2 (lr=1e-4, 2000 epochs) |
| `embeddings/controlvc_vae_expresso_v3.pt` | VAE v3 (lr=1e-4, 5000 epochs, in progress) |
| `output/*.wav` | Generated audio samples |

---

## 8. Dependencies & Environment

- **Python:** 3.11.9 (pyenv)
- **PyTorch:** 2.0.1 (via venv)
- **torchaudio:** 2.9.1
- **fairseq:** 0.12.2 (requires dataclass monkey-patch for Python 3.11)
- **hydra-core:** 1.3.2 (upgraded from 1.0.7)
- **omegaconf:** 2.3.0 (upgraded from 2.0.6)
- **control-vc repo:** `/Users/steve/repos/control-vc` (contains checkpoints and fairseq_feature_reader.py)
- **HuBERT checkpoint:** `control-vc/checkpoints/hubert_base_ls960.pt`
- **K-means model:** `control-vc/checkpoints/km.bin`
- **Vocoder:** `control-vc/checkpoints/embed_f0stat2/g_00350000.pth`

### Venv patches (not version-controlled)
These files in `.venv/lib/python3.11/site-packages/fairseq/` were manually patched:
- `dataclass/configs.py` — `field(default_factory=...)` for FairseqConfig fields
- `dataclass/initialize.py` — handle MISSING defaults in hydra_init loop
- `checkpoint_utils.py` — `weights_only=False` for torch.load

---

## 9. Lingering Questions

### Resolved
1. ~~**Is speaker embedding the right place to control style?**~~ **No.** Confirmed empirically — D_VECTOR doesn't encode style. Pivoted to F0.
2. ~~**What is beta in the label loss?**~~ beta=1. Loss uses `.sum()` not `.mean()`, so label loss is ~23x smaller in gradient magnitude than recon loss.
3. ~~**Should we validate style presence in embeddings first?**~~ Yes, did this. Separability ratio = 0.88 (not separable).

### Still Open

1. **How do we make F0 control DP-compatible?** This is the central research question. See Section 10.1 below for a full analysis.

2. **Did the controllable VAE ever work with OpenVoice?** If Joe saw audible style differences with OpenVoice embeddings, the embedding-based approach might work for some VC systems. Need to ask.

3. **What F0 features are speaker-identifying?** Mean F0, F0 range, speaking rate, and intonation patterns are all biometric. We need to understand which features carry identity vs. style to design the right DP mechanism.

4. **Do we need the full 11 styles?** Longform (3 samples) and singing (3 samples) are useless. essentials (80 samples) is marginal. Probably should reduce to 7-8 styles for cleaner training.

5. **How does the noise budget compose across embedding + F0?** If we apply DP noise to both speaker embedding AND F0 features, the total privacy cost combines via composition. What's the right epsilon split?

---

## 10. Novelty & Paper Potential

### 10.1 The DP-Compatible F0 Problem (Key Research Question)

**The gap in our current system:**

Right now, the pipeline has two independent pieces:
1. **Speaker embedding** → VAE + DP noise → anonymized embedding (has formal privacy guarantee)
2. **F0 contour** → deterministic preset transform → modified pitch (no privacy guarantee)

The F0 contour is **biometric information**. Your pitch patterns, speaking rhythm, and intonation help identify you as a speaker. If we apply DP noise to the speaker embedding but leave F0 unprotected (or only apply a fixed transform), an attacker could potentially re-identify the speaker from the F0 alone — defeating the privacy guarantee on the embedding side.

**What "DP-compatible F0 control" means:**

Instead of hardcoded presets like `pitch_shift=1.25`, we need a mechanism where:
- F0 features pass through a noise mechanism with a formal privacy guarantee (calibrated epsilon)
- Style can still be controlled by overriding specific dimensions
- The privacy budget accounts for BOTH the embedding and F0 channels

**Three possible approaches:**

**Approach A: F0 statistics through the existing VAE**
- Extract F0 summary statistics (mean, std, range, slope) from the source audio — say 4-6 features
- Concatenate with the 256-dim speaker embedding → 260-262 dim input
- Train a single VAE on the combined representation
- First K latent dims → style labels (which now have F0 information to learn from!)
- DP noise applied to the full latent space
- Reconstruct both embedding + F0 stats; use reconstructed stats to reshape the F0 contour
- **Pros:** Single privacy budget, one model, clean architecture
- **Cons:** Different feature scales (unit-norm embedding vs. raw Hz F0), may need careful normalization

**Approach B: Separate F0 VAE**
- Train a second, smaller VAE specifically on F0 statistics
- Apply DP noise independently to each VAE
- Use composition theorem to compute total privacy cost
- **Pros:** Each model focuses on its domain, easier to tune
- **Cons:** Two models to maintain, composition increases total epsilon

**Approach C: Direct DP mechanism on F0 statistics**
- Skip the VAE for F0 — just L2-clip the F0 stats and add Gaussian noise directly
- Override specific stats with style targets (e.g., force mean_F0 = 200 Hz for "happy")
- Reconstruct the F0 contour from the (noisy) stats
- **Pros:** Simplest implementation, no training needed
- **Cons:** Less expressive, can't learn nuanced style-identity disentanglement

**Recommendation for discussion with Joe:** Approach A is the most elegant and publishable — a single VAE that jointly protects identity (via embedding) and prosody (via F0 stats), while separating style-controllable dims from identity dims. The key experiment would be: do F0 statistics form separable clusters by style? (We already know the answer should be yes — we proved F0 control produces audible differences.)

**What this enables for a paper:**
- "We show that speaker embeddings alone are insufficient for style control (separability ratio 0.88)"
- "We demonstrate that F0 prosody features carry style information that is audibly distinguishable"
- "We propose a joint embedding-prosody VAE that provides DP guarantees across both channels while maintaining controllable style"
- This would be a genuine contribution — most voice anonymization work ignores prosody as an identity channel

---

### Core Contribution
**Controllable differentially private voice conversion** — to our knowledge, no prior work combines:
- Differential privacy on speaker embeddings
- Explicit control over perceptual attributes in the anonymized output
- A VAE architecture that separates controllable (labeled) and free (identity/privacy) latent dimensions

### Why This Matters
Standard DP voice anonymization destroys all speaker information indiscriminately. Controllable DP-VC lets you *choose* what to reveal: "anonymize the speaker, but make them sound happy" or "anonymize but preserve the emotional tone." This has applications in:
- **Call center anonymization:** Strip identity but preserve customer sentiment
- **Witness protection recordings:** Change voice but maintain emotional authenticity
- **Accessible media:** Re-voice content with specific style properties

### Paper Framing Ideas
1. **"Prosody-Aware Differentially Private Voice Conversion"** — argue that existing DP voice anonymization leaks identity through prosody, then show a joint embedding+F0 mechanism that protects both while enabling style control
2. **"Controllable Differential Privacy for Voice Conversion"** — frame as a privacy-utility tradeoff where "utility" includes perceptual style control
3. **"Expressive Voice Anonymization with Formal Privacy Guarantees"** — emphasize the practical application angle

### Key Results to Include
- **Negative result (important!):** Speaker embeddings (D_VECTOR) don't encode style. Separability ratio = 0.88. This is worth reporting — it tells the community that embedding-only style control doesn't work.
- **Positive result:** F0 prosody manipulation produces audibly distinct styles through the VC pipeline. Style lives in prosody, not in speaker embeddings.
- **The gap:** F0 is an unprotected identity channel in current DP voice anonymization systems.

### What Would Strengthen a Paper
- **Quantitative style evaluation:** Use a pretrained emotion classifier on outputs to measure if controlled styles are detectable
- **Speaker verification experiments:** Show that anonymization actually reduces speaker re-identification accuracy
- **F0-based re-identification attack:** Demonstrate that F0 alone can re-identify speakers even after embedding anonymization — this motivates the need for F0 protection
- **Privacy-utility curves:** Plot speaker verification accuracy vs. emotion classification accuracy at different noise levels, for embedding-only vs. embedding+F0 protection
- **Joint VAE ablation:** Compare Approach A (joint VAE) vs. Approach B (separate VAEs) vs. Approach C (direct mechanism)
- **Comparison with naive approach:** Show that just adding noise (without style control) cannot achieve the same style preservation

### Related Work to Position Against
- Voice Privacy Challenge (VPC) 2020-2024 — DP voice anonymization baselines (embedding-only, no F0 protection)
- SpeechFlow / NaturalSpeech — controllable speech synthesis (but no privacy)
- FHVAE / SpeechSplit — disentangled speech representations (but no DP)
- Prosody-based speaker recognition literature — establishes that F0 IS identifying (motivates our work)

---

## 11. Meeting Prep Notes (for April 10)

### What to Demo
- The end-to-end pipeline works: extract → train → infer → audible speech
- HuBERT content encoder is now functional (was broken, produced noise)
- **F0-based style control produces audible differences** (validated April 9)
- Can play back original Trump audio → baseline VC → happy/sad/whisper/laughing variants

### What to Discuss
- **ControlVC D_VECTOR doesn't encode style** — confirmed empirically (separability ratio 0.88) and by Joe ("some systems treat F0 differently"). Not a dead end for the project, just the wrong VC system for style control.
- **OpenVoice is the right target for controllable style** — its embedding captures F0/prosody, so the controllable VAE should produce audible differences. This was the original plan per Joe.
- **Concrete next step:** Re-run the Expresso extraction + VAE training pipeline with OpenVoice instead of ControlVC. The code is already set up for this (`dpvc/openvoice.py` has `get_vae_config()`).
- **F0 prosody control on ControlVC works as a fallback** — we proved direct F0 manipulation produces audible style differences. This could still be useful for ControlVC-based anonymization even if the VAE-based approach moves to OpenVoice.
- **DP question for longer term:** If we do both embedding-based style control (OpenVoice) AND F0 manipulation, how do the privacy budgets compose? Is there a unified approach?
- **The fairseq compat situation** is fragile (monkey-patches for Python 3.11). Worth discussing whether to pin Python 3.10 or migrate to torchaudio's HuBERT.

### Joe's Feedback (April 9, pre-meeting)
> "Some systems treat f0 differently and so randomizing the speaker embedding doesn't sound like a big change. OpenVoice embeds the f0 profile in the speaker embedding, so with OpenVoice you do tend to get audible differences. I think we discussed doing that with OpenVoice (not ControlVC) since OpenVoice has been the easiest to work with in general."

**Implication:** The controllable VAE architecture is sound — the problem is ControlVC's D_VECTOR, not the approach. Switching to OpenVoice should produce audible style differences because its embedding captures F0/prosody.

### Prior Meeting Notes (March 18 call — Expresso plan)

**The agreed-upon plan was always OpenVoice + Expresso:**
1. **Extraction:** Read Expresso wav files, extract speaker embeddings using the OpenVoice wrapper, save embeddings + style labels to a .pt file
   - Reference: `controllable_vae` branch → `examples/openvoice_extract_commonvoice_features.py`
2. **Train VAE with labeled features:** For K labeled features, force the first K latent dims to match the labels via MSE loss during training
   - Reference: `controllable_vae` branch → `examples/openvoice_train_vae_features.py`

**The mechanism (from Joe):**
- Latent representation has N dimensions (e.g., 8)
- If we have K labeled features (e.g., age, gender, accent), the first K dims are forced to equal those labels
- Training loss includes MSE between feature values and corresponding latent dims
- At inference, override those K dims to control the output

**Key URLs from that meeting:**
- Expresso dataset: https://speechbot.github.io/expresso/
- NaturalSpeech3 extraction example: `controllable_vae` branch → `examples/naturalspeech3_extract_commonvoice...`
- OpenVoice extraction: `controllable_vae` branch → `examples/openvoice_extract_commonvoice_features.py`
- OpenVoice VAE training: `controllable_vae` branch → `examples/openvoice_train_vae_features.py`

**What we did instead:** Built the pipeline for ControlVC (which we now know doesn't embed F0 in the speaker embedding). Need to redo with OpenVoice as originally planned.

### 3.7 OpenVoice Expresso Extraction (April 9, branch `feat/openvoice-expresso`)

Per Joe's feedback and the original plan, pivoted to extracting Expresso embeddings with OpenVoice instead of ControlVC. OpenVoice embeds F0/prosody in its speaker embedding, so style control via the controllable VAE should produce audible differences.

**Changes:**
- `dpvc/openvoice.py` — Rewrote `extract_embedding()` to call `tone_color_converter.extract_se()` directly, bypassing OpenVoice's `get_se()` which runs VAD-based audio splitting. The VAD splitting asserts `num_splits > 0` (requires ~10s of speech after VAD), causing 75% of short Expresso utterances to fail. Direct extraction works on any length audio.
- `dpvc/__init__.py` — Uncommented and fixed OpenVoiceWrapper export
- `examples/openvoice_extract_expresso.py` — New extraction script adapted from ControlVC version:
  - Uses pandas for parquet loading (bypasses HuggingFace datasets library issues with incomplete cache)
  - Falls back to HuggingFace `load_dataset` when no `--parquet-dir` given
  - Same one-hot style encoding (+1 active, -1 others) across 11 styles
  - ~15-20 samples/sec on CPU (faster than ControlVC extraction)

**Extraction results:**
- 8,712 total samples across 9 parquet files (only 9 of 12 downloaded — 3 missing shards)
- 0% skip rate (vs. 75% with the old VAD-based extraction)
- Embedding shape: [N, 256, 1] (256-dim, same as ControlVC D_VECTOR)
- Saved to `embeddings/openvoice_expresso_emb.pt`

**Installation notes:**
- OpenVoice installed via `pip install git+https://github.com/myshell-ai/OpenVoice --no-deps` (PyPI package doesn't exist)
- Manual deps: unidecode, inflect, cn2an, pypinyin, jieba, eng_to_ipa, langid, whisper-timestamped
- Checkpoint auto-downloads to `~/.cache/openvoice_checkpoint/` (122 MB)

### What's Committed vs. Uncommitted
- **Committed (feat/controlvc):** VAE port, Expresso pipeline scripts, portability fixes, .gitignore
- **Committed (feat/f0-style-control):** F0 transform in controlvc.py, anonymizer.py, wrapper.py; inference CLI with F0 presets; fairseq Python 3.11 compat patch
- **Committed (feat/openvoice-expresso):** OpenVoice extraction script (initial version)
- **Not committed:** openvoice.py VAD bypass fix, __init__.py export fix, extraction script pandas update, WORKLOG.md, generated artifacts

---

## 12. Reproduction Commands

```bash
# Setup
cd /Users/steve/UVM-plaid/dp-vc
source .venv/bin/activate
export PYTHONPATH=/Users/steve/UVM-plaid/dp-vc

# ===== OpenVoice Pipeline (recommended) =====

# 1. Extract OpenVoice embeddings from Expresso (~8 min on CPU)
python examples/openvoice_extract_expresso.py \
  --output embeddings/openvoice_expresso_emb.pt \
  --parquet-dir ~/.cache/huggingface/hub/datasets--ylacombe--expresso/snapshots/9fb79a189698de3255eff48edd2bc0d9e487adc0/read

# 2. Train controllable VAE on OpenVoice embeddings
python examples/controlvc_train_vae_expresso.py \
  --embeddings embeddings/openvoice_expresso_emb.pt \
  --output embeddings/openvoice_vae_expresso.pt \
  --epochs 2000 --lr 1e-4

# 3. Run controllable inference (TODO: adapt for OpenVoice)
# python examples/openvoice_infer_controllable.py ...

# ===== ControlVC Pipeline (F0 style control) =====

# 1. Extract ControlVC embeddings from Expresso (~10 min on CPU)
export HF_HUB_DISABLE_XET=1
python examples/controlvc_extract_expresso.py \
  --repo-root /Users/steve/repos/control-vc \
  --output embeddings/controlvc_expresso_emb.pt \
  --parquet-dir ~/.cache/huggingface/hub/datasets--ylacombe--expresso/snapshots/9fb79a189698de3255eff48edd2bc0d9e487adc0/read

# 2. Run F0-based style inference (no VAE needed for F0 control)
python examples/controlvc_infer_controllable.py \
  --repo-root /Users/steve/repos/control-vc \
  --source examples/trump_0.wav \
  --out output/trump_happy.wav \
  --style happy --noise-level 0.5
```

### 0.19 Repository Consolidation Planning Closeout (May 4, branch `integration/research-rollup`)

- Opened the consolidation PR from `integration/research-rollup` into `main`:
  - `https://github.com/uvm-plaid/dpvc/pull/3`
- The accepted sequential research line is now represented as one reviewable PR
  instead of many branches.
- The PR intentionally **excludes** the older side branches:
  - `feat/controlvc`
  - `feat/openvoice-expresso`
  - `feat/f0-style-control`
- Those branches remain historical / review-later branches rather than being
  silently folded into the accepted line.

**Post-rollup outcome**
- PR `#3` was merged and then reverted from upstream `main` at Joe's request so `main` could return to being a stable published-work branch
- The controllable-VAE line now continues on the fork `NonMundaneDev/dpvc`
- Canonical research branch: `research/controllable-vae`
- Current experiment focus: mixed-data pseudo-label teacher / acceptance logic
- Remaining near-term follow-ups:
  - add the short Joe-facing metric guide
  - extend the non-Trump sweep to a larger panel
  - finish the reproducibility checklist / dependency pinning work

**Checked-in plan files**
- `IMPLEMENTATION_PLAN_post-consolidation-next-queue.md`
- `IMPLEMENTATION_PLAN_mixed-data-pseudolabel-teacher.md`
