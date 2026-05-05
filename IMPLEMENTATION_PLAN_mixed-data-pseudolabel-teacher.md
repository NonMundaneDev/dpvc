# Implementation Plan: Mixed-Data Pseudolabel Teacher

This is the current experiment slice running on the canonical research branch:

- `research/controllable-vae`

## 1. Why this branch exists

The first mixed-data branch and the mixed-data quality follow-up answered two
important questions:

- schedule choice alone is not enough,
- and first-pass pseudo-label filtering / labeled-data protection is still not
  enough.

The best early mixed-data condition was:
- `mixed_quality_labeled_guarded`
- recall `18.2%`
- novelty `0.0764`
- WER `0.0978`
- MOS delta `-0.1234`

The expanded rare-supply teacher result now changes the branch status:
- `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup`
- recall `47.0%`
- novelty `0.2995`
- mean styled WER `0.2751`
- MOS delta `-0.2640`

The first inference-side calibration follow-up adds a quality-balanced profile:
- `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard`
- recall `47.0%`
- novelty `0.2726`
- mean styled WER `0.2348`
- MOS delta `-0.2081`
- files with any collapse `20`

That answers the first version of the branch question positively: better rare
pseudo-label supply can move recall well above `18.2%`. The unresolved problem
is learning the recall/quality tradeoff directly instead of depending on
manual per-style inference calibration.

So the next highest-value branch is a **decoder-aware or generated-audio style
objective** that preserves the expanded rare-supply recall gain while repairing
WER/MOS during training.

## 2. Branch context

- canonical research branch: `research/controllable-vae`
- historical experiment branch name: `research/mixed-data-pseudolabel-teacher`

## 3. Core question

**Can a decoder-aware or generated-audio style objective preserve the expanded
rare-supply recall gain while reducing the WER/MOS cost, and can it do so
without depending on hand-authored per-style strength profiles?**

## 4. What stays fixed

To keep this branch interpretable, hold fixed:
- the OpenVoice backend
- the 15-dim controllable VAE framing
- the current mixed-data artifact schema
- the 11-speaker evaluation corpus
- the 4-metric stack:
  - emotion recall / emo_sim
  - novelty
  - WER
  - MOS
- the current comparison references:
  - `combined`
  - `commonvoice_cv500_init`
  - `cv500_ft_short_low_lr`
  - `cv500_rich_free_anchor`
  - `mixed_static_balanced`
  - `mixed_labeled_finish`
  - `mixed_quality_labeled_guarded`

## 5. Working hypotheses

### Hypothesis 1
The current pseudo-label teacher is too conservative or too noisy in the wrong
way, which leaves CommonVoice pseudo-style rows clustered around a narrow
neutral/sad basin.

### Hypothesis 2
Acceptance logic needs to be class-aware, not just confidence-thresholded.
Even a decent teacher can fail if `neutral` and `sad` dominate the accepted
pool.

### Hypothesis 3
Better pseudo labels should help more than another pure schedule tweak, because
Finding 18 already showed that schedule + first-pass filtering only nudges the
result.

## 6. Implementation steps

### Step 1. Freeze the current mixed-data baseline

Use the current best mixed-data references as explicit baselines in the branch
summary:
- `results/eval_mixed_data_summary_pass9.csv`
- `results/eval_mixed_quality_summary.csv`
- `results/eval_nontrump_strength_sweep.csv`

These should remain unchanged so the new branch answers one clean question.

### Step 2. Extend pseudo-label generation

Primary target:
- `scripts/annotate_commonvoice_pseudolabels.py`

Add support for:
- teacher selection / checkpoint selection
- saved teacher metadata in the artifact
- class probability logging instead of only top-1 labels
- optional top-k retention for later filtering decisions

Recommended new controls:
- `--teacher-checkpoint`
- `--teacher-config`
- `--save-logits` or `--save-probs`
- `--top-k`
- `--batch-size`

If the current script becomes too messy, split out:
- `scripts/score_commonvoice_pseudolabels.py`
- `scripts/filter_commonvoice_pseudolabels.py`

Preferred design:
- score first
- filter second

That keeps the teacher outputs reusable across multiple acceptance policies.

### Step 3. Add better acceptance logic to the mixed-data builder

Primary target:
- `scripts/build_mixed_training_set.py`

Add or strengthen support for:
- per-class confidence thresholds
- per-class target counts or caps
- per-class fallback logic when a rare class has too few confident rows
- agreement rules if more than one teacher signal is available later
- a clear accepted vs rejected report per style

Recommended new controls:
- `--pseudo-style-thresholds anger=...,happy=...,sad=...`
- `--commonvoice-style-caps neutral=...,sad=...`
- `--commonvoice-style-targets anger=...,fear=...`
- `--acceptance-policy` with values like:
  - `confidence_only`
  - `threshold_plus_caps`
  - `balanced_targets`

### Step 4. Make the artifact reports stronger

Every mixed-data artifact should keep a richer `mixture_report` with:
- teacher identity / checkpoint
- threshold by style
- accepted count by style
- rejected count by style
- cap-skipped count by style
- fallback-selected count by style
- final pseudo-label distribution
- CommonVoice speaker count
- CommonVoice labeled-row count
- dataset masses used in training

That makes later comparisons much easier and reduces branch archaeology.

### Step 5. Train a focused teacher-quality matrix

Minimum recommended condition family:

1. `mixed_teacher_threshold_balanced`
- improved teacher outputs
- threshold + per-class balancing
- static balanced masses

2. `mixed_teacher_labeled_finish`
- improved teacher outputs
- threshold + per-class balancing
- labeled-finish masses

3. `mixed_teacher_labeled_guarded`
- improved teacher outputs
- threshold + per-class balancing
- strongest labeled-data protection

Optional stretch condition:
4. `mixed_teacher_rare_class_guarded`
- explicitly protects rare pseudo classes such as `anger` and `fear`
- only if the accepted pseudo pool is large enough to justify it

Suggested checkpoints:
- `embeddings/openvoice_vae_mixed_teacher_threshold_balanced.pt`
- `embeddings/openvoice_vae_mixed_teacher_labeled_finish.pt`
- `embeddings/openvoice_vae_mixed_teacher_labeled_guarded.pt`

### Step 6. Generate matched evaluation corpora

Suggested output dirs:
- `output/mixed_teacher_threshold_balanced_eval/`
- `output/mixed_teacher_labeled_finish_eval/`
- `output/mixed_teacher_labeled_guarded_eval/`

### Step 7. Run the full metric stack

Run:
- `examples/eval_emotion.py`
- `examples/eval_novelty.py`
- `examples/eval_wer.py`
- `examples/eval_mos.py`

Compare directly against:
- `mixed_static_balanced`
- `mixed_labeled_finish`
- `mixed_quality_labeled_guarded`
- `combined`

### Step 8. Add a dedicated summarizer for the branch

Recommended new script:
- `scripts/summarize_mixed_teacher_results.py`

Expected outputs:
- `results/eval_mixed_teacher_summary.csv`
- `results/eval_mixed_teacher_collapse.csv`

The summary should explicitly answer:
- did the stronger teacher beat `mixed_quality_labeled_guarded` on recall?
- did it preserve or improve novelty?
- did it preserve enough WER / MOS?
- did it reduce identity collapse or just move the error somewhere else?

## 7. Success criteria

A successful branch should do at least one of these clearly:
- push recall meaningfully above `18.2%`
- improve novelty without losing the mixed-data WER gains
- reduce identity collapse while preserving the current best mixed-data recall

If it fails, that is still useful if the failure is cleanly characterized.

## 8. Validation

Record these explicitly in `WORKLOG.md` at branch closeout:
- `Validation`: The stronger pseudo-label teacher path is reproducible from checked-in scripts.
- `Validation`: Every teacher-quality condition has a named artifact, checkpoint, corpus, and result bundle.
- `Validation`: The comparison explicitly answers whether teacher quality beats the current mixed-data quality baseline.
- `Validation`: The branch isolates pseudo-label teacher / acceptance changes rather than mixing in unrelated architecture changes.

## 9. Follow-up tasks to preserve

If this branch helps only a little:
- compare one-clip-per-speaker vs two-clips-per-speaker CommonVoice sampling
- add prototype-space or teacher-space style targets during mixed-data training
- revisit Expresso label mapping and rare-style preservation again

If this branch works well:
- rerun the non-Trump style-strength sweep on the new best checkpoint
- expand the sweep to a larger panel
- add style presets backed by metrics
- add repeated-seed uncertainty before freezing any paper table

## 10. Current implementation status

The branch now has the first real teacher-focused plumbing in place:

- `scripts/annotate_commonvoice_pseudolabels.py`
  - records `pseudo_style_topk_labels`
  - records `pseudo_style_topk_scores`
  - optionally records mapped per-style score dictionaries
  - records `pseudo_style_teacher` metadata so later artifacts can report which
    teacher produced the pseudo labels

- `scripts/filter_commonvoice_pseudolabels.py`
  - adds a reusable row-level acceptance step
  - supports `confidence_only`, `threshold_plus_caps`, and `balanced_targets`
  - writes `pseudo_style_selected*` fields plus `pseudo_style_filter_report`

- `scripts/build_mixed_training_set.py`
  - now supports `--acceptance-policy`
  - supports `--commonvoice-style-targets`
  - preserves richer `mixture_report` fields including target shortfalls,
    selected reasons, and teacher/filter metadata

- `examples/openvoice_train_vae_mixed.py`
  - prints teacher and acceptance-policy metadata when present in the artifact

Real local validation completed so far:

- `scripts/annotate_commonvoice_pseudolabels.py`
  - smoke-tested on `8` real CommonVoice clips
  - confirmed that `pseudo_style_topk_labels`, `pseudo_style_topk_scores`,
    `pseudo_style_score_map`, `pseudo_style_teacher`, and
    `pseudo_style_report` are written as expected
  - full `1202`-row rescoring now completed to:
    - `embeddings/openvoice_commonvoice_cv500_pseudo_scored.pt`
  - current teacher distribution at the report threshold:
    - `anger=11`
    - `disgust=46`
    - `fear=5`
    - `happy=51`
    - `neutral=640`
    - `sad=336`

- `scripts/filter_commonvoice_pseudolabels.py`
  - validated on the full `1202`-row rescored CommonVoice artifact
  - balanced-target acceptance with the current threshold profile selected:
    - `anger=6`
    - `disgust=36`
    - `fear=4`
    - `happy=42`
    - `neutral=120`
    - `sad=110`

- `scripts/build_mixed_training_set.py`
  - built the first real local mixed teacher base:
    - `embeddings/openvoice_mixed_teacher_base.pt`
  - the rebuilt artifact now carries teacher/report/filter provenance inside
    `mixture_report`, so later checkpoints can be traced back without digging
    through branch history
  - current composition:
    - `500` CommonVoice rows / speakers
    - `232` labeled CommonVoice rows
    - selected CommonVoice pseudo-style counts in the mixed artifact:
      - `anger=5`
      - `disgust=30`
      - `fear=4`
      - `happy=35`
      - `neutral=78`
      - `sad=80`

- `examples/openvoice_train_vae_mixed.py`
  - trained the first teacher-focused checkpoint family from
    `embeddings/openvoice_mixed_teacher_base.pt`
  - real checkpoints now on disk:
    - `embeddings/openvoice_vae_mixed_teacher_threshold_balanced.pt`
    - `embeddings/openvoice_vae_mixed_teacher_labeled_finish.pt`
    - `embeddings/openvoice_vae_mixed_teacher_labeled_guarded.pt`
  - training used the same schedule discipline as the earlier mixed-data line:
    - `mixed_teacher_threshold_balanced` = `static_balanced`
    - `mixed_teacher_labeled_finish` = `labeled_finish` with `--schedule-epochs 1000`
    - `mixed_teacher_labeled_guarded` = `labeled_finish` with `--schedule-epochs 1000` and end masses `CommonVoice=0.10,CREMA-D=0.45,Expresso=0.45`

- `scripts/annotate_commonvoice_latent_prototypes.py`
  - added a genuinely different pseudo-label teacher that scores CommonVoice
    rows against combined-VAE latent style prototypes instead of
    `emotion2vec_plus_large`
  - real prototype artifacts now on disk:
    - `embeddings/openvoice_commonvoice_cv500_pseudo_prototype.pt`
    - `embeddings/openvoice_commonvoice_cv500_pseudo_prototype_filtered.pt`
    - `embeddings/openvoice_mixed_teacher_prototype_base.pt`
    - `embeddings/openvoice_vae_mixed_teacher_prototype_balanced.pt`
  - the prototype condition now has a matched corpus and full metric bundle:
    - `output/mixed_teacher_prototype_balanced_eval/`
    - `results/eval_emotion_mixed_teacher_mixed_teacher_prototype_balanced.csv`
    - `results/eval_novelty_mixed_teacher_mixed_teacher_prototype_balanced.csv`
    - `results/eval_wer_mixed_teacher_mixed_teacher_prototype_balanced.csv`
    - `results/eval_mos_mixed_teacher_mixed_teacher_prototype_balanced.csv`
  - result: `mixed_teacher_prototype_balanced` ties the best mixed-data recall
    at `18.2%`, improves novelty to `0.0854`, and slightly lowers identity /
    mixed collapse, but gives back WER/MOS versus
    `mixed_teacher_threshold_balanced`

- `mixed_teacher_prototype_guarded`
  - built the strong guarded prototype variant from
    `embeddings/openvoice_commonvoice_cv500_pseudo_prototype_filtered.pt`
    using pseudo-confidence scaling, lower CommonVoice pseudo row weight
    (`0.50`), stronger true row weight (`1.50`), and a labeled-heavy finish
    schedule
  - real artifacts now on disk:
    - `embeddings/openvoice_mixed_teacher_prototype_guarded_base.pt`
    - `embeddings/openvoice_vae_mixed_teacher_prototype_guarded.pt`
  - the guarded prototype condition now has a matched corpus and full metric
    bundle:
    - `output/mixed_teacher_prototype_guarded_eval/`
    - `results/eval_emotion_mixed_teacher_mixed_teacher_prototype_guarded.csv`
    - `results/eval_novelty_mixed_teacher_mixed_teacher_prototype_guarded.csv`
    - `results/eval_wer_mixed_teacher_mixed_teacher_prototype_guarded.csv`
    - `results/eval_mos_mixed_teacher_mixed_teacher_prototype_guarded.csv`
  - result: `mixed_teacher_prototype_guarded` ties the `18.2%` recall ceiling
    and improves WER versus the unguarded prototype (`0.0920` vs `0.1009`),
    but erases the prototype novelty advantage (`0.0761` vs `0.0854`) and
    worsens identity/mixed collapse

- `scripts/combine_commonvoice_pseudolabel_teachers.py`
  - added a reusable hybrid teacher combiner that merges filtered emotion2vec
    CommonVoice labels with filtered combined-VAE latent prototype labels
  - tested `prototype_extra_priority`, where emotion2vec supplies canonical
    emotion rows and the prototype teacher supplies `confused`, `enunciated`,
    and `whisper`
  - real hybrid artifacts now on disk:
    - `embeddings/openvoice_commonvoice_cv500_pseudo_hybrid_extra_priority.pt`
    - `embeddings/openvoice_mixed_teacher_hybrid_extra_base.pt`
    - `embeddings/openvoice_vae_mixed_teacher_hybrid_extra_balanced.pt`
  - the hybrid condition now has a matched corpus and full metric bundle:
    - `output/mixed_teacher_hybrid_extra_balanced_eval/`
    - `results/eval_emotion_mixed_teacher_mixed_teacher_hybrid_extra_balanced.csv`
    - `results/eval_novelty_mixed_teacher_mixed_teacher_hybrid_extra_balanced.csv`
    - `results/eval_wer_mixed_teacher_mixed_teacher_hybrid_extra_balanced.csv`
    - `results/eval_mos_mixed_teacher_mixed_teacher_hybrid_extra_balanced.csv`
  - result: `mixed_teacher_hybrid_extra_balanced` produces the best
    mixed-teacher novelty so far (`0.0860`) and slightly fewer files with any
    collapse, but recall falls back to `16.7%` and MOS worsens, so it is a
    useful tradeoff/negative rather than the new overall reference

- style-space distillation follow-up
  - extended `examples/openvoice_train_vae_mixed.py` and `dpvc/utils.py` so
    mixed-data training can use a frozen VAE teacher as a continuous style-space
    target on selected encoder mean dimensions
  - real style-distillation artifacts now on disk:
    - `embeddings/openvoice_vae_mixed_teacher_hybrid_style_distill_balanced.pt`
    - `output/mixed_teacher_hybrid_style_distill_balanced_eval/`
    - `results/eval_emotion_mixed_teacher_mixed_teacher_hybrid_style_distill_balanced.csv`
    - `results/eval_novelty_mixed_teacher_mixed_teacher_hybrid_style_distill_balanced.csv`
    - `results/eval_wer_mixed_teacher_mixed_teacher_hybrid_style_distill_balanced.csv`
    - `results/eval_mos_mixed_teacher_mixed_teacher_hybrid_style_distill_balanced.csv`
  - tested command shape:
    - `--style-teacher-checkpoint embeddings/openvoice_vae_combined.pt`
    - `--style-teacher-weight 0.25`
    - `--style-teacher-datasets CommonVoice`
    - `--style-teacher-dims 0-8`
  - result: `mixed_teacher_hybrid_style_distill_balanced` preserves the hard
    hybrid novelty gain (`0.0861`), improves MOS delta (`-0.1072` vs.
    `-0.1190`), and reduces collapse counts, but recall remains `16.7%`

- style-teacher weight sweep
  - tested the same continuous style-space objective at teacher weights `0.10`
    and `0.50`, keeping the hybrid mixed artifact, frozen combined-VAE teacher,
    style dims `0-8`, `CommonVoice` teacher rows, static balanced schedule,
    inference corpus, and four-metric stack fixed
  - real artifacts now on disk:
    - `embeddings/openvoice_vae_mixed_teacher_hybrid_style_distill_w010_balanced.pt`
    - `embeddings/openvoice_vae_mixed_teacher_hybrid_style_distill_w050_balanced.pt`
    - `output/mixed_teacher_hybrid_style_distill_w010_balanced_eval/`
    - `output/mixed_teacher_hybrid_style_distill_w050_balanced_eval/`
    - `results/eval_emotion_mixed_teacher_mixed_teacher_hybrid_style_distill_w010_balanced.csv`
    - `results/eval_novelty_mixed_teacher_mixed_teacher_hybrid_style_distill_w010_balanced.csv`
    - `results/eval_wer_mixed_teacher_mixed_teacher_hybrid_style_distill_w010_balanced.csv`
    - `results/eval_mos_mixed_teacher_mixed_teacher_hybrid_style_distill_w010_balanced.csv`
    - `results/eval_emotion_mixed_teacher_mixed_teacher_hybrid_style_distill_w050_balanced.csv`
    - `results/eval_novelty_mixed_teacher_mixed_teacher_hybrid_style_distill_w050_balanced.csv`
    - `results/eval_wer_mixed_teacher_mixed_teacher_hybrid_style_distill_w050_balanced.csv`
    - `results/eval_mos_mixed_teacher_mixed_teacher_hybrid_style_distill_w050_balanced.csv`
  - result: all three tested weights (`0.10`, `0.25`, `0.50`) remain at
    `16.7%` recall; `0.50` improves WER (`0.0821`) and identity/mixed collapse
    (`56` / `48`), while `0.25` remains the stronger novelty/MOS tradeoff

- target-dimension style-teacher mask follow-up
  - extended the mixed-data trainer so style-teacher loss can be applied only
    to each row's accepted style dimension, with optional accepted-label
    masking, per-style row weights, and pseudo-label confidence scaling
  - new public training flags:
    - `--style-teacher-target-mode {all_dims,target_dim}`
    - `--style-teacher-require-label`
    - `--style-teacher-style-weights`
    - `--style-teacher-confidence-power`
  - real artifacts now on disk:
    - `embeddings/openvoice_vae_mixed_teacher_hybrid_style_distill_targetmask_balanced.pt`
    - `output/mixed_teacher_hybrid_style_distill_targetmask_balanced_eval/`
    - `results/eval_emotion_mixed_teacher_mixed_teacher_hybrid_style_distill_targetmask_balanced.csv`
    - `results/eval_novelty_mixed_teacher_mixed_teacher_hybrid_style_distill_targetmask_balanced.csv`
    - `results/eval_wer_mixed_teacher_mixed_teacher_hybrid_style_distill_targetmask_balanced.csv`
    - `results/eval_mos_mixed_teacher_mixed_teacher_hybrid_style_distill_targetmask_balanced.csv`
  - result: target masking, per-style row weights, and confidence scaling still
    remain at `16.7%` recall; the run preserves similar novelty (`0.0852`) but
    worsens WER/MOS versus the best global `0.25` style-distillation condition

- per-style diagnostic follow-up
  - added `scripts/analyze_mixed_teacher_style_diagnostics.py`
  - joined label supply, teacher encoder means, student encoder means,
    generated-output metrics, and collapse flags for
    `mixed_teacher_hybrid_style_distill_targetmask_balanced`
  - real diagnostic artifacts now on disk:
    - `results/eval_mixed_teacher_style_diagnostics_targetmask.csv`
    - `results/eval_mixed_teacher_style_diagnostics_targetmask.md`
  - result: canonical emotion pseudo labels often do not have teacher target-dim
    dominance (`anger=0.0000`, `fear=0.0000`, `happy=0.1000` teacher top1
    rates), rare classes are undersupplied (`anger=4`, `fear=4` active rows),
    and `sad` can align latently while still decoding to neutral-classified
    audio

- labeled-first curriculum follow-up
  - added a `labeled_warmup` mixed-data schedule that starts with CREMA-D and
    Expresso only, then ramps CommonVoice back to a balanced mix
  - added `--style-teacher-weight-final` so teacher-style loss can ramp from
    `0.0` to `0.25` instead of being active from the first epoch
  - real artifacts now on disk:
    - `embeddings/openvoice_vae_mixed_teacher_hybrid_style_distill_labeled_warmup.pt`
    - `output/mixed_teacher_hybrid_style_distill_labeled_warmup_eval/`
    - `results/eval_emotion_mixed_teacher_mixed_teacher_hybrid_style_distill_labeled_warmup.csv`
    - `results/eval_novelty_mixed_teacher_mixed_teacher_hybrid_style_distill_labeled_warmup.csv`
    - `results/eval_wer_mixed_teacher_mixed_teacher_hybrid_style_distill_labeled_warmup.csv`
    - `results/eval_mos_mixed_teacher_mixed_teacher_hybrid_style_distill_labeled_warmup.csv`
    - `results/eval_mixed_teacher_style_diagnostics_labeled_warmup.csv`
    - `results/eval_mixed_teacher_style_diagnostics_labeled_warmup.md`
  - result: `mixed_teacher_hybrid_style_distill_labeled_warmup` improves
    novelty (`0.0930`) and reduces identity/any-collapse counts (`54` / `61`),
    but recall remains `16.7%`, so schedule-only curriculum with the current
    teacher is not enough

- listening and rare-supply audit follow-up
  - added `scripts/build_listening_report.py` so every generated corpus can
    produce an HTML listening report plus subjective-rating CSV
  - generated:
    - `results/listening_mixed_teacher_hybrid_style_distill_labeled_warmup.html`
    - `results/listening_mixed_teacher_hybrid_style_distill_labeled_warmup_ratings.csv`
  - added `scripts/audit_commonvoice_pseudolabel_supply.py` and generated:
    - `results/commonvoice_pseudolabel_supply_audit.csv`
    - `results/commonvoice_pseudolabel_supply_audit.md`
  - result: the current local CommonVoice subset has only `1202` validated rows,
    and the hybrid artifact selects only `anger=5` and `fear=4` before mixed
    speaker-first sampling, so the rare-class bottleneck is data supply rather
    than just loss weighting

- CommonVoice rare-class supply preflight
  - added `scripts/plan_commonvoice_rare_supply_expansion.py` so the data-first
    rare-supply path has a reproducible go/no-go gate before extraction,
    scoring, filtering, hybrid combining, mixed-artifact construction, or
    training
  - downloaded/build an expanded local CommonVoice corpus at
    `/Users/steve/datasets/cv-corpus-21.0-2025-03-14/en` from the validated TSV
    plus first English validated audio shard; `validated_full.tsv` preserves
    full metadata and active `validated.tsv` is filtered to locally available
    clips
  - generated:
    - `results/commonvoice_rare_supply_expansion_preflight.json`
    - `results/commonvoice_rare_supply_expansion_preflight.md`
  - result: literal `/data/cv-corpus-21.0-2025-03-14/en` is not creatable in
    this macOS session because `/` is read-only, but the expanded user-space
    corpus now passes the gate with `40000` usable rows / `20537` speakers
  - based on checked-in selected rare-label rates, a credible rare-supply
    extraction should target at least `22538` usable CommonVoice rows before
    another model run

- expanded CommonVoice extraction and resumable teacher scoring
  - extracted the expanded local corpus into:
    - `embeddings/openvoice_commonvoice_cvrare_expanded_emb.pt`
  - validated:
    - `25910` embeddings
    - `13308` unique speakers
    - tensor shape `(25910, 256, 1)`
    - `0` missing clip files
    - `0` unreadable clip files
  - upgraded `scripts/annotate_commonvoice_pseudolabels.py` for expanded
    teacher runs:
    - `--batch-size`
    - `--checkpoint-every`
    - `--checkpoint-path`
    - automatic checkpoint resume
    - `--no-resume`
    - `--fail-on-error`
    - per-row `pseudo_style_error`
    - `--stop-when-accepted-targets anger=50,fear=50`
  - smoke-tested the scorer:
    - 16-row run wrote checkpoint + final artifact and resumed with `0`
      pending rows
    - 64-row timing run confirmed the full emotion2vec teacher pass remains
      multi-hour, so checkpoint/resume and target-seeking are required
  - completed target-seeking command:
    - `.venv/bin/python scripts/annotate_commonvoice_pseudolabels.py --embeddings embeddings/openvoice_commonvoice_cvrare_expanded_emb.pt --output embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_scored.pt --save-style-score-map --report-threshold 0.60 --batch-size 4 --checkpoint-every 500 --stop-when-accepted-targets anger=50,fear=50`

Immediate next execution steps on this branch:

1. Treat the first decoder-prototype pilot and its guarded readout as
   implemented cautionary baselines for decoder-aware training: the unguarded
   run reaches `42.4%` recall and `0.3008` novelty gain, and the guarded run
   improves WER/MOS to `0.2592` / `-0.1787`, but neither beats the expanded
   rare-supply `sad/enunciated` guard on recall, WER, and collapse.
2. Design the safer follow-up objective: lower decoder-prototype weights,
   style-specific prototype subsets, or offline generated-audio calibration
   from emotion2vec / WER / MOS feedback.
3. Use
   `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup` as the strongest
   checked-in controllability/novelty reference (`47.0%` recall, `0.2995`
   novelty gain), while keeping `combined` as the cleanest original quality
   baseline.
4. Prioritize perceptual review from
   `results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.html`,
   with `results/listening_mixed_teacher_cvrare_decoder_proto_labeled_warmup.html`
   as the decoder-prototype comparison.
5. Compare one-clip-per-speaker versus two-clips-per-speaker CommonVoice
   sampling only after the decoder-aware objective is tested; the current
   selected-pseudo preservation mode already keeps rare selected rows without
   bloating all speakers to three clips.
6. Compare prototype-only versus hybrid teacher targets inside the same
   decoder-aware objective only after diagnostics confirm whether the failure is
   teacher calibration or decoder/output alignment.
7. Use `scripts/run_generated_audio_eval_suite.py` as the default closeout path
   for generated-audio corpora so emotion, novelty, WER, MOS, summary/collapse,
   and listening artifacts are regenerated consistently.
8. Convert the hand-authored `cvrare_sad_enunc_guard` profile into a small
   reproducible grid/optimizer only after the decoder-aware objective baseline
   is defined.

Expanded rare-supply gate status:

- `embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_scored.pt` annotated
  `6380/25910` rows and met `anger=50,fear=50` at the target threshold.
- `embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_hybrid_extra_priority.pt`
  selected `645` pseudo rows across the nine unified styles.
- `scripts/build_mixed_training_set.py --commonvoice-preserve-selected-pseudo`
  keeps one-clip-per-speaker speaker breadth while preserving selected pseudo
  rows that would otherwise be dropped by the clip cap.
- `embeddings/openvoice_mixed_teacher_cvrare_hybrid_extra_base.pt` has
  `14195` rows, `13308` CommonVoice speakers, and selected CommonVoice counts
  `anger=50` / `fear=50`.
- Generated-audio evaluation now validates the data-side hypothesis:
  `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup` reaches `47.0%`
  recall and `0.2995` novelty gain, but mean styled WER rises to `0.2751` and
  MOS delta falls to `-0.2640`.
- Inference-side per-style calibration now validates a stronger
  quality-balanced readout:
  `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard`
  preserves `47.0%` recall while improving mean styled WER to `0.2348`, MOS
  delta to `-0.2081`, and files with any collapse to `20`.
- The first decoder-prototype pilot is implemented and evaluated:
  `mixed_teacher_cvrare_decoder_proto_labeled_warmup` reaches `42.4%` recall,
  `0.3008` novelty gain, `0.2863` mean styled WER, `-0.2148` MOS delta, and
  `27` files with any collapse. It is a reproducible negative/cautionary
  result: decoded-embedding prototype matching alone did not learn the
  inference-side guard's quality-balanced repair.
- Applying the same `cvrare_sad_enunc_guard` profile to the decoder-prototype
  checkpoint improves WER/MOS (`0.2592`, `-0.1787`) but leaves recall at
  `42.4%` and increases files with any collapse to `28`, so inference
  calibration repairs only part of the decoder-prototype failure.
