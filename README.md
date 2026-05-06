# Differentially Private Anonymization via Voice Control

This repository provides a library for defining differentially private speaker anonymization systems using existing voice control models. The approach works for any voice control system that separates utterance information into constant-length speaker information (e.g. a speaker embedding) and time-varying content information (e.g. semantic features).

[Click here for full documentation](https://jnear.w3.uvm.edu/dpvc/)

## Current work — controllable DP voice conversion

Upstream `main` is now back to Joe's intended role: a stable reflection of the
published work. Ongoing controllable-VAE research has moved to the fork:

- [NonMundaneDev/dpvc](https://github.com/NonMundaneDev/dpvc)

Canonical research branch:

- **`research/controllable-vae`**

Current experiment focus on that branch:

- **mixed-data generated-audio calibration and style-specific reranking**

Immediate next queue:

1. use `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` as the current quality-balanced profile, `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup` as the strongest high-novelty result, and `combined` as the cleanest original quality baseline
2. treat the decoder-prototype, failure-targeted target-dim, and anti-neutral prototype-margin runs as verified cautionary baselines, not as the new reference
3. run perceptual review from the A/B dashboard `results/listening_mixed_teacher_cvrare_strength_grid_ab_review.html`, starting with the priority rows in `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority.md`
4. only promote style-specific presets after perceptual review; the grid improves some hard-style rows but is not yet a safe global strength increase
5. then add the Joe-facing metric guide, broaden the non-Trump sweep, and finish the reproducibility checklist / dependency pinning work

The dedicated next-step plans live in:

- **[`IMPLEMENTATION_PLAN_post-consolidation-next-queue.md`](IMPLEMENTATION_PLAN_post-consolidation-next-queue.md)** — historical record of the consolidation / rollback sequence
- **[`IMPLEMENTATION_PLAN_mixed-data-pseudolabel-teacher.md`](IMPLEMENTATION_PLAN_mixed-data-pseudolabel-teacher.md)** — the technical plan for the current experiment slice on `research/controllable-vae`

We’ve extended the library with a **controllable** VAE that exposes 9 style knobs (anger, confused, disgust, enunciated, fear, happy, neutral, sad, whisper) on top of the DP anonymization pipeline. Primary entry points:

- **[`examples/README.md`](examples/README.md)** — end-to-end reproduction guide (extraction → training → controllable inference → evaluation).
- **[`FINDINGS.md`](FINDINGS.md)** — 35 paper-facing findings with methodology and per-row takeaways.
- **[`WORKLOG.md`](WORKLOG.md)** — roadmap and progress tracking.
- **[`results/`](results/)** — raw evaluation CSVs (emotion2vec Recall/emo_sim, WER, predicted MOS) backing the findings.

OpenVoice is the **canonical controllable pipeline**. ControlVC remains in the
repository as a useful DP baseline and wrapper reference, but not as the
recommended path for style control.

Current best checked-in result: the expanded rare-supply mixed teacher is now
the strongest controllability / novelty condition, while the original
**combined** OpenVoice model remains the cleanest quality baseline. The later
CommonVoice finetune, objective, rich-objective, and
partial-label studies sharpened that conclusion by showing that neither simple
gentler CommonVoice finetuning, nor simple scalar objective reweighting, nor
the first richer teacher/anchor CommonVoice objectives, nor validation-scale
weak-label CommonVoice pretraining recover the combined model's tradeoff. The
mixed-data schedule branch and mixed-data pseudo-label quality follow-up then
showed that combined-data training can improve WER and move recall slightly,
but still does not match the `combined` model's control/novelty balance. The
first mixed-data pseudo-label teacher family narrows that story further:
`mixed_teacher_threshold_balanced` matches the best mixed-data recall
(`18.2%`) while improving WER, MOS, novelty, and identity collapse versus
`mixed_quality_labeled_guarded`, but it still does not beat the `combined`
model or break the mixed-data recall ceiling. A softer mapped-score
teacher-agreement follow-up (`mixed_teacher_mapped015_balanced`) nudges
novelty a bit higher (`0.0818`), but drops back to `16.7%` recall and gives
back WER/MOS. A genuinely different combined-VAE latent prototype teacher
(`mixed_teacher_prototype_balanced`) restores the `18.2%` recall tie, improves
novelty to `0.0854`, and slightly lowers identity / mixed collapse, but gives
back WER and MOS versus `mixed_teacher_threshold_balanced`. At that stage,
`mixed_teacher_threshold_balanced` was the best overall mixed-data teacher
reference and `mixed_teacher_prototype_balanced` the best novelty/coverage
candidate. The guarded prototype follow-up (`mixed_teacher_prototype_guarded`)
keeps the `18.2%` recall tie and repairs WER somewhat (`0.0920` vs `0.1009`),
but it erases the prototype novelty advantage (`0.0761` vs `0.0854`) and
worsens identity collapse, which set up the multi-teacher test rather than
another simply weaker prototype-supervision run. The
hybrid prototype+emotion2vec follow-up (`mixed_teacher_hybrid_extra_balanced`)
then tests that multi-teacher hypothesis directly: it reaches the best
mixed-teacher novelty so far (`0.0860`) and slightly lowers files with any
collapse, but drops recall back to `16.7%` and worsens MOS delta (`-0.1190`).
That kept `mixed_teacher_threshold_balanced` as the historical mixed-data
teacher reference and motivated a richer continuous teacher objective. The
style-space distillation follow-up
(`mixed_teacher_hybrid_style_distill_balanced`) uses the frozen combined VAE as
a teacher on CommonVoice style dims `0-8`; it preserves the hybrid novelty
gain (`0.0861`), improves MOS delta versus hard hybrid labels (`-0.1072` vs
`-0.1190`), and reduces identity / mixed / any-collapse counts (`58` / `50` /
`63`), but recall remains stuck at `16.7%`. That makes style-space
distillation a better tradeoff than hard hybrid row labels, but not the
recall breakthrough. The follow-up global weight sweep (`0.10`, `0.25`,
`0.50`) confirmed that scalar teacher-loss calibration is not enough: all three
weights remain at `16.7%` recall. Weight `0.50` improves WER (`0.0821`) and
identity/mixed collapse (`56` / `48`), while `0.25` remains the better
novelty/MOS tradeoff. The target-dimension style-teacher mask follow-up
(`mixed_teacher_hybrid_style_distill_targetmask_balanced`) then tested
class-specific teacher loss, per-style row weights, and confidence scaling; it
also stayed at `16.7%` recall, preserved similar novelty (`0.0852`), and
worsened WER/MOS versus global `0.25` style distillation. The next useful step
was diagnostic or curriculum-driven rather than another latent-only mask/weight
variant. The per-style diagnostic then localized the failure: canonical
CommonVoice pseudo labels often do not make the intended frozen-teacher style
dimension dominant (`anger=0.0000`, `fear=0.0000`, `happy=0.1000` teacher
target-top1 rates), `anger` and `fear` have only `4` active teacher rows each,
and `sad` can align latently while still decoding to neutral-classified audio.
That made the next training move a labeled-first curriculum. The curriculum
condition (`mixed_teacher_hybrid_style_distill_labeled_warmup`) protects
CREMA-D / Expresso style axes first, then ramps CommonVoice teacher geometry
from `0.0` to `0.25`; it improves secondary axes (`0.0930` novelty gain, `54`
identity-collapse files, `61` files with any collapse), but recall remains
`16.7%`. The rare-class supply preflight then made the data-side constraint
explicit and was rerun after expanding the local CommonVoice corpus. Literal
`/data/cv-corpus-21.0-2025-03-14/en` is not creatable in this macOS session
because the root filesystem is read-only, so the usable stable local corpus is
`/Users/steve/datasets/cv-corpus-21.0-2025-03-14/en`. It preserves the full
English metadata as `validated_full.tsv`, uses an active `validated.tsv`
filtered to the locally extracted `40000` MP3 clips from the first validated
audio shard, and yields `40000` usable rows / `20537` speakers. That clears the
current `22538` usable-row rare-supply preflight target. OpenVoice extraction
from that expanded corpus now produced
`embeddings/openvoice_commonvoice_cvrare_expanded_emb.pt` with `25910`
embeddings, `13308` unique speakers, and zero missing/unreadable clips. The
resumable target-seeking emotion2vec scorer then annotated `6380/25910` rows,
the expanded hybrid teacher artifact selected `645` pseudo-labeled rows, and
the final mixed artifact
`embeddings/openvoice_mixed_teacher_cvrare_hybrid_extra_base.pt` preserves
`13308` CommonVoice speakers while keeping `anger=50` and `fear=50` CommonVoice
pseudo rows in the training set. The expanded rare-supply checkpoint
(`mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup`) is now evaluated:
it reaches `47.0%` emotion recall and `0.2995` novelty gain, beating both the
prior mixed-teacher recall ceiling (`18.2%`) and the combined-only novelty
reference (`0.2599`). The tradeoff is clear: mean styled WER rises to `0.2751`
and MOS delta falls to `-0.2640`, so this is the strongest controllability /
novelty result so far but not yet the cleanest quality result. A follow-up
per-style strength calibration shows that the narrow
`mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard`
profile keeps the `47.0%` recall gain while improving mean styled WER to
`0.2348`, MOS delta to `-0.2081`, and files with any collapse to `20`; this is
the recommended quality-balanced listening/eval profile for the expanded
checkpoint. The broader `content_guard` repairs WER/MOS more aggressively but
drops recall to `40.9%`, so it is a diagnostic profile rather than the new
reference. The first decoder-prototype training pilot
(`mixed_teacher_cvrare_decoder_proto_labeled_warmup`) adds a real
decoder-aware objective, but the initial global-strength result does not beat
the inference-side guard: recall falls to `42.4%`, WER rises to `0.2863`, MOS
delta is `-0.2148`, and files with any collapse rise to `27`, even though
novelty remains high at `0.3008`. Applying the same `sad/enunciated` inference
guard to the decoder-prototype checkpoint improves WER to `0.2592` and MOS
delta to `-0.1787`, but recall remains `42.4%` and collapse rises to `28`
files. Lowering the decoder-prototype final weight to `0.005`
(`mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup`) also fails to
recover the reference tradeoff: recall remains `42.4%`, novelty is high
(`0.3032`), WER is still worse (`0.2782`), MOS delta is `-0.2122`, and any
collapse remains higher (`26` files). This makes the decoder-prototype family
a useful negative/cautionary result and keeps the expanded rare-supply
`sad/enunciated` guard as the current quality-balanced reference. The new
generated-audio failure-mining artifact confirms that reference has the lowest
row-level failure score and localizes the remaining hard styles to `disgust`,
`fear`, and `anger`. The follow-up failure-conditioned selector marks `anger`
and `disgust` ready for a positive target objective and blocks `fear` because
there are no clean fear targets under the current reference. That targeted
`anger`/`disgust` follow-up has now been tested:
`mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` keeps
novelty high (`0.2960`) and reduces content collapse to `1`, but recall drops
to `39.4%` and style-to-neutral collapse rises to `26`. The current
`sad/enunciated` guard therefore remains the quality-balanced reference, and
the next training direction should add an explicit anti-neutral or
generated-audio-calibrated output signal rather than more target-dim teacher
pressure. The first anti-neutral prototype-margin follow-up
(`mixed_teacher_cvrare_antineutral_labeled_warmup`) tests that hypothesis in
embedding space and is also negative: recall reaches only `40.9%`, novelty
stays high (`0.2962`), WER is `0.2609`, and style-to-neutral collapse remains
high at `25`. That rules out another tempting proxy and moves the next best
work toward actual generated-audio strength grids or reranking. The first
generated-audio style-strength grid is now checked in for `anger`, `disgust`,
and `fear`: `anger_s10` improves anger recall from `1/11` to `3/11`,
`fear_s7p5` improves fear recall from `3/11` to `6/11` but with high WER
(`0.5231`), and `disgust_s10` raises novelty without improving recall while
severely hurting MOS (`-0.6039`). This makes the grid a useful
audio-calibrated reranking artifact, not a new universal default. The non-Trump
strength sweep adds a narrower inference-side result:
`5.0` remains the safest default, `7.5` is a useful stronger option for styles
like `whisper` and `confused`, and `10.0-12.5` look more like high-novelty
specialized settings than new defaults. The main summary artifacts are:

- [`results/eval_ablation_summary_pass4.csv`](results/eval_ablation_summary_pass4.csv)
- [`results/eval_commonvoice_finetune_summary_pass5.csv`](results/eval_commonvoice_finetune_summary_pass5.csv)
- [`results/eval_commonvoice_objective_summary_pass6.csv`](results/eval_commonvoice_objective_summary_pass6.csv)
- [`results/eval_commonvoice_rich_objectives_summary_pass7.csv`](results/eval_commonvoice_rich_objectives_summary_pass7.csv)
- [`results/eval_commonvoice_partial_label_summary_pass8.csv`](results/eval_commonvoice_partial_label_summary_pass8.csv)
- [`results/eval_mixed_data_summary_pass9.csv`](results/eval_mixed_data_summary_pass9.csv)
- [`results/eval_mixed_quality_summary.csv`](results/eval_mixed_quality_summary.csv)
- [`results/eval_mixed_teacher_summary.csv`](results/eval_mixed_teacher_summary.csv)
- [`results/eval_mixed_teacher_style_diagnostics_targetmask.md`](results/eval_mixed_teacher_style_diagnostics_targetmask.md)
- [`results/eval_mixed_teacher_style_diagnostics_labeled_warmup.md`](results/eval_mixed_teacher_style_diagnostics_labeled_warmup.md)
- [`results/eval_mixed_teacher_style_diagnostics_cvrare_labeled_warmup.md`](results/eval_mixed_teacher_style_diagnostics_cvrare_labeled_warmup.md)
- [`results/eval_mixed_teacher_cvrare_strength_profiles_summary.md`](results/eval_mixed_teacher_cvrare_strength_profiles_summary.md)
- [`results/eval_mixed_teacher_cvrare_decoder_proto_summary.md`](results/eval_mixed_teacher_cvrare_decoder_proto_summary.md)
- [`results/eval_mixed_teacher_generated_audio_failure_mining.md`](results/eval_mixed_teacher_generated_audio_failure_mining.md)
- [`results/eval_mixed_teacher_failure_conditioned_targets.md`](results/eval_mixed_teacher_failure_conditioned_targets.md)
- [`results/eval_mixed_teacher_cvrare_strength_grid_ranking.md`](results/eval_mixed_teacher_cvrare_strength_grid_ranking.md)
- [`results/listening_mixed_teacher_cvrare_strength_grid_ab_review.html`](results/listening_mixed_teacher_cvrare_strength_grid_ab_review.html)
- [`results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority.md`](results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority.md)
- [`results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.html`](results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.html)
- [`results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.html`](results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.html)
- [`results/listening_mixed_teacher_cvrare_decoder_proto_labeled_warmup.html`](results/listening_mixed_teacher_cvrare_decoder_proto_labeled_warmup.html)
- [`results/listening_mixed_teacher_cvrare_decoder_proto_labeled_warmup_sad_enunc_guard.html`](results/listening_mixed_teacher_cvrare_decoder_proto_labeled_warmup_sad_enunc_guard.html)
- [`results/listening_mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup.html`](results/listening_mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup.html)
- [`results/listening_mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup.html`](results/listening_mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup.html)
- [`results/listening_mixed_teacher_cvrare_antineutral_labeled_warmup.html`](results/listening_mixed_teacher_cvrare_antineutral_labeled_warmup.html)
- [`results/listening_mixed_teacher_cvrare_strength_grid_anger_s10.html`](results/listening_mixed_teacher_cvrare_strength_grid_anger_s10.html)
- [`results/listening_mixed_teacher_cvrare_strength_grid_disgust_s10.html`](results/listening_mixed_teacher_cvrare_strength_grid_disgust_s10.html)
- [`results/listening_mixed_teacher_cvrare_strength_grid_fear_s7p5.html`](results/listening_mixed_teacher_cvrare_strength_grid_fear_s7p5.html)
- [`configs/style_strength_profiles/cvrare_sad_enunc_guard.json`](configs/style_strength_profiles/cvrare_sad_enunc_guard.json)
- [`results/commonvoice_pseudolabel_supply_audit_rare_supply.md`](results/commonvoice_pseudolabel_supply_audit_rare_supply.md)
- [`results/commonvoice_rare_supply_expansion_preflight.md`](results/commonvoice_rare_supply_expansion_preflight.md)
- [`results/eval_nontrump_strength_sweep.csv`](results/eval_nontrump_strength_sweep.csv)
- [`results/eval_nontrump_strength_sweep_summary.md`](results/eval_nontrump_strength_sweep_summary.md)

## Installation

Clone this repository, then install with the extras you need. The active
OpenVoice path is covered by package extras; the ControlVC baseline has a
separate setup guide because it depends on an external repo plus Python 3.10 /
fairseq compatibility work.

```bash
# Core library only
pip install -e .

# + OpenVoice backend (required for the controllable pipeline)
pip install -e ".[openvoice]"

# + Expresso dataset extraction
pip install -e ".[openvoice,expresso]"

# + Evaluation pipeline (emotion2vec, novelty, Whisper WER, predicted MOS)
pip install -e ".[openvoice,expresso,eval]"
```

Tested OpenVoice stabilization and reproducibility environment in `.venv`:

- `torch==2.9.1`
- `torchaudio==2.9.1`
- `numpy==2.3.5`
- `librosa==0.9.1`
- `soundfile==0.13.1`
- `datasets==4.8.4`
- `pandas==3.0.2`
- `funasr==1.3.1`
- `openai-whisper==20250625`
- `jiwer==4.0.0`

For ControlVC-specific setup, use [`docs/controlvc_setup.md`](docs/controlvc_setup.md).

## Example: basic DP anonymization (OpenVoice)

```python
import dpvc
vc_wrapper = dpvc.OpenVoiceWrapper()
anonymizer = dpvc.Anonymizer(vc_wrapper)
anonymizer.anonymize(src_path, output_path, noise_level=1.0)
```

`src_path` is an input .wav, `output_path` is the anonymized output, and `noise_level` controls the magnitude of DP noise added to the speaker embedding.

See also:

- `examples/openvoice_inference.py` — basic anonymization (no style control).
- `examples/openvoice_train_vae.py` — train a custom DP-VAE for the anonymizer.
- `examples/openvoice_infer_controllable.py` — **controllable** style-aware inference (the current headline flow; see [`examples/README.md`](examples/README.md) for the full pipeline).
- `examples/openvoice_extract_commonvoice.py` + `examples/openvoice_pretrain_vae_commonvoice.py` — Common Voice pretraining path, including validation-scale weak supervision from metadata and pseudo labels.
- `scripts/build_mixed_training_set.py` + `examples/openvoice_train_vae_mixed.py` — mixed-data bootstrap path that combines pseudo-labeled CommonVoice with labeled CREMA-D and Expresso under schedule-controlled sampling, including optional style-space teacher distillation via `--style-teacher-checkpoint`, `--style-teacher-weight`, `--style-teacher-weight-final`, `--style-teacher-dims`, `--style-teacher-datasets`, `--style-teacher-target-mode`, `--style-teacher-require-label`, `--style-teacher-style-weights`, and `--style-teacher-confidence-power`.
- `scripts/prepare_commonvoice_subset.py` — helper for turning downloaded Common Voice shards into a filtered local `validated.tsv` + `clips/` subset.
- `scripts/annotate_commonvoice_pseudolabels.py` — adds confidence-scored pseudo-style labels to a Common Voice embedding artifact, with batch size, checkpoint/resume, per-row error recording, and target-seeking stop controls for expanded-corpus teacher runs.
- `scripts/annotate_commonvoice_latent_prototypes.py` — scores Common Voice rows against combined-VAE latent style prototypes as an alternate pseudo-label teacher.
- `scripts/combine_commonvoice_pseudolabel_teachers.py` — combines filtered emotion2vec and latent-prototype CommonVoice pseudo labels into a reusable hybrid teacher artifact.
- `scripts/audit_commonvoice_pseudolabel_supply.py` — audits rare-style pseudo-label supply before more CommonVoice weighting/curriculum experiments.
- `scripts/plan_commonvoice_rare_supply_expansion.py` — preflights local CommonVoice corpus size/speaker availability and emits a go/no-go command plan before rebuilding rare-class pseudo labels.
- `scripts/analyze_mixed_teacher_style_diagnostics.py` — joins label supply, teacher/student latent geometry, generated metrics, and collapse rows for mixed-teacher conditions.
- `scripts/build_listening_report.py` — creates an HTML listening report plus subjective-rating CSV from any generation manifest.
- `scripts/prepare_ablation_embeddings.py` — builds the `cremad_only` and `expresso_only` evaluation ablation datasets in the unified label format.
- `scripts/run_ablation_inference.py` — generates the evaluation ablation matrix corpora, including the naive unlabeled-latent baseline.
- `scripts/summarize_ablation_results.py` — builds the condition summary table and collapse taxonomy for the paper.
- `scripts/summarize_commonvoice_finetune_ablation.py` — compares the CommonVoice finetune variants against `combined` and the original `cv500` init.
- `scripts/summarize_commonvoice_objective_ablation.py` — compares the CommonVoice objective variants against `combined`, the raw `cv500` init, and the best CommonVoice finetune recipe.
- `scripts/summarize_commonvoice_rich_objectives.py` — compares the CommonVoice rich-objective variants against `combined`, the raw `cv500` init, and the best CommonVoice finetune recipe.
- `scripts/summarize_commonvoice_partial_label.py` — compares the CommonVoice partial-label variants against `combined`, the raw `cv500` init, the best CommonVoice finetune recipe, and the best CommonVoice rich-objective reference.
- `scripts/summarize_mixed_data_results.py` — summarizes the mixed-data pseudolabel mix schedule matrix against the strongest earlier CommonVoice references.
- `docs/controlvc_setup.md` — ControlVC baseline setup and smoke-test path.

## Evaluation

The evaluation scripts under `examples/` cover the three EmoVoice-style axes
plus our speaker-novelty proof:

- `examples/eval_emotion.py` — emotion2vec_plus_large Recall Rate + emo_sim (target alignment)
- `examples/eval_novelty.py` — OpenVoice native speaker-embedding novelty vs source and vs baseline conversion (speaker shift / proof of novelty)
- `examples/eval_wer.py` — OpenAI Whisper drift-from-baseline Word Error Rate (content preservation)
- `examples/eval_mos.py` — torchaudio SQUIM_SUBJECTIVE predicted MOS (naturalness)

CSV outputs from our runs live in [`results/`](results/). Schemas and reproduction steps are in [`results/README.md`](results/README.md).

For perceptual review, generate a browser-playable listening report from any
`generation_manifest.jsonl`:

```bash
python scripts/build_listening_report.py \
  --manifest output/mixed_teacher_hybrid_style_distill_labeled_warmup_eval/generation_manifest.jsonl \
  --input-tag mixed_teacher \
  --out results/listening_mixed_teacher_hybrid_style_distill_labeled_warmup.html
```

Open the generated HTML in a browser and score the companion `_ratings.csv`
template for target emotion, naturalness, intelligibility, and identity shift.

For the current paper matrix, start with:

- [`results/eval_ablation_summary_pass4.csv`](results/eval_ablation_summary_pass4.csv)
- [`results/eval_ablation_collapse_pass4.csv`](results/eval_ablation_collapse_pass4.csv)
- [`results/eval_commonvoice_finetune_summary_pass5.csv`](results/eval_commonvoice_finetune_summary_pass5.csv)
- [`results/eval_commonvoice_objective_summary_pass6.csv`](results/eval_commonvoice_objective_summary_pass6.csv)
- [`results/eval_commonvoice_rich_objectives_summary_pass7.csv`](results/eval_commonvoice_rich_objectives_summary_pass7.csv)
- [`results/eval_commonvoice_partial_label_summary_pass8.csv`](results/eval_commonvoice_partial_label_summary_pass8.csv)

## Example: NaturalSpeech 3

Install NaturalSpeech 3's FACodec with:

```
pip install git+https://github.com/lifeiteng/naturalspeech3_facodec.git
```

Then use the wrapper as follows:

```
import dpvc
vc_wrapper = dpvc.NaturalSpeech3Wrapper()
anonymizer = dpvc.Anonymizer(vc_wrapper)
anonymizer.anonymize(src_path, output_path, noise_level=1.0)
```

## Building Documentation

The documentation is built with [MkDocs](https://www.mkdocs.org/):

```bash
pip install mkdocs "mkdocstrings[python]" mkdocs-material
mkdocs build
```
