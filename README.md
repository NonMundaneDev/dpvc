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

- **mixed-data style-supervision diagnostics and curriculum**

Immediate next queue:

1. test a labeled-first curriculum or decoder-aware style objective, because the per-style diagnostic shows canonical teacher geometry and decoded emotion are misaligned
2. rebuild or rebalance rare canonical CommonVoice pseudo-label supply before more weighting experiments, especially `anger` and `fear`
3. keep `mixed_teacher_threshold_balanced` as the best overall mixed-data teacher reference, while treating `mixed_teacher_hybrid_style_distill_balanced` as the strongest novelty/naturalness tradeoff result from the hybrid teacher line
4. add the Joe-facing metric guide, broaden the non-Trump sweep, and finish the reproducibility checklist / dependency pinning work

The dedicated next-step plans live in:

- **[`IMPLEMENTATION_PLAN_post-consolidation-next-queue.md`](IMPLEMENTATION_PLAN_post-consolidation-next-queue.md)** — historical record of the consolidation / rollback sequence
- **[`IMPLEMENTATION_PLAN_mixed-data-pseudolabel-teacher.md`](IMPLEMENTATION_PLAN_mixed-data-pseudolabel-teacher.md)** — the technical plan for the current experiment slice on `research/controllable-vae`

We’ve extended the library with a **controllable** VAE that exposes 9 style knobs (anger, confused, disgust, enunciated, fear, happy, neutral, sad, whisper) on top of the DP anonymization pipeline. Primary entry points:

- **[`examples/README.md`](examples/README.md)** — end-to-end reproduction guide (extraction → training → controllable inference → evaluation).
- **[`FINDINGS.md`](FINDINGS.md)** — 28 paper-facing findings with methodology and per-row takeaways.
- **[`WORKLOG.md`](WORKLOG.md)** — roadmap and progress tracking.
- **[`results/`](results/)** — raw evaluation CSVs (emotion2vec Recall/emo_sim, WER, predicted MOS) backing the findings.

OpenVoice is the **canonical controllable pipeline**. ControlVC remains in the
repository as a useful DP baseline and wrapper reference, but not as the
recommended path for style control.

Current best checked-in result: the **combined** OpenVoice model remains the
best tradeoff across controllability, speaker novelty, intelligibility, and
naturalness. The later CommonVoice finetune, objective, rich-objective, and
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
back WER and MOS versus `mixed_teacher_threshold_balanced`. That makes
`mixed_teacher_threshold_balanced` the best overall mixed-data teacher
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
That kept `mixed_teacher_threshold_balanced` as the best overall mixed-data
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
That makes the next training move a labeled-first curriculum or decoder-aware
style objective, not another latent-only calibration variant. The
non-Trump strength sweep adds a narrower inference-side result: `5.0` remains
the safest default, `7.5` is a useful stronger option for styles like
`whisper` and `confused`, and `10.0-12.5` look more like high-novelty
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
- `scripts/build_mixed_training_set.py` + `examples/openvoice_train_vae_mixed.py` — mixed-data bootstrap path that combines pseudo-labeled CommonVoice with labeled CREMA-D and Expresso under schedule-controlled sampling, including optional style-space teacher distillation via `--style-teacher-checkpoint`, `--style-teacher-weight`, `--style-teacher-dims`, `--style-teacher-datasets`, `--style-teacher-target-mode`, `--style-teacher-require-label`, `--style-teacher-style-weights`, and `--style-teacher-confidence-power`.
- `scripts/prepare_commonvoice_subset.py` — helper for turning downloaded Common Voice shards into a filtered local `validated.tsv` + `clips/` subset.
- `scripts/annotate_commonvoice_pseudolabels.py` — adds confidence-scored pseudo-style labels to a Common Voice embedding artifact.
- `scripts/annotate_commonvoice_latent_prototypes.py` — scores Common Voice rows against combined-VAE latent style prototypes as an alternate pseudo-label teacher.
- `scripts/combine_commonvoice_pseudolabel_teachers.py` — combines filtered emotion2vec and latent-prototype CommonVoice pseudo labels into a reusable hybrid teacher artifact.
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
