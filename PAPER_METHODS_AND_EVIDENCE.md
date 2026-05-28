# Paper Methods and Evidence Packet

**Date:** 2026-05-28
**Branch:** `docs/paper-methods-and-evidence`
**Purpose:** turn the current research state into a paper-facing methods and
evidence outline before starting another training branch.

## Top-Line Position

The strongest current paper story is:

> We built a controllable speaker-generation / anonymization pipeline on top of
> OpenVoice speaker embeddings. The system can move speaker/style attributes in
> controlled directions, and the strongest mixed-data result now reaches about
> `47.0%` emotion recall with strong speaker novelty. The current best
> quality-balanced demo is the expanded rare-supply checkpoint with the
> `cvrare_sad_enunc_guard` inference profile. Recent stronger style-strength
> candidates and first-pass CommonVoice age/gender controls are diagnostic, not
> promoted results.

This should be the paper-writing posture:

- lead with controllable speaker generation / anonymization, not perfect
  emotion conversion;
- present emotion/style as the best validated controllable attribute family;
- present CommonVoice age/gender controls as implemented infrastructure and
  future work, because the first listening panel did not show perceptible
  age/gender control;
- separate objective metrics from perceptual listening, and do not promote a
  setting unless both support the claim.

## What To Listen To First

Serve the repository root:

```bash
cd /Users/steve/UVM-plaid/dp-vc
python3 -m http.server 8000
```

Open:

```text
http://localhost:8000/results/listening_evidence_demo_index.html
```

The first detailed report to inspect is:

```text
http://localhost:8000/results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.html
```

## Proposed Paper Method

### Problem Framing

The method targets voice-to-voice controllable speaker generation. A source
utterance keeps its content, but the speaker representation is transformed
through a controllable latent space. Differentially private anonymization is
one downstream use: generate a different speaker identity while retaining
content and controlling desired speaker/style attributes.

### Active Voice Conversion Path

OpenVoice is the active controllable path:

- extract source speaker embeddings;
- train a controllable VAE over speaker embeddings;
- reserve latent dimensions for controllable style attributes;
- decode a modified latent vector back to an OpenVoice speaker embedding;
- synthesize voice-to-voice outputs through the OpenVoice path.

ControlVC remains useful as a DP baseline and wrapper reference, but the repo
findings show it is not the active style-control path.

### Controllable Latent Layout

Current main model family:

- latent dimensionality: `15`
- style dimensions: `0-8`
- style labels:
  - `anger`
  - `confused`
  - `disgust`
  - `enunciated`
  - `fear`
  - `happy`
  - `neutral`
  - `sad`
  - `whisper`
- free speaker dimensions: remaining dimensions in the style-only model family

The CommonVoice metadata-control branch added first-pass scalar metadata
controls:

- `dim_9`: gender scalar
- `dim_10`: age scalar

Those controls train and generate, but the first perceptual panel sounded
identical or like generic speaker/timbre shifts. They are not paper-facing
positive results yet.

### Training Data

Current evidence uses three main data sources:

- CREMA-D: labeled emotion speech, useful for canonical emotion supervision.
- Expresso: labeled expressive speech, useful for additional speaking-style
  supervision.
- CommonVoice English: broad speaker coverage, used for speaker diversity and
  pseudo-labeled / metadata-bearing rows.

The current stable local CommonVoice path is:

```text
/Users/steve/datasets/cv-corpus-21.0-2025-03-14/en
```

For the expanded rare-supply run:

- local clips: `40000`
- usable rows: `40000`
- local speakers: `20537`
- extracted OpenVoice embeddings: `25910`
- extracted speakers: `13308`

The mixed artifact preserved expanded rare pseudo-label supply, including
`anger=50` and `fear=50` selected CommonVoice pseudo rows.

### Supervision Strategy

The strongest current training path is not naive CommonVoice pretraining. The
successful sequence was:

1. protect true labeled CREMA-D / Expresso rows;
2. use CommonVoice for broader speaker coverage;
3. pseudo-label selected CommonVoice rows with teacher models;
4. preserve rare-class supply for hard styles;
5. use style-space teacher distillation with a labeled-first warmup;
6. apply inference-side per-style strength calibration for quality control.

The quality-balanced reference condition is:

```text
mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard
```

The underlying checkpoint is:

```text
embeddings/openvoice_vae_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.pt
```

The strength profile is:

```text
configs/style_strength_profiles/cvrare_sad_enunc_guard.json
```

### Evaluation Stack

The current evaluation stack is intentionally multi-axis:

- style / emotion control: `emotion2vec_plus_large` recall and `emo_sim`
- content preservation: Whisper WER
- naturalness: SQUIM subjective MOS proxy
- speaker novelty: OpenVoice embedding-space novelty gain versus baseline
- perceptual validation: browser listening panels and collaborator review

The key rule for paper writing:

> Objective metrics can identify candidates, but a setting should not be
> promoted as a preset or demo claim unless perceptual listening also supports
> it.

## Claim-To-Evidence Map

| Claim | Evidence | Artifact | Caveat |
|-------|----------|----------|--------|
| OpenVoice is the active controllable path. | Finding 1 shows ControlVC is a useful DP baseline but not the style-control path. | `FINDINGS.md` Finding 1 | This does not make ControlVC irrelevant; it remains useful for baseline/wrapper comparisons. |
| Speaker diversity matters for style control. | Findings 2, 4, and 6 show OpenVoice style directions become perceptible and generalize better with more speakers. | `FINDINGS.md` Findings 2, 4, 6 | Some styles and speaker-style pairs still collapse. |
| Naive CommonVoice pretraining is not enough. | Findings 10-16 show CommonVoice improves WER/content but tends to wash out style or identity without better supervision. | `FINDINGS.md` Findings 10-16 | Negative results narrow the method; they are not the final CommonVoice result. |
| Mixed-data training needed rare-class CommonVoice supply. | Finding 30 shows expanded rare supply produced the first large recall jump. | `FINDINGS.md` Finding 30 | The unguarded run has WER/MOS quality cost. |
| The best current quality-balanced result preserves the recall gain while reducing quality damage. | Finding 31 shows `cvrare_sad_enunc_guard` keeps `47.0%` recall, improves WER to `0.2348`, improves MOS delta to `-0.2081`, and reduces any-collapse files to `20`. | `FINDINGS.md` Finding 31; `results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.html` | It is an inference-side calibration profile, not a final learned repair objective. |
| Stronger metric-selected style settings should not be promoted yet. | Finding 35 plus Joe's five-row review showed `0/5` candidate wins. | `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority_joe_2026-05-19.md` | The grid remains useful for diagnostics and candidate mining. |
| CommonVoice age/gender controls are implemented but not perceptually validated. | The metadata-control branch trained and generated a panel, but local listening found identical/generic timbre shifts. | `results/listening_metadata_w010_labeled_warmup.md`; `IMPLEMENTATION_PLAN_commonvoice-metadata-controls.md` | Treat as future work, not a current paper result. |

## Current Positive Results

Use these as the core paper/evidence backbone:

- OpenVoice speaker embeddings can support controllable style directions.
- Style control is perceptible for several styles, with `whisper` especially
  strong.
- Expanded rare-class CommonVoice supply changed the mixed-data outcome from
  low-recall diagnostic runs to a serious `47.0%` recall result.
- The `cvrare_sad_enunc_guard` profile is the current best recall/quality
  compromise.
- The project now has a reproducible evaluation stack: emotion recall, WER,
  MOS proxy, novelty, collapse taxonomy, and listening dashboards.

## Current Non-Claims

Do not claim:

- age/gender controls work perceptually;
- `anger_s10` or `fear_s7p5` are better presets;
- objective classifier wins are sufficient without listening;
- formal privacy accounting is already complete;
- the current method is the final best architecture.

## Anticipated Questions And Short Answers

### What is the main result in one sentence?

The current system can perform controllable voice-to-voice speaker generation
with several audible style controls, and the best mixed-data setup reaches
about `47.0%` emotion recall while preserving useful speaker novelty.

### What is the strongest number and its caveat?

The strongest current number is `47.0%` emotion recall from the expanded
rare-supply mixed teacher. The caveat is that the unguarded version damages
WER/MOS, so the current demo reference uses the `cvrare_sad_enunc_guard`
profile to reduce that quality cost.

### Is this supervised, unsupervised, or weakly supervised?

It is mixed supervision. CREMA-D and Expresso provide true labels. CommonVoice
provides broad speaker coverage plus weak pseudo labels from teacher models and
metadata labels where available.

### What is pseudo-labeling?

Pseudo-labeling means using a pretrained model to assign provisional labels to
otherwise unlabeled CommonVoice clips. We then filter, cap, and record those
selected rows before mixing them with true labeled rows.

### What does filtering mean here?

Filtering is confidence-gated weak-label selection. It is not hand-picking
nice examples. The pipeline records accepted/rejected counts, class coverage,
and artifact metadata so the selection process is auditable.

### What does collapse mean?

Collapse is split into axes:

- content collapse: high WER / unintelligible audio;
- style collapse to neutral: a non-neutral target is classified as neutral;
- identity collapse to baseline: low OpenVoice novelty gain versus baseline;
- mixed collapse: multiple axes fail together.

### Where does differential privacy enter?

The DP framing comes from perturbing speaker representations / latent speaker
codes to generate anonymized speaker identities. Formal epsilon accounting and
privacy-utility curves remain open paper tasks.

### What should Joe listen to?

Start with:

```text
results/listening_evidence_demo_index.html
```

Then inspect:

```text
results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.html
```

### What happened with age/gender controls?

The repo now has first-pass CommonVoice age/gender control plumbing, training,
inference flags, and a listening panel. The first local perceptual review did
not hear interpretable age/gender changes, so this remains diagnostic
infrastructure and future work.

### What should happen next?

The immediate next move is paper-method consolidation, not another blind
training run. The next research moves should be targeted:

- probe whether OpenVoice embeddings encode recoverable age/gender;
- add an external speaker-verifier / EER-style novelty check;
- design a generated-audio-calibrated content repair loop for hard styles;
- add repeated-seed confidence intervals before freezing paper tables.

## Next Research Queue

1. Paper-method documentation and evidence cleanup.
2. External speaker-verifier novelty check.
3. Metadata separability probe before more age/gender training.
4. Generated-audio/content-repair loop for `anger`, `disgust`, and `fear`.
5. Repeated-seed confidence intervals for final candidate tables.
6. Formal DP accounting and privacy-utility curves.

## Validation

- This packet does not add a new model result.
- It reuses verified findings and listening artifacts already recorded in
  `FINDINGS.md`, `WORKLOG.md`, and `results/`.
- `FINDINGS.md` should not be changed by this documentation branch unless a
  new verified experiment is added later.
