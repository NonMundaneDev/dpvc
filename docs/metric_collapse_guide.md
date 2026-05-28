# Metric and Collapse Guide

This guide defines the project metrics in plain language for collaborators and
paper drafting.

## Main Axes

### Style / Emotion Control

**Question:** does the generated voice sound like the requested style?

Primary numeric metric:

- `emotion2vec_plus_large` recall: whether the model's top predicted emotion
  label matches the target label.

Important caveat:

- Some project controls, such as `whisper`, `confused`, and `enunciated`, do
  not have direct emotion2vec labels. For those styles, listening and secondary
  embedding movement are more informative than recall.

### Content Preservation

**Question:** are the words still understandable?

Primary numeric metric:

- WER, or word error rate, computed by transcribing generated audio and
  comparing it to the source/reference text.

Collapse name:

- `content_collapse`: WER is very high or the generated audio is effectively
  unintelligible.

### Naturalness

**Question:** does the audio sound plausible rather than broken or synthetic in
a distracting way?

Primary numeric metric:

- Predicted MOS from SQUIM subjective quality scoring.

Interpretation:

- MOS is a proxy for human naturalness judgment. It is useful for relative
  comparisons but does not replace human listening.

### Identity / Speaker Novelty

**Question:** does the generated speaker move away from the source speaker or
baseline conversion?

Primary numeric metric:

- OpenVoice embedding-space novelty gain versus baseline.
- SpeechBrain ECAPA external speaker-verifier similarity and baseline-relative
  novelty gain.

Collapse name:

- `identity_collapse_to_baseline`: the generated output is still intelligible,
  but its speaker embedding does not move meaningfully beyond the baseline
  conversion.

Important clarification:

- Identity collapse is not WER. A file can be intelligible and still have low
  speaker novelty.

### Metadata Separability

**Question:** do labels such as CommonVoice `gender`, `age`, or `accent` have
recoverable structure in the embeddings or VAE latents?

Primary numeric metric:

- nearest-centroid macro-F1 versus majority and permutation baselines.

Important caveat:

- Separability is not perceptual controllability. A model can encode a gender
  correlate while a listener hears only a generic timbre or identity shift.

## Collapse Taxonomy

### `content_collapse`

The output is hard to understand, usually reflected by high WER.

### `style_collapse_to_neutral`

The target is a non-neutral style, but the emotion model predicts neutral.

### `identity_collapse_to_baseline`

The output does not move far enough from baseline identity in OpenVoice
embedding space.

### `mixed_collapse`

More than one collapse axis occurs in the same generated file, for example high
WER and low speaker novelty.

## How To Explain This In Meetings

Use this wording:

> Collapse does not always mean garbage audio. Sometimes it means the audio is
> understandable but falls back toward a neutral style or baseline-like speaker
> identity. We track those as separate axes: content, style, and identity.

## Current Interpretation

The current evidence says:

- The expanded rare-supply mixed teacher improves emotion recall and speaker
  novelty.
- The `sad/enunciated` guard is the current quality-balanced listening profile.
- The style-strength grid found metric gains for some hard styles, but Joe's
  first perceptual review found no candidate wins. The content-repair gate now
  formalizes this and promotes `0/33` hard-style strength rows, so those
  strength settings remain diagnostic rather than presets.
- The first CommonVoice age/gender metadata-control checkpoint trained and
  generated audio, but local perceptual review heard identical outputs or
  generic timbre/identity shifts rather than interpretable age/gender control.
- The metadata separability probe found gender strongly recoverable in raw
  embeddings and metadata-control VAE latents, but age is weak and accent does
  not survive strongly in VAE latents.
- The next metadata-control research task should be narrow and listening-first:
  a balanced gender-focused follow-up only if the paper needs metadata controls,
  not a broad age/gender/accent sweep.

## Paper-Readiness Rule

Use metrics to find candidates, but use listening to decide whether a candidate
can become a demo or paper claim.

Current status:

- `cvrare_sad_enunc_guard`: paper/demo reference candidate.
- external ECAPA verifier: corroborates the current guard's identity shift
  (`0.3594` mean styled novelty gain vs baseline; `6/99` styled rows accepted
  as source at a proxy threshold).
- `anger_s10` / `fear_s7p5`: diagnostic candidates only; Joe heard no
  perceptual win in the five-row review and the generated-audio content-repair
  gate promotes no current hard-style strength preset.
- generated-audio objective plan: selects `anger` and `disgust` for training
  repair, blocks `fear`, and still requires generated-audio evaluation plus
  listening before it can become a paper result.
- CommonVoice age/gender controls: implemented diagnostic infrastructure only;
  the first perceptual gate failed; separability probing says gender has
  objective structure, while age/accent remain weak or diagnostic.
