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

Collapse name:

- `identity_collapse_to_baseline`: the generated output is still intelligible,
  but its speaker embedding does not move meaningfully beyond the baseline
  conversion.

Important clarification:

- Identity collapse is not WER. A file can be intelligible and still have low
  speaker novelty.

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
  first perceptual review found no candidate wins, so those strength settings
  remain diagnostic rather than presets.
- The first CommonVoice age/gender metadata-control checkpoint trained and
  generated audio, but local perceptual review heard identical outputs or
  generic timbre/identity shifts rather than interpretable age/gender control.
- The next high-value documentation task is to consolidate the current
  style-control method and evidence. The next metadata-control research task
  should first test whether OpenVoice embeddings contain recoverable age/gender
  signal before spending more training compute.

## Paper-Readiness Rule

Use metrics to find candidates, but use listening to decide whether a candidate
can become a demo or paper claim.

Current status:

- `cvrare_sad_enunc_guard`: paper/demo reference candidate.
- `anger_s10` / `fear_s7p5`: diagnostic candidates only; Joe heard no
  perceptual win in the five-row review.
- CommonVoice age/gender controls: implemented diagnostic infrastructure only;
  the first perceptual gate failed.
