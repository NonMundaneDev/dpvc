# Pre-Meeting Async Feedback: Joe Near

**Date:** 2026-05-28
**Context:** Joe reviewed the focused audio-calibrated `anger` / `disgust`
listening bundle over Microsoft Teams before the May 28 update meeting.

## 1. Executive Summary For The Upcoming Meeting

Joe's feedback changes the next research direction. He did **not** hear
`disgust` as a perceptually successful style. He heard the whole focused panel
as subtle, with only weak source-dependent `anger` signal in the early CREMA-D
rows and mostly neutral behavior elsewhere.

The most important point is his interpretation of the training data:

> many CREMA-D `disgust` training examples also sound neutral, so we should not
> expect the model to create a strong perceptual signal for a label that is weak
> in the source data.

That means the next best work is not another attempt to force `disgust` to be
discernible. The better direction is to distinguish:

- controls with clear perceptual signal that can support the paper/demo;
- labels that are weak or subtle in the training data and should be treated as
  limitations or excluded from headline claims.

## 2. What Joe Said In Teams

Joe's core points:

- The focused `anger` / `disgust` outputs are subtle overall.
- If asked to identify the emotion without labels, he might not be able to tell
  for any of them.
- Early rows up to `cremad_1076` sound slightly more angry in the `anger`
  condition, even when the model predicts neutral.
- Later `anger` rows sound more neutral.
- `disgust` sounds neutral to him for all rows.
- Some CREMA-D training examples labeled `disgust` also sound neutral.
- We should not spend too much effort making weak/subtle labels strongly
  discernible when the training data itself may not contain a major difference.

## 3. Interpretation To Bring Into The Meeting

Before Joe's review, Stephen's first listening pass suggested:

- `disgust` might be a perceptual success that emotion2vec missed;
- `anger` might be the main content-repair target.

After Joe's review, the stronger interpretation is:

- `disgust` is **not confirmed** as a perceptual success.
- The emotion2vec `0/11` `disgust` result is probably not just a metric miss;
  it may reflect weak perceptual signal in the training data.
- `anger` has some source-dependent perceptual signal, but it is not robust
  enough to become the next headline control without careful selection.
- The research should stop chasing hard-style discernibility for labels that
  are weak in the training corpus.

## 4. What This Means For The Paper Story

The paper story should shift away from:

> "Can we make every CREMA-D/Expresso emotion label strongly audible?"

Toward:

> "The framework can control speaker/style attributes when the underlying
> training signal is perceptually meaningful; weak or ambiguous labels expose a
> limitation of dataset-driven control rather than a failure of the architecture
> alone."

This is a more defensible research claim. It also aligns with Joe's earlier
point that the contribution is broader controllable speaker generation, not
state-of-the-art emotion conversion.

## 5. Recommended Discussion Agenda

Recommended order to discuss with Joe:

1. Add a small training-data perceptual audit for the labels we keep trying to
   optimize, especially `disgust`, `anger`, and possibly `fear`.
2. Re-rank controllable styles by human-perceptual strength, not just by
   emotion2vec recall.
3. Focus demos and paper examples on styles/attributes with clear signal.
4. Treat subtle labels as limitations or future work unless additional data
   with stronger perceptual examples is added.
5. Confirm whether we should continue paper-methods documentation around the
   current reference result instead of running another latent/prototype tweak
   for `disgust`.

## 6. Communication Notes For Stephen

What you did right:

- You asked Joe for perceptual validation before promoting the checkpoint.
- You framed the question narrowly enough that his feedback was actionable.
- You avoided making the `disgust` claim before confirmation.

What to improve next time:

- When a style sounds good to you but the metric disagrees, frame it as a
  hypothesis until another listener confirms it.
- Ask Joe to compare the generated samples against actual training examples for
  the same label. His `disgust` observation came from doing exactly that, and it
  exposed the real bottleneck.
- Keep emphasizing training-signal quality: the model cannot reliably amplify a
  perceptual distinction that the data labels do not clearly contain.

## 7. One-Sentence Status For The Meeting

The audio-calibrated checkpoint is diagnostic: it does not create a robust
perceptual `disgust` control, `anger` is only weakly/source-dependently audible,
and Joe's review suggests the next work should audit and prioritize training
labels by perceptual strength rather than forcing subtle CREMA-D emotions to
become obvious.
