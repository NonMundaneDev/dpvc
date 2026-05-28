# Joe Perceptual Review — Audio-Calibrated Anger/Disgust Focus

Date: 2026-05-28

Panel:

- `results/listening_mixed_teacher_cvrare_audio_calibrated_anger_disgust_focus.html`

Checkpoint:

- `mixed_teacher_cvrare_audio_calibrated_labeled_warmup`

Scope:

- Focused review of `anger` and `disgust` rows from the audio-calibrated
  checkpoint.

Joe's feedback:

> I think they are all quite subtle and if you asked me what emotion is being
> conveyed I might not be able to tell for any of them. I do think the early
> ones (up to cremad_1076) sound slightly more angry for the "angry" condition,
> even though the model predicts neutral for some of those. The later ones sound
> more neutral to me. The "disgust" condition sounds neutral to me, for all of
> them.
>
> However, I don't think we should focus too much on making these sound
> discernable. I have also gone back to listen to some of the creama-d training
> examples, and many of the the "disgust" examples also sound neutral to me! I
> don't think it will be possible to magically create some signal here when the
> training data doesn't actually have major differences for some conditions.

Interpretation:

- Joe did **not** confirm `disgust` as a perceptual success. For him, `disgust`
  sounds neutral across the focused panel.
- `anger` has a weak, source-dependent perceptual signal: earlier CREMA-D rows
  up to `cremad_1076` sound slightly more angry, while later rows sound more
  neutral.
- The limiting factor may be the training data itself: if CREMA-D `disgust`
  examples are perceptually close to neutral, more objective pressure cannot
  reliably create a strong, generalizable `disgust` signal.
- Next work should not over-optimize subtle hard emotions. It should prioritize
  controls with clearer perceptual signal and document weak-label/training-data
  limitations.
