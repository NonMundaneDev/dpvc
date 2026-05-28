# Meeting Brief for Joe Near

**Project:** DPVC / controllable voice-to-voice speaker generation
**Meeting date:** May 28, 2026
**Prepared for:** Stephen Oladele
**Current branch:** `research/generated-audio-calibrated-training`
**Research fork:** `https://github.com/NonMundaneDev/dpvc`
**Updated after:** Joe's pre-meeting Teams listening feedback

## 1. The 90-Second Opening

Use this first, before details:

> Since our last meeting, I followed up on the perceptual-review and broader-control directions. The strongest checkpoint is still the expanded CommonVoice rare-supply mixed teacher with the `sad/enunciated` guard: it keeps about `47%` emotion recall while improving quality compared with the unguarded version. I also tested age/gender metadata controls, external speaker verification, generated-audio content repair, and a training-side audio-calibrated objective. The newest audio-calibrated checkpoint is not better on aggregate metrics. My first listen suggested `disgust` might be perceptually stronger than emotion2vec reported, but your pre-meeting review did not confirm that: you heard `disgust` as neutral and `anger` as only subtly/source-dependently angry. That is useful because it points away from trying to force weak labels and toward auditing which labels actually have clear perceptual training signal.

## 2. Main Meeting Goal

The meeting should answer four questions:

1. Does Joe agree that the right interpretation is weak/ambiguous training signal for `disgust`, not simply a model repair problem?
2. Should we deprioritize `disgust` as a headline control unless stronger examples/data are added?
3. Which styles or controls have enough perceptual signal to support the paper/demo now?
4. Should the next two-week direction be:
   - a training-data perceptual audit and style-priority pass,
   - paper-methods documentation around the current strongest result,
   - a focused repair for a higher-signal style such as `anger`,
   - or another experimental direction Joe thinks is higher value?

## 3. What Changed Since the May 14 Meeting

### A. Joe's first five-row review was incorporated

Joe reviewed the five priority A/B rows from the previous strength-grid bundle.

His result:

- `4/5` rows sounded identical to the reference.
- `1/5` preferred the reference because the candidate had an unnatural pitch change.
- No candidate clearly beat the current guard.

Interpretation:

- The strength grid is diagnostic, not a preset.
- We should not promote `anger_s10` or `fear_s7p5` just because metrics liked them.
- Human listening remains necessary before making style-control claims.

### B. Age/gender metadata controls were tested, but not validated perceptually

We implemented the first CommonVoice metadata-control path and listened locally.

Result:

- The first age/gender control panel sounded mostly identical or like generic speaker/timbre shifts.
- A metadata separability probe found that **gender is objectively recoverable** in embeddings/latents.
- Age and accent are weaker and should not be pushed as current claims.

Interpretation:

- Metadata controls are infrastructure and diagnostic evidence, not a current paper result.
- We should not claim age/gender perceptual control yet.

### C. External speaker verifier corroborates identity shift

We added a SpeechBrain ECAPA external speaker-verifier check.

For the current `sad/enunciated` guard:

- ECAPA mean styled novelty gain vs baseline: about `0.3594`.
- Only `6/99` styled rows were accepted as the source speaker at the proxy threshold.

Interpretation:

- Speaker movement is real across an independent verifier, not only OpenVoice's native embedding space.
- Final privacy/EER claims still need independent labeled speaker-verification trials.

### D. Generated-audio content-repair gate promoted no current hard-style strength candidate

We turned Joe's perceptual review into a reproducible gate.

Result:

- No `anger`, `disgust`, or `fear` strength-grid candidate was promoted.
- The only objective-pass rows were blocked by perceptual feedback.
- `disgust` had no objective-pass repair row in the grid.

Interpretation:

- The next repair should not be "increase style strength."
- It should use generated-audio behavior more directly.

### E. First audio-calibrated training checkpoint was trained and evaluated

Checkpoint:

- `mixed_teacher_cvrare_audio_calibrated_labeled_warmup`

Top-line comparison:

| Condition | Recall | Mean WER | MOS delta | Novelty gain | Style-to-neutral | Any collapse |
|-----------|--------|----------|-----------|--------------|------------------|--------------|
| current `sad/enunciated` guard | `46.97%` | `0.2348` | `-0.2081` | `0.2726` | `18` | `20` |
| audio-calibrated training | `40.91%` | `0.2465` | `-0.1236` | `0.2351` | `29` | `37` |

Targeted styles:

| Style | emotion2vec recall | Stephen's first listening read | Joe's pre-meeting read |
|-------|--------------------|--------------------------------|------------------------|
| `disgust` | `0/11` | sounded convincingly disgusted and intelligible | neutral for all rows |
| `anger` | `3/11` | some style change, but distorted / less intelligible | slightly angry only in early rows; later rows neutral |

Interpretation:

- Aggregate metrics say this checkpoint should not replace the current guard.
- Joe's review says `disgust` is not currently a confirmed perceptual success.
- `anger` has weak/source-dependent signal, but it is not robust enough yet to become a headline claim without careful selection.
- The more important bottleneck may be label/training-signal quality, especially if many CREMA-D `disgust` examples themselves sound neutral.

## 4. The Current Best Story

The current research story is stronger than it was two weeks ago, but more nuanced:

1. **The best aggregate result is still the expanded CommonVoice rare-supply mixed teacher with the `sad/enunciated` guard.**
2. **External speaker verification supports the identity-shift claim.**
3. **Age/gender controls are not yet perceptually validated.**
4. **The newest audio-calibrated checkpoint is diagnostic, not a new reference.**
5. **Joe's pre-meeting review shifts the next bottleneck toward training-signal quality: some labels may be too perceptually subtle to support strong controls from the current data.**

## 5. What Not To Overclaim

Avoid saying:

- "The audio-calibrated checkpoint is the new best model."
- "Disgust is solved."
- "emotion2vec is wrong."
- "Joe confirmed the disgust result."
- "Age/gender controls work."
- "We have final privacy guarantees."
- "Speaker novelty equals formal privacy."

Safer wording:

- "The current aggregate reference is still the `sad/enunciated` guard."
- "Joe's pre-meeting review did not confirm `disgust`; this now looks like weak/ambiguous label signal rather than a simple metric miss."
- "Age/gender controls have signal in the latent space, but first-pass perceptual control is not established."
- "External ECAPA corroborates identity shift, but final privacy/EER needs independent trials and DP accounting."

## 6. What To Ask Joe

Ask these directly:

1. Is my corrected interpretation right: `disgust` should be treated as weak/ambiguous training signal, not a current model success?
2. Should we stop optimizing `disgust` as a hard repair target unless we add stronger perceptual examples?
3. Which labels or controls do you think are worth keeping in the headline demo/paper story?
4. Would a small perceptual audit of CREMA-D/Expresso labels be useful before another model run?
5. Should the current strongest result be enough to begin paper method/evaluation writing while we keep limitations explicit?
6. Should age/gender controls remain active, or should they become future work until emotion/style is cleaner?

## 7. What Was Sent To Joe

Already sent the focused review bundle:

- `results/joe_audio_calibrated_review_bundle_2026-05-28.zip`

It is self-contained and does not require the full OpenVoice/CommonVoice setup.
Joe already listened and replied over Teams, so do not resend it unless he asks
for the link/path again.

If Joe asks whether this is what we needed, use this wording:

```text
Yes, this was exactly useful. It changes my interpretation in a more
conservative direction: I should not claim `disgust` works perceptually if you
hear it as neutral, and your point about CREMA-D `disgust` examples sounding
neutral suggests the label itself may be weak. I would like to use today to
decide whether the next step should be a small training-data perceptual audit
and style-priority pass before another model run.
```

## 8. What You Should Study Before the Meeting

### Collapse taxonomy

- `content_collapse`: high WER / content changed or became unintelligible.
- `style_collapse_to_neutral`: target style is non-neutral, but emotion2vec predicts neutral.
- `identity_collapse_to_baseline`: generated voice did not move far enough away from the same-source baseline in speaker embedding space.
- `mixed_collapse`: two or more collapse axes fire for the same file.

### Metric meanings

- Emotion recall: does emotion2vec's top label match the requested emotion?
- emo_sim: emotion embedding similarity to baseline; useful when labels miss or styles have no exact class.
- WER: content preservation / intelligibility proxy.
- MOS delta: predicted naturalness vs same-speaker baseline.
- Novelty gain: how far the generated speaker moves away from the source/baseline.
- ECAPA accept-as-source: independent verifier check for whether generated audio still sounds like the source speaker to a verifier.

### Pseudo-labeling

Use this wording:

> CommonVoice does not have emotion labels, so we use a pretrained emotion model to assign weak emotion labels. Filtering means keeping only rows that pass confidence, class-balance, or target-count rules, and recording what was accepted/rejected so the process is auditable.

### Supervision type

Say:

> The emotion part is supervised on CREMA-D/Expresso, weakly supervised on CommonVoice through pseudo labels, and evaluated with both objective metrics and human listening. It is not purely unsupervised.

## 9. Anticipated Questions And Answers

### Q: What is the main result since last time?

A: The new audio-calibrated checkpoint is not a new aggregate best model. It revealed a useful negative/diagnostic result: Joe did not hear a robust `disgust` control, and he noted that many CREMA-D `disgust` examples also sound neutral. That moves the next question toward training-label perceptual quality.

### Q: Which model is the current reference?

A: The current reference is still `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard`: `46.97%` recall, `0.2348` mean WER, `0.2726` novelty gain, and `20` files with any collapse.

### Q: Why use human listening in addition to emotion2vec?

A: We should not blindly replace the metric. Human listening is a guard against both false positives and false negatives. In this case, Joe's listening made the result more conservative: it prevented us from overclaiming `disgust` as a metric miss when it may actually be weak/neutral training signal.

### Q: Did age/gender controls work?

A: Not perceptually yet. Gender is separable in embeddings/latents, but the first listening panel did not produce clear age/gender control. It should remain diagnostic/future work unless Joe wants it prioritized.

### Q: Is privacy solved?

A: No. We have stronger evidence of identity shift, including ECAPA external verifier results, but formal DP accounting and independent EER trials remain open paper tasks.

### Q: What should happen next?

A: Confirm the interpretation with Joe during the meeting, then run a small training-data perceptual audit / style-priority pass before another model run. The likely next paper move is to foreground controls with clear perceptual signal and treat weak labels like `disgust` as limitations unless stronger data is added.

## 10. Proposed Next Two-Week Plan

Recommended order:

1. Confirm with Joe that his pre-meeting review should be interpreted as a training-signal warning, especially for `disgust`.
2. After the meeting, record the actual meeting outcome separately from the pre-meeting Teams feedback.
3. Add a training-data perceptual audit / style-priority branch for CREMA-D/Expresso labels.
4. Re-rank current controls by human-perceptual strength and decide which belong in the paper/demo.
5. Add eval-suite preflight for `ffmpeg` / `torchcodec`.
6. Start paper-methods documentation around the current reference result, metrics, and limitations.

## 11. Meeting Outcome Template

After the meeting, paste notes back into Codex using this structure:

```text
Joe's read on disgust after discussion:

Joe's read on anger after discussion:

Does Joe accept the weak/ambiguous training-signal framing?

Which labels or controls should remain in the paper/demo story?

Should age/gender stay active or move to future work?

Should we start paper writing/method docs now?

Agreed next branch/task:

Any exact phrases Joe used that should go into FINDINGS or WORKLOG:
```
