# Meeting Brief for Joe Near

**Project:** DPVC / controllable voice-to-voice speaker generation
**Meeting date:** May 28, 2026
**Prepared for:** Stephen Oladele
**Current branch:** `research/generated-audio-calibrated-training`
**Research fork:** `https://github.com/NonMundaneDev/dpvc`

## 1. The 90-Second Opening

Use this first, before details:

> Since our last meeting, I followed up on the perceptual-review and broader-control directions. The strongest checkpoint is still the expanded CommonVoice rare-supply mixed teacher with the `sad/enunciated` guard: it keeps about `47%` emotion recall while improving quality compared with the unguarded version. I also tested age/gender metadata controls, external speaker verification, generated-audio content repair, and a training-side audio-calibrated objective. The newest audio-calibrated checkpoint is not better on aggregate metrics, but my first listening pass found something important: `disgust` sounds perceptually good and intelligible even though emotion2vec scores it as `0/11`; `anger` has audible style pressure but damages intelligibility. So today I want your help deciding whether `disgust` is a real perceptual success that the metric misses, and whether the next repair should focus on anger content preservation rather than more generic style pressure.

## 2. Main Meeting Goal

The meeting should answer three questions:

1. Does Joe agree that the audio-calibrated `disgust` outputs sound perceptually disgusted and intelligible?
2. Does Joe agree that `anger` has style pressure but needs content/intelligibility repair?
3. Should the next two-week direction be:
   - focused anger content repair,
   - metric/perceptual calibration for `disgust`,
   - paper-methods documentation around the current strongest result,
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

| Style | emotion2vec recall | Stephen's first listening read |
|-------|--------------------|--------------------------------|
| `disgust` | `0/11` | sounds convincingly disgusted and intelligible |
| `anger` | `3/11` | some style change, but distorted / less intelligible |

Interpretation:

- Aggregate metrics say this checkpoint should not replace the current guard.
- Human listening suggests `disgust` may be a metric-calibration miss, not a true style-control failure.
- `anger` is likely the real next content-repair target.

## 4. The Current Best Story

The current research story is stronger than it was two weeks ago, but more nuanced:

1. **The best aggregate result is still the expanded CommonVoice rare-supply mixed teacher with the `sad/enunciated` guard.**
2. **External speaker verification supports the identity-shift claim.**
3. **Age/gender controls are not yet perceptually validated.**
4. **The newest audio-calibrated checkpoint creates an important metric-vs-human disagreement for `disgust`.**
5. **The next bottleneck may be less "make the emotion metric higher" and more "separate perceptual style success from classifier mismatch, while repairing content damage for anger."**

## 5. What Not To Overclaim

Avoid saying:

- "The audio-calibrated checkpoint is the new best model."
- "Disgust is solved."
- "emotion2vec is wrong."
- "Age/gender controls work."
- "We have final privacy guarantees."
- "Speaker novelty equals formal privacy."

Safer wording:

- "The current aggregate reference is still the `sad/enunciated` guard."
- "My first listening pass suggests `disgust` may be perceptually successful despite emotion2vec missing it; I need Joe's confirmation."
- "Age/gender controls have signal in the latent space, but first-pass perceptual control is not established."
- "External ECAPA corroborates identity shift, but final privacy/EER needs independent trials and DP accounting."

## 6. What To Ask Joe

Ask these directly:

1. When you listen to the focused `disgust` rows, do they sound genuinely disgusted, or am I over-reading generic timbre change?
2. For `anger`, do you hear useful style control, or is the content distortion too severe to count as useful?
3. If human listening says `disgust` works but emotion2vec says `0/11`, how should we report that in the paper?
4. Should we add a small human perceptual study as a primary/secondary evaluation for hard styles?
5. Is the current strongest result close enough to start paper method/evaluation writing, while keeping anger repair as follow-up?
6. Should age/gender controls remain in scope now, or should they become future work until emotion/style is cleaner?

## 7. What To Send Joe

Send the focused review bundle:

- `results/joe_audio_calibrated_review_bundle_2026-05-28.zip`

It is self-contained and does not require the full OpenVoice/CommonVoice setup.

Copy-paste message:

```text
Hi Joe,

For today's follow-up, I made a smaller focused listening bundle for the newest audio-calibrated checkpoint. The question is narrower than last time:

1. Does the `disgust` style actually sound disgusted and intelligible to you, even though emotion2vec reports 0/11 recall?
2. Does `anger` sound emotionally stronger but too distorted / less intelligible?

To run:

cd joe_audio_calibrated_review_bundle_2026-05-28
python3 -m http.server 8000

Then open:

http://localhost:8000/results/listening_mixed_teacher_cvrare_audio_calibrated_anger_disgust_focus.html

Please listen source -> baseline -> disgust/anger for each speaker. If sending the CSV back is inconvenient, short text notes are fine:

Disgust: convincing / mixed / not convincing; intelligibility: good / mixed / poor.
Anger: convincing / mixed / not convincing; intelligibility: good / mixed / poor.
Recommendation: treat disgust as a perceptual success? focus next repair on anger content?

The key thing I want to avoid is over-trusting emotion2vec if human listeners hear a valid disgust style, but also avoid promoting anything that damages content.
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

A: The new audio-calibrated checkpoint is not a new aggregate best model, but it revealed an important perceptual/metric split: `disgust` sounds promising to me despite emotion2vec scoring it as `0/11`, while `anger` needs content repair.

### Q: Which model is the current reference?

A: The current reference is still `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard`: `46.97%` recall, `0.2348` mean WER, `0.2726` novelty gain, and `20` files with any collapse.

### Q: Why trust human listening over emotion2vec?

A: We should not blindly replace the metric. The right interpretation is that hard styles may require perceptual validation because emotion2vec's prototype for "disgust" may not match the generated acoustic expression. That is why Joe's review matters.

### Q: Did age/gender controls work?

A: Not perceptually yet. Gender is separable in embeddings/latents, but the first listening panel did not produce clear age/gender control. It should remain diagnostic/future work unless Joe wants it prioritized.

### Q: Is privacy solved?

A: No. We have stronger evidence of identity shift, including ECAPA external verifier results, but formal DP accounting and independent EER trials remain open paper tasks.

### Q: What should happen next?

A: First, Joe should review the focused `anger`/`disgust` bundle. If he confirms `disgust`, we treat it as a perceptual success with metric mismatch and focus repair on `anger` intelligibility. If he does not confirm it, then both hard styles remain unsolved and the next step is stronger generated-audio feedback/selection.

## 10. Proposed Next Two-Week Plan

Recommended order:

1. Get Joe's focused review of `disgust` and `anger`.
2. Encode his review into the ratings/notes artifacts.
3. If `disgust` is confirmed, update findings to say emotion2vec under-scores that style and prepare a small perceptual-review protocol.
4. Start an anger content-repair branch: preserve the audible anger style pressure while lowering WER/distortion.
5. Add eval-suite preflight for `ffmpeg` / `torchcodec`.
6. Start paper-methods documentation around the current reference result, metrics, and limitations.

## 11. Meeting Outcome Template

After the meeting, paste notes back into Codex using this structure:

```text
Joe's read on disgust:

Joe's read on anger:

Does Joe accept metric-vs-human mismatch framing?

Should age/gender stay active or move to future work?

Should we start paper writing/method docs now?

Agreed next branch/task:

Any exact phrases Joe used that should go into FINDINGS or WORKLOG:
```
