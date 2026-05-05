# Key Findings — Controllable DP Voice Conversion

**Last updated:** 2026-05-05 (Finding 32 adds the decoder-prototype pilot family and generated-audio failure mining)
**Authors:** Stephen Oladele, Joe Near

---

## Thesis

Controllable differentially private voice conversion: anonymize a speaker's identity via calibrated noise on speaker embeddings, while explicitly controlling perceptual style attributes (emotion, whisper, etc.) in the output. No prior work combines DP guarantees with explicit style controllability.

---

## Finding 1: Speaker Embeddings Encode Style Only in Some VC Systems

**ControlVC's D_VECTOR does not encode style.**
- Separability ratio: 0.88 (need >>1)
- Between-style centroid distance (0.030) < within-style variance (0.034)
- Even at 5x amplification of style latent dims, zero audible difference
- Root cause: D_VECTOR encodes speaker identity; F0/prosody are handled separately

**OpenVoice's speaker embedding does encode style.**
- OpenVoice embeds F0/prosody profile into the speaker embedding
- Whisper achieves separability ratio 1.30 (clearly separable)
- Confirmed by Joe: "OpenVoice embeds the F0 profile in the speaker embedding"

**Implication for the field:** Researchers must verify that their chosen VC system's speaker embedding actually carries the features they want to control. This is not guaranteed and varies across systems.

---

## Finding 2: Speaker Diversity is Critical for Learning Generalizable Style

**3 speakers (Expresso only):** VAE memorizes speaker-specific tonal patterns instead of learning style.
- Only whisper (ratio 1.30) and sad (ratio 0.39) produce audible differences
- Happy/laughing/confused indistinguishable from baseline
- Within-style variance dominated by speaker identity, not emotion

**91 speakers (CREMA-D):** VAE forced to learn cross-speaker emotion patterns.
- All 6 emotions acoustically distinct
- Happy works for the first time (highest F0 variance)

**94 speakers (CREMA-D + Expresso combined):** Best of both worlds.
- All 9 styles perceptually distinct and acoustically confirmed
- Happy shows 2x F0 variation over baseline

**Progression:**

| Model | Speakers | Working styles | Happy? | Takeaway |
|-------|----------|---------------|--------|----------|
| ControlVC + Expresso | 3 | 0 of 11 | No | Embedding doesn't carry style at all — no amount of VAE training can recover what isn't there. Dead end for ControlVC. |
| OpenVoice + Expresso | 3 | 2 of 11 (whisper, sad) | No | Right embedding, wrong training set — too few speakers, so the VAE memorizes speaker identity instead of learning generalizable style. Only the most extreme styles (whisper, sad) break through. |
| OpenVoice + CREMA-D | 91 | 6 of 6 | Yes | Speaker diversity is the unlock — with 91 speakers the VAE is forced to disentangle style from identity. Happy works for the first time. |
| OpenVoice + Combined | 94 | 9 of 9 | Yes | Combines CREMA-D's 91-speaker backbone with Expresso's unique styles (confused, enunciated, whisper). All 9 styles controllable. |

---

## Finding 3: Style Control Survives Differential Privacy Noise

Tested at noise levels 0, 0.1, 0.3, 0.5, and 1.0 with the combined VAE (v6, 9 styles, 94 speakers).

**Metric — spectral centroid (brightness).** Spectral centroid is the "center of mass" of the short-time power spectrum, reported in Hz. A higher centroid means more energy sits in the high frequencies, which listeners perceive as a brighter or sharper voice (e.g., whisper, anger); a lower centroid is dimmer/darker (e.g., neutral, sad). We use deltas from the uncontrolled baseline, so units are Hz-shift relative to what OpenVoice would have produced with no style control. **Why brightness and not F0:** Finding 6 showed that brightness is the one acoustic correlate that moves consistently across source speakers; F0 does not. Brightness is therefore the metric we trust for measuring whether style control is working.

**Noise scale.** Gaussian DP noise with standard deviation equal to `noise_level` is added to the VAE-decoded speaker embedding. `noise=0` is no privacy; `noise=1.0` is heavy privacy (formal ε TBD — see Open Questions).

**Brightness (spectral centroid) deltas from baseline persist across noise levels:**

| Style | noise=0 | noise=0.1 | noise=0.3 | noise=0.5 | noise=1.0 | Takeaway |
|-------|---------|-----------|-----------|-----------|-----------|----------|
| whisper | +945 | +631 | +280 | +393 | +266 | Brightest style at every noise level — whisper's high-frequency breathy signature is the most privacy-robust of any style we tested. |
| anger | +369 | +177 | -136 | -93 | -113 | Direction *flips* under noise — the bright-edge signature of anger is fragile past noise=0.3. Anger is the least noise-robust controllable style. |
| neutral | -533 | -678 | -759 | -527 | -442 | Most stable downward shift — neutral's "dim, subdued" signature stays intact across every noise level. Safe choice under strong privacy. |
| sad | -272 | -414 | -705 | -668 | -633 | Consistent downward shift like neutral — confirms sad is acoustically closer to the low-energy end than to anger/whisper. |

**Style differences diminish but do not disappear under DP noise.** This is the expected privacy-utility tradeoff — higher noise provides stronger privacy at the cost of some style fidelity.

**Unexpected finding:** At high noise levels (0.3+), the uncontrolled baseline becomes unintelligible (F0 collapses to 0), but style-controlled outputs retain speech-like qualities. **Style control partially rescues speech intelligibility from noise destruction.** This means controllable DP-VC produces *better* outputs than uncontrolled DP-VC at the same privacy level.

---

## Finding 4: Acoustic Signatures Match Emotional Expectations

Combined model (v6) acoustic analysis, all deltas from uncontrolled baseline:

| Style | dF0 Mean | dF0 Std | dF0 Range | dBrightness | Acoustic signature |
|-------|----------|---------|-----------|-------------|-------------------|
| anger | +13.2 | -3.9 | +4.4 | +369 | Sharp, bright, edgy |
| confused | +2.5 | -3.8 | -29.5 | +111 | Hesitant, narrow range |
| disgust | -25.6 | +6.1 | +24.3 | -156 | Low, withdrawn |
| enunciated | -10.5 | -3.9 | +5.2 | +274 | Crisp, bright articulation |
| fear | +17.4 | +4.8 | +80.8 | -155 | Tense, highly variable |
| happy | +26.8 | +33.2 | +240.7 | +84 | Most animated — 2x F0 variation |
| neutral | -46.0 | +6.7 | +38.8 | -533 | Flat, subdued, dimmest |
| sad | -0.3 | +11.2 | +44.3 | -272 | Darker, more varied |
| whisper | -128.6 | -30.8 | -143.6 | +945 | F0 near zero, breathy/airy |

These patterns align with established prosodic correlates of emotion in the speech science literature.

---

## Finding 5: Embedding Separability Alone Does Not Predict VAE Success

CREMA-D raw embedding separability ratio (0.54) was *worse* than Expresso (0.62), yet the CREMA-D VAE produced better perceptual results. This is because:

1. Linear centroid analysis misses nonlinear structure that the VAE encoder learns
2. Speaker diversity matters more than raw separability — with 91 speakers, the VAE has enough variation to disentangle style from identity
3. The VAE's label loss during training is a better predictor of downstream controllability than raw embedding separability

---

## Finding 6: Style Generalizes Across Speakers via Brightness, Not F0

Tested on 5 source speakers (1 known male + 4 CREMA-D speakers with 5-8s audio) at noise=0, control_strength=5.0.

**Spectral brightness (centroid) is direction-consistent across speakers for 7/9 styles:**

| Style | trump | spk1007 | spk1023 | spk1045 | spk1076 | Consistent? | Takeaway |
|-------|-------|---------|---------|---------|---------|-------------|----------|
| anger | +795 | collapsed | +364 | +561 | +398 | YES | Brighter in every non-collapsed speaker — the one collapse is a speaker-specific edge case, not a control failure. |
| confused | +536 | +84 | -180 | -79 | +71 | NO | "Confused" doesn't have a single brightness signature — speakers express hesitation via different acoustic means (pause structure, pitch contour, etc.), not consistent spectral shift. |
| enunciated | +759 | -412 | -731 | -616 | -658 | NO | Sign flip between trump (+759) and all CREMA-D speakers (negative) — enunciation is likely domain-sensitive; it behaves differently on scripted CREMA-D voices than on the conversational Trump sample. |
| fear | +259 | +580 | +375 | +558 | +106 | YES | Consistently brighter — tense, elevated formants hold across all tested speakers. |
| happy | +550 | +342 | +297 | +191 | +290 | YES | Reliable — happy voices are measurably brighter in every speaker we tested. Matches Finding 4's 2× F0 variation. |
| neutral | -325 | collapsed | collapsed | -43 | -184 | YES | Direction consistent where defined, but two collapses suggest "neutral" can push already-neutral source voices into degenerate territory. |
| sad | +117 | +358 | +21 | +243 | +57 | YES | Surprisingly upward — CREMA-D's "sad" has a tense, pleading quality rather than a mellow one, so it lands as slightly *brighter*, not darker. |
| whisper | +973 | +956 | +350 | +720 | +758 | YES | Strongest and most consistent cross-speaker signal of any style — whisper is the one style that generalizes nearly perfectly. |

**F0 direction is NOT consistent** — only 1/9 styles (happy) showed the same F0 shift direction across all speakers. This is expected: OpenVoice encodes timbre/tonal color in the speaker embedding, not F0 directly. F0 changes are a secondary effect that depends on how each source voice interacts with the modified embedding.

**Collapses:** 4/45 outputs (9%) had F0=0 (unintelligible). This occurred when the combination of source speaker embedding + style control pushed the reconstruction into a degenerate region. Affects anger and neutral for spk1007, and disgust/neutral for spk1023.

**At noise=0.1:** Brightness consistency maintained (7/9), but collapses increased to 5/45 (11%). The privacy noise makes some speakers more vulnerable to collapse.

**Control strength tradeoff:**
- strength=5.0: 7/9 consistent, 4 collapses
- strength=3.0: 3/9 consistent, 2 collapses
- strength=2.0: 4/9 consistent, 2 collapses

Higher control strength gives better cross-speaker consistency but increases collapse risk. This suggests that the latent dimensions need to be pushed far enough to dominate the source speaker's baseline characteristics, but this can exceed the decoder's valid input range for some speakers.

**Joe's feedback (April 16 call):** F0 inconsistency is expected and not a problem — different people express emotion through pitch differently. F0 is not a knob we should control; the knobs are the style labels themselves (anger, happy, etc.). Brightness and F0 are measurement metrics, not user-facing controls. Joe also noted that collapses are expected when the VAE goes outside its training distribution and don't require a perfect fix.

**Implication:** The system works across diverse speakers for the majority of styles, with spectral brightness as the reliable cross-speaker acoustic correlate. Papers should report brightness as the primary measurement metric rather than F0.

---

## Finding 7: Emotion Classification Reveals a Training Gap, Not a Control Gap

### Methodology

**What emotion2vec_plus_large is.** A self-supervised universal speech-emotion encoder (Ma et al., ACL 2024; released as `iic/emotion2vec_plus_large` on HuggingFace, loaded here through `funasr`). It takes a raw waveform and produces (1) a fixed-dimensional emotion embedding and (2) softmax probabilities over 9 Chinese-English bilingual labels (`angry`, `disgusted`, `fearful`, `happy`, `neutral`, `sad`, `surprised`, `other`, `<unk>`). It is the current state-of-the-art universal emotion encoder, and it is what the EmoVoice paper (arxiv 2504.12867) uses to benchmark emotional TTS — so using it keeps us numerically comparable to the EmoVoice evaluation pipeline.

**How we score each file.** For every generated `.wav` we run emotion2vec, take the argmax label, strip the bilingual prefix (e.g. `生气/angry` → `angry`), and compare it to our target style. Three of our nine styles (`confused`, `enunciated`, `whisper`) have no emotion2vec counterpart — we report emo_sim only for those, with no Recall Rate.

**Design choice — `_plus_large` over the base model.** The `_plus` variant is fine-tuned on additional labeled emotion corpora (IEMOCAP-style), which gives higher absolute accuracy on our CREMA-D/Expresso-style inputs than the base SSL model. This is the same variant EmoVoice reports, so the numbers are directly comparable.

**Primary metric — Recall Rate.** Does the argmax predicted emotion equal the target style? This is the EmoVoice paper's primary controllability metric. Random-chance baseline over 9 classes ≈ 11%.

**Secondary metric — emo_sim.** Cosine similarity between the emotion2vec embedding of a generated file and the embedding of the *same speaker's uncontrolled baseline*. Measures how far the style control pushes the file through emotion-embedding space, regardless of whether the push lands on the target label. Useful for (a) styles emotion2vec can't label directly, and (b) sanity-checking that "zero recall" doesn't mean "identical to baseline."

**How this relates to the project.** Recall Rate is the headline number Joe asked us to produce before anything else (April 16 call). Getting it above chance proves the VAE controls emotion; getting it to >50% is the bar for a paper-quality result. emo_sim is our fallback signal when the classifier can't help us.

### Results — initial 5-speaker run at control_strength=5.0, noise=0.0:

| Style | Recall Rate | emotion2vec prediction | Takeaway |
|-------|------------|----------------------|----------|
| anger | 2/5 (40%) | mixed: `angry`, `<unk>` | Partial success — 2 samples land on target; the rest are off-distribution enough that emotion2vec refuses to label them. |
| disgust | 0/5 (0%) | mostly `disgusted` → `sad`/`<unk>` | Classifier *does* see disgust-like features but not strongly enough to clear the argmax threshold. Signal is there, calibration is off. |
| fear | 0/5 (0%) | `worried`/`happy`/other | emotion2vec's "fearful" class is narrower than our "fear" — training-data mismatch, not a control failure. |
| happy | 0/5 (0%) | `disgusted`/`<unk>` | The most striking miss. Our "happy" is acoustically real (Finding 4: 2× F0 variation) but emotion2vec labels it as disgust — the latent direction may point at sarcasm/mockery, not joy. |
| neutral | 3/5 (60%) | mostly correct | Neutral is the easiest category for any classifier and the smallest perturbation — expected ceiling. |
| sad | 1/5 (20%) | `neutral`/`disgusted` | Our sad comes out flat enough to read as neutral — our acoustic "sad" doesn't match emotion2vec's prototype. |
| **Overall** | **6/30 (20%)** | | Roughly 2× random chance, well short of the >50% target the paper needs. |

**Full-corpus run (258 files, 27 speaker-variant configs, strength sweep included):**

Reproduced via `python examples/eval_emotion.py --input output/diverse_speakers/ --out output/eval_emotion_full.csv`.

| Style | Recall Rate | Mean emo_sim | emo_sim range | Takeaway |
|-------|-------------|--------------|---------------|----------|
| anger | 8/27 (30%) | 0.937 | 0.851–0.998 | Best recall of any emotion — anger has a recognizable acoustic signature (sharp, bright) that survives the CREMA-D → wild-audio transfer. |
| disgust | 3/27 (11%) | 0.970 | 0.939–0.994 | Recall ≈ random chance — acoustic signature exists but is too subtle for emotion2vec to pick out reliably. |
| fear | 0/27 (0%) | 0.960 | 0.893–0.998 | Zero recall despite high emo_sim: we *are* perturbing the embedding, but not in a direction emotion2vec recognizes as fear. Training-distribution mismatch. |
| happy | 0/27 (0%) | 0.961 | 0.849–0.999 | Same pattern — happy is acoustically real (Finding 4) but classifier-invisible. The strongest evidence that recall is a training-coverage gap, not an architecture gap. |
| neutral | 18/27 (67%) | 0.976 | 0.889–0.998 | Strongest category — validates that the pipeline works end-to-end when the target's acoustic signature is broad enough for emotion2vec. |
| sad | 7/27 (26%) | 0.966 | 0.898–0.998 | Above chance, below ceiling — similar training-gap story as disgust. |
| **Overall** | **36/162 (22.2%)** | — | — | Consistent with the 20% 5-speaker run — not a sample-size problem. Motivates CommonVoice pre-training (Phase 1.5). |
| confused | n/a (no e2v class) | 0.943 | 0.888–0.990 | Embedding is clearly shifted from baseline — confused is a real style even without a recall number. |
| enunciated | n/a (no e2v class) | 0.959 | 0.925–0.995 | Least embedding-disruptive of the non-emotional styles — closest to baseline in emotion-space. |
| whisper | n/a (no e2v class) | **0.875** | **0.616–0.994** | Most embedding-disruptive style of any kind. Validates emo_sim as a useful probe for non-labeled controls: it reveals that whisper genuinely perturbs the emotional signature, even though no classifier class exists for it. |

Per-style emo_sim ranges and the 20% vs 22.2% difference are both consistent with a training-coverage issue rather than a measurement artifact.

**Control strength test on a single speaker (Trump), noise=0.0:**

| Style | s=5.0 | s=10.0 | s=20.0 | Takeaway |
|-------|-------|--------|--------|----------|
| anger | `<unk>` | `<unk>` | `<unk>` | Strength doesn't help — the Trump-speaker "anger" region is simply outside emotion2vec's label space at every intensity. |
| disgust | `neutral` | `neutral` | `disgusted` ✓ | Strength *does* help — disgust reaches target at s=20. Latent direction is correct but weakly encoded; pushing harder works. |
| fear | `disgusted` | `disgusted` | `disgusted` | Latent direction is *wrong* (not weak) — pushing harder just makes it "more disgusted." A strength dial cannot fix a direction error. |
| happy | `disgusted` | `disgusted` | `disgusted` | Same story as fear — the axis we labeled "happy" points into emotion2vec's disgust region. Evidence the VAE learned *something*, but mislabeled. |
| neutral | `sad` | `sad` | `sad` | Odd — controlled "neutral" reads as sad. Likely the label encoding pulls toward low-energy regions during training. |
| sad | `neutral` | `sad` ✓ | `sad` ✓ | Recoverable with strength — sad is weakly encoded but directionally correct. |

**Interpretation of the strength sweep:** the three outcomes — "strength fixes it" (disgust, sad), "strength doesn't help" (anger), "strength makes it more wrong" (fear, happy) — tell us exactly where the training gap lives. Anger/fear/happy need *better training data*, not a bigger strength dial. This is the direct motivation for Phase 1.5 (CommonVoice pre-training).

**Interpretation:** Increasing control strength does help some styles (sad, disgust reach their target at higher values), but doesn't fix anger, happy, or fear. This is not a strength problem — the latent dimensions for those styles don't map cleanly to the emotion2vec categories the classifier was trained on. The VAE learned acoustic features for emotion, but the specific acoustic signature it produces for e.g. "angry" may differ from what emotion2vec considers "angry."

**Root cause hypothesis:** The model was trained on CREMA-D (scripted emotional speech, 91 speakers) and Expresso (expressive storytelling speech, 3 speakers). Both datasets have different recording conditions from natural conversational speech. The emotion representations the VAE learns may be dataset-specific rather than universal. Training on a larger and more diverse base (e.g., CommonVoice for acoustic pre-training) should improve cross-domain generalisation.

**emo_sim scores (0.87–0.97) are consistently high** — every output is emotionally coherent and close in embedding space to baseline speech. This is expected: we're controlling a few latent dimensions, not rewriting the full embedding.

**Whisper is the most embedding-disruptive style (new observation, April 17).** Across all 15 whisper outputs, emo_sim mean = 0.875 and min = 0.616 — substantially lower than every other style. This validates emo_sim as a secondary signal for styles that emotion2vec cannot classify directly (whisper, confused, enunciated): even when Recall Rate is undefined, emo_sim reveals whether the style is actually perturbing the emotional signature. Whisper perturbs it the most, which matches its acoustic profile (near-zero F0, breathy/airy texture).

**Implication:** The evaluation pipeline is working — emotion2vec is sensitive enough to detect the failures. The next step is improving training coverage, not changing the architecture. CommonVoice pre-training (Phase 1.5) is directly motivated by this finding.

---

## Finding 8: Style Control Preserves Intelligibility (WER Sanity Check)

### Methodology

**What WER is.** Word Error Rate is the standard ASR evaluation metric: (substitutions + deletions + insertions) / reference words, after normalization. Lower is better; 0 means the hypothesis and reference transcripts are identical. We compute WER with `jiwer`, which handles the alignment plus a normalization chain (lowercase, strip punctuation, collapse whitespace) so that `"Hello, world."` and `"hello world"` score as identical.

**What we use as the ASR.** OpenAI Whisper `base` model — 74M params, multilingual. We chose `base` over `large` because this is a **sanity check**, not a transcription benchmark: we only need it to be sensitive enough to detect intelligibility loss caused by our control knob. `base` is fast (~real-time on CPU) and its error modes are well-understood.

**Design choice — drift-from-baseline, not absolute WER.** For each `<speaker>_<style>.wav` the reference is the *same speaker's* `<speaker>_baseline.wav` transcription, not a ground-truth script. This is a deliberate methodological choice: we don't care whether Whisper gets the words right in absolute terms (the source audio may be noisy, accented, or hard for Whisper regardless of our pipeline). We care whether the style control *changes what Whisper hears*. A drift-from-baseline WER of 0.15 means "style control caused Whisper to disagree with itself on 15% of words" — a clean attribution to our intervention. `eval_wer.py` also supports `--reference-text` for absolute WER on known scripts, which we'll use when we report on a held-out test set.

**How this relates to the project.** One of the claims of a controllable DP-VC system is that the controls are orthogonal to the content channel — you change *how* something is said, not *what* is said. WER is the direct test of that claim. Without it we cannot rule out "style control works by scrambling the phonemes."

### Results (258 files, 73 scored against a same-speaker baseline):

| Style | Mean WER | Median WER | Min | Max | Takeaway |
|-------|----------|------------|-----|-----|----------|
| happy | **0.110** | 0.000 | 0 | 0.750 | Best-preserving style — median 0 means Whisper recovers the exact baseline transcript most of the time. Happy is "free" on the content channel. |
| neutral | 0.130 | 0.000 | 0 | 0.750 | Expected — neutral is the smallest embedding perturbation. |
| sad | 0.150 | 0.000 | 0 | 0.750 | Slow, flat delivery but ASR still recovers the words — median 0 confirms it. |
| anger | 0.166 | 0.125 | 0 | 0.750 | Low mean WER despite the large acoustic shift — anger stays articulate. Content channel intact. |
| fear | 0.188 | 0.200 | 0 | 0.750 | Moderate — tense, variable F0 hurts ASR slightly but not catastrophically. |
| disgust | 0.200 | 0.143 | 0 | 0.750 | Moderate — low, withdrawn voice degrades recovery slightly. Still acceptable. |
| enunciated | 0.244 | 0.214 | 0 | 0.600 | Counter-intuitive — "crisp articulation" doesn't help Whisper here, probably because Expresso's storytelling-style enunciation is stylistically unusual. |
| confused | 0.275 | 0.250 | 0 | 1.000 | Hesitant delivery genuinely confuses the ASR — content drift is real, not just acoustic. |
| **whisper** | **0.356** | **0.286** | 0 | **1.000** | Worst. Two factors compound: Whisper-the-model is broadly poor on whispered speech regardless, *and* high-strength whisper occasionally collapses entirely (max = 1.0). |

**Interpretation:**

1. **6 of 9 styles preserve intelligibility well** (median WER ≤ 0.20). For happy/neutral/sad the median is exactly 0 — Whisper recovers the same transcription as the baseline.
2. **Whisper is the only style with systemic intelligibility loss** (mean 0.356). Part of this is intrinsic to whispered speech — ASR systems are broadly worse on whisper. But some of it is real degradation from our control, especially at high strength.
3. **WER max of 1.00 occurs for whisper, confused, and a few outliers in other styles.** These are the collapse cases already catalogued in Finding 6 (9% of speaker-style combinations produce degenerate output) showing up in the content-recovery metric.
4. **WER and emo_sim are consistent.** Whisper has both the lowest mean emo_sim (0.875 — largest embedding shift) AND the highest mean WER (0.356 — largest content drift). Both metrics agree that whisper is the style most perturbing to "normal" speech, which is what whisper should be.

**Implication:** The style control is essentially **orthogonal to the content channel**. We can turn style knobs without breaking what the speaker says. This is the sanity-check result we needed before claiming "controllable speaker generation": the system genuinely controls speaker properties without damaging the content channel, which is the promise of separating speaker embedding from HuBERT content codes.

---

## Finding 9: Predicted MOS — Naturalness Holds for Most Styles

### Methodology

**What MOS is.** Mean Opinion Score — a standard 1-to-5 subjective scale for perceptual audio quality, where 1 = "bad" and 5 = "excellent." Traditionally collected via human listening panels; modern papers use **predicted MOS** from a model trained on large panels of human ratings (much cheaper, reproducible, highly correlated with real panels).

**Which predictor we use — and why not UTMOS.** The EmoVoice paper (arxiv 2504.12867) uses UTMOS (Saeki et al., 2022), which is the dominant predicted-MOS model in TTS literature. UTMOS is distributed via `sarulab-speech/utmos`, which in turn requires a from-source build of fairseq. Our codebase monkey-patches fairseq for OpenVoice compatibility, and the two fairseq versions conflict — we cannot install UTMOS without breaking inference. We substitute torchaudio's `SQUIM_SUBJECTIVE` (Meta, 2023), which predicts the same quantity (subjective MOS on the same 1–5 scale) trained on comparable datasets (BVCC + DAPS). SQUIM and UTMOS are not numerically identical, but they measure the same construct and rank-order audio similarly — which is all we need for a *relative* comparison across styles.

**How SQUIM_SUBJECTIVE works — the "non-matching reference" design.** SQUIM takes two inputs: the audio to score, and a *reference* waveform used only to condition the model (not to compare against). The reference can be any clean-speech file — the model uses it to calibrate its judgment of what "natural speech" looks like for this speaker/channel, then scores the test audio on its own merits. This is why `eval_mos.py` defaults to using each file's same-speaker baseline as reference: gives the model a consistent conditioning anchor per speaker. Pass `--reference` to override with a single fixed waveform.

**Design choice — per-speaker baseline delta.** For each style we report `Δ vs same-speaker baseline = MOS(style) − MOS(baseline)` for that speaker. This removes per-speaker MOS variation (some source voices are just noisier than others) and isolates the style knob's naturalness cost.

**How this relates to the project.** Recall Rate (Finding 7) tells us if the style hits the target; WER (Finding 8) tells us if the words survive; MOS tells us if the output still *sounds like a person*. Without MOS, a style-control system could achieve high recall and low WER by producing robotic, artifact-laden audio that a human would rate as unusable. MOS closes that loop.

### Results (258 files, 150 scored — the others had no same-speaker baseline available):

| Style | Mean MOS | Median | Range | Δ vs same-speaker baseline | Takeaway |
|-------|----------|--------|-------|----------------------------|----------|
| baseline | 4.055 | 3.980 | 3.925–4.448 | — | OpenVoice's naturalness ceiling — "good" quality before any style control is applied. Sets the upper bound. |
| fear | 4.074 | 3.979 | 3.966–4.444 | +0.019 | Statistically indistinguishable from baseline — fear is free on the naturalness axis. |
| sad | 4.068 | 3.978 | 3.898–4.438 | +0.013 | No measurable cost. |
| neutral | 4.045 | 3.974 | 3.916–4.402 | −0.010 | No meaningful cost. |
| happy | 4.025 | 3.984 | 3.392–4.411 | −0.030 | Tiny mean cost, but note the low end (3.39) — occasional samples are noticeably degraded, probably collapse-adjacent cases. |
| disgust | 4.021 | 3.972 | 3.872–4.411 | −0.034 | Small cost, no outliers — disgust is a benign style in naturalness terms. |
| enunciated | 3.932 | 3.947 | 3.069–4.448 | −0.123 | First style with a measurable cost, but still "good." The low-end (3.07) reveals that aggressive enunciation can sound unnaturally over-articulated. |
| anger | 3.760 | 3.975 | **1.407**–4.377 | −0.294 (bimodal; median healthy) | Bimodal failure — median is still fine (3.98), but a few outliers crash to 1.4 MOS and drag the mean. These are the collapse cases from Finding 6 showing up in MOS. |
| confused | 3.704 | 3.808 | 2.892–4.421 | −0.351 | Systemic cost — every sample is a little less natural, no clean cases. Hesitant/halting speech intrinsically scores lower. |
| **whisper** | **3.623** | 3.904 | **2.503**–4.170 | **−0.432** | Worst naturalness cost overall — whisper is genuinely hard for any speech synthesis system because its acoustic profile (near-zero F0, breath) is atypical for the MOS model's training data. |

**Interpretation:**

1. **Baseline MOS ≈ 4.05** — the OpenVoice VC pipeline itself produces "good"-to-"very good" naturalness, before any style control is applied. This sets the naturalness ceiling we're operating under.
2. **6 of 9 styles stay within 0.12 MOS of baseline** (fear, sad, neutral, happy, disgust, enunciated). The style knob preserves perceived naturalness for most emotions.
3. **Three styles degrade naturalness measurably**: whisper (−0.43), confused (−0.35), anger (−0.29). The anger degradation is bimodal — median is fine (3.98) but high-strength trump samples drop to 1.4 MOS. Whisper and confused are systemic: every sample is somewhat degraded.

**Cross-metric convergence (the most important observation).** All three independent evaluation metrics — emo_sim (Finding 7), WER (Finding 8), and predicted MOS (this finding) — rank the same three styles as hardest: **whisper, confused, anger**. Each metric measures a different quantity (embedding shift, content-channel drift, naturalness perception), yet they converge on the same ordering.

| Style | emo_sim mean | WER mean | MOS mean | Worst-on-all? | Takeaway |
|-------|--------------|----------|----------|---------------|----------|
| whisper | 0.875 | 0.356 | 3.623 | ✓ | Hardest style on every axis — embedding shift, content drift, and naturalness all agree. This is where training coverage matters most. |
| confused | 0.943 | 0.275 | 3.704 | ✓ | Second-hardest — hesitant delivery disrupts all three channels, but less severely than whisper. |
| anger | 0.937 | 0.166 | 3.760 | ✓ (on MOS & emo_sim) | Third-hardest — good on WER, bimodal on MOS. Failures are collapse cases, not the style itself being unnatural. A collapse-detector would recover most of anger's MOS. |

This triangulation is what we'd want to see before trusting any single metric. It means the signal is real — these styles are genuinely harder for our current model, not an artifact of any one evaluator.

**Implication:** For the paper, we can report MOS to confirm that style control preserves naturalness for most styles, and flag whisper/confused/anger as the training-coverage edge cases. Combined with WER (intelligibility) and emotion2vec (target alignment), we have a three-axis evaluation that mirrors the EmoVoice (arxiv 2504.12867) pipeline.

---

## Finding 10: Validation-Scale CommonVoice Pre-Training Improves WER but Collapses Style Toward Neutral

### Methodology

To test the Phase 1.5 hypothesis without waiting for a full Common Voice mirror, we built a **local validation-scale English subset** from downloaded Common Voice 21.0 shards:

- local clips available: `28,116`
- local speakers matched in `validated.tsv`: `6,795`
- training subset used for this pass: **`500` speakers / `1,202` clips** (`cv500`)

Pipeline:

1. Build a local `validated.tsv` + `clips/` subset with `scripts/prepare_commonvoice_subset.py`
2. Extract OpenVoice speaker embeddings with `examples/openvoice_extract_commonvoice.py`
3. Run reconstruction-only pre-training with `examples/openvoice_pretrain_vae_commonvoice.py`
4. Finetune the existing CREMA-D + Expresso controllable VAE with `examples/openvoice_train_vae_combined.py --init-checkpoint ...`
5. Re-run the same 110-file evaluation corpus and compare against the current combined-only baseline

This is an intentionally conservative test: **same controllable training recipe, same source speakers, same evaluation scripts**. The only change is the CommonVoice initialization.

### Results (`cv500` candidate vs current combined-only baseline)

| Metric | Combined-only baseline | CommonVoice `cv500` init | Delta | Takeaway |
|--------|------------------------|--------------------------|-------|----------|
| Emotion recall | `17/66 = 25.8%` | `11/66 = 16.7%` | `-9.1 pts` | Worse overall controllability |
| Neutral recall | `9/11 = 81.8%` | `11/11 = 100%` | `+18.2 pts` | Model got even better at staying neutral |
| Anger recall | `3/11 = 27.3%` | `0/11 = 0%` | `-27.3 pts` | Previously recoverable style collapsed |
| Disgust recall | `1/11 = 9.1%` | `0/11 = 0%` | `-9.1 pts` | No gain |
| Sad recall | `4/11 = 36.4%` | `0/11 = 0%` | `-36.4 pts` | Lost one of the stronger styles |
| Neutral predictions across all 110 files | `70/110` | `109/110` | `+39 files` | Strong neutral-collapse signature |
| Mean WER across all 99 non-baseline style rows | `0.235` | `0.084` | `-0.151` | Much better content preservation |
| Median WER across all 99 non-baseline style rows | `0.143` | `0.000` | `-0.143` | Most candidate outputs transcribe exactly like baseline |
| Whisper mean WER | `0.448` | `0.111` | `-0.337` | Even the hardest style became far easier for Whisper |

The emotion2vec output distribution makes the failure mode unambiguous: the `cv500` candidate predicts **`neutral` for 109 of 110 files**. Mean `emo_sim` values also jump extremely close to baseline (`0.979–0.998` across styles), which is exactly what we'd expect from a model that learned a tighter reconstruction prior but weakened its style offsets.

### Interpretation

This is a **real negative result**, and an informative one:

1. **Naive reconstruction-only CommonVoice pre-training is not automatically helpful for controllable style generalization.** On this validation-scale run it made the model *less* expressive, not more.
2. **The gain is real on the content channel.** WER improved substantially, so the pre-training is doing something meaningful: it is regularizing the speaker embedding toward more stable, baseline-like speech.
3. **The failure mode is over-regularization, not random noise.** The model did not become chaotic; it became too conservative. Almost everything collapsed back toward the neutral/baseline region.
4. **This does not kill the CommonVoice direction.** It narrows the hypothesis. The question is no longer "does broader speaker coverage help at all?" The question is: **how do we keep the broader speaker prior without washing out the labeled style axes?**

### Implication

For the paper, this gives us a strong and honest intermediate result:

- **positive:** CommonVoice pre-training can improve intelligibility/content preservation dramatically
- **negative:** the current `cv500` recipe hurts controllability by collapsing style toward neutral
- **next step:** scale the subset and test gentler finetuning or partial freezing before claiming CommonVoice pre-training improves emotion recall

This is exactly the kind of result worth reporting because it turns a vague Phase 1.5 idea into a concrete research question with a measurable failure mode.

---

## Finding 11: Speaker Novelty Metric Confirms Real Identity Shift in the Combined Model and Collapse in the `cv500` Model

### Methodology

We added a **speaker novelty metric** in the native OpenVoice speaker-embedding space. For each generated output we extract:

- the source speaker embedding from the original source audio
- the generated speaker embedding from the output waveform
- the same-speaker baseline conversion embedding when available

Then we compute:

- **similarity** = cosine similarity(source, generated)
- **distance** = `1 - similarity`
- **novelty gain vs baseline** = similarity(source, baseline) - similarity(source, generated)

The key quantity is **novelty gain vs baseline**. It answers the paper question more cleanly than raw similarity alone:

- positive value: the style-controlled output moved farther from the source than plain voice conversion already does
- near-zero value: the style-controlled output is not meaningfully more novel than the baseline conversion

This is a **proof-of-novelty** metric in our framing, not yet a full privacy metric. It uses the system's native embedding space and is intended to answer "did we generate a genuinely shifted speaker identity?" before we add an independent speaker-verification/EER pipeline.

We ran it on the same two 110-file validation corpora used in CommonVoice pretraining pipeline:

1. combined-only checkpoint: `results/eval_novelty_pass2_combined.csv`
2. CommonVoice `cv500` init checkpoint: `results/eval_novelty_pass2_cv500.csv`

### Results

| Metric | Combined-only model | CommonVoice `cv500` model | Takeaway |
|--------|---------------------|---------------------------|----------|
| Mean source-to-baseline similarity | `0.9101` | `0.8682` | Both baseline conversions remain relatively close to the source, though `cv500` baseline is already a bit farther away |
| Mean source-to-styled similarity | `0.6502` | `0.8313` | Combined-only styles are much farther from source; `cv500` styles stay much closer |
| **Mean novelty gain vs baseline** | **`0.2599`** | **`0.0369`** | Combined-only model creates substantially more novel speakers; `cv500` barely moves beyond baseline conversion |

Per-style novelty gain vs baseline (higher = more novel than the baseline conversion):

| Style | Combined-only | `cv500` | Takeaway |
|-------|---------------|---------|----------|
| whisper | `0.6561` | `0.0337` | Strongest example of the collapse: whisper is highly speaker-shifting in the combined model, but almost baseline-like after `cv500` pretraining |
| confused | `0.5589` | `0.0280` | Same story — style difference largely disappears |
| enunciated | `0.4495` | `0.0023` | Nearly no novelty beyond baseline under `cv500` |
| anger | `0.1506` | `0.0371` | Novelty compressed |
| fear | `0.1027` | `0.1015` | The one style that remains comparably speaker-shifting in both models |

The raw similarity values also line up with intuition from the earlier findings:

- In the combined-only model, **whisper** and **confused** are the most identity-shifting styles, which matches their cross-metric difficulty in WER and MOS.
- In the `cv500` model, nearly every style clusters close to the source/baseline region, which matches the emotion-classifier collapse toward `neutral`.

### Interpretation

This finding is useful for two reasons:

1. **It validates the main model on a new axis.** The combined-only controllable model is not just changing classifier labels or acoustic surface features; it is producing speaker embeddings that move substantially away from the source relative to baseline conversion.
2. **It independently confirms the `cv500` failure mode.** The CommonVoice-pretrained model does not just collapse in emotion2vec space; it also stops producing meaningfully novel speaker identities for most styles. That makes the CommonVoice pretraining pipeline negative result much harder to dismiss as a quirk of one evaluator.

In other words, the novelty metric and the emotion metric tell the same story from different angles:

- **combined-only model**: expressive, identity-shifting, but imperfectly aligned to target emotion labels
- **`cv500` model**: more conservative, more transcript-stable, but much less novel and much less expressive

### Implication

For the paper, this closes an important gap in the evaluation stack:

- emotion2vec Recall/emo_sim: does the style target land?
- WER: does the content survive?
- MOS: does it still sound natural?
- **Novelty**: does the output become a genuinely different speaker profile?

It also sharpens the CommonVoice follow-up agenda. The next question is no longer just "can pre-training improve recall?" It is:

**Can we preserve the intelligibility gains of broader pre-training without collapsing both emotion control and speaker novelty back toward the baseline identity?**

---

## Finding 12: The Combined Model Best Balances Control, Novelty, Intelligibility, and Naturalness

### Methodology

We ran a **five-condition ablation matrix** on the same 11-speaker validation corpus used in Findings 10 and 11, with the same evaluation stack:

- emotion2vec Recall Rate + emo_sim
- native OpenVoice novelty gain vs baseline
- Whisper drift-from-baseline WER
- SQUIM_SUBJECTIVE predicted MOS delta vs baseline

The five conditions were:

1. **combined** — the main 9-style CREMA-D + Expresso checkpoint
2. **CommonVoice `cv500` init** — the CommonVoice pretraining pipeline finetuned checkpoint
3. **CREMA-D only** — a speaker-diverse 6-style checkpoint trained only on CREMA-D labels
4. **Expresso only** — a 6-style checkpoint trained only on the balanced Expresso subset
5. **naive baseline** — the combined checkpoint, but with deterministic random control vectors applied only to the six *unlabeled* latent dims (9-14), L2-matched to the style-control strength

That naive condition is important. It answers a harder question than "does noise change the speaker?" It asks:

**Can arbitrary latent perturbation create something that looks as good as labeled style control?**

### Results

| Condition | Supported styles | Emotional rows scored | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Takeaway |
|-----------|------------------|-----------------------|--------|--------------------------|----------|----------------|----------|
| combined | 9 | 66 | 25.8% | 0.2599 | 0.2353 | -0.0792 | Best overall tradeoff — non-trivial target alignment and non-trivial identity shift, with moderate WER/MOS cost |
| CommonVoice `cv500` init | 9 | 66 | 16.7% | 0.0369 | 0.0844 | -0.0123 | Sounds stable and transcribes well, but style and novelty collapse back toward baseline |
| CREMA-D only | 6 | 66 | 16.7% | 0.0158 | 0.0534 | +0.0059 | Excellent stability, weak speaker shift — speaker-diverse emotion data alone is not enough |
| Expresso only | 6 | 33 | 36.4% | -0.0008 | 0.0549 | -0.0120 | Superficially decent recall on a tiny emotional subset, but almost no identity shift at all |
| naive free-dim baseline | 9 | 66 | 18.2% | 0.4708 | 0.1579 | -0.6453 | High novelty without clean control — random latent pushes are not a substitute for labeled style axes |

### Collapse taxonomy

To keep "collapse" from meaning everything and nothing, we tracked four failure types:

- **content collapse**: `WER >= 0.8`
- **style collapse to neutral**: target is emotional, but emotion2vec predicts `neutral`
- **identity collapse to baseline**: novelty gain vs baseline `<= 0.05`
- **mixed collapse**: at least two collapse axes on the same file

Condition-level summary:

| Condition | Content collapse | Style collapse to neutral | Identity collapse to baseline | Mixed collapse | Takeaway |
|-----------|------------------|---------------------------|-------------------------------|----------------|----------|
| combined | 6 | 37 | 7 | 4 | Fewer failures overall, but the failures that remain are spread across content, style, and identity |
| CommonVoice `cv500` init | 0 | 54 | 72 | 34 | Classic conservative collapse: no catastrophic audio failures, but the model washes back toward neutral and baseline identity |
| CREMA-D only | 0 | 55 | 61 | 50 | Very stable audio, but almost every row is a style/identity collapse rather than a successful controlled speaker shift |
| Expresso only | 0 | 21 | 66 | 21 | The dominant failure is identity collapse — outputs stay too close to the baseline speaker profile |
| naive free-dim baseline | 1 | 41 | 0 | 0 | The failure is the opposite of `cv500`: lots of speaker movement, but poor emotional targeting and degraded naturalness |

### Interpretation

This ablation answers several paper-critical questions at once:

1. **The combined model is still the main model.** It is not the cleanest on every single metric, but it is the only condition that jointly preserves meaningful target alignment and meaningful identity shift.
2. **Single-dataset stability is not enough.** `CREMA-D only`, `Expresso only`, and `cv500` all show that a model can sound natural and transcribe well while still failing the actual controllable-speaker objective.
3. **Novelty alone is not the goal.** The naive baseline produces *more* novelty than the combined model, but its recall is worse and its MOS collapses by `-0.6453`. This is the clearest evidence yet that the project is not "make the embedding move"; it is "make the embedding move in a useful, controllable way."
4. **The CommonVoice direction remains open but narrowed.** `cv500` is better framed now as a stability-biased initialization that over-regularizes the system unless the finetuning recipe changes.

### Implication

evaluation ablation matrix turns the repo from "we have a working system" into "we have a defensible comparison story":

- **combined** is the paper's current headline model
- **CommonVoice `cv500` init** is a real negative result and a focused follow-up agenda
- **CREMA-D only** and **Expresso only** explain *why* the combined condition is necessary
- **naive free-dim baseline** shows why labeled style supervision matters beyond raw novelty

That is the comparison structure the paper needs.

---

## Finding 13: Simple Gentler Finetuning Does Not Fix the CommonVoice Collapse

### Methodology

CommonVoice finetune ablation kept the **CommonVoice `cv500` pretrained checkpoint fixed** and asked a
much narrower question than Finding 10:

**Was the conservative `cv500` collapse mainly caused by an overly aggressive finetuning policy?**

To test that, we held the dataset and evaluation stack constant and changed
only the finetuning recipe used to adapt the CommonVoice-pretrained VAE onto
the combined CREMA-D + Expresso labels.

We compared the existing references:

1. **combined** — the main paper checkpoint
2. **CommonVoice `cv500` init** — the original CommonVoice pretraining pipeline finetune

against five new finetuning variants:

1. **`cv500_ft_short`** — fewer finetune epochs (`1000` instead of `3000`)
2. **`cv500_ft_low_lr`** — lower learning rate (`3e-7` instead of `1e-6`)
3. **`cv500_ft_short_low_lr`** — fewer epochs plus lower learning rate
4. **`cv500_ft_freeze_decoder`** — finetune only the encoder/latent side
5. **`cv500_ft_freeze_encoder`** — finetune only the decoder

All conditions used the same:

- CommonVoice init checkpoint: `embeddings/openvoice_vae_commonvoice_cv500.pt`
- combined finetuning embeddings: `embeddings/openvoice_combined_emb.pt`
- 11-speaker validation corpus format
- four-metric evaluation stack from Findings 10-12

### Results

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Takeaway |
|-----------|--------|--------------------------|----------|----------------|----------|
| combined | 25.8% | 0.2599 | 0.2353 | -0.0792 | Still the paper model: the only condition with both meaningful control and meaningful identity shift |
| CommonVoice `cv500` init | 16.7% | 0.0369 | 0.0844 | -0.0123 | Stable but collapsed reference point |
| `cv500_ft_short` | 16.7% | 0.0558 | 0.0390 | -0.0108 | Better novelty than raw `cv500`, but no recall recovery |
| `cv500_ft_low_lr` | 16.7% | 0.0495 | 0.0340 | -0.0142 | Slight novelty gain, even lower WER, still no control recovery |
| `cv500_ft_short_low_lr` | 16.7% | 0.0692 | 0.0791 | -0.0441 | Best novelty recovery among the CommonVoice variants, still far from combined |
| `cv500_ft_freeze_decoder` | 16.7% | 0.0192 | 0.0330 | -0.0154 | Worst novelty of the sweep; freezing the decoder makes identity collapse worse |
| `cv500_ft_freeze_encoder` | 18.2% | 0.0594 | 0.0905 | -0.1261 | Only variant with any recall gain, but it gives back too much naturalness |

### Collapse behavior

The collapse summary makes the failure mode more precise.

| Condition | Style collapse to neutral | Identity collapse to baseline | Mixed collapse | Takeaway |
|-----------|---------------------------|-------------------------------|----------------|----------|
| CommonVoice `cv500` init | 54 | 72 | 34 | Original conservative collapse pattern |
| `cv500_ft_short` | 55 | 55 | 26 | Identity collapse improves, but style collapse does not |
| `cv500_ft_low_lr` | 55 | 61 | 29 | Slightly better than raw `cv500`, still mostly conservative collapse |
| `cv500_ft_short_low_lr` | 55 | 44 | 22 | Best partial recovery: identity collapse drops the most, but style collapse stays unchanged |
| `cv500_ft_freeze_decoder` | 54 | 88 | 49 | Decoder freeze is too blunt; it worsens the identity-collapse failure |
| `cv500_ft_freeze_encoder` | 54 | 63 | 33 | Slight recall gain, only modest identity recovery, worse naturalness |

### Interpretation

This pass narrows the CommonVoice agenda in an important way:

1. **The `cv500` failure is not just “too many finetune steps.”** Shorter training and lower LR help only marginally.
2. **The `cv500` failure is not solved by coarse freezing either.** Whole-module encoder/decoder freezes are too blunt to restore the combined model's tradeoff.
3. **Novelty is easier to recover than emotional alignment.** Several variants improve novelty over raw `cv500`, but four of the five do not improve recall at all.
4. **The best partial recipe is `cv500_ft_short_low_lr`, not because it wins outright, but because it reduces identity collapse the most without destroying WER/MOS.**
5. **The decoder-freeze result is particularly informative.** It suggests that preserving the pretrained decoder too rigidly keeps the model trapped near the conservative CommonVoice prior.

### Implication

CommonVoice finetune ablation changes the next-step recommendation for the paper:

- We should **not** frame the CommonVoice issue as something that simple “gentler finetuning” already solved.
- We **can** say that modest identity recovery is possible without giving back all the intelligibility gain.
- The next meaningful experiments are now narrower and more defensible:
  - larger-scale CommonVoice with the best partial recipe (`short_low_lr`)
  - better pretraining objectives, not just reconstruction-only pretraining
  - finer-grained freeze or loss-weight schedules rather than coarse encoder/decoder freezing

That is a much sharper research position than the repo had after Finding 10 alone.

---

## Finding 14: Simple Objective Reweighting Does Not Recover Controllability After CommonVoice Pretraining

### Methodology

CommonVoice objective ablation asked a narrower follow-up question than Finding 13:

**Was the CommonVoice `cv500` collapse mainly an objective-balance problem that we
could fix by reweighting the existing reconstruction, KL, and label losses
during combined finetuning?**

To test that, we kept the data, evaluation corpus, and four-metric stack fixed,
started from the same CommonVoice-pretrained init checkpoint, and changed only
the loss weights used during combined finetuning.

We kept these pieces unchanged:

- CommonVoice init checkpoint: `embeddings/openvoice_vae_commonvoice_cv500.pt`
- combined finetuning embeddings: `embeddings/openvoice_combined_emb.pt`
- 11-speaker validation corpus format
- same evaluation stack from Findings 10-13
- same best CommonVoice finetune ablation training schedule foundation: `1000` epochs at `3e-7`

We compared three references:

1. **combined** — the current paper checkpoint
2. **CommonVoice `cv500` init** — the original conservative CommonVoice result
3. **`cv500_ft_short_low_lr`** — the best partial-recovery finetune recipe from Finding 13

against four new objective variants:

1. **`cv500_obj_label2`** — label weight `2.0`, reconstruction weight `1.0`
2. **`cv500_obj_label4`** — label weight `4.0`, reconstruction weight `1.0`
3. **`cv500_obj_label_ramp`** — label weight ramps from `1.0` to `4.0`
4. **`cv500_obj_recon_half_label2`** — reconstruction weight `0.5`, label weight `2.0`

### Results

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Takeaway |
|-----------|--------|--------------------------|----------|----------------|----------|
| combined | 25.8% | 0.2599 | 0.2353 | -0.0792 | Still the only condition with both meaningful control and meaningful identity shift |
| CommonVoice `cv500` init | 16.7% | 0.0369 | 0.0844 | -0.0123 | Conservative collapse reference point |
| `cv500_ft_short_low_lr` | 16.7% | 0.0692 | 0.0791 | -0.0441 | Best CommonVoice partial-recovery reference from Finding 13 |
| `cv500_obj_label2` | 16.7% | 0.0589 | 0.0862 | -0.0300 | Best of the new objective variants, but still below `cv500_ft_short_low_lr` |
| `cv500_obj_label4` | 16.7% | 0.0496 | 0.1222 | -0.0228 | Larger label weight hurts WER without recovering control |
| `cv500_obj_label_ramp` | 16.7% | 0.0442 | 0.0832 | -0.0126 | Preserves MOS closest to raw `cv500`, but stays conservative |
| `cv500_obj_recon_half_label2` | 16.7% | 0.0500 | 0.1146 | -0.0226 | Lower reconstruction weight is still not enough to recover recall or novelty |

### Collapse behavior

The collapse summary confirms that the new objective variants did not escape the
same underlying failure mode.

| Condition | Style collapse to neutral | Identity collapse to baseline | Mixed collapse | Takeaway |
|-----------|---------------------------|-------------------------------|----------------|----------|
| CommonVoice `cv500` init | 54 | 72 | 34 | Original conservative collapse pattern |
| `cv500_ft_short_low_lr` | 55 | 44 | 22 | Best CommonVoice finetune ablation identity-recovery reference |
| `cv500_obj_label2` | 55 | 53 | 28 | Some identity recovery over raw `cv500`, still worse than best CommonVoice finetune ablation recipe |
| `cv500_obj_label4` | 55 | 55 | 32 | Stronger label weight worsens WER and does not reduce style collapse |
| `cv500_obj_label_ramp` | 55 | 57 | 35 | Label ramp behaves almost like the original conservative collapse |
| `cv500_obj_recon_half_label2` | 55 | 56 | 32 | Reduced reconstruction pressure still does not recover control |

### Interpretation

CommonVoice objective ablation sharpens the CommonVoice story again:

1. **Simple scalar objective reweighting is not enough.** None of the four variants improves recall beyond `16.7%`.
2. **Objective tuning helps less than the best CommonVoice finetune recipe.** Even the best new variant (`cv500_obj_label2`) trails `cv500_ft_short_low_lr` on novelty and identity collapse.
3. **The label-ramp result is particularly informative.** It preserves MOS close to the original `cv500` run, but only by staying near the same conservative identity/style collapse basin.
4. **The CommonVoice failure is now unlikely to be just a basic weight-balance issue.** CommonVoice finetune ablation ruled out simple finetune-policy changes; CommonVoice objective ablation rules out simple scalar reweighting of the existing losses.
5. **The next objective work needs to be richer than rebalancing the current losses.** The more promising directions are now partial-label pretraining, teacher or latent anchoring, curriculum objectives, or larger-scale data with stronger supervision.

### Implication

CommonVoice objective ablation changes how we should describe the CommonVoice line in the paper:

- We should **not** claim that simple objective tuning already solves the CommonVoice collapse.
- We **can** say that the failure has been narrowed substantially: it survives both gentler finetuning (Finding 13) and simple loss reweighting (Finding 14).
- The next serious CommonVoice experiments should focus on richer supervision or larger-scale training, not more scalar weight sweeps.

That is a stronger and more honest research position than we had after Finding
13 alone.

---

## Finding 15: Richer Teacher/Anchor Objectives Still Do Not Recover Recall After CommonVoice Pretraining

### Methodology

CommonVoice rich-objective ablation asked the next sharper follow-up question after Finding 14:

**Was the CommonVoice `cv500` collapse mainly a problem of missing structural
supervision during combined finetuning, such that teacher guidance on the style
latent dims or anchoring of the non-style dims could preserve the CommonVoice
prior while restoring control?**

To test that, we kept the data, the validation corpus, and the four-metric
stack fixed again. We still started from the same CommonVoice-pretrained init
checkpoint and the same combined labeled embeddings. The only change was that
combined finetuning now included richer auxiliary losses that explicitly target
different parts of the latent space:

- **style dims (`0-8`)**: the labeled emotion/style subspace
- **free dims (`9-14`)**: the unlabeled remainder of the latent code

We kept these pieces unchanged:

- CommonVoice init checkpoint: `embeddings/openvoice_vae_commonvoice_cv500.pt`
- combined finetuning embeddings: `embeddings/openvoice_combined_emb.pt`
- 11-speaker validation corpus format
- same evaluation stack from Findings 10-14
- same best CommonVoice finetune ablation training schedule foundation: `1000` epochs at `3e-7`

We compared three references:

1. **combined** — the current paper checkpoint
2. **CommonVoice `cv500` init** — the original conservative CommonVoice result
3. **`cv500_ft_short_low_lr`** — the best partial-recovery finetune recipe from Finding 13

against three new rich-objective variants:

1. **`cv500_rich_teacher_style`** — style-teacher distillation on dims `0-8` using the combined checkpoint as the teacher
2. **`cv500_rich_free_anchor`** — free-dim anchoring on dims `9-14` using the CommonVoice-initialized checkpoint as the frozen anchor
3. **`cv500_rich_teacher_plus_anchor`** — both losses together

### Results

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Takeaway |
|-----------|--------|--------------------------|----------|----------------|----------|
| combined | 25.8% | 0.2599 | 0.2353 | -0.0792 | Still the only condition with both meaningful control and meaningful identity shift |
| CommonVoice `cv500` init | 16.7% | 0.0369 | 0.0844 | -0.0123 | Conservative collapse reference point |
| `cv500_ft_short_low_lr` | 16.7% | 0.0692 | 0.0791 | -0.0441 | Best CommonVoice partial-recovery reference from Finding 13 |
| `cv500_rich_teacher_style` | 16.7% | 0.0574 | 0.0851 | -0.0200 | Style teacher loss preserves quality fairly well, but still does not recover control |
| `cv500_rich_free_anchor` | 16.7% | 0.0646 | 0.0724 | -0.0079 | Best CommonVoice rich-objective ablation variant for WER/MOS, but still below `cv500_ft_short_low_lr` on novelty |
| `cv500_rich_teacher_plus_anchor` | 16.7% | 0.0566 | 0.0809 | -0.0161 | Combining both losses does not create a better tradeoff than the simpler variants |

### Collapse behavior

The collapse summary shows that richer supervision as implemented here still
does not escape the same conservative failure shape.

| Condition | Style collapse to neutral | Identity collapse to baseline | Mixed collapse | Takeaway |
|-----------|---------------------------|-------------------------------|----------------|----------|
| CommonVoice `cv500` init | 54 | 72 | 34 | Original conservative collapse pattern |
| `cv500_ft_short_low_lr` | 55 | 44 | 22 | Best CommonVoice finetune ablation identity-recovery reference |
| `cv500_rich_teacher_style` | 55 | 51 | 28 | Teacher guidance recovers some identity shift over raw `cv500`, but still trails the best CommonVoice finetune ablation recipe |
| `cv500_rich_free_anchor` | 55 | 50 | 25 | Best CommonVoice rich-objective ablation stability tradeoff, but style collapse remains unchanged and identity collapse still exceeds the best CommonVoice finetune ablation result |
| `cv500_rich_teacher_plus_anchor` | 54 | 53 | 27 | Combining the two losses does not materially reduce the collapse counts |

### Interpretation

CommonVoice rich-objective ablation narrows the CommonVoice story again:

1. **This first richer-objective family still does not recover recall.** All three variants stay flat at `16.7%`.
2. **Latent teacher/anchor supervision helps stability more than controllability.** The best CommonVoice rich-objective ablation variant (`cv500_rich_free_anchor`) improves WER and MOS over `cv500_ft_short_low_lr`, but not novelty enough to beat it, and not recall at all.
3. **The style-teacher loss did not rescue the style subspace.** If teacher guidance on the style dims alone were enough, `cv500_rich_teacher_style` should have improved recall or reduced neutral collapse. It did neither.
4. **The combined teacher+anchor objective also stayed conservative.** Adding both losses together still leaves the model in roughly the same basin of flat recall and high identity collapse.
5. **The remaining bottleneck likely sits earlier than this finetuning stage.** After CommonVoice finetune ablation (gentler finetuning), CommonVoice objective ablation (scalar reweighting), and CommonVoice rich-objective ablation (teacher/anchor supervision), the strongest remaining hypotheses are:
   - the CommonVoice stage needs richer supervision or partial labels itself
   - the current reconstruction-only pretraining objective over-shapes the latent space before labeled finetuning begins
   - or a larger-scale setup is needed, but only once a stronger objective survives on this validation-scale corpus

### Implication

CommonVoice rich-objective ablation changes the paper position again, in a useful way:

- We should **not** claim that richer finetune-time supervision already solves the CommonVoice collapse.
- We **can** say that the failure has been narrowed further: it survives gentler finetuning (Finding 13), scalar loss reweighting (Finding 14), and the first richer teacher/anchor objective family (Finding 15).
- The next CommonVoice experiments should focus on richer pretraining supervision, partial-label or pseudo-label objectives on CommonVoice itself, or larger-scale training once a better objective is identified.

That is a stronger and more defensible research position than we had after
Finding 14 alone.

---

## Finding 16: Validation-Scale Weak-Label CommonVoice Pretraining Still Does Not Recover Recall

### Methodology

CommonVoice partial-label pretraining moved the supervision earlier in the pipeline.

After Finding 15, the strongest remaining hypothesis was:

**Maybe the CommonVoice collapse is already being baked in during the
reconstruction-only CommonVoice stage, so the right intervention is to add weak
labels during CommonVoice pretraining itself rather than only during combined
finetuning.**

To test that, we kept the validation-scale `cv500` setup, the combined
finetuning data, the 11-speaker evaluation corpus, and the same four-metric
stack fixed. The changed variable was the CommonVoice pretraining objective.

We started from the same local CommonVoice artifact and measured how much real
metadata support it actually had:

- age known on `193/1202` clips
- gender known on `184/1202`
- accent known on `190/1202`

That meant the metadata-only condition was genuinely weak supervision, not a
dense relabeling of the corpus.

We also generated pseudo-style labels from audio using a frozen emotion model.
At the reporting/training threshold `0.60`, the accepted pseudo-style counts
were:

- `neutral = 640`
- `sad = 336`
- `happy = 51`
- `disgust = 46`
- `anger = 11`
- `fear = 5`

So the pseudo-style supervision was also weak in two ways:

1. it was teacher-generated rather than ground truth
2. it was heavily imbalanced toward the conservative classes

We therefore used inverse-frequency balancing for both metadata classes and
pseudo-style rows during CommonVoice pretraining.

We compared four reference points:

1. **combined** — the main paper checkpoint
2. **CommonVoice `cv500` init** — the raw reconstruction-only CommonVoice result
3. **`cv500_ft_short_low_lr`** — the best CommonVoice finetune-only recovery recipe
4. **`cv500_rich_free_anchor`** — the strongest CommonVoice rich-objective ablation richer-objective reference

against three new CommonVoice pretraining variants:

1. **`cv500_pl_meta`** — metadata-only weak supervision on the free dims
2. **`cv500_pl_pseudo_style`** — pseudo-style-only weak supervision on the style dims
3. **`cv500_pl_meta_plus_pseudo`** — both weak signals together

### Results

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Takeaway |
|-----------|--------|--------------------------|----------|----------------|----------|
| combined | 25.8% | 0.2599 | 0.2353 | -0.0792 | Still the only condition with both meaningful control and meaningful speaker shift |
| CommonVoice `cv500` init | 16.7% | 0.0369 | 0.0844 | -0.0123 | Conservative CommonVoice collapse reference |
| `cv500_ft_short_low_lr` | 16.7% | 0.0692 | 0.0791 | -0.0441 | Best CommonVoice finetune ablation partial-recovery reference |
| `cv500_rich_free_anchor` | 16.7% | 0.0646 | 0.0724 | -0.0079 | Best CommonVoice rich-objective ablation richer-objective reference |
| `cv500_pl_meta` | 16.7% | 0.0570 | 0.0918 | -0.0505 | Metadata-only supervision nudges novelty, but not enough to beat the stronger earlier CommonVoice variants |
| `cv500_pl_pseudo_style` | 16.7% | 0.0190 | 0.0263 | -0.0229 | Pseudo-style supervision greatly improves intelligibility, but collapses novelty and identity shift |
| `cv500_pl_meta_plus_pseudo` | 16.7% | 0.0181 | 0.0285 | -0.0148 | Hybrid weak supervision mostly follows the pseudo-style conservative basin |

### Collapse behavior

The collapse summary is especially informative here because it shows that the
weak-label variants do not fail in the same way.

| Condition | Style collapse to neutral | Identity collapse to baseline | Mixed collapse | Takeaway |
|-----------|---------------------------|-------------------------------|----------------|----------|
| CommonVoice `cv500` init | 54 | 72 | 34 | Original conservative collapse reference |
| `cv500_ft_short_low_lr` | 55 | 44 | 22 | Best CommonVoice finetune ablation identity-recovery reference |
| `cv500_rich_free_anchor` | 55 | 50 | 25 | Best CommonVoice rich-objective ablation richer-objective reference |
| `cv500_pl_meta` | 54 | 59 | 33 | Metadata-only supervision reduces identity collapse somewhat vs raw `cv500`, but not enough to beat the best earlier CommonVoice variants |
| `cv500_pl_pseudo_style` | 55 | 85 | 47 | Pseudo-style supervision preserves content, but collapses identity much harder |
| `cv500_pl_meta_plus_pseudo` | 55 | 90 | 49 | Adding metadata does not rescue the pseudo-style collapse pattern |

### Interpretation

CommonVoice partial-label pretraining narrows the CommonVoice story again:

1. **Weak labels during CommonVoice pretraining still do not recover recall.** All three new variants stay flat at `16.7%`.
2. **Metadata-only supervision helps novelty a little, but not enough.** `cv500_pl_meta` improves novelty over raw `cv500` (`0.0570` vs. `0.0369`), but still trails both the best CommonVoice finetune ablation and best CommonVoice rich-objective ablation references.
3. **Pseudo-style supervision helps intelligibility much more than control.** Both pseudo-style variants drive WER down sharply (`0.0263` and `0.0285`), but they do so by collapsing toward baseline identity: identity-collapse counts rise to `85-90`.
4. **The current pseudo labels likely reinforce the conservative basin instead of escaping it.** The label distribution is dominated by `neutral` and `sad`, and even with balancing the resulting models remain highly conservative on speaker shift.
5. **The remaining CommonVoice bottleneck now looks deeper than this first weak-label family.** After CommonVoice finetune ablation (gentler finetuning), CommonVoice objective ablation (scalar reweighting), CommonVoice rich-objective ablation (teacher/anchor finetuning), and CommonVoice partial-label pretraining (weak-label pretraining), we still have no recall recovery and no CommonVoice variant that matches the combined model's tradeoff.

### Implication

CommonVoice partial-label pretraining changes the paper position again, in a useful way:

- We should **not** claim that weak metadata or pseudo-label supervision on this validation-scale CommonVoice setup already solves the collapse.
- We **can** say that the CommonVoice failure has now survived four increasingly informed fixes: gentler finetuning (Finding 13), scalar reweighting (Finding 14), richer finetune-time supervision (Finding 15), and weak-label pretraining (Finding 16).
- The next CommonVoice experiments should focus on better pseudo-label quality, prototype- or teacher-space targets during CommonVoice pretraining, stronger curricula, or larger-scale runs once a better supervision recipe survives on the validation corpus.

That is a stronger and more honest research position than we had after Finding
15 alone.

---

## Finding 17: Mixed-Data Schedule Changes Improve WER More Than They Improve Control

### Methodology

The mixed-data pseudolabel mix experiment answered Joe's April 30 question
directly: what happens if we stop treating CommonVoice as a separate pretraining
stage and instead train one mixed artifact from:

- pseudo-labeled CommonVoice
- CREMA-D
- Expresso

We fixed the mixed artifact itself first:

- `500` CommonVoice speakers with one clip per speaker
- `546` CREMA-D rows
- `279` Expresso rows
- `1,325` total rows
- `1,287/1,325` labeled rows

The accepted CommonVoice pseudo-label mix in that artifact remained heavily
skewed:

- `neutral = 207`
- `sad = 170`
- `happy = 37`
- `disgust = 34`
- `anger = 10`
- `fear = 4`

We then trained three schedule variants from the same mixed artifact:

1. **`mixed_static_balanced`** — equal dataset mass every epoch
2. **`mixed_cv_warmup`** — CommonVoice-heavy early, then balanced
3. **`mixed_labeled_finish`** — balanced early, then CREMA-D/Expresso-heavy late

Everything else stayed fixed:

- same 11-speaker evaluation corpus
- same four-metric stack: emotion, novelty, WER, MOS
- same inference style strength (`5.0`) and noise level (`0.0`)

### Results

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Identity collapse | Takeaway |
|-----------|--------|--------------------------|----------|----------------|-------------------|----------|
| `combined` | 25.8% | 0.2599 | 0.2353 | -0.0792 | 7 | Still the only condition with clearly stronger control and novelty together |
| `cv500_ft_short_low_lr` | 16.7% | 0.0692 | 0.0791 | -0.0441 | 44 | Best earlier CommonVoice finetune reference |
| `cv500_rich_free_anchor` | 16.7% | 0.0646 | 0.0724 | -0.0079 | 50 | Best earlier richer-objective CommonVoice reference |
| `cv500_pl_meta` | 16.7% | 0.0570 | 0.0918 | -0.0505 | 59 | Best earlier weak-label metadata-only reference |
| `mixed_static_balanced` | 16.7% | 0.0865 | 0.0825 | -0.1316 | 58 | Best novelty of the mixed schedules, but no recall gain and worse naturalness |
| `mixed_cv_warmup` | 16.7% | 0.0738 | 0.0700 | -0.1341 | 60 | Better WER than static, but worse novelty and still no recall gain |
| `mixed_labeled_finish` | 16.7% | 0.0828 | 0.0606 | -0.1425 | 64 | Best WER of the mixed schedules, but still no recall gain and the worst identity collapse of the three |

### Collapse behavior

The schedule comparison is useful because it shows that the branch is not
failing on content collapse. It is failing on style and identity collapse.

| Condition | Style collapse to neutral | Identity collapse to baseline | Mixed collapse | Files with any collapse |
|-----------|---------------------------|-------------------------------|----------------|-------------------------|
| `mixed_static_balanced` | 55 | 58 | 47 | 66 |
| `mixed_cv_warmup` | 54 | 60 | 49 | 65 |
| `mixed_labeled_finish` | 55 | 64 | 49 | 70 |

### Interpretation

The mixed-data pseudolabel mix result changes the story in an important but
measured way:

1. **Changing the dataset schedule alone does not recover recall.** All three mixed-data schedules stay flat at `16.7%`.
2. **Mixed-data training helps WER more than it helps control.** `mixed_labeled_finish` reaches the best WER of the three (`0.0606`), and `mixed_cv_warmup` also beats the best earlier CommonVoice finetune reference on WER (`0.0700` vs. `0.0791`).
3. **Mixed-data training helps novelty somewhat, but not nearly enough.** `mixed_static_balanced` (`0.0865`) and `mixed_labeled_finish` (`0.0828`) beat the best earlier CommonVoice references on novelty, but remain far below the `combined` model (`0.2599`).
4. **The three schedules stay in the same conservative basin.** Style-collapse counts remain `54-55`, identity-collapse counts remain `58-64`, and none of the schedules shows a credible escape from the neutral / baseline-identity pattern.
5. **The schedule question is now narrower.** `mixed_cv_warmup` is not better than the other two overall, and `mixed_labeled_finish` mainly buys intelligibility rather than control. The next mixed-data gains likely require better pseudo-label quality, per-class filtering/caps, or stronger labeled-data protection rather than more schedule variants alone.

### Implication

The first mixed-data run does **not** justify claiming that simple
CommonVoice+CREMA-D+Expresso mixing solves the controllability gap.

It **does** justify a more precise claim:

- mixing data helps the CommonVoice line keep or improve its intelligibility story
- schedule choice modestly changes novelty vs. WER tradeoffs
- but the core controllability bottleneck survives the first real combined-data run

That is still a useful paper result because it tells us that:

- the April 30 data-mixing idea was worth testing
- it did not fail trivially
- and the remaining blocker is likely supervision quality or representation,
  not just the absence of a mixed-data schedule

---

## Finding 18: Better Mixed-Data Pseudo-Label Filtering Produces Only a Narrow Recall Gain

### Methodology

The mixed-data pseudolabel quality follow-up kept the same overall mixed-data
goal as Finding 17, but tightened the CommonVoice side of the training line in
two ways:

1. **Better pseudo-label filtering and balancing inside the artifact builder**
2. **Stronger explicit labeled-data protection during mixed training**

The new artifact still preserved Joe's speaker-breadth-first heuristic:

- `500` CommonVoice speakers
- `1` clip per speaker
- `546` CREMA-D rows
- `279` Expresso rows
- `1,325` total rows

But the CommonVoice pseudo-label acceptance logic became much stricter. The
improved artifact:

- lowered labeled CommonVoice rows from `462` to `300`
- kept `200` CommonVoice rows as unlabeled fallback rows
- enforced per-style thresholds:
  - `neutral=0.995`
  - `sad=0.98`
  - `happy=0.92`
  - `disgust=0.92`
  - `anger=0.90`
  - `fear=0.90`
- enforced CommonVoice pseudo-style caps:
  - `neutral=120`
  - `sad=110`
  - `happy=60`
  - `disgust=50`
  - `anger=20`
  - `fear=20`

The resulting selected CommonVoice pseudo-style mix became:

- `neutral = 120`
- `sad = 110`
- `happy = 32`
- `disgust = 29`
- `anger = 5`
- `fear = 4`

We then trained three new mixed-data conditions:

1. **`mixed_quality_static_balanced`** — improved artifact, equal dataset mass
2. **`mixed_quality_labeled_finish`** — improved artifact, labeled-heavy finish
3. **`mixed_quality_labeled_guarded`** — improved artifact, stronger labeled-data protection with end masses `CommonVoice=0.10`, `CREMA-D=0.45`, `Expresso=0.45`

Everything else stayed fixed:

- same 11-speaker evaluation corpus
- same four-metric stack: emotion, novelty, WER, MOS
- same inference style strength (`5.0`) and noise level (`0.0`)

### Results

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Identity collapse | Takeaway |
|-----------|--------|--------------------------|----------|----------------|-------------------|----------|
| `combined` | 25.8% | 0.2599 | 0.2353 | -0.0792 | 7 | Still the only condition with clearly stronger control and novelty together |
| `mixed_static_balanced` | 16.7% | 0.0865 | 0.0825 | -0.1316 | 58 | Best novelty of the original mixed schedules |
| `mixed_labeled_finish` | 16.7% | 0.0828 | 0.0606 | -0.1425 | 64 | Best WER of the original mixed schedules |
| `mixed_quality_static_balanced` | 16.7% | 0.0770 | 0.0911 | -0.1130 | 64 | Cleaner pseudo-label mix, but no control gain |
| `mixed_quality_labeled_finish` | 16.7% | 0.0763 | 0.0649 | -0.1232 | 65 | Keeps most of the earlier WER story, but no recall gain |
| `mixed_quality_labeled_guarded` | 18.2% | 0.0764 | 0.0978 | -0.1234 | 69 | First mixed-data recall bump, but not a clean overall win |

### Collapse behavior

The quality follow-up is useful because it finally moves recall, but it still
does so inside the same broad collapse regime:

| Condition | Style collapse to neutral | Identity collapse to baseline | Mixed collapse | Files with any collapse |
|-----------|---------------------------|-------------------------------|----------------|-------------------------|
| `mixed_quality_static_balanced` | 54 | 64 | 47 | 68 |
| `mixed_quality_labeled_finish` | 55 | 65 | 48 | 69 |
| `mixed_quality_labeled_guarded` | 53 | 69 | 48 | 69 |

### Interpretation

The mixed-data pseudolabel quality follow-up sharpens Finding 17 rather than
replacing it:

1. **Better pseudo-label filtering plus stronger labeled-data protection can move recall a little.** `mixed_quality_labeled_guarded` becomes the first mixed-data condition to improve recall above `16.7%`, reaching `18.2%`.
2. **The gain is narrow and costly.** The best new recall condition gives back WER versus `mixed_labeled_finish` (`0.0978` vs. `0.0606`) and gives back novelty versus `mixed_static_balanced` (`0.0764` vs. `0.0865`).
3. **Better class balance is not enough by itself to escape the conservative basin.** Identity collapse stays high (`64-69`), style collapse stays high (`53-55`), and none of the new conditions approaches the `combined` model's novelty or recall.
4. **The branch still produces a useful checkpoint choice.** `mixed_quality_labeled_guarded` is the most control-capable mixed-data checkpoint so far, even though it is not a clean overall winner.
5. **The mixed-data bottleneck is now more specific.** The next improvement likely needs a stronger pseudo-label teacher, better class-balanced pseudo-label acceptance, or stronger style supervision, not just stricter filtering and dataset-mass protection.

### Implication

This branch gives us a more precise mixed-data claim:

- the first mixed-data schedules improved WER more than recall (Finding 17)
- the first pseudo-label-quality follow-up can recover a **small** amount of recall
- but the mixed-data line still does not achieve a convincing control/intelligibility balance

That is useful for the paper because it shows the mixed-data direction is not
dead, but also not solved. We now know that:

- mixed-data schedule choice alone is too weak
- stricter pseudo-label filtering and labeled-data protection help only a little
- the next mixed-data gains probably require stronger supervision quality, not
  just more careful bookkeeping

---

## Finding 19: Style Strength Above `5.0` Is Useful for Some Non-Trump Styles, But Not as a New Global Default

### Methodology

This sweep tested Joe's qualitative claim directly: can style strengths above
`5.0` still be useful on non-Trump examples?

We fixed:

- checkpoint: `mixed_quality_labeled_guarded`
- panel: four checked-in non-Trump speakers
  - `female_1_cremad_1002`
  - `female_2_cremad_1012`
  - `male_1_cremad_1003`
  - `male_2_cremad_1051`
- noise level: `0.0`
- seed: `42`
- generation mode: `--all-styles`

We swept:

- `5.0`
- `7.5`
- `10.0`
- `12.5`

and scored each strength with the same four metrics:

- emotion recall / emo_sim
- novelty gain
- WER
- MOS

### Results

| Strength | Recall | Mean emo_sim | Mean novelty gain | Mean WER | Mean MOS delta | Takeaway |
|----------|--------|--------------|-------------------|----------|----------------|----------|
| `5.0` | 20.8% | 0.9874 | 0.0789 | 0.1472 | -0.1312 | Safest overall setting on this panel |
| `7.5` | 16.7% | 0.9773 | 0.1246 | 0.1681 | -0.1701 | Best stronger-than-default compromise |
| `10.0` | 16.7% | 0.9726 | 0.1570 | 0.2028 | -0.2287 | Higher novelty, noticeably worse overall quality |
| `12.5` | 16.7% | 0.9728 | 0.1761 | 0.2444 | -0.2119 | Highest novelty, but clearly not a balanced default |

Focus-style behavior:

| Strength | Style | Recall | emo_sim | Novelty gain | WER | MOS delta |
|----------|-------|--------|---------|--------------|-----|-----------|
| `5.0` | `confused` | - | 0.9796 | 0.2871 | 0.1125 | -0.5253 |
| `5.0` | `happy` | 0.0000 | 0.9951 | -0.0325 | 0.0625 | -0.0182 |
| `5.0` | `sad` | 0.2500 | 0.9845 | 0.0338 | 0.0625 | -0.0166 |
| `5.0` | `whisper` | - | 0.9617 | 0.3651 | 0.3000 | -0.6416 |
| `7.5` | `confused` | - | 0.9625 | 0.4549 | 0.1125 | -0.6577 |
| `7.5` | `happy` | 0.0000 | 0.9897 | -0.0239 | 0.1875 | -0.0211 |
| `7.5` | `sad` | 0.0000 | 0.9887 | 0.0367 | 0.0625 | -0.0182 |
| `7.5` | `whisper` | - | 0.8953 | 0.5123 | 0.3000 | -0.7248 |
| `10.0` | `confused` | - | 0.9691 | 0.5341 | 0.1125 | -0.9787 |
| `10.0` | `happy` | 0.0000 | 0.9902 | -0.0326 | 0.2500 | -0.0233 |
| `10.0` | `sad` | 0.0000 | 0.9857 | 0.0437 | 0.0625 | -0.0179 |
| `10.0` | `whisper` | - | 0.8750 | 0.6042 | 0.3000 | -0.7918 |
| `12.5` | `confused` | - | 0.9600 | 0.5694 | 0.1125 | -1.0095 |
| `12.5` | `happy` | 0.0000 | 0.9895 | -0.0332 | 0.2500 | -0.0208 |
| `12.5` | `sad` | 0.0000 | 0.9866 | 0.0649 | 0.0625 | -0.0175 |
| `12.5` | `whisper` | - | 0.8867 | 0.6521 | 0.3000 | -0.7141 |

### Interpretation

This sweep answers Joe's question more precisely:

1. **`5.0` is not a hard ceiling.** Higher strengths clearly increase novelty for `whisper` and `confused` on this non-Trump panel.
2. **Higher strength is a tradeoff knob, not a free upgrade.** As strength rises, novelty improves, but overall recall, WER, and MOS all worsen.
3. **`7.5` is the best stronger-than-default compromise on this panel.** It gives a substantial novelty boost over `5.0` (`0.1246` vs. `0.0789` overall, `0.5123` vs. `0.3651` for `whisper`) without the steeper WER/MOS penalties seen by `10.0` and `12.5`.
4. **`whisper` benefits most from higher strength.** Its novelty rises monotonically from `0.3651` to `0.6521`, while WER stays flat at `0.3000`. The cost is naturalness, not content preservation.
5. **Not all styles benefit.** `happy` stays negative on novelty gain at every strength, and `sad` only has non-zero recall at `5.0`.

### Implication

The repo guidance should change, but only narrowly:

- keep `5.0` as the safest general default
- treat `7.5` as a documented stronger option for styles like `whisper` and
  `confused` on non-Trump inputs
- treat `10.0-12.5` as higher-risk, higher-novelty settings better suited to
  demos or targeted style-specific use, not new defaults

---

## Finding 20: The First Mixed-Data Pseudo-Label Teacher Family Improves the Tradeoff Slightly, But Still Does Not Break the Recall Ceiling

### Methodology

This follow-up kept the same mixed-data framing as Findings 17 and 18:

- same OpenVoice backend
- same 15-dim controllable VAE
- same `11`-speaker evaluation corpus
- same four-metric stack: emotion recall / emo_sim, novelty, WER, MOS
- same inference settings: `style_strength=5.0`, `noise_level=0.0`, `seed=42`

The changed variable was the **teacher-focused CommonVoice pseudo-label path**
that fed the mixed artifact:

- `scripts/annotate_commonvoice_pseudolabels.py` preserved top-k teacher
  labels/scores and teacher metadata
- `scripts/filter_commonvoice_pseudolabels.py` selected a balanced-target
  pseudo-label pool from the scored CommonVoice artifact
- `scripts/build_mixed_training_set.py` rebuilt
  `embeddings/openvoice_mixed_teacher_base.pt` from that score -> filter path

The rebuilt mixed teacher artifact kept:

- `500` CommonVoice speakers / rows
- `546` CREMA-D rows
- `279` Expresso rows
- `1,325` total rows
- `1,057` labeled rows overall

The selected CommonVoice pseudo-style counts in the real mixed artifact were:

- `anger = 5`
- `disgust = 30`
- `fear = 4`
- `happy = 35`
- `neutral = 78`
- `sad = 80`

From this artifact we trained three new teacher-family conditions:

1. **`mixed_teacher_threshold_balanced`** — teacher-filtered artifact, static balanced dataset masses
2. **`mixed_teacher_labeled_finish`** — teacher-filtered artifact, labeled-heavy finish
3. **`mixed_teacher_labeled_guarded`** — teacher-filtered artifact, strongest labeled-data protection with end masses `CommonVoice=0.10`, `CREMA-D=0.45`, `Expresso=0.45`

### Results

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Identity collapse | Takeaway |
|-----------|--------|--------------------------|----------|----------------|-------------------|----------|
| `combined` | 25.8% | 0.2599 | 0.2353 | -0.0792 | 7 | Still the only condition with clearly stronger control and novelty together |
| `mixed_quality_labeled_guarded` | 18.2% | 0.0764 | 0.0978 | -0.1234 | 69 | Previous best mixed-data recall condition |
| `mixed_teacher_threshold_balanced` | 18.2% | 0.0785 | 0.0829 | -0.1012 | 62 | Matches best mixed-data recall while improving every other measured axis vs. `mixed_quality_labeled_guarded` |
| `mixed_teacher_labeled_finish` | 16.7% | 0.0763 | 0.0727 | -0.1150 | 68 | Better WER than the prior best recall condition, but recall falls back to the conservative basin |
| `mixed_teacher_labeled_guarded` | 18.2% | 0.0760 | 0.1095 | -0.1173 | 70 | Preserves the recall bump, but gives back too much WER and identity stability |

### Collapse behavior

The first teacher family still lives in the same broad collapse regime as the
earlier mixed-data branches:

| Condition | Style collapse to neutral | Identity collapse to baseline | Mixed collapse | Files with any collapse |
|-----------|---------------------------|-------------------------------|----------------|-------------------------|
| `mixed_teacher_threshold_balanced` | 53 | 62 | 49 | 66 |
| `mixed_teacher_labeled_finish` | 54 | 68 | 51 | 68 |
| `mixed_teacher_labeled_guarded` | 54 | 70 | 51 | 70 |

### Interpretation

This branch is a useful refinement, but not a breakthrough:

1. **The first teacher-family run does not move recall above `18.2%`.** The best teacher conditions only tie the mixed-data pseudo-label quality branch on recall.
2. **`mixed_teacher_threshold_balanced` is still a real improvement.** It matches `mixed_quality_labeled_guarded` on recall while improving novelty (`0.0785` vs. `0.0764`), WER (`0.0829` vs. `0.0978`), MOS delta (`-0.1012` vs. `-0.1234`), and identity collapse (`62` vs. `69`).
3. **The guarded teacher schedule is not the answer.** `mixed_teacher_labeled_guarded` preserves the recall bump but ends up with the worst WER and identity collapse inside the teacher family.
4. **The current teacher remains the likely bottleneck.** The branch-specific rescoring already showed that `emotion2vec_plus_large` reproduces the earlier neutral/sad-heavy pseudo-label distribution almost exactly, and the evaluation confirms that cleaner use of the same teacher only helps a little.
5. **The next decisive mixed-data experiment is now narrower.** We should compare an alternative pseudo-label teacher or a multi-teacher / agreement rule rather than repeating more schedule variants on the same single-teacher family.

### Implication

Finding 20 narrows the mixed-data story again:

- schedule choice alone was too weak (Finding 17)
- stricter pseudo-label filtering and labeled-data protection moved recall only a little (Finding 18)
- the first cleaner teacher-family run improves the tradeoff slightly, but still does not break the `18.2%` recall ceiling

That is useful because it points the next branch much more clearly at:

- a stronger teacher
- teacher-agreement acceptance
- or a richer supervision target

instead of just another iteration of the same single-teacher filtering logic.

---

## Finding 21: A Softer Same-Teacher Agreement Rule Raises Novelty Slightly, But Still Loses the Overall Tradeoff

### Methodology

After Finding 20, the narrow next question was whether a **softer agreement
rule** could clean up CommonVoice pseudo-label acceptance without starving the
rare classes.

This follow-up kept the same teacher model:

- `iic/emotion2vec_plus_large`

But changed the scoring/filtering path:

- `scripts/annotate_commonvoice_pseudolabels.py` now scores:
  - the full clip
  - a center-crop secondary view
- it preserves:
  - top-k teacher labels/scores
  - mapped per-style score maps
  - row-level agreement metadata
- `scripts/filter_commonvoice_pseudolabels.py` now supports:
  - `--secondary-agreement-mode exact`
  - `--secondary-agreement-mode mapped_score`

The tested agreement condition used:

- `mapped_score` agreement mode
- `0.15` per-style secondary support thresholds
- the same balanced-target pseudo-style policy as the earlier teacher branch

This produced:

- `embeddings/openvoice_commonvoice_cv500_pseudo_agreement_mapped015_filtered.pt`
- `embeddings/openvoice_mixed_teacher_mapped015_base.pt`
- `embeddings/openvoice_vae_mixed_teacher_mapped015_balanced.pt`

The final evaluation condition was:

1. **`mixed_teacher_mapped015_balanced`** — mapped-score agreement, static balanced dataset masses

### Results

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Identity collapse | Takeaway |
|-----------|--------|--------------------------|----------|----------------|-------------------|----------|
| `mixed_teacher_threshold_balanced` | 18.2% | 0.0785 | 0.0829 | -0.1012 | 62 | Current best teacher-family reference |
| `mixed_teacher_mapped015_balanced` | 16.7% | 0.0818 | 0.1090 | -0.1115 | 63 | Slightly more novel, but loses the broader tradeoff |

### Interpretation

1. **The mapped-score agreement rule does not beat the teacher-family reference.** It falls back to `16.7%` recall, so it gives up the main thing the best teacher-family condition was preserving.
2. **Its only clear win is novelty, and that win is small.** Novelty rises from `0.0785` to `0.0818`, but not enough to offset worse WER and MOS.
3. **This makes the current bottleneck clearer.** Reusing the same teacher more carefully, even with a softer agreement rule, is still not enough.
4. **The next useful branch should change the teacher, not just the agreement heuristic.** A genuinely different pseudo-label teacher or a richer multi-teacher rule is now more justified than more polishing on the same single-teacher path.

### Implication

Finding 21 strengthens the mixed-data teacher story:

- the first cleaner teacher-family run improved the tradeoff a little (Finding 20)
- a softer agreement rule on the same teacher does **not** carry that further

So the next meaningful mixed-data teacher gain now likely requires:

- a different pseudo-label teacher
- a multi-teacher agreement rule
- or stronger class-balanced supervision that is not tied to the same
  neutral/sad-heavy teacher distribution

---

## Finding 22: A Combined-VAE Latent Prototype Teacher Improves Novelty and Coverage, But Still Does Not Break the Recall Ceiling

### Methodology

After Finding 21, the next clean test was to change the pseudo-label teacher
instead of polishing `emotion2vec_plus_large` filtering.

The alternate teacher uses the existing combined controllable VAE as a
prototype model:

- encode the labeled combined CREMA-D + Expresso artifact with
  `embeddings/openvoice_vae_combined.pt`
- build one prototype per style from the VAE style dims `0-8`
- encode the CommonVoice `cv500` rows with the same VAE encoder
- assign pseudo-style labels by similarity to the style prototypes
- filter with the same balanced-target path used by the earlier teacher branch

This produced:

- `scripts/annotate_commonvoice_latent_prototypes.py`
- `embeddings/openvoice_commonvoice_cv500_pseudo_prototype.pt`
- `embeddings/openvoice_commonvoice_cv500_pseudo_prototype_filtered.pt`
- `embeddings/openvoice_mixed_teacher_prototype_base.pt`
- `embeddings/openvoice_vae_mixed_teacher_prototype_balanced.pt`

The important difference from Findings 20-21 is that this teacher is not an
emotion classifier. It is a **model-internal latent geometry teacher**: it asks
which combined-VAE style prototype each CommonVoice row is nearest to.

### Pseudo-label coverage

The prototype teacher produced broad all-style coverage after filtering:

| Style | Selected before mixing | Selected in mixed artifact |
|-------|------------------------|----------------------------|
| anger | 40 | 25 |
| confused | 40 | 25 |
| disgust | 40 | 33 |
| enunciated | 40 | 29 |
| fear | 36 | 21 |
| happy | 40 | 27 |
| neutral | 40 | 23 |
| sad | 40 | 34 |
| whisper | 12 | 10 |

This is a major coverage improvement over the emotion2vec teacher, which was
restricted to canonical emotion classes and stayed heavily neutral/sad.

### Results

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Identity collapse | Mixed collapse | Takeaway |
|-----------|--------|--------------------------|----------|----------------|-------------------|----------------|----------|
| `mixed_teacher_threshold_balanced` | 18.2% | 0.0785 | 0.0829 | -0.1012 | 62 | 49 | Best overall mixed-data teacher reference |
| `mixed_teacher_mapped015_balanced` | 16.7% | 0.0818 | 0.1090 | -0.1115 | 63 | 51 | Softer same-teacher agreement raised novelty but lost recall/WER/MOS |
| `mixed_teacher_prototype_balanced` | 18.2% | 0.0854 | 0.1009 | -0.1086 | 60 | 48 | Best mixed-teacher novelty and slightly lower collapse, but still not the best overall tradeoff |

### Interpretation

1. **Changing the teacher helps, but not enough.** The prototype teacher ties the best mixed-data recall at `18.2%` and improves novelty to `0.0854`, the best mixed-teacher novelty result so far.
2. **Coverage improves substantially.** Unlike emotion2vec, the prototype teacher supplies pseudo labels for `confused`, `enunciated`, and `whisper`, which are important because those styles are part of the actual controllable interface.
3. **WER and MOS reveal over-steering.** The prototype model gives back intelligibility and naturalness versus `mixed_teacher_threshold_balanced`, so broader pseudo-label coverage alone is not a clean win.
4. **The recall ceiling remains.** Even a genuinely different teacher does not move the mixed-data line above `18.2%` recall on the current corpus.
5. **The next variant should be guarded, not simply larger.** The promising direction is to keep the prototype teacher's coverage but reduce its risk with pseudo-confidence scaling, lower pseudo row weight, stronger true-label protection, or multi-teacher agreement with emotion2vec.

### Implication

Finding 22 makes the mixed-data supervision bottleneck sharper:

- same-teacher cleanup was too weak (Findings 20-21)
- a different latent prototype teacher improves novelty and coverage
- but the model still trades off WER/MOS and remains below the combined-only checkpoint on controllability

The best current mixed-data teacher reference remains
`mixed_teacher_threshold_balanced`. Finding 23 tests the guarded prototype
path directly; after that result, the best next research move is a
prototype+emotion2vec agreement rule or an intermediate prototype guard, not
another unguarded pseudo-label expansion.

---

## Finding 23: Strong Prototype Guardrails Preserve Recall, But Remove the Prototype Teacher's Novelty Advantage

### Methodology

Finding 22 showed that the latent prototype teacher was the first alternate
teacher to broaden CommonVoice pseudo-label coverage across all 9 controllable
styles, but that the unguarded prototype run gave back WER/MOS versus
`mixed_teacher_threshold_balanced`.

This follow-up kept the same prototype pseudo-label pool and changed only the
training guardrails:

- reused `embeddings/openvoice_commonvoice_cv500_pseudo_prototype_filtered.pt`
- rebuilt the mixed artifact with:
  - `--pseudo-confidence-scale`
  - `--pseudo-row-weight 0.50`
  - `--true-row-weight 1.50`
- trained with a labeled-heavy finish schedule:
  - `--schedule labeled_finish`
  - `--schedule-epochs 1000`
  - `--schedule-end-masses CommonVoice=0.10,CREMA-D=0.45,Expresso=0.45`

This produced:

- `embeddings/openvoice_mixed_teacher_prototype_guarded_base.pt`
- `embeddings/openvoice_vae_mixed_teacher_prototype_guarded.pt`
- `output/mixed_teacher_prototype_guarded_eval/`
- the full emotion / novelty / WER / MOS result bundle under
  `results/eval_*_mixed_teacher_mixed_teacher_prototype_guarded.csv`

### Results

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Identity collapse | Mixed collapse | Takeaway |
|-----------|--------|--------------------------|----------|----------------|-------------------|----------------|----------|
| `mixed_teacher_threshold_balanced` | 18.2% | 0.0785 | 0.0829 | -0.1012 | 62 | 49 | Best overall mixed-data teacher reference |
| `mixed_teacher_prototype_balanced` | 18.2% | 0.0854 | 0.1009 | -0.1086 | 60 | 48 | Best mixed-teacher novelty and slightly lower collapse, but worse WER/MOS |
| `mixed_teacher_prototype_guarded` | 18.2% | 0.0761 | 0.0920 | -0.1081 | 71 | 51 | Guardrails improve WER versus the unguarded prototype, but remove the novelty/collapse advantage |

The guarded prototype condition kept the same selected pseudo-style coverage
inside the real mixed artifact as the unguarded prototype condition:

| Style | Selected CommonVoice pseudo rows |
|-------|----------------------------------|
| anger | 25 |
| confused | 25 |
| disgust | 33 |
| enunciated | 29 |
| fear | 21 |
| happy | 27 |
| neutral | 23 |
| sad | 34 |
| whisper | 10 |

### Interpretation

1. **The strong guard did not improve recall.** `mixed_teacher_prototype_guarded` ties the same `18.2%` mixed-data ceiling as the threshold and prototype teachers.
2. **WER improved relative to the unguarded prototype, but not enough to beat the threshold teacher.** Mean WER improved from `0.1009` to `0.0920`, while `mixed_teacher_threshold_balanced` remains better at `0.0829`.
3. **The guard erased the prototype teacher's main advantage.** Novelty fell from `0.0854` to `0.0761`, below both the unguarded prototype and the threshold teacher.
4. **Collapse behavior worsened.** Identity collapse rose to `71`, compared with `60` for the unguarded prototype and `62` for the threshold teacher.
5. **The problem is not just "prototype labels need less weight."** A strong pseudo-weight reduction plus labeled-heavy finish makes the run safer on WER but pushes the model back toward the same conservative identity-collapse basin.

### Implication

Finding 23 is a useful negative result:

- changing the teacher can improve coverage and novelty (Finding 22)
- but simply guarding that teacher harder does not preserve the benefit
- this set up the hybrid teacher test reported in Finding 24; after that
  result, richer style-space supervision is a stronger next move than more
  hard pseudo-label arbitration or another strong schedule guard on the same
  prototype pseudo labels

For the paper story, `mixed_teacher_threshold_balanced` remains the best
overall mixed-data teacher reference, while `mixed_teacher_prototype_balanced`
remains the strongest evidence that alternate teacher geometry can add useful
coverage/novelty if we can control its WER/collapse cost.

---

## Finding 24: Hybrid Teacher Labels Produce the Best Mixed-Teacher Novelty, But Lose Recall and Naturalness

### Methodology

Finding 23 left a precise question: can we keep emotion2vec's stronger
canonical-emotion precision while using the latent prototype teacher only for
the extra controllable styles that emotion2vec cannot label directly?

This follow-up added `scripts/combine_commonvoice_pseudolabel_teachers.py`,
which combines two already-filtered CommonVoice artifacts:

- emotion2vec pseudo labels from
  `embeddings/openvoice_commonvoice_cv500_pseudo_filtered.pt`
- latent prototype pseudo labels from
  `embeddings/openvoice_commonvoice_cv500_pseudo_prototype_filtered.pt`

The tested policy was `prototype_extra_priority`:

- use emotion2vec-selected rows for canonical emotions:
  `anger`, `disgust`, `fear`, `happy`, `neutral`, `sad`
- use prototype-selected rows for extra styles:
  `confused`, `enunciated`, `whisper`

This produced:

- `embeddings/openvoice_commonvoice_cv500_pseudo_hybrid_extra_priority.pt`
- `embeddings/openvoice_mixed_teacher_hybrid_extra_base.pt`
- `embeddings/openvoice_vae_mixed_teacher_hybrid_extra_balanced.pt`
- `output/mixed_teacher_hybrid_extra_balanced_eval/`
- the full emotion / novelty / WER / MOS result bundle under
  `results/eval_*_mixed_teacher_mixed_teacher_hybrid_extra_balanced.csv`

### Results

Hybrid pseudo-label selection before one-clip-per-speaker mixing:

| Style | Selected rows |
|-------|---------------|
| anger | 5 |
| confused | 40 |
| disgust | 32 |
| enunciated | 40 |
| fear | 4 |
| happy | 37 |
| neutral | 109 |
| sad | 106 |
| whisper | 12 |

Hybrid pseudo-label component sources:

| Component source | Rows |
|------------------|------|
| emotion2vec | 293 |
| prototype | 92 |
| unselected | 817 |

Top-line comparison:

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Identity collapse | Style collapse | Mixed collapse | Files with any collapse | Takeaway |
|-----------|--------|--------------------------|----------|----------------|-------------------|----------------|----------------|-------------------------|----------|
| `mixed_teacher_threshold_balanced` | 18.2% | 0.0785 | 0.0829 | -0.1012 | 62 | 53 | 49 | 66 | Best overall mixed-data teacher reference |
| `mixed_teacher_prototype_balanced` | 18.2% | 0.0854 | 0.1009 | -0.1086 | 60 | 54 | 48 | 66 | Best prototype novelty/coverage before hybrid, but worse WER/MOS |
| `mixed_teacher_prototype_guarded` | 18.2% | 0.0761 | 0.0920 | -0.1081 | 71 | 52 | 51 | 72 | Guardrails improve WER versus prototype but erase novelty/collapse advantage |
| `mixed_teacher_hybrid_extra_balanced` | 16.7% | 0.0860 | 0.0931 | -0.1190 | 61 | 55 | 51 | 65 | Best mixed-teacher novelty and slightly fewer files with any collapse, but recall/MOS are worse |

### Interpretation

1. **The hybrid teacher did not break the recall ceiling.** It actually fell back to `16.7%`, below the `18.2%` threshold/prototype/guarded runs.
2. **It produced the best mixed-teacher novelty so far.** Mean novelty gain rose to `0.0860`, narrowly above the unguarded prototype result (`0.0854`).
3. **The novelty gain is not a clean overall win.** WER (`0.0931`) remains worse than `mixed_teacher_threshold_balanced` (`0.0829`), and MOS delta worsens to `-0.1190`.
4. **Prototype extra-style labels are useful, but hard row-label mixing is too blunt.** The hybrid result supports the idea that prototype geometry contributes coverage/novelty, especially for `confused`, `enunciated`, and `whisper`, but selected hard pseudo labels still do not recover target emotion alignment.
5. **Rare canonical supply remains a bottleneck.** After speaker-first mixing, only `anger=4` and `fear=4` CommonVoice pseudo rows survive, so the canonical-emotion side of the hybrid teacher is still underpowered.

### Implication

Finding 24 narrows the next research step:

- more hard pseudo-label arbitration is unlikely to be enough
- prototype/style information should move into a richer objective, such as a
  style-space auxiliary loss, prototype distillation target, or per-style
  curriculum
- `mixed_teacher_threshold_balanced` remains the best overall mixed-data
  teacher reference
- `mixed_teacher_hybrid_extra_balanced` becomes the strongest evidence that
  alternate teacher geometry can buy novelty/coverage, but at a measurable
  recall/naturalness cost

For the paper story, this is a useful negative/tradeoff result: the broad
CommonVoice speaker prior is still valuable for intelligibility, but the
current route for injecting style into that prior is too weak when it is
represented only as hard pseudo labels.

---

## Finding 25: Continuous Style-Space Distillation Improves the Hybrid Teacher Tradeoff, But Still Does Not Recover Recall

### Methodology

Finding 24 showed that the hybrid emotion2vec + latent-prototype teacher
contains useful geometry: it produced the best mixed-teacher novelty so far,
but using that teacher only as selected hard pseudo labels hurt recall and
naturalness.

This follow-up changed the supervision mechanism rather than the data artifact:

- kept the hybrid mixed artifact fixed:
  `embeddings/openvoice_mixed_teacher_hybrid_extra_base.pt`
- trained a mixed VAE with a frozen style teacher:
  `embeddings/openvoice_vae_combined.pt`
- added a continuous MSE loss on VAE encoder mean dimensions `0-8`
- applied that loss only to `CommonVoice` rows in the mixed artifact
- used teacher weight `0.25` with the same `static_balanced` schedule

This produced:

- `embeddings/openvoice_vae_mixed_teacher_hybrid_style_distill_balanced.pt`
- `output/mixed_teacher_hybrid_style_distill_balanced_eval/`
- the full emotion / novelty / WER / MOS result bundle under
  `results/eval_*_mixed_teacher_mixed_teacher_hybrid_style_distill_balanced.csv`
- regenerated summary and collapse tables:
  `results/eval_mixed_teacher_summary.csv` and
  `results/eval_mixed_teacher_collapse.csv`

### Results

Top-line comparison:

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Identity collapse | Style collapse | Mixed collapse | Files with any collapse | Takeaway |
|-----------|--------|--------------------------|----------|----------------|-------------------|----------------|----------------|-------------------------|----------|
| `mixed_teacher_threshold_balanced` | 18.2% | 0.0785 | 0.0829 | -0.1012 | 62 | 53 | 49 | 66 | Best overall mixed-data teacher reference |
| `mixed_teacher_prototype_balanced` | 18.2% | 0.0854 | 0.1009 | -0.1086 | 60 | 54 | 48 | 66 | Best prototype novelty/coverage before hybrid, but worse WER/MOS |
| `mixed_teacher_hybrid_extra_balanced` | 16.7% | 0.0860 | 0.0931 | -0.1190 | 61 | 55 | 51 | 65 | Best hard-label hybrid novelty, but recall/MOS are worse |
| `mixed_teacher_hybrid_style_distill_balanced` | 16.7% | 0.0861 | 0.0938 | -0.1072 | 58 | 55 | 50 | 63 | Preserves hybrid novelty and improves MOS/collapse, but recall remains stuck |

### Interpretation

1. **Continuous style-space supervision is better than hard hybrid labels for naturalness/collapse.** MOS delta improves from `-0.1190` to `-0.1072`, identity collapse falls from `61` to `58`, mixed collapse falls from `51` to `50`, and files with any collapse fall from `65` to `63`.
2. **It preserves the hybrid teacher's novelty benefit.** Mean novelty gain is `0.0861`, narrowly above the hard hybrid result (`0.0860`) and above the prototype-only result (`0.0854`).
3. **It still does not recover target emotion recall.** Recall remains `16.7%`, so the neutral-basin failure is not caused only by hard-label arbitration.
4. **The threshold teacher remains the best overall mixed-data reference.** `mixed_teacher_threshold_balanced` still wins on recall (`18.2%`), WER (`0.0829`), and MOS delta (`-0.1012`).
5. **The next bottleneck is calibration, not just objective form.** A continuous teacher target helps the tradeoff, but it likely needs per-style masks, confidence weighting, or a curriculum to push rare canonical emotions out of the neutral basin.

### Implication

Finding 25 strengthens the paper story in two ways:

- it shows the project did not stop at hard pseudo labels; we tested a richer
  latent-space supervision mechanism,
- and it narrows the remaining research gap to calibrated teacher supervision
  or architecture/curriculum changes, rather than simply "try better labels."

For the paper, this is a useful tradeoff finding: the broad CommonVoice speaker
prior can be combined with continuous teacher geometry to preserve novelty and
naturalness better than hard hybrid labels, but current mixed-data supervision
still cannot match the combined-only controllability reference.

---

## Finding 26: Global Style-Teacher Weight Calibration Does Not Recover Recall

### Methodology

Finding 25 showed that continuous style-space distillation at teacher weight
`0.25` improved the hybrid teacher tradeoff but did not recover recall. The
natural next question was whether the teacher loss was simply underpowered or
overweighted.

This follow-up held the data, schedule, style dims, teacher checkpoint,
inference corpus, and evaluation stack fixed, changing only the scalar
style-teacher weight:

- training artifact:
  `embeddings/openvoice_mixed_teacher_hybrid_extra_base.pt`
- frozen style teacher:
  `embeddings/openvoice_vae_combined.pt`
- style-teacher dims:
  `0-8`
- teacher rows:
  `CommonVoice`
- schedule:
  `static_balanced`
- compared weights:
  `0.10`, `0.25`, and `0.50`

This produced two new checked conditions:

- `mixed_teacher_hybrid_style_distill_w010_balanced`
- `mixed_teacher_hybrid_style_distill_w050_balanced`

Each condition has a matched 110-row evaluation corpus and full emotion /
novelty / WER / MOS metric bundle under:

- `output/mixed_teacher_hybrid_style_distill_w010_balanced_eval/`
- `output/mixed_teacher_hybrid_style_distill_w050_balanced_eval/`
- `results/eval_*_mixed_teacher_mixed_teacher_hybrid_style_distill_w010_balanced.csv`
- `results/eval_*_mixed_teacher_mixed_teacher_hybrid_style_distill_w050_balanced.csv`
- regenerated `results/eval_mixed_teacher_summary.csv` and
  `results/eval_mixed_teacher_collapse.csv`

### Results

Top-line comparison:

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Identity collapse | Style collapse | Mixed collapse | Files with any collapse | Takeaway |
|-----------|--------|--------------------------|----------|----------------|-------------------|----------------|----------------|-------------------------|----------|
| `mixed_teacher_threshold_balanced` | 18.2% | 0.0785 | 0.0829 | -0.1012 | 62 | 53 | 49 | 66 | Best overall mixed-data teacher reference |
| `mixed_teacher_hybrid_style_distill_w010_balanced` | 16.7% | 0.0854 | 0.0924 | -0.1161 | 62 | 54 | 51 | 65 | Lower teacher weight preserves novelty but gives back MOS/collapse |
| `mixed_teacher_hybrid_style_distill_balanced` | 16.7% | 0.0861 | 0.0938 | -0.1072 | 58 | 55 | 50 | 63 | Best style-distillation novelty/MOS tradeoff so far |
| `mixed_teacher_hybrid_style_distill_w050_balanced` | 16.7% | 0.0840 | 0.0821 | -0.1196 | 56 | 55 | 48 | 63 | Higher teacher weight improves WER and identity/mixed collapse, but not recall or MOS |

### Interpretation

1. **A global teacher-weight sweep does not recover target-emotion recall.** All three tested weights (`0.10`, `0.25`, `0.50`) stay at `16.7%` recall.
2. **The scalar weight mostly trades off intelligibility/collapse against novelty/MOS.** Weight `0.50` gives the best WER (`0.0821`) and lowest identity/mixed collapse (`56` / `48`), but it loses novelty (`0.0840`) and MOS delta (`-0.1196`) relative to the `0.25` condition.
3. **The `0.25` condition remains the best style-distillation tradeoff, not a recall breakthrough.** It has the best style-distillation novelty (`0.0861`) and MOS delta (`-0.1072`), but it remains below `mixed_teacher_threshold_balanced` on recall.
4. **The bottleneck is not just global teacher-loss strength.** If the loss were globally underpowered, `0.50` should have moved recall. It did not.
5. **The next intervention should be class-specific.** The evidence points toward per-style masks, confidence weighting, rare-class protection, or curriculum rather than another scalar teacher-weight tweak.

### Implication

Finding 26 is useful because it closes a tempting but shallow branch of the
search space. We now have evidence that continuous teacher geometry helps the
novelty/naturalness/collapse tradeoff, but simply turning that geometry up or
down does not move the model out of the neutral-basin recall failure.

For the paper story, this strengthens the argument that mixed-data broad
speaker coverage is not enough by itself: the model needs more structured
style supervision, especially for low-supply canonical emotions.

---

## Finding 27: Target-Dimension Style-Teacher Masks Still Do Not Break the Neutral Basin

### Methodology

Finding 26 showed that changing the global style-teacher loss weight did not
recover recall. The next hypothesis was that the teacher loss was too diffuse:
matching all style dimensions for each CommonVoice row might preserve the
teacher's broad latent geometry while failing to push the specific accepted
target style axis.

This follow-up held the mixed artifact, frozen teacher, style dims, schedule,
inference corpus, and four-metric evaluation stack fixed, then changed the
teacher loss to be class-specific:

- training artifact:
  `embeddings/openvoice_mixed_teacher_hybrid_extra_base.pt`
- frozen style teacher:
  `embeddings/openvoice_vae_combined.pt`
- style-teacher dims:
  `0-8`
- teacher rows:
  labeled / accepted CommonVoice rows only (`267/1325`)
- target mode:
  `target_dim`, meaning each row only matches the frozen teacher on its
  accepted style dimension
- row weighting:
  rare canonical emotions upweighted, neutral downweighted
- confidence scaling:
  row weights multiplied by `confidence^0.5`
- schedule:
  `static_balanced`

This produced one new checked condition:

- `mixed_teacher_hybrid_style_distill_targetmask_balanced`

The condition has a matched 110-row evaluation corpus and full emotion /
novelty / WER / MOS metric bundle under:

- `output/mixed_teacher_hybrid_style_distill_targetmask_balanced_eval/`
- `results/eval_*_mixed_teacher_mixed_teacher_hybrid_style_distill_targetmask_balanced.csv`
- regenerated `results/eval_mixed_teacher_summary.csv` and
  `results/eval_mixed_teacher_collapse.csv`

### Results

Top-line comparison:

| Condition | Recall | Novelty gain vs baseline | Mean WER | Mean MOS delta | Identity collapse | Style collapse | Mixed collapse | Files with any collapse | Takeaway |
|-----------|--------|--------------------------|----------|----------------|-------------------|----------------|----------------|-------------------------|----------|
| `mixed_teacher_threshold_balanced` | 18.2% | 0.0785 | 0.0829 | -0.1012 | 62 | 53 | 49 | 66 | Best overall mixed-data teacher reference |
| `mixed_teacher_hybrid_style_distill_balanced` | 16.7% | 0.0861 | 0.0938 | -0.1072 | 58 | 55 | 50 | 63 | Best global style-distillation novelty/MOS tradeoff |
| `mixed_teacher_hybrid_style_distill_targetmask_balanced` | 16.7% | 0.0852 | 0.1062 | -0.1181 | 60 | 54 | 51 | 63 | Target masks and confidence weighting do not recover recall |
| `mixed_teacher_hybrid_style_distill_w050_balanced` | 16.7% | 0.0840 | 0.0821 | -0.1196 | 56 | 55 | 48 | 63 | Higher global weight improves WER/collapse but not recall or MOS |

Emotion recall remained `11/66 = 16.7%`, entirely from neutral:

- anger: `0/11`
- disgust: `0/11`
- fear: `0/11`
- happy: `0/11`
- neutral: `11/11`
- sad: `0/11`

The novelty metric still moved for extra-style axes that emotion2vec cannot
directly score:

- `whisper` novelty gain: `0.3714`
- `confused` novelty gain: `0.2411`
- overall mean novelty gain: `0.0852`

### Interpretation

1. **Per-style target masking does not recover target-emotion recall.** The model remains in the same classifier-visible neutral basin as the global style-distillation runs.
2. **The style-space teacher can still move speaker/style geometry without moving emotion2vec target labels.** Novelty remains high for `whisper` and `confused`, but canonical emotion recall does not improve.
3. **The failure is not simply "all style dimensions are diluting the target axis."** If that were the main problem, `target_dim` should have helped. It did not.
4. **Confidence weighting and rare-class row weights are not sufficient at the current pseudo-label scale.** Upweighting tiny accepted CommonVoice classes (`anger=4`, `fear=4`) cannot compensate for weak or mismatched target geometry.
5. **The next intervention should be diagnostic or curriculum-driven.** The strongest next step is to measure where teacher means, student means, generated emotion predictions, novelty, and collapse diverge by style and speaker, then test a labeled-first curriculum or decoder-aware style objective.

### Implication

Finding 27 closes another tempting branch of the mixed-data search space:
class-specific teacher masking by itself is not enough. The result strengthens
the paper narrative that broad unlabeled/pseudo-labeled speaker coverage helps
some quality and novelty axes, but recovering controllable emotion requires
style supervision that is both class-specific and decoder-aware.

For the paper, this is a useful negative result because it narrows the remaining
gap from "try per-style weights" to a sharper question: can a curriculum,
stronger pseudo-label supply, or generated-audio/decoder-aware objective align
the latent style axes with classifier-visible emotion without losing the
mixed-data intelligibility gains?

---

## Finding 28: Per-Style Diagnostics Localize the Mixed-Teacher Failure

### Methodology

After Finding 27, the next risk was wasting another training turn on a guessed
objective. We therefore built a per-style diagnostic that joins:

- CommonVoice / CREMA-D / Expresso label supply in the mixed artifact,
- frozen combined-VAE teacher encoder means,
- target-masked mixed-teacher student encoder means,
- generated-output emotion2vec, novelty, WER, and MOS metrics,
- and collapse taxonomy rows.

The diagnostic was run on:

- mixed artifact:
  `embeddings/openvoice_mixed_teacher_hybrid_extra_base.pt`
- frozen teacher:
  `embeddings/openvoice_vae_combined.pt`
- student checkpoint:
  `embeddings/openvoice_vae_mixed_teacher_hybrid_style_distill_targetmask_balanced.pt`
- condition:
  `mixed_teacher_hybrid_style_distill_targetmask_balanced`

Artifacts:

- `scripts/analyze_mixed_teacher_style_diagnostics.py`
- `results/eval_mixed_teacher_style_diagnostics_targetmask.csv`
- `results/eval_mixed_teacher_style_diagnostics_targetmask.md`

The most important latent diagnostic is **target top1 rate**: among the selected
style dims `0-8`, how often is the intended target style dimension the largest
teacher or student encoder-mean dimension for rows accepted as that style?

### Results

| Style | Active CommonVoice teacher rows | Teacher target top1 | Student target top1 | Recall | Neutral prediction rate | Novelty gain | Takeaway |
|-------|---------------------------------|---------------------|---------------------|--------|-------------------------|--------------|----------|
| anger | 4 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.0020 | Too few rows and no teacher target dominance |
| disgust | 27 | 0.2222 | 0.0741 | 0.0000 | 1.0000 | 0.0251 | Teacher target usually not dominant |
| fear | 4 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.0036 | Too few rows and no teacher target dominance |
| happy | 30 | 0.1000 | 0.2333 | 0.0000 | 0.9091 | -0.0011 | More rows, but weak teacher target geometry |
| sad | 78 | 0.2692 | 0.7308 | 0.0000 | 1.0000 | 0.0232 | Student can align latent sad, but generated audio still reads neutral |
| confused | 22 | 1.0000 | 0.6818 | n/a | 0.8182 | 0.2411 | Latent geometry moves novelty, but this is not an emotion2vec recall class |
| whisper | 9 | 1.0000 | 0.7778 | n/a | 0.3636 | 0.3714 | Strong latent novelty, but high WER/MOS cost remains |

### Interpretation

1. **The canonical-emotion failure is visible before decoding.** For `anger`, `fear`, `happy`, `disgust`, `neutral`, and `sad`, the frozen teacher's accepted CommonVoice rows often do not rank the intended style dimension first. That means the pseudo-style labels and teacher latent geometry are not well aligned for the styles that matter to emotion2vec recall.
2. **Rare-class weighting cannot solve absent supply.** `anger` and `fear` each have only `4` active CommonVoice teacher rows. Upweighting four rows does not create a stable class manifold.
3. **Sad exposes a decoder/output-level mismatch.** The student makes the sad target dimension top-ranked for `73.1%` of active sad rows, but generated sad samples are still classified as neutral. This shows that latent target dominance can be insufficient if the decoder maps that latent state to classifier-neutral audio.
4. **Extra styles behave differently from canonical emotions.** `confused` and `whisper` have coherent latent teacher geometry and strong novelty gains, but they are not counted by emotion2vec recall and they carry quality/intelligibility costs.
5. **The next intervention should be curriculum or decoder-aware, not another latent-only mask.** We now know where the failure sits: weak canonical teacher geometry, rare pseudo-label supply, and latent-to-output mismatch.

### Implication

Finding 28 sharpens the mixed-data story substantially. The issue is not merely
that CommonVoice is broad and weakly labeled. The accepted pseudo labels for
canonical emotions often do not correspond to a teacher latent state where the
intended style dimension is dominant, and even when the student can align a
latent dimension (`sad`), the generated audio can remain classifier-neutral.

For the paper, this supports a more rigorous negative-result narrative: broad
speaker coverage plus pseudo labels improves some stability/quality axes, but
recovering controllable emotion needs stronger class-specific supply and an
objective that is aware of generated audio behavior, not just latent teacher
agreement.

---

## Finding 29: Labeled-First Curriculum Improves Novelty/Collapse but Not Emotion Recall

### Methodology

Finding 28 suggested a natural next test: maybe the continuous CommonVoice
teacher signal was arriving too early, before the model had stabilized the
CREMA-D / Expresso labeled style axes. We therefore added a labeled-first
curriculum:

- schedule `labeled_warmup`
- initial data mix: `CommonVoice=0.00`, `CREMA-D=0.50`, `Expresso=0.50`
- final data mix: `CommonVoice=0.33`, `CREMA-D=0.33`, `Expresso=0.33`
- teacher-style weight ramp: `0.0 -> 0.25`
- teacher rows: CommonVoice only
- teacher dims: style dims `0-8`
- schedule ramp: first `1000` epochs of a `3000` epoch run

Training artifact:

- `embeddings/openvoice_vae_mixed_teacher_hybrid_style_distill_labeled_warmup.pt`

Evaluation artifacts:

- `output/mixed_teacher_hybrid_style_distill_labeled_warmup_eval/`
- `results/eval_emotion_mixed_teacher_mixed_teacher_hybrid_style_distill_labeled_warmup.csv`
- `results/eval_novelty_mixed_teacher_mixed_teacher_hybrid_style_distill_labeled_warmup.csv`
- `results/eval_wer_mixed_teacher_mixed_teacher_hybrid_style_distill_labeled_warmup.csv`
- `results/eval_mos_mixed_teacher_mixed_teacher_hybrid_style_distill_labeled_warmup.csv`
- `results/eval_mixed_teacher_style_diagnostics_labeled_warmup.csv`
- `results/eval_mixed_teacher_style_diagnostics_labeled_warmup.md`

The same deterministic 110-row, 11-speaker evaluation corpus and four-metric
stack were used, so the result is directly comparable to the previous
mixed-teacher rows.

### Results

| Condition | Recall | Novelty gain | Mean WER | MOS delta | Identity collapse | Files with any collapse | Takeaway |
|-----------|--------|--------------|----------|-----------|-------------------|-------------------------|----------|
| `mixed_teacher_threshold_balanced` | `18.2%` | `0.0785` | `0.0829` | `-0.1012` | `62` | `66` | Best overall mixed-data teacher reference |
| `mixed_teacher_hybrid_style_distill_balanced` | `16.7%` | `0.0861` | `0.0938` | `-0.1072` | `58` | `63` | Best previous style-distillation tradeoff |
| `mixed_teacher_hybrid_style_distill_labeled_warmup` | `16.7%` | `0.0930` | `0.0924` | `-0.1093` | `54` | `61` | Higher novelty and less collapse, but recall still neutral-only |
| `mixed_teacher_hybrid_style_distill_targetmask_balanced` | `16.7%` | `0.0852` | `0.1062` | `-0.1181` | `60` | `63` | Target masking did not help recall |

### Interpretation

1. **Curriculum timing alone does not recover target emotion recall.** The
   labeled-first run remains at `16.7%` emotion recall, again entirely from
   neutral.
2. **The curriculum does improve secondary axes.** Novelty rises to `0.0930`,
   the strongest checked-in mixed-teacher novelty so far, while identity
   collapse falls to `54` and files with any collapse fall to `61`.
3. **The bottleneck is now sharper.** Protecting labeled axes before adding
   CommonVoice teacher geometry helps the latent / identity tradeoff, but it
   does not make generated audio classifier-visible as anger, fear, happy, sad,
   or disgust.
4. **Another schedule-only variant is unlikely to be the best next move.** The
   next intervention should either add stronger rare-class supply or introduce
   a decoder-aware / generated-audio style objective.

### Implication

Finding 29 is a useful negative result with a narrow positive signal. It shows
that labeled-first curriculum is not a recall breakthrough, but it can improve
novelty and collapse behavior. For the paper story, this supports the claim
that the remaining bottleneck is not just data-mix timing; it is the mismatch
between latent teacher geometry and classifier-visible generated emotion.

The next paper-relevant question is therefore: can a decoder-aware style loss
or better rare-class CommonVoice supply turn the novelty/collapse gains into
actual emotion recall?

---

## April 30 Meeting Alignment with Joe

The April 30 call with Joe did **not** change the scientific findings above,
but it did sharpen what we needed to test next. That recommendation has now
been exercised by the mixed-data branch, so this section is best read as:

- what Joe pushed us to try,
- what we completed afterward,
- and what remains open now.

The most important alignment points were:

1. **The biggest missing experiment really was the first real mixed-data run.**
   Joe's summary was that we still had not trained one VAE on
   **CommonVoice + CREMA-D + Expresso together** while retaining both broad
   speaker diversity and controllable emotion. We have now completed that run,
   and the result is Finding 17: the first mixed-data schedules improved WER
   more than recall.
2. **Pseudo-labeled CommonVoice remains the most promising bootstrap path.**
   Joe explicitly endorsed the idea of using pretrained emotion models to add
   labels to CommonVoice and then combining that data with the smaller labeled
   datasets. The mixed-data branch did not prove that this works yet, but it
   does keep this direction alive as the best current data-side bootstrap path.
3. **Data mixing matters more than another small finetuning tweak, but schedule
   choice alone is not enough.**
   Joe warned that a naive combined run could simply behave like CommonVoice if
   the smaller labeled datasets are not protected in the mixture. That means
   explicit mixture control was the right thing to test first. After Finding 17,
   the remaining data-side question is now stronger labeled-data protection and
   better pseudo-label filtering, not just another simple schedule sweep.
4. **Speaker breadth is now an explicit heuristic.**
   Joe's current assumption is that for CommonVoice, getting at least one clip
   per speaker may matter more than maximizing total clip count. The first
   mixed-data artifact followed that heuristic, and the next follow-up should
   compare one clip per speaker against two clips per speaker directly.
5. **Architecture is still secondary to data for the immediate next step.**
   Joe agrees that the current linear latent control may not be optimal, but
   he still put data composition and supervision quality ahead of architecture
   changes for the next branch.
6. **`style_strength = 5.0` should not be treated as a hard ceiling.**
   Joe qualitatively found that higher strengths could still work well,
   especially for whisper on non-Trump examples. Future sweeps should treat
   strength as style- and speaker-dependent rather than fixed.

This meeting therefore changed the next-step framing from:

- "improve CommonVoice pseudo-label quality in isolation"

to:

- "**run the first sampled mixed-data experiment with pseudo-labeled CommonVoice
  plus explicit mixture/schedule control**."

That experiment, its first quality follow-up, the first non-Trump
  style-strength sweep, the teacher-family branch, the first style-space
  distillation follow-up, the global style-teacher weight sweep, the
  target-dimension mask follow-up, the first per-style diagnostic, and the
  first labeled-first curriculum are now complete, so the current follow-up
  framing is:

- "**move from latent-only mixed-data teacher calibration and schedule-only
  curriculum to decoder-aware style objectives or stronger rare-class supply,
  while keeping the new style-strength guidance as a documented inference-side
  finding**."

---

## Finding 30: Expanded Rare-Class CommonVoice Supply Produces the First Large Mixed-Teacher Recall Jump, With a Clear Quality Tradeoff

### Methodology

Finding 28 showed that the previous mixed-teacher artifact had only `4` active
`anger` rows and `4` active `fear` rows after filtering and speaker-first
sampling. We therefore tested the data-side hypothesis directly: if rare
CommonVoice pseudo-label supply is fixed, does generated-audio emotion recall
move out of the neutral basin?

The expanded run used:

- branch: `research/controllable-vae`
- local corpus: `/Users/steve/datasets/cv-corpus-21.0-2025-03-14/en`
- expanded OpenVoice extraction:
  `embeddings/openvoice_commonvoice_cvrare_expanded_emb.pt`
- extracted CommonVoice embeddings: `25910`
- unique CommonVoice speakers: `13308`
- missing/unreadable clips: `0`
- target-seeking emotion2vec scorer stop condition: `anger=50,fear=50`
- rows annotated before stop: `6380/25910`
- hybrid selected pseudo rows: `645`
- final mixed artifact:
  `embeddings/openvoice_mixed_teacher_cvrare_hybrid_extra_base.pt`
- final mixed rows: `14195`
- final CommonVoice rows: `13370`
- true labeled rows: `825` (`CREMA-D=546`, `Expresso=279`)

The mixed builder used `--commonvoice-preserve-selected-pseudo`, which keeps
one-clip-per-speaker speaker breadth but adds selected pseudo-labeled rows that
would otherwise be lost behind the per-speaker cap. This preserved the expanded
rare-label gate in the final training artifact:

- `anger=50`
- `confused=50`
- `disgust=79`
- `enunciated=50`
- `fear=50`
- `happy=80`
- `neutral=116`
- `sad=120`
- `whisper=50`

We then trained:

- `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup`
- checkpoint:
  `embeddings/openvoice_vae_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.pt`
- schedule: `labeled_warmup`
- frozen teacher: `embeddings/openvoice_vae_combined.pt`
- teacher-style weight ramp: `0.0 -> 0.25`
- teacher dims: `0-8`

Evaluation used the same deterministic 110-row, 11-speaker corpus as the
previous mixed-teacher comparisons:

- style strength: `5.0`
- noise level: `0.0`
- seed: `42`
- emotion recall / emo_sim
- speaker novelty gain
- WER
- predicted MOS
- collapse taxonomy
- browser listening report

### Results

| Condition | Recall | Novelty gain | Mean styled WER | MOS delta | Identity collapse | Files with any collapse | Takeaway |
|-----------|--------|--------------|-----------------|-----------|-------------------|-------------------------|----------|
| `combined` | `25.8%` | `0.2599` | `0.2353` | `-0.0792` | `7` | `46` | Still the cleanest original controllable baseline |
| `mixed_teacher_threshold_balanced` | `18.2%` | `0.0785` | `0.0829` | `-0.1012` | `62` | `66` | Best previous mixed-teacher reference |
| `mixed_teacher_hybrid_style_distill_labeled_warmup` | `16.7%` | `0.0930` | `0.0924` | `-0.1093` | `54` | `61` | Best previous style-distillation novelty/collapse variant |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup` | `47.0%` | `0.2995` | `0.2751` | `-0.2640` | `1` | `25` | First large mixed-teacher recall/novelty jump, but with a quality/content cost |

Per-style canonical recall for the expanded rare-supply run:

| Style | Recall | Neutral predictions | Readout |
|-------|--------|---------------------|---------|
| `anger` | `1/11` | `7/11` | Still weak; expanded supply alone does not calibrate anger |
| `disgust` | `2/11` | `9/11` | Still mostly neutral-classified |
| `fear` | `3/11` | `0/11` | Moves off zero; errors are no longer just neutral collapse |
| `happy` | `5/11` | `1/11` | Largest canonical improvement outside neutral/sad |
| `neutral` | `10/11` | `10/11` | Strong, as expected |
| `sad` | `10/11` | `1/11` | Strong recall, but carries the worst WER burden |

### Diagnostics

The expanded diagnostic report is:

- `results/eval_mixed_teacher_style_diagnostics_cvrare_labeled_warmup.md`

It shows that supply is no longer the only bottleneck:

| Style | Teacher rows | Teacher target top-1 | Student target top-1 | Recall | WER | MOS delta |
|-------|--------------|----------------------|----------------------|--------|-----|-----------|
| `anger` | `50` | `0.2600` | `0.3600` | `0.0909` | `0.1740` | `-0.0532` |
| `disgust` | `79` | `0.1899` | `0.1519` | `0.1818` | `0.1529` | `-0.2315` |
| `fear` | `50` | `0.1400` | `0.2000` | `0.2727` | `0.3461` | `-0.3903` |
| `happy` | `80` | `0.2375` | `0.0625` | `0.4545` | `0.3981` | `-0.0026` |
| `neutral` | `116` | `0.0948` | `0.0603` | `0.9091` | `0.1269` | `-0.1472` |
| `sad` | `120` | `0.3167` | `0.1917` | `0.9091` | `0.5847` | `-0.0311` |

### Interpretation

1. **Rare-class supply was a real bottleneck.** Moving from `4-5` active
   `anger` / `fear` rows to `50` each produced the first large mixed-teacher
   recall jump: from the previous mixed-teacher ceiling of `18.2%` to `47.0%`.
2. **The expanded run also beats the combined model on novelty.** Mean novelty
   gain rises to `0.2995`, above the combined model's `0.2599`. This means the
   model is not merely becoming more conservative; it is making larger speaker
   embedding moves.
3. **The price is intelligibility and naturalness.** Mean styled WER rises to
   `0.2751`, MOS delta falls to `-0.2640`, and `enunciated`, `sad`, `happy`,
   and `fear` need perceptual review before this can be treated as a clean win.
4. **Expanded supply does not fully calibrate the teacher.** Teacher target
   top-1 rates remain weak for canonical emotion classes, so the next
   bottleneck is pseudo-label calibration / generated-output alignment, not raw
   row scarcity.
5. **This is now close to Joe's paper-quality recall target.** The system is
   within a few points of `>50%` emotion recall on the standard 11-speaker
   panel, but the quality tradeoff means the next experiment should preserve
   this gain while repairing WER/MOS.

### Implication

Finding 30 is the strongest positive mixed-data result so far. It changes the
story from "CommonVoice breadth improves quality but cannot recover control" to
"CommonVoice breadth can recover control when rare pseudo-label supply is
protected, but the current latent teacher objective pays for that control with
content and naturalness degradation."

For the paper, this supports both a stronger result and a sharper limitation:
speaker breadth plus class-balanced pseudo-label supply can produce
near-paper-target controllability, but robust controllable DP voice conversion
needs decoder-aware or generated-audio style supervision to keep the outputs
intelligible and natural.

The listening artifact for subjective review is:

- `results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.html`

---

## Finding 31: Per-Style Strength Calibration Preserves the Expanded Recall Gain While Repairing Part of the Quality Cost

### Methodology

Finding 30 showed that the expanded rare-supply checkpoint is the strongest
controllability / novelty result so far, but its generated audio pays a real
quality cost. We therefore tested whether the problem is partly an inference
calibration issue: some styles may need less latent push than the global
`style_strength=5.0` default.

The follow-up added:

- `scripts/run_ablation_inference.py --style-strength-map`
- `scripts/run_generated_audio_eval_suite.py`
- `configs/style_strength_profiles/cvrare_content_guard.json`
- `configs/style_strength_profiles/cvrare_sad_enunc_guard.json`

Both profiles used the same deterministic 110-row evaluation setup:

- checkpoint:
  `embeddings/openvoice_vae_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.pt`
- source panel: `11` checked-in source speakers
- noise level: `0.0`
- seed: `42`
- metric stack: emotion recall / emo_sim, speaker novelty, WER, MOS, collapse
  taxonomy, and browser listening reports

The two tested profiles were:

- **content guard:** reduce `sad`, `enunciated`, `fear`, `happy`, and
  `confused`
- **sad/enunciated guard:** reduce only the clearest quality-risk styles
  (`sad`, `enunciated`, and `confused`) while leaving canonical emotions at the
  original strength where possible

### Results

| Condition | Recall | Novelty gain | Mean styled WER | MOS delta | Content collapse | Style-to-neutral collapse | Identity collapse | Mixed collapse | Files with any collapse | Takeaway |
|-----------|--------|--------------|-----------------|-----------|------------------|---------------------------|-------------------|----------------|-------------------------|----------|
| `combined` | `25.8%` | `0.2599` | `0.2353` | `-0.0792` | `6` | `37` | `7` | `4` | `46` | Cleanest original quality baseline |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup` | `47.0%` | `0.2995` | `0.2751` | `-0.2640` | `7` | `18` | `1` | `1` | `25` | Strongest novelty, but quality/content cost |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_content_guard` | `40.9%` | `0.2653` | `0.2133` | `-0.1989` | `1` | `24` | `1` | `2` | `24` | Best WER/MOS repair, but loses recall |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `47.0%` | `0.2726` | `0.2348` | `-0.2081` | `2` | `18` | `1` | `1` | `20` | Best current recall/quality Pareto point |

Per-style canonical recall confirms why the narrower guard is preferred:

| Style | Unguarded recall | Content guard recall | Sad/enunciated guard recall |
|-------|------------------|----------------------|-----------------------------|
| `anger` | `1/11` | `1/11` | `1/11` |
| `disgust` | `2/11` | `2/11` | `2/11` |
| `fear` | `3/11` | `1/11` | `3/11` |
| `happy` | `5/11` | `3/11` | `5/11` |
| `neutral` | `10/11` | `10/11` | `10/11` |
| `sad` | `10/11` | `10/11` | `10/11` |

### Interpretation

1. **Some quality damage is inference-calibration damage.** Lowering the
   strongest quality-risk styles improves WER, MOS, and collapse without
   retraining the VAE.
2. **Broadly lowering difficult canonical emotions is too expensive.** The
   content guard repairs WER/MOS more, but drops recall to `40.9%` by weakening
   `fear` and `happy`.
3. **The `sad/enunciated` guard is the best current demo/eval profile.** It
   preserves the expanded run's `47.0%` recall, keeps novelty above the
   combined baseline (`0.2726` vs `0.2599`), reduces WER from `0.2751` to
   `0.2348`, improves MOS delta from `-0.2640` to `-0.2081`, and lowers files
   with any collapse from `25` to `20`.
4. **This does not replace the next training objective.** A hand-authored
   style-strength profile is useful evidence and a better listening preset, but
   the paper-quality method should still move toward decoder-aware or
   generated-audio style supervision.

### Implication

Finding 31 sharpens the Finding 30 story. Expanded rare-class CommonVoice
supply really does recover control, and part of the apparent quality cost can
be reduced with per-style inference calibration. The paper should present the
unguarded run as the high-novelty result and the `sad/enunciated` guard as the
current best quality-balanced variant, while clearly labeling the guard as an
inference-side calibration step.

Recommended listening and diagnostic artifacts:

- `results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.html`
- `results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_content_guard.html`
- `results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.html`

---

## Finding 32: The First Decoder-Prototype Objective Preserves Novelty, But Does Not Learn the Quality-Balanced Repair

### Methodology

Finding 31 showed that the expanded rare-supply checkpoint can be made more
usable with a hand-authored per-style inference guard. The next question was
whether training could learn a similar repair directly.

This follow-up added a decoder-prototype objective to mixed-data VAE training:

- build one real embedding prototype per style from true-labeled
  CREMA-D/Expresso rows
- encode a training row and take `model.last_mu`
- apply an inference-like style control to the latent (`target_only` mode)
- decode the controlled latent back into OpenVoice embedding space
- match that decoded embedding to the corresponding real style prototype

The pilot trained from the expanded rare-supply checkpoint:

- base artifact:
  `embeddings/openvoice_mixed_teacher_cvrare_hybrid_extra_base.pt`
- init checkpoint:
  `embeddings/openvoice_vae_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.pt`
- output checkpoint:
  `embeddings/openvoice_vae_mixed_teacher_cvrare_decoder_proto_labeled_warmup.pt`
- style-teacher ramp: `0.0 -> 0.25`
- decoder-prototype ramp: `0.0 -> 0.02`
- decoder-prototype rows: selected CommonVoice rows with labels
- prototype source: true-labeled rows only
- generation: same deterministic 110-row source panel, `style_strength=5.0`,
  `noise_level=0.0`, seed `42`

A lower-weight follow-up kept the same setup but reduced the
decoder-prototype ramp to `0.0 -> 0.005`, saving:

- `embeddings/openvoice_vae_mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup.pt`

### Results

| Condition | Recall | Novelty gain | Mean styled WER | MOS delta | Content collapse | Style-to-neutral collapse | Identity collapse | Mixed collapse | Files with any collapse | Takeaway |
|-----------|--------|--------------|-----------------|-----------|------------------|---------------------------|-------------------|----------------|-------------------------|----------|
| `combined` | `25.8%` | `0.2599` | `0.2353` | `-0.0792` | `6` | `37` | `7` | `4` | `46` | Cleanest original quality baseline |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup` | `47.0%` | `0.2995` | `0.2751` | `-0.2640` | `7` | `18` | `1` | `1` | `25` | Strongest high-novelty expanded run |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `47.0%` | `0.2726` | `0.2348` | `-0.2081` | `2` | `18` | `1` | `1` | `20` | Current quality-balanced reference |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `42.4%` | `0.3008` | `0.2863` | `-0.2148` | `4` | `22` | `1` | `0` | `27` | High novelty, but worse recall/WER/collapse than the guard |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup_sad_enunc_guard` | `42.4%` | `0.2718` | `0.2592` | `-0.1787` | `4` | `23` | `2` | `1` | `28` | Improves decoder-prototype WER/MOS, but not recall/collapse |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `42.4%` | `0.3032` | `0.2782` | `-0.2122` | `3` | `22` | `1` | `0` | `26` | Lower prototype weight improves novelty/collapse slightly, but still misses the reference |

Generated-audio failure mining then joined emotion, novelty, WER, MOS, and
collapse labels at the speaker/style row level:

| Condition | Any-failure rows | Emotion misses | High-WER rows | Low-MOS rows | Mean failure score |
|-----------|------------------|----------------|---------------|--------------|--------------------|
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `58/99` | `35` | `31` | `14` | `1.9899` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `65/99` | `38` | `37` | `17` | `2.2929` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup_sad_enunc_guard` | `61/99` | `38` | `32` | `15` | `2.2929` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `66/99` | `38` | `36` | `17` | `2.2525` |

Per-style canonical recall for the decoder-prototype pilot:

| Style | Recall |
|-------|--------|
| `anger` | `2/11` |
| `disgust` | `0/11` |
| `fear` | `1/11` |
| `happy` | `6/11` |
| `neutral` | `9/11` |
| `sad` | `10/11` |

### Interpretation

1. **The decoder-aware training path is real and reproducible.** The repo now
   has a training objective that supervises decoded, style-controlled
   embeddings rather than only latent style geometry.
2. **The first version is not the new best checkpoint.** It improves recall
   over the original `combined` baseline, but falls below both expanded
   rare-supply readouts (`42.4%` vs `47.0%`).
3. **Novelty remains strong, but quality does not.** Novelty is slightly higher
   than the unguarded expanded run (`0.3008` vs `0.2995`), but WER is worse
   (`0.2863`) and files with any collapse increase to `27`.
4. **The objective changes which styles fail.** `happy` improves to `6/11`,
   but `fear` and `disgust` weaken, so the prototype target is not uniformly
   repairing canonical emotion control.
5. **Manual inference calibration still wins the current Pareto comparison.**
   The `sad/enunciated` guard keeps `47.0%` recall with better WER, better MOS,
   and fewer collapse files than the decoder-prototype pilot.
6. **Applying the same guard to the decoder-prototype checkpoint repairs only
   part of the problem.** WER improves from `0.2863` to `0.2592`, and MOS delta
   improves from `-0.2148` to `-0.1787`, but recall remains `42.4%` and files
   with any collapse increase from `27` to `28`.
7. **Lowering the prototype weight is not enough.** The `0.005` run improves
   novelty to `0.3032` and lowers any-collapse to `26`, but recall remains
   `42.4%` and WER remains much worse than the current guard (`0.2782` vs
   `0.2348`).
8. **Row-level failure mining confirms the next target styles.** Across the
   compared conditions, persistent failures concentrate in `disgust`
   (`43/44` failures, mean recall `0.0455`), `fear` (`39/44`, mean recall
   `0.1364`), and `anger` (`37/44`, mean recall `0.1591`). `enunciated`
   fails often too (`29/44`), but mostly through WER/MOS tradeoffs rather than
   canonical emotion recall.

### Implication

Finding 32 is a cautionary result, not a dead end. It confirms that decoded
embedding supervision can be added cleanly, and the guarded readout confirms
that inference calibration can still repair some decoder-prototype
naturalness/content damage. But naive prototype matching is not enough to learn
the quality-balanced repair that the manual strength guard discovered, and
simple scalar prototype-weight reduction is too blunt. The next training-side
work should be directly tied to generated-audio behavior. The failure-mining
artifact now makes that concrete: prioritize `emotion_miss + style_to_neutral`
rows for `anger` / `disgust` / `fear`, and do not use high-WER or very-low-MOS
rows as direct positive style targets unless the objective explicitly repairs
content.

Recommended listening artifacts:

- `results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.html`
- `results/listening_mixed_teacher_cvrare_decoder_proto_labeled_warmup.html`
- `results/listening_mixed_teacher_cvrare_decoder_proto_labeled_warmup_sad_enunc_guard.html`
- `results/listening_mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup.html`
- `results/eval_mixed_teacher_generated_audio_failure_mining.md`

---

## Open Questions

1. **What are the formal privacy guarantees?** We need to compute epsilon for each noise level and report privacy-utility curves.
2. ~~**Does style control generalize across source speakers?**~~ → **Answered in Finding 6.** Brightness generalizes (7/9 styles); F0 does not. Some speaker-style combinations collapse.
3. ~~**How do we evaluate emotion controllability?**~~ → **Answered in Finding 7.** emotion2vec Recall Rate + emo_sim (per EmoVoice) is the primary metric. Recall is 20% — training gap identified.
4. **Can CommonVoice-style broad speaker coverage improve recall once we mix the datasets together more carefully?** Mostly answered in Findings 30-32: yes, if rare pseudo-label supply is expanded and selected rows are preserved through speaker-first sampling. The expanded rare-supply mixed teacher reaches `47.0%` emotion recall and `0.2995` novelty gain, and the `sad/enunciated` strength guard keeps `47.0%` recall while improving WER/MOS. The decoder-prototype pilots preserve novelty but do not beat that guard, and generated-audio failure mining localizes the remaining hard styles to `disgust`, `fear`, and `anger`.
5. **Can we train age/gender and emotion knobs simultaneously?** CommonVoice has age/gender, CREMA-D has emotion. Can a single VAE learn all at once when each training stage only labels a subset? Unknown — Joe flagged this as an open research question.
6. **Can an independent speaker verifier confirm the novelty signal?** Finding 11 uses OpenVoice's native embedding space. The next step is an external speaker encoder / EER-style check.
7. **Can an adversary re-identify speakers from F0 alone?** If so, embedding-only DP is insufficient — motivates joint protection.
8. **What is the minimum speaker count for style learning?** We jumped from 3 to 91. Where's the threshold?
9. **Can we interpolate between styles?** E.g., 50% happy + 50% sad — does the output sound bittersweet?
10. **How to prevent collapses?** 9% of speaker-style combinations produce unintelligible output in the combined-only model, and the `cv500` CommonVoice run adds a second collapse mode: style washing back to neutral. CommonVoice finetune ablation shows that coarse whole-module freezing is not enough, CommonVoice objective ablation shows that simple scalar loss-weight schedules are not enough, CommonVoice rich-objective ablation shows that the first teacher/anchor supervision family still does not fix the neutral-collapse pattern, and CommonVoice partial-label pretraining shows that weak metadata / pseudo-label supervision mostly trades controllability for stronger intelligibility instead of escaping the collapse basin. Can we use better pseudo labels, stronger pretraining objectives, prototype/teacher-space targets, or detect/reject bad combinations?
11. **How stable are the ablation conclusions across seeds?** evaluation ablation matrix used a single deterministic seed and one validation corpus. We should add repeated-seed confidence intervals before freezing paper tables.
12. **What stronger mixed-data intervention, beyond schedule choice and first-pass pseudo-label filtering, can recover recall?** Finding 30 shows that stronger rare-class supply is the first intervention that materially recovers recall: `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup` reaches `47.0%` recall and `0.2995` novelty gain. Finding 31 shows that a narrow style-strength guard can preserve that recall while reducing the quality/content cost (`0.2348` mean styled WER, `-0.2081` MOS delta). Finding 32 shows that the first naive decoder-prototype objective does not learn that repair (`42.4%` recall, `0.2863` WER, `27` collapse files), the same guard repairs only WER/MOS (`0.2592`, `-0.1787`) while recall stays `42.4%`, and lowering the prototype weight to `0.005` still stays at `42.4%` recall with worse WER (`0.2782`). The remaining mixed-data question is now narrower: can a failure-conditioned objective improve `anger`/`disgust`/`fear` without sacrificing the current guard's WER/MOS?
13. **How high can style strength go before useful control turns into collapse?** The first non-Trump sweep (Finding 19) shows that `5.0` is not a hard ceiling: `7.5` is a reasonable stronger setting for `whisper` and `confused` on the current 4-speaker panel, while `10.0-12.5` push novelty higher at a clear WER/MOS cost. The open question is whether that pattern holds on a broader source panel and on the `combined` checkpoint, not just `mixed_quality_labeled_guarded`.

---

## Paper Framing

**Title candidates:**
1. "Controllable Differentially Private Voice Conversion"
2. "Expressive Voice Anonymization with Formal Privacy Guarantees"
3. "Style-Preserving Speaker Anonymization via Controllable VAE"

**Core argument (refined after April 16 call + April 16 evening message from Joe):** The problem we solve is **controllable speaker generation for voice-to-voice systems** — a more general problem than either VoicePrivacy (which preserves existing emotion) or TTS (which generates from text). A single controllable VAE enables multiple downstream applications:

1. **Identity change with controlled style** — anonymize a speaker while targeting a specific emotion (the demo we currently emphasize)
2. **Emotion change without identity change** — modify style latent dims while keeping the rest of the embedding fixed; same-sounding person, different mood
3. **Random speaker generation with control** — sample freely from the VAE to produce speakers that never existed, with specified properties
4. **Random speaker near an existing one** — sample from a neighborhood in latent space around a reference speaker

Privacy / DP noise is **one application** of use cases (3) and (4), not the paper's headline. Joe's assessment (April 16 evening): "I'm not sure about this, but our solution might end up being the state of the art for this more general thing."

**Key results to highlight:**
1. Negative result: not all VC embeddings encode style (ControlVC fails, OpenVoice works)
2. Speaker diversity is the critical training requirement (3 vs 91 speakers)
3. 9 styles controllable and acoustically verified
4. Style differences survive DP noise (privacy-utility tradeoff, not cliff)
5. Style control rescues intelligibility under heavy noise (bonus finding)
6. Style generalizes across diverse speakers (7/9 via brightness, collapses expected)
7. emotion2vec evaluation reveals training gap: 22.2% recall at current scale — motivates CommonVoice pre-training
8. Style control preserves intelligibility (WER sanity check): 6 of 9 styles have median WER ≤ 0.20 against same-speaker baseline; whisper is the only style with systemic loss
9. Predicted MOS confirms naturalness preservation for 6 of 9 styles; emo_sim + WER + MOS converge on the same three hardest styles (whisper, confused, anger) — cross-metric triangulation validates the signal
10. Validation-scale CommonVoice pre-training (`cv500`) is a mixed result: WER improves sharply, but emotion recall drops and the model collapses toward neutral. The CommonVoice direction remains promising, but the naive recipe is not paper-ready yet.
11. Native OpenVoice novelty evaluation confirms that the combined-only model produces genuinely shifted speaker identities relative to the source, while the `cv500` CommonVoice checkpoint largely collapses that shift back toward baseline voice identity.
12. The evaluation ablation matrix ablation matrix shows that the combined model remains the best overall tradeoff. Single-dataset and `cv500` conditions are more stable but collapse toward baseline identity and/or neutral emotion, while the naive random-latent baseline proves that raw novelty without target alignment is not the paper objective.
13. CommonVoice finetune ablation shows that simple gentler finetuning from the CommonVoice `cv500` init only partially recovers novelty and does not recover the combined model's controllability. The CommonVoice direction remains open, but the next gains likely require better objectives or larger-scale adaptation rather than just lighter finetuning.
14. CommonVoice objective ablation shows that simple objective reweighting during CommonVoice finetuning also does not recover recall or beat the best CommonVoice finetune ablation novelty recovery. The next CommonVoice gains likely require richer supervision, partial-label objectives, or larger-scale training rather than more scalar weight sweeps.
15. CommonVoice rich-objective ablation shows that the first richer teacher/anchor supervision family during combined finetuning still does not recover recall or beat the best CommonVoice finetune ablation novelty recovery. The next CommonVoice gains likely require richer supervision earlier in the pipeline, stronger pretraining objectives, partial-label CommonVoice training, or larger-scale follow-up once a stronger objective survives on the validation corpus.
16. CommonVoice partial-label pretraining shows that validation-scale weak metadata / pseudo-label supervision during CommonVoice pretraining still does not recover recall. Metadata-only supervision nudges novelty but not enough to beat the best earlier CommonVoice variants, while pseudo-style supervision greatly improves WER at the cost of much stronger identity collapse. The next CommonVoice gains likely require better pseudo-label quality, prototype- or teacher-space targets, or stronger pretraining curricula rather than the current weak-label recipe alone.
17. The mixed-data pseudolabel mix experiment shows that the first real CommonVoice + CREMA-D + Expresso run improves WER more than it improves control. Static balanced, CommonVoice warmup, and labeled-data finish all remain stuck at `16.7%` recall. `mixed_labeled_finish` achieves the best WER (`0.0606`), `mixed_static_balanced` achieves the best novelty (`0.0865`), but all three retain heavy style and identity collapse and remain far below the `combined` model on controllability.
18. Better mixed-data pseudo-label filtering plus stronger labeled-data protection can move recall a little: `mixed_quality_labeled_guarded` becomes the first mixed-data condition to improve recall above `16.7%`, reaching `18.2%`, but the gain comes with worse WER (`0.0978`) and weaker novelty (`0.0764`) than the best original mixed schedules.
19. A small non-Trump style-strength sweep shows that `5.0` is still the safest global default, `7.5` is a useful stronger compromise for styles like `whisper` and `confused`, and `10.0-12.5` behave more like high-novelty style-specific settings than new global defaults.
20. The first mixed-data pseudo-label teacher checkpoint family does not raise recall above `18.2%`, but `mixed_teacher_threshold_balanced` matches the best mixed-data recall while improving WER, MOS, novelty, and identity collapse versus `mixed_quality_labeled_guarded`. The next mixed-data gains likely require a stronger pseudo-label teacher or agreement rule rather than more tuning of the same single-teacher family.
21. A softer mapped-score agreement rule on the same teacher (`mixed_teacher_mapped015_balanced`) nudges novelty slightly higher (`0.0818`) but drops back to `16.7%` recall and gives back WER/MOS versus `mixed_teacher_threshold_balanced`. That makes the next mixed-data teacher step narrower: change the teacher or move to a richer multi-teacher rule, not just a softer same-teacher agreement heuristic.
22. A combined-VAE latent prototype teacher (`mixed_teacher_prototype_balanced`) gives broader CommonVoice pseudo-label coverage and the best mixed-teacher novelty so far (`0.0854`) while tying the best mixed-data recall (`18.2%`) and slightly reducing identity/mixed collapse. It still gives back WER/MOS versus `mixed_teacher_threshold_balanced`, so it is a promising coverage/novelty teacher but not yet the overall mixed-data reference.
23. Strong guardrails on the prototype teacher (`mixed_teacher_prototype_guarded`) preserve the `18.2%` recall tie and improve WER versus the unguarded prototype (`0.0920` vs `0.1009`), but erase the prototype novelty advantage (`0.0761` vs `0.0854`) and worsen identity collapse (`71`). The next teacher step should be multi-teacher or intermediate-guarded rather than simply weaker prototype supervision.
24. A hybrid emotion2vec + latent-prototype teacher (`mixed_teacher_hybrid_extra_balanced`) produces the best mixed-teacher novelty so far (`0.0860`) and slightly lowers files with any collapse (`65`), but recall falls back to `16.7%` and MOS worsens (`-0.1190`). The next mixed-data step should use richer style-space supervision, prototype distillation, or per-style curriculum rather than more hard pseudo-label arbitration.
25. Continuous style-space distillation on the hybrid teacher artifact (`mixed_teacher_hybrid_style_distill_balanced`) preserves the best mixed-teacher novelty (`0.0861`) and improves MOS/collapse versus hard hybrid labels (`-0.1072` MOS delta, `58` identity collapse, `63` files with any collapse), but recall remains `16.7%`. The next step is calibrated teacher supervision, not just replacing hard pseudo labels with one continuous loss.
26. A global style-teacher weight sweep (`0.10`, `0.25`, `0.50`) does not recover recall: all three style-distillation weights stay at `16.7%`. Higher weight (`0.50`) improves mean WER (`0.0821`) and identity/mixed collapse (`56` / `48`) but loses novelty/MOS, while `0.25` remains the best style-distillation novelty/MOS tradeoff. The next step should be per-style masks, confidence weighting, or curriculum, not another scalar teacher-weight tweak.
27. Target-dimension style-teacher masking with per-style row weights and confidence scaling still does not recover recall: `mixed_teacher_hybrid_style_distill_targetmask_balanced` remains at `16.7%`, preserves only a similar novelty signal (`0.0852`), and worsens WER/MOS versus the best global style-distillation setting. The next step should be diagnostic or curriculum-driven, not another latent-only mask/weight variant.
28. Per-style diagnostics localize the mixed-teacher failure: canonical emotion pseudo-labels often do not correspond to teacher-latent target dominance (`anger=0.0000`, `fear=0.0000`, `happy=0.1000` teacher target-top1 rates), rare classes have too little accepted supply (`anger=4`, `fear=4`), and `sad` shows that latent alignment can still decode to neutral-classified audio. The next step should be labeled-first curriculum, stronger rare-class supply, or decoder-aware style supervision.
29. A labeled-first curriculum protects CREMA-D/Expresso style axes before introducing CommonVoice teacher geometry and improves secondary axes (`0.0930` novelty gain, `54` identity-collapse files, `61` files with any collapse), but still remains at `16.7%` recall. The next step should move beyond schedule-only curriculum to stronger rare-class supply or decoder-aware/generated-audio style supervision.
30. Expanded rare-class CommonVoice supply is the first mixed-teacher intervention to produce a large recall jump: `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup` reaches `47.0%` emotion recall and `0.2995` novelty gain after preserving `anger=50` and `fear=50` selected CommonVoice pseudo rows. The limitation is quality/content cost (`0.2751` mean styled WER, `-0.2640` MOS delta), so the next paper-critical step is decoder-aware or generated-audio style supervision that keeps the recall gain while repairing WER/MOS.
31. Per-style strength calibration shows that part of the expanded rare-supply quality cost is repairable at inference time: `cvrare_sad_enunc_guard` keeps `47.0%` recall, keeps novelty above the combined baseline (`0.2726` vs `0.2599`), improves WER from `0.2751` to `0.2348`, improves MOS delta from `-0.2640` to `-0.2081`, and lowers files with any collapse from `25` to `20`. The result is useful for demos and paper tables, but should be presented as inference-side calibration rather than the final training method.
32. The decoder-prototype objective family is a cautionary baseline, not the new reference: the first run reaches `42.4%` recall and `0.3008` novelty but worsens WER/collapse, the guarded readout improves WER/MOS without recovering recall, and the lower-weight `0.005` run still stays at `42.4%` recall with `0.2782` WER. Generated-audio failure mining confirms the current guard still has the lowest row-level failure score and localizes persistent failures to `disgust`, `fear`, and `anger`; the next training-side step should be failure-conditioned, not another scalar prototype-weight sweep.

**Evaluation approach (per Joe, April 16 + EmoVoice paper):**
- **Primary:** emotion2vec Recall Rate + emo_sim (per EmoVoice pipeline) — measures whether generated outputs express the intended emotion
- **Secondary:** Speaker novelty in native OpenVoice embedding space — proof that the output speaker identity actually shifts away from the source
- **Tertiary:** Word error rate via Whisper — intelligibility sanity check
- **Quaternary:** Privacy/speaker verification (add as "we can also do this")
- Baseline for recall: random chance = ~11% (9 classes); current model = 20%; target: >50% for paper
