# Post-Meeting Debrief: Joe Near Update

**Meeting date:** May 14, 2026
**Transcript source:** `/Users/steve/Downloads/Stephen _ Joe voice privacy project transcripts.md`
**Purpose:** capture research direction, communication feedback, corrected explanations, and future meeting-prep rules.

## 1. Executive Summary

The meeting went well strategically. Joe was not discouraged by imperfect emotion control; he explicitly reframed the goal away from beating emotion-control systems and toward demonstrating controllable speaker generation with multiple controllable attributes. His strongest new guidance was:

- do not over-focus on emotion as the only feature;
- add CommonVoice age/gender controls because those labels already exist;
- listen to the five-row A/B bundle and compare human perception with metric-selected candidates;
- clarify what `identity collapse` means;
- if listening and age/gender controls look good, the project may be close to paper-writing mode;
- next major writing work should document architecture, training data, and evaluation strategy clearly enough to justify the claims.

## 2. Research Takeaways From Joe

### 2.1 Emotion does not need to be perfect

Joe said emotion control does not need to be world-class. The contribution is not “best emotion conversion”; it is controllable speaker generation / anonymization where emotion is one controllable feature.

Implication:

- The paper can use strong, perceptually obvious styles as the headline examples.
- `whisper` remains especially important because it is clear and audible.
- `anger` may also be a good target if the A/B review confirms it.
- Weak styles like `enunciated` may not be worth over-optimizing, especially if even the training data has subtle differences.

### 2.2 Add age and gender controls next

Joe explicitly asked whether age/gender labels from CommonVoice had been incorporated. They have not.

Implication:

- This is now a top-priority experiment.
- It better matches the broader claim: “we can control arbitrary speaker attributes when labels exist,” not just emotion.
- It also helps the paper because age/gender are metadata labels from the broad CommonVoice speaker corpus, while emotion/style comes from CREMA-D/Expresso plus pseudo labels.

### 2.3 Clarify identity collapse

Joe asked how identity collapse was measured. The answer in the meeting was too uncertain and mixed up WER, emotion, and identity.

Correct definition in the current repo:

- `content_collapse`: high WER, currently `WER >= 0.8`.
- `style_collapse_to_neutral`: target style is non-neutral, but emotion2vec predicts `neutral`.
- `identity_collapse_to_baseline`: OpenVoice novelty gain versus baseline is very low, currently `novelty_gain_vs_baseline <= 0.05`.
- `mixed_collapse`: two or more collapse axes fire for the same generated file.

Important correction:

- WER is about content/intelligibility, not identity.
- Emotion recall is about style match, not identity.
- Identity collapse is currently a proxy for “the generated speaker did not move meaningfully away from baseline/source behavior in OpenVoice embedding space.”

### 2.4 The word “collapse” can sound too catastrophic

Joe interpreted “collapse” as possibly meaning the output is garbage/unintelligible. He clarified that what we often mean is less severe: the model still produces audio, but it may sound like a default/baseline speaker or neutral style.

Future language:

- Use “style collapsed to neutral” instead of generic “collapse.”
- Use “identity collapsed toward baseline” instead of generic “identity collapse.”
- Use “content collapse / high WER” only when intelligibility genuinely fails.

### 2.5 Paper-writing may be close

Joe said that if the examples sound good and the added age/gender controls work, the next step may be documenting:

- exact architecture,
- exact training data and schedule,
- exact evaluation strategy,
- why the evaluation strategy is justified.

Implication:

- We should start organizing the paper skeleton and method/evaluation description soon, not only keep running experiments.

## 3. What Stephen Did Right

1. **You connected the update to Joe’s April 30 guidance.** You correctly framed the work as the mixed-data experiment Joe recommended.
2. **You gave the important numerical trend.** You communicated that the initial mixed run stayed near `16%`, filtering got to `18%`, and rare-supply/mixed teacher work reached the `40%+` range.
3. **You emphasized perceptual review.** This was exactly right. Joe agreed to listen, and the project needs human judgment before promoting presets.
4. **You were transparent about limitations.** You did not pretend the system was perfect; you surfaced intelligibility and quality issues.
5. **You responded well to Joe’s reframing.** When Joe said emotion does not need to be perfect and age/gender matters, you accepted the shift and aligned on the next two-week direction.
6. **You handled Ivoline’s conceptual questions collaboratively.** You tried to explain pseudo-labeling and filtering in accessible terms, and Joe helped sharpen the explanation.

## 4. What To Improve Next Time

### 4.1 Start with the conclusion first

The first 8 minutes were heavy with details before the crisp takeaway landed.

Better opening:

> “We tested your mixed-data recommendation. The short version is: the first mixed schedules did not improve recall, but expanding rare CommonVoice pseudo-label supply changed the result substantially. We now get about `47%` recall, but quality/intelligibility is the bottleneck. So I sent a five-row A/B bundle for perceptual review before we do more training.”

Then give details.

### 4.2 Keep metric definitions crisp

The identity-collapse answer was the weakest moment. You correctly admitted uncertainty, but the answer mixed identity, WER, and emotion.

Memorize this:

> “In our current tables, identity collapse means low speaker novelty gain versus the baseline: the generated output did not move far enough in OpenVoice speaker-embedding space. WER is separate; that measures content/intelligibility collapse. Emotion recall is also separate; that measures whether the output was classified as the target style.”

### 4.3 Do not call filtering “cherry picking” or “hope for the best”

That wording undermines rigor.

Better wording:

> “Filtering is a confidence-gated weak-label selection step. We score CommonVoice clips with a pretrained emotion model, keep rows that pass per-style confidence thresholds or target/cap rules, and record accepted/rejected counts so the data-selection process is auditable.”

### 4.4 Do not say “EmoVoice classifier” for the filtering model

The model used in our pipeline is `emotion2vec_plus_large` / `iic/emotion2vec_plus_large`. EmoVoice is the evaluation paper/pipeline we were aligning with, not the specific classifier name.

Correct wording:

> “We use emotion2vec_plus_large, the model family used in the EmoVoice-style evaluation setup, to score emotion labels.”

### 4.5 Separate identity preservation from anonymization

At several points, “retaining identity” and “changing identity/anonymization” blurred together.

Better framing:

- For anonymization / new speaker generation: we want identity shift / novelty.
- For content preservation: we want same words / low WER.
- For controlled attributes: we want style/age/gender to move in intended directions.
- For some demos, we may also want “same speaker but different emotion,” but that is a different mode from anonymization.

### 4.6 Be careful with “official CommonVoice” vs “Hugging Face shards”

You mentioned using Hugging Face shards and not listening to all of them. That is honest, but it can raise reproducibility/data-quality questions.

Better wording:

> “For the expanded run, we used a local CommonVoice English subset prepared into `validated.tsv + clips/`, and the scripts record usable rows, speakers, missing/unreadable audio, and selected pseudo-label counts. We did not manually listen to all CommonVoice clips; quality control is through metadata validation, readable-audio checks, and downstream perceptual review.”

## 5. Knowledge Gaps To Close

1. **Collapse taxonomy.** Know exactly how content/style/identity/mixed collapse are computed.
2. **Pseudo-label pipeline.** Be able to explain score -> threshold/filter -> cap/target -> selected counts -> mixed training artifact.
3. **Model names.** Say `emotion2vec_plus_large`, not “EmoVoice classifier.”
4. **DP mechanism.** Explain that DP comes from calibrated noise in the latent/speaker representation; formal epsilon/privacy accounting remains an open paper task.
5. **Age/gender controls.** Understand CommonVoice metadata quality and how age/gender labels map into controllable latent dimensions.
6. **Paper claim.** The claim is broader controllable speaker generation / anonymization, not perfect emotion conversion.
7. **Evaluation justification.** Be ready to explain why we use emotion recall, WER, MOS, novelty, collapse taxonomy, and perceptual listening.

## 6. Future Meeting-Prep Rule For Codex

For every future Joe-facing meeting brief, include an **Anticipated Questions and Answers** section.

Minimum Q&A set:

1. What exactly changed since the last meeting?
2. What is the main result in one sentence?
3. What is the strongest number, and what is its caveat?
4. What does each metric mean?
5. What does “collapse” mean here?
6. How was the model trained?
7. What data was used?
8. What is pseudo-labeling/filtering?
9. Is this supervised, unsupervised, or weakly supervised?
10. Where does differential privacy enter?
11. What should Joe do before the next meeting?
12. What are we not claiming yet?

## 7. Updated Research Plan

### Immediate: finish perceptual review

Joe should run the five-row review bundle and fill the ratings CSV.

Decision after ratings:

- If `anger_s10` or `fear_s7p5` wins perceptually, create a checked-in style-strength profile and evaluate it.
- If not, keep the grid as diagnostic and move to generated-audio/content repair.

### Next implementation focus: age/gender controls

New branch/milestone title:

- **CommonVoice Metadata Controls: Age and Gender**

Goal:

- Use CommonVoice age/gender metadata as additional controllable attributes alongside emotion/style.

Expected work:

1. Audit CommonVoice metadata coverage and label distributions.
2. Decide label mapping for age buckets and gender categories.
3. Extend mixed embedding artifacts to preserve age/gender labels when present.
4. Extend controllable VAE training to supervise additional dims for age/gender.
5. Add inference controls for age/gender.
6. Evaluate whether controls move generated outputs perceptually and/or through a classifier/proxy.
7. Check whether age/gender controls interfere with emotion controls.

### Parallel documentation focus: paper-readiness

Start drafting durable docs for:

- architecture,
- data and training schedule,
- evaluation strategy and justification,
- metric definitions,
- perceptual-review protocol.

Joe explicitly said that once the current outputs and age/gender controls are acceptable, the next step may be writing the paper.

## 8. Docs To Update

### `WORKLOG.md`

Add May 14 meeting alignment:

- Joe says emotion does not need to be perfect.
- Add age/gender controls as a `NOW` item.
- Clarify identity-collapse explanation task.
- Add paper architecture/evaluation documentation as a near-term task.
- Add future meeting-brief Q&A requirement.

### `FINDINGS.md`

Do not add a new scientific finding yet. This meeting produced direction and interpretation, not a verified result.

Update later only after:

- Joe/Stephen perceptual ratings are summarized, or
- age/gender controls produce evaluated results.

### `README.md`

Update the current next queue after we implement the worklog changes:

- perceptual review,
- age/gender metadata controls,
- metric guide / paper documentation.

### New recommended docs

1. `MEETING_DEBRIEF_JOE_2026-05-14.md` — this file.
2. `docs/research_metrics.md` — plain-English metric and collapse guide.
3. `docs/paper_method_and_evaluation_plan.md` — architecture/training/evaluation description for paper drafting.

## 9. Correct Short Answers For Future Meetings

### What is pseudo-labeling?

> CommonVoice does not have emotion labels, so we use a pretrained emotion classifier to assign approximate emotion labels to CommonVoice clips. These are pseudo-labels because they are model-generated, not human ground truth.

### What is pseudo-label filtering?

> Filtering is the step where we keep only pseudo-labeled rows that pass confidence, class-balance, target-count, or cap rules. It prevents noisy or overrepresented labels, especially neutral/sad, from dominating training.

### Is this supervised or unsupervised?

> It is mixed. CREMA-D/Expresso are supervised. CommonVoice reconstruction is unsupervised. CommonVoice pseudo-label training is weakly/semi-supervised because we train on model-generated labels.

### What is identity collapse?

> In our current metric tables, identity collapse means the generated output has very low novelty gain versus the baseline in OpenVoice speaker-embedding space. It is not WER. WER measures content/intelligibility.

### Are we trying to preserve identity or anonymize it?

> Depends on the mode. For anonymized/new-speaker generation, we want identity shift with controlled attributes. For an emotion-change demo, we may keep identity closer and change style only. The VAE supports both kinds of experiments, but they should be reported separately.

### What is the core paper claim now?

> We are building controllable speaker generation for voice-to-voice systems under a DP/anonymization framing. Emotion is one attribute, not the whole claim. The broader claim is that labeled attributes can become controllable dimensions of the generated speaker representation.

## 10. Bottom Line

The meeting upgraded the next plan. Before this call, the immediate queue was mostly “perceptual review before more training.” After this call, the queue is:

1. finish perceptual review;
2. add age/gender metadata controls;
3. clarify metric/collapse definitions;
4. start paper-method/evaluation documentation;
5. keep emotion optimization pragmatic rather than perfectionist.
