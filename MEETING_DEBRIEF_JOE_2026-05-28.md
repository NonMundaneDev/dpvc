# Meeting Debrief: Joe Near

**Date:** 2026-05-28
**Source of truth:** `/Users/steve/Downloads/Stephen _ Joe voice privacy project Meeting Transcript.md`
**Secondary source checked:** `/Users/steve/Downloads/Stephen _ Joe voice privacy project AI-Generated Meeting Summary Notes.md`
**Branch:** `research/generated-audio-calibrated-training`

## 1. Executive Summary

Joe's main guidance is that the system is basically working and the project
should now pivot from open-ended model improvement to paper-facing evaluation,
justification, and simplification.

The next work should not be another attempt to make `anger` or `disgust`
stronger. Instead, it should answer paper-critical evaluation questions:

1. Which controls are actually perceptible and separable in the training data?
2. Which controls should be in the headline paper/demo story?
3. Can gender control be made clear enough, since Joe considers it important?
4. Can age be framed as a low-priority broad-bucket classification task?
5. Can accent be explicitly excluded because the current OpenVoice speaker
   embedding path is not expected to control it?

Joe's highest-level instruction was: stop asking "what can I add to make it
better?" and start asking "what do we need to justify in the paper?"

## 2. Joe's Main Technical Conclusions

### A. The system is basically working

Joe said the thing is basically working the way we want it to. His advice was
not to focus on making it better, but to focus on writing the paper and making
sure the evaluation is solid.

Implication:

- The current reference result is strong enough to become the paper/evaluation
  anchor.
- New experiments should be small, bounded, and directly tied to paper
  justification.
- Avoid adding complexity unless it answers a reviewer-facing question.

### B. Do not keep trying to force `anger` and `disgust`

Joe agreed with the direction that weak/subtle emotions may simply not contain
enough perceptual training signal. His position is that if the training data is
not perceptually separable, then the classifier cannot detect it and the VAE
cannot reasonably learn it.

Implication:

- `disgust` should no longer be treated as a hard-style repair target unless
  stronger perceptual examples are added.
- `anger` may have some signal, but the next step is not "increase strength";
  it is to decide whether `anger` belongs among the clearly separable controls.
- The emotion selection problem becomes an evaluation-design problem, not a
  model-capacity problem.

### C. Select top emotions by training-data separability

Joe suggested running the emotion classifier on the training data itself and
using per-label F1 or similar separability metrics to justify which emotions are
included in the headline story.

His expected outcome:

- `whisper` is likely easy/high-F1.
- `disgust` is likely low-F1 even on the training data.
- The top three or top few emotions should be selected based on evidence, not
  preference.

Implication:

- Build a training-data style separability audit.
- Report per-style F1/confusion on CREMA-D/Expresso training examples.
- Use that table to justify the paper/demo control set.

### D. Gender is important and should be repaired if unclear

Joe said gender is important because it is usually easy to perceive and because
simultaneous gender-plus-emotion control is a meaningful demo.

He also noted that his earlier CommonVoice-only gender experiment sounded
clearly controllable, even without classifier evaluation.

Likely explanations if our current gender control is weak:

- We used too little CommonVoice gender-labeled data.
- The combined emotion+gender setting may interfere with perceptual clarity.
- The latent dimensionality may be too small once more controllable features are
  added.

Implication:

- Run a narrow gender-focused follow-up, not a broad metadata sweep.
- Extract many more gender-known CommonVoice rows, not just a tiny sampled
  subset.
- Test gender alone and gender plus a small number of strong emotions.
- Try a bounded latent-dimension comparison if needed.

### E. Accent should be explicitly out of scope

Joe said accent should not be touched in the current setup. His reasoning is
that systems like OpenVoice may encode accent in the content representation,
not the speaker embedding. Since our VAE manipulates speaker embeddings, we
should not expect it to control accent.

Implication:

- Do not spend time trying to repair accent control in this branch.
- Docs should explicitly say accent is out of scope for the current embedding
  path.
- This is not a weakness reviewers will be surprised by; it is a known
  disentanglement issue in speech systems.

### F. Age is lowest priority and should be broad buckets only

Joe said age would be nice if it works, but it is lowest priority. Fine-grained
age control is not perceptually realistic; broad buckets may be more defensible.

Possible bucket framing:

- young / child-like, if available in data;
- broad adult / middle range;
- older / elderly.

Implication:

- Do not treat age as a continuous scalar claim.
- If age stays in scope, make it a broad classification experiment.
- Stop quickly if bucket separability is weak.

## 3. Updated Research Direction

The next branch should be paper/evaluation-oriented:

**Recommended branch:** `research/control-selection-evaluation`

Primary goals:

1. Run a training-data separability audit for emotion/style labels.
2. Use the audit to select the top emotion/style controls for the paper/demo.
3. Repair or re-test gender control with more gender-known CommonVoice data.
4. Explicitly move accent out of scope.
5. Reframe age as optional broad buckets and keep it low priority.
6. Start tightening paper-methods documentation around the current strongest
   checkpoint instead of adding more model complexity.

## 4. What Should Change In Repo Docs

### WORKLOG.md

Update immediately:

- Add the May 28 meeting direction update.
- Move `disgust` repair from `[NOW]` to blocked/deprioritized unless stronger
  perceptual training examples are added.
- Promote training-data separability audit to `[NOW]`.
- Promote gender-focused follow-up to `[NOW]` or `[SOON]`.
- Mark accent control as out of scope for current OpenVoice speaker-embedding
  path.
- Reframe age as low-priority broad-bucket classification.
- Add paper simplification/evaluation justification as the dominant current
  task.

### FINDINGS.md

Review, but do not add a new empirical finding unless we run the audit. The
meeting itself is guidance, not a new model result.

Recommended update:

- Add a short "May 28 Meeting Alignment" note under paper framing or open
  questions, clearly labeled as interpretation/guidance rather than a measured
  finding.
- Update Open Question 5 to reflect Joe's gender priority and accent exclusion.
- Update the paper framing to say controls should be selected by training-data
  separability and perceptual clarity.

### results/README.md

No immediate result artifact update is required unless we add the debrief as a
paper-facing artifact. The next result artifact should be the separability audit
CSV/Markdown.

### Meeting docs

- Keep `ASYNC_FEEDBACK_JOE_2026-05-28.md` as the pre-meeting Teams review.
- Keep this file as the actual post-meeting debrief.
- Future Joe briefs must include anticipated questions and short answers.

## 5. Feedback For Stephen

### What you did right

- You sent Joe a focused review bundle before the meeting, which made the
  discussion concrete.
- You did not overclaim the `disgust` result after Joe disagreed with your first
  listening read.
- You correctly proposed that weak labels should be audited instead of forced.
- You asked Joe to choose the next two-week direction instead of continuing the
  rabbit hole alone.
- You surfaced the real scope question: paper now versus more experiments.
- You asked whether age/gender/emotion should all remain active, which led Joe
  to give a useful priority ordering.

### What you should improve

- Lead with the decision you need. The first 10 minutes contained too many
  experiment threads before Joe got a crisp question. Next time start with:
  "I need help choosing between paper/evaluation, gender repair, or another
  emotion repair. My recommendation is paper/evaluation plus bounded gender."
- Keep one clean reference model name ready. The phrase around the current
  `sad/enunciated` guard became hard to follow. Use a short alias like
  "current reference guard" and then give the checkpoint name only if Joe asks.
- Do not explain every branch unless it changes the decision. Joe did not need
  the full chain of generated-audio repair details during the meeting.
- Avoid saying a metric is for the wrong thing. ECAPA speaker verification is
  for identity shift / source similarity, not intelligibility.
- Do not treat every weak result as something to fix. Joe is steering you to
  choose defensible controls and justify exclusions.
- When discussing data, distinguish clearly between a data shortage, label
  ambiguity, and architecture capacity. Joe mentioned all three as different
  possibilities for gender/emotion behavior.
- If the AI summary is used, verify it against the transcript. It incorrectly
  implies Joe should share the ECAPA paper; the transcript says Stephen offered
  to send it to Joe.

### Knowledge gaps to close before the next meeting

- What emotion2vec per-class F1/confusion looks like on the actual training
  examples, not generated outputs.
- How many CommonVoice rows have reliable gender metadata locally, and how many
  can be extracted without hardcoded paths.
- Whether gender control works alone before testing gender plus emotion.
- Whether the current latent dimension is constraining multi-control behavior.
- Which style controls should be selected for the paper based on evidence.
- How to explain accent as a content/speaker disentanglement limitation.

## 6. Next Two-Week Plan

### Workstream 1: Training-data separability audit

Purpose:

- Justify which emotion/style controls belong in the paper.

Tasks:

- Run emotion2vec on CREMA-D/Expresso training clips.
- Compute per-label precision, recall, F1, support, and confusion matrix.
- Rank labels by separability.
- Recommend top controls for paper/demo.
- Mark weak labels as limitations/future work.

Validation:

- CSV and Markdown summary exist.
- The selected top controls are justified by training-data metrics.
- The summary explicitly discusses `disgust`, `anger`, `whisper`, and other
  candidate controls.

### Workstream 2: Gender-focused CommonVoice follow-up

Purpose:

- Determine whether gender control can be made perceptually clear enough for
  the paper/demo.

Tasks:

- Extract a larger local CommonVoice subset with known gender metadata.
- Train or evaluate gender-only control first.
- Then test gender plus top separable emotion controls.
- Compare current latent dimensionality against one bounded larger latent size
  if needed.

Validation:

- Gender-known row counts are reported.
- Listening panel exists for gender-only and gender-plus-style examples.
- A simple classifier or separability probe reports whether gender is preserved
  in embeddings/latents.
- If gender remains weak, the failure is categorized as data, interaction, or
  capacity.

### Workstream 3: Paper simplification and methods scaffold

Purpose:

- Convert the project from experiment accumulation into a paper.

Tasks:

- Create/update a paper outline around the current reference checkpoint.
- Add a control-selection rationale section.
- Add limitations: weak labels, accent out of scope, age low priority, formal DP
  accounting still open.
- Keep the method story simple: OpenVoice embeddings, controllable VAE,
  selected controls, evaluation stack.

Validation:

- Paper outline has a clear claim and evaluation table plan.
- Each headline control has a justification.
- Each excluded control has a defensible reason.

## 7. Future Meeting Communication Template

Use this structure for Joe:

1. **Decision needed today:** one sentence.
2. **Current best result:** one checkpoint alias plus 2-3 numbers.
3. **What changed since last time:** maximum three bullets.
4. **My interpretation:** one paragraph.
5. **What I need from you:** 2-4 direct questions.
6. **Proposed next two weeks:** one recommended path plus one fallback.
7. **Anticipated questions:** short answers prepared in advance.

Example opening:

> The system appears to work well enough to pivot toward the paper. I need your
> help deciding which controls are defensible headline claims. My recommendation
> is: select top emotions using training-data separability, fix gender with more
> gender-known CommonVoice data if possible, exclude accent, and keep age as a
> low-priority broad-bucket test.

## 8. Notes To Preserve For Project Memory

- Future meeting briefs must include anticipated questions and concise answers.
- Teach Stephen the difference between model improvement, evaluation
  justification, and paper framing before meetings.
- For Joe-facing updates, lead with the decision needed, not the full branch
  history.
- Use "current reference guard" as the short name before giving long checkpoint
  IDs.
- Remind Stephen that ECAPA validates identity shift, not intelligibility.
- Prioritize paper/evaluation simplification unless Joe explicitly asks for
  more model complexity.
