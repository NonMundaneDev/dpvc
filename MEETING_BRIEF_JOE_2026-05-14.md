# Meeting Brief for Joe Near

**Project:** DPVC / controllable voice-to-voice speaker generation
**Meeting date:** May 14, 2026
**Previous update meeting:** April 30, 2026
**Next planned update window:** around May 28, 2026
**Prepared for:** Stephen Oladele

## 1. Purpose of Today's Meeting

The goal of today's meeting is to give Joe a clear update on what changed since the April 30 call, align on the current best interpretation, and agree on what Stephen and Joe should each do before the next update meeting.

The main message is:

> Since the April 30 meeting, we tested Joe's recommended mixed-data direction seriously. The strongest result is now much better than before: expanded CommonVoice rare-class supply plus mixed-data teacher supervision reaches about `47%` emotion recall and beats the old combined baseline on novelty. But the remaining bottleneck is no longer just data scale; it is generated-audio calibration and perceptual quality. Before more training, Joe should listen to the five-row A/B review and tell us whether the stronger candidates actually sound better to a human listener.

## 2. What Joe Asked Us To Try Last Time

From the April 30 call, Joe's main guidance was:

1. Test a real mixed-data setup with **CommonVoice + CREMA-D + Expresso** instead of improving CommonVoice in isolation.
2. Use pretrained emotion models to pseudo-label CommonVoice, because that seemed like the most promising way to bootstrap emotion control.
3. Protect the small labeled datasets during training, because otherwise CommonVoice may dominate and wash out controllability.
4. Prioritize speaker breadth in CommonVoice sampling: at least one clip per speaker may matter more than many clips from fewer speakers.
5. Treat architecture changes as important, but secondary to getting the data mixture and supervision quality right first.
6. Do not treat `style_strength = 5.0` as a hard ceiling; Joe heard useful results at higher strengths on non-Trump examples.
7. Keep the research easier to review through a single findings/briefing document instead of expecting Joe to inspect every branch.

## 3. What We Completed Since April 30

### A. Repository and collaboration cleanup

- We moved ongoing controllable-VAE research out of upstream `main` after Joe clarified that upstream `main` should remain a stable reflection of the published work.
- The active research line now lives in the fork:
  - `NonMundaneDev/dpvc`
  - branch: `research/controllable-vae`
- This directly addresses the branch-sprawl / upstream-main concern.

### B. First mixed-data experiment: CommonVoice + CREMA-D + Expresso

We ran the first sampled mixed-data setup Joe recommended.

Result:

- Mixed-data schedules improved WER/content preservation more than controllability.
- Emotion recall stayed around `16.7%` in the first schedule family.
- Conclusion: Joe's mixed-data direction was right to test, but schedule choice alone was not enough.

### C. Improved pseudo-label filtering and labeled-data protection

We tightened CommonVoice pseudo-label filtering and protected labeled CREMA-D / Expresso rows more strongly.

Result:

- First small mixed-data recall bump: `18.2%`.
- Still not a clean win, because WER/novelty tradeoffs remained.
- Conclusion: better filtering helps, but we needed stronger rare-class supply and better teacher calibration.

### D. Teacher-family experiments and richer pseudo-label diagnostics

We built a reusable pseudo-label pipeline:

- score CommonVoice rows,
- filter pseudo labels,
- build mixed training artifacts,
- preserve reports/metadata inside artifacts,
- compare teacher policies.

We tested:

- emotion2vec teacher labels,
- latent-prototype teacher labels,
- hybrid teacher labels,
- style-space distillation,
- labeled-first curriculum,
- scalar teacher-weight sweeps,
- target masks.

Result:

- These experiments improved diagnostics and sometimes novelty/stability.
- They did not break the earlier low-recall ceiling until we expanded rare-class supply.

### E. Expanded CommonVoice rare-class supply

We expanded the local English CommonVoice corpus and used it to get more rare pseudo-labeled rows for hard classes like `anger` and `fear`.

Key local corpus:

- `/Users/steve/datasets/cv-corpus-21.0-2025-03-14/en`
- `40000` usable rows
- `20537` speakers

Key result:

- Expanded rare-supply mixed teacher reached about `47.0%` emotion recall.
- Novelty gain reached about `0.2995`, above the old combined-only reference (`0.2599`).
- This is the strongest controllability / novelty result so far.

Tradeoff:

- Mean styled WER worsened to about `0.2751`.
- MOS delta worsened to about `-0.2640`.
- So this is a big controllability result, but not yet the cleanest quality result.

### F. Per-style strength calibration

We tested whether the quality issue was partly an inference calibration issue.

Best current quality-balanced profile:

- `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard`

Result:

- Preserves `47.0%` recall.
- Improves WER from about `0.2751` to `0.2348`.
- Improves MOS delta from about `-0.2640` to `-0.2081`.
- Reduces files with any collapse from `25` to `20`.

Interpretation:

- Some damage is caused by over-driving certain style dimensions at inference time.
- Per-style strength profiles help, but they are still manual calibration, not the final training-side answer.

### G. Decoder-aware and anti-neutral training attempts

We tested training-side ideas meant to learn the quality repair automatically:

- decoder-prototype objective,
- lower-weight decoder-prototype variant,
- failure-conditioned target-dim supervision,
- anti-neutral prototype-margin objective.

Result:

- These are useful negative/cautionary results.
- None beats the current `sad/enunciated` guarded expanded checkpoint.
- They show that embedding-space proxy objectives can look reasonable while generated audio still fails.

Conclusion:

- The next useful calibration signal should come from actual generated audio, not just latent/embedding proxies.

### H. Generated-audio style-strength grid and A/B review

We ran a generated-audio strength grid for the hardest styles:

- `anger`
- `disgust`
- `fear`

Best cells:

| Style | Best cell | Result | Caveat |
|-------|-----------|--------|--------|
| `anger` | `anger_s10` | improves recall from `1/11` to `3/11` | moderate WER cost, needs listening |
| `fear` | `fear_s7p5` | improves recall from `3/11` to `6/11` | high WER/content risk |
| `disgust` | `disgust_s10` | improves novelty | does not improve recall and hurts MOS badly |

We then built:

- a full 33-row A/B review dashboard,
- an objective triage sheet,
- and a five-row priority-only perceptual review bundle for Joe.

## 4. Current Best Interpretation

The April 30 direction worked, but in a more nuanced way than simply "combine the datasets."

What is now clear:

1. **Rare-class CommonVoice pseudo-label supply was a real bottleneck.** Expanding it produced the first large recall jump.
2. **The system is now close to Joe's earlier paper-quality recall target.** `47%` is not above `50%`, but it is much closer than the old `16-26%` range.
3. **The remaining issue is quality/calibration, not just more raw data.** The model can now produce stronger emotion movement, but some outputs pay for it with WER/MOS damage.
4. **Generated-audio metrics are more trustworthy than latent proxy losses for the next step.** Several proxy objectives failed to beat manual/generated-audio calibration.
5. **We should not promote stronger style settings until human listening confirms them.** Metrics say `anger_s10` and `fear_s7p5` are interesting; Joe's ears should decide whether they are actually useful.

## 5. What To Discuss With Joe Today

### Main update to say out loud

> We tested the mixed-data direction you recommended. The first mixed runs improved WER but not recall. After improving pseudo-label filtering and expanding rare-class CommonVoice supply, the result changed substantially: the best mixed teacher now reaches about `47%` emotion recall and higher novelty than the old combined baseline. The remaining problem is that stronger control can hurt intelligibility/naturalness. So the next gate is perceptual: before training another model, we need to listen to the five most informative A/B rows and decide whether the metric-selected candidates actually sound better.

### Discussion point 1: Is the paper story now stronger?

Suggested framing:

- Earlier story: CommonVoice improves intelligibility but collapses controllability.
- Updated story: CommonVoice can recover controllability when rare pseudo-label supply is expanded and protected, but quality calibration becomes the next bottleneck.
- This is a stronger and more publishable story because it has both a positive result and a precise remaining limitation.

Ask Joe:

- Does he agree this is now the right framing?
- Would he present the `47%` result as the main positive result, with the `sad/enunciated` guard as the quality-balanced variant?

### Discussion point 2: Human listening before more training

Ask Joe to run the five-row review bundle.

Bundle created locally:

- `/Users/steve/UVM-plaid/dp-vc/results/joe_priority_review_bundle_2026-05-14.zip`

What Joe needs to answer:

- Does `anger_s10` sound better than the reference guard for anger?
- Does `fear_s7p5` sound better than the reference guard for fear?
- Are any of the metric gains actually perceptually unacceptable?
- Should either candidate become a style-specific preset?

Important instruction:

- The candidate should not win just because emotion2vec classified it better.
- The candidate should win only if it sounds more like the target style while remaining intelligible and natural.

### Discussion point 3: What should happen if Joe's listening disagrees with the metrics?

Proposed policy:

- If Joe prefers the candidate: promote it to a checked-in candidate style profile and rerun evaluation.
- If Joe prefers the reference: keep the grid as diagnostic and do not promote the stronger setting.
- If Joe says the candidate is more emotional but less usable: treat it as evidence for content repair / generated-audio-calibrated training.

### Discussion point 4: Next two-week research focus

Proposed order:

1. Finish the perceptual review and summarize ratings.
2. If ratings support `anger_s10` or `fear_s7p5`, create a candidate style-strength profile and run the full metric stack.
3. If ratings do not support them, move to fear/content repair and avoid preset promotion.
4. In parallel, improve Joe-facing reproducibility: metric guide, setup checklist, and artifact map.

### Discussion point 5: Reproducibility expectations

Be explicit:

- The five-row listening bundle is self-contained and does not require Joe to rebuild OpenVoice outputs.
- Full model reproduction is still heavier because generated WAVs and checkpoints are local artifacts, while the repo tracks scripts/results/docs.
- Before the next meeting, we should decide whether to package the key generated audio artifacts more formally, possibly with a release asset, Git LFS, or a small artifact bundle.

## 6. Concrete Tasks Before the Next Update Meeting

### Stephen's tasks before May 28

1. Send Joe the self-contained five-row listening bundle.
2. Keep the fork branch `research/controllable-vae` as the canonical active research branch.
3. After Joe fills the ratings CSV, summarize the perceptual ratings.
4. Based on ratings, either:
   - create and evaluate a checked-in style-strength profile, or
   - keep the grid diagnostic and move to content repair.
5. Add a short Joe-facing metric guide explaining:
   - emotion recall,
   - emo_sim,
   - novelty,
   - WER,
   - MOS,
   - collapse labels.
6. Add a reproducibility checklist that separates:
   - quick listening review,
   - rerunning metrics,
   - regenerating audio,
   - retraining models.
7. Decide how to package generated audio artifacts so Joe can review without depending on Steve-local output directories.

### Joe's tasks before May 28

1. Unzip and run the five-row perceptual review bundle.
2. Fill the priority ratings CSV.
3. Send the filled CSV back to Stephen.
4. Give qualitative notes on whether:
   - `anger_s10` should become a candidate preset,
   - `fear_s7p5` is useful despite WER risk,
   - `disgust_s10` should stay diagnostic,
   - the reference guard still sounds best overall.
5. Tell Stephen whether the current paper framing makes sense:
   - mixed-data rare pseudo-label supply unlocks controllability,
   - generated-audio calibration is now the main bottleneck.
6. Optional: tell Stephen whether the lab machine / cluster access is worth using for the next larger run, or whether current local resources are enough for now.

## 7. Proposed Two-Week Timeline

### May 14-16

- Stephen sends Joe the listening bundle.
- Joe confirms he can open the dashboard and play audio.
- Stephen writes the metric-reading guide.

### May 17-22

- Joe listens and fills the ratings CSV when available.
- Stephen prepares the rating summarizer path and reproducibility checklist.
- If ratings arrive early, Stephen creates the candidate style profile and starts evaluation.

### May 23-27

- Stephen summarizes perceptual ratings.
- Stephen either evaluates the new style preset or pivots to content repair.
- Stephen prepares the next brief for the May 28 update.

### May 28 meeting

Decision point:

- Promote a style-specific preset, or
- Treat the grid as diagnostic and prioritize generated-audio/content repair.

## 8. Suggested Meeting Agenda

1. Reconfirm collaboration setup: upstream `main` is stable; research continues on the fork.
2. Summarize what was completed since April 30.
3. Walk through the main result: expanded rare-supply mixed teacher reaches about `47%` recall.
4. Explain the quality tradeoff and why manual strength calibration helped.
5. Explain why decoder/anti-neutral proxy objectives did not beat the current guard.
6. Ask Joe to run the five-row perceptual review bundle.
7. Agree on the May 28 deliverables.
8. Ask if Joe wants any change in paper framing or artifact packaging.

## 9. One-Minute Version

> Since our April 30 meeting, we tested the main thing you recommended: combining CommonVoice with CREMA-D and Expresso using pseudo-labeled CommonVoice and explicit mixture control. The first versions mostly improved WER but not recall. After improving pseudo-label filtering and expanding rare-class CommonVoice supply, we got a much stronger result: about `47%` emotion recall and novelty above the old combined baseline. The tradeoff is quality: WER and MOS get worse unless we use a per-style guard. We tried several training-side proxy fixes, but none beat the current guarded result. So the next best step is perceptual review. I have a self-contained five-row A/B listening bundle for you. If the stronger candidates sound genuinely better, we promote them to style-specific presets; if not, we keep them diagnostic and work on content repair.

## 10. What Not To Overclaim

Avoid saying:

- "We solved controllable emotion."
- "Higher style strength is better globally."
- "The candidate presets are ready."
- "CommonVoice pretraining alone fixed the problem."

Safer wording:

- "We now have a much stronger mixed-data result, but quality calibration is the next bottleneck."
- "The generated-audio grid found promising candidate settings for anger and fear, but they need human listening before promotion."
- "The current guarded expanded checkpoint is the best quality-balanced reference."

## 11. Files and Artifacts To Mention

Main repo/fork:

- `https://github.com/NonMundaneDev/dpvc/tree/research/controllable-vae`

Main documents:

- `FINDINGS.md`
- `WORKLOG.md`
- `IMPLEMENTATION_PLAN_mixed-data-pseudolabel-teacher.md`

Listening bundle:

- `results/joe_priority_review_bundle_2026-05-14.zip`

Priority review artifacts inside the bundle:

- `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority.html`
- `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority_ratings.csv`
- `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority.md`

Key current reference condition:

- `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard`

Key candidate settings:

- `anger_s10`
- `fear_s7p5`

## 12. Bottom Line

For today's meeting, the clean ask is:

> Joe, please listen to the five-row A/B bundle before the next meeting and send back the ratings CSV. That will tell us whether to promote the metric-selected stronger style settings or treat them as diagnostic and move to generated-audio/content repair.
