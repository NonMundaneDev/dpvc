# Canonical Evidence and Demo Packet

**Date:** 2026-05-26
**Branch:** `research/evidence-demo-packet`
**Base research line:** `research/controllable-vae`

## Top-Line Answer

We already have a substantial controllable-voice result. The strongest current
claim is not that every emotion control is perfect. The claim is that the
OpenVoice controllable VAE can generate/anonymize speakers while moving labeled
perceptual attributes in controlled directions, with several styles that are
perceptually clear and measurable.

The latest Joe listening review should be read narrowly: the higher-strength
metric-selected `anger_s10` / `fear_s7p5` candidates did not sound better than
the current guarded reference. That does not invalidate the core system. It
means we should keep the strength grid diagnostic and avoid promoting those
specific candidates as presets.

## What To Listen To First

Serve the repository root locally:

```bash
cd /Users/steve/UVM-plaid/dp-vc
python3 -m http.server 8000
```

Then open the canonical listening index:

```text
http://localhost:8000/results/listening_evidence_demo_index.html
```

The index points to the full reports, but starts with a small embedded quick
listen panel for two source speakers and four useful controls: `anger`,
`happy`, `sad`, and `whisper`.

## Evidence Tiers

### Tier 1: Current Quality-Balanced Demo

Use this as the current default demo surface:

- `results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.html`

Why it matters:

- It preserves the expanded rare-supply recall result at about `47%`.
- It improves content/naturalness relative to the unguarded expanded checkpoint.
- It is the safest current reference for listening demos.

### Tier 2: Strongest High-Novelty / High-Recall Checkpoint

Use this to explain the scientific tradeoff:

- `results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup.html`

Why it matters:

- This is the stronger expanded rare-supply result: about `47%` emotion recall
  and `0.2995` novelty gain.
- It demonstrates that broader CommonVoice pseudo-label supply can recover a
  large part of the recall gap.
- It has a content/naturalness cost, which is why the guarded readout remains
  the demo reference.

### Tier 3: Joe's Perceptual Gate on Metric-Selected Strength Candidates

Use these to explain what we should *not* overclaim:

- `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority.html`
- `results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority_joe_2026-05-19.md`

Joe's result:

- `4/5` rows sounded identical to the reference guard.
- `1/5` rows preferred the reference because the candidate had an unnatural
  pitch change.
- `0/5` candidate rows won perceptually.

Interpretation:

- The strength grid is useful as an analysis tool.
- `anger_s10` and `fear_s7p5` should not become checked-in presets yet.
- Objective classifier gains are not enough unless human listening agrees.

### Tier 4: Historical Strong Evidence

Use these findings to explain why this research line is substantial:

- `FINDINGS.md` Finding 1: OpenVoice is the right controllable path; ControlVC
  is a useful DP baseline but a negative style-control result.
- `FINDINGS.md` Finding 2 / 4 / 6: OpenVoice plus CREMA-D/Expresso produces
  perceptibly distinct controls, with `whisper` especially strong and robust.
- `FINDINGS.md` Finding 30 / 31: expanded rare-supply mixed teacher reaches
  paper-relevant recall while exposing the quality tradeoff.
- `FINDINGS.md` Finding 35: generated-audio strength reranking is diagnostic,
  not a safe default.

## What We Can Say Now

Good wording:

> We have a controllable speaker-generation/anonymization system with clear
> audible controls for several styles, especially whisper, and a quantitative
> mixed-data result around 47% emotion recall. The newest listening review says
> metric-selected strength increases are not perceptually better enough to
> promote yet, so the next research move is to add CommonVoice age/gender
> controls and show the framework supports more than emotion.

Avoid saying:

- The strength grid solved anger/fear.
- Joe rejected the whole system.
- The model is only an emotion converter.
- The next step is simply more strength or more emotion tuning.

## Next Research Step

The next implementation branch should be CommonVoice metadata controls:

- Suggested branch: `research/commonvoice-metadata-controls`
- Goal: add age/gender controls from CommonVoice metadata while preserving the
  existing emotion/style controls.
- Why: this directly tests Joe's broader paper framing: controllable speaker
  generation with multiple labeled attributes, not only emotion conversion.

## Open Caveats

- Current listening artifacts depend on local generated audio in `output/`.
  The checked-in HTML indexes are lightweight, but collaborators need either a
  generated-output bundle or rerun commands to play every clip.
- Joe's latest review covered five high-priority A/B rows, not a full human
  listening study.
- Formal DP accounting remains a paper task.
- Age/gender metadata quality and label imbalance still need auditing before
  training.
