# Canonical Evidence and Demo Packet

**Created:** 2026-05-26
**Last updated:** 2026-05-28
**Current update branch:** `research/generated-audio-content-repair`
**Base research line:** `research/controllable-vae`

## Top-Line Answer

We already have a substantial controllable-voice result. The strongest current
claim is not that every emotion control is perfect. The claim is that the
OpenVoice controllable VAE can generate/anonymize speakers while moving labeled
perceptual attributes in controlled directions, with several styles that are
perceptually clear and measurable.

Two recent listening gates should be read narrowly:

- Joe's higher-strength `anger_s10` / `fear_s7p5` review found no candidate
  wins over the current guarded reference, and the generated-audio
  content-repair gate now promotes `0/33` hard-style strength rows.
- The first CommonVoice age/gender metadata-control panel sounded identical or
  mostly like generic speaker/timbre shifts.
- The follow-up metadata separability probe shows gender is objectively
  recoverable in the current embeddings/latents, but age/accent remain weak or
  diagnostic and perceptual control is still unproven.

Neither result invalidates the core system. They mean we should keep the
strength grid and metadata controls diagnostic while consolidating the current
positive style-control evidence for paper drafting.

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
- It now has external ECAPA speaker-verifier support for identity shift:
  `0.3594` mean styled novelty gain vs baseline and `6/99` styled rows accepted
  as source at a proxy threshold.
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
- The content-repair gate promotes `0/33` hard-style strength rows.
- Objective classifier gains are not enough unless content, naturalness,
  novelty, and human listening all agree.

### Tier 4: Historical Strong Evidence

Use these findings to explain why this research line is substantial:

- `FINDINGS.md` Finding 1: OpenVoice is the right controllable path; ControlVC
  is a useful DP baseline but a negative style-control result.
- `FINDINGS.md` Finding 2 / 4 / 6: OpenVoice plus CREMA-D/Expresso produces
  perceptibly distinct controls, with `whisper` especially strong and robust.
- `FINDINGS.md` Finding 30 / 31: expanded rare-supply mixed teacher reaches
  paper-relevant recall while exposing the quality tradeoff.
- `FINDINGS.md` Finding 36: an external ECAPA speaker verifier corroborates
  the current guard's identity shift, with a proxy-threshold caveat.
- `FINDINGS.md` Finding 37: CommonVoice gender metadata is separable, while
  age/accent controls remain diagnostic and not perceptually validated.
- `FINDINGS.md` Finding 38: the generated-audio content-repair gate blocks
  all current hard-style strength candidates from preset promotion.
- `FINDINGS.md` Finding 35: generated-audio strength reranking is diagnostic,
  not a safe default.

### Tier 5: Metadata-Control Diagnostic

Use this to explain what is implemented but not yet a claim:

- `results/listening_metadata_w010_labeled_warmup.md`
- `results/listening_metadata_w010_labeled_warmup.html`
- `results/commonvoice_metadata_separability_mixed_metadata_base.md`
- `IMPLEMENTATION_PLAN_commonvoice-metadata-controls.md`

Result:

- The metadata-control path trains and generates.
- Local listening found the age/gender variants effectively identical or like
  generic speaker/timbre shifts.
- Objective separability says gender signal exists, but age/accent are weak
  and the listener-clear control claim is not established.
- Do not promote age/gender control as a current paper result.

## What We Can Say Now

Good wording:

> We have a controllable speaker-generation/anonymization system with clear
> audible controls for several styles, especially whisper, and a quantitative
> mixed-data result around 47% emotion recall. Recent listening gates say
> metric-selected strength increases and first-pass age/gender controls are not
> perceptually ready to promote yet. The metadata probe says gender is
> objectively separable but not yet listener-clear, so the next branch should
> prioritize generated-audio/content repair for the hard emotion styles unless
> we deliberately choose a narrow gender-focused follow-up.

Avoid saying:

- The strength grid solved anger/fear.
- Joe rejected the whole system.
- The model is only an emotion converter.
- The next step is simply more strength or more emotion tuning.
- Age/gender controls work perceptually.

## Next Step

The next research branch should be targeted rather than exploratory:

- generated-audio-calibrated training repair for hard styles, because the
  current strength-grid gate promotes no preset;
- narrow gender-focused metadata-control follow-up only if we need a metadata
  appendix;
- repeated-seed confidence intervals before final tables;
- formal DP accounting before submission.

## Open Caveats

- Current listening artifacts depend on local generated audio in `output/`.
  The checked-in HTML indexes are lightweight, but collaborators need either a
  generated-output bundle or rerun commands to play every clip.
- Joe's latest review covered five high-priority A/B rows, not a full human
  listening study.
- Formal DP accounting remains a paper task.
- Age/gender metadata quality has been audited, first-pass control plumbing
  exists, and metadata separability has been probed. Gender has objective
  signal, but the first perceptual panel failed; future metadata work should
  first test whether OpenVoice embeddings encode recoverable age/gender signal.
