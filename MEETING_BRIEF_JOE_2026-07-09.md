# Meeting Brief for Joe Near

**Project:** DPVC / controllable voice-to-voice speaker generation
**Prepared date:** July 9, 2026
**Meeting date:** next Joe update
**Prepared for:** Stephen Oladele
**Current branch:** `research/commonvoice-gender-followup`
**Research fork:** `https://github.com/NonMundaneDev/dpvc`

## 1. The 90-Second Opening

Use this first:

> Since the May 28 pivot, I stopped treating the project as open-ended model
> repair and turned it into a paper-facing control-selection problem. The
> current reference guard is still the best anchor: it keeps about `47%` emotion
> recall, preserves identity shift, and has the cleanest current quality
> tradeoff. After focused listening, the defensible headline controls are now
> narrow: `neutral` and `sad`. `Sad` is real but subtle/source-dependent. The
> gender follow-up did not pass the listening gate; the `female` and `male`
> controls sounded more like subtle or generic speaker/timbre movement than
> reliable gender control. So my recommended paper story is: identity shift plus
> two perceptually confirmed style controls, with hard emotions and metadata
> controls framed as limitations/future work.

## 2. Main Decision Needed

The meeting should confirm one decision:

> Should we freeze the current paper claim set around the current reference
> guard plus `neutral` and `sad`, and move into paper/evaluation cleanup rather
> than another model-training loop?

Recommended answer to bring in:

> Yes. The current evidence supports a narrow, defensible story. More training
> should wait until there is a specific reviewer-facing gap, a stronger
> perceptual target, or a materially different gender/style objective.

## 3. What Changed Since May 28

### A. Control selection is now resolved for headline style claims

We ran the source-separability and generated-output shortlist gate, then
incorporated Joe's focused `neutral` / `sad` listening feedback.

Current buckets:

| Bucket | Styles |
| --- | --- |
| headline controls | `neutral`, `sad` |
| supported but quality-sensitive | `happy`, `enunciated`, `whisper` |
| diagnostic / limitation | `anger`, `confused`, `disgust`, `fear` |

Interpretation:

- `neutral` is clearly supported.
- `sad` is supported, but phrase it as perceptible and subtle /
  source-dependent.
- Do not claim all nine controls work.
- Do not revive `anger`, `disgust`, or `fear` as headline controls without new
  perceptual evidence.

### B. Gender-only follow-up did not pass the perceptual gate

We followed Joe's recommendation to test gender before gender-plus-emotion.

The run:

- used `1181/1788` matched clips from the preflight-selected CommonVoice gender
  manifest;
- trained a gender-only metadata-control checkpoint;
- produced a four-source listening panel with source, baseline,
  baseline+female, and baseline+male.

Local listening result:

- `female` / `male` controls were not reliably perceptible as gender controls;
- differences sounded subtle, inconsistent, or like generic speaker/timbre
  movement;
- no gender claim should be promoted.

Quick acoustic sanity check after listening:

- a simple F0/centroid diagnostic on the four-source gender review bundle found
  `female`-control median F0 above `male`-control median F0 in only `1/4`
  source pairs;
- median `female` minus `male` F0 delta was `-5.69 Hz`;
- this is only an acoustic proxy, not a perceptual classifier, but it agrees
  with the listening-gate decision not to promote gender control.

Interpretation:

- Gender structure is objectively present in embeddings and metadata-control
  latents, but this scalar speaker-embedding VAE knob did not make it
  listener-clear.
- Do not run gender/speaker-verifier diagnostics as claim evidence for this
  checkpoint after the listening gate failed.
- A future gender retry should be materially stronger, not just more of the
  same checkpoint.

### C. The paper story is simpler now

The best claim set is:

1. OpenVoice speaker embeddings can support controllable voice-to-voice speaker
   generation.
2. The current reference guard changes identity while preserving content enough
   for a usable demo/evaluation anchor.
3. Two style controls are perceptually confirmed: `neutral` and `sad`.
4. Objective metrics and human listening are both required before promoting a
   control.
5. Hard emotions and metadata controls are useful diagnostics, but not
   headline claims.

## 4. Current Reference Guard

Use the short name first:

- **current reference guard**

Full condition name only if Joe asks:

- `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard`

Top-line metrics:

| Metric | Current reference guard |
| --- | ---: |
| emotion recall | `46.97%` |
| mean styled WER | `0.2348` |
| MOS delta | `-0.2081` |
| novelty gain | `0.2726` |
| files with any collapse | `20` |

Important interpretation:

- It is the current quality-balanced reference, not a perfect final model.
- It is strong enough to anchor paper/evaluation writing.
- The control claims should come from the shortlist and listening gates, not
  from aggregate recall alone.

## 5. What Not To Overclaim

Avoid saying:

- "All nine styles work."
- "Gender control works."
- "Gender failed because the model has no gender information."
- "Speaker novelty equals privacy."
- "ECAPA proves intelligibility."
- "The next step is more training."

Safer wording:

- "The current evidence supports two headline controls: `neutral` and `sad`."
- "Gender is objectively recoverable in embeddings and latents, but the current
  scalar control did not produce reliable perceptual gender control."
- "ECAPA corroborates identity shift/source-similarity behavior; WER handles
  content preservation."
- "More training should be tied to a specific paper/reviewer gap, not open-ended
  improvement."

## 6. What To Ask Joe

Ask these directly:

1. Do you agree that the current paper claim should be narrow: identity shift
   plus `neutral` and `sad`, not all nine controls?
2. Do you agree that gender should stay out of the paper claim unless a
   materially stronger gender objective produces listener-clear control?
3. Should we now prioritize paper/evaluation cleanup over another training run?
4. Which missing evidence would matter most before writing: repeated seeds,
   formal DP accounting, independent EER trials, or broader listening?
5. Are `happy`, `enunciated`, or `whisper` worth a small optional secondary
   demo review, or should they stay out of the main story for now?

## 7. Anticipated Questions And Answers

### Q: What is the main result now?

A: The project has moved from "can we make every control work?" to a conservative
paper-facing claim. The current reference guard supports identity movement and
two perceptually confirmed headline style controls: `neutral` and `sad`.

### Q: What happened with gender?

A: The gender-only follow-up trained and generated correctly, but the listening
gate did not pass. The female/male controls were subtle or generic rather than
reliably gendered. This means gender remains diagnostic/future work, not a
paper claim.

### Q: Does that mean gender is absent from the embeddings?

A: No. The separability probe showed gender is strongly recoverable in both raw
OpenVoice embeddings and metadata-control VAE latents. The failure is
perceptual control: the scalar knob did not turn that structure into a
listener-clear gender attribute.

### Q: Why not run a gender classifier or speaker verifier anyway?

A: It could characterize movement, but after the listening gate failed it would
not establish a usable gender-control claim. It risks measuring generic speaker
movement rather than perceptual gender control.

### Q: Which controls should be in the paper?

A: `Neutral` and `sad` as headline style controls. `Happy`, `enunciated`, and
`whisper` can remain secondary or demo-only candidates if Joe wants a small
optional review. `Anger`, `confused`, `disgust`, and `fear` stay limitations
under current evidence.

### Q: Is privacy solved?

A: No. The system has identity-shift evidence, including ECAPA corroboration,
but formal DP accounting and independent speaker-verification/EER trials remain
open paper tasks.

### Q: What should happen next?

A: Paper/evaluation cleanup. Freeze the narrow claim set, make the evidence map
consistent, and identify the smallest missing validation tasks before writing.

## 8. Concepts Stephen Should Be Ready To Explain

### Current reference guard

Short label for the best current quality-balanced profile. Use this phrase
before giving the long checkpoint/profile name.

### Headline control

A control that passed source evidence, generated-output metrics, and perceptual
listening. Today that means `neutral` and `sad`.

### Diagnostic control

A control that teaches us something about the system but should not be claimed
as working. Today this includes hard emotions like `disgust` and metadata
controls like gender.

### Gender separability versus gender control

Separability means gender can be predicted from embeddings or latents. Control
means moving a knob produces an audio output that listeners hear as the target
gender. The current system has the former, not the latter.

### ECAPA / speaker verification

Use it for identity shift or source-similarity evidence. Do not describe it as
an intelligibility metric.

## 9. Proposed Next Work

Recommended next two-week plan:

1. Update paper/evidence docs to make the claim set consistent:
   current reference guard, `neutral`, `sad`, limitations.
2. Refresh the paper methods/evidence packet so it no longer says
   `neutral`/`sad` are merely pending listening and no longer frames gender as
   an active near-term claim.
3. Build a final evidence checklist for:
   - repeated-seed confidence intervals;
   - independent speaker-verification/EER trials;
   - formal DP accounting;
   - optional broader listening for the two headline controls.
4. Only after that, decide whether any new experiment is worth doing.

Do not start with:

- another gender checkpoint;
- another `disgust` repair;
- another scalar metadata-control sweep;
- a larger CommonVoice extraction unless it is attached to a materially
  stronger objective.

## 10. Meeting Outcome Template

After the meeting, paste notes back into Codex using this structure:

```text
Does Joe agree with the narrow claim set?

Does Joe agree that gender remains diagnostic/future work?

Which evidence item matters most before writing?

Does Joe want secondary review for happy/enunciated/whisper?

Any change to the current reference guard?

Next two-week action:
```
