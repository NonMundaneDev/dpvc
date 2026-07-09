# Paper Evidence Checklist

**Date:** 2026-07-09
**Branch:** `research/commonvoice-gender-followup`
**Purpose:** short checklist of what is ready, what should be claimed, and what
still blocks final paper writing.

## Current Claim Set

Use this as the paper-facing claim boundary:

- The active system is OpenVoice-based controllable voice-to-voice speaker
  generation / anonymization.
- The current reference guard is the quality-balanced evidence anchor:
  `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard`.
- The current headline style controls are `neutral` and `sad`.
- `Sad` should be described as perceptible but subtle/source-dependent.
- `Happy`, `enunciated`, and `whisper` are secondary or demo-only candidates,
  not headline controls unless a later focused review promotes them.
- `Anger`, `confused`, `disgust`, and `fear` are diagnostic or limitation
  controls under current evidence.
- Gender, age, and accent controls are diagnostic/future work, not paper claims.

## Ready Evidence

| Evidence item | Status | Primary artifacts |
| --- | --- | --- |
| Current reference guard metrics | Ready as current anchor | `FINDINGS.md` Finding 31; `results/eval_*_mixed_teacher_*_sad_enunc_guard.csv` |
| Headline style-control shortlist | Ready | `results/control_selection_recommendation.md`; `FINDINGS.md` Findings 41-42 |
| `neutral` perceptual support | Ready | `results/listening_control_shortlist_neutral_sad_joe_2026-06-08.md` |
| `sad` perceptual support | Ready with subtle/source-dependent wording | `results/listening_control_shortlist_neutral_sad_joe_2026-06-08.md` |
| External identity-shift corroboration | Ready as proxy evidence | `results/eval_external_speaker_verifier_cvrare_sad_enunc_guard.md`; `FINDINGS.md` Finding 36 |
| Hard-style limitation evidence | Ready | `results/generated_audio_content_repair_gate.md`; `FINDINGS.md` Findings 35, 38, 39 |
| Metadata-control limitation evidence | Ready | `results/listening_metadata_w010_labeled_warmup.md`; `results/listening_gender_followup_available_w005_labeled_warmup_stephen_2026-07-09.md`; `FINDINGS.md` Findings 37, 43 |

## Blocking Before Paper Submission

These are the highest-value missing pieces before final submission:

1. **Formal DP accounting**
   - Define the privacy mechanism precisely.
   - Compute epsilon / privacy-utility curves for the chosen noise settings.
   - Separate identity-shift evidence from formal privacy claims.

2. **Independent speaker-verification / EER trial**
   - Build an independent labeled speaker-verification trial CSV.
   - Report EER or a defensible equivalent beyond the current proxy threshold.
   - Keep ECAPA framed as identity-shift/source-similarity evidence, not
     intelligibility evidence.

3. **Repeated-seed confidence intervals**
   - Repeat the final candidate training/evaluation enough to report stability.
   - Prioritize the current reference guard and main baselines rather than every
     historical ablation.

4. **Paper-ready listening evidence**
   - Decide whether Joe's existing `neutral` / `sad` review is enough for the
     paper, or whether a small broader listener panel is required.
   - Keep the task narrow: confirm `neutral`, confirm subtle `sad`, and avoid
     reopening all nine controls.

5. **Methods/evidence consistency pass**
   - Ensure `README.md`, `FINDINGS.md`, `PAPER_METHODS_AND_EVIDENCE.md`,
     `results/README.md`, and the Joe brief all use the same claim boundary.
   - Remove or qualify stale wording that says `neutral` / `sad` are only
     pending, or that gender is an active near-term paper claim.

## Optional, Not Blocking

- Focused secondary review for `happy`, `enunciated`, or `whisper`.
- A materially stronger future gender retry, but only if it uses a stronger
  objective or capacity comparison rather than scaling the failed-gate scalar
  checkpoint unchanged.
- Larger CommonVoice extraction for gender only if paired with that materially
  stronger gender objective.

## Do Not Start Next

- Another `disgust` repair loop without new perceptual training evidence.
- Another scalar age/gender/accent sweep.
- A gender classifier or speaker-verifier pass used to promote the failed
  gender checkpoint after the listening gate failed.
- A larger CommonVoice extraction as a substitute for a clearer objective.

## Immediate Next Task

Use the checklist to drive paper cleanup:

1. Make the paper outline use the current claim set.
2. Decide which blocking evidence item Joe wants first.
3. Implement the smallest validation task that answers that blocker.
