# Meeting Brief for Joe Near

**Project:** DPVC / controllable voice-to-voice speaker generation
**Prepared:** July 15, 2026
**Meeting:** July 16, 2026
**Branch:** `research/commonvoice-gender-followup`

## 1. The 90-Second Opening

> I took your questions as a request to recover the broader control story rather
> than freeze the paper at neutral and sad. I ran two bounded studies. First, I
> compared the old combined style model and the current reference guard on the
> same 11 speakers and all nine controls. That confirms the old model really did
> have stronger anger behavior, while the current model has preserved whisper
> movement and substantially improved its intelligibility. Second, I recovered
> your two exact March CommonVoice metadata checkpoints from git history and
> reran age and gender at the trained endpoints and at the stronger old demo
> setting. The first age/gender model produces a consistent female-above-male F0
> direction on all five speakers, with a 39 Hz median gap at the trained setting,
> low mean WER, and little MOS loss. The listening gate is still required, but
> this is materially stronger than our failed recent gender checkpoint. I want
> to use tomorrow to decide which recovered controls enter the final experiment
> matrix, then finish the remaining paper evidence in July.

## 2. Direct Answers to Your Questions

### Is neutral/sad enough?

No. I agree that two styles alone are too narrow for the intended paper. The
new evidence gives us a principled way to test expansion without claiming all
nine controls.

### Did whisper and anger work before?

Yes, in our earlier combined CREMA-D + Expresso experiment, not in a separate
Joe emotion experiment.

- Legacy anger target recall was `27.3%`, versus `9.1%` in the current guard.
- Current whisper retains almost the same novelty movement (`0.643` versus
  `0.656`) while improving mean WER (`0.289` versus `0.448`) and eliminating
  the three legacy high-WER collapses.
- Current happy reaches `45.5%` recall versus `0%` in the legacy model, but its
  mean WER is `0.398`, so it needs row-level listening before promotion.

This means controls were not simply lost. Anger regressed in target alignment;
whisper remains active and cleaner; happy improved objectively but is
quality-sensitive.

### Was the old gender experiment yours or ours?

The two March checkpoints were Joe's CommonVoice-only metadata models:

- commit `de65862`: age + gender;
- commit `8bfb1fe`: age + gender + accent.

They did not contain emotion controls. The older whisper/anger evidence came
from our later combined style model.

### Have age, gender, and emotion worked simultaneously with pseudo-labels?

Not yet as a perceptually validated joint system.

- The code can train style and metadata dimensions together.
- The mixed metadata checkpoint generated age/gender/emotion outputs, but its
  age/gender listening panel sounded identical or like generic timbre movement.
- The later gender-only available-subset checkpoint also failed Stephen's
  perceptual gate.
- The recovered historical models show a much stronger gender-direction signal,
  but they are metadata-only and therefore do not establish joint control.

The remaining joint question is narrow: can the successful historical gender
objective be reproduced alongside the best style subset without destroying
either control family?

## 3. New Result A: Matched Style-Control Recovery

The study uses the same `11` sources and `99` styled rows per condition.

| Style | Legacy recall | Current recall | Legacy WER | Current WER | Main read |
| --- | ---: | ---: | ---: | ---: | --- |
| anger | `27.3%` | `9.1%` | `0.159` | `0.174` | real regression in target alignment |
| happy | `0%` | `45.5%` | `0.115` | `0.398` | candidate, but quality-sensitive |
| neutral | `81.8%` | `90.9%` | `0.120` | `0.127` | strong current control |
| sad | `36.4%` | `90.9%` | `0.159` | `0.296` | strong but source-dependent |
| whisper | n/a | n/a | `0.448` | `0.289` | movement preserved; needs listening |

Bundle:
`results/control_recovery_old_vs_current_review_bundle_2026-07-15/index.html`

## 4. New Result B: Historical Metadata Recovery

Both exact historical checkpoints were run over five source speakers at:

- strength `1`: the trained `-1/+1` label endpoints;
- strength `2`: the extrapolated setting used in the old demo code.

| Model | Control | Strength | Direction | Median F0 gap | Mean WER | Mean MOS delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| age/gender v1 | gender | `1` | `5/5` | `39.05 Hz` | `0.057` | `-0.039` |
| age/gender v1 | gender | `2` | `5/5` | `99.98 Hz` | `0.200` | `-0.009` |
| age/gender/accent v2 | gender | `1` | `5/5` | `15.33 Hz` | `0.041` | `-0.058` |
| age/gender/accent v2 | gender | `2` | `5/5` | `67.21 Hz` | `0.071` | `-0.136` |

Important limits:

- F0 direction is an acoustic sanity check, not proof of perceived gender.
- Strength `2` is outside the training label range.
- Age remains weak and hard to interpret perceptually; do not promote it from
  this diagnostic.

Bundle:
`results/historical_metadata_recovery_review_bundle_2026-07-15/index.html`

## 5. Listening Decision Before the Meeting

Priority order for Stephen:

1. Historical `joe_age_gender_v1`, gender strength `1`, all five speakers.
2. The same model at gender strength `2` to judge added clarity versus content
   damage.
3. Current-versus-legacy `whisper` on the five matched rows.
4. Current `happy`, then legacy/current `anger`, only if time remains.

Decision rule:

- If v1 gender is consistently heard as the named endpoints at strength `1`,
  treat gender as a recovered candidate and design one final joint
  gender-plus-style replication.
- If only strength `2` is clear, report an extrapolation-dependent recovery and
  discuss whether its WER cost is acceptable.
- If it remains timbre-only, retain gender as a limitation despite the strong
  acoustic direction.
- Promote whisper or happy only if the intended control is audible and content
  remains usable across speakers.

## 6. Decisions to Ask Joe For

1. Does the historical gender result justify one final joint replication if
   Stephen's listening gate passes?
2. Should whisper be the first non-emotion secondary control reviewed for the
   paper?
3. Is happy worth keeping despite its current WER cost, or should the emotion
   subset remain neutral/sad plus a targeted anger repair?
4. Should age remain a limitation/broad-bucket future item rather than a July
   objective?
5. Which submission blocker should be completed first: formal DP accounting,
   independent EER, repeated-seed intervals, or a broader listener study?

## 7. Recommended July Closeout

1. **July 16:** freeze the candidate control set after listening and Joe's
   decision.
2. **July 17-21:** run at most one final joint replication, only if historical
   gender passes the perceptual gate; otherwise stop model development.
3. **July 20-25:** complete the highest-priority privacy/evaluation blocker and
   repeated-seed table for the frozen reference.
4. **July 24-28:** run the final bounded listening study on the selected control
   set.
5. **July 27-31:** draft methods, experiments, results, limitations, and the
   evidence table from checked-in artifacts.

The stop rule is important: July work should close a defined evidence gap. It
should not reopen broad model search.
