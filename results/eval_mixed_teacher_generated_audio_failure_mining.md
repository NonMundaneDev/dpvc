# Generated-Audio Failure Mining

Input tag: `mixed_teacher`

Summary source: `results/eval_mixed_teacher_summary.csv`

Collapse source: `results/eval_mixed_teacher_collapse.csv`

## Purpose

Join generated-audio metrics at the speaker/style row level so the next training objective can target observed failures rather than aggregate averages.

## Compared Conditions

| Condition | Recall | Novelty | WER | MOS delta | Any collapse |
|-----------|--------|---------|-----|-----------|--------------|
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `0.4697` | `0.2726` | `0.2348` | `-0.2081` | `20` |
| `mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` | `0.3939` | `0.2960` | `0.2651` | `-0.2072` | `28` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `0.4091` | `0.2962` | `0.2609` | `-0.2065` | `27` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `0.4242` | `0.3008` | `0.2863` | `-0.2148` | `27` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `0.4242` | `0.3032` | `0.2782` | `-0.2122` | `26` |

## Failure Labels

- `emotion_miss`: emotion2vec target exists and the generated file missed it.
- `high_wer`: WER >= `0.3`.
- `low_mos_delta`: MOS delta vs baseline <= `-0.5`.
- `low_novelty`: novelty gain vs baseline <= `0.05`.
- collapse labels come from the existing collapse taxonomy CSV.

## Condition-Level Failure Counts

| Condition | Styled rows | Any failure | Emotion misses | High WER | Low MOS delta | Low novelty | Mean failure score |
|-----------|-------------|-------------|----------------|----------|---------------|-------------|--------------------|
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `99` | `65` | `39` | `32` | `15` | `2` | `2.2727` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `99` | `65` | `38` | `37` | `17` | `1` | `2.2929` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `99` | `66` | `38` | `36` | `17` | `1` | `2.2525` |
| `mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` | `99` | `67` | `40` | `33` | `16` | `1` | `2.2929` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `99` | `58` | `35` | `31` | `14` | `1` | `1.9899` |

## Observed Readout

- Lowest row-level failure score: `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` (`58` any-failure rows, mean score `1.9899`).
- Highest row-level failure score: `mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` (`67` any-failure rows, mean score `2.2929`).
- Most persistent style failures: `disgust` `54/55` failures (98.2%, mean emotion recall `0.0364`); `fear` `51/55` failures (92.7%, mean emotion recall `0.1091`); `anger` `46/55` failures (83.6%, mean emotion recall `0.1636`); `enunciated` `38/55` failures (69.1%).
- Objective-design implication: prioritize generated-audio rows with `emotion_miss` plus `style_to_neutral` for anger/disgust/fear, and keep high-WER or very-low-MOS rows out of direct positive targets unless the goal is content repair.

## Style-Level Readout

| Condition | Style | Rows | Any failure | Emotion recall | Mean WER | Mean MOS delta | Mean novelty |
|-----------|-------|------|-------------|----------------|----------|----------------|--------------|
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `anger` | `11` | `9` | `0.1818` | `0.1210` | `-0.0258` | `0.2508` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `confused` | `11` | `6` | `` | `0.2877` | `-0.2528` | `0.3297` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `disgust` | `11` | `11` | `0.0000` | `0.1789` | `-0.3124` | `0.2408` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `enunciated` | `11` | `8` | `` | `0.3234` | `-0.7748` | `0.3761` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `fear` | `11` | `11` | `0.0000` | `0.2795` | `-0.3058` | `0.3201` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `happy` | `11` | `8` | `0.4545` | `0.3450` | `0.0077` | `0.1803` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `neutral` | `11` | `2` | `0.9091` | `0.1594` | `-0.0539` | `0.2189` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `sad` | `11` | `6` | `0.9091` | `0.3412` | `-0.0252` | `0.1079` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `whisper` | `11` | `4` | `` | `0.3120` | `-0.1152` | `0.6411` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `anger` | `11` | `9` | `0.1818` | `0.2087` | `-0.0270` | `0.2524` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `confused` | `11` | `6` | `` | `0.2785` | `-0.2909` | `0.3339` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `disgust` | `11` | `11` | `0.0000` | `0.1789` | `-0.3097` | `0.2449` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `enunciated` | `11` | `8` | `` | `0.3234` | `-0.7845` | `0.3851` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `fear` | `11` | `10` | `0.0909` | `0.3201` | `-0.3341` | `0.3283` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `happy` | `11` | `6` | `0.5455` | `0.4143` | `0.0096` | `0.1926` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `neutral` | `11` | `3` | `0.8182` | `0.2049` | `-0.0483` | `0.2190` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `sad` | `11` | `7` | `0.9091` | `0.3360` | `-0.0234` | `0.1083` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `whisper` | `11` | `5` | `` | `0.3120` | `-0.1250` | `0.6426` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `anger` | `11` | `9` | `0.1818` | `0.1957` | `-0.0132` | `0.2585` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `confused` | `11` | `6` | `` | `0.2785` | `-0.2830` | `0.3356` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `disgust` | `11` | `11` | `0.0000` | `0.1789` | `-0.3144` | `0.2474` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `enunciated` | `11` | `8` | `` | `0.3104` | `-0.7807` | `0.3856` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `fear` | `11` | `10` | `0.0909` | `0.3071` | `-0.3232` | `0.3327` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `happy` | `11` | `7` | `0.5455` | `0.4078` | `0.0076` | `0.1914` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `neutral` | `11` | `3` | `0.8182` | `0.1854` | `-0.0549` | `0.2232` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `sad` | `11` | `7` | `0.9091` | `0.3133` | `-0.0240` | `0.1100` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `whisper` | `11` | `5` | `` | `0.3272` | `-0.1237` | `0.6444` |
| `mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` | `anger` | `11` | `9` | `0.1818` | `0.1957` | `-0.0235` | `0.2511` |
| `mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` | `confused` | `11` | `6` | `` | `0.2682` | `-0.2560` | `0.3297` |
| `mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` | `disgust` | `11` | `11` | `0.0000` | `0.1724` | `-0.3082` | `0.2403` |
| `mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` | `enunciated` | `11` | `8` | `` | `0.3818` | `-0.7872` | `0.3763` |
| `mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` | `fear` | `11` | `11` | `0.0909` | `0.2812` | `-0.3125` | `0.3186` |
| `mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` | `happy` | `11` | `8` | `0.4545` | `0.3158` | `0.0077` | `0.1793` |
| `mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` | `neutral` | `11` | `3` | `0.8182` | `0.1659` | `-0.0485` | `0.2181` |
| `mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` | `sad` | `11` | `7` | `0.8182` | `0.3234` | `-0.0246` | `0.1080` |
| `mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` | `whisper` | `11` | `4` | `` | `0.2817` | `-0.1116` | `0.6424` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `anger` | `11` | `10` | `0.0909` | `0.1740` | `-0.0532` | `0.2962` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `confused` | `11` | `5` | `` | `0.2152` | `-0.4156` | `0.2764` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `disgust` | `11` | `10` | `0.1818` | `0.1529` | `-0.2315` | `0.2296` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `enunciated` | `11` | `6` | `` | `0.1542` | `-0.5339` | `0.1686` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `fear` | `11` | `9` | `0.2727` | `0.3071` | `-0.3903` | `0.3485` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `happy` | `11` | `7` | `0.4545` | `0.3981` | `-0.0026` | `0.1687` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `neutral` | `11` | `2` | `0.9091` | `0.1269` | `-0.1472` | `0.2351` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `sad` | `11` | `5` | `0.9091` | `0.2955` | `-0.0318` | `0.0877` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `whisper` | `11` | `4` | `` | `0.2889` | `-0.0668` | `0.6430` |

## Failure Mode Totals

| Condition | Mode | Count |
|-----------|------|-------|
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `content_collapse` | `1` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `emotion_miss` | `39` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `high_wer` | `32` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `identity_collapse` | `2` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `low_mos_delta` | `15` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `low_novelty` | `2` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `mixed_collapse` | `1` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `style_to_neutral` | `25` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `content_collapse` | `4` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `emotion_miss` | `38` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `high_wer` | `37` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `identity_collapse` | `1` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `low_mos_delta` | `17` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `low_novelty` | `1` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `style_to_neutral` | `22` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `content_collapse` | `3` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `emotion_miss` | `38` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `high_wer` | `36` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `identity_collapse` | `1` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `low_mos_delta` | `17` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `low_novelty` | `1` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `style_to_neutral` | `22` |
| `mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` | `content_collapse` | `1` |
| `mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` | `emotion_miss` | `40` |
| `mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` | `high_wer` | `33` |
| `mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` | `identity_collapse` | `1` |
| `mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` | `low_mos_delta` | `16` |
| `mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` | `low_novelty` | `1` |
| `mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` | `style_to_neutral` | `26` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `content_collapse` | `2` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `emotion_miss` | `35` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `high_wer` | `31` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `identity_collapse` | `1` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `low_mos_delta` | `14` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `low_novelty` | `1` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `mixed_collapse` | `1` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `style_to_neutral` | `18` |

## Highest-Priority Rows

| Condition | Speaker | Style | Score | Modes | Predicted | Target | WER | MOS delta | Novelty | File |
|-----------|---------|-------|-------|-------|-----------|--------|-----|-----------|---------|------|
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `cremad_1006` | `sad` | `10` | `emotion_miss;style_to_neutral;identity_collapse;mixed_collapse;low_novelty` | `neutral` | `sad` | `0.0000` | `-0.0593` | `0.0498` | `cremad_1006_sad.wav` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `cremad_1006` | `sad` | `10` | `emotion_miss;style_to_neutral;identity_collapse;mixed_collapse;low_novelty` | `neutral` | `sad` | `0.0000` | `-0.0731` | `0.0379` | `cremad_1006_sad.wav` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `cremad_1003` | `fear` | `8` | `emotion_miss;content_collapse;high_wer;low_mos_delta` | `sad` | `fearful` | `0.8571` | `-0.5590` | `0.3755` | `cremad_1003_fear.wav` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `male_2_cremad_1051` | `fear` | `7` | `emotion_miss;style_to_neutral;high_wer;low_mos_delta` | `neutral` | `fearful` | `0.7500` | `-1.0221` | `0.2921` | `male_2_cremad_1051_fear.wav` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `male_1_cremad_1003` | `happy` | `7` | `emotion_miss;content_collapse;high_wer` | `sad` | `happy` | `1.0000` | `-0.0023` | `0.2228` | `male_1_cremad_1003_happy.wav` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `male_2_cremad_1051` | `fear` | `7` | `emotion_miss;style_to_neutral;high_wer;low_mos_delta` | `neutral` | `fearful` | `0.7500` | `-1.0139` | `0.2993` | `male_2_cremad_1051_fear.wav` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `male_1_cremad_1003` | `happy` | `7` | `emotion_miss;content_collapse;high_wer` | `sad` | `happy` | `1.0000` | `-0.0043` | `0.2225` | `male_1_cremad_1003_happy.wav` |
| `mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup` | `male_2_cremad_1051` | `fear` | `7` | `emotion_miss;style_to_neutral;high_wer;low_mos_delta` | `neutral` | `fearful` | `0.7500` | `-0.9960` | `0.3045` | `male_2_cremad_1051_fear.wav` |
| `mixed_teacher_cvrare_failure_targeted_style_teacher_labeled_warmup` | `male_2_cremad_1051` | `fear` | `7` | `emotion_miss;style_to_neutral;high_wer;low_mos_delta` | `neutral` | `fearful` | `0.7500` | `-1.0245` | `0.2871` | `male_2_cremad_1051_fear.wav` |
| `mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard` | `male_2_cremad_1051` | `happy` | `7` | `emotion_miss;content_collapse;high_wer` | `sad` | `happy` | `1.2500` | `-0.0038` | `0.2005` | `male_2_cremad_1051_happy.wav` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `cremad_1003` | `disgust` | `6` | `emotion_miss;style_to_neutral;low_mos_delta` | `neutral` | `disgusted` | `0.0000` | `-0.8770` | `0.2469` | `cremad_1003_disgust.wav` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `cremad_1004` | `disgust` | `6` | `emotion_miss;style_to_neutral;low_mos_delta` | `neutral` | `disgusted` | `0.0000` | `-1.2326` | `0.2825` | `cremad_1004_disgust.wav` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `cremad_1007` | `disgust` | `6` | `emotion_miss;style_to_neutral;high_wer` | `neutral` | `disgusted` | `0.3750` | `+0.0778` | `0.1669` | `cremad_1007_disgust.wav` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `cremad_1023` | `disgust` | `6` | `emotion_miss;style_to_neutral;high_wer` | `neutral` | `disgusted` | `0.4286` | `+0.0297` | `0.1760` | `cremad_1023_disgust.wav` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `female_2_cremad_1012` | `disgust` | `6` | `emotion_miss;style_to_neutral;low_mos_delta` | `neutral` | `disgusted` | `0.2000` | `-1.3859` | `0.3747` | `female_2_cremad_1012_disgust.wav` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `male_1_cremad_1003` | `fear` | `6` | `emotion_miss;style_to_neutral;low_mos_delta` | `neutral` | `fearful` | `0.0000` | `-0.6533` | `0.3646` | `male_1_cremad_1003_fear.wav` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `male_2_cremad_1051` | `anger` | `6` | `emotion_miss;style_to_neutral;high_wer` | `neutral` | `angry` | `0.7500` | `+0.0005` | `0.2176` | `male_2_cremad_1051_anger.wav` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `male_2_cremad_1051` | `disgust` | `6` | `emotion_miss;style_to_neutral;high_wer` | `neutral` | `disgusted` | `0.7500` | `-0.0770` | `0.1424` | `male_2_cremad_1051_disgust.wav` |
| `mixed_teacher_cvrare_antineutral_labeled_warmup` | `male_2_cremad_1051` | `happy` | `6` | `emotion_miss;style_to_neutral;high_wer` | `neutral` | `happy` | `0.7500` | `-0.0015` | `0.1712` | `male_2_cremad_1051_happy.wav` |
| `mixed_teacher_cvrare_decoder_proto_labeled_warmup` | `cremad_1003` | `disgust` | `6` | `emotion_miss;style_to_neutral;low_mos_delta` | `neutral` | `disgusted` | `0.0000` | `-0.9323` | `0.2563` | `cremad_1003_disgust.wav` |

## Training Implication

- Use this artifact to choose the next decoder-aware objective by style and failure mode.
- Prefer target-style/generated-audio failures over another global scalar loss sweep.
- Inspect rows with high emotion-miss plus high WER or low MOS before turning them into training targets; some may be bad supervision examples rather than useful corrective targets.
