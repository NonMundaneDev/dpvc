# Historical Metadata Recovery: Quality Summary

WER is measured against each model's same-source baseline with Whisper `base`.
MOS delta uses SQUIM_SUBJECTIVE against the same baseline.

| Model | Attribute | Strength | F0 direction | Median F0 delta | Mean WER | Median WER | WER >= 0.8 | Mean MOS delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `joe_age_gender_v1` | `gender` | 1 | 5/5 | 39.0485 Hz | 0.0571 | 0.0000 | 0/10 | -0.0387 |
| `joe_age_gender_v1` | `gender` | 2 | 5/5 | 99.9816 Hz | 0.2000 | 0.1071 | 0/10 | -0.0090 |
| `joe_age_gender_v1` | `age` | 1 | 5/5 | 11.9376 Hz | 0.1107 | 0.0000 | 0/10 | -0.0223 |
| `joe_age_gender_v1` | `age` | 2 | 3/5 | 10.2887 Hz | 0.2179 | 0.0000 | 0/10 | -0.0112 |
| `joe_age_gender_accent_v2` | `gender` | 1 | 5/5 | 15.3342 Hz | 0.0412 | 0.0000 | 0/10 | -0.0578 |
| `joe_age_gender_accent_v2` | `gender` | 2 | 5/5 | 67.2102 Hz | 0.0706 | 0.0000 | 0/10 | -0.1363 |
| `joe_age_gender_accent_v2` | `age` | 1 | 5/5 | 7.2549 Hz | 0.0294 | 0.0000 | 0/10 | -0.0476 |
| `joe_age_gender_accent_v2` | `age` | 2 | 4/5 | 14.8546 Hz | 0.1494 | 0.0000 | 0/10 | -0.1521 |

## Reading

- `joe_age_gender_v1` has the strongest gender-direction acoustic movement, including at the trained `1` endpoint.
- Strength `2` is an extrapolation beyond the training labels. It is useful for diagnosis but needs a stricter perceptual and intelligibility gate.
- The quality metrics characterize preservation only. They do not establish that listeners hear the intended age or gender.
- The listening dashboard remains the promotion gate for any paper-facing metadata-control claim.
