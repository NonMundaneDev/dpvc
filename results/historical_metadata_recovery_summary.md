# Historical Metadata-Control Recovery

This study re-runs Joe's exact March 2026 CommonVoice metadata checkpoints.
It tests the original latent ordering and label polarity rather than the current metadata-control interface.

## Provenance

- `joe_age_gender_v1`: git commit `de65862`, `examples/openvoice_vae_features.pt` (CommonVoice age + gender); `8` latent dimensions
- `joe_age_gender_accent_v2`: git commit `8bfb1fe`, `examples/openvoice_vae_features2.pt` (CommonVoice age + gender + accent); `8` latent dimensions
- Sources: `5`
- Historical dimensions: age `0`, gender `1`
- Historical polarity: male `-1`, female `+1`; teens `-1`, sixties `+1`
- Strength `1` is the trained endpoint; strength `2` reproduces the extrapolated old demo setting.
- Noise `0`, seed `42`

## Acoustic Sanity Check

| Model | Attribute | Strength | Pairs | Positive endpoint higher F0 | Median F0 delta | Median centroid delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `joe_age_gender_accent_v2` | `age` | 1 | 5 | 5/5 | 7.2549 Hz | -340.8260 Hz |
| `joe_age_gender_accent_v2` | `age` | 2 | 5 | 4/5 | 14.8546 Hz | -736.2752 Hz |
| `joe_age_gender_accent_v2` | `gender` | 1 | 5 | 5/5 | 15.3342 Hz | -240.2316 Hz |
| `joe_age_gender_accent_v2` | `gender` | 2 | 5 | 5/5 | 67.2102 Hz | -510.2463 Hz |
| `joe_age_gender_v1` | `age` | 1 | 5 | 5/5 | 11.9376 Hz | -671.2181 Hz |
| `joe_age_gender_v1` | `age` | 2 | 5 | 3/5 | 10.2887 Hz | -1198.5185 Hz |
| `joe_age_gender_v1` | `gender` | 1 | 5 | 5/5 | 39.0485 Hz | 586.3817 Hz |
| `joe_age_gender_v1` | `gender` | 2 | 5 | 5/5 | 99.9816 Hz | 1205.3094 Hz |

## Decision Rule

- F0 and spectral centroid are sanity checks, not age/gender classifiers.
- Promote a checkpoint only if listeners consistently hear the named endpoint across source speakers while speech remains intelligible and natural.
- If only strength `2` is audible, report that the old demo depended on latent extrapolation and evaluate its quality cost before treating it as a control.
- If differences remain speaker/timbre-only, keep historical metadata control out of the paper claim.
