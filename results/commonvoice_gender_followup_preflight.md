# CommonVoice Gender Follow-Up Preflight

Corpus: `/Users/steve/datasets/cv-corpus-21.0-2025-03-14/en`
Validated TSV: `/Users/steve/datasets/cv-corpus-21.0-2025-03-14/en/validated.tsv`
Local clip files: `40000`
Seed: `42`

## Recommendation

- GO: enough balanced gender-known speaker coverage for a gender-only follow-up preflight.
- This is a metadata/data-readiness result only; it does not train or claim perceptual gender control.
- If Joe's style-listening feedback is unresolved, keep this as preparation rather than starting training.

## Gender-Known Local Rows

- Gender-known local clips: `5291`
- Gender-known local speakers: `2775`

| Gender | Clips | Speakers | Multi-clip speakers |
| --- | ---: | ---: | ---: |
| `female` | `907` | `494` | `294` |
| `male` | `4384` | `2281` | `1490` |

## Deterministic Candidate Plan

- Speaker cap per gender: `500`
- Clips per speaker: `2`
- Selected speakers: `994`
- Selected clips: `1788`
- Speaker manifest: `results/commonvoice_gender_followup_speakers.csv`

| Gender | Selected speakers | Selected clips |
| --- | ---: | ---: |
| `female` | `494` | `788` |
| `male` | `500` | `1000` |

## Next Command Shape

Use the speaker manifest as the input contract for the next extraction/training branch. The follow-up should test gender-only first, then gender plus only the shortlisted style controls if gender-only is perceptually plausible.
