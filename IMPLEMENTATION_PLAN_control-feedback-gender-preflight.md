# Listening Feedback Ingestion and Gender Follow-Up Preflight

Branch: `research/control-feedback-gender-preflight`

## Goal

Keep progress moving while Joe reviews the focused `neutral`/`sad` listening
bundle, without starting another model run prematurely.

This branch adds two pieces of infrastructure:

- a reproducible way to ingest Joe/Stephen perceptual feedback into the
  control-selection ledger;
- a local CommonVoice gender-readiness preflight for the next narrow metadata
  follow-up.

## Scope

In scope:

- Ingest filled ratings CSVs or Teams-style plain-text feedback into
  `results/control_selection_perceptual_evidence.csv`.
- Optionally rerun `scripts/build_control_selection_recommendation.py`.
- Verify the local CommonVoice English corpus has enough binary gender-known
  speakers and clips for a gender-only follow-up.
- Emit a deterministic speaker manifest for the future branch.

Out of scope:

- No new training.
- No new generated-audio claim.
- No age/accent retry.
- No `FINDINGS.md` update unless a verified perceptual or generated-audio
  finding appears.

## Implementation Tasks

- [x] Add `scripts/ingest_control_selection_feedback.py`.
- [x] Add tests for feedback summarization, manual status overrides, and
  canonical ledger ordering.
- [x] Add `scripts/preflight_commonvoice_gender_followup.py`.
- [x] Add tests for CommonVoice gender normalization, local-row filtering,
  deterministic speaker selection, and preflight recommendation logic.
- [x] Run the gender preflight against
  `/Users/steve/datasets/cv-corpus-21.0-2025-03-14/en`.
- [x] Update `WORKLOG.md`, `README.md`, `examples/README.md`, and
  `results/README.md`.

## Validation

- [x] Unit tests passed:
  `python3 -m unittest tests.test_ingest_control_selection_feedback tests.test_preflight_commonvoice_gender_followup`
- [x] Compile check passed with a temp bytecode cache:
  `env PYTHONPYCACHEPREFIX=/private/tmp/dpvc_pycache python3 -m py_compile scripts/ingest_control_selection_feedback.py scripts/preflight_commonvoice_gender_followup.py tests/test_ingest_control_selection_feedback.py tests/test_preflight_commonvoice_gender_followup.py`
- [x] Gender preflight wrote:
  `results/commonvoice_gender_followup_preflight.md`,
  `results/commonvoice_gender_followup_preflight.json`, and
  `results/commonvoice_gender_followup_speakers.csv`
- [x] Feedback ingestion smoke wrote a temp ledger at
  `/private/tmp/control_selection_perceptual_evidence_smoke.csv` without
  changing the real perceptual ledger before Joe replies.

## Current Result

The local CommonVoice corpus has enough balanced gender-known speaker coverage
for a gender-only follow-up preflight:

- `40000` local validated rows / clips
- `5291` gender-known local clips
- `2775` gender-known local speakers
- selected deterministic plan: `994` speakers, `1788` clips
- recommendation: `GO`

This is data-readiness only. It is not a perceptual gender-control finding.

## Next Use

When Joe replies to the `neutral`/`sad` focused listening bundle:

```bash
python scripts/ingest_control_selection_feedback.py \
    --feedback-text /path/to/joe_feedback.txt \
    --style-status neutral=supported \
    --style-status sad=mixed \
    --rerun-recommendation
```

If the shortlist gate is resolved, start the next gender-only branch from:

```bash
results/commonvoice_gender_followup_speakers.csv
```
