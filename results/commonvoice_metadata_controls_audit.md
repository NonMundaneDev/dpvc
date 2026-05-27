# CommonVoice Metadata Controls Audit

Corpus: `/Users/steve/datasets/cv-corpus-21.0-2025-03-14/en`
Validated TSV: `/Users/steve/datasets/cv-corpus-21.0-2025-03-14/en/validated.tsv`
Local clips: `40000`

## Recommendation

- Add a first-pass metadata-control experiment.
- Use `gender` as one binary scalar control: `female=-1`, `male=1`.
- Use `age` as one ordinal scalar control over known CommonVoice age buckets.
- Keep latent dimensionality at `15` for the first pass: style dims `0-8`, metadata dims `9-10`, free dims `11-14`.
- Train with masked partial-label losses so rows missing age or gender still contribute reconstruction/style information.

## Local Validated Subset

- Rows: `40000`
- Unique speakers: `20537`
- Age-control rows: `5504` (13.8%)
- Gender-control rows: `5291` (13.2%)
- Rows with both age and gender controls: `5258` (13.1%)

### Age Counts

- `twenties`: `2637`
- `thirties`: `1106`
- `teens`: `788`
- `fourties`: `521`
- `fifties`: `265`
- `sixties`: `114`
- `seventies`: `63`
- `eighties`: `9`
- `nineties`: `1`

### Gender Counts

- `male_masculine`: `4384`
- `female_feminine`: `907`
- `non-binary`: `6`
- `do_not_wish_to_say`: `2`

### Age Bin Counts

- `older`: `452`
- `young`: `3425`
- `adult`: `1627`

## Extracted Embedding Artifact

Artifact: `/Users/steve/UVM-plaid/dp-vc/embeddings/openvoice_commonvoice_cvrare_expanded_emb.pt`
- Rows: `25910`
- Unique speakers: `13308`
- Age-control rows: `3566` (13.8%)
- Gender-control rows: `3429` (13.2%)
- Rows with both age and gender controls: `3403` (13.1%)

### Existing Metadata Report In Artifact

- `age`: known `3566` / `25910`, unique `9`
- `gender`: known `3437` / `25910`, unique `4`
- `accent`: known `3590` / `25910`, unique `266`

## Implementation Implications

- The raw/extracted CommonVoice artifact already preserves per-row age/gender metadata.
- Mixed artifacts should preserve per-row metadata scalars and masks, not only aggregate CommonVoice metadata.
- Metadata controls need masked label loss so missing age/gender values do not become false zeros.
- The first training target should be conservative: `gender_binary` and `age_ordinal`, not a large one-hot demographic taxonomy.
