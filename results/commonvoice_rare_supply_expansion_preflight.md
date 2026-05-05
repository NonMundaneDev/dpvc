# CommonVoice Rare-Class Supply Expansion Preflight

This report checks whether a local CommonVoice English corpus is large enough to justify rebuilding the pseudo-label pool for rare canonical styles before another mixed-data model run.

## Decision

- `GO`: at least one candidate corpus meets the current row and speaker gate.
- rare styles: `anger, fear`
- target selected rare rows per style: `50`
- recommended minimum usable rows: `22538`
- minimum usable speakers: `2000`

## Candidate Corpora

| path | usable rows | usable speakers | missing clips | decision |
| --- | ---: | ---: | ---: | --- |
| `/data/cv-corpus-21.0-2025-03-14/en` | 0 | 0 | 0 | NO-GO |
| `/Users/steve/datasets/cv-corpus-21.0-2025-03-14/en` | 40000 | 20537 | 0 | GO |
| `/Users/steve/datasets/cv-corpus-21.0-2025-03-14-subset/en` | 1202 | 500 | 0 | NO-GO |

### Why `/data/cv-corpus-21.0-2025-03-14/en` is blocked

- corpus path does not exist
- usable rows 0 < recommended 22538
- usable speakers 0 < minimum 2000

### Why `/Users/steve/datasets/cv-corpus-21.0-2025-03-14-subset/en` is blocked

- usable rows 1202 < recommended 22538
- usable speakers 500 < minimum 2000

## Rare-Rate Estimate

The row target is derived from the checked-in pseudo-labeled CommonVoice artifacts, then multiplied by a safety factor so the next run is not just another tiny rare-class sample.

| reference artifact | rows | selected anger | selected fear |
| --- | ---: | ---: | ---: |
| `embeddings/openvoice_commonvoice_cv500_pseudo_filtered.pt` | 1202 | 6 | 4 |
| `embeddings/openvoice_commonvoice_cv500_pseudo_hybrid_extra_priority.pt` | 1202 | 5 | 4 |

| style | rows needed before safety |
| --- | ---: |
| `anger` | 12020 |
| `fear` | 15025 |

## Next Commands

Run these in order. Stop after the supply audit if `anger` and `fear` still fail to reach the target selected-row count.

### preflight

```bash
python scripts/plan_commonvoice_rare_supply_expansion.py --corpus-path /Users/steve/datasets/cv-corpus-21.0-2025-03-14/en
```

### extract

```bash
python examples/openvoice_extract_commonvoice.py --corpus-path /Users/steve/datasets/cv-corpus-21.0-2025-03-14/en --output embeddings/openvoice_commonvoice_cvrare_expanded_emb.pt --seed 42 --max-speakers 13308 --max-clips-per-speaker 3
```

### emotion2vec_score

```bash
python scripts/annotate_commonvoice_pseudolabels.py --embeddings embeddings/openvoice_commonvoice_cvrare_expanded_emb.pt --output embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_scored.pt --save-style-score-map --report-threshold 0.60
```

### emotion2vec_filter

```bash
python scripts/filter_commonvoice_pseudolabels.py --input embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_scored.pt --output embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_filtered.pt --default-threshold 0.60 --style-targets anger=50,fear=50,disgust=80,happy=80,neutral=120,sad=120 --acceptance-policy balanced_targets
```

### prototype_score

```bash
python scripts/annotate_commonvoice_latent_prototypes.py --commonvoice embeddings/openvoice_commonvoice_cvrare_expanded_emb.pt --combined embeddings/openvoice_combined_emb.pt --checkpoint embeddings/openvoice_vae_combined.pt --output embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_prototype.pt
```

### prototype_filter

```bash
python scripts/filter_commonvoice_pseudolabels.py --input embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_prototype.pt --output embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_prototype_filtered.pt --default-threshold 0.35 --style-targets confused=50,enunciated=50,whisper=50 --acceptance-policy balanced_targets
```

### hybrid_combine

```bash
python scripts/combine_commonvoice_pseudolabel_teachers.py --emotion2vec embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_filtered.pt --prototype embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_prototype_filtered.pt --output embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_hybrid_extra_priority.pt --policy prototype_extra_priority
```

### supply_audit

```bash
python scripts/audit_commonvoice_pseudolabel_supply.py --artifacts embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_scored.pt embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_filtered.pt embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_hybrid_extra_priority.pt --out-csv results/commonvoice_pseudolabel_supply_audit_rare_supply.csv --out-md results/commonvoice_pseudolabel_supply_audit_rare_supply.md
```

### mixed_artifact_after_audit_passes

```bash
python scripts/build_mixed_training_set.py --commonvoice embeddings/openvoice_commonvoice_cvrare_expanded_pseudo_hybrid_extra_priority.pt --output embeddings/openvoice_mixed_teacher_rare_supply_base.pt --acceptance-policy artifact_selected --commonvoice-prefer-pseudo --commonvoice-max-clips-per-speaker 2 --pseudo-confidence-scale --pseudo-row-weight 0.75 --true-row-weight 1.25
```

### train_after_audit_passes

```bash
python examples/openvoice_train_vae_mixed.py --embeddings embeddings/openvoice_mixed_teacher_rare_supply_base.pt --output embeddings/openvoice_vae_mixed_teacher_rare_supply_balanced.pt --schedule static_balanced
```

## Validation

- `Validation`: the preflight scanned every candidate path supplied to the script.
- `Validation`: the gate uses local `validated.tsv` rows with matching files in `clips/`, not a remote dataset shortcut.
- `Validation`: the current decision is reproducible from the JSON report next to this file.
