# CommonVoice Metadata Separability Probe

Artifact: `embeddings/openvoice_mixed_teacher_cvrare_hybrid_extra_metadata_base.pt`
VAE checkpoint: `embeddings/openvoice_vae_mixed_teacher_cvrare_metadata_w010_labeled_warmup.pt`
Seed: `42`
Minimum class count: `20`
Top-k classes: `8`
Permutations: `100`

## Interpretation Rule

This probe uses a nearest-centroid classifier with deterministic stratified splits.
A metadata field is useful for control only if it is separable beyond a majority baseline and a train-label permutation baseline.
This does not prove perceptual controllability; it only says whether the metadata has recoverable structure in the tested feature space.

## Results

| Field | Space | Used rows | Classes | Accuracy | Macro F1 | Majority F1 | F1 delta | Perm F1 | p(F1) | Sep. ratio | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gender | embedding | 1795 | 2 | 0.9265 | 0.8797 | 0.4531 | 0.4266 | 0.4575 | 0.0099 | 0.6985 | strongly separable |
| gender | vae_mu | 1795 | 2 | 0.9198 | 0.8670 | 0.4531 | 0.4139 | 0.4547 | 0.0099 | 1.1708 | strongly separable |
| age | embedding | 1885 | 7 | 0.2267 | 0.1499 | 0.0936 | 0.0563 | 0.0993 | 0.0099 | 0.2141 | weak / diagnostic only |
| age | vae_mu | 1885 | 7 | 0.1992 | 0.1412 | 0.0936 | 0.0476 | 0.0845 | 0.0099 | 0.2459 | weak / diagnostic only |
| accent | embedding | 1621 | 8 | 0.2346 | 0.1501 | 0.0821 | 0.0680 | 0.0768 | 0.0099 | 0.2307 | weak / diagnostic only |
| accent | vae_mu | 1621 | 8 | 0.1531 | 0.1016 | 0.0821 | 0.0195 | 0.0617 | 0.0396 | 0.1853 | not meaningfully separable |

## Class Counts

- `gender` / `embedding`: `{"female": 309, "male": 1486}`
- `gender` / `vae_mu`: `{"female": 309, "male": 1486}`
- `age` / `embedding`: `{"fifties": 91, "forties": 176, "seventies": 24, "sixties": 33, "teens": 269, "thirties": 374, "twenties": 918}`
- `age` / `vae_mu`: `{"fifties": 91, "forties": 176, "seventies": 24, "sixties": 33, "teens": 269, "thirties": 374, "twenties": 918}`
- `accent` / `embedding`: `{"Australian English": 58, "Canadian English": 80, "England English": 242, "India and South Asia (India, Pakistan, Sri Lanka)": 373, "Irish English": 22, "Scottish English": 24, "Southern African (South Africa, Zimbabwe, Namibia)": 32, "United States English": 790}`
- `accent` / `vae_mu`: `{"Australian English": 58, "Canadian English": 80, "England English": 242, "India and South Asia (India, Pakistan, Sri Lanka)": 373, "Irish English": 22, "Scottish English": 24, "Southern African (South Africa, Zimbabwe, Namibia)": 32, "United States English": 790}`

## Recommended Use

- If age/gender are weak here, do not spend more cycles on scalar age/gender control without better labels or a stronger perceptual target.
- If they are separable in raw embeddings but not VAE latents, the bottleneck is likely the current VAE objective/capacity rather than CommonVoice metadata itself.
- If they are separable in both, the next training branch should use this probe to select rows/classes and then run a listening-first metadata-control panel.
