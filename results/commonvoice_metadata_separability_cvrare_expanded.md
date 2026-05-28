# CommonVoice Metadata Separability Probe

Artifact: `embeddings/openvoice_commonvoice_cvrare_expanded_emb.pt`
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
| gender | embedding | 3429 | 2 | 0.9452 | 0.9077 | 0.4555 | 0.4522 | 0.4495 | 0.0099 | 0.7062 | strongly separable |
| gender | vae_mu | 3429 | 2 | 0.9417 | 0.9005 | 0.4555 | 0.4450 | 0.4443 | 0.0099 | 1.1765 | strongly separable |
| age | embedding | 3559 | 7 | 0.2295 | 0.1901 | 0.0937 | 0.0964 | 0.0889 | 0.0099 | 0.2159 | weak / diagnostic only |
| age | vae_mu | 3559 | 7 | 0.1665 | 0.1343 | 0.0937 | 0.0406 | 0.0794 | 0.0099 | 0.2546 | weak / diagnostic only |
| accent | embedding | 2881 | 8 | 0.2420 | 0.1902 | 0.0819 | 0.1084 | 0.0687 | 0.0099 | 0.2684 | moderately separable |
| accent | vae_mu | 2881 | 8 | 0.1822 | 0.1446 | 0.0819 | 0.0627 | 0.0578 | 0.0099 | 0.3224 | weak / diagnostic only |

## Class Counts

- `gender` / `embedding`: `{"female": 562, "male": 2867}`
- `gender` / `vae_mu`: `{"female": 562, "male": 2867}`
- `age` / `embedding`: `{"fifties": 173, "forties": 333, "seventies": 36, "sixties": 69, "teens": 491, "thirties": 720, "twenties": 1737}`
- `age` / `vae_mu`: `{"fifties": 173, "forties": 333, "seventies": 36, "sixties": 69, "teens": 491, "thirties": 720, "twenties": 1737}`
- `accent` / `embedding`: `{"Australian English": 95, "Canadian English": 139, "England English": 409, "Hong Kong English": 42, "India and South Asia (India, Pakistan, Sri Lanka)": 693, "Scottish English": 42, "Southern African (South Africa, Zimbabwe, Namibia)": 60, "United States English": 1401}`
- `accent` / `vae_mu`: `{"Australian English": 95, "Canadian English": 139, "England English": 409, "Hong Kong English": 42, "India and South Asia (India, Pakistan, Sri Lanka)": 693, "Scottish English": 42, "Southern African (South Africa, Zimbabwe, Namibia)": 60, "United States English": 1401}`

## Recommended Use

- If age/gender are weak here, do not spend more cycles on scalar age/gender control without better labels or a stronger perceptual target.
- If they are separable in raw embeddings but not VAE latents, the bottleneck is likely the current VAE objective/capacity rather than CommonVoice metadata itself.
- If they are separable in both, the next training branch should use this probe to select rows/classes and then run a listening-first metadata-control panel.
