"""
Probe whether Common Voice metadata is separable in embedding or VAE-latent space.

This is a diagnostic, not a new training recipe. It answers a narrower question:
do labels such as gender, age, or accent occupy recoverable structure in the
current OpenVoice embedding artifacts before we keep trying to steer them as
explicit controls?
"""

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from dpvc.model_embedding_vae import VariationalAutoencoder


GENDER_NORMALIZATION = {
    "male": "male",
    "male_masculine": "male",
    "female": "female",
    "female_feminine": "female",
}

AGE_NORMALIZATION = {
    "fourties": "forties",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe CommonVoice metadata separability in embedding/latent space"
    )
    parser.add_argument(
        "--artifact",
        required=True,
        help="CommonVoice or mixed-data .pt artifact containing data plus metadata lists",
    )
    parser.add_argument(
        "--fields",
        default="gender,age,accent",
        help="Comma-separated metadata fields to probe",
    )
    parser.add_argument(
        "--vae-checkpoint",
        default=None,
        help="Optional VAE checkpoint. If supplied, also probes encoder mu latents.",
    )
    parser.add_argument(
        "--latent-dims",
        type=int,
        default=None,
        help="Override latent dims for --vae-checkpoint. Otherwise inferred from state_dict.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional deterministic row cap after known-label filtering (0 = all rows)",
    )
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=20,
        help="Minimum rows required per class after normalization",
    )
    parser.add_argument(
        "--top-k-classes",
        type=int,
        default=12,
        help="Keep at most this many most common classes per field (0 = no cap)",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.25,
        help="Stratified test fraction",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=100,
        help="Number of train-label permutations for a simple null baseline",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed",
    )
    parser.add_argument(
        "--out-csv",
        default="results/commonvoice_metadata_separability_probe.csv",
        help="Per-field CSV output path",
    )
    parser.add_argument(
        "--out-md",
        default="results/commonvoice_metadata_separability_probe.md",
        help="Markdown summary output path",
    )
    return parser.parse_args()


def clean_value(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    return text


def normalize_metadata(field: str, value) -> Optional[str]:
    cleaned = clean_value(value)
    if cleaned is None:
        return None
    if field == "gender":
        return GENDER_NORMALIZATION.get(cleaned, cleaned)
    if field == "age":
        return AGE_NORMALIZATION.get(cleaned, cleaned)
    if field == "accent":
        return cleaned.strip()
    return cleaned


def flatten_data(data: torch.Tensor) -> torch.Tensor:
    if data.ndim == 3 and data.shape[-1] == 1:
        data = data.squeeze(-1)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D embeddings after flattening, got shape {tuple(data.shape)}")
    return data.float()


def load_artifact(path: Path) -> Dict:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict):
        raise ValueError(f"{path} is not a dict artifact")
    if "data" not in artifact:
        raise ValueError(f"{path} is missing required key 'data'")
    artifact = dict(artifact)
    artifact["data"] = flatten_data(artifact["data"])
    return artifact


def values_for_field(artifact: Dict, field: str) -> List:
    if field in artifact:
        return list(artifact[field])
    aliases = {
        "age": ["metadata_age_raw"],
        "gender": ["metadata_gender_raw"],
        "accent": ["metadata_accent_raw", "accents"],
    }
    for key in aliases.get(field, []):
        if key in artifact:
            return list(artifact[key])
    return [None] * int(artifact["data"].shape[0])


def select_labeled_rows(
    values: Sequence,
    field: str,
    min_class_count: int,
    top_k_classes: int,
    max_rows: int,
    seed: int,
) -> Tuple[List[int], List[str], Counter]:
    labels_by_index = []
    for index, value in enumerate(values):
        label = normalize_metadata(field, value)
        if label is not None:
            labels_by_index.append((index, label))

    raw_counts = Counter(label for _, label in labels_by_index)
    eligible = {
        label
        for label, count in raw_counts.most_common(top_k_classes or None)
        if count >= min_class_count
    }
    filtered = [(index, label) for index, label in labels_by_index if label in eligible]

    if max_rows and len(filtered) > max_rows:
        rng = random.Random(seed)
        by_label: Dict[str, List[int]] = defaultdict(list)
        for row_pos, (_, label) in enumerate(filtered):
            by_label[label].append(row_pos)
        sampled_positions = []
        for label, positions in sorted(by_label.items()):
            class_budget = max(1, round(max_rows * len(positions) / len(filtered)))
            rng.shuffle(positions)
            sampled_positions.extend(positions[:class_budget])
        if len(sampled_positions) > max_rows:
            rng.shuffle(sampled_positions)
            sampled_positions = sampled_positions[:max_rows]
        filtered = [filtered[pos] for pos in sorted(sampled_positions)]

    indices = [index for index, _ in filtered]
    labels = [label for _, label in filtered]
    return indices, labels, raw_counts


def standardize(train: torch.Tensor, test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return (train - mean) / std, (test - mean) / std


def stratified_split(
    labels: Sequence[str],
    test_fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    by_label: Dict[str, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[label].append(index)

    train_indices = []
    test_indices = []
    for label, indices in sorted(by_label.items()):
        if len(indices) < 2:
            raise ValueError(f"Class {label!r} has fewer than 2 rows after filtering")
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * test_fraction)))
        n_test = min(n_test, len(shuffled) - 1)
        test_indices.extend(shuffled[:n_test])
        train_indices.extend(shuffled[n_test:])

    train_indices.sort()
    test_indices.sort()
    return train_indices, test_indices


def nearest_centroid_predict(
    train_x: torch.Tensor,
    train_y: Sequence[str],
    test_x: torch.Tensor,
) -> List[str]:
    labels = sorted(set(train_y))
    centroids = []
    for label in labels:
        label_indices = [index for index, y in enumerate(train_y) if y == label]
        centroids.append(train_x[label_indices].mean(dim=0))
    centroid_tensor = torch.stack(centroids, dim=0)
    distances = torch.cdist(test_x, centroid_tensor)
    predictions = distances.argmin(dim=1).tolist()
    return [labels[index] for index in predictions]


def accuracy_score(true_y: Sequence[str], pred_y: Sequence[str]) -> float:
    if not true_y:
        return 0.0
    correct = sum(1 for true, pred in zip(true_y, pred_y) if true == pred)
    return correct / len(true_y)


def macro_f1_score(true_y: Sequence[str], pred_y: Sequence[str]) -> float:
    labels = sorted(set(true_y) | set(pred_y))
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(1 for true, pred in zip(true_y, pred_y) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(true_y, pred_y) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(true_y, pred_y) if true == label and pred != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores)


def majority_predictions(train_y: Sequence[str], n: int) -> List[str]:
    majority = Counter(train_y).most_common(1)[0][0]
    return [majority] * n


def centroid_separation_ratio(features: torch.Tensor, labels: Sequence[str]) -> float:
    unique = sorted(set(labels))
    if len(unique) < 2:
        return 0.0
    centroids = []
    within_distances = []
    for label in unique:
        indices = [index for index, y in enumerate(labels) if y == label]
        class_x = features[indices]
        centroid = class_x.mean(dim=0)
        centroids.append(centroid)
        within_distances.extend(torch.norm(class_x - centroid, dim=1).tolist())
    centroid_tensor = torch.stack(centroids, dim=0)
    pairwise = torch.pdist(centroid_tensor)
    between = float(pairwise.mean().item()) if pairwise.numel() else 0.0
    within = sum(within_distances) / len(within_distances) if within_distances else 0.0
    if within <= 1e-8:
        return 0.0
    return between / within


def evaluate_feature_space(
    features: torch.Tensor,
    labels: Sequence[str],
    test_fraction: float,
    seed: int,
    permutations: int,
) -> Dict[str, float]:
    if len(set(labels)) < 2:
        raise ValueError("Need at least two metadata classes to probe separability")

    train_idx, test_idx = stratified_split(labels, test_fraction, seed)
    train_x_raw = features[train_idx]
    test_x_raw = features[test_idx]
    train_y = [labels[index] for index in train_idx]
    test_y = [labels[index] for index in test_idx]
    train_x, test_x = standardize(train_x_raw, test_x_raw)

    pred_y = nearest_centroid_predict(train_x, train_y, test_x)
    majority_y = majority_predictions(train_y, len(test_y))

    perm_acc = []
    perm_f1 = []
    rng = random.Random(seed + 137)
    for _ in range(permutations):
        shuffled_labels = list(labels)
        rng.shuffle(shuffled_labels)
        perm_train_idx, perm_test_idx = stratified_split(
            shuffled_labels,
            test_fraction,
            rng.randrange(1_000_000_000),
        )
        perm_train_raw = features[perm_train_idx]
        perm_test_raw = features[perm_test_idx]
        perm_train_y = [shuffled_labels[index] for index in perm_train_idx]
        perm_test_y = [shuffled_labels[index] for index in perm_test_idx]
        perm_train_x, perm_test_x = standardize(perm_train_raw, perm_test_raw)
        perm_pred = nearest_centroid_predict(perm_train_x, perm_train_y, perm_test_x)
        perm_acc.append(accuracy_score(perm_test_y, perm_pred))
        perm_f1.append(macro_f1_score(perm_test_y, perm_pred))

    acc = accuracy_score(test_y, pred_y)
    f1 = macro_f1_score(test_y, pred_y)
    baseline_acc = accuracy_score(test_y, majority_y)
    baseline_f1 = macro_f1_score(test_y, majority_y)
    sep_ratio = centroid_separation_ratio(features, labels)
    if permutations:
        p_acc = (1 + sum(value >= acc for value in perm_acc)) / (permutations + 1)
        p_f1 = (1 + sum(value >= f1 for value in perm_f1)) / (permutations + 1)
        perm_acc_mean = sum(perm_acc) / len(perm_acc)
        perm_f1_mean = sum(perm_f1) / len(perm_f1)
    else:
        p_acc = p_f1 = perm_acc_mean = perm_f1_mean = float("nan")

    return {
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "accuracy": acc,
        "macro_f1": f1,
        "majority_accuracy": baseline_acc,
        "majority_macro_f1": baseline_f1,
        "accuracy_delta_vs_majority": acc - baseline_acc,
        "macro_f1_delta_vs_majority": f1 - baseline_f1,
        "permutation_accuracy_mean": perm_acc_mean,
        "permutation_macro_f1_mean": perm_f1_mean,
        "accuracy_p_value": p_acc,
        "macro_f1_p_value": p_f1,
        "centroid_separation_ratio": sep_ratio,
    }


def infer_vae_shape(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    try:
        latent_dims = int(state_dict["encoder.to_mu.weight"].shape[0])
        input_dim = int(state_dict["encoder.net.0.weight"].shape[1])
    except KeyError as exc:
        raise ValueError("Could not infer VAE shape from checkpoint state_dict") from exc
    return latent_dims, input_dim


def encode_latents(
    features: torch.Tensor,
    checkpoint_path: Path,
    latent_dims: Optional[int] = None,
) -> torch.Tensor:
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise ValueError(f"{checkpoint_path} is not a state_dict checkpoint")
    inferred_latent_dims, input_dim = infer_vae_shape(state_dict)
    latent_dims = latent_dims or inferred_latent_dims
    model = VariationalAutoencoder(input_dim=input_dim, latent_dims=latent_dims)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        mu, _ = model.encoder(features.float())
    return mu


def format_float(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return ""
    return f"{value:.4f}"


def class_count_json(labels: Sequence[str]) -> str:
    return json.dumps(dict(Counter(labels).most_common()), sort_keys=True)


def run_probe(args: argparse.Namespace) -> List[Dict]:
    artifact = load_artifact(Path(args.artifact))
    fields = [field.strip() for field in args.fields.split(",") if field.strip()]
    feature_spaces = [("embedding", artifact["data"])]
    if args.vae_checkpoint:
        latents = encode_latents(
            artifact["data"],
            Path(args.vae_checkpoint),
            latent_dims=args.latent_dims,
        )
        feature_spaces.append(("vae_mu", latents))

    rows = []
    for field in fields:
        values = values_for_field(artifact, field)
        indices, labels, raw_counts = select_labeled_rows(
            values=values,
            field=field,
            min_class_count=args.min_class_count,
            top_k_classes=args.top_k_classes,
            max_rows=args.max_rows,
            seed=args.seed,
        )
        if len(set(labels)) < 2:
            rows.append({
                "field": field,
                "feature_space": "all",
                "status": "skipped",
                "reason": "fewer_than_two_classes_after_filtering",
                "raw_known": sum(raw_counts.values()),
                "raw_class_counts": json.dumps(dict(raw_counts.most_common()), sort_keys=True),
            })
            continue

        for feature_name, features in feature_spaces:
            selected_features = features[indices]
            metrics = evaluate_feature_space(
                features=selected_features,
                labels=labels,
                test_fraction=args.test_fraction,
                seed=args.seed,
                permutations=args.permutations,
            )
            rows.append({
                "field": field,
                "feature_space": feature_name,
                "status": "ok",
                "n_known_raw": sum(raw_counts.values()),
                "n_used": len(labels),
                "n_classes": len(set(labels)),
                "class_counts": class_count_json(labels),
                **metrics,
            })
    return rows


def write_csv(rows: Sequence[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "field",
        "feature_space",
        "status",
        "reason",
        "n_known_raw",
        "raw_known",
        "n_used",
        "n_classes",
        "n_train",
        "n_test",
        "class_counts",
        "raw_class_counts",
        "accuracy",
        "macro_f1",
        "majority_accuracy",
        "majority_macro_f1",
        "accuracy_delta_vs_majority",
        "macro_f1_delta_vs_majority",
        "permutation_accuracy_mean",
        "permutation_macro_f1_mean",
        "accuracy_p_value",
        "macro_f1_p_value",
        "centroid_separation_ratio",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def verdict(row: Dict) -> str:
    if row.get("status") != "ok":
        return "skipped"
    acc_delta = float(row.get("accuracy_delta_vs_majority", 0.0))
    f1_delta = float(row.get("macro_f1_delta_vs_majority", 0.0))
    p_value = float(row.get("macro_f1_p_value", 1.0))
    if f1_delta >= 0.20 and p_value <= 0.05:
        return "strongly separable"
    if f1_delta >= 0.10 and p_value <= 0.10:
        return "moderately separable"
    if acc_delta > 0.02 or f1_delta > 0.02:
        return "weak / diagnostic only"
    return "not meaningfully separable"


def write_markdown(rows: Sequence[Dict], path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CommonVoice Metadata Separability Probe",
        "",
        f"Artifact: `{args.artifact}`",
        f"VAE checkpoint: `{args.vae_checkpoint or 'not used'}`",
        f"Seed: `{args.seed}`",
        f"Minimum class count: `{args.min_class_count}`",
        f"Top-k classes: `{args.top_k_classes}`",
        f"Permutations: `{args.permutations}`",
        "",
        "## Interpretation Rule",
        "",
        "This probe uses a nearest-centroid classifier with deterministic stratified splits.",
        "A metadata field is useful for control only if it is separable beyond a majority baseline and a train-label permutation baseline.",
        "This does not prove perceptual controllability; it only says whether the metadata has recoverable structure in the tested feature space.",
        "",
        "## Results",
        "",
        "| Field | Space | Used rows | Classes | Accuracy | Macro F1 | Majority F1 | F1 delta | Perm F1 | p(F1) | Sep. ratio | Verdict |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        if row.get("status") != "ok":
            lines.append(
                f"| {row.get('field')} | {row.get('feature_space')} | 0 | 0 |  |  |  |  |  |  |  | skipped: {row.get('reason')} |"
            )
            continue
        lines.append(
            "| {field} | {space} | {n_used} | {n_classes} | {acc} | {f1} | {base_f1} | {f1_delta} | {perm_f1} | {p_f1} | {sep} | {verdict} |".format(
                field=row["field"],
                space=row["feature_space"],
                n_used=row["n_used"],
                n_classes=row["n_classes"],
                acc=format_float(float(row["accuracy"])),
                f1=format_float(float(row["macro_f1"])),
                base_f1=format_float(float(row["majority_macro_f1"])),
                f1_delta=format_float(float(row["macro_f1_delta_vs_majority"])),
                perm_f1=format_float(float(row["permutation_macro_f1_mean"])),
                p_f1=format_float(float(row["macro_f1_p_value"])),
                sep=format_float(float(row["centroid_separation_ratio"])),
                verdict=verdict(row),
            )
        )
    lines.extend([
        "",
        "## Class Counts",
        "",
    ])
    for row in rows:
        counts = row.get("class_counts") or row.get("raw_class_counts")
        lines.append(f"- `{row.get('field')}` / `{row.get('feature_space')}`: `{counts}`")
    lines.extend([
        "",
        "## Recommended Use",
        "",
        "- If age/gender are weak here, do not spend more cycles on scalar age/gender control without better labels or a stronger perceptual target.",
        "- If they are separable in raw embeddings but not VAE latents, the bottleneck is likely the current VAE objective/capacity rather than CommonVoice metadata itself.",
        "- If they are separable in both, the next training branch should use this probe to select rows/classes and then run a listening-first metadata-control panel.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    rows = run_probe(args)
    write_csv(rows, Path(args.out_csv))
    write_markdown(rows, Path(args.out_md), args)
    print(f"Wrote {len(rows)} probe rows to {args.out_csv}")
    print(f"Wrote Markdown summary to {args.out_md}")


if __name__ == "__main__":
    main()
