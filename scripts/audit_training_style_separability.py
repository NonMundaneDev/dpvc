"""
Audit whether source training labels are separable before claiming controls.

This script evaluates the original labeled CREMA-D / Expresso training audio
with the same emotion2vec model used for generated-output evaluation. It also
uses the extracted emotion2vec embeddings for a simple held-out nearest-centroid
style classifier, so styles without a direct emotion2vec output label (for
example `whisper`) can still be assessed as separable in emotion2vec space.

The audit is intended to answer Joe Near's May 28 question: which controls are
defensible headline paper/demo claims, and which labels are too weak or
ambiguous in the source data to keep forcing as repair targets?
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf


UNIFIED_STYLES = [
    "anger",
    "confused",
    "disgust",
    "enunciated",
    "fear",
    "happy",
    "neutral",
    "sad",
    "whisper",
]

CREMAD_STYLES = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
EXPRESSO_SUPPORTED_STYLES = ["confused", "enunciated", "happy", "neutral", "sad", "whisper"]
EXPRESSO_MAP = {
    "default": "neutral",
    "confused": "confused",
    "enunciated": "enunciated",
    "happy": "happy",
    "sad": "sad",
    "whisper": "whisper",
}
EXPRESSO_ONLY = {"confused", "enunciated", "whisper"}

STYLE_TO_E2V = {
    "anger": "angry",
    "confused": None,
    "disgust": "disgusted",
    "enunciated": None,
    "fear": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "whisper": None,
}


@dataclass(frozen=True)
class SourceClip:
    dataset: str
    row_index: int
    source_id: str
    speaker_id: str
    source_label: str
    unified_style: str
    audio: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        default="cremad,expresso",
        help="Comma-separated datasets to audit: cremad,expresso",
    )
    parser.add_argument(
        "--cremad-mode",
        choices=["speaker_emotion", "all"],
        default="speaker_emotion",
        help=(
            "CREMA-D row policy. `speaker_emotion` mirrors the existing "
            "OpenVoice extraction: one clip per speaker/emotion."
        ),
    )
    parser.add_argument(
        "--expresso-mode",
        choices=["unified_balanced", "all_supported"],
        default="unified_balanced",
        help=(
            "Expresso row policy. `unified_balanced` mirrors the mixed-data "
            "builder: one row per speaker/shared style and capped "
            "Expresso-only styles."
        ),
    )
    parser.add_argument(
        "--expresso-only-cap",
        type=int,
        default=90,
        help="Cap for Expresso-only styles in unified_balanced mode",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=0,
        help="Optional deterministic cap per unified style after dataset selection (0 = no cap)",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=0,
        help="Optional deterministic cap across all selected rows (0 = no cap)",
    )
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=5,
        help="Minimum rows per class for embedding-space separability classifier",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.25,
        help="Deterministic stratified test fraction for embedding-space classifier",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument(
        "--model",
        default="iic/emotion2vec_plus_large",
        help="funasr emotion2vec model id",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cached HuggingFace datasets/model files where possible",
    )
    parser.add_argument(
        "--out-prefix",
        default="results/training_style_separability",
        help="Output path prefix for rows/by-label/confusion/summary artifacts",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print one inference progress row every N clips (1 = every clip, 0 = only first/last)",
    )
    return parser.parse_args()


def canonical_label(raw: str) -> str:
    if "/" in raw:
        raw = raw.split("/")[-1]
    return raw.strip().lower()


def normalize_dataset_names(raw: str) -> List[str]:
    names = []
    for item in raw.split(","):
        name = item.strip().lower()
        if not name:
            continue
        if name not in {"cremad", "expresso"}:
            raise ValueError(f"Unknown dataset {name!r}; expected cremad or expresso")
        names.append(name)
    if not names:
        raise ValueError("At least one dataset must be selected")
    return names


def load_cremad(mode: str) -> List[SourceClip]:
    from datasets import Audio, load_dataset

    dataset = load_dataset("AbstractTTS/CREMA-D", split="train")
    dataset = dataset.cast_column("audio", Audio(decode=False))

    clips: List[SourceClip] = []
    seen = set()
    for index, row in enumerate(dataset):
        emotion = row.get("major_emotion")
        if emotion not in CREMAD_STYLES:
            continue
        file_id = str(row.get("file", index))
        speaker = file_id.split("_")[0]
        if mode == "speaker_emotion":
            key = (speaker, emotion)
            if key in seen:
                continue
            seen.add(key)
        clips.append(
            SourceClip(
                dataset="CREMA-D",
                row_index=index,
                source_id=file_id,
                speaker_id=str(speaker),
                source_label=str(emotion),
                unified_style=str(emotion),
                audio=row["audio"],
            )
        )
    return clips


def load_expresso(mode: str, expresso_only_cap: int, seed: int) -> List[SourceClip]:
    from datasets import Audio, load_dataset

    dataset = load_dataset("ylacombe/expresso", "read", split="train")
    dataset = dataset.cast_column("audio", Audio(decode=False))

    candidates: Dict[str, List[SourceClip]] = defaultdict(list)
    seen_shared = set()
    for index, row in enumerate(dataset):
        source_style = row.get("style")
        unified = EXPRESSO_MAP.get(source_style)
        if unified not in EXPRESSO_SUPPORTED_STYLES:
            continue
        speaker = str(row.get("speaker_id", ""))
        if mode == "unified_balanced" and unified not in EXPRESSO_ONLY:
            key = (speaker, unified)
            if key in seen_shared:
                continue
            seen_shared.add(key)
        source_id = f"expresso_{index}_{speaker}_{source_style}"
        candidates[unified].append(
            SourceClip(
                dataset="Expresso",
                row_index=index,
                source_id=source_id,
                speaker_id=speaker,
                source_label=str(source_style),
                unified_style=unified,
                audio=row["audio"],
            )
        )

    rng = random.Random(seed)
    clips: List[SourceClip] = []
    for style in EXPRESSO_SUPPORTED_STYLES:
        rows = list(candidates.get(style, []))
        if mode == "unified_balanced" and style in EXPRESSO_ONLY and expresso_only_cap:
            rng.shuffle(rows)
            rows = rows[:expresso_only_cap]
            rows.sort(key=lambda clip: clip.row_index)
        clips.extend(rows)
    return clips


def cap_rows(
    clips: Sequence[SourceClip],
    max_per_label: int,
    max_total: int,
    seed: int,
) -> List[SourceClip]:
    rng = random.Random(seed)
    selected = list(clips)
    if max_per_label:
        by_style: Dict[str, List[SourceClip]] = defaultdict(list)
        for clip in selected:
            by_style[clip.unified_style].append(clip)
        selected = []
        for style in sorted(by_style):
            rows = list(by_style[style])
            rng.shuffle(rows)
            selected.extend(sorted(rows[:max_per_label], key=lambda clip: (clip.dataset, clip.row_index)))
    if max_total and len(selected) > max_total:
        rng.shuffle(selected)
        selected = selected[:max_total]
    return sorted(selected, key=lambda clip: (clip.dataset, clip.unified_style, clip.row_index))


def decode_audio(audio: object) -> Tuple[np.ndarray, int]:
    if not isinstance(audio, dict):
        raise ValueError(f"Expected datasets audio dict, got {type(audio).__name__}")
    if audio.get("bytes") is not None:
        array, sample_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    elif audio.get("path"):
        array, sample_rate = sf.read(audio["path"], dtype="float32")
    else:
        raise ValueError("Audio row has neither bytes nor path")

    if array.ndim > 1:
        array = array.mean(axis=1)
    return np.asarray(array, dtype=np.float32), int(sample_rate)


def load_emotion_model(model_id: str, offline: bool):
    from funasr import AutoModel

    if offline:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return AutoModel(model=model_id, hub="hf", disable_update=True)


def infer_clip(model, clip: SourceClip) -> Dict:
    waveform, sample_rate = decode_audio(clip.audio)
    rec = model.generate(
        waveform,
        granularity="utterance",
        extract_embedding=True,
        audio_fs=sample_rate,
    )
    result = rec[0]
    labels = [canonical_label(label) for label in result["labels"]]
    scores = [float(score) for score in result["scores"]]
    top_index = int(np.argmax(scores))
    return {
        "predicted": labels[top_index],
        "score": scores[top_index],
        "labels": labels,
        "scores": scores,
        "embedding": np.asarray(result["feats"], dtype=np.float32).reshape(-1),
    }


def stratified_split(labels: Sequence[str], test_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    train: List[int] = []
    test: List[int] = []
    by_label: Dict[str, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[label].append(index)
    for label, indices in sorted(by_label.items()):
        if len(indices) < 2:
            raise ValueError(f"Class {label!r} has fewer than two rows")
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_fraction))
        n_test = min(n_test, len(shuffled) - 1)
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])
    return sorted(train), sorted(test)


def nearest_centroid_predict(
    train_x: np.ndarray,
    train_y: Sequence[str],
    test_x: np.ndarray,
) -> List[str]:
    labels = sorted(set(train_y))
    centroids = []
    for label in labels:
        indices = [index for index, value in enumerate(train_y) if value == label]
        centroids.append(train_x[indices].mean(axis=0))
    centroid_matrix = np.stack(centroids, axis=0)
    distances = ((test_x[:, None, :] - centroid_matrix[None, :, :]) ** 2).sum(axis=2)
    return [labels[index] for index in distances.argmin(axis=1).tolist()]


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std


def precision_recall_f1(true_y: Sequence[str], pred_y: Sequence[str]) -> Dict[str, Dict[str, float]]:
    labels = sorted(set(true_y) | set(pred_y))
    metrics = {}
    for label in labels:
        tp = sum(1 for true, pred in zip(true_y, pred_y) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(true_y, pred_y) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(true_y, pred_y) if true == label and pred != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for true in true_y if true == label),
            "predicted": sum(1 for pred in pred_y if pred == label),
        }
    return metrics


def run_embedding_classifier(
    row_records: Sequence[Dict],
    min_class_count: int,
    test_fraction: float,
    seed: int,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict]]:
    labels = [row["unified_style"] for row in row_records]
    counts = Counter(labels)
    eligible = {label for label, count in counts.items() if count >= min_class_count}
    indices = [index for index, label in enumerate(labels) if label in eligible]
    if len(eligible) < 2:
        return {}, []

    selected_labels = [labels[index] for index in indices]
    features = np.stack([row_records[index]["embedding"] for index in indices], axis=0)
    train_idx, test_idx = stratified_split(selected_labels, test_fraction, seed)
    train_x_raw = features[train_idx]
    test_x_raw = features[test_idx]
    train_y = [selected_labels[index] for index in train_idx]
    test_y = [selected_labels[index] for index in test_idx]
    train_x, test_x = standardize(train_x_raw, test_x_raw)
    pred_y = nearest_centroid_predict(train_x, train_y, test_x)
    metrics = precision_recall_f1(test_y, pred_y)
    confusion = [
        {"confusion_type": "embedding_classifier", "target": true, "predicted": pred}
        for true, pred in zip(test_y, pred_y)
    ]
    return metrics, confusion


def build_direct_metrics(row_records: Sequence[Dict]) -> Tuple[Dict[str, Dict[str, float]], List[Dict]]:
    evaluable = [row for row in row_records if row["direct_evaluable"] == "1"]
    true_y = [row["target_e2v"] for row in evaluable]
    pred_y = [row["predicted"] for row in evaluable]
    e2v_metrics = precision_recall_f1(true_y, pred_y) if evaluable else {}
    by_style: Dict[str, Dict[str, float]] = {}
    for style in UNIFIED_STYLES:
        style_rows = [row for row in row_records if row["unified_style"] == style]
        direct_rows = [row for row in style_rows if row["direct_evaluable"] == "1"]
        correct = sum(1 for row in direct_rows if row["direct_match"] == "1")
        by_style[style] = {
            "support": len(style_rows),
            "direct_support": len(direct_rows),
            "direct_correct": correct,
            "direct_recall": correct / len(direct_rows) if direct_rows else None,
        }
    confusion = [
        {
            "confusion_type": "emotion2vec_direct",
            "target": row["target_e2v"],
            "predicted": row["predicted"],
        }
        for row in evaluable
    ]
    return {"by_e2v_label": e2v_metrics, "by_style": by_style}, confusion


def style_verdict(direct_recall: Optional[float], embedding_f1: Optional[float]) -> str:
    signals = [value for value in [direct_recall, embedding_f1] if value is not None]
    if not signals:
        return "needs non-emotion perceptual review"
    best = max(signals)
    if best >= 0.70:
        return "headline candidate"
    if best >= 0.45:
        return "supported but quality-sensitive"
    if best >= 0.20:
        return "diagnostic / weak"
    return "limitation / do not force"


def write_rows_csv(rows: Sequence[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "row_index",
        "source_id",
        "speaker_id",
        "source_label",
        "unified_style",
        "target_e2v",
        "predicted",
        "score",
        "direct_evaluable",
        "direct_match",
        "labels_json",
        "scores_json",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_by_label_csv(
    rows: Sequence[Dict],
    direct: Dict,
    embedding_metrics: Dict[str, Dict[str, float]],
    path: Path,
) -> List[Dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_rows = []
    direct_by_style = direct["by_style"]
    for style in UNIFIED_STYLES:
        style_rows = [row for row in rows if row["unified_style"] == style]
        if not style_rows:
            continue
        direct_row = direct_by_style.get(style, {})
        embedding_row = embedding_metrics.get(style, {})
        direct_recall = direct_row.get("direct_recall")
        embedding_f1 = embedding_row.get("f1")
        output_rows.append(
            {
                "unified_style": style,
                "support": direct_row.get("support", 0),
                "datasets": ",".join(sorted({row["dataset"] for row in style_rows})),
                "target_e2v": STYLE_TO_E2V.get(style) or "",
                "direct_support": direct_row.get("direct_support", 0),
                "direct_correct": direct_row.get("direct_correct", 0),
                "direct_recall": "" if direct_recall is None else f"{direct_recall:.4f}",
                "embedding_precision": format_optional_float(embedding_row.get("precision")),
                "embedding_recall": format_optional_float(embedding_row.get("recall")),
                "embedding_f1": format_optional_float(embedding_f1),
                "embedding_test_support": int(embedding_row.get("support", 0)),
                "verdict": style_verdict(direct_recall, embedding_f1),
            }
        )

    fieldnames = [
        "unified_style",
        "support",
        "datasets",
        "target_e2v",
        "direct_support",
        "direct_correct",
        "direct_recall",
        "embedding_precision",
        "embedding_recall",
        "embedding_f1",
        "embedding_test_support",
        "verdict",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    return output_rows


def write_confusion_csv(confusion_rows: Sequence[Dict], path: Path) -> None:
    counts = Counter(
        (row["confusion_type"], row["target"], row["predicted"]) for row in confusion_rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["confusion_type", "target", "predicted", "count"],
            lineterminator="\n",
        )
        writer.writeheader()
        for (confusion_type, target, predicted), count in sorted(counts.items()):
            writer.writerow(
                {
                    "confusion_type": confusion_type,
                    "target": target,
                    "predicted": predicted,
                    "count": count,
                }
            )


def format_optional_float(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.4f}"


def write_summary(
    by_label_rows: Sequence[Dict],
    rows_path: Path,
    by_label_path: Path,
    confusion_path: Path,
    path: Path,
    args: argparse.Namespace,
) -> None:
    headline = [row for row in by_label_rows if row["verdict"] == "headline candidate"]
    limitations = [row for row in by_label_rows if row["verdict"] == "limitation / do not force"]
    lines = [
        "# Training Style Separability Audit",
        "",
        "This audit evaluates source training clips, not generated outputs. It is a",
        "control-selection diagnostic: labels that are weak in the source data should",
        "not become open-ended repair targets without stronger data.",
        "",
        "## Configuration",
        "",
        f"- Datasets: `{args.datasets}`",
        f"- CREMA-D mode: `{args.cremad_mode}`",
        f"- Expresso mode: `{args.expresso_mode}`",
        f"- Expresso-only cap: `{args.expresso_only_cap}`",
        f"- Max per label: `{args.max_per_label or 'none'}`",
        f"- Max total: `{args.max_total or 'none'}`",
        f"- Minimum classifier class count: `{args.min_class_count}`",
        f"- Seed: `{args.seed}`",
        f"- Progress print interval: `{args.progress_every}`",
        "",
        "## Artifacts",
        "",
        f"- Rows: `{rows_path}`",
        f"- Per-label summary: `{by_label_path}`",
        f"- Confusion tables: `{confusion_path}`",
        "",
        "## Per-Style Results",
        "",
        "| Style | Support | Datasets | emotion2vec target | Direct recall | Embedding F1 | Verdict |",
        "| --- | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for row in by_label_rows:
        lines.append(
            "| {style} | {support} | {datasets} | {target} | {direct} | {embed} | {verdict} |".format(
                style=row["unified_style"],
                support=row["support"],
                datasets=row["datasets"],
                target=row["target_e2v"] or "n/a",
                direct=row["direct_recall"] or "n/a",
                embed=row["embedding_f1"] or "n/a",
                verdict=row["verdict"],
            )
        )

    lines.extend(["", "## Initial Recommendation", ""])
    if headline:
        lines.append("Headline candidates from this audit:")
        lines.append("")
        for row in headline:
            lines.append(f"- `{row['unified_style']}` ({row['verdict']})")
    else:
        lines.append("No label cleared the headline-candidate threshold in this run.")

    if limitations:
        lines.append("")
        lines.append("Labels that should not be forced without stronger data:")
        lines.append("")
        for row in limitations:
            lines.append(f"- `{row['unified_style']}` ({row['verdict']})")

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `Direct recall` only applies to styles with an emotion2vec output label.",
            "- `Embedding F1` is a held-out nearest-centroid classifier over emotion2vec embeddings.",
            "- A high embedding F1 for styles such as `whisper` means the style is separable in emotion2vec space even without a direct emotion2vec class.",
            "- This audit does not prove generated control quality; it decides which source labels are fair to claim or repair.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run(args: argparse.Namespace) -> Dict[str, Path]:
    if args.offline:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    dataset_names = normalize_dataset_names(args.datasets)
    clips: List[SourceClip] = []
    if "cremad" in dataset_names:
        clips.extend(load_cremad(args.cremad_mode))
    if "expresso" in dataset_names:
        clips.extend(load_expresso(args.expresso_mode, args.expresso_only_cap, args.seed))
    clips = cap_rows(clips, args.max_per_label, args.max_total, args.seed)
    if not clips:
        raise SystemExit("No source clips selected for audit")

    print(f"Selected {len(clips)} source clips")
    for style, count in sorted(Counter(clip.unified_style for clip in clips).items()):
        print(f"  {style:12s} {count:5d}")

    print(f"\nLoading {args.model}...")
    model = load_emotion_model(args.model, args.offline)
    print("Model loaded.\n")

    row_records = []
    for index, clip in enumerate(clips, 1):
        pred = infer_clip(model, clip)
        target_e2v = STYLE_TO_E2V.get(clip.unified_style)
        direct_evaluable = target_e2v is not None
        direct_match = bool(direct_evaluable and pred["predicted"] == target_e2v)
        row = {
            "dataset": clip.dataset,
            "row_index": clip.row_index,
            "source_id": clip.source_id,
            "speaker_id": clip.speaker_id,
            "source_label": clip.source_label,
            "unified_style": clip.unified_style,
            "target_e2v": target_e2v or "",
            "predicted": pred["predicted"],
            "score": f"{pred['score']:.4f}",
            "direct_evaluable": "1" if direct_evaluable else "0",
            "direct_match": "1" if direct_match else "0" if direct_evaluable else "",
            "labels_json": json.dumps(pred["labels"]),
            "scores_json": json.dumps([round(score, 6) for score in pred["scores"]]),
            "embedding": pred["embedding"],
        }
        row_records.append(row)
        should_print = index == 1 or index == len(clips)
        if args.progress_every > 0 and index % args.progress_every == 0:
            should_print = True
        if should_print:
            print(
                f"[{index:4d}/{len(clips)}] {clip.dataset:8s} {clip.unified_style:12s} "
                f"-> {pred['predicted']:12s} ({pred['score']:.2f})"
            )

    direct_metrics, direct_confusion = build_direct_metrics(row_records)
    embedding_metrics, embedding_confusion = run_embedding_classifier(
        row_records,
        min_class_count=args.min_class_count,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    out_prefix = Path(args.out_prefix)
    rows_path = out_prefix.with_name(out_prefix.name + "_rows.csv")
    by_label_path = out_prefix.with_name(out_prefix.name + "_by_label.csv")
    confusion_path = out_prefix.with_name(out_prefix.name + "_confusion.csv")
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.md")

    write_rows_csv(row_records, rows_path)
    by_label_rows = write_by_label_csv(row_records, direct_metrics, embedding_metrics, by_label_path)
    write_confusion_csv([*direct_confusion, *embedding_confusion], confusion_path)
    write_summary(by_label_rows, rows_path, by_label_path, confusion_path, summary_path, args)

    print("\nArtifacts written:")
    print(f"  {rows_path}")
    print(f"  {by_label_path}")
    print(f"  {confusion_path}")
    print(f"  {summary_path}")

    return {
        "rows": rows_path,
        "by_label": by_label_path,
        "confusion": confusion_path,
        "summary": summary_path,
    }


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
