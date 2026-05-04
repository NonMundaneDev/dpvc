"""
Combine CommonVoice pseudo labels from emotion2vec and latent prototypes.

The mixed-data branch now has two useful but incomplete teachers:

- emotion2vec: better precision for canonical emotion classes, but no labels
  for confused, enunciated, or whisper.
- combined-VAE latent prototypes: broader 9-style coverage, but more risk of
  over-steering canonical emotions.

This script creates a reusable CommonVoice artifact with row-level
`pseudo_style_selected_*` decisions that downstream mixed-data builders can
consume via `--acceptance-policy artifact_selected`.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch


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

DEFAULT_CANONICAL_STYLES = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
DEFAULT_PROTOTYPE_EXTRA_STYLES = ["confused", "enunciated", "whisper"]


def parse_style_list(raw: str, default: list[str]) -> list[str]:
    if not raw.strip():
        return list(default)
    styles = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [style for style in styles if style not in UNIFIED_STYLES]
    if unknown:
        raise ValueError(f"Unknown styles: {unknown}")
    return styles


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--emotion2vec",
        default="embeddings/openvoice_commonvoice_cv500_pseudo_filtered.pt",
        help="Filtered emotion2vec CommonVoice artifact",
    )
    ap.add_argument(
        "--prototype",
        default="embeddings/openvoice_commonvoice_cv500_pseudo_prototype_filtered.pt",
        help="Filtered latent-prototype CommonVoice artifact",
    )
    ap.add_argument(
        "--output",
        default="embeddings/openvoice_commonvoice_cv500_pseudo_hybrid_extra_priority.pt",
        help="Output artifact with hybrid pseudo-label selection decisions",
    )
    ap.add_argument(
        "--policy",
        default="prototype_extra_priority",
        choices=[
            "prototype_extra_priority",
            "emotion_priority",
            "prototype_extra_priority_neutral_sad_only",
        ],
        help=(
            "Hybrid arbitration policy. prototype_extra_priority gives "
            "prototype-selected extra styles priority because emotion2vec has no "
            "direct labels for them."
        ),
    )
    ap.add_argument(
        "--canonical-styles",
        default=",".join(DEFAULT_CANONICAL_STYLES),
        help="Comma-separated styles handled by the emotion2vec component",
    )
    ap.add_argument(
        "--prototype-extra-styles",
        default=",".join(DEFAULT_PROTOTYPE_EXTRA_STYLES),
        help="Comma-separated styles handled by the prototype component",
    )
    return ap.parse_args()


def as_bool_mask(data, key: str, n_rows: int) -> list[bool]:
    mask = data.get(key)
    if mask is None:
        return [False] * n_rows
    if torch.is_tensor(mask):
        return [bool(value) for value in mask.view(-1).tolist()]
    return [bool(value) for value in mask]


def scalar_at(data, key: str, idx: int, default=0.0):
    values = data.get(key)
    if values is None or idx >= len(values):
        return default
    value = values[idx]
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return float(value.reshape(-1)[0].item())
    return float(value)


def value_at(data, key: str, idx: int, default=None):
    values = data.get(key)
    if values is None or idx >= len(values):
        return default
    return values[idx]


def build_index_map(reference_paths, other_paths):
    if reference_paths == other_paths:
        return list(range(len(reference_paths)))
    by_path = {str(path): idx for idx, path in enumerate(other_paths)}
    missing = [str(path) for path in reference_paths if str(path) not in by_path]
    if missing:
        raise ValueError(
            f"Artifacts do not align: {len(missing)} reference clip paths are missing "
            f"from the secondary artifact; first missing path: {missing[0]}"
        )
    return [by_path[str(path)] for path in reference_paths]


def selected_style(data, mask, idx: int):
    if not mask[idx]:
        return None
    return value_at(data, "pseudo_style_selected", idx)


def choose_label(args, emotion_style, prototype_style, canonical_styles, extra_styles):
    if args.policy == "prototype_extra_priority":
        if prototype_style in extra_styles:
            return "prototype", prototype_style, "prototype_extra_priority"
        if emotion_style in canonical_styles:
            return "emotion2vec", emotion_style, "emotion2vec_canonical"
        return None, None, "unselected"

    if args.policy == "emotion_priority":
        if emotion_style in canonical_styles:
            return "emotion2vec", emotion_style, "emotion2vec_canonical"
        if prototype_style in extra_styles:
            return "prototype", prototype_style, "prototype_extra"
        return None, None, "unselected"

    if args.policy == "prototype_extra_priority_neutral_sad_only":
        if prototype_style in extra_styles and emotion_style in {None, "neutral", "sad"}:
            return "prototype", prototype_style, "prototype_extra_priority_neutral_sad"
        if emotion_style in canonical_styles:
            return "emotion2vec", emotion_style, "emotion2vec_canonical"
        if prototype_style in extra_styles:
            return "prototype", prototype_style, "prototype_extra"
        return None, None, "unselected"

    raise ValueError(f"Unsupported policy: {args.policy}")


def main():
    args = parse_args()
    canonical_styles = set(parse_style_list(args.canonical_styles, DEFAULT_CANONICAL_STYLES))
    extra_styles = set(parse_style_list(args.prototype_extra_styles, DEFAULT_PROTOTYPE_EXTRA_STYLES))

    overlap = canonical_styles & extra_styles
    if overlap:
        raise ValueError(f"Styles cannot be both canonical and prototype-extra: {sorted(overlap)}")

    emotion_data = torch.load(args.emotion2vec, weights_only=False, map_location="cpu")
    prototype_data = torch.load(args.prototype, weights_only=False, map_location="cpu")
    n_rows = len(emotion_data.get("clip_paths", []))
    proto_index = build_index_map(
        [str(path) for path in emotion_data.get("clip_paths", [])],
        [str(path) for path in prototype_data.get("clip_paths", [])],
    )

    emotion_mask = as_bool_mask(emotion_data, "pseudo_style_selected_mask", n_rows)
    prototype_mask_all = as_bool_mask(
        prototype_data,
        "pseudo_style_selected_mask",
        len(prototype_data.get("clip_paths", [])),
    )

    chosen_style = [None] * n_rows
    chosen_confidence = [0.0] * n_rows
    chosen_source = ["unselected"] * n_rows
    chosen_reason = ["unselected"] * n_rows
    chosen_raw_label = [None] * n_rows
    chosen_topk_labels = [None] * n_rows
    chosen_topk_scores = [None] * n_rows
    chosen_score_map = [None] * n_rows
    selected_mask = [False] * n_rows

    selected_counts = Counter()
    source_counts = Counter()
    emotion_component_counts = Counter()
    prototype_component_counts = Counter()
    conflict_counts = Counter()

    for idx in range(n_rows):
        proto_idx = proto_index[idx]
        emotion_style = selected_style(emotion_data, emotion_mask, idx)
        prototype_style = selected_style(prototype_data, prototype_mask_all, proto_idx)

        if emotion_style in canonical_styles:
            emotion_component_counts[emotion_style] += 1
        if prototype_style in extra_styles:
            prototype_component_counts[prototype_style] += 1
        if emotion_style and prototype_style and emotion_style != prototype_style:
            conflict_counts[f"emotion={emotion_style}|prototype={prototype_style}"] += 1

        source, style, reason = choose_label(
            args,
            emotion_style,
            prototype_style,
            canonical_styles,
            extra_styles,
        )
        if style is None:
            source_counts["unselected"] += 1
            continue

        if source == "emotion2vec":
            confidence = scalar_at(emotion_data, "pseudo_style_selected_confidence", idx)
            raw_label = value_at(emotion_data, "pseudo_style_raw_label", idx)
            topk_labels = value_at(emotion_data, "pseudo_style_topk_labels", idx)
            topk_scores = value_at(emotion_data, "pseudo_style_topk_scores", idx)
            score_map = value_at(emotion_data, "pseudo_style_score_map", idx)
        elif source == "prototype":
            confidence = scalar_at(prototype_data, "pseudo_style_selected_confidence", proto_idx)
            raw_label = value_at(prototype_data, "pseudo_style_raw_label", proto_idx)
            topk_labels = value_at(prototype_data, "pseudo_style_topk_labels", proto_idx)
            topk_scores = value_at(prototype_data, "pseudo_style_topk_scores", proto_idx)
            score_map = value_at(prototype_data, "pseudo_style_score_map", proto_idx)
        else:
            raise ValueError(f"Unexpected source: {source}")

        chosen_style[idx] = style
        chosen_confidence[idx] = float(confidence)
        chosen_source[idx] = source
        chosen_reason[idx] = reason
        chosen_raw_label[idx] = raw_label
        chosen_topk_labels[idx] = topk_labels
        chosen_topk_scores[idx] = topk_scores
        chosen_score_map[idx] = score_map
        selected_mask[idx] = True
        selected_counts[style] += 1
        source_counts[source] += 1

    report = {
        "model": "hybrid_emotion2vec_canonical_latent_prototype_extra",
        "policy": args.policy,
        "emotion2vec_artifact": args.emotion2vec,
        "prototype_artifact": args.prototype,
        "canonical_styles": sorted(canonical_styles),
        "prototype_extra_styles": sorted(extra_styles),
        "rows_annotated": int(n_rows),
        "selected_counts": dict(selected_counts),
        "source_counts": dict(source_counts),
        "emotion_component_selected_counts": dict(emotion_component_counts),
        "prototype_extra_component_selected_counts": dict(prototype_component_counts),
        "conflict_counts": dict(conflict_counts),
        "emotion2vec_teacher": emotion_data.get("pseudo_style_teacher"),
        "prototype_teacher": prototype_data.get("pseudo_style_teacher"),
        "emotion2vec_filter_report": emotion_data.get("pseudo_style_filter_report"),
        "prototype_filter_report": prototype_data.get("pseudo_style_filter_report"),
    }

    enriched = dict(emotion_data)
    enriched["pseudo_style"] = chosen_style
    enriched["pseudo_style_confidence"] = chosen_confidence
    enriched["pseudo_style_raw_label"] = chosen_raw_label
    enriched["pseudo_style_topk_labels"] = chosen_topk_labels
    enriched["pseudo_style_topk_scores"] = chosen_topk_scores
    enriched["pseudo_style_score_map"] = chosen_score_map
    enriched["pseudo_style_source"] = report["model"]
    enriched["pseudo_style_component_source"] = chosen_source
    enriched["pseudo_style_selected"] = chosen_style
    enriched["pseudo_style_selected_mask"] = torch.tensor(selected_mask, dtype=torch.bool)
    enriched["pseudo_style_selected_reason"] = chosen_reason
    enriched["pseudo_style_selected_confidence"] = torch.tensor(
        chosen_confidence,
        dtype=torch.float32,
    ).unsqueeze(1)
    enriched["pseudo_style_teacher"] = {
        "model": report["model"],
        "policy": args.policy,
        "emotion2vec_artifact": args.emotion2vec,
        "prototype_artifact": args.prototype,
        "canonical_styles": sorted(canonical_styles),
        "prototype_extra_styles": sorted(extra_styles),
    }
    enriched["pseudo_style_report"] = report
    enriched["pseudo_style_filter_report"] = {
        "acceptance_policy": "hybrid_teacher",
        "policy": args.policy,
        "selected_counts": dict(selected_counts),
        "source_counts": dict(source_counts),
        "canonical_styles": sorted(canonical_styles),
        "prototype_extra_styles": sorted(extra_styles),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(enriched, output_path)

    print(f"Saved hybrid CommonVoice pseudo-label artifact to {output_path}")
    print(f"Policy: {args.policy}")
    print("Selected counts:")
    for style in UNIFIED_STYLES:
        count = selected_counts.get(style, 0)
        if count:
            print(f"  {style:11s}: {count}")
    print("Source counts:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source:11s}: {count}")


if __name__ == "__main__":
    main()
