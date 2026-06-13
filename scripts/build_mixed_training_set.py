"""
Build a sampled mixed-data OpenVoice training artifact from CommonVoice,
CREMA-D, and Expresso.

This script is the first implementation step for the mixed-data bootstrap line:
- keep CommonVoice for speaker breadth,
- keep CREMA-D and Expresso for emotion/style supervision,
- and preserve enough metadata that later schedule and evaluation results are
  interpretable.

The output artifact is designed for `examples/openvoice_train_vae_mixed.py`.
"""

from __future__ import annotations

import argparse
import os
import random
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
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

CREMAD_STYLES = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
EXPRESSO_STYLES = [
    "default",
    "confused",
    "enunciated",
    "happy",
    "laughing",
    "sad",
    "whisper",
    "emphasis",
    "essentials",
    "longform",
    "singing",
]
EXPRESSO_MAP = {
    "default": "neutral",
    "confused": "confused",
    "enunciated": "enunciated",
    "happy": "happy",
    "sad": "sad",
    "whisper": "whisper",
}
EXPRESSO_ONLY = {"confused", "enunciated", "whisper"}
EXPRESSO_SUPPORTED = ["confused", "enunciated", "happy", "neutral", "sad", "whisper"]
DATASETS = ["CommonVoice", "CREMA-D", "Expresso"]
AGE_CONTROL_ORDER = [
    "teens",
    "twenties",
    "thirties",
    "forties",
    "fifties",
    "sixties",
    "seventies",
    "eighties",
    "nineties",
]
AGE_NORMALIZATION = {
    "fourties": "forties",
}
GENDER_NORMALIZATION = {
    "female": "female",
    "female_feminine": "female",
    "male": "male",
    "male_masculine": "male",
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--commonvoice",
        default="embeddings/openvoice_commonvoice_cv500_pseudo.pt",
        help="Pseudo-labeled CommonVoice embedding artifact",
    )
    ap.add_argument(
        "--cremad",
        default="embeddings/openvoice_cremad_emb.pt",
        help="CREMA-D embedding artifact",
    )
    ap.add_argument(
        "--expresso",
        default="embeddings/openvoice_expresso_emb.pt",
        help="Expresso embedding artifact",
    )
    ap.add_argument(
        "--parquet-dir",
        default=None,
        help="Expresso parquet directory. Auto-detected from the HF cache if omitted.",
    )
    ap.add_argument(
        "--output",
        default="embeddings/openvoice_mixed_base.pt",
        help="Output mixed-data artifact (.pt)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for all sampling (default: 42)",
    )
    ap.add_argument(
        "--commonvoice-max-speakers",
        type=int,
        default=None,
        help="Optional cap on CommonVoice speaker count after speaker-first sampling",
    )
    ap.add_argument(
        "--commonvoice-min-clips-per-speaker",
        type=int,
        default=1,
        help="Minimum clips to try to keep per CommonVoice speaker (default: 1)",
    )
    ap.add_argument(
        "--commonvoice-max-clips-per-speaker",
        type=int,
        default=1,
        help="Maximum CommonVoice clips per speaker (default: 1)",
    )
    ap.add_argument(
        "--commonvoice-prefer-pseudo",
        action="store_true",
        help="Prefer pseudo-labeled CommonVoice rows when choosing per-speaker clips",
    )
    ap.add_argument(
        "--commonvoice-preserve-selected-pseudo",
        action="store_true",
        help=(
            "After speaker-first sampling, add remaining accepted CommonVoice "
            "pseudo-labeled rows from sampled speakers so rare selected labels are "
            "not lost to the per-speaker clip cap."
        ),
    )
    ap.add_argument(
        "--pseudo-style-threshold",
        type=float,
        default=0.60,
        help="Confidence threshold for treating CommonVoice pseudo styles as labeled (default: 0.60)",
    )
    ap.add_argument(
        "--pseudo-style-thresholds",
        default="",
        help=(
            "Optional per-style pseudo-label thresholds, e.g. "
            "neutral=0.995,sad=0.98,happy=0.92"
        ),
    )
    ap.add_argument(
        "--commonvoice-style-caps",
        default="",
        help="Optional per-style cap for selected CommonVoice pseudo labels, e.g. neutral=150,sad=120",
    )
    ap.add_argument(
        "--commonvoice-style-targets",
        default="",
        help="Optional per-style selected-row targets, e.g. anger=30,fear=30,happy=80",
    )
    ap.add_argument(
        "--acceptance-policy",
        default="threshold_plus_caps",
        choices=["confidence_only", "threshold_plus_caps", "balanced_targets", "artifact_selected"],
        help="How CommonVoice pseudo labels become selectable rows (default: threshold_plus_caps)",
    )
    ap.add_argument(
        "--expresso-only-cap",
        type=int,
        default=90,
        help="Cap Expresso-only styles (confused/enunciated/whisper) to this many samples (default: 90)",
    )
    ap.add_argument(
        "--pseudo-confidence-scale",
        action="store_true",
        help="Scale CommonVoice pseudo-label row weights by the pseudo-label confidence",
    )
    ap.add_argument(
        "--pseudo-row-weight",
        type=float,
        default=1.0,
        help="Base row-weight multiplier for CommonVoice pseudo labels (default: 1.0)",
    )
    ap.add_argument(
        "--true-row-weight",
        type=float,
        default=1.0,
        help="Base row-weight multiplier for true labels from CREMA-D/Expresso (default: 1.0)",
    )
    args = ap.parse_args()
    if (
        args.commonvoice_max_clips_per_speaker is not None
        and args.commonvoice_max_clips_per_speaker < args.commonvoice_min_clips_per_speaker
    ):
        ap.error(
            "--commonvoice-max-clips-per-speaker must be >= "
            "--commonvoice-min-clips-per-speaker"
        )
    return args


def parse_style_caps(raw: str) -> Dict[str, int]:
    if not raw.strip():
        return {}
    caps = {}
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(f"Expected style cap in name=count form, got: {item}")
        name, value = item.split('=', 1)
        style = name.strip()
        if style not in UNIFIED_STYLES:
            raise ValueError(f"Unknown style in --commonvoice-style-caps: {style}")
        caps[style] = int(value.strip())
    return caps


def parse_style_targets(raw: str) -> Dict[str, int]:
    return parse_style_caps(raw)


def parse_style_thresholds(raw: str, default: float) -> Dict[str, float]:
    thresholds = {style: float(default) for style in UNIFIED_STYLES}
    if not raw.strip():
        return thresholds
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(f"Expected pseudo threshold in name=value form, got: {item}")
        name, value = item.split('=', 1)
        style = name.strip()
        if style not in UNIFIED_STYLES:
            raise ValueError(f"Unknown style in --pseudo-style-thresholds: {style}")
        thresholds[style] = float(value.strip())
    return thresholds


def resolve_expresso_parquet_dir(parquet_dir: Optional[str]) -> Path:
    if parquet_dir:
        path = Path(parquet_dir).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"Expresso parquet dir not found: {path}")
        return path

    matches = sorted(
        glob(
            str(
                Path.home()
                / ".cache"
                / "huggingface"
                / "hub"
                / "datasets--ylacombe--expresso"
                / "snapshots"
                / "*"
                / "read"
            )
        )
    )
    if not matches:
        raise FileNotFoundError(
            "Could not auto-detect an Expresso parquet cache. Pass --parquet-dir explicitly."
        )
    return Path(matches[-1])


def onehot_target(style: str) -> List[float]:
    values = [-1.0] * len(UNIFIED_STYLES)
    values[UNIFIED_STYLES.index(style)] = 1.0
    return values


def build_empty_target() -> List[float]:
    return [0.0] * len(UNIFIED_STYLES)


def clean_metadata_value(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        value = value.reshape(-1)[0].item()
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def normalize_age(value):
    value = clean_metadata_value(value)
    if value is None:
        return None
    return AGE_NORMALIZATION.get(value, value)


def normalize_gender(value):
    value = clean_metadata_value(value)
    if value is None:
        return None
    return GENDER_NORMALIZATION.get(value, value)


def age_to_ordinal_scalar(value):
    value = normalize_age(value)
    if value not in AGE_CONTROL_ORDER:
        return None
    if len(AGE_CONTROL_ORDER) == 1:
        return 0.0
    idx = AGE_CONTROL_ORDER.index(value)
    return -1.0 + (2.0 * idx / (len(AGE_CONTROL_ORDER) - 1))


def gender_to_binary_scalar(value):
    value = normalize_gender(value)
    if value == "female":
        return -1.0
    if value == "male":
        return 1.0
    return None


def payload_value(payload, key, row_idx):
    values = payload.get(key)
    if values is None:
        return None
    try:
        return values[row_idx]
    except (IndexError, TypeError, KeyError):
        return None


def row_metadata_controls(row, source_payload):
    if row["dataset"] != "CommonVoice":
        return {
            "age_raw": None,
            "gender_raw": None,
            "accent_raw": None,
            "age_scalar": 0.0,
            "age_mask": 0.0,
            "gender_scalar": 0.0,
            "gender_mask": 0.0,
        }

    row_idx = row["row_idx"]
    age_raw = clean_metadata_value(payload_value(source_payload, "age", row_idx))
    gender_raw = clean_metadata_value(payload_value(source_payload, "gender", row_idx))
    accent_raw = clean_metadata_value(payload_value(source_payload, "accent", row_idx))
    if accent_raw is None:
        accent_raw = clean_metadata_value(payload_value(source_payload, "accents", row_idx))

    age_scalar = age_to_ordinal_scalar(age_raw)
    gender_scalar = gender_to_binary_scalar(gender_raw)
    return {
        "age_raw": age_raw,
        "gender_raw": gender_raw,
        "accent_raw": accent_raw,
        "age_scalar": float(age_scalar) if age_scalar is not None else 0.0,
        "age_mask": 1.0 if age_scalar is not None else 0.0,
        "gender_scalar": float(gender_scalar) if gender_scalar is not None else 0.0,
        "gender_mask": 1.0 if gender_scalar is not None else 0.0,
    }


def build_metadata_control_report(source_datasets, age_raw, gender_raw, age_mask, gender_mask):
    age_mask_int = [int(value > 0) for value in age_mask]
    gender_mask_int = [int(value > 0) for value in gender_mask]
    both = [int(a and g) for a, g in zip(age_mask_int, gender_mask_int)]
    by_dataset = {}
    for dataset in DATASETS:
        dataset_indexes = [
            idx for idx, source in enumerate(source_datasets)
            if source == dataset
        ]
        if not dataset_indexes:
            continue
        by_dataset[dataset] = {
            "rows": len(dataset_indexes),
            "age_control_rows": int(sum(age_mask_int[idx] for idx in dataset_indexes)),
            "gender_control_rows": int(sum(gender_mask_int[idx] for idx in dataset_indexes)),
            "age_gender_control_rows": int(sum(both[idx] for idx in dataset_indexes)),
        }

    return {
        "schema": {
            "metadata_gender_scalar": "female/female_feminine=-1, male/male_masculine=1",
            "metadata_gender_mask": "1 when gender scalar is known, else 0",
            "metadata_age_ordinal_scalar": (
                "CommonVoice age bucket mapped youngest=-1 to oldest=1"
            ),
            "metadata_age_mask": "1 when age scalar is known, else 0",
        },
        "recommended_control_dims": {
            "gender_dim": 9,
            "age_dim": 10,
            "free_dims": [11, 12, 13, 14],
        },
        "age_control_order": list(AGE_CONTROL_ORDER),
        "gender_normalization": dict(GENDER_NORMALIZATION),
        "rows": len(source_datasets),
        "age_control_rows": int(sum(age_mask_int)),
        "gender_control_rows": int(sum(gender_mask_int)),
        "age_gender_control_rows": int(sum(both)),
        "by_dataset": by_dataset,
        "age_raw_counts": dict(Counter(value for value in age_raw if value is not None)),
        "gender_raw_counts": dict(Counter(value for value in gender_raw if value is not None)),
    }


def load_expresso_metadata(expresso_data, parquet_dir: Path) -> pd.DataFrame:
    files = sorted(
        os.path.join(parquet_dir, name)
        for name in os.listdir(parquet_dir)
        if name.endswith('.parquet')
    )
    if not files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    df = pd.concat(
        [pd.read_parquet(path, columns=['speaker_id', 'style']) for path in files],
        ignore_index=True,
    )
    if len(df) != len(expresso_data['data']):
        if 'ids' not in expresso_data:
            raise ValueError(
                "Expresso parquet metadata length does not match the extracted embeddings, "
                "and no ids field is available to realign them."
            )
        row_indices = [int(x) for x in expresso_data['ids'].view(-1).tolist()]
        df = df.iloc[row_indices].reset_index(drop=True)
        if len(df) != len(expresso_data['data']):
            raise ValueError(
                "Failed to realign Expresso parquet metadata with the extracted embeddings."
            )
    return df


def resolve_cremad_rows(cremad_data):
    rows = []
    for idx in range(len(cremad_data['data'])):
        style = None
        for candidate in CREMAD_STYLES:
            if cremad_data[f'emotion_{candidate}'][idx].item() > 0:
                style = candidate
                break
        if style is None:
            raise ValueError(f"Failed to resolve a CREMA-D style for row {idx}")
        speaker_id = str(int(cremad_data['speakers'][idx].item())) if 'speakers' in cremad_data else f"cremad_{idx}"
        rows.append({
            'dataset': 'CREMA-D',
            'row_idx': idx,
            'speaker_id': speaker_id,
            'clip_path': None,
            'style': style,
            'label_source': 'true',
            'label_confidence': 1.0,
        })
    return rows


def resolve_expresso_rows(expresso_data, parquet_df, cap, seed):
    expresso_by_label = defaultdict(list)
    seen_shared = set()
    for row_idx in range(len(expresso_data['data'])):
        source_style = None
        for style in EXPRESSO_STYLES:
            if expresso_data[f'style_{style}'][row_idx].item() > 0:
                source_style = style
                break
        if source_style not in EXPRESSO_MAP:
            continue
        unified = EXPRESSO_MAP[source_style]
        if unified in EXPRESSO_ONLY:
            expresso_by_label[unified].append(row_idx)
            continue
        speaker_id = str(parquet_df.iloc[row_idx]['speaker_id'])
        key = (speaker_id, unified)
        if key not in seen_shared:
            seen_shared.add(key)
            expresso_by_label[unified].append(row_idx)

    rng = np.random.default_rng(seed)
    rows = []
    for style in EXPRESSO_SUPPORTED:
        indices = list(expresso_by_label.get(style, []))
        if style in EXPRESSO_ONLY and len(indices) > cap:
            indices = sorted(rng.choice(indices, cap, replace=False).tolist())
        for row_idx in indices:
            rows.append({
                'dataset': 'Expresso',
                'row_idx': row_idx,
                'speaker_id': str(parquet_df.iloc[row_idx]['speaker_id']),
                'clip_path': None,
                'style': style,
                'label_source': 'true',
                'label_confidence': 1.0,
            })
    return rows


def _scalar_confidence(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return float(value.reshape(-1)[0].item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def accepted_commonvoice_style(cv_data, row_idx: int, threshold_map: Dict[str, float], acceptance_policy: str):
    if acceptance_policy == "artifact_selected":
        selected_mask = cv_data.get('pseudo_style_selected_mask')
        selected_style = payload_value(cv_data, 'pseudo_style_selected', row_idx)
        selected_reason = payload_value(
            cv_data, 'pseudo_style_selected_reason', row_idx
        ) or 'artifact_missing'
        selected_conf = None
        if selected_mask is None:
            raise ValueError(
                "acceptance-policy=artifact_selected requires pseudo_style_selected_mask "
                "from scripts/filter_commonvoice_pseudolabels.py"
            )
        selected_conf_tensor = cv_data.get('pseudo_style_selected_confidence')
        if selected_conf_tensor is not None:
            selected_conf = _scalar_confidence(selected_conf_tensor[row_idx])
        if bool(selected_mask[row_idx]) and selected_style in UNIFIED_STYLES:
            return selected_style, selected_conf, selected_reason or 'artifact_selected'
        return None, selected_conf, selected_reason or 'artifact_rejected'

    style = payload_value(cv_data, 'pseudo_style', row_idx)
    confidence = payload_value(cv_data, 'pseudo_style_confidence', row_idx)
    if style is None or confidence is None:
        return None, None, 'missing_style_or_confidence'
    confidence = _scalar_confidence(confidence)
    if confidence is None:
        return None, None, 'bad_confidence'
    if style not in UNIFIED_STYLES:
        return None, confidence, 'unmapped_style'
    if confidence < threshold_map.get(style, 0.0):
        return None, confidence, 'below_threshold'
    return style, confidence, 'threshold_pass'


def select_commonvoice_rows(cv_data, args, style_caps, threshold_map):
    style_targets = parse_style_targets(args.commonvoice_style_targets)
    speaker_to_rows = defaultdict(list)
    for row_idx, speaker_id in enumerate(cv_data['speaker_ids']):
        speaker_to_rows[str(speaker_id)].append(row_idx)

    speakers = list(speaker_to_rows.keys())
    random.Random(args.seed).shuffle(speakers)
    if args.commonvoice_max_speakers is not None:
        speakers = speakers[:args.commonvoice_max_speakers]

    rng = random.Random(args.seed)
    selected = []
    selected_style_counts = Counter()
    skipped_by_cap_counts = Counter()
    skipped_by_target_counts = Counter()
    candidate_reason_counts = Counter()
    selected_reason_counts = Counter()
    fallback_forced_unlabeled_counts = Counter()
    selected_labeled_rows = 0
    selected_row_indices = set()
    sampled_speaker_ids = set(speakers)

    for speaker_id in speakers:
        candidates = speaker_to_rows[speaker_id][:]
        rng.shuffle(candidates)

        def sort_key(row_idx):
            style, conf, _reason = accepted_commonvoice_style(
                cv_data, row_idx, threshold_map, args.acceptance_policy
            )
            has_label = style is not None
            is_neutral = style == 'neutral'
            target_remaining = 0
            if style is not None and style in style_targets:
                target_remaining = max(style_targets[style] - selected_style_counts[style], 0)

            if has_label and target_remaining > 0:
                priority = 0
            elif args.commonvoice_prefer_pseudo and has_label and not is_neutral:
                priority = 1
            elif args.commonvoice_prefer_pseudo and has_label and is_neutral:
                priority = 2
            elif has_label:
                priority = 3
            else:
                priority = 4
            return (priority, -target_remaining, -(conf or -1.0), row_idx)

        ordered = sorted(candidates, key=sort_key)
        target_count = len(ordered)
        if args.commonvoice_max_clips_per_speaker is not None:
            target_count = min(target_count, args.commonvoice_max_clips_per_speaker)
        if target_count <= 0:
            continue

        chosen = []
        used = set()
        for row_idx in ordered:
            if len(chosen) >= target_count:
                break
            style, confidence, candidate_reason = accepted_commonvoice_style(
                cv_data, row_idx, threshold_map, args.acceptance_policy
            )
            candidate_reason_counts[candidate_reason] += 1
            if (
                style is not None
                and args.acceptance_policy == 'balanced_targets'
                and style in style_targets
                and selected_style_counts[style] >= style_targets[style]
            ):
                skipped_by_target_counts[style] += 1
                continue
            if style is not None and style in style_caps and selected_style_counts[style] >= style_caps[style]:
                skipped_by_cap_counts[style] += 1
                continue
            selection_reason = 'pseudo_selected' if style is not None else candidate_reason
            chosen.append((row_idx, style, confidence, selection_reason))
            used.add(row_idx)
            if style is not None:
                selected_style_counts[style] += 1
                selected_labeled_rows += 1
            selected_reason_counts[selection_reason] += 1

        if len(chosen) < target_count:
            for row_idx in ordered:
                if row_idx in used:
                    continue
                style, confidence, candidate_reason = accepted_commonvoice_style(
                    cv_data, row_idx, threshold_map, args.acceptance_policy
                )
                selection_reason = 'speaker_breadth_fallback'
                if (
                    style is not None
                    and args.acceptance_policy == 'balanced_targets'
                    and style in style_targets
                    and selected_style_counts[style] >= style_targets[style]
                ):
                    skipped_by_target_counts[style] += 1
                    fallback_forced_unlabeled_counts[style] += 1
                    style, confidence = None, None
                    selection_reason = 'fallback_after_target_limit'
                if style is not None and style in style_caps and selected_style_counts[style] >= style_caps[style]:
                    skipped_by_cap_counts[style] += 1
                    fallback_forced_unlabeled_counts[style] += 1
                    style, confidence = None, None
                    selection_reason = 'fallback_after_cap_limit'
                elif style is None and candidate_reason not in {'threshold_pass', 'artifact_selected'}:
                    selection_reason = f'speaker_breadth_fallback:{candidate_reason}'
                chosen.append((row_idx, style, confidence, selection_reason))
                used.add(row_idx)
                if style is not None:
                    selected_style_counts[style] += 1
                    selected_labeled_rows += 1
                    selection_reason = 'pseudo_selected'
                selected_reason_counts[selection_reason] += 1
                if len(chosen) >= target_count:
                    break

        for row_idx, style, confidence, selection_reason in chosen:
            selected.append({
                'dataset': 'CommonVoice',
                'row_idx': row_idx,
                'speaker_id': str(cv_data['speaker_ids'][row_idx]),
                'clip_path': cv_data['clip_paths'][row_idx] if 'clip_paths' in cv_data else None,
                'style': style,
                'label_source': 'pseudo' if style is not None else 'none',
                'label_confidence': float(confidence) if confidence is not None else 0.0,
                'selection_reason': selection_reason,
            })
            selected_row_indices.add(row_idx)

    if args.commonvoice_preserve_selected_pseudo:
        for row_idx, speaker_id in enumerate(cv_data['speaker_ids']):
            speaker_id = str(speaker_id)
            if speaker_id not in sampled_speaker_ids or row_idx in selected_row_indices:
                continue
            style, confidence, candidate_reason = accepted_commonvoice_style(
                cv_data, row_idx, threshold_map, args.acceptance_policy
            )
            candidate_reason_counts[candidate_reason] += 1
            if style is None:
                continue
            if (
                args.acceptance_policy == 'balanced_targets'
                and style in style_targets
                and selected_style_counts[style] >= style_targets[style]
            ):
                skipped_by_target_counts[style] += 1
                continue
            if style in style_caps and selected_style_counts[style] >= style_caps[style]:
                skipped_by_cap_counts[style] += 1
                continue

            selected.append({
                'dataset': 'CommonVoice',
                'row_idx': row_idx,
                'speaker_id': speaker_id,
                'clip_path': cv_data['clip_paths'][row_idx] if 'clip_paths' in cv_data else None,
                'style': style,
                'label_source': 'pseudo',
                'label_confidence': float(confidence) if confidence is not None else 0.0,
                'selection_reason': 'pseudo_preserved_after_speaker_cap',
            })
            selected_row_indices.add(row_idx)
            selected_style_counts[style] += 1
            selected_labeled_rows += 1
            selected_reason_counts['pseudo_preserved_after_speaker_cap'] += 1

    selection_report = {
        'acceptance_policy': args.acceptance_policy,
        'style_targets': style_targets,
        'preserve_selected_pseudo': bool(args.commonvoice_preserve_selected_pseudo),
        'skipped_by_cap_counts': dict(skipped_by_cap_counts),
        'skipped_by_target_counts': dict(skipped_by_target_counts),
        'fallback_forced_unlabeled_counts': dict(fallback_forced_unlabeled_counts),
        'candidate_reason_counts': dict(candidate_reason_counts),
        'selected_reason_counts': dict(selected_reason_counts),
        'selected_speaker_count': len(speakers),
        'selected_labeled_row_count': int(selected_labeled_rows),
        'target_shortfall_counts': {
            style: max(target - selected_style_counts.get(style, 0), 0)
            for style, target in style_targets.items()
        },
    }
    return selected, selection_report


def build_rows(args):
    cv_data = torch.load(args.commonvoice, weights_only=False, map_location='cpu')
    cremad_data = torch.load(args.cremad, weights_only=False, map_location='cpu')
    expresso_data = torch.load(args.expresso, weights_only=False, map_location='cpu')
    parquet_dir = resolve_expresso_parquet_dir(args.parquet_dir)
    expresso_df = load_expresso_metadata(expresso_data, parquet_dir)

    style_caps = parse_style_caps(args.commonvoice_style_caps)
    threshold_map = parse_style_thresholds(
        args.pseudo_style_thresholds,
        args.pseudo_style_threshold,
    )
    commonvoice_rows, commonvoice_selection_report = select_commonvoice_rows(
        cv_data,
        args,
        style_caps,
        threshold_map,
    )
    cremad_rows = resolve_cremad_rows(cremad_data)
    expresso_rows = resolve_expresso_rows(expresso_data, expresso_df, args.expresso_only_cap, args.seed)

    rows = commonvoice_rows + cremad_rows + expresso_rows
    payloads = {
        'CommonVoice': cv_data,
        'CREMA-D': cremad_data,
        'Expresso': expresso_data,
    }
    return rows, payloads, parquet_dir, threshold_map, commonvoice_selection_report


def compute_style_row_weights(rows, args):
    style_counts = Counter(row['style'] for row in rows if row['style'] is not None)
    total = sum(style_counts.values())
    present = len(style_counts) or 1
    for row in rows:
        if row['style'] is None:
            row['style_row_weight'] = 0.0
            continue
        base = total / (present * style_counts[row['style']])
        if row['label_source'] == 'pseudo':
            base *= args.pseudo_row_weight
            if args.pseudo_confidence_scale:
                base *= row['label_confidence']
        else:
            base *= args.true_row_weight
        row['style_row_weight'] = float(base)


def normalize_embedding(embedding):
    tensor = torch.as_tensor(embedding)
    if tensor.dim() == 0:
        raise ValueError("Encountered scalar embedding; expected a vector-like tensor")
    return tensor.squeeze()


def build_save_dict(rows, payloads, args, parquet_dir, threshold_map, commonvoice_selection_report):
    compute_style_row_weights(rows, args)

    embeddings = []
    targets = defaultdict(list)
    speaker_ids = []
    clip_paths = []
    source_datasets = []
    style_sources = []
    style_confidences = []
    acceptance_reasons = []
    style_mask = []
    style_row_weights = []
    metadata_age_raw = []
    metadata_gender_raw = []
    metadata_accent_raw = []
    metadata_age_scalars = []
    metadata_age_mask = []
    metadata_gender_scalars = []
    metadata_gender_mask = []

    for row in rows:
        source_payload = payloads[row['dataset']]
        emb = normalize_embedding(source_payload['data'][row['row_idx']])
        embeddings.append(emb)

        if row['style'] is None:
            target = build_empty_target()
            mask = 0.0
        else:
            target = onehot_target(row['style'])
            mask = 1.0

        for idx, style_name in enumerate(UNIFIED_STYLES):
            targets[f'label_{style_name}'].append(torch.tensor(target[idx], dtype=torch.float32))

        speaker_ids.append(row['speaker_id'])
        clip_paths.append(row['clip_path'])
        source_datasets.append(row['dataset'])
        style_sources.append(row['label_source'])
        style_confidences.append(float(row['label_confidence']))
        acceptance_reasons.append(row.get('selection_reason', row['label_source']))
        style_mask.append(mask)
        style_row_weights.append(float(row['style_row_weight']))
        metadata = row_metadata_controls(row, source_payload)
        metadata_age_raw.append(metadata['age_raw'])
        metadata_gender_raw.append(metadata['gender_raw'])
        metadata_accent_raw.append(metadata['accent_raw'])
        metadata_age_scalars.append(metadata['age_scalar'])
        metadata_age_mask.append(metadata['age_mask'])
        metadata_gender_scalars.append(metadata['gender_scalar'])
        metadata_gender_mask.append(metadata['gender_mask'])

    save_dict = {
        'data': torch.stack(embeddings, dim=0),
        'supported_styles': list(UNIFIED_STYLES),
        'speaker_ids': speaker_ids,
        'clip_paths': clip_paths,
        'source_dataset': source_datasets,
        'style_label_source': style_sources,
        'style_label_acceptance_reason': acceptance_reasons,
        'style_label_confidence': torch.tensor(style_confidences, dtype=torch.float32).unsqueeze(1),
        'style_label_mask': torch.tensor(style_mask, dtype=torch.float32).unsqueeze(1),
        'style_label_row_weight': torch.tensor(style_row_weights, dtype=torch.float32).unsqueeze(1),
        'metadata_age_raw': metadata_age_raw,
        'metadata_gender_raw': metadata_gender_raw,
        'metadata_accent_raw': metadata_accent_raw,
        'metadata_age_ordinal_scalar': torch.tensor(
            metadata_age_scalars, dtype=torch.float32
        ).unsqueeze(1),
        'metadata_age_mask': torch.tensor(metadata_age_mask, dtype=torch.float32).unsqueeze(1),
        'metadata_gender_scalar': torch.tensor(
            metadata_gender_scalars, dtype=torch.float32
        ).unsqueeze(1),
        'metadata_gender_mask': torch.tensor(metadata_gender_mask, dtype=torch.float32).unsqueeze(1),
    }
    for key, values in targets.items():
        save_dict[key] = torch.stack(values).unsqueeze(1)

    dataset_counts = Counter(source_datasets)
    dataset_labeled_counts = Counter(
        row['dataset'] for row in rows if row['style'] is not None
    )
    dataset_unique_speakers = {
        dataset: len({row['speaker_id'] for row in rows if row['dataset'] == dataset})
        for dataset in DATASETS
        if dataset_counts.get(dataset)
    }
    raw_pseudo_counts = Counter(
        style
        for style in payloads['CommonVoice'].get('pseudo_style', [])
        if style in UNIFIED_STYLES
    )
    threshold_accepted_counts = Counter()
    threshold_rejected_counts = Counter()
    for style, confidence in zip(
        payloads['CommonVoice'].get('pseudo_style', []),
        payloads['CommonVoice'].get('pseudo_style_confidence', []),
    ):
        if style not in UNIFIED_STYLES or confidence is None:
            continue
        confidence = float(confidence)
        if confidence >= threshold_map.get(style, 0.0):
            threshold_accepted_counts[style] += 1
        else:
            threshold_rejected_counts[style] += 1
    selected_pseudo_counts = Counter(
        row['style'] for row in rows if row['dataset'] == 'CommonVoice' and row['style'] is not None
    )
    metadata_control_report = build_metadata_control_report(
        source_datasets,
        metadata_age_raw,
        metadata_gender_raw,
        metadata_age_mask,
        metadata_gender_mask,
    )
    save_dict['metadata_control_report'] = metadata_control_report
    save_dict['mixture_report'] = {
        'seed': args.seed,
        'parquet_dir': str(parquet_dir),
        'commonvoice_source_artifact': args.commonvoice,
        'cremad_source_artifact': args.cremad,
        'expresso_source_artifact': args.expresso,
        'pseudo_style_threshold': args.pseudo_style_threshold,
        'pseudo_style_thresholds': dict(threshold_map),
        'commonvoice_style_caps': parse_style_caps(args.commonvoice_style_caps),
        'commonvoice_style_targets': parse_style_targets(args.commonvoice_style_targets),
        'commonvoice_acceptance_policy': args.acceptance_policy,
        'commonvoice_selection': {
            'max_speakers': args.commonvoice_max_speakers,
            'min_clips_per_speaker': args.commonvoice_min_clips_per_speaker,
            'max_clips_per_speaker': args.commonvoice_max_clips_per_speaker,
            'prefer_pseudo': args.commonvoice_prefer_pseudo,
            'preserve_selected_pseudo': bool(args.commonvoice_preserve_selected_pseudo),
        },
        'row_weight_config': {
            'pseudo_confidence_scale': bool(args.pseudo_confidence_scale),
            'pseudo_row_weight': float(args.pseudo_row_weight),
            'true_row_weight': float(args.true_row_weight),
        },
        'dataset_counts': dict(dataset_counts),
        'dataset_labeled_counts': dict(dataset_labeled_counts),
        'dataset_unique_speakers': dataset_unique_speakers,
        'commonvoice_raw_pseudo_style_counts': dict(raw_pseudo_counts),
        'commonvoice_threshold_accepted_style_counts': dict(threshold_accepted_counts),
        'commonvoice_threshold_rejected_style_counts': dict(threshold_rejected_counts),
        'commonvoice_selected_pseudo_style_counts': dict(selected_pseudo_counts),
        'metadata_control_report': metadata_control_report,
        'commonvoice_skipped_by_cap_counts': dict(
            commonvoice_selection_report.get('skipped_by_cap_counts', {})
        ),
        'commonvoice_skipped_by_target_counts': dict(
            commonvoice_selection_report.get('skipped_by_target_counts', {})
        ),
        'commonvoice_fallback_forced_unlabeled_counts': dict(
            commonvoice_selection_report.get('fallback_forced_unlabeled_counts', {})
        ),
        'commonvoice_candidate_reason_counts': dict(
            commonvoice_selection_report.get('candidate_reason_counts', {})
        ),
        'commonvoice_selected_reason_counts': dict(
            commonvoice_selection_report.get('selected_reason_counts', {})
        ),
        'commonvoice_selected_speaker_count': int(
            commonvoice_selection_report.get('selected_speaker_count', 0)
        ),
        'commonvoice_selected_labeled_row_count': int(
            commonvoice_selection_report.get('selected_labeled_row_count', 0)
        ),
        'commonvoice_target_shortfall_counts': dict(
            commonvoice_selection_report.get('target_shortfall_counts', {})
        ),
        'style_counts_all_labeled_rows': dict(Counter(row['style'] for row in rows if row['style'] is not None)),
        'label_source_counts': dict(Counter(style_sources)),
        'style_label_acceptance_reason_counts': dict(Counter(acceptance_reasons)),
    }
    if 'pseudo_style_report' in payloads['CommonVoice']:
        save_dict['mixture_report']['commonvoice_pseudo_style_report'] = payloads['CommonVoice'][
            'pseudo_style_report'
        ]
    if 'pseudo_style_teacher' in payloads['CommonVoice']:
        save_dict['mixture_report']['commonvoice_pseudo_style_teacher'] = payloads['CommonVoice'][
            'pseudo_style_teacher'
        ]
    if 'pseudo_style_filter_report' in payloads['CommonVoice']:
        save_dict['mixture_report']['commonvoice_pseudo_style_filter_report'] = payloads['CommonVoice'][
            'pseudo_style_filter_report'
        ]
    if 'metadata_report' in payloads['CommonVoice']:
        save_dict['commonvoice_metadata_report'] = payloads['CommonVoice']['metadata_report']
    if 'pseudo_style_report' in payloads['CommonVoice']:
        save_dict['commonvoice_pseudo_style_report'] = payloads['CommonVoice']['pseudo_style_report']
    if 'pseudo_style_teacher' in payloads['CommonVoice']:
        save_dict['commonvoice_pseudo_style_teacher'] = payloads['CommonVoice']['pseudo_style_teacher']
    if 'pseudo_style_filter_report' in payloads['CommonVoice']:
        save_dict['commonvoice_pseudo_style_filter_report'] = payloads['CommonVoice']['pseudo_style_filter_report']
    return save_dict


def print_report(save_dict):
    report = save_dict['mixture_report']
    print('Mixed-data training artifact report')
    print(f"  total rows: {save_dict['data'].shape[0]}")
    print(f"  acceptance policy: {report.get('commonvoice_acceptance_policy')}")
    print('  dataset counts:')
    for dataset, count in report['dataset_counts'].items():
        labeled = report['dataset_labeled_counts'].get(dataset, 0)
        speakers = report['dataset_unique_speakers'].get(dataset, 0)
        print(f"    {dataset:11s} rows={count:4d} labeled={labeled:4d} speakers={speakers:4d}")
    print('  label sources:')
    for name, count in report['label_source_counts'].items():
        print(f"    {name:11s} {count:4d}")
    print('  labeled style counts:')
    for style in UNIFIED_STYLES:
        count = report['style_counts_all_labeled_rows'].get(style, 0)
        if count:
            print(f"    {style:11s} {count:4d}")
    pseudo_counts = report.get('commonvoice_selected_pseudo_style_counts', {})
    if pseudo_counts:
        print('  selected CommonVoice pseudo-style counts:')
        for style in UNIFIED_STYLES:
            count = pseudo_counts.get(style, 0)
            if count:
                print(f"    {style:11s} {count:4d}")
    rejected_counts = report.get('commonvoice_threshold_rejected_style_counts', {})
    if rejected_counts:
        print('  threshold-rejected CommonVoice pseudo-style counts:')
        for style in UNIFIED_STYLES:
            count = rejected_counts.get(style, 0)
            if count:
                print(f"    {style:11s} {count:4d}")
    skipped_by_cap = report.get('commonvoice_skipped_by_cap_counts', {})
    if skipped_by_cap:
        print('  cap-skipped CommonVoice pseudo-style counts:')
        for style in UNIFIED_STYLES:
            count = skipped_by_cap.get(style, 0)
            if count:
                print(f"    {style:11s} {count:4d}")
    skipped_by_target = report.get('commonvoice_skipped_by_target_counts', {})
    if skipped_by_target:
        print('  target-skipped CommonVoice pseudo-style counts:')
        for style in UNIFIED_STYLES:
            count = skipped_by_target.get(style, 0)
            if count:
                print(f"    {style:11s} {count:4d}")
    shortfalls = report.get('commonvoice_target_shortfall_counts', {})
    if shortfalls:
        print('  CommonVoice target shortfalls:')
        for style in UNIFIED_STYLES:
            count = shortfalls.get(style, 0)
            if count:
                print(f"    {style:11s} {count:4d}")
    metadata_report = report.get('metadata_control_report', {})
    if metadata_report:
        print('  metadata control rows:')
        print(
            "    age          "
            f"{metadata_report.get('age_control_rows', 0):4d}/"
            f"{metadata_report.get('rows', 0)}"
        )
        print(
            "    gender       "
            f"{metadata_report.get('gender_control_rows', 0):4d}/"
            f"{metadata_report.get('rows', 0)}"
        )
        print(
            "    age+gender   "
            f"{metadata_report.get('age_gender_control_rows', 0):4d}/"
            f"{metadata_report.get('rows', 0)}"
        )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    rows, payloads, parquet_dir, threshold_map, commonvoice_selection_report = build_rows(args)
    save_dict = build_save_dict(
        rows,
        payloads,
        args,
        parquet_dir,
        threshold_map,
        commonvoice_selection_report,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, output_path)
    print_report(save_dict)
    print(f"Saved mixed-data artifact to {output_path}")


if __name__ == '__main__':
    main()
