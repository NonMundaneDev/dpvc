"""
Train a controllable VAE on a sampled mixed-data artifact.

This script is designed for the first CommonVoice + CREMA-D + Expresso mixed
training experiments. It expects an artifact built by:

    python scripts/build_mixed_training_set.py --output embeddings/openvoice_mixed_base.pt

Usage:
    python examples/openvoice_train_vae_mixed.py \
        --embeddings embeddings/openvoice_mixed_base.pt \
        --output embeddings/openvoice_vae_mixed_static_balanced.pt \
        --schedule static_balanced

    python examples/openvoice_train_vae_mixed.py \
        --embeddings embeddings/openvoice_mixed_base.pt \
        --output embeddings/openvoice_vae_mixed_cv_warmup.pt \
        --schedule cv_warmup

    python examples/openvoice_train_vae_mixed.py \
        --embeddings embeddings/openvoice_mixed_base.pt \
        --output embeddings/openvoice_vae_mixed_labeled_finish.pt \
        --schedule labeled_finish
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

import dpvc


DEFAULT_SCHEDULES = ["static_balanced", "cv_warmup", "labeled_warmup", "labeled_finish"]
DATASET_NAMES = ["CommonVoice", "CREMA-D", "Expresso"]


def resolve_device():
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_module_requires_grad(module, trainable):
    for param in module.parameters():
        param.requires_grad = trainable


def format_param_count(model):
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total


def parse_dataset_masses(raw):
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    masses = {}
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(f"Expected dataset mass in name=value form, got: {item}")
        name, value = item.split('=', 1)
        dataset = name.strip()
        if dataset not in DATASET_NAMES:
            raise ValueError(f"Unknown dataset in mass config: {dataset}")
        masses[dataset] = float(value.strip())
    return masses


def parse_style_weights(raw, supported_styles):
    weights = {style: 1.0 for style in supported_styles}
    raw = (raw or "").strip()
    if not raw:
        return weights
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(f"Expected style weight in name=value form, got: {item}")
        name, value = item.split('=', 1)
        style = name.strip()
        if style not in weights:
            raise ValueError(f"Unknown style in teacher style-weight config: {style}")
        weights[style] = float(value.strip())
    return weights


def parse_style_strengths(raw, supported_styles, default_strength):
    strengths = {style: float(default_strength) for style in supported_styles}
    raw = (raw or "").strip()
    if not raw:
        return strengths
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(f"Expected style strength in name=value form, got: {item}")
        name, value = item.split('=', 1)
        style = name.strip()
        if style not in strengths:
            raise ValueError(f"Unknown style in decoder prototype strength config: {style}")
        strength = float(value.strip())
        if strength < 0:
            raise ValueError(f"Decoder prototype strength must be non-negative: {item}")
        strengths[style] = strength
    return strengths


def parse_style_names(raw, supported_styles, default_styles=None):
    raw = (raw or "").strip()
    if not raw:
        selected = list(default_styles or supported_styles)
    else:
        selected = [item.strip() for item in raw.split(',') if item.strip()]
    unknown = set(selected) - set(supported_styles)
    if unknown:
        raise ValueError(f"Unknown style(s): {sorted(unknown)}")
    return selected


def parse_label_sources(raw):
    raw = (raw or "true").strip()
    if raw.lower() in {"all", "all_labeled", "true,pseudo", "pseudo,true"}:
        return {"true", "pseudo"}
    selected = {item.strip() for item in raw.split(',') if item.strip()}
    valid = {"true", "pseudo"}
    unknown = selected - valid
    if unknown:
        raise ValueError(
            f"Unknown decoder prototype source(s): {sorted(unknown)}; "
            "expected true, pseudo, or all_labeled"
        )
    return selected


def parse_dim_list(raw):
    dims = []
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        if '-' in item:
            start, end = item.split('-', 1)
            dims.extend(range(int(start), int(end) + 1))
        else:
            dims.append(int(item))
    return sorted(set(dims))


def build_dataset_mask(source_datasets, raw, device):
    raw = (raw or "").strip()
    if not raw or raw.lower() == "all":
        return torch.ones(len(source_datasets), dtype=torch.float32, device=device)
    selected = {item.strip() for item in raw.split(',') if item.strip()}
    unknown = selected - set(DATASET_NAMES)
    if unknown:
        raise ValueError(f"Unknown dataset(s) in teacher dataset filter: {sorted(unknown)}")
    return torch.tensor(
        [1.0 if str(dataset) in selected else 0.0 for dataset in source_datasets],
        dtype=torch.float32,
        device=device,
    )


def build_decoder_prototypes(
    embeddings,
    style_targets,
    style_label_mask,
    style_label_sources,
    supported_styles,
    source_filter,
    min_count,
):
    label_sources = list(style_label_sources or [])
    if len(label_sources) != embeddings.shape[0]:
        raise ValueError(
            "Mixed artifact is missing style_label_source entries needed for "
            "decoder prototype construction"
        )

    source_mask = torch.tensor(
        [str(source) in source_filter for source in label_sources],
        dtype=torch.bool,
        device=embeddings.device,
    )
    label_mask = style_label_mask.view(-1) > 0
    prototypes = []
    counts = {}
    for idx, style in enumerate(supported_styles):
        style_mask = style_targets[:, idx].view(-1) > 0
        active = label_mask & source_mask & style_mask
        count = int(active.sum().item())
        if count < min_count:
            raise ValueError(
                f"Decoder prototype source {sorted(source_filter)} has only "
                f"{count} rows for style {style!r}; require at least {min_count}"
            )
        prototypes.append(embeddings[active].mean(dim=0))
        counts[style] = count
    return torch.stack(prototypes, dim=0), counts


def build_style_teacher_row_weights(
    style_targets,
    style_label_mask,
    style_label_confidence,
    supported_styles,
    raw_style_weights,
    confidence_power,
    require_label,
):
    style_weights = parse_style_weights(raw_style_weights, supported_styles)
    row_weights = torch.ones(
        style_targets.shape[0],
        dtype=torch.float32,
        device=style_targets.device,
    )
    label_mask = style_label_mask.view(-1) > 0

    if require_label:
        row_weights = row_weights * label_mask.float()

    if raw_style_weights:
        style_indices = torch.argmax(style_targets, dim=1)
        for idx, style in enumerate(supported_styles):
            style_mask = label_mask & (style_indices == idx)
            row_weights[style_mask] *= float(style_weights[style])

    if confidence_power and confidence_power > 0:
        if style_label_confidence is None:
            raise ValueError(
                "--style-teacher-confidence-power requires style_label_confidence "
                "in the mixed artifact"
            )
        confidence = style_label_confidence.view(-1).clamp(min=0.0)
        row_weights = row_weights * torch.pow(confidence, confidence_power)

    return row_weights.view(-1, 1), style_weights


def build_selected_style_row_weights(
    style_targets,
    style_label_mask,
    style_label_confidence,
    supported_styles,
    selected_styles,
    raw_style_weights,
    confidence_power,
):
    row_weights, style_weights = build_style_teacher_row_weights(
        style_targets,
        style_label_mask,
        style_label_confidence,
        supported_styles,
        raw_style_weights,
        confidence_power,
        require_label=True,
    )
    style_indices = torch.argmax(style_targets, dim=1)
    selected_indices = torch.tensor(
        [supported_styles.index(style) for style in selected_styles],
        dtype=torch.long,
        device=style_targets.device,
    )
    selected_mask = torch.isin(style_indices, selected_indices)
    row_weights = row_weights * selected_mask.float().view(-1, 1)
    return row_weights, style_weights


def parse_metadata_control_targets(raw):
    targets = [item.strip() for item in (raw or "").split(",") if item.strip()]
    if not targets:
        targets = ["gender", "age"]
    invalid = [target for target in targets if target not in {"gender", "age"}]
    if invalid:
        raise ValueError(
            f"Unsupported metadata-control targets: {invalid}. "
            "Supported targets: ['gender', 'age']"
        )
    duplicates = [target for target, count in Counter(targets).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate metadata-control targets: {duplicates}")
    return targets


def build_metadata_controls(data, args, supported_styles, device):
    if args.metadata_control_weight <= 0:
        return None, None, [], {}

    available_specs = {
        "gender": (
            "gender",
            "metadata_gender_scalar",
            "metadata_gender_mask",
            args.metadata_gender_dim,
        ),
        "age": (
            "age",
            "metadata_age_ordinal_scalar",
            "metadata_age_mask",
            args.metadata_age_dim,
        ),
    }
    selected_targets = parse_metadata_control_targets(args.metadata_control_targets)
    target_specs = [available_specs[target] for target in selected_targets]
    dims = [spec[3] for spec in target_specs]
    if len(set(dims)) != len(dims):
        raise ValueError(f"Metadata control dims must be unique, got {dims}")
    if any(dim < 0 or dim >= args.latent_dims for dim in dims):
        raise ValueError(
            f"Metadata control dims must be within [0, {args.latent_dims - 1}], "
            f"got {dims}"
        )
    style_width = len(supported_styles)
    overlapping = [dim for dim in dims if dim < style_width]
    if overlapping:
        raise ValueError(
            "Metadata control dims must not overlap style dims "
            f"[0, {style_width - 1}], got {overlapping}"
        )

    targets = []
    masks = []
    counts = {}
    for name, target_key, mask_key, dim in target_specs:
        if target_key not in data or mask_key not in data:
            raise ValueError(
                f"--metadata-control-weight requires {target_key} and {mask_key} "
                "in the mixed artifact. Rebuild it with scripts/build_mixed_training_set.py."
            )
        target = data[target_key].to(device).float().view(-1, 1)
        mask = data[mask_key].to(device).float().view(-1, 1)
        targets.append(target)
        masks.append(mask)
        counts[name] = int((mask.view(-1) > 0).sum().item())
        if counts[name] == 0:
            print(f"WARNING: metadata control target {name!r} has zero labeled rows")

    metadata_targets = torch.cat(targets, dim=1)
    metadata_mask = torch.cat(masks, dim=1)
    total_labeled = int((metadata_mask > 0).sum().item())
    if total_labeled == 0:
        raise ValueError("No labeled metadata-control targets are available")

    report = {
        "enabled": True,
        "weight": float(args.metadata_control_weight),
        "targets": selected_targets,
        "dims": {name: int(dim) for name, _, _, dim in target_specs},
        "labeled_rows": counts,
        "artifact_report": data.get("metadata_control_report", {}),
    }
    return metadata_targets, metadata_mask, dims, report


def load_generated_audio_objective_plan(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def apply_generated_audio_objective_plan(args, plan):
    overrides = dict(plan.get("trainer_overrides") or {})
    fields = [
        "style_teacher_target_mode",
        "style_teacher_require_label",
        "style_teacher_style_weights",
        "decoder_prototype_style_weights",
        "decoder_prototype_style_strengths",
        "anti_neutral_styles",
        "anti_neutral_style_weights",
        "anti_neutral_style_strengths",
    ]
    applied = {}
    for field in fields:
        if field in overrides:
            setattr(args, field, overrides[field])
            applied[field] = overrides[field]

    return {
        "enabled": True,
        "objective_name": plan.get("objective_name", ""),
        "reference_condition": plan.get("reference_condition", ""),
        "selected_styles": list(plan.get("selected_styles") or []),
        "blocked_styles": list(plan.get("blocked_styles") or []),
        "applied_overrides": applied,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--embeddings",
        default="embeddings/openvoice_mixed_base.pt",
        help="Path to mixed-data embeddings artifact",
    )
    ap.add_argument(
        "--output",
        default="embeddings/openvoice_vae_mixed_static_balanced.pt",
        help="Output VAE checkpoint path",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=3000,
        help="Training epochs (default: 3000)",
    )
    ap.add_argument(
        "--latent-dims",
        type=int,
        default=15,
        help="Latent dimensions (default: 15)",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-6,
        help="Learning rate (default: 1e-6)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed (default: 42)",
    )
    ap.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional checkpoint to load before mixed-data training",
    )
    ap.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the full encoder during training",
    )
    ap.add_argument(
        "--freeze-decoder",
        action="store_true",
        help="Freeze the full decoder during training",
    )
    ap.add_argument(
        "--recon-weight",
        type=float,
        default=1.0,
        help="Reconstruction-loss weight (default: 1.0)",
    )
    ap.add_argument(
        "--kl-weight",
        type=float,
        default=1.0,
        help="KL-loss weight (default: 1.0)",
    )
    ap.add_argument(
        "--label-weight",
        type=float,
        default=1.0,
        help="Style-label loss weight (default: 1.0)",
    )
    ap.add_argument(
        "--metadata-control-weight",
        type=float,
        default=0.0,
        help=(
            "Masked direct latent-control loss for CommonVoice age/gender "
            "metadata (default: 0.0, disabled)"
        ),
    )
    ap.add_argument(
        "--metadata-control-targets",
        default="gender,age",
        help=(
            "Comma-separated metadata targets for direct latent-control loss "
            "(supported: gender, age; default: gender,age)"
        ),
    )
    ap.add_argument(
        "--metadata-gender-dim",
        type=int,
        default=9,
        help="Latent dim for gender scalar supervision (default: 9)",
    )
    ap.add_argument(
        "--metadata-age-dim",
        type=int,
        default=10,
        help="Latent dim for age ordinal supervision (default: 10)",
    )
    ap.add_argument(
        "--metadata-control-report",
        default=None,
        help="Optional JSON path for the metadata-control training config/report",
    )
    ap.add_argument(
        "--generated-audio-objective-plan",
        default=None,
        help=(
            "Optional JSON plan produced by "
            "scripts/plan_generated_audio_calibrated_objective.py. The plan "
            "sets hard-style teacher/prototype/anti-neutral objective flags "
            "from generated-audio evidence."
        ),
    )
    ap.add_argument(
        "--generated-audio-objective-report",
        default=None,
        help="Optional JSON path recording applied generated-audio objective overrides",
    )
    ap.add_argument(
        "--style-teacher-checkpoint",
        default=None,
        help=(
            "Optional frozen VAE checkpoint for continuous style-space "
            "distillation during mixed-data training"
        ),
    )
    ap.add_argument(
        "--style-teacher-weight",
        type=float,
        default=0.0,
        help="Continuous style-teacher loss weight (default: 0.0)",
    )
    ap.add_argument(
        "--style-teacher-weight-final",
        type=float,
        default=None,
        help=(
            "Optional final teacher weight for schedule-epoch interpolation "
            "(default: disabled)"
        ),
    )
    ap.add_argument(
        "--style-teacher-dims",
        default="0-8",
        help="Comma-separated or ranged style dims for teacher loss (default: 0-8)",
    )
    ap.add_argument(
        "--style-teacher-datasets",
        default="CommonVoice",
        help=(
            "Datasets receiving the teacher-style loss, comma-separated or 'all' "
            "(default: CommonVoice)"
        ),
    )
    ap.add_argument(
        "--style-teacher-target-mode",
        default="all_dims",
        choices=["all_dims", "target_dim"],
        help=(
            "Use teacher loss on all selected dims or only each row's accepted "
            "style target dim (default: all_dims)"
        ),
    )
    ap.add_argument(
        "--style-teacher-require-label",
        action="store_true",
        help="Apply teacher-style loss only to rows with an accepted style label",
    )
    ap.add_argument(
        "--style-teacher-style-weights",
        default="",
        help=(
            "Optional per-style row weights for teacher loss, e.g. "
            "anger=4,fear=4,neutral=0.25"
        ),
    )
    ap.add_argument(
        "--style-teacher-confidence-power",
        type=float,
        default=0.0,
        help=(
            "Optional exponent for multiplying teacher row weights by label "
            "confidence^power (default: 0.0, disabled)"
        ),
    )
    ap.add_argument(
        "--decoder-prototype-weight",
        type=float,
        default=0.0,
        help=(
            "Decoder-space style-prototype loss weight. This decodes a "
            "style-controlled latent and matches it to real labeled style "
            "embedding prototypes (default: 0.0, disabled)"
        ),
    )
    ap.add_argument(
        "--decoder-prototype-weight-final",
        type=float,
        default=None,
        help="Optional final decoder-prototype loss weight for schedule interpolation",
    )
    ap.add_argument(
        "--decoder-prototype-datasets",
        default="CommonVoice",
        help=(
            "Datasets receiving decoder-prototype loss, comma-separated or 'all' "
            "(default: CommonVoice)"
        ),
    )
    ap.add_argument(
        "--decoder-prototype-source",
        default="true",
        help=(
            "Rows used to build style prototypes: true, pseudo, or all_labeled "
            "(default: true)"
        ),
    )
    ap.add_argument(
        "--decoder-prototype-min-count",
        type=int,
        default=5,
        help="Minimum rows required per style when building prototypes (default: 5)",
    )
    ap.add_argument(
        "--decoder-prototype-strength",
        type=float,
        default=5.0,
        help="Default style strength used inside the decoder-prototype loss (default: 5.0)",
    )
    ap.add_argument(
        "--decoder-prototype-style-strengths",
        default="",
        help=(
            "Optional per-style strengths for decoder-prototype loss, e.g. "
            "sad=3.5,enunciated=2.5"
        ),
    )
    ap.add_argument(
        "--decoder-prototype-style-weights",
        default="",
        help=(
            "Optional per-style row weights for decoder-prototype loss. Use "
            "this for generated-audio-calibrated hard-style repair so "
            "non-target styles do not receive prototype pressure."
        ),
    )
    ap.add_argument(
        "--decoder-prototype-control-mode",
        default="target_only",
        choices=["target_only", "all_style_dims"],
        help=(
            "How to apply style controls before decoding for prototype loss. "
            "target_only mirrors inference by setting only the target style dim "
            "(default: target_only)"
        ),
    )
    ap.add_argument(
        "--anti-neutral-weight",
        type=float,
        default=0.0,
        help=(
            "Decoded-teacher anti-neutral margin loss weight. This decodes a "
            "style-controlled latent, re-encodes it with the frozen style "
            "teacher, and penalizes neutral beating the target style "
            "(default: 0.0, disabled)"
        ),
    )
    ap.add_argument(
        "--anti-neutral-weight-final",
        type=float,
        default=None,
        help="Optional final anti-neutral loss weight for schedule interpolation",
    )
    ap.add_argument(
        "--anti-neutral-mode",
        default="teacher_margin",
        choices=["teacher_margin", "prototype_margin"],
        help=(
            "Anti-neutral loss mode. teacher_margin re-encodes decoded embeddings "
            "with the frozen style teacher; prototype_margin makes decoded "
            "embeddings closer to the target style prototype than the neutral "
            "prototype (default: teacher_margin)"
        ),
    )
    ap.add_argument(
        "--anti-neutral-datasets",
        default="CommonVoice",
        help=(
            "Datasets receiving anti-neutral loss, comma-separated or 'all' "
            "(default: CommonVoice)"
        ),
    )
    ap.add_argument(
        "--anti-neutral-styles",
        default="anger,disgust",
        help=(
            "Comma-separated target styles for anti-neutral loss "
            "(default: anger,disgust)"
        ),
    )
    ap.add_argument(
        "--anti-neutral-style-weights",
        default="",
        help=(
            "Optional per-style row weights for anti-neutral loss, e.g. "
            "anger=3,disgust=3"
        ),
    )
    ap.add_argument(
        "--anti-neutral-margin",
        type=float,
        default=0.5,
        help=(
            "Margin requiring target style teacher dim to beat neutral dim "
            "after decoding (default: 0.5)"
        ),
    )
    ap.add_argument(
        "--anti-neutral-strength",
        type=float,
        default=5.0,
        help="Default style strength used inside anti-neutral loss (default: 5.0)",
    )
    ap.add_argument(
        "--anti-neutral-style-strengths",
        default="",
        help=(
            "Optional per-style strengths for anti-neutral loss, e.g. "
            "anger=5,disgust=5"
        ),
    )
    ap.add_argument(
        "--anti-neutral-control-mode",
        default="target_only",
        choices=["target_only", "all_style_dims"],
        help=(
            "How to apply style controls before decoding for anti-neutral loss "
            "(default: target_only)"
        ),
    )
    ap.add_argument(
        "--schedule",
        default="static_balanced",
        choices=DEFAULT_SCHEDULES,
        help="Dataset-mixture schedule (default: static_balanced)",
    )
    ap.add_argument(
        "--schedule-epochs",
        type=int,
        default=0,
        help="Epochs over which a non-static schedule interpolates (default: full run)",
    )
    ap.add_argument(
        "--static-masses",
        default="",
        help="Optional dataset masses for static schedule, e.g. CommonVoice=0.2,CREMA-D=0.4,Expresso=0.4",
    )
    ap.add_argument(
        "--schedule-start-masses",
        default="",
        help="Optional dataset masses at the start of a non-static schedule",
    )
    ap.add_argument(
        "--schedule-end-masses",
        default="",
        help="Optional dataset masses at the end of a non-static schedule",
    )
    args = ap.parse_args()
    generated_audio_objective_report = {"enabled": False}
    if args.generated_audio_objective_plan:
        generated_audio_plan = load_generated_audio_objective_plan(
            args.generated_audio_objective_plan
        )
        generated_audio_objective_report = apply_generated_audio_objective_plan(
            args,
            generated_audio_plan,
        )
        print(f"Generated-audio objective plan: {args.generated_audio_objective_plan}")
        print(
            "Generated-audio selected styles: "
            f"{generated_audio_objective_report['selected_styles']}"
        )
        print(
            "Generated-audio blocked styles: "
            f"{generated_audio_objective_report['blocked_styles']}"
        )
    anti_neutral_requested = (
        args.anti_neutral_weight > 0
        or (args.anti_neutral_weight_final or 0.0) > 0
    )

    if args.freeze_encoder and args.freeze_decoder:
        ap.error("Refusing to freeze both encoder and decoder; nothing would remain trainable")
    if (
        (
            args.style_teacher_weight > 0
            or (args.style_teacher_weight_final or 0.0) > 0
            or (anti_neutral_requested and args.anti_neutral_mode == "teacher_margin")
        )
        and not args.style_teacher_checkpoint
    ):
        ap.error(
            "--style-teacher-weight, --style-teacher-weight-final, "
            "or teacher-margin anti-neutral loss requires --style-teacher-checkpoint"
        )
    if args.decoder_prototype_weight < 0 or (args.decoder_prototype_weight_final or 0.0) < 0:
        ap.error("Decoder-prototype weights must be non-negative")
    if args.anti_neutral_weight < 0 or (args.anti_neutral_weight_final or 0.0) < 0:
        ap.error("Anti-neutral weights must be non-negative")
    if args.decoder_prototype_strength < 0:
        ap.error("--decoder-prototype-strength must be non-negative")
    if args.anti_neutral_strength < 0:
        ap.error("--anti-neutral-strength must be non-negative")
    if args.anti_neutral_margin < 0:
        ap.error("--anti-neutral-margin must be non-negative")
    if args.decoder_prototype_min_count < 1:
        ap.error("--decoder-prototype-min-count must be >= 1")
    if args.metadata_control_weight < 0:
        ap.error("--metadata-control-weight must be non-negative")

    dpvc.utils.set_seed(args.seed)
    device = resolve_device()
    static_masses = parse_dataset_masses(args.static_masses)
    schedule_start_masses = parse_dataset_masses(args.schedule_start_masses)
    schedule_end_masses = parse_dataset_masses(args.schedule_end_masses)
    style_teacher_dims = parse_dim_list(args.style_teacher_dims)
    if any(dim < 0 or dim >= args.latent_dims for dim in style_teacher_dims):
        raise ValueError(
            f"Style-teacher dims must be within [0, {args.latent_dims - 1}], "
            f"got {style_teacher_dims}"
        )

    data = torch.load(args.embeddings, weights_only=False)
    embeddings = data['data'].to(device).squeeze()
    print(f"Embeddings shape: {embeddings.shape}")
    mixture_report = data.get('mixture_report', {})
    teacher_info = data.get('commonvoice_pseudo_style_teacher') or {}
    filter_info = data.get('commonvoice_pseudo_style_filter_report') or {}

    supported_styles = data.get('supported_styles')
    if not supported_styles:
        supported_styles = [
            key.replace('label_', '')
            for key in sorted(data.keys())
            if key.startswith('label_')
        ]
    label_keys = [f'label_{style}' for style in supported_styles]
    missing_keys = [key for key in label_keys if key not in data]
    if missing_keys:
        raise ValueError(f"Missing label keys in mixed artifact: {missing_keys}")

    style_targets = torch.cat([data[key] for key in label_keys], dim=1).to(device)
    style_label_mask = data.get('style_label_mask')
    if style_label_mask is None:
        raise ValueError("Mixed artifact is missing style_label_mask")
    style_label_mask = style_label_mask.to(device)

    style_label_row_weights = data.get('style_label_row_weight')
    if style_label_row_weights is not None:
        style_label_row_weights = style_label_row_weights.to(device)
    style_label_confidence = data.get('style_label_confidence')
    if style_label_confidence is not None:
        style_label_confidence = style_label_confidence.to(device)

    source_datasets = data.get('source_dataset')
    if source_datasets is None:
        raise ValueError("Mixed artifact is missing source_dataset")
    metadata_targets = None
    metadata_label_mask = None
    metadata_control_dims = []
    metadata_control_report = {"enabled": False}
    (
        metadata_targets,
        metadata_label_mask,
        metadata_control_dims,
        metadata_control_report,
    ) = build_metadata_controls(data, args, supported_styles, device)
    teacher_requested = (
        args.style_teacher_checkpoint
        and (
            args.style_teacher_weight > 0
            or (args.style_teacher_weight_final or 0.0) > 0
            or (anti_neutral_requested and args.anti_neutral_mode == "teacher_margin")
        )
    )
    style_teacher_mask = None
    if teacher_requested:
        style_teacher_mask = build_dataset_mask(
            source_datasets,
            args.style_teacher_datasets,
            device,
        )
    style_teacher_row_weights = None
    style_teacher_style_weights = None
    if teacher_requested:
        if args.style_teacher_target_mode == "target_dim":
            max_dim = max(style_teacher_dims) if style_teacher_dims else -1
            if max_dim >= style_targets.shape[1]:
                raise ValueError(
                    "--style-teacher-target-mode target_dim requires teacher dims "
                    f"inside the style-label range [0, {style_targets.shape[1] - 1}], "
                    f"got {style_teacher_dims}"
                )
        style_teacher_row_weights, style_teacher_style_weights = build_style_teacher_row_weights(
            style_targets,
            style_label_mask,
            style_label_confidence,
            supported_styles,
            args.style_teacher_style_weights,
            args.style_teacher_confidence_power,
            args.style_teacher_require_label,
        )

    anti_neutral_mask = None
    anti_neutral_row_weights = None
    anti_neutral_strengths = None
    anti_neutral_strength_map = None
    anti_neutral_styles = []
    anti_neutral_neutral_index = None
    if anti_neutral_requested:
        if "neutral" not in supported_styles:
            raise ValueError("Anti-neutral loss requires a 'neutral' supported style")
        anti_neutral_neutral_index = supported_styles.index("neutral")
        anti_neutral_styles = parse_style_names(
            args.anti_neutral_styles,
            supported_styles,
            default_styles=["anger", "disgust"],
        )
        if "neutral" in anti_neutral_styles:
            raise ValueError("--anti-neutral-styles cannot include neutral")
        anti_neutral_mask = build_dataset_mask(
            source_datasets,
            args.anti_neutral_datasets,
            device,
        )
        anti_neutral_row_weights, anti_neutral_style_weights = build_selected_style_row_weights(
            style_targets,
            style_label_mask,
            style_label_confidence,
            supported_styles,
            anti_neutral_styles,
            args.anti_neutral_style_weights,
            confidence_power=0.0,
        )
        anti_neutral_strength_map = parse_style_strengths(
            args.anti_neutral_style_strengths,
            supported_styles,
            args.anti_neutral_strength,
        )
        anti_neutral_strengths = torch.tensor(
            [anti_neutral_strength_map[style] for style in supported_styles],
            dtype=torch.float32,
            device=device,
        )

    decoder_prototype_requested = (
        args.decoder_prototype_weight > 0
        or (args.decoder_prototype_weight_final or 0.0) > 0
        or (anti_neutral_requested and args.anti_neutral_mode == "prototype_margin")
    )
    decoder_prototype_targets = None
    decoder_prototype_mask = None
    decoder_prototype_row_weights = None
    decoder_prototype_strengths = None
    decoder_prototype_counts = None
    decoder_prototype_strength_map = None
    decoder_prototype_style_weights = None
    if decoder_prototype_requested:
        prototype_sources = parse_label_sources(args.decoder_prototype_source)
        decoder_prototype_targets, decoder_prototype_counts = build_decoder_prototypes(
            embeddings,
            style_targets,
            style_label_mask,
            data.get('style_label_source'),
            supported_styles,
            prototype_sources,
            args.decoder_prototype_min_count,
        )
        decoder_prototype_mask = build_dataset_mask(
            source_datasets,
            args.decoder_prototype_datasets,
            device,
        )
        if args.decoder_prototype_style_weights:
            decoder_prototype_row_weights, decoder_prototype_style_weights = (
                build_style_teacher_row_weights(
                    style_targets,
                    style_label_mask,
                    style_label_confidence,
                    supported_styles,
                    args.decoder_prototype_style_weights,
                    confidence_power=0.0,
                    require_label=True,
                )
            )
            if style_label_row_weights is not None:
                decoder_prototype_row_weights = (
                    decoder_prototype_row_weights * style_label_row_weights
                )
        else:
            decoder_prototype_row_weights = style_label_row_weights
        decoder_prototype_strength_map = parse_style_strengths(
            args.decoder_prototype_style_strengths,
            supported_styles,
            args.decoder_prototype_strength,
        )
        decoder_prototype_strengths = torch.tensor(
            [decoder_prototype_strength_map[style] for style in supported_styles],
            dtype=torch.float32,
            device=device,
        )

    labeled_rows = int((style_label_mask.view(-1) > 0).sum().item())
    print(f"Supported styles: {supported_styles}")
    print(f"Labeled rows: {labeled_rows}/{len(embeddings)}")
    if teacher_info:
        print(f"Pseudo-label teacher: {teacher_info.get('model')}")
        if teacher_info.get('teacher_checkpoint'):
            print(f"Teacher checkpoint: {teacher_info.get('teacher_checkpoint')}")
    if mixture_report.get('commonvoice_acceptance_policy'):
        print(f"CommonVoice acceptance policy: {mixture_report['commonvoice_acceptance_policy']}")
    if filter_info.get('acceptance_policy'):
        print(f"Pre-filtered CommonVoice policy: {filter_info['acceptance_policy']}")
    if metadata_control_report.get("enabled"):
        print(f"Metadata control dims: {metadata_control_report['dims']}")
        print(f"Metadata control labeled rows: {metadata_control_report['labeled_rows']}")

    model = dpvc.VariationalAutoencoder(
        latent_dims=args.latent_dims,
        input_dim=embeddings.shape[-1],
    ).to(device)
    if args.init_checkpoint:
        print(f"Loading init checkpoint from {args.init_checkpoint}")
        model.load_state_dict(
            torch.load(args.init_checkpoint, weights_only=True, map_location=device)
        )

    if args.freeze_encoder:
        set_module_requires_grad(model.encoder, trainable=False)
    if args.freeze_decoder:
        set_module_requires_grad(model.decoder, trainable=False)

    style_teacher_model = None
    if teacher_requested:
        print(f"Loading style teacher checkpoint from {args.style_teacher_checkpoint}")
        style_teacher_model = dpvc.VariationalAutoencoder(
            latent_dims=args.latent_dims,
            input_dim=embeddings.shape[-1],
        ).to(device)
        style_teacher_model.load_state_dict(
            torch.load(args.style_teacher_checkpoint, weights_only=True, map_location=device)
        )
        style_teacher_model.eval()
        set_module_requires_grad(style_teacher_model, trainable=False)
        print(f"Style teacher dims: {style_teacher_dims}")
        print(f"Style teacher datasets: {args.style_teacher_datasets}")
        if args.style_teacher_weight_final is not None:
            print(f"Style teacher weight final: {args.style_teacher_weight_final}")
        print(f"Style teacher target mode: {args.style_teacher_target_mode}")
        print(f"Style teacher require label: {args.style_teacher_require_label}")
        if args.style_teacher_style_weights:
            print(f"Style teacher style weights: {style_teacher_style_weights}")
        if args.style_teacher_confidence_power > 0:
            print(f"Style teacher confidence power: {args.style_teacher_confidence_power}")
    if anti_neutral_requested:
        print(f"Anti-neutral datasets: {args.anti_neutral_datasets}")
        print(f"Anti-neutral styles: {anti_neutral_styles}")
        print(f"Anti-neutral mode: {args.anti_neutral_mode}")
        print(f"Anti-neutral margin: {args.anti_neutral_margin}")
        print(f"Anti-neutral control mode: {args.anti_neutral_control_mode}")
        print(f"Anti-neutral strengths: {anti_neutral_strength_map}")
        if args.anti_neutral_weight_final is not None:
            print(f"Anti-neutral weight final: {args.anti_neutral_weight_final}")
        if args.anti_neutral_style_weights:
            print(f"Anti-neutral style weights: {anti_neutral_style_weights}")
    if decoder_prototype_requested:
        print(f"Decoder prototype source: {args.decoder_prototype_source}")
        print(f"Decoder prototype counts: {decoder_prototype_counts}")
        print(f"Decoder prototype datasets: {args.decoder_prototype_datasets}")
        print(f"Decoder prototype control mode: {args.decoder_prototype_control_mode}")
        print(f"Decoder prototype strengths: {decoder_prototype_strength_map}")
        if args.decoder_prototype_style_weights:
            print(f"Decoder prototype style weights: {decoder_prototype_style_weights}")

    trainable_params, total_params = format_param_count(model)
    print(f"Freeze encoder: {args.freeze_encoder}")
    print(f"Freeze decoder: {args.freeze_decoder}")
    print(f"Trainable parameters: {trainable_params}/{total_params}")
    print(f"Schedule: {args.schedule}")
    if args.schedule != 'static_balanced':
        print(
            f"Schedule epochs: {args.schedule_epochs if args.schedule_epochs > 0 else args.epochs}"
        )
    if static_masses:
        print(f"Static masses: {static_masses}")
    if schedule_start_masses:
        print(f"Schedule start masses: {schedule_start_masses}")
    if schedule_end_masses:
        print(f"Schedule end masses: {schedule_end_masses}")

    dpvc.utils.train_mixed_autoencoder(
        model,
        embeddings,
        style_targets,
        style_label_mask,
        source_datasets=source_datasets,
        epochs=args.epochs,
        lr=args.lr,
        recon_weight=args.recon_weight,
        kl_weight=args.kl_weight,
        label_weight=args.label_weight,
        schedule=args.schedule,
        schedule_epochs=args.schedule_epochs,
        style_label_row_weights=style_label_row_weights,
        metadata_targets=metadata_targets,
        metadata_label_mask=metadata_label_mask,
        metadata_control_dims=metadata_control_dims,
        metadata_control_weight=args.metadata_control_weight,
        static_masses=static_masses,
        schedule_start_masses=schedule_start_masses,
        schedule_end_masses=schedule_end_masses,
        style_teacher_model=style_teacher_model,
        style_teacher_weight=args.style_teacher_weight,
        style_teacher_weight_final=args.style_teacher_weight_final,
        style_teacher_dims=style_teacher_dims,
        style_teacher_mask=style_teacher_mask,
        style_teacher_row_weights=style_teacher_row_weights,
        style_teacher_target_mode=args.style_teacher_target_mode,
        decoder_prototype_targets=decoder_prototype_targets,
        decoder_prototype_weight=args.decoder_prototype_weight,
        decoder_prototype_weight_final=args.decoder_prototype_weight_final,
        decoder_prototype_mask=decoder_prototype_mask,
        decoder_prototype_row_weights=decoder_prototype_row_weights,
        decoder_prototype_strengths=decoder_prototype_strengths,
        decoder_prototype_control_mode=args.decoder_prototype_control_mode,
        anti_neutral_weight=args.anti_neutral_weight,
        anti_neutral_weight_final=args.anti_neutral_weight_final,
        anti_neutral_mask=anti_neutral_mask,
        anti_neutral_row_weights=anti_neutral_row_weights,
        anti_neutral_strengths=anti_neutral_strengths,
        anti_neutral_margin=args.anti_neutral_margin,
        anti_neutral_neutral_index=anti_neutral_neutral_index,
        anti_neutral_control_mode=args.anti_neutral_control_mode,
        anti_neutral_mode=args.anti_neutral_mode,
        anti_neutral_prototype_targets=(
            decoder_prototype_targets if args.anti_neutral_mode == "prototype_margin" else None
        ),
    )

    torch.save(model.state_dict(), args.output)
    print(f"Saved mixed-data VAE checkpoint to {args.output}")
    if args.metadata_control_report:
        report_path = Path(args.metadata_control_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata_control_report, handle, indent=2)
        print(f"Saved metadata-control report to {report_path}")
    if args.generated_audio_objective_report:
        report_path = Path(args.generated_audio_objective_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(generated_audio_objective_report, handle, indent=2)
        print(f"Saved generated-audio objective report to {report_path}")


if __name__ == '__main__':
    main()
