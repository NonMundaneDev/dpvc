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

    if args.freeze_encoder and args.freeze_decoder:
        ap.error("Refusing to freeze both encoder and decoder; nothing would remain trainable")
    if (
        (args.style_teacher_weight > 0 or (args.style_teacher_weight_final or 0.0) > 0)
        and not args.style_teacher_checkpoint
    ):
        ap.error("--style-teacher-weight > 0 or --style-teacher-weight-final > 0 requires --style-teacher-checkpoint")
    if args.decoder_prototype_weight < 0 or (args.decoder_prototype_weight_final or 0.0) < 0:
        ap.error("Decoder-prototype weights must be non-negative")
    if args.decoder_prototype_strength < 0:
        ap.error("--decoder-prototype-strength must be non-negative")
    if args.decoder_prototype_min_count < 1:
        ap.error("--decoder-prototype-min-count must be >= 1")

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
    teacher_requested = (
        args.style_teacher_checkpoint
        and (args.style_teacher_weight > 0 or (args.style_teacher_weight_final or 0.0) > 0)
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

    decoder_prototype_requested = (
        args.decoder_prototype_weight > 0
        or (args.decoder_prototype_weight_final or 0.0) > 0
    )
    decoder_prototype_targets = None
    decoder_prototype_mask = None
    decoder_prototype_row_weights = None
    decoder_prototype_strengths = None
    decoder_prototype_counts = None
    decoder_prototype_strength_map = None
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
    if decoder_prototype_requested:
        print(f"Decoder prototype source: {args.decoder_prototype_source}")
        print(f"Decoder prototype counts: {decoder_prototype_counts}")
        print(f"Decoder prototype datasets: {args.decoder_prototype_datasets}")
        print(f"Decoder prototype control mode: {args.decoder_prototype_control_mode}")
        print(f"Decoder prototype strengths: {decoder_prototype_strength_map}")

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
    )

    torch.save(model.state_dict(), args.output)
    print(f"Saved mixed-data VAE checkpoint to {args.output}")


if __name__ == '__main__':
    main()
