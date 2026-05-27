import torch
import random
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import requests
import zipfile
import contextlib
from collections import Counter
from typing import List, Optional, Sequence


def set_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

def extract_embeddings(vc_wrapper, dataset: List[str]) -> torch.Tensor:
    """Extract speaker embeddings from many source .wav files"""
    embeddings = []
    print('Extracting embeddings...')
    for wav_file in tqdm(dataset):
        try:
            with contextlib.redirect_stdout(None):
                embedding = vc_wrapper.extract_embedding(wav_file)
                embeddings.append(embedding)
        except Exception as e:
            print('Error extracting embedding:', e)

    return torch.vstack(embeddings).squeeze()


def _interpolate_weight(start, end, epoch, schedule_epochs):
    if end is None or schedule_epochs <= 0:
        return start
    progress = min(max(epoch, 0), schedule_epochs) / schedule_epochs
    return start + (end - start) * progress


def _slice_or_none(tensor: torch.Tensor, dims: Optional[Sequence[int]]):
    if dims is None:
        return None
    if not dims:
        return tensor.new_zeros((tensor.shape[0], 0))
    return tensor[:, list(dims)]


def _apply_style_controls_to_latents(
        latents: torch.Tensor,
        style_targets: torch.Tensor,
        style_strengths: torch.Tensor,
        control_mode: str):
    controlled = latents.clone()
    num_styles = style_targets.shape[1]
    if num_styles > controlled.shape[1]:
        raise ValueError(
            f"Style target width ({num_styles}) exceeds latent width ({controlled.shape[1]})"
        )

    if control_mode == "all_style_dims":
        controlled[:, :num_styles] = style_targets * style_strengths.view(1, -1)
        return controlled

    if control_mode != "target_only":
        raise ValueError(f"Unsupported decoder prototype control mode: {control_mode}")

    target_indexes = torch.argmax(style_targets, dim=1)
    row_indexes = torch.arange(controlled.shape[0], device=controlled.device)
    controlled[row_indexes, target_indexes] = style_strengths[target_indexes]
    return controlled


def is_missing_metadata(value):
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def summarize_categorical_values(values, top_n=10):
    total = len(values)
    normalized = [str(value) for value in values if not is_missing_metadata(value)]
    counts = Counter(normalized)
    return {
        'total': total,
        'known': len(normalized),
        'missing': total - len(normalized),
        'unique_known': len(counts),
        'top_values': [
            {'value': value, 'count': count}
            for value, count in counts.most_common(top_n)
        ],
    }


def build_commonvoice_metadata_report(age_values, gender_values, accent_values):
    return {
        'age': summarize_categorical_values(age_values),
        'gender': summarize_categorical_values(gender_values),
        'accent': summarize_categorical_values(accent_values),
    }


def train_autoencoder(model, embeddings, epochs=1000, labels=None, lr=1e-5,
                      recon_weight=1.0, kl_weight=1.0, label_weight=1.0,
                      recon_weight_final=None, kl_weight_final=None,
                      label_weight_final=None, schedule_epochs=0,
                      style_teacher_model=None, style_teacher_weight=0.0,
                      style_teacher_weight_final=None,
                      free_anchor_model=None, free_anchor_weight=0.0,
                      free_anchor_weight_final=None,
                      style_dims=None, free_dims=None):
    BATCH_SIZE = min(256, len(embeddings))
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters remain in the model")
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    if labels is not None:
        num_labels = len(labels.keys())
        print('Label ordering:', [f for f in labels])
        label_vals = [list(labels[f]) for f in labels]
        label_tensor = torch.tensor(label_vals).to(embeddings.device).squeeze().T
    else:
        num_labels = 0
        label_tensor = None

    if schedule_epochs <= 0:
        schedule_epochs = 0

    print('Loss weights:')
    print(f'  recon: {recon_weight}'
          + (f' -> {recon_weight_final}' if recon_weight_final is not None else ''))
    print(f'  kl   : {kl_weight}'
          + (f' -> {kl_weight_final}' if kl_weight_final is not None else ''))
    print(f'  label: {label_weight}'
          + (f' -> {label_weight_final}' if label_weight_final is not None else ''))
    print(f'  teacher-style: {style_teacher_weight}'
          + (f' -> {style_teacher_weight_final}' if style_teacher_weight_final is not None else ''))
    print(f'  free-anchor : {free_anchor_weight}'
          + (f' -> {free_anchor_weight_final}' if free_anchor_weight_final is not None else ''))
    if schedule_epochs:
        print(f'  schedule epochs: {schedule_epochs}')
    if style_teacher_model is not None:
        print(f'  style dims for teacher loss: {list(style_dims or [])}')
    if free_anchor_model is not None:
        print(f'  free dims for anchor loss: {list(free_dims or [])}')

    print(f'Training autoencoder for {epochs} epochs...')
    for epoch in tqdm(range(epochs)):
        current_recon_weight = _interpolate_weight(
            recon_weight, recon_weight_final, epoch, schedule_epochs)
        current_kl_weight = _interpolate_weight(
            kl_weight, kl_weight_final, epoch, schedule_epochs)
        current_label_weight = _interpolate_weight(
            label_weight, label_weight_final, epoch, schedule_epochs)
        current_style_teacher_weight = _interpolate_weight(
            style_teacher_weight, style_teacher_weight_final, epoch, schedule_epochs)
        current_free_anchor_weight = _interpolate_weight(
            free_anchor_weight, free_anchor_weight_final, epoch, schedule_epochs)

        with torch.no_grad():
            indexes = torch.randperm(embeddings.shape[0])
            embeddings_batches = torch.split(embeddings[indexes], BATCH_SIZE)
            if label_tensor is not None:
                labels_batches = torch.split(label_tensor[indexes], BATCH_SIZE)
            else:
                labels_batches = [None] * len(embeddings_batches)

        for embeddings_b, labels_b in zip(embeddings_batches, labels_batches):
            optimizer.zero_grad()

            reconstructed = model(embeddings_b)
            recon_loss = ((embeddings_b - reconstructed)**2).sum()
            kl_loss = model.kl
            student_mu = model.last_mu

            if labels_b is not None:
                label_loss = ((model.last_z[:, :num_labels] - labels_b)**2).sum()
            else:
                label_loss = embeddings_b.new_tensor(0.0)

            if style_teacher_model is not None and style_dims:
                with torch.no_grad():
                    teacher_mu, _ = style_teacher_model.encoder(embeddings_b)
                teacher_style_loss = (
                    (_slice_or_none(student_mu, style_dims) - _slice_or_none(teacher_mu, style_dims)) ** 2
                ).sum()
            else:
                teacher_style_loss = embeddings_b.new_tensor(0.0)

            if free_anchor_model is not None and free_dims:
                with torch.no_grad():
                    anchor_mu, _ = free_anchor_model.encoder(embeddings_b)
                free_anchor_loss = (
                    (_slice_or_none(student_mu, free_dims) - _slice_or_none(anchor_mu, free_dims)) ** 2
                ).sum()
            else:
                free_anchor_loss = embeddings_b.new_tensor(0.0)

            weighted_recon = current_recon_weight * recon_loss
            weighted_kl = current_kl_weight * kl_loss
            weighted_label = current_label_weight * label_loss
            weighted_teacher_style = current_style_teacher_weight * teacher_style_loss
            weighted_free_anchor = current_free_anchor_weight * free_anchor_loss
            loss = (
                weighted_recon
                + weighted_kl
                + weighted_label
                + weighted_teacher_style
                + weighted_free_anchor
            )

            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(
                f'loss: {loss.item():.2f}  '
                f'recon: {recon_loss.item():.2f} (w={current_recon_weight:.2f})  '
                f'kl: {kl_loss.item():.2f} (w={current_kl_weight:.2f})  '
                f'label: {label_loss.item():.2f} (w={current_label_weight:.2f})  '
                f'teacher: {teacher_style_loss.item():.2f} (w={current_style_teacher_weight:.2f})  '
                f'anchor: {free_anchor_loss.item():.2f} (w={current_free_anchor_weight:.2f})'
            )

    print('Ending loss:', loss.item())


def _normalize_dataset_masses(raw_masses, present_datasets):
    filtered = {
        dataset: max(float(raw_masses.get(dataset, 0.0)), 0.0)
        for dataset in present_datasets
    }
    total = sum(filtered.values())
    if total <= 0:
        uniform = 1.0 / max(len(present_datasets), 1)
        return {dataset: uniform for dataset in present_datasets}
    return {dataset: value / total for dataset, value in filtered.items()}


def _dataset_epoch_masses(schedule, epoch, schedule_epochs, present_datasets,
                          static_masses=None, schedule_start_masses=None,
                          schedule_end_masses=None):
    if schedule == "static_balanced":
        return _normalize_dataset_masses(
            static_masses or {"CommonVoice": 1.0, "CREMA-D": 1.0, "Expresso": 1.0},
            present_datasets,
        )

    if schedule_epochs <= 0:
        schedule_epochs = 1
    progress = min(max(epoch, 0), schedule_epochs) / schedule_epochs

    if schedule == "cv_warmup":
        start = {"CommonVoice": 0.70, "CREMA-D": 0.15, "Expresso": 0.15}
        end = {"CommonVoice": 1.0, "CREMA-D": 1.0, "Expresso": 1.0}
    elif schedule == "labeled_warmup":
        start = {"CommonVoice": 0.0, "CREMA-D": 1.0, "Expresso": 1.0}
        end = {"CommonVoice": 1.0, "CREMA-D": 1.0, "Expresso": 1.0}
    elif schedule == "labeled_finish":
        start = {"CommonVoice": 1.0, "CREMA-D": 1.0, "Expresso": 1.0}
        end = {"CommonVoice": 0.20, "CREMA-D": 0.40, "Expresso": 0.40}
    else:
        raise ValueError(f"Unsupported mixed-data schedule: {schedule}")
    if schedule_start_masses is not None:
        start = schedule_start_masses
    if schedule_end_masses is not None:
        end = schedule_end_masses

    raw = {
        dataset: start.get(dataset, 0.0) + (end.get(dataset, 0.0) - start.get(dataset, 0.0)) * progress
        for dataset in present_datasets
    }
    return _normalize_dataset_masses(raw, present_datasets)


def train_mixed_autoencoder(model, embeddings, style_targets, style_label_mask,
                            source_datasets, epochs=1000, lr=1e-5,
                            recon_weight=1.0, kl_weight=1.0,
                            label_weight=1.0, schedule="static_balanced",
                            schedule_epochs=0, style_label_row_weights=None,
                            metadata_targets=None,
                            metadata_label_mask=None,
                            metadata_control_dims=None,
                            metadata_control_weight=0.0,
                            static_masses=None, schedule_start_masses=None,
                            schedule_end_masses=None,
                            style_teacher_model=None,
                            style_teacher_weight=0.0,
                            style_teacher_weight_final=None,
                            style_teacher_dims=None,
                            style_teacher_mask=None,
                            style_teacher_row_weights=None,
                            style_teacher_target_mode="all_dims",
                            decoder_prototype_targets=None,
                            decoder_prototype_weight=0.0,
                            decoder_prototype_weight_final=None,
                            decoder_prototype_mask=None,
                            decoder_prototype_row_weights=None,
                            decoder_prototype_strengths=None,
                            decoder_prototype_control_mode="target_only",
                            anti_neutral_weight=0.0,
                            anti_neutral_weight_final=None,
                            anti_neutral_mask=None,
                            anti_neutral_row_weights=None,
                            anti_neutral_strengths=None,
                            anti_neutral_margin=0.5,
                            anti_neutral_neutral_index=None,
                            anti_neutral_control_mode="target_only",
                            anti_neutral_mode="teacher_margin",
                            anti_neutral_prototype_targets=None):
    BATCH_SIZE = min(256, len(embeddings))
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters remain in the model")
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    metadata_control_dims = list(metadata_control_dims or [])
    if metadata_control_weight > 0:
        if metadata_targets is None or metadata_label_mask is None:
            raise ValueError(
                "metadata_control_weight > 0 requires metadata_targets and "
                "metadata_label_mask"
            )
        if not metadata_control_dims:
            raise ValueError(
                "metadata_control_weight > 0 requires metadata_control_dims"
            )
    if metadata_targets is not None:
        if metadata_targets.shape[0] != embeddings.shape[0]:
            raise ValueError(
                "metadata_targets row count must match embeddings: "
                f"{metadata_targets.shape[0]} != {embeddings.shape[0]}"
            )
        if metadata_targets.shape[1] != len(metadata_control_dims):
            raise ValueError(
                "metadata_targets width must match metadata_control_dims: "
                f"{metadata_targets.shape[1]} != {len(metadata_control_dims)}"
            )
    if metadata_label_mask is not None:
        if metadata_targets is None:
            raise ValueError("metadata_label_mask requires metadata_targets")
        if metadata_label_mask.shape != metadata_targets.shape:
            raise ValueError(
                "metadata_label_mask shape must match metadata_targets: "
                f"{metadata_label_mask.shape} != {metadata_targets.shape}"
            )
    if metadata_control_dims and (
        min(metadata_control_dims) < 0 or max(metadata_control_dims) >= model.latent_dims
    ):
        raise ValueError(
            "metadata_control_dims must be inside the VAE latent width: "
            f"{metadata_control_dims} for latent_dims={model.latent_dims}"
        )

    if schedule_epochs <= 0 and (
        schedule != "static_balanced"
        or style_teacher_weight_final is not None
        or decoder_prototype_weight_final is not None
        or anti_neutral_weight_final is not None
    ):
        schedule_epochs = epochs

    source_datasets = [str(dataset) for dataset in source_datasets]
    dataset_counts = Counter(source_datasets)
    present_datasets = [dataset for dataset in ["CommonVoice", "CREMA-D", "Expresso"] if dataset_counts.get(dataset)]
    if not present_datasets:
        raise ValueError("No source datasets found for mixed-data training")

    print("Mixed-data training config:")
    print(f"  recon weight : {recon_weight}")
    print(f"  kl weight    : {kl_weight}")
    print(f"  label weight : {label_weight}")
    print(f"  metadata-control weight: {metadata_control_weight}")
    print(f"  teacher-style weight: {style_teacher_weight}"
          + (f" -> {style_teacher_weight_final}" if style_teacher_weight_final is not None else ""))
    print(f"  decoder-prototype weight: {decoder_prototype_weight}"
          + (f" -> {decoder_prototype_weight_final}" if decoder_prototype_weight_final is not None else ""))
    print(f"  anti-neutral weight: {anti_neutral_weight}"
          + (f" -> {anti_neutral_weight_final}" if anti_neutral_weight_final is not None else ""))
    print(f"  schedule     : {schedule}")
    if schedule != "static_balanced":
        print(f"  schedule epochs: {schedule_epochs}")
    if static_masses is not None:
        print(f"  static masses : {static_masses}")
    if schedule_start_masses is not None:
        print(f"  start masses  : {schedule_start_masses}")
    if schedule_end_masses is not None:
        print(f"  end masses    : {schedule_end_masses}")
    print(f"  styles       : {style_targets.shape[1]}")
    for dataset in present_datasets:
        print(f"  dataset rows {dataset:11s}: {dataset_counts[dataset]}")
    labeled_rows = int((style_label_mask.view(-1) > 0).sum().item())
    print(f"  labeled rows : {labeled_rows}/{len(embeddings)}")
    metadata_control_enabled = (
        metadata_targets is not None
        and metadata_label_mask is not None
        and metadata_control_dims
        and metadata_control_weight > 0
    )
    if metadata_targets is not None and metadata_label_mask is not None:
        metadata_counts = metadata_label_mask.sum(dim=0).detach().cpu().tolist()
        print(f"  metadata-control dims: {metadata_control_dims}")
        print(
            "  metadata-control rows by dim: "
            f"{[int(value) for value in metadata_counts]}"
        )
    teacher_enabled = (
        style_teacher_model is not None
        and (
            style_teacher_weight > 0
            or (style_teacher_weight_final is not None and style_teacher_weight_final > 0)
        )
    )
    if teacher_enabled:
        teacher_active = torch.ones(len(embeddings), dtype=torch.bool, device=embeddings.device)
        if style_teacher_mask is not None:
            teacher_active = teacher_active & (style_teacher_mask.view(-1) > 0)
        if style_teacher_row_weights is not None:
            teacher_active = teacher_active & (style_teacher_row_weights.view(-1) > 0)
        teacher_rows = int(teacher_active.sum().item())
        print(f"  teacher-style dims: {list(style_teacher_dims or [])}")
        print(f"  teacher-style rows: {teacher_rows}/{len(embeddings)}")
        print(f"  teacher target mode: {style_teacher_target_mode}")
        if style_teacher_row_weights is not None:
            nonzero_weights = style_teacher_row_weights.view(-1)[style_teacher_row_weights.view(-1) > 0]
            if nonzero_weights.numel() > 0:
                print(
                    "  teacher row weights: "
                    f"mean={nonzero_weights.mean().item():.3f} "
                    f"min={nonzero_weights.min().item():.3f} "
                    f"max={nonzero_weights.max().item():.3f}"
                )
    decoder_prototype_enabled = (
        decoder_prototype_targets is not None
        and (
            decoder_prototype_weight > 0
            or (
                decoder_prototype_weight_final is not None
                and decoder_prototype_weight_final > 0
            )
        )
    )
    anti_neutral_enabled = (
        anti_neutral_weight > 0
        or (
            anti_neutral_weight_final is not None
            and anti_neutral_weight_final > 0
        )
    )
    if decoder_prototype_enabled:
        if decoder_prototype_targets.shape[0] != style_targets.shape[1]:
            raise ValueError(
                "decoder_prototype_targets must have one row per style target "
                f"({style_targets.shape[1]}), got {decoder_prototype_targets.shape[0]}"
            )
        if decoder_prototype_strengths is None:
            decoder_prototype_strengths = torch.ones(
                style_targets.shape[1],
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
        decoder_active = style_label_mask.view(-1) > 0
        if decoder_prototype_mask is not None:
            decoder_active = decoder_active & (decoder_prototype_mask.view(-1) > 0)
        if decoder_prototype_row_weights is not None:
            decoder_active = decoder_active & (decoder_prototype_row_weights.view(-1) > 0)
        decoder_rows = int(decoder_active.sum().item())
        nonzero_strengths = decoder_prototype_strengths.detach().cpu().tolist()
        print(f"  decoder-prototype rows: {decoder_rows}/{len(embeddings)}")
        print(f"  decoder-prototype control mode: {decoder_prototype_control_mode}")
        print(f"  decoder-prototype strengths: {[round(float(v), 4) for v in nonzero_strengths]}")
        if decoder_prototype_row_weights is not None:
            nonzero_weights = decoder_prototype_row_weights.view(-1)[
                decoder_prototype_row_weights.view(-1) > 0
            ]
            if nonzero_weights.numel() > 0:
                print(
                    "  decoder-prototype row weights: "
                    f"mean={nonzero_weights.mean().item():.3f} "
                    f"min={nonzero_weights.min().item():.3f} "
                    f"max={nonzero_weights.max().item():.3f}"
                )
    if anti_neutral_enabled:
        if anti_neutral_neutral_index is None:
            raise ValueError(
                "anti_neutral_neutral_index is required when anti-neutral is enabled"
            )
        if anti_neutral_mode == "teacher_margin" and style_teacher_model is None:
            raise ValueError("teacher_margin anti-neutral mode requires style_teacher_model")
        if anti_neutral_mode == "prototype_margin" and anti_neutral_prototype_targets is None:
            raise ValueError(
                "prototype_margin anti-neutral mode requires anti_neutral_prototype_targets"
            )
        if anti_neutral_mode not in {"teacher_margin", "prototype_margin"}:
            raise ValueError(f"Unsupported anti_neutral_mode: {anti_neutral_mode}")
        if anti_neutral_strengths is None:
            anti_neutral_strengths = torch.ones(
                style_targets.shape[1],
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
        anti_active = style_label_mask.view(-1) > 0
        if anti_neutral_mask is not None:
            anti_active = anti_active & (anti_neutral_mask.view(-1) > 0)
        if anti_neutral_row_weights is not None:
            anti_active = anti_active & (anti_neutral_row_weights.view(-1) > 0)
        anti_rows = int(anti_active.sum().item())
        nonzero_strengths = anti_neutral_strengths.detach().cpu().tolist()
        print(f"  anti-neutral rows: {anti_rows}/{len(embeddings)}")
        print(f"  anti-neutral mode: {anti_neutral_mode}")
        print(f"  anti-neutral neutral index: {anti_neutral_neutral_index}")
        print(f"  anti-neutral margin: {anti_neutral_margin}")
        print(f"  anti-neutral control mode: {anti_neutral_control_mode}")
        print(f"  anti-neutral strengths: {[round(float(v), 4) for v in nonzero_strengths]}")
        if anti_neutral_row_weights is not None:
            nonzero_weights = anti_neutral_row_weights.view(-1)[
                anti_neutral_row_weights.view(-1) > 0
            ]
            if nonzero_weights.numel() > 0:
                print(
                    "  anti-neutral row weights: "
                    f"mean={nonzero_weights.mean().item():.3f} "
                    f"min={nonzero_weights.min().item():.3f} "
                    f"max={nonzero_weights.max().item():.3f}"
                )

    print(f"Training mixed-data autoencoder for {epochs} epochs...")
    for epoch in tqdm(range(epochs)):
        current_style_teacher_weight = _interpolate_weight(
            style_teacher_weight,
            style_teacher_weight_final,
            epoch,
            schedule_epochs,
        )
        current_decoder_prototype_weight = _interpolate_weight(
            decoder_prototype_weight,
            decoder_prototype_weight_final,
            epoch,
            schedule_epochs,
        )
        current_anti_neutral_weight = _interpolate_weight(
            anti_neutral_weight,
            anti_neutral_weight_final,
            epoch,
            schedule_epochs,
        )
        dataset_masses = _dataset_epoch_masses(
            schedule,
            epoch,
            schedule_epochs,
            present_datasets,
            static_masses=static_masses,
            schedule_start_masses=schedule_start_masses,
            schedule_end_masses=schedule_end_masses,
        )
        per_row_probs = torch.tensor(
            [dataset_masses[dataset] / dataset_counts[dataset] for dataset in source_datasets],
            dtype=torch.float32,
            device=embeddings.device,
        )
        per_row_probs = per_row_probs / per_row_probs.sum()
        sampled_indexes = torch.multinomial(
            per_row_probs,
            num_samples=embeddings.shape[0],
            replacement=True,
        )
        index_batches = torch.split(sampled_indexes, BATCH_SIZE)

        for batch_indexes in index_batches:
            embeddings_b = embeddings[batch_indexes]
            optimizer.zero_grad()

            reconstructed = model(embeddings_b)
            recon_loss = ((embeddings_b - reconstructed)**2).sum()
            kl_loss = model.kl
            teacher_style_loss = embeddings_b.new_tensor(0.0)
            decoder_prototype_loss = embeddings_b.new_tensor(0.0)
            anti_neutral_loss = embeddings_b.new_tensor(0.0)
            metadata_loss = embeddings_b.new_tensor(0.0)

            batch_mask = style_label_mask[batch_indexes].view(-1) > 0
            if batch_mask.any():
                target_b = style_targets[batch_indexes][batch_mask]
                student_b = model.last_z[batch_mask, :style_targets.shape[1]]
                row_loss = ((student_b - target_b)**2).sum(dim=1)
                if style_label_row_weights is not None:
                    weights_b = style_label_row_weights[batch_indexes].view(-1)[batch_mask]
                    label_loss = (row_loss * weights_b).sum()
                else:
                    label_loss = row_loss.sum()
            else:
                label_loss = embeddings_b.new_tensor(0.0)

            if metadata_control_enabled:
                metadata_mask_b = metadata_label_mask[batch_indexes] > 0
                if metadata_mask_b.any():
                    metadata_target_b = metadata_targets[batch_indexes]
                    metadata_student_b = _slice_or_none(
                        model.last_z,
                        metadata_control_dims,
                    )
                    metadata_errors = (
                        (metadata_student_b - metadata_target_b) ** 2
                    ) * metadata_mask_b.to(metadata_student_b.dtype)
                    metadata_loss = metadata_errors.sum()

            if (
                style_teacher_model is not None
                and current_style_teacher_weight > 0
                and style_teacher_dims
            ):
                if style_teacher_mask is not None:
                    teacher_batch_mask = style_teacher_mask[batch_indexes].view(-1) > 0
                else:
                    teacher_batch_mask = torch.ones(
                        embeddings_b.shape[0],
                        dtype=torch.bool,
                        device=embeddings_b.device,
                    )
                if teacher_batch_mask.any():
                    with torch.no_grad():
                        teacher_mu, _ = style_teacher_model.encoder(embeddings_b)
                    teacher_errors = (
                        _slice_or_none(model.last_mu[teacher_batch_mask], style_teacher_dims)
                        - _slice_or_none(teacher_mu[teacher_batch_mask], style_teacher_dims)
                    ) ** 2
                    if style_teacher_target_mode == "target_dim":
                        dim_weights = _slice_or_none(
                            style_targets[batch_indexes][teacher_batch_mask],
                            style_teacher_dims,
                        ) > 0
                        dim_weights = dim_weights.to(teacher_errors.dtype)
                        teacher_errors = teacher_errors * dim_weights
                    elif style_teacher_target_mode != "all_dims":
                        raise ValueError(
                            f"Unsupported style_teacher_target_mode: {style_teacher_target_mode}"
                        )
                    teacher_row_loss = teacher_errors.sum(dim=1)
                    if style_teacher_row_weights is not None:
                        row_weights = style_teacher_row_weights[batch_indexes].view(-1)[
                            teacher_batch_mask
                        ]
                        teacher_row_loss = teacher_row_loss * row_weights
                    teacher_style_loss = teacher_row_loss.sum()

            if decoder_prototype_enabled and current_decoder_prototype_weight > 0:
                decoder_batch_mask = style_label_mask[batch_indexes].view(-1) > 0
                if decoder_prototype_mask is not None:
                    decoder_batch_mask = (
                        decoder_batch_mask
                        & (decoder_prototype_mask[batch_indexes].view(-1) > 0)
                    )
                if decoder_prototype_row_weights is not None:
                    decoder_batch_mask = (
                        decoder_batch_mask
                        & (decoder_prototype_row_weights[batch_indexes].view(-1) > 0)
                    )
                if decoder_batch_mask.any():
                    decoder_targets_b = style_targets[batch_indexes][decoder_batch_mask]
                    controlled_latents = _apply_style_controls_to_latents(
                        model.last_mu[decoder_batch_mask],
                        decoder_targets_b,
                        decoder_prototype_strengths,
                        decoder_prototype_control_mode,
                    )
                    decoded_style_embeddings = model.decoder(controlled_latents)
                    prototype_indexes = torch.argmax(decoder_targets_b, dim=1)
                    prototype_targets_b = decoder_prototype_targets[prototype_indexes]
                    decoder_row_loss = (
                        (decoded_style_embeddings - prototype_targets_b) ** 2
                    ).sum(dim=1)
                    if decoder_prototype_row_weights is not None:
                        row_weights = decoder_prototype_row_weights[batch_indexes].view(-1)[
                            decoder_batch_mask
                        ]
                        decoder_row_loss = decoder_row_loss * row_weights
                    decoder_prototype_loss = decoder_row_loss.sum()

            if anti_neutral_enabled and current_anti_neutral_weight > 0:
                anti_batch_mask = style_label_mask[batch_indexes].view(-1) > 0
                if anti_neutral_mask is not None:
                    anti_batch_mask = (
                        anti_batch_mask
                        & (anti_neutral_mask[batch_indexes].view(-1) > 0)
                    )
                if anti_neutral_row_weights is not None:
                    anti_batch_mask = (
                        anti_batch_mask
                        & (anti_neutral_row_weights[batch_indexes].view(-1) > 0)
                    )
                if anti_batch_mask.any():
                    anti_targets_b = style_targets[batch_indexes][anti_batch_mask]
                    controlled_latents = _apply_style_controls_to_latents(
                        model.last_mu[anti_batch_mask],
                        anti_targets_b,
                        anti_neutral_strengths,
                        anti_neutral_control_mode,
                    )
                    decoded_style_embeddings = model.decoder(controlled_latents)
                    target_indexes = torch.argmax(anti_targets_b, dim=1)
                    row_indexes = torch.arange(
                        target_indexes.shape[0],
                        device=target_indexes.device,
                    )
                    if anti_neutral_mode == "teacher_margin":
                        teacher_decoded_mu, _ = style_teacher_model.encoder(
                            decoded_style_embeddings
                        )
                        target_scores = teacher_decoded_mu[row_indexes, target_indexes]
                        neutral_scores = teacher_decoded_mu[:, anti_neutral_neutral_index]
                        anti_row_loss = torch.relu(
                            neutral_scores - target_scores + anti_neutral_margin
                        )
                    elif anti_neutral_mode == "prototype_margin":
                        target_prototypes = anti_neutral_prototype_targets[target_indexes]
                        neutral_prototypes = anti_neutral_prototype_targets[
                            anti_neutral_neutral_index
                        ].view(1, -1)
                        target_dist = (
                            (decoded_style_embeddings - target_prototypes) ** 2
                        ).sum(dim=1)
                        neutral_dist = (
                            (decoded_style_embeddings - neutral_prototypes) ** 2
                        ).sum(dim=1)
                        anti_row_loss = torch.relu(
                            target_dist - neutral_dist + anti_neutral_margin
                        )
                    else:
                        raise ValueError(f"Unsupported anti_neutral_mode: {anti_neutral_mode}")
                    if anti_neutral_row_weights is not None:
                        row_weights = anti_neutral_row_weights[batch_indexes].view(-1)[
                            anti_batch_mask
                        ]
                        anti_row_loss = anti_row_loss * row_weights
                    anti_neutral_loss = anti_row_loss.sum()

            loss = (
                recon_weight * recon_loss
                + kl_weight * kl_loss
                + label_weight * label_loss
                + metadata_control_weight * metadata_loss
                + current_style_teacher_weight * teacher_style_loss
                + current_decoder_prototype_weight * decoder_prototype_loss
                + current_anti_neutral_weight * anti_neutral_loss
            )
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            masses_str = ", ".join(
                f"{dataset}={dataset_masses[dataset]:.2f}" for dataset in present_datasets
            )
            print(
                f"loss: {loss.item():.2f}  "
                f"recon: {recon_loss.item():.2f} (w={recon_weight:.2f})  "
                f"kl: {kl_loss.item():.2f} (w={kl_weight:.2f})  "
                f"label: {label_loss.item():.2f} (w={label_weight:.2f})  "
                f"metadata: {metadata_loss.item():.2f} "
                f"(w={metadata_control_weight:.2f})  "
                f"teacher: {teacher_style_loss.item():.2f} (w={current_style_teacher_weight:.2f})  "
                f"decoder_proto: {decoder_prototype_loss.item():.2f} "
                f"(w={current_decoder_prototype_weight:.4f})  "
                f"anti_neutral: {anti_neutral_loss.item():.2f} "
                f"(w={current_anti_neutral_weight:.4f})  "
                f"mix: {masses_str}"
            )

    print('Ending loss:', loss.item())


def train_commonvoice_pretrain(model, embeddings, epochs=1000, lr=1e-5,
                               recon_weight=1.0, kl_weight=1.0,
                               metadata_specs=None, metadata_weight=0.0,
                               pseudo_style_targets=None,
                               pseudo_style_mask=None,
                               pseudo_style_row_weights=None,
                               pseudo_style_weight=0.0,
                               style_dims=None, free_dims=None):
    BATCH_SIZE = min(256, len(embeddings))
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters remain in the model")

    metadata_specs = metadata_specs or []
    head_params = []
    for spec in metadata_specs:
        head_params.extend(spec['head'].parameters())

    optimizer = torch.optim.Adam(trainable_params + head_params, lr=lr)

    print('CommonVoice pretraining loss weights:')
    print(f'  recon        : {recon_weight}')
    print(f'  kl           : {kl_weight}')
    print(f'  metadata     : {metadata_weight}')
    print(f'  pseudo-style : {pseudo_style_weight}')
    if style_dims is not None:
        print(f'  style dims   : {list(style_dims)}')
    if free_dims is not None:
        print(f'  free dims    : {list(free_dims)}')
    for spec in metadata_specs:
        print(
            f"  metadata target {spec['name']}: "
            f"{spec['known_count']} labeled rows across {len(spec['classes'])} classes"
        )
    if pseudo_style_mask is not None:
        print(f'  pseudo-style labeled rows: {int(pseudo_style_mask.sum().item())}')

    print(f'Training CommonVoice pretrain model for {epochs} epochs...')
    for epoch in tqdm(range(epochs)):
        with torch.no_grad():
            indexes = torch.randperm(embeddings.shape[0], device=embeddings.device)
            index_batches = torch.split(indexes, BATCH_SIZE)

        for batch_indexes in index_batches:
            embeddings_b = embeddings[batch_indexes]
            optimizer.zero_grad()

            reconstructed = model(embeddings_b)
            recon_loss = ((embeddings_b - reconstructed)**2).sum()
            kl_loss = model.kl

            metadata_loss = embeddings_b.new_tensor(0.0)
            if metadata_specs and free_dims:
                free_repr = _slice_or_none(model.last_mu, free_dims)
                for spec in metadata_specs:
                    target_indices = spec['indices'][batch_indexes]
                    mask = target_indices >= 0
                    if mask.any():
                        logits = spec['head'](free_repr[mask])
                        metadata_loss = metadata_loss + torch.nn.functional.cross_entropy(
                            logits,
                            target_indices[mask],
                            reduction='sum',
                            weight=spec.get('class_weights'),
                        )

            pseudo_style_loss = embeddings_b.new_tensor(0.0)
            if (
                pseudo_style_targets is not None
                and pseudo_style_mask is not None
                and style_dims
            ):
                batch_mask = pseudo_style_mask[batch_indexes]
                if batch_mask.any():
                    style_target = pseudo_style_targets[batch_indexes][batch_mask]
                    student_style = _slice_or_none(model.last_mu[batch_mask], style_dims)
                    row_loss = ((student_style - style_target[:, list(style_dims)]) ** 2).sum(dim=1)
                    if pseudo_style_row_weights is not None:
                        row_weights = pseudo_style_row_weights[batch_indexes][batch_mask]
                        pseudo_style_loss = (row_loss * row_weights).sum()
                    else:
                        pseudo_style_loss = row_loss.sum()

            loss = (
                recon_weight * recon_loss
                + kl_weight * kl_loss
                + metadata_weight * metadata_loss
                + pseudo_style_weight * pseudo_style_loss
            )

            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(
                f'loss: {loss.item():.2f}  '
                f'recon: {recon_loss.item():.2f} (w={recon_weight:.2f})  '
                f'kl: {kl_loss.item():.2f} (w={kl_weight:.2f})  '
                f'metadata: {metadata_loss.item():.2f} (w={metadata_weight:.2f})  '
                f'pseudo-style: {pseudo_style_loss.item():.2f} (w={pseudo_style_weight:.2f})'
            )

    print('Ending loss:', loss.item())

def extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    """Extracts a zip file if not already extracted."""
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # You can check if already extracted by verifying members
        existing = all((extract_dir / name).exists() for name in zf.namelist())
        if not existing:
            print(f"Extracting {zip_path} -> {extract_dir}")
            zf.extractall(extract_dir)

def download_file(url, dest):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

def ensure_checkpoint(checkpoint_url):
    cache_dir = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "openvoice_checkpoint"
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "checkpoints.zip"
    ckpt_path = cache_dir / "checkpoints"

    if not zip_path.exists():
        print(f"Downloading model checkpoints to {zip_path}...")
        download_file(checkpoint_url, zip_path)
        extract_zip(zip_path, ckpt_path)

    return ckpt_path
