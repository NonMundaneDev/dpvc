"""
Annotate CommonVoice embeddings with combined-VAE latent prototype labels.

This is a local alternative pseudo-label teacher for the mixed-data research
line. Instead of using an external emotion classifier, it uses the current
combined OpenVoice VAE as a teacher:

1. encode labeled CREMA-D + Expresso embeddings with the combined checkpoint
2. build one latent-space prototype per controllable style
3. encode CommonVoice embeddings and assign each row to the nearest prototype

The output artifact follows the same pseudo_style_* schema used by
annotate_commonvoice_pseudolabels.py, so the existing filter and mixed-artifact
builder can consume it without special cases.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

import dpvc
from dpvc import utils


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


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--commonvoice",
        default="embeddings/openvoice_commonvoice_cv500_emb.pt",
        help="CommonVoice embedding artifact to annotate",
    )
    ap.add_argument(
        "--combined",
        default="embeddings/openvoice_combined_emb.pt",
        help="Combined CREMA-D + Expresso embedding artifact with style labels",
    )
    ap.add_argument(
        "--checkpoint",
        default="embeddings/openvoice_vae_combined.pt",
        help="Combined VAE checkpoint used as the latent prototype teacher",
    )
    ap.add_argument(
        "--output",
        default="embeddings/openvoice_commonvoice_cv500_pseudo_prototype.pt",
        help="Output CommonVoice artifact with pseudo_style_* annotations",
    )
    ap.add_argument(
        "--latent-dims",
        type=int,
        default=15,
        help="VAE latent dimensionality (default: 15)",
    )
    ap.add_argument(
        "--style-dims",
        default="0,1,2,3,4,5,6,7,8",
        help="Comma-separated latent dims used for prototype matching",
    )
    ap.add_argument(
        "--metric",
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Prototype scoring metric (default: cosine)",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.08,
        help="Softmax temperature for converting prototype scores to confidences",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many ranked prototype labels/scores to store per row",
    )
    ap.add_argument(
        "--report-threshold",
        type=float,
        default=0.35,
        help="Confidence threshold used only for the printed/saved report",
    )
    return ap.parse_args()


def resolve_device():
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_dims(raw: str) -> list[int]:
    dims = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            dims.append(int(item))
    if not dims:
        raise ValueError("--style-dims must include at least one dimension")
    return dims


def load_vae(checkpoint, input_dim, latent_dims, device):
    model = dpvc.VariationalAutoencoder(
        input_dim=input_dim,
        latent_dims=latent_dims,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def encode_mu(model, embeddings, batch_size=256):
    chunks = []
    for start in range(0, embeddings.shape[0], batch_size):
        batch = embeddings[start:start + batch_size]
        mu, _ = model.encoder(batch)
        chunks.append(mu.cpu())
    return torch.cat(chunks, dim=0)


def build_prototypes(combined_data, combined_mu, style_dims):
    prototypes = {}
    counts = {}
    for style in UNIFIED_STYLES:
        key = f"label_{style}"
        if key not in combined_data:
            continue
        mask = combined_data[key].view(-1) > 0
        if not mask.any():
            continue
        style_latents = combined_mu[mask][:, style_dims]
        prototypes[style] = style_latents.mean(dim=0)
        counts[style] = int(mask.sum().item())
    missing = [style for style in UNIFIED_STYLES if style not in prototypes]
    if missing:
        raise ValueError(f"Could not build prototypes for styles: {missing}")
    return prototypes, counts


def score_latents(cv_mu, prototypes, metric, temperature):
    styles = list(UNIFIED_STYLES)
    proto_matrix = torch.stack([prototypes[style] for style in styles], dim=0)
    cv_style = cv_mu
    if metric == "cosine":
        logits = F.normalize(cv_style, dim=1) @ F.normalize(proto_matrix, dim=1).T
    elif metric == "euclidean":
        logits = -torch.cdist(cv_style, proto_matrix, p=2)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    probs = torch.softmax(logits / temperature, dim=1)
    return styles, logits, probs


def main():
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    style_dims = parse_dims(args.style_dims)
    device = resolve_device()

    cv_data = torch.load(args.commonvoice, weights_only=False)
    combined_data = torch.load(args.combined, weights_only=False)
    cv_embeddings = cv_data["data"].squeeze(-1).to(device)
    combined_embeddings = combined_data["data"].squeeze(-1).to(device)

    teacher = load_vae(
        args.checkpoint,
        input_dim=combined_embeddings.shape[-1],
        latent_dims=args.latent_dims,
        device=device,
    )

    print(f"Encoding combined artifact: {args.combined}")
    combined_mu = encode_mu(teacher, combined_embeddings).cpu()
    print(f"Encoding CommonVoice artifact: {args.commonvoice}")
    cv_mu_full = encode_mu(teacher, cv_embeddings).cpu()

    prototypes, prototype_counts = build_prototypes(combined_data, combined_mu, style_dims)
    styles, logits, probs = score_latents(
        cv_mu_full[:, style_dims],
        prototypes,
        args.metric,
        args.temperature,
    )

    pseudo_style = []
    pseudo_style_confidence = []
    pseudo_style_raw_label = []
    pseudo_style_topk_labels = []
    pseudo_style_topk_scores = []
    pseudo_style_score_map = []
    pseudo_style_margin = []
    counts = Counter()
    accepted_counts = Counter()
    score_totals = defaultdict(float)

    top_k = max(args.top_k, 1)
    for row_idx in range(probs.shape[0]):
        row_probs = probs[row_idx]
        ranked = torch.argsort(row_probs, descending=True).tolist()
        top_idx = ranked[0]
        second_idx = ranked[1] if len(ranked) > 1 else top_idx
        style = styles[top_idx]
        confidence = float(row_probs[top_idx].item())
        margin = float((row_probs[top_idx] - row_probs[second_idx]).item())
        top_indexes = ranked[:top_k]
        score_map = {
            styles[idx]: float(row_probs[idx].item())
            for idx in range(len(styles))
        }

        pseudo_style.append(style)
        pseudo_style_confidence.append(confidence)
        pseudo_style_raw_label.append(style)
        pseudo_style_topk_labels.append([styles[idx] for idx in top_indexes])
        pseudo_style_topk_scores.append([float(row_probs[idx].item()) for idx in top_indexes])
        pseudo_style_score_map.append(score_map)
        pseudo_style_margin.append(margin)
        counts[style] += 1
        if confidence >= args.report_threshold:
            accepted_counts[style] += 1
        for mapped_style, score in score_map.items():
            score_totals[mapped_style] += float(score)

    metadata_report = cv_data.get("metadata_report")
    if metadata_report is None:
        metadata_report = utils.build_commonvoice_metadata_report(
            cv_data.get("age", []),
            cv_data.get("gender", []),
            cv_data.get("accent", []),
        )

    report = {
        "model": "combined_vae_latent_prototype",
        "teacher_checkpoint": args.checkpoint,
        "combined_artifact": args.combined,
        "metric": args.metric,
        "temperature": float(args.temperature),
        "style_dims": style_dims,
        "top_k": int(top_k),
        "report_threshold": float(args.report_threshold),
        "rows_annotated": int(probs.shape[0]),
        "prototype_counts": prototype_counts,
        "mapped_style_counts": dict(counts),
        "accepted_style_counts": dict(accepted_counts),
        "mapped_style_score_totals": {
            style: float(score)
            for style, score in sorted(score_totals.items())
        },
    }

    enriched = dict(cv_data)
    enriched["pseudo_style"] = pseudo_style
    enriched["pseudo_style_confidence"] = pseudo_style_confidence
    enriched["pseudo_style_raw_label"] = pseudo_style_raw_label
    enriched["pseudo_style_topk_labels"] = pseudo_style_topk_labels
    enriched["pseudo_style_topk_scores"] = pseudo_style_topk_scores
    enriched["pseudo_style_score_map"] = pseudo_style_score_map
    enriched["pseudo_style_margin"] = pseudo_style_margin
    enriched["pseudo_style_source"] = "combined_vae_latent_prototype"
    enriched["pseudo_style_teacher"] = {
        "model": "combined_vae_latent_prototype",
        "teacher_checkpoint": args.checkpoint,
        "combined_artifact": args.combined,
        "metric": args.metric,
        "temperature": float(args.temperature),
        "style_dims": style_dims,
    }
    enriched["pseudo_style_report"] = report
    enriched["metadata_report"] = metadata_report

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(enriched, output_path)

    print(f"Saved prototype pseudo-labeled CommonVoice artifact to {output_path}")
    print(f"Rows annotated: {probs.shape[0]}")
    print("Accepted pseudo-style counts at report threshold:")
    for style, count in sorted(accepted_counts.items()):
        print(f"  {style:11s}: {count}")


if __name__ == "__main__":
    main()
