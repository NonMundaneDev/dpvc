"""
Build per-style diagnostics for mixed-data style-teacher experiments.

The report joins three views that are easy to inspect separately but hard to
reason about together:

1. label supply in the mixed training artifact,
2. teacher-vs-student latent geometry on the same artifact rows,
3. generated-output metrics and collapse flags for one evaluated condition.

Example:

    python scripts/analyze_mixed_teacher_style_diagnostics.py \
        --mixed-artifact embeddings/openvoice_mixed_teacher_hybrid_extra_base.pt \
        --teacher-checkpoint embeddings/openvoice_vae_combined.pt \
        --student-checkpoint embeddings/openvoice_vae_mixed_teacher_hybrid_style_distill_targetmask_balanced.pt \
        --condition mixed_teacher_hybrid_style_distill_targetmask_balanced \
        --out-csv results/eval_mixed_teacher_style_diagnostics_targetmask.csv \
        --out-md results/eval_mixed_teacher_style_diagnostics_targetmask.md
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import torch

import dpvc


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mixed-artifact",
        default="embeddings/openvoice_mixed_teacher_hybrid_extra_base.pt",
        help="Mixed-data training artifact to diagnose",
    )
    ap.add_argument(
        "--teacher-checkpoint",
        default="embeddings/openvoice_vae_combined.pt",
        help="Frozen style-teacher VAE checkpoint",
    )
    ap.add_argument(
        "--student-checkpoint",
        default="embeddings/openvoice_vae_mixed_teacher_hybrid_style_distill_targetmask_balanced.pt",
        help="Trained student checkpoint to compare against the teacher",
    )
    ap.add_argument(
        "--condition",
        default="mixed_teacher_hybrid_style_distill_targetmask_balanced",
        help="Evaluated condition name used in metric CSV filenames",
    )
    ap.add_argument(
        "--input-tag",
        default="mixed_teacher",
        help="Metric file tag, e.g. mixed_teacher",
    )
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--collapse-csv", default="results/eval_mixed_teacher_collapse.csv")
    ap.add_argument("--style-dims", default="0-8")
    ap.add_argument("--latent-dims", type=int, default=15)
    ap.add_argument("--teacher-datasets", default="CommonVoice")
    ap.add_argument(
        "--out-csv",
        default="results/eval_mixed_teacher_style_diagnostics_targetmask.csv",
    )
    ap.add_argument(
        "--out-md",
        default="results/eval_mixed_teacher_style_diagnostics_targetmask.md",
    )
    return ap.parse_args()


def parse_dim_list(raw):
    dims = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            dims.extend(range(int(start), int(end) + 1))
        else:
            dims.append(int(item))
    return sorted(set(dims))


def parse_datasets(raw):
    raw = (raw or "").strip()
    if not raw or raw.lower() == "all":
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def mean(values):
    values = [value for value in values if value is not None and not math.isnan(value)]
    return statistics.mean(values) if values else None


def fmt(value, digits=4):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric_path(results_dir, metric, tag, condition):
    return Path(results_dir) / f"eval_{metric}_{tag}_{condition}.csv"


def load_vae(checkpoint, input_dim, latent_dims, device):
    model = dpvc.VariationalAutoencoder(input_dim=input_dim, latent_dims=latent_dims).to(device)
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


def style_targets_from_artifact(data, styles):
    return torch.cat([data[f"label_{style}"] for style in styles], dim=1)


def style_index_map(style_targets):
    return torch.argmax(style_targets, dim=1).tolist()


def source_counts_for_indexes(sources, indexes):
    counter = Counter()
    for idx in indexes:
        counter[str(sources[idx])] += 1
    return counter


def row_source_counts(label_sources, indexes):
    counter = Counter()
    for idx in indexes:
        source = label_sources[idx] if idx < len(label_sources) else "unknown"
        counter[str(source)] += 1
    return counter


def tensor_values(tensor, indexes):
    if tensor is None:
        return []
    flat = tensor.view(-1).cpu()
    return [float(flat[idx]) for idx in indexes]


def top1_rate(mu, indexes, style_dims, target_dim):
    if not indexes:
        return None
    vals = mu[indexes][:, style_dims]
    winners = vals.argmax(dim=1)
    local_target = style_dims.index(target_dim)
    return float((winners == local_target).float().mean().item())


def margin_mean(mu, indexes, style_dims, target_dim):
    if not indexes:
        return None
    vals = mu[indexes][:, style_dims]
    local_target = style_dims.index(target_dim)
    target_vals = vals[:, local_target]
    if vals.shape[1] == 1:
        return None
    others = torch.cat([vals[:, :local_target], vals[:, local_target + 1:]], dim=1)
    margins = target_vals - others.max(dim=1).values
    return float(margins.mean().item())


def latent_stats(teacher_mu, student_mu, indexes, target_dim, style_dims):
    if not indexes:
        return {
            "teacher_target_mean": None,
            "student_target_mean": None,
            "signed_gap_mean": None,
            "abs_gap_mean": None,
            "teacher_target_top1_rate": None,
            "student_target_top1_rate": None,
            "teacher_target_margin_mean": None,
            "student_target_margin_mean": None,
        }
    teacher_vals = teacher_mu[indexes, target_dim]
    student_vals = student_mu[indexes, target_dim]
    gap = student_vals - teacher_vals
    return {
        "teacher_target_mean": float(teacher_vals.mean().item()),
        "student_target_mean": float(student_vals.mean().item()),
        "signed_gap_mean": float(gap.mean().item()),
        "abs_gap_mean": float(gap.abs().mean().item()),
        "teacher_target_top1_rate": top1_rate(teacher_mu, indexes, style_dims, target_dim),
        "student_target_top1_rate": top1_rate(student_mu, indexes, style_dims, target_dim),
        "teacher_target_margin_mean": margin_mean(teacher_mu, indexes, style_dims, target_dim),
        "student_target_margin_mean": margin_mean(student_mu, indexes, style_dims, target_dim),
    }


def generated_metrics(results_dir, input_tag, condition, collapse_csv):
    by_style = defaultdict(lambda: defaultdict(list))
    pred_counts = defaultdict(Counter)
    recall_counts = defaultdict(lambda: {"matches": 0, "scored": 0})

    for row in read_csv(metric_path(results_dir, "emotion", input_tag, condition)):
        style = row["style"]
        if style == "baseline":
            continue
        pred_counts[style][row["predicted"]] += 1
        if row.get("match") in {"0", "1"}:
            recall_counts[style]["scored"] += 1
            recall_counts[style]["matches"] += int(row["match"])
        if row.get("emo_sim"):
            by_style[style]["emo_sim"].append(float(row["emo_sim"]))
        if row.get("score"):
            by_style[style]["emotion_score"].append(float(row["score"]))

    for row in read_csv(metric_path(results_dir, "novelty", input_tag, condition)):
        style = row["style"]
        if style == "baseline":
            continue
        if row.get("novelty_gain_vs_baseline"):
            by_style[style]["novelty_gain"].append(float(row["novelty_gain_vs_baseline"]))
        if row.get("similarity"):
            by_style[style]["speaker_similarity"].append(float(row["similarity"]))

    for row in read_csv(metric_path(results_dir, "wer", input_tag, condition)):
        style = row["style"]
        if style == "baseline":
            continue
        if row.get("wer") and row.get("ref_source") != "self":
            by_style[style]["wer"].append(float(row["wer"]))

    for row in read_csv(metric_path(results_dir, "mos", input_tag, condition)):
        style = row["style"]
        if style == "baseline":
            continue
        if row.get("mos"):
            by_style[style]["mos"].append(float(row["mos"]))
        if row.get("delta_vs_baseline"):
            by_style[style]["mos_delta"].append(float(row["delta_vs_baseline"]))

    collapse_counts = defaultdict(Counter)
    for row in read_csv(collapse_csv):
        if row.get("condition") != condition:
            continue
        style = row["style"]
        axes = 0
        for key in ("content_collapse", "style_collapse_to_neutral", "identity_collapse_to_baseline", "mixed_collapse"):
            value = int(row.get(key) or 0)
            collapse_counts[style][key] += value
            if key != "mixed_collapse":
                axes += value
        if axes > 0:
            collapse_counts[style]["files_with_any_collapse"] += 1
        collapse_counts[style]["collapse_rows"] += 1

    output = {}
    styles = set(by_style) | set(pred_counts) | set(recall_counts) | set(collapse_counts)
    for style in styles:
        scored = recall_counts[style]["scored"]
        matches = recall_counts[style]["matches"]
        total_preds = sum(pred_counts[style].values())
        neutral_preds = pred_counts[style].get("neutral", 0)
        top_pred, top_count = ("", 0)
        if pred_counts[style]:
            top_pred, top_count = pred_counts[style].most_common(1)[0]
        output[style] = {
            "emotion_matches": matches,
            "emotion_scored": scored,
            "emotion_recall": matches / scored if scored else None,
            "neutral_prediction_rate": neutral_preds / total_preds if total_preds else None,
            "predicted_top": top_pred,
            "predicted_top_count": top_count,
            "mean_emo_sim": mean(by_style[style]["emo_sim"]),
            "mean_emotion_score": mean(by_style[style]["emotion_score"]),
            "mean_novelty_gain": mean(by_style[style]["novelty_gain"]),
            "mean_speaker_similarity": mean(by_style[style]["speaker_similarity"]),
            "mean_wer": mean(by_style[style]["wer"]),
            "mean_mos": mean(by_style[style]["mos"]),
            "mean_mos_delta": mean(by_style[style]["mos_delta"]),
            "content_collapse_count": collapse_counts[style]["content_collapse"],
            "style_collapse_count": collapse_counts[style]["style_collapse_to_neutral"],
            "identity_collapse_count": collapse_counts[style]["identity_collapse_to_baseline"],
            "mixed_collapse_count": collapse_counts[style]["mixed_collapse"],
            "files_with_any_collapse": collapse_counts[style]["files_with_any_collapse"],
            "generated_rows": total_preds,
        }
    return output


def write_csv(path, rows):
    fieldnames = list(rows[0].keys()) if rows else []
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_markdown(path, condition, rows):
    rare = [row for row in rows if int(row["active_teacher_rows"] or 0) < 10]
    neutral_stuck = [
        row
        for row in rows
        if row["emotion_scored"]
        and row["emotion_recall"] == "0.0000"
        and row["neutral_prediction_rate"]
        and float(row["neutral_prediction_rate"]) >= 0.80
    ]
    teacher_misaligned = [
        row
        for row in rows
        if row["teacher_target_top1_rate"]
        and float(row["teacher_target_top1_rate"]) < 0.50
    ]
    student_not_matching = [
        row
        for row in rows
        if row["teacher_target_top1_rate"]
        and row["student_target_top1_rate"]
        and float(row["teacher_target_top1_rate"]) - float(row["student_target_top1_rate"]) >= 0.25
    ]

    compact_rows = []
    for row in rows:
        compact_rows.append([
            row["style"],
            row["active_teacher_rows"],
            row["pseudo_labeled_rows"],
            row["teacher_target_top1_rate"],
            row["student_target_top1_rate"],
            row["emotion_recall"],
            row["neutral_prediction_rate"],
            row["mean_novelty_gain"],
            row["mean_wer"],
            row["mean_mos_delta"],
            row["files_with_any_collapse"],
        ])

    lines = [
        f"# Mixed-Teacher Style Diagnostics: `{condition}`",
        "",
        "This report joins training-artifact label supply, teacher/student latent geometry, and generated-output metrics by style.",
        "",
        "## Diagnostic Table",
        "",
        markdown_table(
            [
                "style",
                "teacher rows",
                "pseudo rows",
                "teacher top1",
                "student top1",
                "recall",
                "neutral pred",
                "novelty",
                "WER",
                "MOS delta",
                "any collapse",
            ],
            compact_rows,
        ),
        "",
        "## Automatic Readout",
        "",
    ]

    if rare:
        rare_text = ", ".join(f"`{row['style']}={row['active_teacher_rows']}`" for row in rare)
        lines.append(f"- Rare active teacher supply: {rare_text}. Weighting these rows cannot substitute for more examples.")
    else:
        lines.append("- Active teacher supply is not extremely sparse for any style under the current threshold.")

    if neutral_stuck:
        stuck_text = ", ".join(f"`{row['style']}`" for row in neutral_stuck)
        lines.append(f"- Emotion-scored styles still collapsing to neutral: {stuck_text}.")
    else:
        lines.append("- No emotion-scored style has both zero recall and at least 80% neutral predictions.")

    if teacher_misaligned:
        teacher_text = ", ".join(
            f"`{row['style']}` teacher-top1={row['teacher_target_top1_rate']}"
            for row in teacher_misaligned
        )
        lines.append(f"- Teacher latent target is often not the top style dim for: {teacher_text}.")
    else:
        lines.append("- Teacher latent target dims are top-ranked for at least half of active rows in every style with active teacher rows.")

    if student_not_matching:
        student_text = ", ".join(
            f"`{row['style']}` teacher={row['teacher_target_top1_rate']} student={row['student_target_top1_rate']}"
            for row in student_not_matching
        )
        lines.append(f"- Student target-dim alignment trails the teacher strongly for: {student_text}.")
    else:
        lines.append("- Student target-dim alignment is not dramatically below teacher alignment by this coarse top1 diagnostic.")

    lines.extend([
        "",
        "## Recommendation",
        "",
        "- Do not spend the next turn on another latent-only scalar/mask variant without first addressing the diagnostic failure mode above.",
        "- Prefer a labeled-first curriculum if student latent alignment trails the teacher, or a decoder-aware/generated-audio style objective if latent alignment exists but emotion recall remains neutral.",
        "- Treat rare canonical classes as a data-supply issue, not merely a weighting issue.",
        "",
    ])

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    device = "cpu"
    data = torch.load(args.mixed_artifact, map_location=device, weights_only=False)
    embeddings = data["data"].to(device)
    styles = list(data.get("supported_styles") or [])
    if not styles:
        raise ValueError("Mixed artifact is missing supported_styles")

    style_dims = parse_dim_list(args.style_dims)
    if len(style_dims) < len(styles):
        raise ValueError(
            f"--style-dims must include at least {len(styles)} dims for {len(styles)} styles"
        )
    dim_by_style = {style: style_dims[idx] for idx, style in enumerate(styles)}
    selected_datasets = parse_datasets(args.teacher_datasets)

    style_targets = style_targets_from_artifact(data, styles)
    target_indices = style_index_map(style_targets)
    label_mask = data["style_label_mask"].view(-1) > 0
    label_confidence = data.get("style_label_confidence")
    row_weights = data.get("style_label_row_weight")
    sources = [str(source) for source in data.get("source_dataset", [])]
    label_sources = [str(source) for source in data.get("style_label_source", [])]

    teacher = load_vae(args.teacher_checkpoint, embeddings.shape[1], args.latent_dims, device)
    student = load_vae(args.student_checkpoint, embeddings.shape[1], args.latent_dims, device)
    teacher_mu = encode_mu(teacher, embeddings)
    student_mu = encode_mu(student, embeddings)

    generated = generated_metrics(
        args.results_dir,
        args.input_tag,
        args.condition,
        args.collapse_csv,
    )

    rows = []
    for style_idx, style in enumerate(styles):
        target_dim = dim_by_style[style]
        labeled_indexes = [
            idx for idx, is_labeled in enumerate(label_mask.tolist())
            if is_labeled and target_indices[idx] == style_idx
        ]
        active_indexes = [
            idx
            for idx in labeled_indexes
            if selected_datasets is None or sources[idx] in selected_datasets
        ]
        source_counts = source_counts_for_indexes(sources, labeled_indexes)
        label_source_counts = row_source_counts(label_sources, labeled_indexes)
        confidence_values = tensor_values(label_confidence, active_indexes)
        row_weight_values = tensor_values(row_weights, active_indexes)
        lstats = latent_stats(teacher_mu, student_mu, active_indexes, target_dim, style_dims)
        gstats = generated.get(style, {})

        row = {
            "condition": args.condition,
            "style": style,
            "target_dim": target_dim,
            "labeled_rows": len(labeled_indexes),
            "active_teacher_rows": len(active_indexes),
            "commonvoice_labeled_rows": source_counts.get("CommonVoice", 0),
            "cremad_labeled_rows": source_counts.get("CREMA-D", 0),
            "expresso_labeled_rows": source_counts.get("Expresso", 0),
            "true_labeled_rows": label_source_counts.get("true", 0),
            "pseudo_labeled_rows": label_source_counts.get("pseudo", 0),
            "mean_confidence": fmt(mean(confidence_values)),
            "min_confidence": fmt(min(confidence_values) if confidence_values else None),
            "max_confidence": fmt(max(confidence_values) if confidence_values else None),
            "mean_row_weight": fmt(mean(row_weight_values)),
            "teacher_target_mean": fmt(lstats["teacher_target_mean"]),
            "student_target_mean": fmt(lstats["student_target_mean"]),
            "signed_gap_mean": fmt(lstats["signed_gap_mean"]),
            "abs_gap_mean": fmt(lstats["abs_gap_mean"]),
            "teacher_target_top1_rate": fmt(lstats["teacher_target_top1_rate"]),
            "student_target_top1_rate": fmt(lstats["student_target_top1_rate"]),
            "teacher_target_margin_mean": fmt(lstats["teacher_target_margin_mean"]),
            "student_target_margin_mean": fmt(lstats["student_target_margin_mean"]),
            "generated_rows": gstats.get("generated_rows", 0),
            "emotion_matches": gstats.get("emotion_matches", 0),
            "emotion_scored": gstats.get("emotion_scored", 0),
            "emotion_recall": fmt(gstats.get("emotion_recall")),
            "neutral_prediction_rate": fmt(gstats.get("neutral_prediction_rate")),
            "predicted_top": gstats.get("predicted_top", ""),
            "predicted_top_count": gstats.get("predicted_top_count", 0),
            "mean_emo_sim": fmt(gstats.get("mean_emo_sim")),
            "mean_emotion_score": fmt(gstats.get("mean_emotion_score")),
            "mean_novelty_gain": fmt(gstats.get("mean_novelty_gain")),
            "mean_speaker_similarity": fmt(gstats.get("mean_speaker_similarity")),
            "mean_wer": fmt(gstats.get("mean_wer")),
            "mean_mos": fmt(gstats.get("mean_mos")),
            "mean_mos_delta": fmt(gstats.get("mean_mos_delta")),
            "content_collapse_count": gstats.get("content_collapse_count", 0),
            "style_collapse_count": gstats.get("style_collapse_count", 0),
            "identity_collapse_count": gstats.get("identity_collapse_count", 0),
            "mixed_collapse_count": gstats.get("mixed_collapse_count", 0),
            "files_with_any_collapse": gstats.get("files_with_any_collapse", 0),
        }
        rows.append(row)

    write_csv(args.out_csv, rows)
    write_markdown(args.out_md, args.condition, rows)
    print(f"Wrote style diagnostics CSV to {args.out_csv}")
    print(f"Wrote style diagnostics report to {args.out_md}")


if __name__ == "__main__":
    main()
