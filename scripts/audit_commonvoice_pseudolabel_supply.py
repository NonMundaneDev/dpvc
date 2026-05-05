"""
Audit CommonVoice pseudo-label supply in scored, filtered, and mixed artifacts.

This report is meant to prevent blind threshold or weighting experiments. It
answers how many candidate rows exist for each style, how many survive artifact
filters, and how many are actually selected into a mixed-data training set.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import torch


STYLES = [
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
THRESHOLDS = [0.35, 0.50, 0.60, 0.75, 0.90]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--artifacts",
        nargs="+",
        default=[
            "embeddings/openvoice_commonvoice_cv500_pseudo_scored.pt",
            "embeddings/openvoice_commonvoice_cv500_pseudo_hybrid_extra_priority.pt",
            "embeddings/openvoice_mixed_teacher_hybrid_extra_base.pt",
        ],
        help="Artifacts to audit",
    )
    ap.add_argument(
        "--out-csv",
        default="results/commonvoice_pseudolabel_supply_audit.csv",
        help="Output CSV path",
    )
    ap.add_argument(
        "--out-md",
        default="results/commonvoice_pseudolabel_supply_audit.md",
        help="Output Markdown path",
    )
    return ap.parse_args()


def speaker_count(data):
    if "speaker_ids" not in data:
        return 0
    return len(set(str(value) for value in data["speaker_ids"]))


def scalar(value):
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


def score_threshold_counts(data):
    rows = []
    score_maps = data.get("pseudo_style_score_map")
    if not score_maps:
        return rows
    for style in STYLES:
        for threshold in THRESHOLDS:
            count = 0
            for score_map in score_maps:
                if not isinstance(score_map, dict):
                    continue
                score = scalar(score_map.get(style))
                if score is not None and score >= threshold:
                    count += 1
            rows.append({
                "style": style,
                "threshold": threshold,
                "score_count": count,
            })
    return rows


def audit_commonvoice_artifact(path, data):
    rows = []
    pseudo_counts = Counter(style for style in data.get("pseudo_style", []) if style in STYLES)
    selected_style = data.get("pseudo_style_selected", [])
    selected_mask = data.get("pseudo_style_selected_mask", [])
    selected_counts = Counter()
    if selected_style and selected_mask is not None:
        for style, selected in zip(selected_style, selected_mask):
            if bool(selected) and style in STYLES:
                selected_counts[style] += 1
    threshold_rows = score_threshold_counts(data)
    threshold_by_style = {
        (row["style"], row["threshold"]): row["score_count"]
        for row in threshold_rows
    }
    for style in STYLES:
        row = {
            "artifact": path,
            "artifact_type": "commonvoice",
            "rows_total": len(data.get("data", [])),
            "speaker_count": speaker_count(data),
            "style": style,
            "raw_pseudo_count": pseudo_counts.get(style, 0),
            "artifact_selected_count": selected_counts.get(style, 0),
            "mixed_commonvoice_count": "",
            "mixed_total_style_count": "",
        }
        for threshold in THRESHOLDS:
            row[f"score_ge_{threshold:.2f}"] = threshold_by_style.get((style, threshold), "")
        rows.append(row)
    return rows


def audit_mixed_artifact(path, data):
    rows = []
    source_dataset = list(data.get("source_dataset", []))
    style_sources = list(data.get("style_label_source", []))
    labels = {style: data.get(f"label_{style}") for style in STYLES}
    cv_counts = Counter()
    total_counts = Counter()
    for idx, dataset in enumerate(source_dataset):
        active_style = None
        for style in STYLES:
            values = labels.get(style)
            if values is None:
                continue
            value = values[idx]
            if torch.is_tensor(value):
                is_active = float(value.reshape(-1)[0].item()) > 0
            else:
                is_active = float(value) > 0
            if is_active:
                active_style = style
                break
        if active_style is None:
            continue
        total_counts[active_style] += 1
        if dataset == "CommonVoice" and idx < len(style_sources) and style_sources[idx] == "pseudo":
            cv_counts[active_style] += 1

    for style in STYLES:
        rows.append({
            "artifact": path,
            "artifact_type": "mixed",
            "rows_total": len(data.get("data", [])),
            "speaker_count": speaker_count(data),
            "style": style,
            "raw_pseudo_count": "",
            "artifact_selected_count": "",
            "mixed_commonvoice_count": cv_counts.get(style, 0),
            "mixed_total_style_count": total_counts.get(style, 0),
            **{f"score_ge_{threshold:.2f}": "" for threshold in THRESHOLDS},
        })
    return rows


def audit_artifact(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if "source_dataset" in data:
        return audit_mixed_artifact(path, data)
    return audit_commonvoice_artifact(path, data)


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "artifact",
        "artifact_type",
        "rows_total",
        "speaker_count",
        "style",
        "raw_pseudo_count",
        "artifact_selected_count",
        "mixed_commonvoice_count",
        "mixed_total_style_count",
        *[f"score_ge_{threshold:.2f}" for threshold in THRESHOLDS],
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_md(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for row in rows:
        artifact = row["artifact"]
        if artifact not in artifacts:
            artifacts.append(artifact)

    lines = [
        "# CommonVoice Pseudo-Label Supply Audit",
        "",
        "This report tracks rare-style supply before running more mixed-data weighting or curriculum experiments.",
        "",
    ]
    for artifact in artifacts:
        subset = [row for row in rows if row["artifact"] == artifact]
        if not subset:
            continue
        first = subset[0]
        lines.extend([
            f"## `{artifact}`",
            "",
            f"- type: `{first['artifact_type']}`",
            f"- rows: `{first['rows_total']}`",
            f"- speakers: `{first['speaker_count']}`",
            "",
            "| style | raw pseudo | selected | mixed CV | mixed total | score>=0.60 | score>=0.90 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in subset:
            lines.append(
                f"| `{row['style']}` | {row['raw_pseudo_count'] or 0} | "
                f"{row['artifact_selected_count'] or 0} | {row['mixed_commonvoice_count'] or 0} | "
                f"{row['mixed_total_style_count'] or 0} | {row['score_ge_0.60'] or 0} | "
                f"{row['score_ge_0.90'] or 0} |"
            )
        lines.append("")

    rare_notes = []
    for row in rows:
        if row["style"] in {"anger", "fear"}:
            selected = row.get("artifact_selected_count") or row.get("mixed_commonvoice_count") or 0
            try:
                selected = int(selected)
            except (TypeError, ValueError):
                selected = 0
            if selected and selected < 10:
                rare_notes.append(f"`{Path(row['artifact']).name}` has `{row['style']}={selected}` selected rows.")
    lines.extend([
        "## Readout",
        "",
    ])
    if rare_notes:
        lines.extend(f"- {note}" for note in rare_notes)
    lines.extend([
        "- If rare canonical styles remain below roughly tens of rows, prefer extracting/scoring more CommonVoice data over another loss-weight tweak.",
        "- If raw score counts are high but selected counts are low, adjust filters; if raw counts are low, the local corpus is the bottleneck.",
        "",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    rows = []
    for artifact in args.artifacts:
        rows.extend(audit_artifact(artifact))
    write_csv(args.out_csv, rows)
    write_md(args.out_md, rows)
    print(f"Wrote audit CSV to {args.out_csv}")
    print(f"Wrote audit report to {args.out_md}")


if __name__ == "__main__":
    main()
