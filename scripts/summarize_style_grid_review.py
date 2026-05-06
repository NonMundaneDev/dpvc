#!/usr/bin/env python3
"""
Summarize an A/B style-strength listening review.

This script creates an objective-assisted triage sheet for the A/B dashboard
and, once ratings are filled, summarizes subjective preferences.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from build_style_grid_review import (
    DEFAULT_CANDIDATES,
    build_pairs,
    parse_candidate_spec,
    read_manifest,
)


RATING_FIELDS = [
    "reference_target_match_1_5",
    "candidate_target_match_1_5",
    "reference_intelligibility_1_5",
    "candidate_intelligibility_1_5",
    "reference_naturalness_1_5",
    "candidate_naturalness_1_5",
    "reference_identity_shift_1_5",
    "candidate_identity_shift_1_5",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--reference-manifest",
        default=(
            "output/mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard_eval/"
            "generation_manifest.jsonl"
        ),
    )
    ap.add_argument(
        "--reference-tag",
        default="mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard",
    )
    ap.add_argument("--reference-input-tag", default="mixed_teacher")
    ap.add_argument("--candidate", action="append", default=None)
    ap.add_argument("--candidate-input-tag", default="mixed_teacher_strength_grid")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument(
        "--ratings",
        default="results/listening_mixed_teacher_cvrare_strength_grid_ab_review_ratings.csv",
    )
    ap.add_argument(
        "--out-csv",
        default="results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority.csv",
    )
    ap.add_argument(
        "--out-md",
        default="results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority.md",
    )
    return ap.parse_args()


def to_float(value, default=None):
    if value in ("", None):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def to_int(value, default=0):
    if value in ("", None):
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def metric_value(pair, side, key, default=None):
    return to_float(pair[f"{side}_metrics"].get(key), default)


def classify_pair(pair):
    ref_match = to_int(pair["reference_metrics"].get("match"))
    cand_match = to_int(pair["candidate_metrics"].get("match"))
    ref_wer = metric_value(pair, "reference", "wer", 0.0)
    cand_wer = metric_value(pair, "candidate", "wer", 0.0)
    ref_mos = metric_value(pair, "reference", "mos_delta", 0.0)
    cand_mos = metric_value(pair, "candidate", "mos_delta", 0.0)
    ref_novelty = metric_value(pair, "reference", "novelty", 0.0)
    cand_novelty = metric_value(pair, "candidate", "novelty", 0.0)

    match_delta = cand_match - ref_match
    wer_delta = cand_wer - ref_wer
    mos_delta = cand_mos - ref_mos
    novelty_delta = cand_novelty - ref_novelty

    quality_risk = cand_wer >= 0.50 or cand_mos <= -0.75
    severe_risk = cand_wer >= 0.80 or cand_mos <= -1.00

    if match_delta > 0 and not quality_risk:
        bucket = "clean_target_gain"
        priority = 1
        recommendation = "listen first; candidate may be preset-worthy if it sounds natural"
    elif match_delta > 0:
        bucket = "target_gain_quality_risk"
        priority = 2
        recommendation = "listen early; candidate gains target label but quality/content may block preset"
    elif match_delta == 0 and novelty_delta > 0.03 and not quality_risk:
        bucket = "novelty_gain_no_recall_gain"
        priority = 3
        recommendation = "listen if novelty matters; do not treat as recall improvement"
    elif cand_match < ref_match:
        bucket = "candidate_regression"
        priority = 4
        recommendation = "likely reject unless perceptual target strength contradicts metrics"
    elif severe_risk or (wer_delta > 0.15 or mos_delta < -0.25):
        bucket = "metric_trap"
        priority = 4
        recommendation = "likely reject; objective metrics show no recall gain and worse quality"
    else:
        bucket = "tie_or_minor_change"
        priority = 3
        recommendation = "lower priority; use for sanity check"

    return {
        "bucket": bucket,
        "priority": priority,
        "recommendation": recommendation,
        "reference_match": ref_match,
        "candidate_match": cand_match,
        "match_delta": match_delta,
        "reference_wer": ref_wer,
        "candidate_wer": cand_wer,
        "wer_delta": wer_delta,
        "reference_mos_delta": ref_mos,
        "candidate_mos_delta": cand_mos,
        "mos_delta_delta": mos_delta,
        "reference_novelty": ref_novelty,
        "candidate_novelty": cand_novelty,
        "novelty_delta": novelty_delta,
        "quality_risk": int(quality_risk),
        "severe_risk": int(severe_risk),
    }


def format_float(value):
    if value is None:
        return ""
    return f"{value:.4f}"


def build_priority_rows(pairs):
    rows = []
    for pair in pairs:
        labels = classify_pair(pair)
        row = {
            "priority": labels["priority"],
            "bucket": labels["bucket"],
            "style": pair["style"],
            "source_stem": pair["source"],
            "recommendation": labels["recommendation"],
            "reference_tag": pair["reference_tag"],
            "candidate_tag": pair["candidate_tag"],
            "reference_audio": pair["reference_file"],
            "candidate_audio": pair["candidate_file"],
        }
        for key, value in labels.items():
            if key in {"bucket", "priority", "recommendation"}:
                continue
            row[key] = format_float(value) if isinstance(value, float) else value
        rows.append(row)
    return sorted(rows, key=lambda row: (int(row["priority"]), row["style"], row["source_stem"]))


def write_csv(path, rows):
    fieldnames = [
        "priority",
        "bucket",
        "style",
        "source_stem",
        "recommendation",
        "reference_match",
        "candidate_match",
        "match_delta",
        "reference_wer",
        "candidate_wer",
        "wer_delta",
        "reference_mos_delta",
        "candidate_mos_delta",
        "mos_delta_delta",
        "reference_novelty",
        "candidate_novelty",
        "novelty_delta",
        "quality_risk",
        "severe_risk",
        "reference_tag",
        "candidate_tag",
        "reference_audio",
        "candidate_audio",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_ratings(path):
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_ratings(ratings):
    filled = [
        row for row in ratings if row.get("prefer_reference_or_candidate", "").strip()
    ]
    preferences = Counter(
        row.get("prefer_reference_or_candidate", "").strip().lower()
        for row in filled
    )
    by_style = defaultdict(Counter)
    for row in filled:
        by_style[row.get("style", "")][
            row.get("prefer_reference_or_candidate", "").strip().lower()
        ] += 1

    numeric_counts = Counter()
    numeric_sums = Counter()
    for row in filled:
        for field in RATING_FIELDS:
            value = to_float(row.get(field))
            if value is None:
                continue
            numeric_counts[field] += 1
            numeric_sums[field] += value

    averages = {
        field: numeric_sums[field] / numeric_counts[field]
        for field in numeric_counts
    }
    return filled, preferences, by_style, averages


def write_md(path, rows, ratings_path):
    counts = Counter(row["bucket"] for row in rows)
    by_style = defaultdict(Counter)
    for row in rows:
        by_style[row["style"]][row["bucket"]] += 1

    ratings = read_ratings(ratings_path)
    filled, preferences, style_preferences, averages = summarize_ratings(ratings)

    lines = [
        "# Style-Strength A/B Review Priority",
        "",
        "This is an objective-assisted triage sheet for the listening dashboard.",
        "It is not a substitute for perceptual review.",
        "",
        f"Priority rows: `{len(rows)}`",
        f"Ratings file: `{ratings_path}`",
        "",
        "## Bucket Counts",
        "",
        "| Bucket | Count |",
        "|--------|-------|",
    ]
    for bucket, count in sorted(counts.items()):
        lines.append(f"| `{bucket}` | `{count}` |")

    lines.extend(["", "## Counts By Style", ""])
    for style in sorted(by_style):
        lines.append(f"### `{style}`")
        lines.append("")
        lines.append("| Bucket | Count |")
        lines.append("|--------|-------|")
        for bucket, count in sorted(by_style[style].items()):
            lines.append(f"| `{bucket}` | `{count}` |")
        lines.append("")

    lines.extend([
        "## Listen First",
        "",
        "| Priority | Style | Source | Bucket | Ref match | Cand match | Cand WER | Cand MOS delta | Recommendation |",
        "|----------|-------|--------|--------|-----------|------------|----------|----------------|----------------|",
    ])
    for row in rows:
        if int(row["priority"]) > 2:
            continue
        lines.append(
            "| `{priority}` | `{style}` | `{source}` | `{bucket}` | `{ref_match}` | `{cand_match}` | `{cand_wer}` | `{cand_mos}` | {rec} |".format(
                priority=row["priority"],
                style=row["style"],
                source=row["source_stem"],
                bucket=row["bucket"],
                ref_match=row["reference_match"],
                cand_match=row["candidate_match"],
                cand_wer=row["candidate_wer"],
                cand_mos=row["candidate_mos_delta"],
                rec=row["recommendation"],
            )
        )

    traps = [row for row in rows if row["bucket"] in {"metric_trap", "candidate_regression"}]
    lines.extend([
        "",
        "## Likely Reject / Metric-Trap Rows",
        "",
        "| Style | Source | Bucket | Ref match | Cand match | WER delta | MOS delta delta |",
        "|-------|--------|--------|-----------|------------|-----------|-----------------|",
    ])
    for row in traps:
        lines.append(
            f"| `{row['style']}` | `{row['source_stem']}` | `{row['bucket']}` | `{row['reference_match']}` | `{row['candidate_match']}` | `{row['wer_delta']}` | `{row['mos_delta_delta']}` |"
        )

    lines.extend(["", "## Human Ratings Summary", ""])
    if not filled:
        lines.append("No filled perceptual ratings yet.")
    else:
        lines.append(f"Filled rows: `{len(filled)}`")
        lines.append("")
        lines.append("| Preference | Count |")
        lines.append("|------------|-------|")
        for preference, count in sorted(preferences.items()):
            lines.append(f"| `{preference}` | `{count}` |")
        lines.append("")
        lines.append("### Preference By Style")
        lines.append("")
        for style in sorted(style_preferences):
            lines.append(f"- `{style}`: {dict(style_preferences[style])}")
        if averages:
            lines.append("")
            lines.append("### Mean Rating Scores")
            lines.append("")
            lines.append("| Field | Mean |")
            lines.append("|-------|------|")
            for field, value in sorted(averages.items()):
                lines.append(f"| `{field}` | `{value:.2f}` |")

    lines.extend([
        "",
        "## Recommended Use",
        "",
        "1. Start with priority `1` and `2` rows in the A/B dashboard.",
        "2. Prefer a candidate only if human listening agrees with the objective gain.",
        "3. Do not promote `disgust_s10` unless listening contradicts the MOS warning.",
        "",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    candidate_specs = [
        parse_candidate_spec(raw)
        for raw in (args.candidate if args.candidate is not None else DEFAULT_CANDIDATES)
    ]
    pairs, _, _ = build_pairs(
        reference_rows=read_manifest(args.reference_manifest),
        candidate_specs=candidate_specs,
        results_dir=Path(args.results_dir),
        reference_input_tag=args.reference_input_tag,
        reference_tag=args.reference_tag,
        candidate_input_tag=args.candidate_input_tag,
    )
    if not pairs:
        raise SystemExit("No matched A/B pairs found")

    rows = build_priority_rows(pairs)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_csv, rows)
    write_md(args.out_md, rows, args.ratings)
    print(f"Wrote priority CSV to {args.out_csv}")
    print(f"Wrote priority summary to {args.out_md}")
    print(f"Rows summarized: {len(rows)}")


if __name__ == "__main__":
    main()
