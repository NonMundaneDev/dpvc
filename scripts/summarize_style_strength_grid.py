#!/usr/bin/env python3
"""
Summarize and rank generated-audio style-strength grid conditions.

Input is the summary CSV produced by summarize_mixed_teacher_results.py for a
grid-specific input tag. Conditions are expected to be named like:

  <condition-prefix>_<style>_s<strength-token>

Example:

  mixed_teacher_cvrare_strength_grid_disgust_s7p5
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


SUMMARY_KEYS = [
    "emotion_recall",
    "mean_novelty_gain_vs_baseline",
    "mean_wer",
    "mean_mos_delta_vs_baseline",
    "content_collapse_count",
    "style_collapse_to_neutral_count",
    "identity_collapse_to_baseline_count",
    "mixed_collapse_count",
    "files_with_any_collapse",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", required=True, help="Grid summary CSV")
    ap.add_argument("--condition-prefix", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    return ap.parse_args()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_float(value, default=None):
    if value in ("", None):
        return default
    return float(value)


def parse_int(value, default=0):
    if value in ("", None):
        return default
    return int(value)


def parse_strength(token):
    return float(token.replace("p", ".").replace("m", "-"))


def parse_condition(condition, prefix):
    stem = f"{prefix}_"
    if not condition.startswith(stem):
        return None
    tail = condition[len(stem):]
    if "_s" not in tail:
        return None
    style, strength_token = tail.rsplit("_s", 1)
    if not style or not strength_token:
        return None
    try:
        strength = parse_strength(strength_token)
    except ValueError:
        return None
    return style, strength


def sort_key(row):
    recall = parse_float(row.get("emotion_recall"), -1.0)
    novelty = parse_float(row.get("mean_novelty_gain_vs_baseline"), -math.inf)
    wer = parse_float(row.get("mean_wer"), math.inf)
    mos_delta = parse_float(row.get("mean_mos_delta_vs_baseline"), -math.inf)
    any_collapse = parse_int(row.get("files_with_any_collapse"), 999999)
    neutral_collapse = parse_int(row.get("style_collapse_to_neutral_count"), 999999)
    # Higher recall is primary; ties prefer fewer failures, lower WER, higher
    # MOS delta, then higher novelty.
    return (-recall, any_collapse, neutral_collapse, wer, -mos_delta, -novelty)


def write_csv(path, rows):
    fieldnames = [
        "style",
        "rank",
        "strength",
        "condition",
        "recommendation",
        "styles_present",
        "sources_count",
        "rows_total",
    ] + SUMMARY_KEYS
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return value if value not in ("", None) else "-"


def write_md(path, rows, prefix):
    best = [row for row in rows if row["rank"] == "1"]
    lines = [
        "# Style-Strength Grid Ranking",
        "",
        f"Condition prefix: `{prefix}`",
        "",
        "## Best Per Style",
        "",
        "| Style | Strength | Recall | Novelty | WER | MOS delta | Style-to-neutral | Any collapse | Condition |",
        "|-------|----------|--------|---------|-----|-----------|------------------|--------------|-----------|",
    ]
    for row in best:
        lines.append(
            "| `{style}` | `{strength}` | `{recall}` | `{novelty}` | `{wer}` | `{mos}` | `{neutral}` | `{any_collapse}` | `{condition}` |".format(
                style=row["style"],
                strength=row["strength"],
                recall=fmt(row["emotion_recall"]),
                novelty=fmt(row["mean_novelty_gain_vs_baseline"]),
                wer=fmt(row["mean_wer"]),
                mos=fmt(row["mean_mos_delta_vs_baseline"]),
                neutral=fmt(row["style_collapse_to_neutral_count"]),
                any_collapse=fmt(row["files_with_any_collapse"]),
                condition=row["condition"],
            )
        )

    lines.extend(
        [
            "",
            "## Full Ranking",
            "",
            "| Style | Rank | Strength | Recall | Novelty | WER | MOS delta | Style-to-neutral | Any collapse |",
            "|-------|------|----------|--------|---------|-----|-----------|------------------|--------------|",
        ]
    )
    for row in rows:
        lines.append(
            "| `{style}` | `{rank}` | `{strength}` | `{recall}` | `{novelty}` | `{wer}` | `{mos}` | `{neutral}` | `{any_collapse}` |".format(
                style=row["style"],
                rank=row["rank"],
                strength=row["strength"],
                recall=fmt(row["emotion_recall"]),
                novelty=fmt(row["mean_novelty_gain_vs_baseline"]),
                wer=fmt(row["mean_wer"]),
                mos=fmt(row["mean_mos_delta_vs_baseline"]),
                neutral=fmt(row["style_collapse_to_neutral_count"]),
                any_collapse=fmt(row["files_with_any_collapse"]),
            )
        )

    lines.extend(
        [
            "",
            "## Ranking Rule",
            "",
            "Rows are ranked within each style by: higher emotion recall, fewer files with any collapse, fewer style-to-neutral collapses, lower WER, higher MOS delta, then higher novelty.",
            "",
            "Use this as an audio-calibrated candidate selector. A strength should still be checked perceptually before becoming a default.",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    rows = []
    for row in read_csv(args.summary):
        parsed = parse_condition(row["condition"], args.condition_prefix)
        if not parsed:
            continue
        style, strength = parsed
        rows.append({"style": style, "strength": strength, **row})
    if not rows:
        raise SystemExit(
            f"No rows found in {args.summary} for prefix {args.condition_prefix!r}"
        )

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["style"]].append(row)

    ranked_rows = []
    for style in sorted(grouped):
        style_rows = sorted(grouped[style], key=sort_key)
        for rank, row in enumerate(style_rows, start=1):
            ranked = {
                "style": style,
                "rank": str(rank),
                "strength": f"{row['strength']:g}",
                "condition": row["condition"],
                "recommendation": "best_for_style" if rank == 1 else "",
                "styles_present": row.get("styles_present", ""),
                "sources_count": row.get("sources_count", ""),
                "rows_total": row.get("rows_total", ""),
            }
            for key in SUMMARY_KEYS:
                ranked[key] = row.get(key, "")
            ranked_rows.append(ranked)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_csv, ranked_rows)
    write_md(args.out_md, ranked_rows, args.condition_prefix)
    print(f"Wrote grid ranking CSV to {args.out_csv}")
    print(f"Wrote grid ranking summary to {args.out_md}")


if __name__ == "__main__":
    main()
