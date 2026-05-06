"""
Select clean generated-audio failures for the next targeted training objective.

The failure-mining CSV labels generated speaker/style rows with emotion,
content, naturalness, novelty, and collapse failures. This script turns that
row-level evidence into a conservative target plan: keep rows that look like
style-control failures, reject rows whose failures are confounded with content
or naturalness, and emit a small training-plan summary.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_FAILURE_CSV = "results/eval_mixed_teacher_generated_audio_failure_mining.csv"
DEFAULT_REFERENCE_CONDITION = "mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard"
DEFAULT_TARGET_STYLES = ["anger", "disgust", "fear"]
DEFAULT_REQUIRED_MODES = ["emotion_miss", "style_to_neutral"]
DEFAULT_EXCLUDE_MODES = [
    "content_collapse",
    "high_wer",
    "low_mos_delta",
    "identity_collapse",
    "mixed_collapse",
    "low_novelty",
]
STYLE_ORDER = [
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
    ap.add_argument("--failure-csv", default=DEFAULT_FAILURE_CSV)
    ap.add_argument("--reference-condition", default=DEFAULT_REFERENCE_CONDITION)
    ap.add_argument(
        "--target-styles",
        default=",".join(DEFAULT_TARGET_STYLES),
        help="Comma-separated styles to consider for targeted follow-up.",
    )
    ap.add_argument(
        "--required-modes",
        default=",".join(DEFAULT_REQUIRED_MODES),
        help="Comma-separated failure modes required for positive target selection.",
    )
    ap.add_argument(
        "--exclude-modes",
        default=",".join(DEFAULT_EXCLUDE_MODES),
        help="Comma-separated failure modes that reject a row as too confounded.",
    )
    ap.add_argument(
        "--min-selected-per-style",
        type=int,
        default=3,
        help="Minimum clean selected rows before a style is considered ready.",
    )
    ap.add_argument(
        "--focus-weight",
        type=float,
        default=3.0,
        help="Suggested style weight for ready target styles.",
    )
    ap.add_argument(
        "--other-weight",
        type=float,
        default=0.0,
        help="Suggested style weight for styles not selected by this target plan.",
    )
    ap.add_argument(
        "--out-csv",
        default="results/eval_mixed_teacher_failure_conditioned_targets.csv",
    )
    ap.add_argument(
        "--out-json",
        default="results/eval_mixed_teacher_failure_conditioned_targets.json",
    )
    ap.add_argument(
        "--out-md",
        default="results/eval_mixed_teacher_failure_conditioned_targets.md",
    )
    return ap.parse_args()


def parse_list(raw):
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def split_modes(row):
    return {mode for mode in row.get("failure_modes", "").split(";") if mode}


def style_sort_key(style):
    try:
        return (STYLE_ORDER.index(style), style)
    except ValueError:
        return (len(STYLE_ORDER), style)


def decide_row(row, target_styles, required_modes, exclude_modes):
    if row.get("style") not in target_styles:
        return "skip_non_target_style", "style is outside target-styles"

    modes = split_modes(row)
    missing = sorted(required_modes - modes)
    excluded = sorted(exclude_modes & modes)
    if missing:
        return "reject_missing_required_modes", "missing " + ",".join(missing)
    if excluded:
        return "reject_confounded_failure", "excluded " + ",".join(excluded)
    return "select_positive_style_target", "clean style failure"


def build_records(args):
    target_styles = set(parse_list(args.target_styles))
    required_modes = set(parse_list(args.required_modes))
    exclude_modes = set(parse_list(args.exclude_modes))
    rows = read_csv(args.failure_csv)
    records = []
    for row in rows:
        if row.get("style") not in target_styles:
            continue
        decision, reason = decide_row(row, target_styles, required_modes, exclude_modes)
        record = {
            "condition": row.get("condition", ""),
            "speaker": row.get("speaker", ""),
            "style": row.get("style", ""),
            "file": row.get("file", ""),
            "decision": decision,
            "selected": "1" if decision == "select_positive_style_target" else "0",
            "decision_reason": reason,
            "failure_score": row.get("failure_score", ""),
            "failure_modes": row.get("failure_modes", ""),
            "emotion_predicted": row.get("emotion_predicted", ""),
            "emotion_target": row.get("emotion_target", ""),
            "emotion_score": row.get("emotion_score", ""),
            "emo_sim": row.get("emo_sim", ""),
            "wer": row.get("wer", ""),
            "mos_delta_vs_baseline": row.get("mos_delta_vs_baseline", ""),
            "novelty_gain_vs_baseline": row.get("novelty_gain_vs_baseline", ""),
            "reference": row.get("reference", ""),
            "hypothesis": row.get("hypothesis", ""),
        }
        records.append(record)

    records.sort(
        key=lambda row: (
            row["condition"] != args.reference_condition,
            row["condition"],
            style_sort_key(row["style"]),
            row["decision"] != "select_positive_style_target",
            row["speaker"],
        )
    )
    return records, target_styles, required_modes, exclude_modes


def summarize(records, args, target_styles):
    condition_counts = defaultdict(Counter)
    condition_style_counts = defaultdict(lambda: defaultdict(Counter))
    for row in records:
        condition = row["condition"]
        style = row["style"]
        decision = row["decision"]
        condition_counts[condition][decision] += 1
        condition_counts[condition]["total"] += 1
        condition_style_counts[condition][style][decision] += 1
        condition_style_counts[condition][style]["total"] += 1

    reference_counts = condition_style_counts.get(args.reference_condition, {})
    ready_styles = []
    blocked_styles = []
    for style in sorted(target_styles, key=style_sort_key):
        selected = reference_counts.get(style, {}).get("select_positive_style_target", 0)
        total = reference_counts.get(style, {}).get("total", 0)
        if selected >= args.min_selected_per_style:
            ready_styles.append(style)
        else:
            blocked_styles.append({"style": style, "selected": selected, "total": total})

    style_weights = {
        style: (args.focus_weight if style in ready_styles else args.other_weight)
        for style in STYLE_ORDER
    }
    style_weights_arg = ",".join(f"{style}={style_weights[style]:g}" for style in STYLE_ORDER)
    decoder_strengths_arg = ",".join(
        f"{style}=5.0" if style in ready_styles else f"{style}=0.0"
        for style in STYLE_ORDER
    )

    return {
        "reference_condition": args.reference_condition,
        "target_styles": sorted(target_styles, key=style_sort_key),
        "required_modes": parse_list(args.required_modes),
        "exclude_modes": parse_list(args.exclude_modes),
        "min_selected_per_style": args.min_selected_per_style,
        "condition_counts": {
            condition: dict(counts) for condition, counts in sorted(condition_counts.items())
        },
        "condition_style_counts": {
            condition: {
                style: dict(counts)
                for style, counts in sorted(styles.items(), key=lambda item: style_sort_key(item[0]))
            }
            for condition, styles in sorted(condition_style_counts.items())
        },
        "ready_styles": ready_styles,
        "blocked_styles": blocked_styles,
        "recommended_style_teacher_style_weights": style_weights,
        "recommended_style_teacher_style_weights_arg": style_weights_arg,
        "recommended_decoder_prototype_style_strengths_arg": decoder_strengths_arg,
        "recommended_training_notes": [
            "Use ready styles for the first failure-conditioned positive target objective.",
            "Keep blocked styles diagnostic until style failure can be separated from content/naturalness failure.",
            "Use target-dim teacher/prototype supervision for ready styles; do not globally increase all style losses.",
        ],
    }


def write_csv(path, records):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "speaker",
        "style",
        "file",
        "decision",
        "selected",
        "decision_reason",
        "failure_score",
        "failure_modes",
        "emotion_predicted",
        "emotion_target",
        "emotion_score",
        "emo_sim",
        "wer",
        "mos_delta_vs_baseline",
        "novelty_gain_vs_baseline",
        "reference",
        "hypothesis",
    ]
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def write_json(path, summary):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path, records, summary, args):
    selected_reference = [
        row
        for row in records
        if row["condition"] == args.reference_condition
        and row["decision"] == "select_positive_style_target"
    ]
    lines = [
        "# Failure-Conditioned Target Selection",
        "",
        f"Failure CSV: `{args.failure_csv}`",
        "",
        f"Reference condition: `{args.reference_condition}`",
        "",
        "## Selection Rule",
        "",
        f"- Target styles: `{', '.join(summary['target_styles'])}`",
        f"- Required modes: `{', '.join(summary['required_modes'])}`",
        f"- Exclude modes: `{', '.join(summary['exclude_modes'])}`",
        f"- Ready style threshold: at least `{args.min_selected_per_style}` clean selected rows in the reference condition",
        "",
        "A selected row is a clean style-control failure: the generated audio missed the target emotion by collapsing to neutral, but it does not also have high WER, low MOS, low novelty, content collapse, identity collapse, or mixed collapse.",
        "",
        "## Reference Readout",
        "",
        "| Style | Target rows | Selected clean targets | Status |",
        "|-------|-------------|------------------------|--------|",
    ]
    ref_counts = summary["condition_style_counts"].get(args.reference_condition, {})
    for style in summary["target_styles"]:
        counts = ref_counts.get(style, {})
        selected = counts.get("select_positive_style_target", 0)
        total = counts.get("total", 0)
        status = "ready" if style in summary["ready_styles"] else "blocked"
        lines.append(f"| `{style}` | `{total}` | `{selected}` | `{status}` |")

    lines.extend(
        [
            "",
            "## Condition Counts",
            "",
            "| Condition | Target rows | Selected | Missing required modes | Confounded failure |",
            "|-----------|-------------|----------|------------------------|--------------------|",
        ]
    )
    for condition, counts in summary["condition_counts"].items():
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                condition,
                counts.get("total", 0),
                counts.get("select_positive_style_target", 0),
                counts.get("reject_missing_required_modes", 0),
                counts.get("reject_confounded_failure", 0),
            )
        )

    lines.extend(
        [
            "",
            "## Selected Reference Rows",
            "",
            "| Speaker | Style | Predicted | Target | WER | MOS delta | Novelty | File |",
            "|---------|-------|-----------|--------|-----|-----------|---------|------|",
        ]
    )
    for row in selected_reference:
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                row["speaker"],
                row["style"],
                row["emotion_predicted"],
                row["emotion_target"],
                row["wer"],
                row["mos_delta_vs_baseline"],
                row["novelty_gain_vs_baseline"],
                row["file"],
            )
        )

    lines.extend(
        [
            "",
            "## Recommended Training Focus",
            "",
            "- Ready styles: " + (", ".join(f"`{style}`" for style in summary["ready_styles"]) or "`none`"),
            "- Blocked styles: "
            + (
                ", ".join(
                    f"`{item['style']}` (`{item['selected']}/{item['total']}` selected)"
                    for item in summary["blocked_styles"]
                )
                or "`none`"
            ),
            "",
            "Suggested existing-trainer argument for a target-dim style-teacher follow-up:",
            "",
            "```bash",
            "--style-teacher-target-mode target_dim \\",
            "--style-teacher-require-label \\",
            f"--style-teacher-style-weights {summary['recommended_style_teacher_style_weights_arg']}",
            "```",
            "",
            "If the next run uses decoder-prototype controls, the analogous ready-style strength map is:",
            "",
            "```bash",
            f"--decoder-prototype-style-strengths {summary['recommended_decoder_prototype_style_strengths_arg']}",
            "```",
            "",
            "## Interpretation",
            "",
            "- `anger` and `disgust` are ready for a conservative positive target objective under the current reference condition.",
            "- `fear` remains blocked under the current reference because its failures are usually confounded with high WER, low MOS, or non-neutral emotion confusions.",
            "- The next experiment should be targeted and conservative: improve clean neutral-collapse failures without turning content-damaged examples into positive style targets.",
            "",
        ]
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    records, target_styles, _, _ = build_records(args)
    summary = summarize(records, args, target_styles)
    write_csv(args.out_csv, records)
    write_json(args.out_json, summary)
    write_md(args.out_md, records, summary, args)
    print(f"Wrote target decisions to {args.out_csv}")
    print(f"Wrote target summary to {args.out_json}")
    print(f"Wrote target readout to {args.out_md}")
    print(f"Ready styles: {', '.join(summary['ready_styles']) or 'none'}")
    if summary["blocked_styles"]:
        blocked = ", ".join(
            f"{item['style']} ({item['selected']}/{item['total']})"
            for item in summary["blocked_styles"]
        )
        print(f"Blocked styles: {blocked}")


if __name__ == "__main__":
    main()
