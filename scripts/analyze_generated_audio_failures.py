"""
Build a row-level generated-audio failure-mining artifact.

The generated-audio evaluation suite already writes separate CSVs for emotion,
novelty, WER, MOS, and collapse taxonomy. This script joins those outputs so a
training follow-up can inspect the actual speaker/style failures instead of
reasoning only from aggregate summary rows.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_CONDITIONS = [
    "mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard",
    "mixed_teacher_cvrare_decoder_proto_labeled_warmup",
    "mixed_teacher_cvrare_decoder_proto_labeled_warmup_sad_enunc_guard",
    "mixed_teacher_cvrare_decoder_proto_w005_labeled_warmup",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS,
        help="Condition names to join. Defaults to the current cvrare reference and decoder-prototype family.",
    )
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--input-tag", default="mixed_teacher")
    ap.add_argument(
        "--summary",
        default=None,
        help="Summary CSV path. Defaults to results/eval_<input_tag>_summary.csv.",
    )
    ap.add_argument(
        "--collapse",
        default=None,
        help="Collapse CSV path. Defaults to results/eval_<input_tag>_collapse.csv.",
    )
    ap.add_argument(
        "--out-csv",
        default="results/eval_mixed_teacher_generated_audio_failure_mining.csv",
    )
    ap.add_argument(
        "--out-md",
        default="results/eval_mixed_teacher_generated_audio_failure_mining.md",
    )
    ap.add_argument("--high-wer-threshold", type=float, default=0.30)
    ap.add_argument("--low-mos-delta-threshold", type=float, default=-0.50)
    ap.add_argument("--low-novelty-threshold", type=float, default=0.05)
    ap.add_argument("--top-rows", type=int, default=20)
    return ap.parse_args()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def maybe_float(value):
    if value in ("", None):
        return None
    return float(value)


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def metric_path(results_dir, metric, input_tag, condition):
    path = Path(results_dir) / f"eval_{metric}_{input_tag}_{condition}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {metric} CSV for {condition}: {path}")
    return path


def row_key(condition, row):
    file_name = row.get("file") or Path(row.get("generated_file", "")).name
    return condition, file_name


def load_summary(path, conditions):
    if not Path(path).exists():
        return {}
    rows = read_csv(path)
    return {row["condition"]: row for row in rows if row.get("condition") in conditions}


def load_collapse(path, conditions):
    if not Path(path).exists():
        return {}
    collapse = {}
    for row in read_csv(path):
        condition = row.get("condition", "")
        if condition not in conditions:
            continue
        collapse[(condition, row.get("file", ""))] = row
    return collapse


def collect_rows(args):
    conditions = list(dict.fromkeys(args.conditions))
    summary_path = args.summary or str(Path(args.results_dir) / f"eval_{args.input_tag}_summary.csv")
    collapse_path = args.collapse or str(Path(args.results_dir) / f"eval_{args.input_tag}_collapse.csv")
    summary = load_summary(summary_path, conditions)
    collapse = load_collapse(collapse_path, conditions)

    joined = {}
    for condition in conditions:
        emotion_rows = read_csv(metric_path(args.results_dir, "emotion", args.input_tag, condition))
        novelty_rows = read_csv(metric_path(args.results_dir, "novelty", args.input_tag, condition))
        wer_rows = read_csv(metric_path(args.results_dir, "wer", args.input_tag, condition))
        mos_rows = read_csv(metric_path(args.results_dir, "mos", args.input_tag, condition))

        for source, label in [
            (emotion_rows, "emotion"),
            (novelty_rows, "novelty"),
            (wer_rows, "wer"),
            (mos_rows, "mos"),
        ]:
            for row in source:
                key = row_key(condition, row)
                if key[1] == "":
                    continue
                record = joined.setdefault(
                    key,
                    {
                        "condition": condition,
                        "file": key[1],
                        "speaker": row.get("speaker", ""),
                        "style": row.get("style", ""),
                    },
                )
                record["speaker"] = record.get("speaker") or row.get("speaker", "")
                record["style"] = record.get("style") or row.get("style", "")
                if label == "emotion":
                    record.update(
                        {
                            "emotion_predicted": row.get("predicted", ""),
                            "emotion_target": row.get("target", ""),
                            "emotion_match": row.get("match", ""),
                            "emotion_score": row.get("score", ""),
                            "emo_sim": row.get("emo_sim", ""),
                        }
                    )
                elif label == "novelty":
                    record.update(
                        {
                            "novelty_gain_vs_baseline": row.get("novelty_gain_vs_baseline", ""),
                            "similarity": row.get("similarity", ""),
                            "baseline_similarity": row.get("baseline_similarity", ""),
                        }
                    )
                elif label == "wer":
                    record.update(
                        {
                            "wer": row.get("wer", ""),
                            "reference": row.get("reference", ""),
                            "hypothesis": row.get("hypothesis", ""),
                        }
                    )
                elif label == "mos":
                    record.update(
                        {
                            "mos": row.get("mos", ""),
                            "mos_delta_vs_baseline": row.get("delta_vs_baseline", ""),
                        }
                    )

        for key, record in list(joined.items()):
            if key[0] != condition:
                continue
            flags = collapse.get(key, {})
            record["content_collapse"] = flags.get("content_collapse", "0")
            record["style_collapse_to_neutral"] = flags.get("style_collapse_to_neutral", "0")
            record["identity_collapse_to_baseline"] = flags.get("identity_collapse_to_baseline", "0")
            record["mixed_collapse"] = flags.get("mixed_collapse", "0")

    styled = []
    for record in joined.values():
        if record.get("style") == "baseline":
            continue
        add_failure_labels(record, args)
        styled.append(record)

    styled.sort(
        key=lambda row: (
            row["condition"],
            -int(row["failure_score"]),
            row.get("speaker", ""),
            row.get("style", ""),
        )
    )
    return styled, summary, summary_path, collapse_path


def add_failure_labels(record, args):
    modes = []
    score = 0

    target = record.get("emotion_target", "")
    match = record.get("emotion_match", "")
    if target and match == "0":
        modes.append("emotion_miss")
        score += 3

    if record.get("content_collapse") == "1":
        modes.append("content_collapse")
        score += 3
    if record.get("style_collapse_to_neutral") == "1":
        modes.append("style_to_neutral")
        score += 2
    if record.get("identity_collapse_to_baseline") == "1":
        modes.append("identity_collapse")
        score += 2
    if record.get("mixed_collapse") == "1":
        modes.append("mixed_collapse")
        score += 2

    wer = maybe_float(record.get("wer"))
    if wer is not None and wer >= args.high_wer_threshold:
        modes.append("high_wer")
        score += 1

    mos_delta = maybe_float(record.get("mos_delta_vs_baseline"))
    if mos_delta is not None and mos_delta <= args.low_mos_delta_threshold:
        modes.append("low_mos_delta")
        score += 1

    novelty = maybe_float(record.get("novelty_gain_vs_baseline"))
    if novelty is not None and novelty <= args.low_novelty_threshold:
        modes.append("low_novelty")
        score += 1

    record["failure_modes"] = ";".join(modes)
    record["failure_score"] = str(score)
    record["any_failure"] = "1" if modes else "0"


def safe_mean(values):
    return statistics.mean(values) if values else None


def condition_failure_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    summaries = []
    for condition, condition_rows in grouped.items():
        failures = [row for row in condition_rows if row["any_failure"] == "1"]
        summaries.append(
            {
                "condition": condition,
                "styled_rows": len(condition_rows),
                "rows_with_any_failure": len(failures),
                "emotion_misses": sum(1 for row in condition_rows if "emotion_miss" in row["failure_modes"]),
                "high_wer_rows": sum(1 for row in condition_rows if "high_wer" in row["failure_modes"]),
                "low_mos_delta_rows": sum(1 for row in condition_rows if "low_mos_delta" in row["failure_modes"]),
                "low_novelty_rows": sum(1 for row in condition_rows if "low_novelty" in row["failure_modes"]),
                "mean_failure_score": safe_mean([int(row["failure_score"]) for row in condition_rows]),
            }
        )
    return summaries


def style_failure_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["style"])].append(row)
    summaries = []
    for (condition, style), style_rows in sorted(grouped.items()):
        failures = [row for row in style_rows if row["any_failure"] == "1"]
        emotion_scored = [row for row in style_rows if row.get("emotion_match") in {"0", "1"}]
        emotion_hits = sum(1 for row in emotion_scored if row.get("emotion_match") == "1")
        summaries.append(
            {
                "condition": condition,
                "style": style,
                "rows": len(style_rows),
                "rows_with_any_failure": len(failures),
                "emotion_recall": (emotion_hits / len(emotion_scored)) if emotion_scored else None,
                "mean_wer": safe_mean(
                    [
                        maybe_float(row.get("wer"))
                        for row in style_rows
                        if maybe_float(row.get("wer")) is not None
                    ]
                ),
                "mean_mos_delta": safe_mean(
                    [
                        maybe_float(row.get("mos_delta_vs_baseline"))
                        for row in style_rows
                        if maybe_float(row.get("mos_delta_vs_baseline")) is not None
                    ]
                ),
                "mean_novelty": safe_mean(
                    [
                        maybe_float(row.get("novelty_gain_vs_baseline"))
                        for row in style_rows
                        if maybe_float(row.get("novelty_gain_vs_baseline")) is not None
                    ]
                ),
            }
        )
    return summaries


def write_csv(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "speaker",
        "style",
        "file",
        "failure_score",
        "failure_modes",
        "any_failure",
        "emotion_predicted",
        "emotion_target",
        "emotion_match",
        "emotion_score",
        "emo_sim",
        "novelty_gain_vs_baseline",
        "wer",
        "mos",
        "mos_delta_vs_baseline",
        "content_collapse",
        "style_collapse_to_neutral",
        "identity_collapse_to_baseline",
        "mixed_collapse",
        "similarity",
        "baseline_similarity",
        "reference",
        "hypothesis",
    ]
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def observed_readout(condition_summaries, style_summaries):
    lines = []
    if condition_summaries:
        best = min(
            condition_summaries,
            key=lambda row: (
                row["mean_failure_score"] if row["mean_failure_score"] is not None else float("inf"),
                row["rows_with_any_failure"],
            ),
        )
        worst = max(
            condition_summaries,
            key=lambda row: (
                row["mean_failure_score"] if row["mean_failure_score"] is not None else -1,
                row["rows_with_any_failure"],
            ),
        )
        lines.append(
            "- Lowest row-level failure score: `{}` (`{}` any-failure rows, mean score `{}`).".format(
                best["condition"],
                best["rows_with_any_failure"],
                fmt(best["mean_failure_score"]),
            )
        )
        lines.append(
            "- Highest row-level failure score: `{}` (`{}` any-failure rows, mean score `{}`).".format(
                worst["condition"],
                worst["rows_with_any_failure"],
                fmt(worst["mean_failure_score"]),
            )
        )

    style_totals = defaultdict(lambda: {"rows": 0, "failures": 0, "recalls": []})
    for row in style_summaries:
        style = row["style"]
        style_totals[style]["rows"] += row["rows"]
        style_totals[style]["failures"] += row["rows_with_any_failure"]
        if row["emotion_recall"] is not None:
            style_totals[style]["recalls"].append(row["emotion_recall"])

    ranked_styles = sorted(
        style_totals.items(),
        key=lambda item: (-(item[1]["failures"] / item[1]["rows"]), -item[1]["failures"], item[0]),
    )
    if ranked_styles:
        summary = []
        for style, values in ranked_styles[:4]:
            rate = values["failures"] / values["rows"] if values["rows"] else 0.0
            recall = safe_mean(values["recalls"])
            recall_text = f", mean emotion recall `{fmt(recall)}`" if recall is not None else ""
            summary.append(
                f"`{style}` `{values['failures']}/{values['rows']}` failures ({rate:.1%}{recall_text})"
            )
        lines.append("- Most persistent style failures: " + "; ".join(summary) + ".")

    lines.append(
        "- Objective-design implication: prioritize generated-audio rows with `emotion_miss` plus `style_to_neutral` for anger/disgust/fear, and keep high-WER or very-low-MOS rows out of direct positive targets unless the goal is content repair."
    )
    return lines


def write_md(path, rows, summary_rows, summary_path, collapse_path, args):
    condition_summaries = condition_failure_summary(rows)
    style_summaries = style_failure_summary(rows)
    worst_rows = sorted(rows, key=lambda row: (-int(row["failure_score"]), row["condition"], row["speaker"]))[
        : args.top_rows
    ]

    lines = [
        "# Generated-Audio Failure Mining",
        "",
        f"Input tag: `{args.input_tag}`",
        "",
        f"Summary source: `{summary_path}`",
        "",
        f"Collapse source: `{collapse_path}`",
        "",
        "## Purpose",
        "",
        "Join generated-audio metrics at the speaker/style row level so the next training objective can target observed failures rather than aggregate averages.",
        "",
        "## Compared Conditions",
        "",
        "| Condition | Recall | Novelty | WER | MOS delta | Any collapse |",
        "|-----------|--------|---------|-----|-----------|--------------|",
    ]
    for condition in args.conditions:
        row = summary_rows.get(condition, {})
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                condition,
                row.get("emotion_recall", ""),
                row.get("mean_novelty_gain_vs_baseline", ""),
                row.get("mean_wer", ""),
                row.get("mean_mos_delta_vs_baseline", ""),
                row.get("files_with_any_collapse", ""),
            )
        )

    lines.extend(
        [
            "",
            "## Failure Labels",
            "",
            f"- `emotion_miss`: emotion2vec target exists and the generated file missed it.",
            f"- `high_wer`: WER >= `{args.high_wer_threshold}`.",
            f"- `low_mos_delta`: MOS delta vs baseline <= `{args.low_mos_delta_threshold}`.",
            f"- `low_novelty`: novelty gain vs baseline <= `{args.low_novelty_threshold}`.",
            "- collapse labels come from the existing collapse taxonomy CSV.",
            "",
            "## Condition-Level Failure Counts",
            "",
            "| Condition | Styled rows | Any failure | Emotion misses | High WER | Low MOS delta | Low novelty | Mean failure score |",
            "|-----------|-------------|-------------|----------------|----------|---------------|-------------|--------------------|",
        ]
    )
    for row in condition_summaries:
        lines.append(
            "| `{condition}` | `{styled_rows}` | `{rows_with_any_failure}` | `{emotion_misses}` | `{high_wer_rows}` | `{low_mos_delta_rows}` | `{low_novelty_rows}` | `{score}` |".format(
                condition=row["condition"],
                styled_rows=row["styled_rows"],
                rows_with_any_failure=row["rows_with_any_failure"],
                emotion_misses=row["emotion_misses"],
                high_wer_rows=row["high_wer_rows"],
                low_mos_delta_rows=row["low_mos_delta_rows"],
                low_novelty_rows=row["low_novelty_rows"],
                score=fmt(row["mean_failure_score"]),
            )
        )

    lines.extend(
        [
            "",
            "## Observed Readout",
            "",
            *observed_readout(condition_summaries, style_summaries),
            "",
            "## Style-Level Readout",
            "",
            "| Condition | Style | Rows | Any failure | Emotion recall | Mean WER | Mean MOS delta | Mean novelty |",
            "|-----------|-------|------|-------------|----------------|----------|----------------|--------------|",
        ]
    )
    for row in style_summaries:
        lines.append(
            "| `{condition}` | `{style}` | `{rows}` | `{failures}` | `{recall}` | `{wer}` | `{mos}` | `{novelty}` |".format(
                condition=row["condition"],
                style=row["style"],
                rows=row["rows"],
                failures=row["rows_with_any_failure"],
                recall=fmt(row["emotion_recall"]),
                wer=fmt(row["mean_wer"]),
                mos=fmt(row["mean_mos_delta"]),
                novelty=fmt(row["mean_novelty"]),
            )
        )

    mode_counts = Counter()
    for row in rows:
        for mode in row["failure_modes"].split(";"):
            if mode:
                mode_counts[(row["condition"], mode)] += 1

    lines.extend(
        [
            "",
            "## Failure Mode Totals",
            "",
            "| Condition | Mode | Count |",
            "|-----------|------|-------|",
        ]
    )
    for (condition, mode), count in sorted(mode_counts.items()):
        lines.append(f"| `{condition}` | `{mode}` | `{count}` |")

    lines.extend(
        [
            "",
            "## Highest-Priority Rows",
            "",
            "| Condition | Speaker | Style | Score | Modes | Predicted | Target | WER | MOS delta | Novelty | File |",
            "|-----------|---------|-------|-------|-------|-----------|--------|-----|-----------|---------|------|",
        ]
    )
    for row in worst_rows:
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                row["condition"],
                row.get("speaker", ""),
                row.get("style", ""),
                row.get("failure_score", ""),
                row.get("failure_modes", ""),
                row.get("emotion_predicted", ""),
                row.get("emotion_target", ""),
                row.get("wer", ""),
                row.get("mos_delta_vs_baseline", ""),
                row.get("novelty_gain_vs_baseline", ""),
                row.get("file", ""),
            )
        )

    lines.extend(
        [
            "",
            "## Training Implication",
            "",
            "- Use this artifact to choose the next decoder-aware objective by style and failure mode.",
            "- Prefer target-style/generated-audio failures over another global scalar loss sweep.",
            "- Inspect rows with high emotion-miss plus high WER or low MOS before turning them into training targets; some may be bad supervision examples rather than useful corrective targets.",
            "",
        ]
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    rows, summary, summary_path, collapse_path = collect_rows(args)
    write_csv(args.out_csv, rows)
    write_md(args.out_md, rows, summary, summary_path, collapse_path, args)
    print(f"Wrote joined failure CSV to {args.out_csv}")
    print(f"Wrote failure summary to {args.out_md}")


if __name__ == "__main__":
    main()
