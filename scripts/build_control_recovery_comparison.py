#!/usr/bin/env python3
"""Build a matched legacy-versus-current control recovery review bundle."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path


DEFAULT_LEGACY_MANIFEST = "output/pass2_combined_eval/generation_manifest.jsonl"
DEFAULT_CURRENT_MANIFEST = (
    "output/mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_"
    "sad_enunc_guard_eval/generation_manifest.jsonl"
)
DEFAULT_LEGACY_METRICS = {
    "emotion": "results/eval_emotion_pass4_combined.csv",
    "novelty": "results/eval_novelty_pass4_combined.csv",
    "wer": "results/eval_wer_pass4_combined.csv",
    "mos": "results/eval_mos_pass4_combined.csv",
}
DEFAULT_CURRENT_METRICS = {
    "emotion": (
        "results/eval_emotion_mixed_teacher_mixed_teacher_cvrare_hybrid_style_"
        "distill_labeled_warmup_sad_enunc_guard.csv"
    ),
    "novelty": (
        "results/eval_novelty_mixed_teacher_mixed_teacher_cvrare_hybrid_style_"
        "distill_labeled_warmup_sad_enunc_guard.csv"
    ),
    "wer": (
        "results/eval_wer_mixed_teacher_mixed_teacher_cvrare_hybrid_style_"
        "distill_labeled_warmup_sad_enunc_guard.csv"
    ),
    "mos": (
        "results/eval_mos_mixed_teacher_mixed_teacher_cvrare_hybrid_style_"
        "distill_labeled_warmup_sad_enunc_guard.csv"
    ),
}
DEFAULT_STYLES = "anger,happy,neutral,sad,whisper"
DEFAULT_SOURCES = (
    "male_1_cremad_1003,female_1_cremad_1002,cremad_1004,cremad_1023,cremad_1045"
)
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
METRIC_FIELDS = [
    "predicted",
    "target",
    "match",
    "score",
    "emo_sim",
    "novelty_gain_vs_baseline",
    "wer",
    "mos",
    "delta_vs_baseline",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-manifest", default=DEFAULT_LEGACY_MANIFEST)
    parser.add_argument("--current-manifest", default=DEFAULT_CURRENT_MANIFEST)
    parser.add_argument("--legacy-label", default="legacy_combined")
    parser.add_argument("--current-label", default="current_reference_guard")
    for condition, defaults in (
        ("legacy", DEFAULT_LEGACY_METRICS),
        ("current", DEFAULT_CURRENT_METRICS),
    ):
        for metric, default in defaults.items():
            parser.add_argument(f"--{condition}-{metric}", default=default)
    parser.add_argument(
        "--styles",
        default=DEFAULT_STYLES,
        help="Comma-separated styles for the listening dashboard.",
    )
    parser.add_argument(
        "--sources",
        default=DEFAULT_SOURCES,
        help="Comma-separated source stems for the listening dashboard.",
    )
    parser.add_argument(
        "--bundle-dir",
        default="results/control_recovery_old_vs_current_review_bundle_2026-07-15",
    )
    parser.add_argument(
        "--bundle-zip",
        default="results/control_recovery_old_vs_current_review_bundle_2026-07-15.zip",
    )
    parser.add_argument(
        "--summary-out",
        default="results/control_recovery_old_vs_current_summary.md",
    )
    parser.add_argument(
        "--detail-out",
        default="results/control_recovery_old_vs_current.csv",
    )
    parser.add_argument(
        "--aggregate-out",
        default="results/control_recovery_old_vs_current_by_style.csv",
    )
    return parser.parse_args()


def split_csv_arg(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def read_manifest(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Manifest has no rows: {path}")
    return rows


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row):
    return (row.get("source_stem") or row.get("speaker"), row.get("style"))


def metric_key(row):
    return (row.get("speaker") or row.get("source_stem"), row.get("style"))


def index_unique(rows, key_fn, label):
    indexed = {}
    for row in rows:
        key = key_fn(row)
        if not all(key):
            raise ValueError(f"Missing key in {label}: {row}")
        if key in indexed:
            raise ValueError(f"Duplicate key {key} in {label}")
        indexed[key] = row
    return indexed


def load_metrics(paths):
    merged = defaultdict(dict)
    for metric, path in paths.items():
        for row in read_csv(path):
            key = metric_key(row)
            for field in METRIC_FIELDS:
                value = row.get(field)
                if value not in (None, ""):
                    merged[key][field] = value
            merged[key][f"_{metric}_path"] = str(path)
    return merged


def as_float(value):
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean(values):
    cleaned = [value for value in values if value is not None]
    return statistics.fmean(cleaned) if cleaned else None


def fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_detail_rows(
    legacy_rows,
    current_rows,
    legacy_metrics,
    current_metrics,
    legacy_label,
    current_label,
):
    legacy_index = index_unique(legacy_rows, row_key, "legacy manifest")
    current_index = index_unique(current_rows, row_key, "current manifest")
    matched_keys = sorted(set(legacy_index) & set(current_index))
    if not matched_keys:
        raise ValueError("The manifests have no matched source/style rows")
    if set(legacy_index) != set(current_index):
        missing_legacy = sorted(set(current_index) - set(legacy_index))
        missing_current = sorted(set(legacy_index) - set(current_index))
        raise ValueError(
            "Manifests are not fully matched: "
            f"missing legacy={missing_legacy[:5]}, missing current={missing_current[:5]}"
        )

    details = []
    for source, style in matched_keys:
        for label, manifest_row, metrics in (
            (legacy_label, legacy_index[(source, style)], legacy_metrics),
            (current_label, current_index[(source, style)], current_metrics),
        ):
            metric_row = metrics.get((source, style), {})
            details.append(
                {
                    "condition": label,
                    "source_stem": source,
                    "style": style,
                    "source_file": manifest_row.get("source_file", ""),
                    "audio_file": manifest_row.get("output_file", ""),
                    "style_strength": manifest_row.get("style_strength", ""),
                    "predicted": metric_row.get("predicted", ""),
                    "target": metric_row.get("target", ""),
                    "match": metric_row.get("match", ""),
                    "score": metric_row.get("score", ""),
                    "emo_sim": metric_row.get("emo_sim", ""),
                    "novelty_gain_vs_baseline": metric_row.get(
                        "novelty_gain_vs_baseline", ""
                    ),
                    "wer": metric_row.get("wer", ""),
                    "mos": metric_row.get("mos", ""),
                    "mos_delta_vs_baseline": metric_row.get("delta_vs_baseline", ""),
                }
            )
    return details


def aggregate_details(details, legacy_label, current_label):
    grouped = defaultdict(list)
    for row in details:
        if row["style"] != "baseline":
            grouped[(row["condition"], row["style"])].append(row)

    condition_style = {}
    for key, rows in grouped.items():
        scored_matches = [
            as_float(row["match"])
            for row in rows
            if row.get("target") and as_float(row.get("match")) is not None
        ]
        condition_style[key] = {
            "n": len(rows),
            "recall": mean(scored_matches),
            "mean_emo_sim": mean(as_float(row["emo_sim"]) for row in rows),
            "mean_novelty_gain": mean(
                as_float(row["novelty_gain_vs_baseline"]) for row in rows
            ),
            "mean_wer": mean(as_float(row["wer"]) for row in rows),
            "mean_mos_delta": mean(
                as_float(row["mos_delta_vs_baseline"]) for row in rows
            ),
            "content_collapse_count": sum(
                1 for row in rows if (as_float(row["wer"]) or 0.0) >= 0.8
            ),
        }

    aggregates = []
    for style in STYLE_ORDER:
        legacy = condition_style.get((legacy_label, style), {})
        current = condition_style.get((current_label, style), {})
        if not legacy and not current:
            continue
        row = {"style": style, "n": current.get("n") or legacy.get("n") or 0}
        for metric in (
            "recall",
            "mean_emo_sim",
            "mean_novelty_gain",
            "mean_wer",
            "mean_mos_delta",
            "content_collapse_count",
        ):
            legacy_value = legacy.get(metric)
            current_value = current.get(metric)
            row[f"legacy_{metric}"] = legacy_value
            row[f"current_{metric}"] = current_value
            if legacy_value is not None and current_value is not None:
                row[f"delta_{metric}"] = current_value - legacy_value
            else:
                row[f"delta_{metric}"] = None
        aggregates.append(row)
    return aggregates


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path,
    aggregates,
    legacy_label,
    current_label,
    matched_sources,
    styled_outputs_per_condition,
):
    lines = [
        "# Control Recovery: Legacy Combined vs Current Reference Guard",
        "",
        "This study compares two already-generated, fully matched evaluation corpora.",
        "It is a recovery/regression diagnostic, not a substitute for listening.",
        "",
        "## Scope",
        "",
        f"- Legacy condition: `{legacy_label}`",
        f"- Current condition: `{current_label}`",
        f"- Matched sources: `{matched_sources}`",
        f"- Styled outputs per condition: `{styled_outputs_per_condition}`",
        "- Inference seed/noise: matched (`42`, `0.0`)",
        "- Legacy style strength: `5.0` for every style",
        "- Current guard uses its checked-in per-style strength profile",
        "",
        "## Per-Style Comparison",
        "",
        (
            "| Style | Legacy recall | Current recall | Legacy WER | Current WER | "
            "Legacy MOS delta | Current MOS delta | Legacy novelty | Current novelty |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregates:
        lines.append(
            f"| `{row['style']}` | {fmt(row['legacy_recall'])} | "
            f"{fmt(row['current_recall'])} | {fmt(row['legacy_mean_wer'])} | "
            f"{fmt(row['current_mean_wer'])} | {fmt(row['legacy_mean_mos_delta'])} | "
            f"{fmt(row['current_mean_mos_delta'])} | "
            f"{fmt(row['legacy_mean_novelty_gain'])} | "
            f"{fmt(row['current_mean_novelty_gain'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Recall applies only to the six emotion2vec-aligned emotions.",
            "- Whisper, confused, and enunciated require listening; their recall is `n/a`.",
            "- Lower WER is better. A higher MOS delta is better. Higher novelty means more identity movement, not necessarily better control.",
            "- A control is recovered only if the intended attribute is audible and intelligibility/naturalness remain acceptable.",
            "- The matched listening dashboard is the decision gate for `anger`, `happy`, `neutral`, `sad`, and `whisper`.",
            "",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def safe_copy(source, destination):
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_listening_audio(details, styles, sources, bundle_dir, legacy_label, current_label):
    index = {(row["condition"], row["source_stem"], row["style"]): row for row in details}
    copied = {}
    for source in sources:
        source_rows = [row for row in details if row["source_stem"] == source]
        if not source_rows:
            raise ValueError(f"Unknown listening source: {source}")
        source_path = source_rows[0]["source_file"]
        destination = bundle_dir / "audio" / "source" / f"{source}{Path(source_path).suffix}"
        safe_copy(source_path, destination)
        copied[("source", source, "source")] = destination
        for condition in (legacy_label, current_label):
            for style in ["baseline", *styles]:
                row = index.get((condition, source, style))
                if not row:
                    raise ValueError(f"Missing {condition}/{source}/{style}")
                destination = (
                    bundle_dir
                    / "audio"
                    / condition
                    / f"{source}_{style}{Path(row['audio_file']).suffix}"
                )
                safe_copy(row["audio_file"], destination)
                copied[(condition, source, style)] = destination
    return copied


def relative_bundle_path(path, bundle_dir):
    return Path(path).relative_to(bundle_dir).as_posix()


def render_audio(path, label, bundle_dir):
    src = html.escape(relative_bundle_path(path, bundle_dir))
    return (
        f"<div class=\"audio-label\">{html.escape(label)}</div>"
        f"<audio controls preload=\"none\" src=\"{src}\"></audio>"
    )


def metric_block(row):
    fields = [
        ("prediction", row.get("predicted", "")),
        ("match", row.get("match", "")),
        ("emo sim", row.get("emo_sim", "")),
        ("novelty", row.get("novelty_gain_vs_baseline", "")),
        ("WER", row.get("wer", "")),
        ("MOS delta", row.get("mos_delta_vs_baseline", "")),
    ]
    return "<dl>" + "".join(
        f"<dt>{html.escape(label)}</dt><dd>{html.escape(str(value))}</dd>"
        for label, value in fields
    ) + "</dl>"


def write_ratings(path, details, styles, sources, copied, legacy_label, current_label, bundle_dir):
    index = {(row["condition"], row["source_stem"], row["style"]): row for row in details}
    rows = []
    for style in styles:
        for source in sources:
            rows.append(
                {
                    "source_stem": source,
                    "style": style,
                    "legacy_audio": relative_bundle_path(
                        copied[(legacy_label, source, style)], bundle_dir
                    ),
                    "current_audio": relative_bundle_path(
                        copied[(current_label, source, style)], bundle_dir
                    ),
                    "preference_legacy_current_tie": "",
                    "legacy_target_match_1_5": "",
                    "current_target_match_1_5": "",
                    "legacy_intelligibility_1_5": "",
                    "current_intelligibility_1_5": "",
                    "legacy_naturalness_1_5": "",
                    "current_naturalness_1_5": "",
                    "notes": "",
                }
            )
    write_csv(path, rows)


def write_html(path, details, styles, sources, copied, legacy_label, current_label, bundle_dir):
    index = {(row["condition"], row["source_stem"], row["style"]): row for row in details}
    lines = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "<title>Control Recovery Review</title>",
        "<style>",
        ":root{--bg:#f5f5f2;--surface:#fff;--ink:#20211f;--muted:#686b65;--line:#d7d9d3;--legacy:#8a3a2f;--current:#215d50}",
        "*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}",
        "main{max-width:1440px;margin:0 auto;padding:24px 20px 56px}header{border-bottom:1px solid var(--line);padding-bottom:18px}h1{font-size:28px;margin:0 0 6px;letter-spacing:0}h2{font-size:20px;margin:30px 0 10px;letter-spacing:0}",
        ".meta,.note{color:var(--muted)}.protocol{padding:14px 0;border-bottom:1px solid var(--line)}.protocol ol{margin:8px 0;padding-left:22px}",
        ".table-wrap{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:6px}table{width:100%;border-collapse:collapse;min-width:1180px}th,td{padding:10px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}th{font-size:12px;color:var(--muted);font-weight:600;background:#f0f1ed}tr:last-child td{border-bottom:0}",
        ".style{font-weight:700}.legacy{color:var(--legacy)}.current{color:var(--current)}.audio-label{font-size:12px;color:var(--muted);margin:0 0 3px}audio{width:190px;height:34px}dl{display:grid;grid-template-columns:auto 1fr;gap:2px 8px;margin:8px 0 0;font-size:12px}dt{color:var(--muted)}dd{margin:0;font-variant-numeric:tabular-nums}",
        "@media(max-width:720px){main{padding:18px 12px 40px}h1{font-size:24px}.table-wrap{border-radius:4px}}",
        "</style>",
        "</head>",
        "<body><main>",
        "<header><h1>Control Recovery Review</h1>",
        f"<div class=\"meta\">{len(styles) * len(sources)} matched comparisons across {len(sources)} sources</div></header>",
        "<section class=\"protocol\"><strong>Review protocol</strong><ol>",
        "<li>Listen to the source, then both condition baselines.</li>",
        "<li>Compare the legacy and current styled outputs without using metric values as the answer.</li>",
        "<li>Record which output expresses the target more clearly while remaining intelligible and natural.</li>",
        "<li>Use <code>ratings.csv</code>; a metric-only win is not a recovered paper claim.</li>",
        "</ol></section>",
    ]
    for style in styles:
        lines.extend(
            [
                f"<h2>{html.escape(style.title())}</h2>",
                "<div class=\"table-wrap\"><table>",
                "<thead><tr><th>Source</th><th>Source and baselines</th>"
                f"<th class=\"legacy\">{html.escape(legacy_label)}</th><th>Legacy metrics</th>"
                f"<th class=\"current\">{html.escape(current_label)}</th><th>Current metrics</th></tr></thead><tbody>",
            ]
        )
        for source in sources:
            legacy_row = index[(legacy_label, source, style)]
            current_row = index[(current_label, source, style)]
            context_audio = "".join(
                [
                    render_audio(copied[("source", source, "source")], "source", bundle_dir),
                    render_audio(copied[(legacy_label, source, "baseline")], "legacy baseline", bundle_dir),
                    render_audio(copied[(current_label, source, "baseline")], "current baseline", bundle_dir),
                ]
            )
            lines.append(
                "<tr>"
                f"<td><span class=\"style\">{html.escape(source)}</span></td>"
                f"<td>{context_audio}</td>"
                f"<td>{render_audio(copied[(legacy_label, source, style)], 'legacy styled', bundle_dir)}</td>"
                f"<td>{metric_block(legacy_row)}</td>"
                f"<td>{render_audio(copied[(current_label, source, style)], 'current styled', bundle_dir)}</td>"
                f"<td>{metric_block(current_row)}</td>"
                "</tr>"
            )
        lines.extend(["</tbody></table></div>"])
    lines.extend(["</main></body></html>"])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_bundle_readme(path, styles, sources, legacy_label, current_label):
    text = f"""# Control Recovery Review Bundle

Purpose: compare the earlier project-generated combined style checkpoint with the current reference guard on matched source/style rows.

Conditions:

- `{legacy_label}`
- `{current_label}`

Listening scope:

- styles: {', '.join(f'`{style}`' for style in styles)}
- sources: {', '.join(f'`{source}`' for source in sources)}

Open `index.html` through a local HTTP server and record judgments in `ratings.csv`.
The full objective summary covers all nine styles and all eleven matched speakers.
"""
    Path(path).write_text(text, encoding="utf-8")


def write_zip(bundle_dir, zip_path):
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(bundle_dir.parent))


def main():
    args = parse_args()
    styles = split_csv_arg(args.styles)
    sources = split_csv_arg(args.sources)
    unknown_styles = sorted(set(styles) - set(STYLE_ORDER))
    if unknown_styles:
        raise SystemExit(f"Unknown styles: {unknown_styles}")

    legacy_metric_paths = {
        metric: getattr(args, f"legacy_{metric}") for metric in DEFAULT_LEGACY_METRICS
    }
    current_metric_paths = {
        metric: getattr(args, f"current_{metric}") for metric in DEFAULT_CURRENT_METRICS
    }
    legacy_rows = read_manifest(args.legacy_manifest)
    current_rows = read_manifest(args.current_manifest)
    details = build_detail_rows(
        legacy_rows,
        current_rows,
        load_metrics(legacy_metric_paths),
        load_metrics(current_metric_paths),
        args.legacy_label,
        args.current_label,
    )
    aggregates = aggregate_details(details, args.legacy_label, args.current_label)
    matched_sources = len({row["source_stem"] for row in details})
    styled_outputs_per_condition = sum(
        1
        for row in details
        if row["condition"] == args.current_label and row["style"] != "baseline"
    )

    write_csv(args.detail_out, details)
    write_csv(args.aggregate_out, aggregates)
    write_summary(
        args.summary_out,
        aggregates,
        args.legacy_label,
        args.current_label,
        matched_sources,
        styled_outputs_per_condition,
    )

    bundle_dir = Path(args.bundle_dir)
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    copied = copy_listening_audio(
        details,
        styles,
        sources,
        bundle_dir,
        args.legacy_label,
        args.current_label,
    )
    write_html(
        bundle_dir / "index.html",
        details,
        styles,
        sources,
        copied,
        args.legacy_label,
        args.current_label,
        bundle_dir,
    )
    write_ratings(
        bundle_dir / "ratings.csv",
        details,
        styles,
        sources,
        copied,
        args.legacy_label,
        args.current_label,
        bundle_dir,
    )
    safe_copy(args.summary_out, bundle_dir / "summary.md")
    safe_copy(args.aggregate_out, bundle_dir / "comparison_by_style.csv")
    safe_copy(args.detail_out, bundle_dir / "comparison_rows.csv")
    write_bundle_readme(
        bundle_dir / "README.md",
        styles,
        sources,
        args.legacy_label,
        args.current_label,
    )
    write_zip(bundle_dir, args.bundle_zip)

    print(f"Matched manifest rows per condition: {len(legacy_rows)}")
    print(f"Wrote summary: {args.summary_out}")
    print(f"Wrote bundle: {bundle_dir}")
    print(f"Wrote zip: {args.bundle_zip}")


if __name__ == "__main__":
    main()
