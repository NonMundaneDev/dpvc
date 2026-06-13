"""
Build a local HTML listening report for a generated OpenVoice evaluation corpus.

The report groups generated files by source speaker, embeds browser audio
controls, joins available metric CSVs, and writes a perceptual rating template
CSV so subjective notes can be collected beside objective metrics.
"""

from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import os
from collections import defaultdict
from pathlib import Path


METRICS = ("emotion", "novelty", "wer", "mos")
STYLE_ORDER = [
    "baseline",
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
        "--manifest",
        required=True,
        help="generation_manifest.jsonl from an inference/evaluation output directory",
    )
    ap.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing eval_* CSVs (default: results)",
    )
    ap.add_argument(
        "--input-tag",
        default=None,
        help=(
            "Optional result tag between metric and condition, e.g. mixed_teacher. "
            "If omitted, metric CSVs are discovered by condition suffix."
        ),
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output HTML path (default: results/listening_<condition>.html)",
    )
    ap.add_argument(
        "--rating-template",
        default=None,
        help="Output subjective-rating CSV path (default: HTML path with _ratings.csv)",
    )
    ap.add_argument(
        "--title",
        default=None,
        help="Report title (default: condition name)",
    )
    ap.add_argument(
        "--max-sources",
        type=int,
        default=None,
        help="Optional cap on number of source speakers included",
    )
    ap.add_argument(
        "--styles",
        default="",
        help="Optional comma-separated styles to include, preserving baseline if present",
    )
    return ap.parse_args()


def read_manifest(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Manifest has no rows: {path}")
    return rows


def read_csv(path):
    if not path or not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def discover_metric_csv(results_dir, metric, condition, input_tag=None):
    if input_tag:
        path = Path(results_dir) / f"eval_{metric}_{input_tag}_{condition}.csv"
        return str(path) if path.exists() else None

    matches = sorted(glob.glob(str(Path(results_dir) / f"eval_{metric}_*_{condition}.csv")))
    if len(matches) > 1:
        # Prefer exact suffix matches with the longest tag-free condition part last.
        matches = sorted(matches, key=lambda value: (len(value), value))
    return matches[-1] if matches else None


def load_metrics(results_dir, condition, input_tag=None):
    by_file = defaultdict(dict)
    metric_paths = {}
    for metric in METRICS:
        path = discover_metric_csv(results_dir, metric, condition, input_tag)
        metric_paths[metric] = path
        for row in read_csv(path):
            file_name = row.get("file") or Path(row.get("generated_file", "")).name
            if not file_name:
                continue
            by_file[file_name][metric] = row
    return by_file, metric_paths


def rel_src(path, report_path):
    if not path:
        return ""
    report_dir = Path(report_path).resolve().parent
    resolved = Path(path).expanduser().resolve()
    try:
        return os.path.relpath(resolved, report_dir)
    except ValueError:
        return resolved.as_uri()


def fmt(value, default=""):
    if value is None or value == "":
        return default
    return str(value)


def metric_cells(metrics):
    emotion = metrics.get("emotion", {})
    novelty = metrics.get("novelty", {})
    wer = metrics.get("wer", {})
    mos = metrics.get("mos", {})
    pred = emotion.get("predicted", "")
    target = emotion.get("target", "")
    match = emotion.get("match", "")
    novelty_gain = novelty.get("novelty_gain_vs_baseline", "")
    wer_value = wer.get("wer", "")
    mos_value = mos.get("mos", "")
    mos_delta = mos.get("delta_vs_baseline", "")
    return {
        "predicted": pred,
        "target": target,
        "match": match,
        "novelty": novelty_gain,
        "wer": wer_value,
        "mos": mos_value,
        "mos_delta": mos_delta,
    }


def style_sort_key(row):
    style = row.get("style", "")
    try:
        return (STYLE_ORDER.index(style), style)
    except ValueError:
        return (len(STYLE_ORDER), style)


def write_rating_template(path, rows):
    fields = [
        "source_stem",
        "style",
        "gender_control",
        "age_control",
        "audio_file",
        "emotion_match_1_5",
        "naturalness_1_5",
        "intelligibility_1_5",
        "identity_shift_1_5",
        "notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if (
                row.get("style") == "baseline"
                and not row.get("gender_control")
                and not row.get("age_control")
            ):
                continue
            writer.writerow(
                {
                    "source_stem": row.get("source_stem", ""),
                    "style": row.get("style", ""),
                    "gender_control": row.get("gender_control", ""),
                    "age_control": row.get("age_control", ""),
                    "audio_file": row.get("output_file", ""),
                    "emotion_match_1_5": "",
                    "naturalness_1_5": "",
                    "intelligibility_1_5": "",
                    "identity_shift_1_5": "",
                    "notes": "",
                }
            )


def write_html(path, rows, metrics_by_file, metric_paths, title, rating_template):
    condition = rows[0].get("condition", "unknown_condition")
    has_metadata_controls = any(
        row.get("gender_control") or row.get("age_control") for row in rows
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("source_stem", "unknown")].append(row)

    lines = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        ":root { color-scheme: light; --ink:#172016; --muted:#66705f; --line:#d7ddcc; --bg:#f7f4ea; --card:#fffdf6; --accent:#315c3a; }",
        "body { margin:0; font:15px/1.45 Georgia, 'Times New Roman', serif; color:var(--ink); background:linear-gradient(120deg,#f7f4ea,#edf3df); }",
        "main { max-width:1180px; margin:0 auto; padding:32px 20px 60px; }",
        "header { border-bottom:2px solid var(--ink); padding-bottom:18px; margin-bottom:22px; }",
        "h1 { font-size:34px; margin:0 0 8px; letter-spacing:-0.03em; }",
        "h2 { font-size:22px; margin:34px 0 10px; }",
        ".meta { color:var(--muted); }",
        ".panel { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:16px; box-shadow:0 8px 24px rgba(42,51,31,.08); }",
        "table { width:100%; border-collapse:collapse; margin-top:10px; }",
        "th,td { border-bottom:1px solid var(--line); padding:8px; vertical-align:top; text-align:left; }",
        "th { font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); }",
        "audio { width:210px; max-width:100%; }",
        ".style { font-weight:700; color:var(--accent); }",
        ".checklist li { margin:5px 0; }",
        ".small { font-size:13px; color:var(--muted); }",
        "@media (max-width:760px) { table,thead,tbody,tr,th,td { display:block; } th { display:none; } td { border-bottom:0; padding:5px 0; } tr { border-bottom:1px solid var(--line); padding:10px 0; } audio { width:100%; } }",
        "</style>",
        "</head>",
        "<body>",
        "<main>",
        "<header>",
        f"<h1>{html.escape(title)}</h1>",
        f"<div class=\"meta\">Condition: <code>{html.escape(condition)}</code> · rows: {len(rows)} · sources: {len(grouped)}</div>",
        "</header>",
        "<section class=\"panel\">",
        "<h2>How To Listen</h2>",
        "<ol class=\"checklist\">",
        "<li>For each speaker, play the source and baseline first.</li>",
        "<li>Then compare each styled output against the baseline for target emotion, intelligibility, naturalness, and identity shift.</li>",
        "<li>If age/gender controls are present, judge them perceptually and do not assume the requested label was achieved.</li>",
        "<li>Use headphones if possible; whisper/confused often trade novelty for quality.</li>",
        "<li>Record subjective scores in the rating template CSV linked below.</li>",
        "</ol>",
        f"<p class=\"small\">Rating template: <code>{html.escape(str(rating_template))}</code></p>",
        "<p class=\"small\">Metric CSVs: "
        + ", ".join(
            f"{name}=<code>{html.escape(str(path or 'missing'))}</code>"
            for name, path in metric_paths.items()
        )
        + "</p>",
        "</section>",
    ]

    for source, source_rows in sorted(grouped.items()):
        source_rows = sorted(source_rows, key=style_sort_key)
        source_file = source_rows[0].get("source_file")
        metadata_header = "<th>Gender</th><th>Age</th>" if has_metadata_controls else ""
        lines.extend([
            f"<h2>{html.escape(source)}</h2>",
            "<div class=\"panel\">",
            "<div><strong>Source</strong></div>",
            f"<audio controls preload=\"none\" src=\"{html.escape(rel_src(source_file, path))}\"></audio>",
            "<table>",
            (
                "<thead><tr><th>Style</th>"
                + metadata_header
                + "<th>Audio</th><th>Predicted</th><th>Target</th><th>Match</th>"
                + "<th>Novelty</th><th>WER</th><th>MOS</th><th>MOS Δ</th></tr></thead>"
            ),
            "<tbody>",
        ])
        for row in source_rows:
            output_file = row.get("output_file", "")
            file_name = Path(output_file).name
            cells = metric_cells(metrics_by_file.get(file_name, {}))
            metadata_cells = ""
            if has_metadata_controls:
                metadata_cells = (
                    f"<td>{html.escape(fmt(row.get('gender_control')))}</td>"
                    f"<td>{html.escape(fmt(row.get('age_control')))}</td>"
                )
            lines.append(
                "<tr>"
                f"<td class=\"style\">{html.escape(row.get('style', ''))}</td>"
                + metadata_cells
                + f"<td><audio controls preload=\"none\" src=\"{html.escape(rel_src(output_file, path))}\"></audio><div class=\"small\">{html.escape(file_name)}</div></td>"
                + f"<td>{html.escape(fmt(cells['predicted']))}</td>"
                + f"<td>{html.escape(fmt(cells['target']))}</td>"
                + f"<td>{html.escape(fmt(cells['match']))}</td>"
                + f"<td>{html.escape(fmt(cells['novelty']))}</td>"
                + f"<td>{html.escape(fmt(cells['wer']))}</td>"
                + f"<td>{html.escape(fmt(cells['mos']))}</td>"
                + f"<td>{html.escape(fmt(cells['mos_delta']))}</td>"
                + "</tr>"
            )
        lines.extend(["</tbody>", "</table>", "</div>"])

    lines.extend(["</main>", "</body>", "</html>"])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    rows = read_manifest(args.manifest)
    condition = rows[0].get("condition", "unknown_condition")
    include_styles = [item.strip() for item in args.styles.split(",") if item.strip()]
    if include_styles:
        allowed = set(include_styles) | {"baseline"}
        rows = [row for row in rows if row.get("style") in allowed]
    if args.max_sources is not None:
        seen = []
        keep = set()
        for row in rows:
            source = row.get("source_stem", "")
            if source not in keep:
                if len(seen) >= args.max_sources:
                    continue
                seen.append(source)
                keep.add(source)
        rows = [row for row in rows if row.get("source_stem", "") in keep]

    out = args.out or str(Path(args.results_dir) / f"listening_{condition}.html")
    rating_template = args.rating_template or str(Path(out).with_name(Path(out).stem + "_ratings.csv"))
    title = args.title or f"Listening Report: {condition}"

    metrics_by_file, metric_paths = load_metrics(args.results_dir, condition, args.input_tag)
    write_rating_template(rating_template, rows)
    write_html(out, rows, metrics_by_file, metric_paths, title, rating_template)
    print(f"Wrote listening report to {out}")
    print(f"Wrote rating template to {rating_template}")


if __name__ == "__main__":
    main()
