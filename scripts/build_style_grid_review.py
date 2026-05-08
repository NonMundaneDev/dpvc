#!/usr/bin/env python3
"""
Build an A/B listening review for style-strength grid candidates.

The report compares a reference readout against one candidate per style, using
matched source speakers. It is intended for perceptual review before promoting
style-specific strength presets.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path


METRICS = ("emotion", "novelty", "wer", "mos")
DEFAULT_CANDIDATES = [
    "anger=output/mixed_teacher_cvrare_strength_grid/mixed_teacher_cvrare_strength_grid_anger_s10/generation_manifest.jsonl:mixed_teacher_cvrare_strength_grid_anger_s10",
    "disgust=output/mixed_teacher_cvrare_strength_grid/mixed_teacher_cvrare_strength_grid_disgust_s10/generation_manifest.jsonl:mixed_teacher_cvrare_strength_grid_disgust_s10",
    "fear=output/mixed_teacher_cvrare_strength_grid/mixed_teacher_cvrare_strength_grid_fear_s7p5/generation_manifest.jsonl:mixed_teacher_cvrare_strength_grid_fear_s7p5",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--reference-manifest",
        default=(
            "output/mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard_eval/"
            "generation_manifest.jsonl"
        ),
        help="Reference generation manifest.",
    )
    ap.add_argument(
        "--reference-tag",
        default="mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard",
        help="Reference result tag used in eval CSV filenames.",
    )
    ap.add_argument("--reference-input-tag", default="mixed_teacher")
    ap.add_argument(
        "--candidate",
        action="append",
        default=None,
        help=(
            "Candidate spec: style=manifest:result_tag. May be repeated. "
            "Defaults to anger_s10, disgust_s10, and fear_s7p5."
        ),
    )
    ap.add_argument("--candidate-input-tag", default="mixed_teacher_strength_grid")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument(
        "--out",
        default="results/listening_mixed_teacher_cvrare_strength_grid_ab_review.html",
    )
    ap.add_argument(
        "--rating-template",
        default="results/listening_mixed_teacher_cvrare_strength_grid_ab_review_ratings.csv",
    )
    ap.add_argument(
        "--priority-csv",
        default=None,
        help=(
            "Optional triage CSV from summarize_style_grid_review.py. "
            "When set, only matching style/source_stem rows are included."
        ),
    )
    ap.add_argument(
        "--max-priority",
        type=int,
        default=None,
        help="Optional maximum priority rank to keep from --priority-csv.",
    )
    ap.add_argument(
        "--title",
        default="Generated-Audio Strength Grid A/B Review",
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
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric_paths(results_dir, input_tag, result_tag):
    return {
        metric: Path(results_dir) / f"eval_{metric}_{input_tag}_{result_tag}.csv"
        for metric in METRICS
    }


def load_metrics(results_dir, input_tag, result_tag):
    by_file = {}
    paths = metric_paths(results_dir, input_tag, result_tag)
    for metric, path in paths.items():
        for row in read_csv(path):
            file_name = row.get("file") or Path(row.get("generated_file", "")).name
            if not file_name:
                continue
            by_file.setdefault(file_name, {})[metric] = row
    return by_file, paths


def load_priority_keys(path, max_priority=None):
    keys = {}
    for order, row in enumerate(read_csv(path)):
        if max_priority is not None:
            try:
                priority = int(row.get("priority", ""))
            except ValueError:
                continue
            if priority > max_priority:
                continue
        else:
            try:
                priority = int(row.get("priority", ""))
            except ValueError:
                priority = order + 1
        style = (row.get("style") or "").strip()
        source = (row.get("source_stem") or "").strip()
        if style and source:
            keys[(style, source)] = (priority, order)
    if not keys:
        limit = f" with priority <= {max_priority}" if max_priority is not None else ""
        raise ValueError(f"No priority rows found in {path}{limit}")
    return keys


def parse_candidate_spec(raw):
    if "=" not in raw or ":" not in raw:
        raise ValueError(
            f"Invalid candidate spec {raw!r}; expected style=manifest:result_tag"
        )
    style, rest = raw.split("=", 1)
    manifest, result_tag = rest.rsplit(":", 1)
    style = style.strip()
    if not style:
        raise ValueError(f"Invalid candidate spec {raw!r}; style is empty")
    return style, Path(manifest), result_tag.strip()


def index_rows(rows):
    indexed = {}
    for row in rows:
        indexed[(row.get("source_stem"), row.get("style"))] = row
    return indexed


def rel_src(path, report_path):
    if not path:
        return ""
    report_dir = Path(report_path).resolve().parent
    resolved = Path(path).expanduser().resolve()
    try:
        return os.path.relpath(resolved, report_dir)
    except ValueError:
        return resolved.as_uri()


def metric_summary(metrics):
    emotion = metrics.get("emotion", {})
    novelty = metrics.get("novelty", {})
    wer = metrics.get("wer", {})
    mos = metrics.get("mos", {})
    return {
        "predicted": emotion.get("predicted", ""),
        "match": emotion.get("match", ""),
        "score": emotion.get("score", ""),
        "novelty": novelty.get("novelty_gain_vs_baseline", ""),
        "wer": wer.get("wer", ""),
        "mos_delta": mos.get("delta_vs_baseline", ""),
    }


def fmt(value):
    return "" if value is None else str(value)


def render_audio(path, report_path):
    if not path:
        return "<span class=\"missing\">missing</span>"
    name = Path(path).name
    return (
        f"<audio controls preload=\"none\" src=\"{html.escape(rel_src(path, report_path))}\"></audio>"
        f"<div class=\"small\">{html.escape(name)}</div>"
    )


def metric_table(cells):
    rows = [
        ("pred", cells["predicted"]),
        ("match", cells["match"]),
        ("score", cells["score"]),
        ("novelty", cells["novelty"]),
        ("WER", cells["wer"]),
        ("MOS Δ", cells["mos_delta"]),
    ]
    return "<dl>" + "".join(
        f"<dt>{html.escape(label)}</dt><dd>{html.escape(fmt(value))}</dd>"
        for label, value in rows
    ) + "</dl>"


def build_pairs(reference_rows, candidate_specs, results_dir, reference_input_tag, reference_tag, candidate_input_tag):
    reference_index = index_rows(reference_rows)
    reference_metrics, reference_metric_paths = load_metrics(results_dir, reference_input_tag, reference_tag)
    pairs = []
    candidate_metric_paths = {}

    for style, manifest, result_tag in candidate_specs:
        candidate_rows = read_manifest(manifest)
        candidate_index = index_rows(candidate_rows)
        candidate_metrics, paths = load_metrics(results_dir, candidate_input_tag, result_tag)
        candidate_metric_paths[result_tag] = paths
        sources = sorted(
            source
            for source, row_style in reference_index
            if row_style == style and (source, style) in candidate_index
        )
        for source in sources:
            reference_row = reference_index[(source, style)]
            candidate_row = candidate_index[(source, style)]
            baseline_row = reference_index.get((source, "baseline"))
            pairs.append(
                {
                    "style": style,
                    "source": source,
                    "source_file": reference_row.get("source_file", ""),
                    "baseline_file": baseline_row.get("output_file", "") if baseline_row else "",
                    "reference_tag": reference_tag,
                    "reference_file": reference_row.get("output_file", ""),
                    "reference_strength": reference_row.get("style_strength", ""),
                    "reference_metrics": metric_summary(
                        reference_metrics.get(Path(reference_row.get("output_file", "")).name, {})
                    ),
                    "candidate_tag": result_tag,
                    "candidate_file": candidate_row.get("output_file", ""),
                    "candidate_strength": candidate_row.get("style_strength", ""),
                    "candidate_metrics": metric_summary(
                        candidate_metrics.get(Path(candidate_row.get("output_file", "")).name, {})
                    ),
                }
            )
    return pairs, reference_metric_paths, candidate_metric_paths


def write_rating_template(path, pairs):
    fields = [
        "source_stem",
        "style",
        "reference_tag",
        "candidate_tag",
        "reference_audio",
        "candidate_audio",
        "prefer_reference_or_candidate",
        "reference_target_match_1_5",
        "candidate_target_match_1_5",
        "reference_intelligibility_1_5",
        "candidate_intelligibility_1_5",
        "reference_naturalness_1_5",
        "candidate_naturalness_1_5",
        "reference_identity_shift_1_5",
        "candidate_identity_shift_1_5",
        "notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    "source_stem": pair["source"],
                    "style": pair["style"],
                    "reference_tag": pair["reference_tag"],
                    "candidate_tag": pair["candidate_tag"],
                    "reference_audio": pair["reference_file"],
                    "candidate_audio": pair["candidate_file"],
                    "prefer_reference_or_candidate": "",
                    "reference_target_match_1_5": "",
                    "candidate_target_match_1_5": "",
                    "reference_intelligibility_1_5": "",
                    "candidate_intelligibility_1_5": "",
                    "reference_naturalness_1_5": "",
                    "candidate_naturalness_1_5": "",
                    "reference_identity_shift_1_5": "",
                    "candidate_identity_shift_1_5": "",
                    "notes": "",
                }
            )


def write_html(
    path,
    pairs,
    title,
    rating_template,
    reference_metric_paths,
    candidate_metric_paths,
    preserve_pair_order=False,
):
    if preserve_pair_order:
        grouped_items = [("Priority Rows", pairs)]
        styles = sorted({pair["style"] for pair in pairs})
    else:
        grouped = {}
        for pair in pairs:
            grouped.setdefault(pair["style"], []).append(pair)
        grouped_items = sorted(grouped.items())
        styles = sorted(grouped)

    lines = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        ":root { color-scheme: light; --ink:#181713; --muted:#726f64; --line:#ddd5c4; --bg:#faf6eb; --card:#fffdf7; --accent:#8a3f25; --good:#2f6846; --warn:#946f1c; }",
        "body { margin:0; font:15px/1.45 Georgia, 'Times New Roman', serif; background:radial-gradient(circle at 10% 0%, #fff5dd 0, transparent 34%), linear-gradient(135deg,#faf6eb,#eef2df); color:var(--ink); }",
        "main { max-width:1240px; margin:0 auto; padding:34px 20px 64px; }",
        "header { border-bottom:2px solid var(--ink); margin-bottom:20px; padding-bottom:18px; }",
        "h1 { font-size:36px; letter-spacing:-0.035em; margin:0 0 8px; }",
        "h2 { font-size:24px; margin:34px 0 12px; }",
        ".panel { background:var(--card); border:1px solid var(--line); border-radius:18px; padding:16px; box-shadow:0 12px 30px rgba(42,33,21,.08); }",
        ".small, .meta { color:var(--muted); font-size:13px; }",
        "table { width:100%; border-collapse:collapse; margin-top:12px; }",
        "th, td { border-bottom:1px solid var(--line); padding:9px; vertical-align:top; text-align:left; }",
        "th { font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); }",
        "audio { width:190px; max-width:100%; }",
        "dl { display:grid; grid-template-columns:auto 1fr; gap:2px 8px; margin:0; font-size:13px; }",
        "dt { color:var(--muted); }",
        "dd { margin:0; font-variant-numeric:tabular-nums; }",
        ".style { color:var(--accent); font-weight:700; }",
        ".missing { color:var(--warn); font-weight:700; }",
        ".steps li { margin:5px 0; }",
        "@media (max-width:820px) { table,thead,tbody,tr,th,td { display:block; } th { display:none; } tr { border-bottom:1px solid var(--line); padding:12px 0; } td { border:0; padding:5px 0; } audio { width:100%; } }",
        "</style>",
        "</head>",
        "<body>",
        "<main>",
        "<header>",
        f"<h1>{html.escape(title)}</h1>",
        f"<div class=\"meta\">Pairs: {len(pairs)} · Styles: {', '.join(html.escape(style) for style in styles)}</div>",
        "</header>",
        "<section class=\"panel\">",
        "<h2>Review Protocol</h2>",
        "<ol class=\"steps\">",
        "<li>For each row, listen to the source and baseline first.</li>",
        "<li>Compare the reference guard against the candidate without looking only at the objective metrics.</li>",
        "<li>Prefer the candidate only if target style is audibly stronger and speech remains intelligible/natural.</li>",
        "<li>Record scores in the rating CSV; the candidate should not become a preset without perceptual support.</li>",
        "</ol>",
        f"<p class=\"small\">Rating template: <code>{html.escape(str(rating_template))}</code></p>",
        "<p class=\"small\">Reference metric CSVs: "
        + ", ".join(
            f"{name}=<code>{html.escape(str(path))}</code>"
            for name, path in reference_metric_paths.items()
        )
        + "</p>",
        "<p class=\"small\">Candidate metric CSVs: "
        + "; ".join(
            f"{html.escape(tag)} ["
            + ", ".join(
                f"{name}=<code>{html.escape(str(path))}</code>"
                for name, path in paths.items()
            )
            + "]"
            for tag, paths in candidate_metric_paths.items()
        )
        + "</p>",
        "</section>",
    ]

    for style, style_pairs in grouped_items:
        lines.extend([
            f"<h2>{html.escape(style)}</h2>",
            "<div class=\"panel\">",
            "<table>",
            "<thead><tr><th>Style / Source</th><th>Source / Baseline</th><th>Reference Guard</th><th>Reference Metrics</th><th>Candidate</th><th>Candidate Metrics</th></tr></thead>",
            "<tbody>",
        ])
        for pair in style_pairs:
            lines.append(
                "<tr>"
                f"<td><span class=\"style\">{html.escape(pair['style'])}</span><div>{html.escape(pair['source'])}</div></td>"
                f"<td><div>Source</div>{render_audio(pair['source_file'], path)}<div>Baseline</div>{render_audio(pair['baseline_file'], path)}</td>"
                f"<td><div class=\"small\">{html.escape(pair['reference_tag'])} · strength {html.escape(fmt(pair['reference_strength']))}</div>{render_audio(pair['reference_file'], path)}</td>"
                f"<td>{metric_table(pair['reference_metrics'])}</td>"
                f"<td><div class=\"small\">{html.escape(pair['candidate_tag'])} · strength {html.escape(fmt(pair['candidate_strength']))}</div>{render_audio(pair['candidate_file'], path)}</td>"
                f"<td>{metric_table(pair['candidate_metrics'])}</td>"
                "</tr>"
            )
        lines.extend(["</tbody>", "</table>", "</div>"])

    lines.extend(["</main>", "</body>", "</html>"])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    candidate_specs = [
        parse_candidate_spec(raw)
        for raw in (args.candidate if args.candidate is not None else DEFAULT_CANDIDATES)
    ]
    reference_rows = read_manifest(args.reference_manifest)
    pairs, reference_metric_paths, candidate_metric_paths = build_pairs(
        reference_rows=reference_rows,
        candidate_specs=candidate_specs,
        results_dir=results_dir,
        reference_input_tag=args.reference_input_tag,
        reference_tag=args.reference_tag,
        candidate_input_tag=args.candidate_input_tag,
    )
    if args.priority_csv:
        priority_keys = load_priority_keys(args.priority_csv, args.max_priority)
        pairs = [
            pair
            for pair in pairs
            if (pair["style"], pair["source"]) in priority_keys
        ]
        pairs.sort(key=lambda pair: priority_keys[(pair["style"], pair["source"])])
    if not pairs:
        raise SystemExit("No matched reference/candidate pairs found")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.rating_template).parent.mkdir(parents=True, exist_ok=True)
    write_rating_template(args.rating_template, pairs)
    write_html(
        path=args.out,
        pairs=pairs,
        title=args.title,
        rating_template=args.rating_template,
        reference_metric_paths=reference_metric_paths,
        candidate_metric_paths=candidate_metric_paths,
        preserve_pair_order=bool(args.priority_csv),
    )
    print(f"Wrote A/B listening review to {args.out}")
    print(f"Wrote rating template to {args.rating_template}")
    print(f"Matched pairs: {len(pairs)}")


if __name__ == "__main__":
    main()
