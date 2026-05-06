#!/usr/bin/env python3
"""
Run a generated-audio style-strength grid for targeted styles.

This orchestrates the existing inference and evaluation stack without changing
metric definitions:

  1. generate baseline + one target style for each style/strength pair
  2. run emotion, novelty, WER, MOS, listening report, and rating template
  3. write grid-level summary/ranking artifacts

The default grid targets the hard expanded-rare-supply styles that remain stuck
near neutral: anger, disgust, and fear.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_STYLES = "anger,disgust,fear"
DEFAULT_STRENGTHS = "3.0,5.0,7.5,10.0"


def parse_csv_list(raw):
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_float_list(raw):
    values = []
    for item in parse_csv_list(raw):
        value = float(item)
        if value < 0:
            raise ValueError(f"Strength must be non-negative: {item}")
        values.append(value)
    if not values:
        raise ValueError("At least one strength is required")
    return values


def strength_token(value):
    text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m")


def run(cmd, dry_run=False):
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def metric_paths(results_dir, input_tag, result_tag):
    return [
        results_dir / f"eval_{metric}_{input_tag}_{result_tag}.csv"
        for metric in ("emotion", "novelty", "wer", "mos")
    ]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    source_group = ap.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source", help="Single source audio file")
    source_group.add_argument("--source-dir", help="Directory of source audio files")
    ap.add_argument(
        "--condition",
        default="mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup",
        help="run_ablation_inference condition to sweep",
    )
    ap.add_argument("--styles", default=DEFAULT_STYLES)
    ap.add_argument("--strengths", default=DEFAULT_STRENGTHS)
    ap.add_argument(
        "--out-root",
        default="output/mixed_teacher_cvrare_strength_grid",
        help="Root directory for generated grid corpora",
    )
    ap.add_argument(
        "--result-prefix",
        default="mixed_teacher_cvrare_strength_grid",
        help="Prefix for per-grid result tags",
    )
    ap.add_argument(
        "--input-tag",
        default="mixed_teacher_strength_grid",
        help="Input tag for metric CSV grouping",
    )
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--style-strength-map", default=None)
    ap.add_argument("--noise-level", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--latent-dims", type=int, default=15)
    ap.add_argument("--force", action="store_true", help="Regenerate/evaluate even if outputs exist")
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-listening", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    styles = parse_csv_list(args.styles)
    if not styles:
        raise SystemExit("At least one style is required")
    strengths = parse_float_list(args.strengths)

    out_root = Path(args.out_root)
    results_dir = Path(args.results_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    result_tags = []

    for style in styles:
        for strength in strengths:
            tag = f"{args.result_prefix}_{style}_s{strength_token(strength)}"
            result_tags.append(tag)
            out_dir = out_root / tag
            manifest = out_dir / "generation_manifest.jsonl"

            if args.skip_generation:
                print(f"Skipping generation for {tag}")
            elif args.force or not manifest.exists():
                cmd = [
                    py,
                    "scripts/run_ablation_inference.py",
                    "--condition",
                    args.condition,
                    "--out",
                    out_dir,
                    "--styles",
                    style,
                    "--style-strength",
                    str(strength),
                    "--noise-level",
                    str(args.noise_level),
                    "--seed",
                    str(args.seed),
                    "--latent-dims",
                    str(args.latent_dims),
                ]
                if args.source:
                    cmd.extend(["--source", args.source])
                else:
                    cmd.extend(["--source-dir", args.source_dir])
                if args.style_strength_map:
                    cmd.extend(["--style-strength-map", args.style_strength_map])
                run(cmd, args.dry_run)
            else:
                print(f"Reusing existing manifest for {tag}: {manifest}")

            if args.skip_eval:
                print(f"Skipping evaluation for {tag}")
                continue

            expected_eval_outputs = metric_paths(results_dir, args.input_tag, tag)
            if not args.skip_listening:
                expected_eval_outputs.extend([
                    results_dir / f"listening_{tag}.html",
                    results_dir / f"listening_{tag}_ratings.csv",
                ])
            if args.force or not all(path.exists() for path in expected_eval_outputs):
                cmd = [
                    py,
                    "scripts/run_generated_audio_eval_suite.py",
                    "--input",
                    out_dir,
                    "--result-tag",
                    tag,
                    "--input-tag",
                    args.input_tag,
                    "--results-dir",
                    results_dir,
                    "--skip-summary",
                ]
                if args.skip_listening:
                    cmd.append("--skip-listening")
                run(cmd, args.dry_run)
            else:
                print(f"Reusing existing evaluation outputs for {tag}")

    if not args.skip_eval:
        summary_out = results_dir / f"eval_{args.input_tag}_summary.csv"
        collapse_out = results_dir / f"eval_{args.input_tag}_collapse.csv"
        run(
            [
                py,
                "scripts/summarize_mixed_teacher_results.py",
                "--results-dir",
                results_dir,
                "--input-tag",
                args.input_tag,
                "--summary-out",
                summary_out,
                "--collapse-out",
                collapse_out,
            ],
            args.dry_run,
        )
        run(
            [
                py,
                "scripts/summarize_style_strength_grid.py",
                "--summary",
                summary_out,
                "--condition-prefix",
                args.result_prefix,
                "--out-csv",
                results_dir / f"eval_{args.result_prefix}_ranking.csv",
                "--out-md",
                results_dir / f"eval_{args.result_prefix}_ranking.md",
            ],
            args.dry_run,
        )

    print("\nGrid result tags:")
    for tag in result_tags:
        print(f"- {tag}")


if __name__ == "__main__":
    main()
