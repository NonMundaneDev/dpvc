#!/usr/bin/env python3
"""
Run the standard generated-audio evaluation suite for one corpus.

The suite writes emotion, novelty, WER, MOS, optional summary/collapse tables,
and a browser-playable listening report with stable names:

  results/eval_<metric>_<input_tag>_<result_tag>.csv
  results/listening_<result_tag>.html

Use this after `scripts/run_ablation_inference.py` so every generated-audio
result is evaluated and listenable through the same path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Generated audio directory")
    ap.add_argument(
        "--manifest",
        default=None,
        help="Generation manifest. Defaults to <input>/generation_manifest.jsonl",
    )
    ap.add_argument(
        "--result-tag",
        required=True,
        help="Result tag used in output filenames, e.g. mixed_teacher_cvrare_profile",
    )
    ap.add_argument(
        "--input-tag",
        default="mixed_teacher",
        help="Grouping tag used by summary scripts (default: mixed_teacher)",
    )
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--skip-summary", action="store_true")
    ap.add_argument("--skip-listening", action="store_true")
    return ap.parse_args()


def run(cmd):
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise SystemExit(f"Generated audio directory not found: {input_dir}")

    manifest = Path(args.manifest) if args.manifest else input_dir / "generation_manifest.jsonl"
    if not manifest.is_file():
        raise SystemExit(f"Generation manifest not found: {manifest}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "emotion": results_dir / f"eval_emotion_{args.input_tag}_{args.result_tag}.csv",
        "novelty": results_dir / f"eval_novelty_{args.input_tag}_{args.result_tag}.csv",
        "wer": results_dir / f"eval_wer_{args.input_tag}_{args.result_tag}.csv",
        "mos": results_dir / f"eval_mos_{args.input_tag}_{args.result_tag}.csv",
    }

    py = sys.executable
    run([py, "examples/eval_emotion.py", "--input", input_dir, "--out", outputs["emotion"]])
    run([py, "examples/eval_novelty.py", "--manifest", manifest, "--out", outputs["novelty"]])
    run([py, "examples/eval_wer.py", "--input", input_dir, "--out", outputs["wer"]])
    run([py, "examples/eval_mos.py", "--input", input_dir, "--out", outputs["mos"]])

    if not args.skip_summary:
        run([py, "scripts/summarize_mixed_teacher_results.py", "--input-tag", args.input_tag])

    if not args.skip_listening:
        listening_out = results_dir / f"listening_{args.result_tag}.html"
        run([
            py,
            "scripts/build_listening_report.py",
            "--manifest",
            manifest,
            "--input-tag",
            args.input_tag,
            "--out",
            listening_out,
        ])

    print("\nEvaluation outputs:")
    for path in outputs.values():
        print(f"- {path}")
    if not args.skip_listening:
        print(f"- {results_dir / f'listening_{args.result_tag}.html'}")


if __name__ == "__main__":
    main()
