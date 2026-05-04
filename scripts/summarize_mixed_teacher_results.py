"""
Summarize mixed-data pseudo-label teacher results.

This is a thin convenience wrapper around summarize_mixed_data_results.py that
uses the dedicated `mixed_teacher` tag by default.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_SUMMARY = "results/eval_mixed_teacher_summary.csv"
DEFAULT_COLLAPSE = "results/eval_mixed_teacher_collapse.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--input-tag', default='mixed_teacher')
    ap.add_argument('--summary-out', default=DEFAULT_SUMMARY)
    ap.add_argument('--collapse-out', default=DEFAULT_COLLAPSE)
    args = ap.parse_args()

    target = Path(__file__).with_name('summarize_mixed_data_results.py')
    cmd = [
        sys.executable,
        str(target),
        '--results-dir',
        args.results_dir,
        '--input-tag',
        args.input_tag,
        '--summary-out',
        args.summary_out,
        '--collapse-out',
        args.collapse_out,
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
