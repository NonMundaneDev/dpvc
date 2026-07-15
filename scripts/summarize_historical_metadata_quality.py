#!/usr/bin/env python3
"""Summarize WER, MOS, and acoustic metadata-control recovery evidence."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


ENDPOINTS = {
    "gender": ("male", "female"),
    "age": ("teens", "sixties"),
}


def parse_quality(value):
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("quality inputs must use MODEL=WER_CSV=MOS_CSV")
    return parts[0], Path(parts[1]), Path(parts[2])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality",
        action="append",
        type=parse_quality,
        required=True,
        help="Evaluation input as MODEL=WER_CSV=MOS_CSV; may be repeated.",
    )
    parser.add_argument("--acoustic-summary", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-markdown", required=True)
    return parser.parse_args()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def condition_name(attribute, endpoint, strength):
    return f"{endpoint}_s{int(float(strength))}"


def mean(values):
    cleaned = [float(value) for value in values if value not in (None, "")]
    return statistics.fmean(cleaned) if cleaned else None


def fmt(value):
    return "" if value is None else f"{value:.4f}"


def summarize(quality_specs, acoustic_rows):
    acoustic_index = {
        (row["model"], row["attribute"], float(row["strength"])): row
        for row in acoustic_rows
    }
    output = []
    for model, wer_path, mos_path in quality_specs:
        wer_groups = defaultdict(list)
        mos_groups = defaultdict(list)
        for row in read_csv(wer_path):
            if row["wer"] != "":
                wer_groups[row["style"]].append(float(row["wer"]))
        for row in read_csv(mos_path):
            if row["delta_vs_baseline"] != "":
                mos_groups[row["style"]].append(float(row["delta_vs_baseline"]))

        for attribute, endpoints in ENDPOINTS.items():
            for strength in (1.0, 2.0):
                conditions = [condition_name(attribute, endpoint, strength) for endpoint in endpoints]
                wers = [value for condition in conditions for value in wer_groups[condition]]
                mos_deltas = [value for condition in conditions for value in mos_groups[condition]]
                acoustic = acoustic_index[(model, attribute, strength)]
                output.append(
                    {
                        "model": model,
                        "attribute": attribute,
                        "strength": strength,
                        "endpoint_rows": len(wers),
                        "mean_wer": fmt(mean(wers)),
                        "median_wer": fmt(statistics.median(wers) if wers else None),
                        "wer_ge_0_8_count": sum(value >= 0.8 for value in wers),
                        "mean_mos_delta": fmt(mean(mos_deltas)),
                        "median_positive_minus_negative_f0_hz": acoustic[
                            "median_positive_minus_negative_f0_hz"
                        ],
                        "positive_endpoint_higher_f0": acoustic[
                            "positive_endpoint_higher_f0"
                        ],
                        "acoustic_pairs": acoustic["pairs"],
                    }
                )
    return output


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    lines = [
        "# Historical Metadata Recovery: Quality Summary",
        "",
        "WER is measured against each model's same-source baseline with Whisper `base`.",
        "MOS delta uses SQUIM_SUBJECTIVE against the same baseline.",
        "",
        "| Model | Attribute | Strength | F0 direction | Median F0 delta | Mean WER | Median WER | WER >= 0.8 | Mean MOS delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['model']}` | `{row['attribute']}` | {row['strength']:.0f} | "
            f"{row['positive_endpoint_higher_f0']}/{row['acoustic_pairs']} | "
            f"{row['median_positive_minus_negative_f0_hz']} Hz | "
            f"{row['mean_wer']} | {row['median_wer']} | "
            f"{row['wer_ge_0_8_count']}/{row['endpoint_rows']} | {row['mean_mos_delta']} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `joe_age_gender_v1` has the strongest gender-direction acoustic movement, including at the trained `1` endpoint.",
            "- Strength `2` is an extrapolation beyond the training labels. It is useful for diagnosis but needs a stricter perceptual and intelligibility gate.",
            "- The quality metrics characterize preservation only. They do not establish that listeners hear the intended age or gender.",
            "- The listening dashboard remains the promotion gate for any paper-facing metadata-control claim.",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    rows = summarize(args.quality, read_csv(args.acoustic_summary))
    write_csv(args.out_csv, rows)
    write_markdown(args.out_markdown, rows)
    print(f"Wrote {len(rows)} aggregate rows to {args.out_csv}")
    print(f"Wrote summary to {args.out_markdown}")


if __name__ == "__main__":
    main()
