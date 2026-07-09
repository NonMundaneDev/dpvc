"""Acoustic diagnostic for gender-control follow-up review bundles.

This is a lightweight sanity check, not a perceptual gender classifier. It
estimates simple audio proxies such as median F0 and spectral centroid for the
baseline, female-control, and male-control outputs in a generation manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.io import wavfile


@dataclass(frozen=True)
class AudioStats:
    path: str
    sample_rate: int
    duration_sec: float
    rms_dbfs: float
    median_f0_hz: float | None
    voiced_frame_rate: float
    spectral_centroid_hz: float | None


def load_wav_mono(path: str | Path) -> tuple[int, np.ndarray]:
    sample_rate, audio = wavfile.read(path)
    data = np.asarray(audio)
    if data.ndim == 2:
        data = data.mean(axis=1)

    if np.issubdtype(data.dtype, np.integer):
        max_value = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_value
    else:
        data = data.astype(np.float32)

    return int(sample_rate), data


def estimate_f0_values(
    audio: np.ndarray,
    sample_rate: int,
    *,
    fmin: float = 60.0,
    fmax: float = 400.0,
    frame_ms: float = 40.0,
    hop_ms: float = 10.0,
    min_corr: float = 0.32,
) -> tuple[list[float], int]:
    frame_len = max(1, int(sample_rate * frame_ms / 1000.0))
    hop_len = max(1, int(sample_rate * hop_ms / 1000.0))
    if len(audio) < frame_len:
        return [], 0

    min_lag = max(1, int(sample_rate / fmax))
    max_lag = max(min_lag + 1, int(sample_rate / fmin))
    window = np.hanning(frame_len).astype(np.float32)
    rms_floor = max(float(np.sqrt(np.mean(np.square(audio))) * 0.05), 1e-4)

    f0_values: list[float] = []
    total_frames = 0
    for start in range(0, len(audio) - frame_len + 1, hop_len):
        frame = audio[start : start + frame_len].astype(np.float32)
        frame = frame - float(np.mean(frame))
        rms = float(np.sqrt(np.mean(np.square(frame))))
        if rms < rms_floor:
            continue
        total_frames += 1

        corr = np.correlate(frame * window, frame * window, mode="full")[frame_len - 1 :]
        if corr[0] <= 0.0:
            continue
        corr = corr / corr[0]
        search = corr[min_lag : min(max_lag, len(corr))]
        if len(search) == 0:
            continue
        peak_offset = int(np.argmax(search))
        peak_value = float(search[peak_offset])
        if peak_value < min_corr:
            continue
        lag = min_lag + peak_offset
        f0_values.append(float(sample_rate / lag))

    return f0_values, total_frames


def spectral_centroid(audio: np.ndarray, sample_rate: int) -> float | None:
    if len(audio) == 0:
        return None
    centered = audio.astype(np.float32) - float(np.mean(audio))
    spectrum = np.abs(np.fft.rfft(centered))
    total = float(np.sum(spectrum))
    if total <= 0.0:
        return None
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate)
    return float(np.sum(freqs * spectrum) / total)


def analyze_file(path: str | Path) -> AudioStats:
    sample_rate, audio = load_wav_mono(path)
    duration = float(len(audio) / sample_rate) if sample_rate else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-12))
    f0_values, voiced_candidate_frames = estimate_f0_values(audio, sample_rate)
    median_f0 = float(statistics.median(f0_values)) if f0_values else None
    voiced_rate = float(len(f0_values) / voiced_candidate_frames) if voiced_candidate_frames else 0.0
    return AudioStats(
        path=str(Path(path).resolve()),
        sample_rate=sample_rate,
        duration_sec=duration,
        rms_dbfs=rms_dbfs,
        median_f0_hz=median_f0,
        voiced_frame_rate=voiced_rate,
        spectral_centroid_hz=spectral_centroid(audio, sample_rate),
    )


def condition_label(record: dict) -> str:
    control = record.get("gender_control")
    if control in {"female", "male"}:
        return str(control)
    return "baseline"


def load_manifest(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rows.append(
                {
                    "source_stem": record.get("source_stem") or Path(record["source_file"]).stem,
                    "condition": condition_label(record),
                    "source_file": record["source_file"],
                    "generated_file": record.get("output_file", record.get("generated_file")),
                    "gender_control_dim": record.get("gender_control_dim"),
                    "gender_control_value": record.get("gender_control_value"),
                }
            )
    return rows


def clean_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def write_csv(path: str | Path, rows: Sequence[dict]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def median(values: Sequence[float]) -> float | None:
    cleaned = [value for value in values if value is not None]
    return float(statistics.median(cleaned)) if cleaned else None


def write_summary(path: str | Path, *, manifest: Path, pair_rows: Sequence[dict], detail_rows: Sequence[dict]) -> None:
    complete_pairs = [
        row
        for row in pair_rows
        if row["female_median_f0_hz"] and row["male_median_f0_hz"]
    ]
    female_gt_male = [
        row
        for row in complete_pairs
        if float(row["female_median_f0_hz"]) > float(row["male_median_f0_hz"])
    ]
    f0_pair_deltas = [
        float(row["female_minus_male_f0_hz"])
        for row in complete_pairs
        if row["female_minus_male_f0_hz"]
    ]
    centroid_pair_deltas = [
        float(row["female_minus_male_centroid_hz"])
        for row in pair_rows
        if row["female_minus_male_centroid_hz"]
    ]

    lines = [
        "# Gender Follow-Up Acoustic Diagnostic",
        "",
        f"- Manifest: `{manifest.resolve()}`",
        f"- Audio rows: `{len(detail_rows)}`",
        f"- Source groups: `{len(pair_rows)}`",
        f"- Complete female/male F0 pairs: `{len(complete_pairs)}`",
    ]
    if complete_pairs:
        lines.extend(
            [
                f"- Pairs where female-control median F0 > male-control median F0: `{len(female_gt_male)}/{len(complete_pairs)}`",
                f"- Median female-minus-male F0 delta: `{median(f0_pair_deltas):.2f} Hz`",
            ]
        )
    if centroid_pair_deltas:
        lines.append(
            f"- Median female-minus-male spectral-centroid delta: `{median(centroid_pair_deltas):.2f} Hz`"
        )

    lines.extend(
        [
            "",
            "## Source-Level F0 Pairs",
            "",
            "| Source | Baseline F0 | Female-control F0 | Male-control F0 | Female - male |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in pair_rows:
        lines.append(
            "| "
            f"`{row['source_stem']}` | "
            f"`{row['baseline_median_f0_hz'] or 'n/a'}` | "
            f"`{row['female_median_f0_hz'] or 'n/a'}` | "
            f"`{row['male_median_f0_hz'] or 'n/a'}` | "
            f"`{row['female_minus_male_f0_hz'] or 'n/a'}` |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a diagnostic acoustic proxy, not a perceptual gender-control result.",
            "- A listener-clear gender control would usually be expected to move simple cues such as F0 in a consistent direction, but F0 alone is not sufficient for gender perception.",
            "- Use this only to characterize the failed listening gate, not to promote the gender checkpoint.",
        ]
    )

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_rows(records: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    stats_cache: dict[str, AudioStats] = {}
    detail_rows: list[dict] = []
    by_source: dict[str, dict[str, dict]] = {}

    for record in records:
        generated_file = str(Path(record["generated_file"]).expanduser().resolve())
        stats = stats_cache.setdefault(generated_file, analyze_file(generated_file))
        detail_row = {
            "source_stem": record["source_stem"],
            "condition": record["condition"],
            "generated_file": generated_file,
            "duration_sec": clean_float(stats.duration_sec),
            "rms_dbfs": clean_float(stats.rms_dbfs),
            "median_f0_hz": clean_float(stats.median_f0_hz),
            "voiced_frame_rate": clean_float(stats.voiced_frame_rate),
            "spectral_centroid_hz": clean_float(stats.spectral_centroid_hz),
            "gender_control_dim": record.get("gender_control_dim") or "",
            "gender_control_value": record.get("gender_control_value") or "",
        }
        detail_rows.append(detail_row)
        by_source.setdefault(record["source_stem"], {})[record["condition"]] = detail_row

    pair_rows: list[dict] = []
    for source_stem in sorted(by_source):
        group = by_source[source_stem]
        baseline = group.get("baseline", {})
        female = group.get("female", {})
        male = group.get("male", {})
        baseline_f0 = baseline.get("median_f0_hz", "")
        female_f0 = female.get("median_f0_hz", "")
        male_f0 = male.get("median_f0_hz", "")
        female_centroid = female.get("spectral_centroid_hz", "")
        male_centroid = male.get("spectral_centroid_hz", "")

        f0_delta = ""
        if female_f0 and male_f0:
            f0_delta = clean_float(float(female_f0) - float(male_f0))

        centroid_delta = ""
        if female_centroid and male_centroid:
            centroid_delta = clean_float(float(female_centroid) - float(male_centroid))

        pair_rows.append(
            {
                "source_stem": source_stem,
                "baseline_median_f0_hz": baseline_f0,
                "female_median_f0_hz": female_f0,
                "male_median_f0_hz": male_f0,
                "female_minus_male_f0_hz": f0_delta,
                "baseline_spectral_centroid_hz": baseline.get("spectral_centroid_hz", ""),
                "female_spectral_centroid_hz": female_centroid,
                "male_spectral_centroid_hz": male_centroid,
                "female_minus_male_centroid_hz": centroid_delta,
            }
        )

    return detail_rows, pair_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--pair-csv", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    records = load_manifest(manifest)
    detail_rows, pair_rows = build_rows(records)
    write_csv(args.out_csv, detail_rows)
    write_csv(args.pair_csv, pair_rows)
    write_summary(args.summary_out, manifest=manifest, pair_rows=pair_rows, detail_rows=detail_rows)


if __name__ == "__main__":
    main()
