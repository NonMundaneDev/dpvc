#!/usr/bin/env python3
"""
Build a CommonVoice gender-followup embedding artifact from a preflight speaker manifest.

The preflight manifest is the data contract for the gender-only follow-up. This
script intentionally fails if manifest clips are missing from the source
embedding artifact unless --allow-missing is provided, so a partial artifact is
never created by accident.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


DEFAULT_COMMONVOICE = "embeddings/openvoice_commonvoice_cvrare_expanded_emb.pt"
DEFAULT_MANIFEST = "results/commonvoice_gender_followup_speakers.csv"
DEFAULT_OUTPUT = "embeddings/openvoice_commonvoice_gender_followup_available_emb.pt"
DEFAULT_REPORT_JSON = "results/commonvoice_gender_followup_artifact.json"
DEFAULT_REPORT_MD = "results/commonvoice_gender_followup_artifact.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commonvoice-embeddings",
        default=DEFAULT_COMMONVOICE,
        help="Source CommonVoice embedding artifact to subset",
    )
    parser.add_argument(
        "--speaker-manifest",
        default=DEFAULT_MANIFEST,
        help="CSV emitted by scripts/preflight_commonvoice_gender_followup.py",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Subset artifact path")
    parser.add_argument("--report-json", default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", default=DEFAULT_REPORT_MD)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Create an available-subset artifact even when some manifest clips "
            "are not present in the source embedding artifact"
        ),
    )
    return parser.parse_args()


def clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def split_semicolon(value: object) -> List[str]:
    return [item.strip() for item in clean(value).split(";") if item.strip()]


def read_speaker_manifest(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"gender", "speaker_id", "clip_paths"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Speaker manifest is missing required columns: {sorted(missing)}")
        for row in reader:
            gender = clean(row.get("gender"))
            speaker_id = clean(row.get("speaker_id"))
            for clip_path in split_semicolon(row.get("clip_paths")):
                rows.append(
                    {
                        "gender": gender,
                        "speaker_id": speaker_id,
                        "clip_path": clip_path,
                        "clip_name": Path(clip_path).name,
                    }
                )
    if not rows:
        raise ValueError(f"Speaker manifest has no selected clips: {path}")
    return rows


def payload_value(payload: Dict[str, object], key: str, row_idx: int):
    values = payload.get(key)
    if values is None:
        return None
    try:
        return values[row_idx]
    except (IndexError, KeyError, TypeError):
        return None


def build_clip_index(clip_paths: Sequence[object]) -> Dict[str, int]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for idx, clip_path in enumerate(clip_paths):
        grouped[Path(str(clip_path)).name].append(idx)
    duplicates = {name: indexes for name, indexes in grouped.items() if len(indexes) > 1}
    if duplicates:
        sample = ", ".join(sorted(duplicates)[:5])
        raise ValueError(
            "Source embedding artifact has duplicate clip basenames; "
            f"cannot match manifest safely. Examples: {sample}"
        )
    return {name: indexes[0] for name, indexes in grouped.items()}


def build_metadata_report(ages: Sequence[object], genders: Sequence[object], accents: Sequence[object]):
    report = {}
    for field, values in (("age", ages), ("gender", genders), ("accent", accents)):
        known_values = [clean(value) for value in values if clean(value)]
        report[field] = {
            "known": len(known_values),
            "missing": len(values) - len(known_values),
            "total": len(values),
            "unique_known": len(set(known_values)),
            "top_values": [
                {"value": value, "count": count}
                for value, count in Counter(known_values).most_common(10)
            ],
        }
    return report


def counter_dict(counter: Counter) -> Dict[str, int]:
    return dict(sorted(counter.items()))


def build_subset_artifact(
    source: Dict[str, object],
    manifest_rows: Sequence[Dict[str, str]],
    allow_missing: bool = False,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    clip_paths = list(source.get("clip_paths", []))
    if not clip_paths:
        raise ValueError("Source embedding artifact must contain non-empty clip_paths")
    if "data" not in source:
        raise ValueError("Source embedding artifact must contain data")

    by_clip_name = build_clip_index(clip_paths)
    selected_indexes: List[int] = []
    matched_manifest_rows: List[Dict[str, str]] = []
    missing_rows: List[Dict[str, str]] = []
    speaker_mismatches = []

    for row in manifest_rows:
        idx = by_clip_name.get(row["clip_name"])
        if idx is None:
            missing_rows.append(row)
            continue
        source_speaker = clean(payload_value(source, "speaker_ids", idx))
        if source_speaker and source_speaker != row["speaker_id"]:
            speaker_mismatches.append(
                {
                    "clip_name": row["clip_name"],
                    "manifest_speaker_id": row["speaker_id"],
                    "source_speaker_id": source_speaker,
                }
            )
        selected_indexes.append(idx)
        matched_manifest_rows.append(row)

    if missing_rows and not allow_missing:
        raise ValueError(
            f"{len(missing_rows)} missing manifest clips in source embedding artifact; "
            "rerun extraction for the full manifest or pass --allow-missing for an explicit partial artifact"
        )
    if not selected_indexes:
        raise ValueError("No manifest clips matched the source embedding artifact")

    data = source["data"]
    selected_tensor = torch.as_tensor(data)[selected_indexes].clone()
    out_speaker_ids = [
        clean(payload_value(source, "speaker_ids", idx)) or row["speaker_id"]
        for idx, row in zip(selected_indexes, matched_manifest_rows)
    ]
    out_clip_paths = [str(clip_paths[idx]) for idx in selected_indexes]
    out_ages = [payload_value(source, "age", idx) for idx in selected_indexes]
    out_genders = [
        clean(payload_value(source, "gender", idx)) or row["gender"]
        for idx, row in zip(selected_indexes, matched_manifest_rows)
    ]
    out_accents = [payload_value(source, "accent", idx) for idx in selected_indexes]

    matched_by_gender = Counter(row["gender"] for row in matched_manifest_rows)
    missing_by_gender = Counter(row["gender"] for row in missing_rows)
    matched_speakers_by_gender: Dict[str, set[str]] = defaultdict(set)
    for row in matched_manifest_rows:
        matched_speakers_by_gender[row["gender"]].add(row["speaker_id"])

    report = {
        "source_rows": len(clip_paths),
        "selected_clips": len(manifest_rows),
        "matched_clips": len(selected_indexes),
        "missing_clips": len(missing_rows),
        "missing_clip_names_sample": [row["clip_name"] for row in missing_rows[:50]],
        "matched_by_gender": counter_dict(matched_by_gender),
        "missing_by_gender": counter_dict(missing_by_gender),
        "matched_speakers": len(set(out_speaker_ids)),
        "matched_speakers_by_gender": {
            gender: len(speakers)
            for gender, speakers in sorted(matched_speakers_by_gender.items())
        },
        "speaker_mismatches": speaker_mismatches[:50],
        "speaker_mismatch_count": len(speaker_mismatches),
        "allow_missing": bool(allow_missing),
    }

    artifact = {
        "data": selected_tensor,
        "speaker_ids": out_speaker_ids,
        "clip_paths": out_clip_paths,
        "age": out_ages,
        "gender": out_genders,
        "accent": out_accents,
        "metadata_report": build_metadata_report(out_ages, out_genders, out_accents),
        "gender_followup_report": report,
        "source_commonvoice_report": source.get("metadata_report", {}),
        "corpus_path": source.get("corpus_path"),
        "seed": source.get("seed"),
    }
    return artifact, report


def write_json(data: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown(report: Dict[str, object], path: Path, output_path: Path) -> None:
    lines = [
        "# CommonVoice Gender Follow-Up Artifact",
        "",
        f"Output artifact: `{output_path}`",
        f"Source rows: `{report['source_rows']}`",
        f"Manifest-selected clips: `{report['selected_clips']}`",
        f"Matched clips: `{report['matched_clips']}`",
        f"Missing clips: `{report['missing_clips']}`",
        f"Matched speakers: `{report['matched_speakers']}`",
        f"Allow missing: `{report['allow_missing']}`",
        "",
        "## Matched By Gender",
        "",
        "| Gender | Matched clips | Missing clips | Matched speakers |",
        "| --- | ---: | ---: | ---: |",
    ]
    genders = sorted(
        set(report["matched_by_gender"])
        | set(report["missing_by_gender"])
        | set(report["matched_speakers_by_gender"])
    )
    for gender in genders:
        lines.append(
            f"| `{gender}` | `{report['matched_by_gender'].get(gender, 0)}` | "
            f"`{report['missing_by_gender'].get(gender, 0)}` | "
            f"`{report['matched_speakers_by_gender'].get(gender, 0)}` |"
        )
    if report["missing_clip_names_sample"]:
        lines.extend(["", "## Missing Clip Sample", ""])
        for clip_name in report["missing_clip_names_sample"][:20]:
            lines.append(f"- `{clip_name}`")
    if report["missing_clips"]:
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- This is an available-subset artifact, not the full preflight-selected plan.",
                "- To run the full plan, extract the missing manifest clips from the local CommonVoice corpus first.",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, object]:
    source_path = Path(args.commonvoice_embeddings)
    manifest_path = Path(args.speaker_manifest)
    output_path = Path(args.output)

    source = torch.load(source_path, map_location="cpu", weights_only=False)
    manifest_rows = read_speaker_manifest(manifest_path)
    artifact, report = build_subset_artifact(
        source,
        manifest_rows,
        allow_missing=args.allow_missing,
    )
    report = {
        **report,
        "source_artifact": str(source_path),
        "speaker_manifest": str(manifest_path),
        "output_artifact": str(output_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    write_json(report, Path(args.report_json))
    write_markdown(report, Path(args.report_md), output_path)
    return report


def main() -> None:
    report = run(parse_args())
    print(f"Saved gender-followup artifact with {report['matched_clips']} matched clips")
    print(f"Missing manifest clips: {report['missing_clips']}")
    print(f"Report: {report['output_artifact']}")


if __name__ == "__main__":
    main()
