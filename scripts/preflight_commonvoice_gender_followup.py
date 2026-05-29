#!/usr/bin/env python3
"""
Preflight a gender-focused CommonVoice follow-up before any new training.

This script reads a local Common Voice language directory (`validated.tsv` plus
`clips/`), normalizes binary gender metadata, checks local clip availability,
and emits a deterministic speaker-level candidate plan for a gender-only control
experiment. It does not extract embeddings or train a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence


DEFAULT_CORPUS = "/Users/steve/datasets/cv-corpus-21.0-2025-03-14/en"
DEFAULT_OUT_MD = "results/commonvoice_gender_followup_preflight.md"
DEFAULT_OUT_JSON = "results/commonvoice_gender_followup_preflight.json"
DEFAULT_OUT_SPEAKERS = "results/commonvoice_gender_followup_speakers.csv"

GENDER_NORMALIZATION = {
    "male": "male",
    "male_masculine": "male",
    "female": "female",
    "female_feminine": "female",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-path", default=DEFAULT_CORPUS)
    parser.add_argument("--validated-name", default="validated.tsv")
    parser.add_argument("--clips-dir-name", default="clips")
    parser.add_argument("--clips-per-speaker", type=int, default=2)
    parser.add_argument("--speaker-cap-per-gender", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-md", default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-speakers", default=DEFAULT_OUT_SPEAKERS)
    return parser.parse_args()


def clean(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    return text


def normalize_gender(value: object) -> Optional[str]:
    value = clean(value)
    if value is None:
        return None
    return GENDER_NORMALIZATION.get(value, value)


def read_validated(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def local_clip_names(clips_dir: Path) -> set[str]:
    if not clips_dir.is_dir():
        return set()
    return {path.name for path in clips_dir.iterdir() if path.is_file()}


def gender_known_rows(rows: Sequence[Dict[str, str]], clip_names: set[str]) -> List[Dict[str, str]]:
    known = []
    for row in rows:
        gender = normalize_gender(row.get("gender"))
        if gender not in {"male", "female"}:
            continue
        path = clean(row.get("path"))
        client_id = clean(row.get("client_id"))
        if not path or not client_id or path not in clip_names:
            continue
        known.append({**row, "gender_normalized": gender})
    return known


def summarize_counts(rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    by_gender = Counter(row["gender_normalized"] for row in rows)
    speakers_by_gender: Dict[str, set[str]] = defaultdict(set)
    clips_by_speaker: Dict[tuple[str, str], int] = Counter()
    for row in rows:
        gender = row["gender_normalized"]
        speaker = row["client_id"]
        speakers_by_gender[gender].add(speaker)
        clips_by_speaker[(gender, speaker)] += 1
    return {
        "clip_counts_by_gender": dict(sorted(by_gender.items())),
        "speaker_counts_by_gender": {
            gender: len(speakers) for gender, speakers in sorted(speakers_by_gender.items())
        },
        "single_clip_speakers_by_gender": {
            gender: sum(1 for (g, _), count in clips_by_speaker.items() if g == gender and count == 1)
            for gender in sorted(speakers_by_gender)
        },
        "multi_clip_speakers_by_gender": {
            gender: sum(1 for (g, _), count in clips_by_speaker.items() if g == gender and count > 1)
            for gender in sorted(speakers_by_gender)
        },
    }


def select_speakers(
    rows: Sequence[Dict[str, str]],
    clips_per_speaker: int,
    speaker_cap_per_gender: int,
    seed: int,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    by_gender_speaker: Dict[tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_gender_speaker[(row["gender_normalized"], row["client_id"])].append(row)

    selected = []
    for gender in ["female", "male"]:
        speaker_items = [
            (speaker, sorted(items, key=lambda row: row["path"]))
            for (g, speaker), items in by_gender_speaker.items()
            if g == gender
        ]
        # Prefer speakers that can satisfy the requested clip count. Within
        # equally eligible groups, seed-controlled shuffling avoids always
        # taking the same lexical slice when a cap is lower than available data.
        grouped: Dict[tuple[int, int], List[tuple[str, List[Dict[str, str]]]]] = defaultdict(list)
        for item in speaker_items:
            priority = (-min(len(item[1]), clips_per_speaker), -len(item[1]))
            grouped[priority].append(item)
        speaker_items = []
        for priority in sorted(grouped):
            group = sorted(grouped[priority], key=lambda item: item[0])
            rng.shuffle(group)
            speaker_items.extend(group)
        if speaker_cap_per_gender and len(speaker_items) > speaker_cap_per_gender:
            speaker_items = speaker_items[:speaker_cap_per_gender]
            speaker_items.sort(key=lambda item: item[0])
        for speaker, items in speaker_items:
            selected_clips = items[:clips_per_speaker]
            selected.append(
                {
                    "gender": gender,
                    "speaker_id": speaker,
                    "available_clips": len(items),
                    "selected_clips": len(selected_clips),
                    "clip_paths": ";".join(row["path"] for row in selected_clips),
                    "ages": ";".join(sorted({clean(row.get("age")) or "" for row in selected_clips if clean(row.get("age"))})),
                    "accents": ";".join(sorted({clean(row.get("accents")) or clean(row.get("accent")) or "" for row in selected_clips if clean(row.get("accents")) or clean(row.get("accent"))})),
                }
            )
    return selected


def write_speakers_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    fields = ["gender", "speaker_id", "available_clips", "selected_clips", "clip_paths", "ages", "accents"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(summary: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def recommendation(summary: Dict[str, object]) -> str:
    speakers = summary["selected_speaker_counts_by_gender"]
    clips = summary["selected_clip_counts_by_gender"]
    if min(speakers.get("female", 0), speakers.get("male", 0)) >= 250:
        return "GO: enough balanced gender-known speaker coverage for a gender-only follow-up preflight."
    if min(clips.get("female", 0), clips.get("male", 0)) >= 250:
        return "CAUTION: enough clips but speaker coverage is lower than preferred; cap or rebalance before training."
    return "BLOCKED: not enough balanced gender-known local rows for a defensible gender-only follow-up."


def write_markdown(summary: Dict[str, object], path: Path) -> None:
    lines = [
        "# CommonVoice Gender Follow-Up Preflight",
        "",
        f"Corpus: `{summary['corpus_path']}`",
        f"Validated TSV: `{summary['validated_tsv']}`",
        f"Local clip files: `{summary['local_clip_files']}`",
        f"Seed: `{summary['seed']}`",
        "",
        "## Recommendation",
        "",
        f"- {summary['recommendation']}",
        "- This is a metadata/data-readiness result only; it does not train or claim perceptual gender control.",
        "- If Joe's style-listening feedback is unresolved, keep this as preparation rather than starting training.",
        "",
        "## Gender-Known Local Rows",
        "",
        f"- Gender-known local clips: `{summary['gender_known_local_rows']}`",
        f"- Gender-known local speakers: `{summary['gender_known_local_speakers']}`",
        "",
        "| Gender | Clips | Speakers | Multi-clip speakers |",
        "| --- | ---: | ---: | ---: |",
    ]
    counts = summary["counts"]
    for gender in ["female", "male"]:
        lines.append(
            f"| `{gender}` | `{counts['clip_counts_by_gender'].get(gender, 0)}` | "
            f"`{counts['speaker_counts_by_gender'].get(gender, 0)}` | "
            f"`{counts['multi_clip_speakers_by_gender'].get(gender, 0)}` |"
        )
    lines.extend(
        [
            "",
            "## Deterministic Candidate Plan",
            "",
            f"- Speaker cap per gender: `{summary['speaker_cap_per_gender']}`",
            f"- Clips per speaker: `{summary['clips_per_speaker']}`",
            f"- Selected speakers: `{summary['selected_speakers_total']}`",
            f"- Selected clips: `{summary['selected_clips_total']}`",
            f"- Speaker manifest: `{summary['out_speakers']}`",
            "",
            "| Gender | Selected speakers | Selected clips |",
            "| --- | ---: | ---: |",
        ]
    )
    for gender in ["female", "male"]:
        lines.append(
            f"| `{gender}` | `{summary['selected_speaker_counts_by_gender'].get(gender, 0)}` | "
            f"`{summary['selected_clip_counts_by_gender'].get(gender, 0)}` |"
        )
    lines.extend(
        [
            "",
            "## Next Command Shape",
            "",
            "Use the speaker manifest as the input contract for the next extraction/training branch. The follow-up should test gender-only first, then gender plus only the shortlisted style controls if gender-only is perceptually plausible.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, object]:
    corpus = Path(args.corpus_path)
    validated = corpus / args.validated_name
    clips_dir = corpus / args.clips_dir_name
    if not validated.exists():
        raise FileNotFoundError(f"Missing validated TSV: {validated}")
    rows = read_validated(validated)
    clip_names = local_clip_names(clips_dir)
    known = gender_known_rows(rows, clip_names)
    selected = select_speakers(
        known,
        clips_per_speaker=args.clips_per_speaker,
        speaker_cap_per_gender=args.speaker_cap_per_gender,
        seed=args.seed,
    )
    out_speakers = Path(args.out_speakers)
    write_speakers_csv(selected, out_speakers)

    selected_speaker_counts = Counter(row["gender"] for row in selected)
    selected_clip_counts = Counter()
    for row in selected:
        selected_clip_counts[row["gender"]] += int(row["selected_clips"])

    summary = {
        "corpus_path": str(corpus),
        "validated_tsv": str(validated),
        "local_clip_files": len(clip_names),
        "validated_rows": len(rows),
        "gender_known_local_rows": len(known),
        "gender_known_local_speakers": len({row["client_id"] for row in known}),
        "counts": summarize_counts(known),
        "clips_per_speaker": args.clips_per_speaker,
        "speaker_cap_per_gender": args.speaker_cap_per_gender,
        "seed": args.seed,
        "selected_speakers_total": len(selected),
        "selected_clips_total": sum(int(row["selected_clips"]) for row in selected),
        "selected_speaker_counts_by_gender": dict(sorted(selected_speaker_counts.items())),
        "selected_clip_counts_by_gender": dict(sorted(selected_clip_counts.items())),
        "out_speakers": str(out_speakers),
    }
    summary["recommendation"] = recommendation(summary)
    write_json(summary, Path(args.out_json))
    write_markdown(summary, Path(args.out_md))
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_speakers}")
    print(summary["recommendation"])
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
