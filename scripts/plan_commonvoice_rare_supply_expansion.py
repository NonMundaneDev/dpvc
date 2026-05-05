"""
Preflight CommonVoice corpus supply before rebuilding rare-style pseudo labels.

The mixed-teacher branch currently fails on canonical emotion recall partly
because the local CommonVoice pseudo-label pool is too small for rare styles
such as anger and fear. This script checks candidate local CommonVoice
directories before extraction so we do not spend a model run on a corpus that
cannot plausibly supply enough rare pseudo-labeled rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import torch
except ImportError:  # pragma: no cover - report still works without artifacts.
    torch = None


DEFAULT_CANONICAL_CORPUS = "/data/cv-corpus-21.0-2025-03-14/en"
DEFAULT_LOCAL_SUBSET = "/Users/steve/datasets/cv-corpus-21.0-2025-03-14-subset/en"
DEFAULT_REFERENCE_ARTIFACTS = [
    "embeddings/openvoice_commonvoice_cv500_pseudo_filtered.pt",
    "embeddings/openvoice_commonvoice_cv500_pseudo_hybrid_extra_priority.pt",
]
DEFAULT_RARE_STYLES = ["anger", "fear"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--corpus-path",
        action="append",
        default=None,
        help=(
            "Candidate CommonVoice language directory with validated.tsv and clips/. "
            "Can be passed more than once."
        ),
    )
    ap.add_argument(
        "--reference-artifacts",
        nargs="*",
        default=DEFAULT_REFERENCE_ARTIFACTS,
        help="Pseudo-labeled CommonVoice artifacts used to estimate rare-style row rates.",
    )
    ap.add_argument(
        "--rare-styles",
        default=",".join(DEFAULT_RARE_STYLES),
        help="Comma-separated styles that must have enough pseudo-label supply.",
    )
    ap.add_argument(
        "--target-selected-rare-rows",
        type=int,
        default=50,
        help="Target selected pseudo-labeled rows per rare style before mixed sampling.",
    )
    ap.add_argument(
        "--min-usable-rows",
        type=int,
        default=10000,
        help="Minimum usable validated rows with local clips before extraction is allowed.",
    )
    ap.add_argument(
        "--min-usable-speakers",
        type=int,
        default=2000,
        help="Minimum speakers with at least one local validated clip.",
    )
    ap.add_argument(
        "--safety-multiplier",
        type=float,
        default=1.5,
        help="Multiplier applied to rare-rate row estimates to avoid another undersupplied run.",
    )
    ap.add_argument(
        "--extraction-max-clips-per-speaker",
        type=int,
        default=3,
        help="Recommended extraction cap used to derive the speaker cap in the report.",
    )
    ap.add_argument(
        "--artifact-prefix",
        default="openvoice_commonvoice_cvrare_expanded",
        help="Name prefix for recommended embedding and pseudo-label artifacts.",
    )
    ap.add_argument(
        "--out-json",
        default="results/commonvoice_rare_supply_expansion_preflight.json",
        help="Output JSON report path.",
    )
    ap.add_argument(
        "--out-md",
        default="results/commonvoice_rare_supply_expansion_preflight.md",
        help="Output Markdown report path.",
    )
    return ap.parse_args()


def parse_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def tensor_len(value: Any) -> int:
    if value is None:
        return 0
    if hasattr(value, "shape"):
        return int(value.shape[0])
    return len(value)


def load_artifact(path: Path) -> Optional[Dict[str, Any]]:
    if torch is None or not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def artifact_selected_counts(data: Dict[str, Any], rare_styles: Iterable[str]) -> Counter:
    rare_styles = set(rare_styles)
    styles = list(data.get("pseudo_style", []))
    selected_styles = list(data.get("pseudo_style_selected", []))
    selected_mask = data.get("pseudo_style_selected_mask")
    if selected_styles and selected_mask is not None:
        counts = Counter()
        for style, selected in zip(selected_styles, selected_mask):
            if bool(selected) and style in rare_styles:
                counts[str(style)] += 1
        return counts
    return Counter(str(style) for style in styles if style in rare_styles)


def summarize_reference_artifacts(paths: Iterable[str], rare_styles: List[str]) -> Dict[str, Any]:
    summaries = []
    estimates = []
    for raw_path in paths:
        path = Path(raw_path)
        data = load_artifact(path)
        if data is None:
            summaries.append({
                "path": str(path),
                "exists": path.exists(),
                "loaded": False,
                "rows": 0,
                "selected_counts": {},
                "rates": {},
            })
            continue
        rows = tensor_len(data.get("data"))
        counts = artifact_selected_counts(data, rare_styles)
        rates = {}
        for style in rare_styles:
            rate = counts.get(style, 0) / rows if rows else 0.0
            rates[style] = rate
            if rate > 0:
                estimates.append(math.ceil(1.0 / rate))
        summaries.append({
            "path": str(path),
            "exists": True,
            "loaded": True,
            "rows": rows,
            "selected_counts": dict(counts),
            "rates": rates,
        })
    return {
        "artifacts": summaries,
        "rows_per_single_rare_row_observations": estimates,
    }


def count_known(values: Counter) -> int:
    return sum(count for value, count in values.items() if value)


def top_values(values: Counter, limit: int = 5) -> List[Dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in values.most_common(limit)
        if value
    ]


def resolve_accent_column(fieldnames: Optional[List[str]]) -> Optional[str]:
    if not fieldnames:
        return None
    for column in ("accent", "accents"):
        if column in fieldnames:
            return column
    return None


def scan_corpus(raw_path: str) -> Dict[str, Any]:
    corpus_path = Path(raw_path).expanduser()
    validated_path = corpus_path / "validated.tsv"
    clips_dir = corpus_path / "clips"
    result: Dict[str, Any] = {
        "path": str(corpus_path),
        "exists": corpus_path.exists(),
        "validated_exists": validated_path.exists(),
        "clips_exists": clips_dir.is_dir(),
        "validated_rows": 0,
        "rows_with_client_and_path": 0,
        "usable_rows": 0,
        "speaker_count": 0,
        "usable_speaker_count": 0,
        "missing_clip_rows": 0,
        "columns": [],
        "metadata": {},
        "status": "missing",
        "problems": [],
    }
    if not corpus_path.exists():
        result["problems"].append("corpus path does not exist")
        return result
    if not validated_path.exists():
        result["problems"].append("validated.tsv is missing")
        return result
    if not clips_dir.is_dir():
        result["problems"].append("clips/ directory is missing")
        return result

    speakers = set()
    usable_speakers = set()
    metadata_counters = {
        "age": Counter(),
        "gender": Counter(),
        "accent": Counter(),
    }
    with validated_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        result["columns"] = reader.fieldnames or []
        accent_column = resolve_accent_column(reader.fieldnames)
        required_missing = [
            column for column in ("client_id", "path") if column not in result["columns"]
        ]
        if required_missing:
            result["problems"].append(
                "validated.tsv is missing required columns: "
                + ", ".join(required_missing)
            )
            return result

        for row in reader:
            result["validated_rows"] += 1
            client_id = (row.get("client_id") or "").strip()
            clip_rel = (row.get("path") or "").strip()
            if not client_id or not clip_rel:
                continue
            result["rows_with_client_and_path"] += 1
            speakers.add(client_id)
            for field, column in (
                ("age", "age"),
                ("gender", "gender"),
                ("accent", accent_column),
            ):
                value = (row.get(column) or "").strip() if column else ""
                metadata_counters[field][value] += 1
            clip_path = clips_dir / clip_rel
            if not clip_path.exists():
                result["missing_clip_rows"] += 1
                continue
            result["usable_rows"] += 1
            usable_speakers.add(client_id)

    result["speaker_count"] = len(speakers)
    result["usable_speaker_count"] = len(usable_speakers)
    total = result["rows_with_client_and_path"]
    result["metadata"] = {
        field: {
            "known": count_known(counter),
            "missing": counter.get("", 0),
            "total": total,
            "unique_known": len([value for value in counter if value]),
            "top_values": top_values(counter),
        }
        for field, counter in metadata_counters.items()
    }
    result["status"] = "scanned"
    return result


def compute_required_rows(
    reference_summary: Dict[str, Any],
    rare_styles: List[str],
    target_selected_rare_rows: int,
    min_usable_rows: int,
    safety_multiplier: float,
) -> Dict[str, Any]:
    per_style: Dict[str, List[int]] = {style: [] for style in rare_styles}
    for artifact in reference_summary["artifacts"]:
        rows = artifact.get("rows", 0)
        if not rows:
            continue
        counts = artifact.get("selected_counts", {})
        for style in rare_styles:
            count = counts.get(style, 0)
            if count:
                needed = math.ceil((target_selected_rare_rows * rows) / count)
                per_style[style].append(needed)

    style_requirements = {
        style: max(values) if values else None for style, values in per_style.items()
    }
    observed_requirements = [
        value for value in style_requirements.values() if value is not None
    ]
    observed_required = max(observed_requirements) if observed_requirements else None
    if observed_required is None:
        recommended = min_usable_rows
    else:
        recommended = max(min_usable_rows, math.ceil(observed_required * safety_multiplier))
    return {
        "target_selected_rare_rows": target_selected_rare_rows,
        "min_usable_rows_floor": min_usable_rows,
        "safety_multiplier": safety_multiplier,
        "per_style_rows_needed_before_safety": style_requirements,
        "recommended_min_usable_rows": recommended,
    }


def decide_candidates(
    candidates: List[Dict[str, Any]],
    required_rows: Dict[str, Any],
    min_usable_speakers: int,
) -> List[Dict[str, Any]]:
    min_rows = required_rows["recommended_min_usable_rows"]
    decisions = []
    for candidate in candidates:
        problems = list(candidate.get("problems", []))
        if candidate.get("usable_rows", 0) < min_rows:
            problems.append(
                f"usable rows {candidate.get('usable_rows', 0)} < recommended {min_rows}"
            )
        if candidate.get("usable_speaker_count", 0) < min_usable_speakers:
            problems.append(
                "usable speakers "
                f"{candidate.get('usable_speaker_count', 0)} < minimum {min_usable_speakers}"
            )
        decisions.append({
            "path": candidate["path"],
            "go": not problems,
            "problems": problems,
        })
    return decisions


def best_candidate(candidates: List[Dict[str, Any]], decisions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    decision_by_path = {decision["path"]: decision for decision in decisions}
    go_candidates = [
        candidate for candidate in candidates if decision_by_path[candidate["path"]]["go"]
    ]
    if not go_candidates:
        return None
    return max(
        go_candidates,
        key=lambda candidate: (
            candidate.get("usable_rows", 0),
            candidate.get("usable_speaker_count", 0),
        ),
    )


def recommended_commands(
    corpus_path: str,
    artifact_prefix: str,
    required_rows: Dict[str, Any],
    max_clips_per_speaker: int,
    min_usable_speakers: int,
) -> Dict[str, str]:
    target_rows = required_rows["recommended_min_usable_rows"]
    speaker_cap = max(min_usable_speakers, math.ceil(target_rows / max_clips_per_speaker))
    emb = f"embeddings/{artifact_prefix}_emb.pt"
    scored = f"embeddings/{artifact_prefix}_pseudo_scored.pt"
    filtered = f"embeddings/{artifact_prefix}_pseudo_filtered.pt"
    proto = f"embeddings/{artifact_prefix}_pseudo_prototype.pt"
    proto_filtered = f"embeddings/{artifact_prefix}_pseudo_prototype_filtered.pt"
    hybrid = f"embeddings/{artifact_prefix}_pseudo_hybrid_extra_priority.pt"
    mixed = "embeddings/openvoice_mixed_teacher_rare_supply_base.pt"
    checkpoint = "embeddings/openvoice_vae_mixed_teacher_rare_supply_balanced.pt"
    audit_csv = "results/commonvoice_pseudolabel_supply_audit_rare_supply.csv"
    audit_md = "results/commonvoice_pseudolabel_supply_audit_rare_supply.md"
    return {
        "preflight": (
            "python scripts/plan_commonvoice_rare_supply_expansion.py "
            f"--corpus-path {corpus_path}"
        ),
        "extract": (
            "python examples/openvoice_extract_commonvoice.py "
            f"--corpus-path {corpus_path} "
            f"--output {emb} "
            "--seed 42 "
            f"--max-speakers {speaker_cap} "
            f"--max-clips-per-speaker {max_clips_per_speaker}"
        ),
        "emotion2vec_score": (
            "python scripts/annotate_commonvoice_pseudolabels.py "
            f"--embeddings {emb} "
            f"--output {scored} "
            "--save-style-score-map "
            "--report-threshold 0.60"
        ),
        "emotion2vec_filter": (
            "python scripts/filter_commonvoice_pseudolabels.py "
            f"--input {scored} "
            f"--output {filtered} "
            "--default-threshold 0.60 "
            "--style-targets anger=50,fear=50,disgust=80,happy=80,neutral=120,sad=120 "
            "--acceptance-policy balanced_targets"
        ),
        "prototype_score": (
            "python scripts/annotate_commonvoice_latent_prototypes.py "
            f"--commonvoice {emb} "
            "--combined embeddings/openvoice_combined_emb.pt "
            "--checkpoint embeddings/openvoice_vae_combined.pt "
            f"--output {proto}"
        ),
        "prototype_filter": (
            "python scripts/filter_commonvoice_pseudolabels.py "
            f"--input {proto} "
            f"--output {proto_filtered} "
            "--default-threshold 0.35 "
            "--style-targets confused=50,enunciated=50,whisper=50 "
            "--acceptance-policy balanced_targets"
        ),
        "hybrid_combine": (
            "python scripts/combine_commonvoice_pseudolabel_teachers.py "
            f"--emotion2vec {filtered} "
            f"--prototype {proto_filtered} "
            f"--output {hybrid} "
            "--policy prototype_extra_priority"
        ),
        "supply_audit": (
            "python scripts/audit_commonvoice_pseudolabel_supply.py "
            f"--artifacts {scored} {filtered} {hybrid} "
            f"--out-csv {audit_csv} "
            f"--out-md {audit_md}"
        ),
        "mixed_artifact_after_audit_passes": (
            "python scripts/build_mixed_training_set.py "
            f"--commonvoice {hybrid} "
            f"--output {mixed} "
            "--acceptance-policy artifact_selected "
            "--commonvoice-prefer-pseudo "
            "--commonvoice-max-clips-per-speaker 2 "
            "--pseudo-confidence-scale "
            "--pseudo-row-weight 0.75 "
            "--true-row-weight 1.25"
        ),
        "train_after_audit_passes": (
            "python examples/openvoice_train_vae_mixed.py "
            f"--embeddings {mixed} "
            f"--output {checkpoint} "
            "--schedule static_balanced"
        ),
    }


def write_json(path: str, report: Dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def command_block(command: str) -> List[str]:
    return ["```bash", command, "```"]


def write_markdown(path: str, report: Dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CommonVoice Rare-Class Supply Expansion Preflight",
        "",
        "This report checks whether a local CommonVoice English corpus is large enough to justify rebuilding the pseudo-label pool for rare canonical styles before another mixed-data model run.",
        "",
        "## Decision",
        "",
    ]
    if report["go"]:
        lines.append("- `GO`: at least one candidate corpus meets the current row and speaker gate.")
    else:
        lines.append("- `NO-GO`: no scanned corpus meets the current row and speaker gate.")
    lines.extend([
        f"- rare styles: `{', '.join(report['rare_styles'])}`",
        f"- target selected rare rows per style: `{report['required_rows']['target_selected_rare_rows']}`",
        f"- recommended minimum usable rows: `{report['required_rows']['recommended_min_usable_rows']}`",
        f"- minimum usable speakers: `{report['min_usable_speakers']}`",
        "",
        "## Candidate Corpora",
        "",
        "| path | usable rows | usable speakers | missing clips | decision |",
        "| --- | ---: | ---: | ---: | --- |",
    ])
    decision_by_path = {decision["path"]: decision for decision in report["decisions"]}
    for candidate in report["candidates"]:
        decision = decision_by_path[candidate["path"]]
        status = "GO" if decision["go"] else "NO-GO"
        lines.append(
            f"| `{candidate['path']}` | {candidate['usable_rows']} | "
            f"{candidate['usable_speaker_count']} | {candidate['missing_clip_rows']} | "
            f"{status} |"
        )
    lines.append("")
    for decision in report["decisions"]:
        if decision["go"]:
            continue
        lines.append(f"### Why `{decision['path']}` is blocked")
        lines.append("")
        for problem in decision["problems"]:
            lines.append(f"- {problem}")
        lines.append("")

    lines.extend([
        "## Rare-Rate Estimate",
        "",
        "The row target is derived from the checked-in pseudo-labeled CommonVoice artifacts, then multiplied by a safety factor so the next run is not just another tiny rare-class sample.",
        "",
        "| reference artifact | rows | selected anger | selected fear |",
        "| --- | ---: | ---: | ---: |",
    ])
    for artifact in report["reference_summary"]["artifacts"]:
        counts = artifact.get("selected_counts", {})
        lines.append(
            f"| `{artifact['path']}` | {artifact.get('rows', 0)} | "
            f"{counts.get('anger', 0)} | {counts.get('fear', 0)} |"
        )
    lines.extend([
        "",
        "| style | rows needed before safety |",
        "| --- | ---: |",
    ])
    for style, rows in report["required_rows"]["per_style_rows_needed_before_safety"].items():
        display = rows if rows is not None else "n/a"
        lines.append(f"| `{style}` | {display} |")
    lines.append("")

    if report["go"] and report["selected_corpus"]:
        lines.extend([
            "## Next Commands",
            "",
            "Run these in order. Stop after the supply audit if `anger` and `fear` still fail to reach the target selected-row count.",
            "",
        ])
        for name, command in report["commands"].items():
            lines.extend([f"### {name}", "", *command_block(command), ""])
    else:
        lines.extend([
            "## Next Commands",
            "",
            "Do not run a new rare-supply model condition from the current local subset. First mount or download a fuller English CommonVoice corpus at the stable path, then rerun the preflight:",
            "",
            *command_block(report["commands"]["preflight"]),
            "",
            "Expected stable path:",
            "",
            f"- `{DEFAULT_CANONICAL_CORPUS}`",
            "",
            "Required layout:",
            "",
            f"- `{DEFAULT_CANONICAL_CORPUS}/validated.tsv`",
            f"- `{DEFAULT_CANONICAL_CORPUS}/clips/`",
            "",
            "After that preflight returns `GO`, use this gated command family. Keep the stop point after `supply_audit`: only build/train the mixed artifact if `anger` and `fear` reach the target selected-row count.",
            "",
        ])
        for name, command in report["commands"].items():
            if name == "preflight":
                continue
            lines.extend([f"### {name}", "", *command_block(command), ""])

    lines.extend([
        "## Validation",
        "",
        "- `Validation`: the preflight scanned every candidate path supplied to the script.",
        "- `Validation`: the gate uses local `validated.tsv` rows with matching files in `clips/`, not a remote dataset shortcut.",
        "- `Validation`: the current decision is reproducible from the JSON report next to this file.",
        "",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rare_styles = parse_list(args.rare_styles)
    corpus_paths = args.corpus_path or [DEFAULT_CANONICAL_CORPUS, DEFAULT_LOCAL_SUBSET]
    candidates = [scan_corpus(path) for path in corpus_paths]
    reference_summary = summarize_reference_artifacts(args.reference_artifacts, rare_styles)
    required_rows = compute_required_rows(
        reference_summary=reference_summary,
        rare_styles=rare_styles,
        target_selected_rare_rows=args.target_selected_rare_rows,
        min_usable_rows=args.min_usable_rows,
        safety_multiplier=args.safety_multiplier,
    )
    decisions = decide_candidates(
        candidates,
        required_rows=required_rows,
        min_usable_speakers=args.min_usable_speakers,
    )
    selected = best_candidate(candidates, decisions)
    selected_path = selected["path"] if selected else DEFAULT_CANONICAL_CORPUS
    commands = recommended_commands(
        corpus_path=selected_path,
        artifact_prefix=args.artifact_prefix,
        required_rows=required_rows,
        max_clips_per_speaker=args.extraction_max_clips_per_speaker,
        min_usable_speakers=args.min_usable_speakers,
    )
    report = {
        "go": selected is not None,
        "selected_corpus": selected["path"] if selected else None,
        "rare_styles": rare_styles,
        "min_usable_speakers": args.min_usable_speakers,
        "required_rows": required_rows,
        "reference_summary": reference_summary,
        "candidates": candidates,
        "decisions": decisions,
        "commands": commands,
    }
    write_json(args.out_json, report)
    write_markdown(args.out_md, report)
    print(f"Wrote JSON preflight report to {args.out_json}")
    print(f"Wrote Markdown preflight report to {args.out_md}")
    print("Decision:", "GO" if report["go"] else "NO-GO")


if __name__ == "__main__":
    main()
