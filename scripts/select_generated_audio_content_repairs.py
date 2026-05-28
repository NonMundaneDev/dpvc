#!/usr/bin/env python3
"""
Gate generated-audio style-strength candidates with content and listening rules.

The style-strength grid can find target-label gains, but those gains are only
useful if content, naturalness, identity shift, and human listening do not
regress. This script turns the A/B triage sheet into a conservative repair
decision table.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


DEFAULT_PRIORITY_CSV = "results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority.csv"
DEFAULT_RATINGS = "results/listening_mixed_teacher_cvrare_strength_grid_ab_review_priority_ratings_joe_2026-05-19.csv"


@dataclass(frozen=True)
class Thresholds:
    max_candidate_wer: float = 0.30
    max_wer_delta: float = 0.15
    min_candidate_mos_delta: float = -0.75
    max_mos_delta_drop: float = 0.25
    min_novelty_delta: float = -0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priority-csv", default=DEFAULT_PRIORITY_CSV)
    parser.add_argument(
        "--ratings",
        default=DEFAULT_RATINGS,
        help="Optional filled perceptual ratings CSV. Use '' to ignore ratings.",
    )
    parser.add_argument("--out-csv", default="results/generated_audio_content_repair_gate.csv")
    parser.add_argument("--out-md", default="results/generated_audio_content_repair_gate.md")
    parser.add_argument("--out-json", default="results/generated_audio_content_repair_gate.json")
    parser.add_argument("--max-candidate-wer", type=float, default=Thresholds.max_candidate_wer)
    parser.add_argument("--max-wer-delta", type=float, default=Thresholds.max_wer_delta)
    parser.add_argument("--min-candidate-mos-delta", type=float, default=Thresholds.min_candidate_mos_delta)
    parser.add_argument("--max-mos-delta-drop", type=float, default=Thresholds.max_mos_delta_drop)
    parser.add_argument("--min-novelty-delta", type=float, default=Thresholds.min_novelty_delta)
    return parser.parse_args()


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    if not path:
        return []
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value, default: Optional[float] = None) -> Optional[float]:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default: int = 0) -> int:
    if value in ("", None):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_strength_from_candidate_tag(candidate_tag: str) -> Optional[float]:
    match = re.search(r"_s([0-9]+(?:p[0-9]+)?|m[0-9]+(?:p[0-9]+)?)$", candidate_tag or "")
    if not match:
        return None
    token = match.group(1).replace("p", ".").replace("m", "-")
    return float(token)


def rating_key(row: Dict[str, str]) -> tuple:
    return (
        row.get("style", ""),
        row.get("source_stem", ""),
        row.get("reference_tag", ""),
        row.get("candidate_tag", ""),
    )


def load_ratings(path: str | Path) -> Dict[tuple, Dict[str, str]]:
    ratings = {}
    for row in read_csv(path):
        ratings[rating_key(row)] = row
    return ratings


def normalized_preference(rating: Optional[Dict[str, str]]) -> str:
    if not rating:
        return "unreviewed"
    preference = (rating.get("prefer_reference_or_candidate") or "").strip().lower()
    if preference in {"candidate", "reference", "tie", "neither"}:
        return preference
    return "unreviewed"


def objective_failure_reasons(row: Dict[str, str], thresholds: Thresholds) -> List[str]:
    reasons = []
    match_delta = to_int(row.get("match_delta"))
    candidate_wer = to_float(row.get("candidate_wer"), 0.0) or 0.0
    wer_delta = to_float(row.get("wer_delta"), 0.0) or 0.0
    candidate_mos_delta = to_float(row.get("candidate_mos_delta"), 0.0) or 0.0
    mos_delta_delta = to_float(row.get("mos_delta_delta"), 0.0) or 0.0
    novelty_delta = to_float(row.get("novelty_delta"), 0.0) or 0.0

    if match_delta <= 0:
        reasons.append("no_target_gain")
    if to_int(row.get("severe_risk")):
        reasons.append("severe_risk")
    if to_int(row.get("quality_risk")):
        reasons.append("quality_risk")
    if candidate_wer > thresholds.max_candidate_wer:
        reasons.append("candidate_wer")
    if wer_delta > thresholds.max_wer_delta:
        reasons.append("wer_delta")
    if candidate_mos_delta < thresholds.min_candidate_mos_delta:
        reasons.append("candidate_mos_delta")
    if mos_delta_delta < -thresholds.max_mos_delta_drop:
        reasons.append("mos_delta_drop")
    if novelty_delta < thresholds.min_novelty_delta:
        reasons.append("novelty_delta")
    return reasons


def classify_candidate_row(
    row: Dict[str, str],
    rating: Optional[Dict[str, str]],
    thresholds: Thresholds,
) -> Dict[str, str]:
    reasons = objective_failure_reasons(row, thresholds)
    preference = normalized_preference(rating)
    candidate_strength = parse_strength_from_candidate_tag(row.get("candidate_tag", ""))

    if reasons:
        decision = "reject_no_target_gain" if "no_target_gain" in reasons else "reject_quality"
        objective_gate = "fail"
    else:
        objective_gate = "pass"
        if preference == "candidate":
            decision = "promote_candidate"
        elif preference == "reference":
            decision = "blocked_by_reference_preference"
        elif preference == "tie":
            decision = "blocked_by_perceptual_tie"
        elif preference == "neither":
            decision = "blocked_by_perceptual_rejection"
        else:
            decision = "needs_listening"

    return {
        "candidate_strength": "" if candidate_strength is None else f"{candidate_strength:g}",
        "objective_gate": objective_gate,
        "objective_reasons": ";".join(reasons),
        "perceptual_gate": preference,
        "human_preference": preference,
        "human_notes": (rating or {}).get("notes", ""),
        "decision": decision,
    }


def summarize_by_style(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("style", "")].append(row)

    summary = {}
    for style, style_rows in sorted(grouped.items()):
        decisions = Counter(row.get("decision", "") for row in style_rows)
        objective_pass = sum(1 for row in style_rows if row.get("objective_gate") == "pass")
        promoted = decisions.get("promote_candidate", 0)
        needs_listening = decisions.get("needs_listening", 0)
        reviewed_blocked = sum(
            decisions.get(decision, 0)
            for decision in (
                "blocked_by_reference_preference",
                "blocked_by_perceptual_tie",
                "blocked_by_perceptual_rejection",
            )
        )

        if promoted:
            style_decision = "promote_candidate"
        elif objective_pass == 0:
            style_decision = "needs_content_repair"
        elif needs_listening:
            style_decision = "needs_more_listening"
        elif reviewed_blocked:
            style_decision = "diagnostic_only"
        else:
            style_decision = "needs_content_repair"

        summary[style] = {
            "style": style,
            "rows": len(style_rows),
            "objective_pass": objective_pass,
            "promoted": promoted,
            "needs_listening": needs_listening,
            "reviewed_blocked": reviewed_blocked,
            "decision_counts": dict(decisions),
            "style_decision": style_decision,
        }
    return summary


def enrich_rows(
    priority_rows: Sequence[Dict[str, str]],
    ratings: Dict[tuple, Dict[str, str]],
    thresholds: Thresholds,
) -> List[Dict[str, str]]:
    enriched = []
    for row in priority_rows:
        decision = classify_candidate_row(row, ratings.get(rating_key(row)), thresholds)
        enriched.append({**row, **decision})
    return enriched


def promoted_profile(rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    promoted = [row for row in rows if row.get("decision") == "promote_candidate"]
    style_strengths = {}
    promoted_rows = []
    for row in promoted:
        style = row.get("style", "")
        strength = to_float(row.get("candidate_strength"))
        if not style or strength is None:
            continue
        current = style_strengths.get(style)
        if current is None or strength > current:
            style_strengths[style] = strength
        promoted_rows.append({
            "style": style,
            "source_stem": row.get("source_stem", ""),
            "candidate_tag": row.get("candidate_tag", ""),
            "candidate_strength": row.get("candidate_strength", ""),
            "human_preference": row.get("human_preference", ""),
        })
    return {
        "style_strengths": style_strengths,
        "promoted_rows": promoted_rows,
        "note": (
            "No promoted style strengths: objective gains need human candidate wins."
            if not promoted_rows else
            "Promoted strengths still require broader listening before becoming defaults."
        ),
    }


def write_csv(path: str | Path, rows: Sequence[Dict[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "style",
        "source_stem",
        "candidate_strength",
        "decision",
        "objective_gate",
        "objective_reasons",
        "perceptual_gate",
        "human_preference",
        "human_notes",
        "priority",
        "bucket",
        "reference_match",
        "candidate_match",
        "match_delta",
        "reference_wer",
        "candidate_wer",
        "wer_delta",
        "reference_mos_delta",
        "candidate_mos_delta",
        "mos_delta_delta",
        "reference_novelty",
        "candidate_novelty",
        "novelty_delta",
        "quality_risk",
        "severe_risk",
        "reference_tag",
        "candidate_tag",
        "reference_audio",
        "candidate_audio",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def fmt(value) -> str:
    if value in ("", None):
        return "-"
    return str(value)


def write_markdown(
    path: str | Path,
    rows: Sequence[Dict[str, str]],
    style_summary: Dict[str, Dict[str, object]],
    thresholds: Thresholds,
    ratings_path: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    decisions = Counter(row.get("decision", "") for row in rows)
    promoted = [row for row in rows if row.get("decision") == "promote_candidate"]
    needs_listening = [row for row in rows if row.get("decision") == "needs_listening"]
    blocked = [
        row for row in rows
        if row.get("decision") in {
            "blocked_by_reference_preference",
            "blocked_by_perceptual_tie",
            "blocked_by_perceptual_rejection",
        }
    ]

    lines = [
        "# Generated-Audio Content-Repair Gate",
        "",
        "This report applies a conservative gate to hard-style strength-grid candidates.",
        (
            "A candidate needs objective target gain plus content/naturalness/novelty "
            "safety, and if human ratings are present it must win perceptually before "
            "promotion."
        ),
        "",
        f"Ratings: `{ratings_path or 'not used'}`",
        "",
        "## Thresholds",
        "",
        f"- maximum candidate WER: `{thresholds.max_candidate_wer}`",
        f"- maximum WER increase over reference: `{thresholds.max_wer_delta}`",
        f"- minimum candidate MOS delta: `{thresholds.min_candidate_mos_delta}`",
        f"- maximum MOS-delta drop: `{thresholds.max_mos_delta_drop}`",
        f"- minimum novelty delta: `{thresholds.min_novelty_delta}`",
        "",
        "## Decision Counts",
        "",
        "| Decision | Count |",
        "|----------|------:|",
    ]
    for decision, count in sorted(decisions.items()):
        lines.append(f"| `{decision}` | {count} |")

    lines.extend([
        "",
        "## Style Decisions",
        "",
        "| Style | Rows | Objective-pass rows | Needs listening | Perceptually blocked | Promoted | Style decision |",
        "|-------|-----:|--------------------:|----------------:|---------------------:|---------:|----------------|",
    ])
    for style, summary in style_summary.items():
        lines.append(
            f"| `{style}` | {summary['rows']} | {summary['objective_pass']} | "
            f"{summary['needs_listening']} | {summary['reviewed_blocked']} | "
            f"{summary['promoted']} | `{summary['style_decision']}` |"
        )

    lines.extend([
        "",
        "## Promotion Result",
        "",
    ])
    if not promoted:
        lines.append("No style-strength candidate is promoted by the current gate.")
        lines.append("")
        lines.append(
            "This means the grid remains diagnostic: it finds classifier gains, "
            "but those gains have not passed the combined content and perceptual standard."
        )
    else:
        lines.append("| Style | Source | Strength | Candidate | Human preference |")
        lines.append("|-------|--------|----------|-----------|------------------|")
        for row in promoted:
            lines.append(
                f"| `{row['style']}` | `{row['source_stem']}` | "
                f"`{row['candidate_strength']}` | `{row['candidate_tag']}` | "
                f"`{row['human_preference']}` |"
            )

    lines.extend([
        "",
        "## Needs More Listening",
        "",
        "| Style | Source | Strength | Candidate WER | Candidate MOS delta | Novelty delta | Candidate |",
        "|-------|--------|----------|--------------:|--------------------:|--------------:|-----------|",
    ])
    if not needs_listening:
        lines.append("| - | - | - | - | - | - | - |")
    for row in needs_listening:
        lines.append(
            f"| `{row['style']}` | `{row['source_stem']}` | "
            f"`{row['candidate_strength']}` | `{fmt(row.get('candidate_wer'))}` | "
            f"`{fmt(row.get('candidate_mos_delta'))}` | "
            f"`{fmt(row.get('novelty_delta'))}` | `{row['candidate_tag']}` |"
        )

    lines.extend([
        "",
        "## Perceptually Blocked Objective-Pass Rows",
        "",
        "| Style | Source | Strength | Human preference | Notes |",
        "|-------|--------|----------|------------------|-------|",
    ])
    if not blocked:
        lines.append("| - | - | - | - | - |")
    for row in blocked:
        lines.append(
            f"| `{row['style']}` | `{row['source_stem']}` | "
            f"`{row['candidate_strength']}` | `{row['human_preference']}` | "
            f"{row.get('human_notes', '')} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        (
            "- `anger_s10` and `fear_s7p5` still have useful diagnostic rows, "
            "but Joe's review blocks promotion because the perceived result was "
            "tie/reference, not candidate."
        ),
        (
            "- `disgust` has no safe promoted repair under this gate; it needs "
            "a generated-audio/content objective rather than a stronger strength preset."
        ),
        (
            "- The next training-side repair should optimize for generated-audio "
            "behavior directly instead of relying on embedding-space or classifier-only gains."
        ),
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(
    path: str | Path,
    rows: Sequence[Dict[str, str]],
    style_summary: Dict[str, Dict[str, object]],
    thresholds: Thresholds,
) -> None:
    payload = {
        "thresholds": thresholds.__dict__,
        "style_summary": style_summary,
        "profile": promoted_profile(rows),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    thresholds = Thresholds(
        max_candidate_wer=args.max_candidate_wer,
        max_wer_delta=args.max_wer_delta,
        min_candidate_mos_delta=args.min_candidate_mos_delta,
        max_mos_delta_drop=args.max_mos_delta_drop,
        min_novelty_delta=args.min_novelty_delta,
    )
    priority_rows = read_csv(args.priority_csv)
    if not priority_rows:
        raise SystemExit(f"No priority rows found: {args.priority_csv}")
    ratings = load_ratings(args.ratings) if args.ratings else {}
    enriched = enrich_rows(priority_rows, ratings, thresholds)
    style_summary = summarize_by_style(enriched)

    write_csv(args.out_csv, enriched)
    write_markdown(args.out_md, enriched, style_summary, thresholds, args.ratings)
    write_json(args.out_json, enriched, style_summary, thresholds)

    print(f"Wrote content-repair gate CSV to {args.out_csv}")
    print(f"Wrote content-repair gate summary to {args.out_md}")
    print(f"Wrote content-repair gate JSON to {args.out_json}")


if __name__ == "__main__":
    main()
