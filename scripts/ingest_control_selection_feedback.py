#!/usr/bin/env python3
"""
Ingest focused listening feedback into the control-selection perceptual ledger.

The control-selection recommendation intentionally treats human listening as a
separate gate. This helper turns Joe/Stephen feedback for the focused shortlist
into structured rows in `results/control_selection_perceptual_evidence.csv`, then
optionally rebuilds `results/control_selection_recommendation.*`.

It supports two common workflows:

1. a filled ratings CSV from the neutral/sad review bundle;
2. a plain-text response saved to a file plus explicit style-status overrides.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


DEFAULT_LEDGER = "results/control_selection_perceptual_evidence.csv"
DEFAULT_RECOMMENDATION_SCRIPT = "scripts/build_control_selection_recommendation.py"
LEDGER_FIELDS = ["style", "perceptual_status", "listener", "evidence_path", "summary"]
STYLE_ORDER = [
    "anger",
    "confused",
    "disgust",
    "enunciated",
    "fear",
    "happy",
    "neutral",
    "sad",
    "whisper",
]

STYLE_RATING_COLUMNS = {
    "neutral": {
        "target": "neutral_sounds_neutral_1_5",
        "intelligibility": "neutral_intelligibility_1_5",
        "naturalness": "neutral_naturalness_1_5",
    },
    "sad": {
        "target": "sad_sounds_sad_1_5",
        "intelligibility": "sad_intelligibility_1_5",
        "naturalness": "sad_naturalness_1_5",
    },
}

POSITIVE_STATUSES = {"supported", "confirmed", "candidate_preferred"}
NEGATIVE_STATUSES = {"blocked", "blocked_or_subtle", "negative", "reference_preferred"}
NEUTRAL_STATUSES = {"needs_review", "mixed"}
VALID_STATUSES = POSITIVE_STATUSES | NEGATIVE_STATUSES | NEUTRAL_STATUSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings-csv", default=None, help="Filled ratings CSV from a listening bundle")
    parser.add_argument("--feedback-text", default=None, help="Plain-text feedback file to cite in the ledger")
    parser.add_argument("--listener", default="Joe", help="Listener name to record")
    parser.add_argument(
        "--styles",
        default="neutral,sad",
        help="Comma-separated styles to update when deriving from ratings",
    )
    parser.add_argument(
        "--style-status",
        action="append",
        default=[],
        metavar="STYLE=STATUS",
        help="Manual status override, e.g. neutral=supported or sad=blocked",
    )
    parser.add_argument(
        "--style-summary",
        action="append",
        default=[],
        metavar="STYLE=TEXT",
        help="Manual summary override for a style",
    )
    parser.add_argument(
        "--evidence-path",
        default=None,
        help="Path to cite as evidence. Defaults to ratings CSV or feedback text path.",
    )
    parser.add_argument("--ledger", default=DEFAULT_LEDGER, help="Input perceptual evidence ledger")
    parser.add_argument(
        "--out-ledger",
        default=None,
        help="Output ledger path. Defaults to overwriting --ledger.",
    )
    parser.add_argument(
        "--rerun-recommendation",
        action="store_true",
        help="Run scripts/build_control_selection_recommendation.py after updating the ledger",
    )
    return parser.parse_args()


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: Sequence[Dict[str, str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in LEDGER_FIELDS})


def parse_mapping(items: Sequence[str], allowed_statuses: Optional[set[str]] = None) -> Dict[str, str]:
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected STYLE=VALUE, got {item!r}")
        style, value = item.split("=", 1)
        style = style.strip().lower()
        value = value.strip()
        if allowed_statuses is not None:
            value = value.lower()
        if not style or not value:
            raise ValueError(f"Expected non-empty STYLE=VALUE, got {item!r}")
        if allowed_statuses is not None and value not in allowed_statuses:
            raise ValueError(
                f"Invalid status {value!r}; expected one of {sorted(allowed_statuses)}"
            )
        parsed[style] = value
    return parsed


def parse_styles(raw: str) -> List[str]:
    styles = [style.strip().lower() for style in raw.split(",") if style.strip()]
    if not styles:
        raise ValueError("At least one style is required")
    return styles


def to_score(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score <= 0:
        return None
    return score


def average(values: Iterable[Optional[float]]) -> Optional[float]:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def format_average(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def paper_demo_votes(rows: Sequence[Dict[str, str]], style: str) -> Dict[str, int]:
    votes = {"yes": 0, "maybe": 0, "no": 0, "other": 0}
    for row in rows:
        raw = (row.get("paper_demo_candidate") or "").strip().lower()
        if not raw:
            continue
        if raw in votes:
            votes[raw] += 1
        elif style in raw and "yes" in raw:
            votes["yes"] += 1
        elif style in raw and "maybe" in raw:
            votes["maybe"] += 1
        elif "no" in raw or "neither" in raw:
            votes["no"] += 1
        else:
            votes["other"] += 1
    return votes


def summarize_ratings(rows: Sequence[Dict[str, str]], style: str) -> Dict[str, str]:
    if style not in STYLE_RATING_COLUMNS:
        raise ValueError(f"No ratings-column mapping for style {style!r}")
    columns = STYLE_RATING_COLUMNS[style]
    target_avg = average(to_score(row.get(columns["target"])) for row in rows)
    intel_avg = average(to_score(row.get(columns["intelligibility"])) for row in rows)
    nat_avg = average(to_score(row.get(columns["naturalness"])) for row in rows)
    notes = [row.get("notes", "").strip() for row in rows if row.get("notes", "").strip()]
    votes = paper_demo_votes(rows, style)

    if target_avg is None and intel_avg is None and nat_avg is None and not notes and not any(votes.values()):
        status = "needs_review"
    elif votes["no"] > max(votes["yes"], votes["maybe"]) or (
        target_avg is not None and target_avg < 3.0
    ):
        status = "blocked"
    elif all(
        value is not None and value >= 4.0
        for value in (target_avg, intel_avg, nat_avg)
    ) and votes["no"] == 0:
        status = "supported"
    else:
        status = "mixed"

    summary = (
        f"Ratings summary for {style}: target={format_average(target_avg)}, "
        f"intelligibility={format_average(intel_avg)}, naturalness={format_average(nat_avg)}, "
        f"paper_demo_votes={votes}."
    )
    if notes:
        summary += " Notes: " + " | ".join(notes[:5])
    return {"status": status, "summary": summary}


def load_ledger(path: str | Path) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = read_csv(path)
    for row in rows:
        missing = [field for field in LEDGER_FIELDS if field not in row]
        if missing:
            raise ValueError(f"Ledger {path} is missing fields: {missing}")
    return rows


def upsert_ledger_rows(
    ledger_rows: Sequence[Dict[str, str]],
    updates: Sequence[Dict[str, str]],
) -> List[Dict[str, str]]:
    by_style = {row.get("style", ""): dict(row) for row in ledger_rows}
    for update in updates:
        by_style[update["style"]] = dict(update)

    def sort_key(style: str) -> tuple[int, str]:
        try:
            return (STYLE_ORDER.index(style), style)
        except ValueError:
            return (len(STYLE_ORDER), style)

    return [by_style[style] for style in sorted(by_style, key=sort_key)]


def build_updates(args: argparse.Namespace) -> List[Dict[str, str]]:
    styles = parse_styles(args.styles)
    status_overrides = parse_mapping(args.style_status, VALID_STATUSES)
    summary_overrides = parse_mapping(args.style_summary)
    evidence_path = args.evidence_path or args.ratings_csv or args.feedback_text or "manual feedback"

    ratings_rows = read_csv(args.ratings_csv) if args.ratings_csv else []
    feedback_text = ""
    if args.feedback_text:
        feedback_text = Path(args.feedback_text).read_text(encoding="utf-8").strip()

    updates = []
    for style in styles:
        derived = summarize_ratings(ratings_rows, style) if ratings_rows else {
            "status": "needs_review",
            "summary": "Plain-text feedback recorded; use --style-status / --style-summary to classify it.",
        }
        status = status_overrides.get(style, derived["status"])
        summary = summary_overrides.get(style, derived["summary"])
        if feedback_text and style not in summary_overrides:
            summary = f"{summary} Plain-text feedback: {feedback_text}"
        updates.append(
            {
                "style": style,
                "perceptual_status": status,
                "listener": args.listener,
                "evidence_path": evidence_path,
                "summary": summary,
            }
        )
    return updates


def rerun_recommendation(script_path: str = DEFAULT_RECOMMENDATION_SCRIPT) -> None:
    subprocess.run([sys.executable, script_path], check=True)


def run(args: argparse.Namespace) -> List[Dict[str, str]]:
    updates = build_updates(args)
    ledger_path = Path(args.ledger)
    out_path = Path(args.out_ledger or args.ledger)
    ledger_rows = load_ledger(ledger_path)
    updated = upsert_ledger_rows(ledger_rows, updates)
    write_csv(updated, out_path)
    print(f"Wrote {out_path}")
    for update in updates:
        print(f"{update['style']}: {update['perceptual_status']} - {update['summary']}")
    if args.rerun_recommendation:
        rerun_recommendation()
    return updated


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
