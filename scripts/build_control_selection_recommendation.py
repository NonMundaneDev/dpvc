#!/usr/bin/env python3
"""
Build a paper-facing control-selection recommendation from checked-in evidence.

This script combines three gates:

1. source-label separability on original training audio;
2. generated-output metrics for the current reference guard;
3. human/perceptual evidence captured in a small explicit CSV ledger.

The output is a shortlist table, not a new model result. It exists to prevent
weak or merely classifier-visible controls from becoming paper claims without
supporting generated-audio and listening evidence.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


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

DEFAULT_SOURCE = "results/training_style_separability_by_label.csv"
DEFAULT_EMOTION = (
    "results/eval_emotion_mixed_teacher_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.csv"
)
DEFAULT_WER = (
    "results/eval_wer_mixed_teacher_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.csv"
)
DEFAULT_MOS = (
    "results/eval_mos_mixed_teacher_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.csv"
)
DEFAULT_NOVELTY = "results/eval_external_speaker_verifier_cvrare_sad_enunc_guard.csv"
DEFAULT_COLLAPSE = "results/eval_mixed_teacher_collapse.csv"
DEFAULT_COLLAPSE_CONDITION = "mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard"
DEFAULT_PERCEPTUAL = "results/control_selection_perceptual_evidence.csv"
DEFAULT_OUT_CSV = "results/control_selection_recommendation.csv"
DEFAULT_OUT_MD = "results/control_selection_recommendation.md"


CSV_FIELDS = [
    "style",
    "recommendation_bucket",
    "paper_claim_status",
    "source_gate",
    "source_support",
    "source_datasets",
    "source_direct_recall",
    "source_embedding_f1",
    "generated_gate",
    "generated_target_rows",
    "generated_direct_recall",
    "generated_mean_wer",
    "generated_mean_mos_delta",
    "generated_mean_external_novelty",
    "generated_any_collapse_count",
    "perceptual_status",
    "perceptual_listener",
    "perceptual_evidence_path",
    "perceptual_summary",
    "rationale",
    "next_action",
]


BUCKET_ORDER = {
    "headline_control": 0,
    "candidate_headline_pending_listening": 1,
    "supported_but_quality_sensitive": 2,
    "diagnostic_or_limitation": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--emotion", default=DEFAULT_EMOTION)
    parser.add_argument("--wer", default=DEFAULT_WER)
    parser.add_argument("--mos", default=DEFAULT_MOS)
    parser.add_argument("--novelty", default=DEFAULT_NOVELTY)
    parser.add_argument("--collapse", default=DEFAULT_COLLAPSE)
    parser.add_argument("--collapse-condition", default=DEFAULT_COLLAPSE_CONDITION)
    parser.add_argument("--perceptual", default=DEFAULT_PERCEPTUAL)
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-md", default=DEFAULT_OUT_MD)
    return parser.parse_args()


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: Sequence[Dict[str, object]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def to_float(value: object, default: Optional[float] = None) -> Optional[float]:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: object, default: int = 0) -> int:
    parsed = to_float(value)
    return default if parsed is None else int(parsed)


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def fmt(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.4f}"


def style_sort_key(row_or_style: Dict[str, object] | str) -> tuple:
    style = row_or_style if isinstance(row_or_style, str) else str(row_or_style.get("style", ""))
    bucket_rank = 0
    if isinstance(row_or_style, dict):
        bucket_rank = BUCKET_ORDER.get(str(row_or_style.get("recommendation_bucket", "")), 99)
    try:
        style_rank = STYLE_ORDER.index(style)
    except ValueError:
        style_rank = len(STYLE_ORDER)
    return (bucket_rank, style_rank, style)


def classify_source_gate(direct_recall: Optional[float], embedding_f1: Optional[float]) -> str:
    signals = [value for value in (direct_recall, embedding_f1) if value is not None]
    if not signals:
        return "source_unmeasured"
    best = max(signals)
    if best >= 0.70:
        return "source_separable"
    if best >= 0.45:
        return "source_supported_quality_sensitive"
    if best >= 0.20:
        return "source_weak"
    return "source_limitation"


def classify_generated_gate(
    target_rows: int,
    direct_recall: Optional[float],
    mean_wer: Optional[float],
    mean_mos_delta: Optional[float],
    mean_novelty: Optional[float],
) -> str:
    wer = mean_wer if mean_wer is not None else 999.0
    mos_delta = mean_mos_delta if mean_mos_delta is not None else -999.0
    novelty = mean_novelty if mean_novelty is not None else 0.0

    if target_rows > 0:
        recall = direct_recall if direct_recall is not None else 0.0
        if recall >= 0.75 and wer <= 0.30 and mos_delta >= -0.25 and novelty >= 0.20:
            return "generated_strong"
        if recall >= 0.40 and wer <= 0.40 and mos_delta >= -0.40 and novelty >= 0.20:
            return "generated_mixed"
        if recall >= 0.20 or novelty >= 0.30:
            return "generated_diagnostic"
        return "generated_weak"

    if wer <= 0.30 and mos_delta >= -0.25 and novelty >= 0.25:
        return "generated_nonemotion_supported"
    if wer <= 0.35 and mos_delta >= -0.60 and novelty >= 0.25:
        return "generated_nonemotion_quality_sensitive"
    return "generated_nonemotion_weak"


def classify_recommendation(
    source_gate: str,
    generated_gate: str,
    perceptual_status: str,
) -> tuple[str, str, str]:
    normalized_perceptual = (perceptual_status or "needs_review").strip().lower()
    negative_perceptual = normalized_perceptual in {
        "blocked",
        "blocked_or_subtle",
        "negative",
        "reference_preferred",
    }
    positive_perceptual = normalized_perceptual in {"supported", "confirmed", "candidate_preferred"}

    if negative_perceptual:
        return (
            "diagnostic_or_limitation",
            "do not use as a headline claim",
            "Keep diagnostic unless new listening evidence overturns the negative perceptual gate.",
        )

    if source_gate in {"source_weak", "source_limitation", "source_unmeasured"}:
        return (
            "diagnostic_or_limitation",
            "do not use as a headline claim",
            "Source-label evidence is not strong enough for a central claim.",
        )

    if source_gate == "source_supported_quality_sensitive":
        return (
            "supported_but_quality_sensitive",
            "secondary or demo-only candidate; not headline yet",
            "Use only with caveats or targeted listening; source evidence is useful but not strong enough for a headline claim.",
        )

    if generated_gate in {"generated_strong", "generated_nonemotion_supported"}:
        if positive_perceptual:
            return (
                "headline_control",
                "paper-ready headline control",
                "Use as a headline claim, while still reporting objective and listening limits.",
            )
        return (
            "candidate_headline_pending_listening",
            "candidate headline control; needs focused listening before claim",
            "Queue for focused listening before promoting to paper/demo headline.",
        )

    if generated_gate in {"generated_mixed", "generated_nonemotion_quality_sensitive"}:
        return (
            "supported_but_quality_sensitive",
            "secondary or demo-only candidate; not headline yet",
            "Use only with caveats or targeted listening; do not overclaim.",
        )

    return (
        "diagnostic_or_limitation",
        "do not use as a headline claim",
        "Generated-output evidence is too weak or too costly for a paper claim.",
    )


def load_source_metrics(path: str | Path) -> Dict[str, Dict[str, object]]:
    source = {}
    for row in read_csv(path):
        direct_recall = to_float(row.get("direct_recall"))
        embedding_f1 = to_float(row.get("embedding_f1"))
        source[row["unified_style"]] = {
            "source_support": to_int(row.get("support")),
            "source_datasets": row.get("datasets", ""),
            "source_direct_recall": direct_recall,
            "source_embedding_f1": embedding_f1,
            "source_gate": classify_source_gate(direct_recall, embedding_f1),
        }
    return source


def load_generated_metrics(
    emotion_path: str | Path,
    wer_path: str | Path,
    mos_path: str | Path,
    novelty_path: str | Path,
    collapse_path: str | Path,
    collapse_condition: str,
) -> Dict[str, Dict[str, object]]:
    metrics: Dict[str, Dict[str, object]] = {style: {} for style in STYLE_ORDER}

    emotion_rows = read_csv(emotion_path)
    for style in STYLE_ORDER:
        style_rows = [row for row in emotion_rows if row.get("style") == style]
        target_rows = [row for row in style_rows if row.get("target")]
        direct_recall = None
        if target_rows:
            direct_recall = sum(to_int(row.get("match")) for row in target_rows) / len(target_rows)
        metrics[style].update(
            {
                "generated_target_rows": len(target_rows),
                "generated_direct_recall": direct_recall,
                "generated_mean_emo_sim": mean(
                    float(row["emo_sim"]) for row in style_rows if row.get("emo_sim")
                ),
            }
        )

    wer_rows = read_csv(wer_path)
    for style in STYLE_ORDER:
        style_rows = [row for row in wer_rows if row.get("style") == style]
        metrics[style]["generated_mean_wer"] = mean(float(row["wer"]) for row in style_rows)

    mos_rows = read_csv(mos_path)
    for style in STYLE_ORDER:
        style_rows = [row for row in mos_rows if row.get("style") == style and row.get("delta_vs_baseline")]
        metrics[style]["generated_mean_mos_delta"] = mean(float(row["delta_vs_baseline"]) for row in style_rows)

    novelty_rows = read_csv(novelty_path)
    for style in STYLE_ORDER:
        style_rows = [row for row in novelty_rows if row.get("style") == style]
        metrics[style]["generated_mean_external_novelty"] = mean(
            float(row["external_novelty_gain_vs_baseline"]) for row in style_rows
        )

    collapse_rows = [
        row for row in read_csv(collapse_path) if row.get("condition") == collapse_condition
    ]
    for style in STYLE_ORDER:
        style_rows = [row for row in collapse_rows if row.get("style") == style]
        metrics[style]["generated_any_collapse_count"] = sum(
            1
            for row in style_rows
            if any(
                to_int(row.get(flag))
                for flag in (
                    "content_collapse",
                    "style_collapse_to_neutral",
                    "identity_collapse_to_baseline",
                    "mixed_collapse",
                )
            )
        )

    for style, row in metrics.items():
        row["generated_gate"] = classify_generated_gate(
            int(row.get("generated_target_rows", 0)),
            row.get("generated_direct_recall"),
            row.get("generated_mean_wer"),
            row.get("generated_mean_mos_delta"),
            row.get("generated_mean_external_novelty"),
        )
    return metrics


def load_perceptual_evidence(path: str | Path) -> Dict[str, Dict[str, str]]:
    evidence = {}
    for row in read_csv(path):
        evidence[row["style"]] = {
            "perceptual_status": row.get("perceptual_status", "needs_review"),
            "perceptual_listener": row.get("listener", ""),
            "perceptual_evidence_path": row.get("evidence_path", ""),
            "perceptual_summary": row.get("summary", ""),
        }
    return evidence


def build_rationale(row: Dict[str, object]) -> str:
    pieces = []
    pieces.append(str(row["source_gate"]).replace("_", " "))
    pieces.append(str(row["generated_gate"]).replace("_", " "))
    if row.get("perceptual_status"):
        pieces.append(f"perceptual status: {row['perceptual_status']}")
    return "; ".join(pieces)


def build_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    source = load_source_metrics(args.source)
    generated = load_generated_metrics(
        args.emotion,
        args.wer,
        args.mos,
        args.novelty,
        args.collapse,
        args.collapse_condition,
    )
    perceptual = load_perceptual_evidence(args.perceptual)

    rows: List[Dict[str, object]] = []
    for style in STYLE_ORDER:
        row: Dict[str, object] = {"style": style}
        row.update(source.get(style, {}))
        row.update(generated.get(style, {}))
        row.update(
            perceptual.get(
                style,
                {
                    "perceptual_status": "needs_review",
                    "perceptual_listener": "",
                    "perceptual_evidence_path": "",
                    "perceptual_summary": "No style-specific listening evidence recorded.",
                },
            )
        )
        bucket, claim_status, next_action = classify_recommendation(
            str(row.get("source_gate", "source_unmeasured")),
            str(row.get("generated_gate", "generated_weak")),
            str(row.get("perceptual_status", "needs_review")),
        )
        row["recommendation_bucket"] = bucket
        row["paper_claim_status"] = claim_status
        row["rationale"] = build_rationale(row)
        row["next_action"] = next_action
        rows.append(row)

    return sorted(rows, key=style_sort_key)


def csv_ready(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    formatted = []
    for row in rows:
        out = dict(row)
        for key in (
            "source_direct_recall",
            "source_embedding_f1",
            "generated_direct_recall",
            "generated_mean_wer",
            "generated_mean_mos_delta",
            "generated_mean_external_novelty",
        ):
            out[key] = fmt(out.get(key))
        formatted.append(out)
    return formatted


def write_markdown(rows: Sequence[Dict[str, object]], path: str | Path, args: argparse.Namespace) -> None:
    counts = Counter(str(row["recommendation_bucket"]) for row in rows)
    lines = [
        "# Control Selection Recommendation",
        "",
        "This recommendation intersects source training-label separability, generated-output metrics for the current `sad/enunciated` reference guard, and available human listening evidence.",
        "It is deliberately conservative: a source-separable label is not promoted unless generated audio and listening evidence also support the claim.",
        "",
        "## Inputs",
        "",
        f"- Source separability: `{args.source}`",
        f"- Generated emotion metrics: `{args.emotion}`",
        f"- Generated WER metrics: `{args.wer}`",
        f"- Generated MOS metrics: `{args.mos}`",
        f"- External speaker novelty: `{args.novelty}`",
        f"- Collapse diagnostics: `{args.collapse}` (`{args.collapse_condition}`)",
        f"- Perceptual evidence ledger: `{args.perceptual}`",
        "",
        "## Bucket Counts",
        "",
        "| Bucket | Count |",
        "| --- | ---: |",
    ]
    for bucket in sorted(counts, key=lambda item: BUCKET_ORDER.get(item, 99)):
        lines.append(f"| `{bucket}` | `{counts[bucket]}` |")

    lines.extend(
        [
            "",
            "## Recommendation Table",
            "",
            "| Style | Bucket | Source gate | Generated gate | Generated recall | WER | MOS delta | External novelty | Perceptual status | Claim status |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| `{style}` | `{bucket}` | `{source}` | `{generated}` | {recall} | {wer} | {mos} | {novelty} | `{perceptual}` | {claim} |".format(
                style=row["style"],
                bucket=row["recommendation_bucket"],
                source=row.get("source_gate", ""),
                generated=row.get("generated_gate", ""),
                recall=fmt(row.get("generated_direct_recall")) or "n/a",
                wer=fmt(row.get("generated_mean_wer")) or "n/a",
                mos=fmt(row.get("generated_mean_mos_delta")) or "n/a",
                novelty=fmt(row.get("generated_mean_external_novelty")) or "n/a",
                perceptual=row.get("perceptual_status", ""),
                claim=row.get("paper_claim_status", ""),
            )
        )

    lines.extend(["", "## Interpretation", ""])
    headline = [row for row in rows if row["recommendation_bucket"] == "headline_control"]
    pending = [row for row in rows if row["recommendation_bucket"] == "candidate_headline_pending_listening"]
    supported = [row for row in rows if row["recommendation_bucket"] == "supported_but_quality_sensitive"]
    diagnostic = [row for row in rows if row["recommendation_bucket"] == "diagnostic_or_limitation"]

    if headline:
        lines.append("Paper-ready headline controls:")
        lines.append("")
        for row in headline:
            lines.append(f"- `{row['style']}`: {row['paper_claim_status']}")
    else:
        lines.append("No style control is promoted as fully paper-ready by this conservative gate yet, because no current style has both strong objective evidence and positive focused listening evidence recorded.")

    if pending:
        lines.extend(["", "Candidate headline controls needing focused listening:", ""])
        for row in pending:
            lines.append(f"- `{row['style']}`: {row['perceptual_summary']}")

    if supported:
        lines.extend(["", "Supported but quality-sensitive controls:", ""])
        for row in supported:
            lines.append(f"- `{row['style']}`: {row['rationale']}")

    if diagnostic:
        lines.extend(["", "Diagnostic / limitation controls:", ""])
        for row in diagnostic:
            lines.append(f"- `{row['style']}`: {row['perceptual_summary'] or row['rationale']}")

    lines.extend(
        [
            "",
            "## Next Listening Queue",
            "",
            "Use the existing browser report for the current reference guard:",
            "",
            "```bash",
            "python3 -m http.server 8000",
            "# open http://localhost:8000/results/listening_mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard.html",
            "```",
            "",
            "Listen first to the `candidate_headline_pending_listening` rows, then the quality-sensitive rows. A control should become a headline claim only if a listener can hear the intended control without major intelligibility or naturalness loss.",
            "",
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> List[Dict[str, object]]:
    rows = build_rows(args)
    write_csv(csv_ready(rows), args.out_csv)
    write_markdown(rows, args.out_md, args)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")
    for bucket, count in Counter(row["recommendation_bucket"] for row in rows).items():
        print(f"{bucket}: {count}")
    return rows


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
