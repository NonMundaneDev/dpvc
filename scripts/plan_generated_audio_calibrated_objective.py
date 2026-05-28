#!/usr/bin/env python3
"""
Build a trainer-ready objective plan from generated-audio evidence.

The content-repair gate says whether strength-grid candidates are safe to
promote as presets. The failure-conditioned target selector says which hard
styles have clean generated-audio failures worth training against. This script
combines both signals into conservative mixed-trainer overrides.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


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

DEFAULT_GATE_JSON = "results/generated_audio_content_repair_gate.json"
DEFAULT_FAILURE_TARGET_JSON = "results/eval_mixed_teacher_failure_conditioned_targets.json"
DEFAULT_OUT_JSON = "results/generated_audio_calibrated_objective_plan.json"
DEFAULT_OUT_MD = "results/generated_audio_calibrated_objective_plan.md"
DEFAULT_OUT_CSV = "results/generated_audio_calibrated_objective_plan.csv"


@dataclass(frozen=True)
class ObjectiveConfig:
    diagnostic_weight: float = 2.0
    content_repair_weight: float = 3.0
    blocked_weight: float = 0.0
    training_strength: float = 5.0
    output_checkpoint: str = (
        "embeddings/openvoice_vae_mixed_teacher_cvrare_audio_calibrated_labeled_warmup.pt"
    )
    objective_plan_path: str = DEFAULT_OUT_JSON
    embeddings: str = "embeddings/openvoice_mixed_teacher_cvrare_hybrid_extra_base.pt"
    style_teacher_checkpoint: str = "embeddings/openvoice_vae_combined.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-json", default=DEFAULT_GATE_JSON)
    parser.add_argument("--failure-target-json", default=DEFAULT_FAILURE_TARGET_JSON)
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", default=DEFAULT_OUT_MD)
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    parser.add_argument(
        "--diagnostic-weight",
        type=float,
        default=ObjectiveConfig.diagnostic_weight,
    )
    parser.add_argument(
        "--content-repair-weight",
        type=float,
        default=ObjectiveConfig.content_repair_weight,
    )
    parser.add_argument(
        "--blocked-weight",
        type=float,
        default=ObjectiveConfig.blocked_weight,
    )
    parser.add_argument(
        "--training-strength",
        type=float,
        default=ObjectiveConfig.training_strength,
    )
    parser.add_argument("--embeddings", default=ObjectiveConfig.embeddings)
    parser.add_argument("--output-checkpoint", default=ObjectiveConfig.output_checkpoint)
    parser.add_argument(
        "--style-teacher-checkpoint",
        default=ObjectiveConfig.style_teacher_checkpoint,
    )
    return parser.parse_args()


def read_json(path: str | Path) -> Dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def style_sort_key(style: str) -> tuple:
    try:
        return (STYLE_ORDER.index(style), style)
    except ValueError:
        return (len(STYLE_ORDER), style)


def format_number(value: float) -> str:
    return f"{float(value):g}"


def format_style_map(values: Mapping[str, float], styles: Iterable[str] = STYLE_ORDER) -> str:
    return ",".join(
        f"{style}={format_number(values.get(style, 0.0))}" for style in styles
    )


def blocked_style_lookup(failure_summary: Mapping) -> Dict[str, Dict]:
    return {
        str(item.get("style", "")): dict(item)
        for item in failure_summary.get("blocked_styles", [])
        if item.get("style")
    }


def choose_style_action(
    style: str,
    gate_decision: str,
    ready_styles: set[str],
    blocked_styles: Mapping[str, Dict],
    config: ObjectiveConfig,
) -> Dict[str, object]:
    if style not in ready_styles:
        blocked = blocked_styles.get(style, {})
        return {
            "style": style,
            "selected": False,
            "action": "blocked_no_clean_targets",
            "weight": config.blocked_weight,
            "strength": 0.0,
            "reason": (
                f"failure target selector found {blocked.get('selected', 0)}/"
                f"{blocked.get('total', 0)} clean rows"
            ),
        }

    if gate_decision == "needs_content_repair":
        return {
            "style": style,
            "selected": True,
            "action": "train_content_repair",
            "weight": config.content_repair_weight,
            "strength": config.training_strength,
            "reason": "gate found no safe preset, but clean generated-audio failures exist",
        }

    if gate_decision in {"diagnostic_only", "needs_more_listening"}:
        return {
            "style": style,
            "selected": True,
            "action": "train_conservative_repair",
            "weight": config.diagnostic_weight,
            "strength": config.training_strength,
            "reason": "gate keeps preset diagnostic, but clean generated-audio failures exist",
        }

    if gate_decision == "promote_candidate":
        return {
            "style": style,
            "selected": False,
            "action": "hold_promoted_candidate",
            "weight": config.blocked_weight,
            "strength": 0.0,
            "reason": "style already has a promoted preset candidate; do not train repair here",
        }

    return {
        "style": style,
        "selected": False,
        "action": "blocked_by_gate",
        "weight": config.blocked_weight,
        "strength": 0.0,
        "reason": f"unsupported or absent gate decision: {gate_decision or 'missing'}",
    }


def build_recommended_command(config: ObjectiveConfig) -> str:
    parts = [
        "python examples/openvoice_train_vae_mixed.py",
        f"--embeddings {config.embeddings}",
        f"--output {config.output_checkpoint}",
        "--schedule labeled_warmup",
        "--schedule-epochs 1500",
        "--epochs 3000",
        "--lr 1e-6",
        "--style-teacher-checkpoint " + config.style_teacher_checkpoint,
        "--style-teacher-weight 0.0",
        "--style-teacher-weight-final 0.25",
        "--decoder-prototype-weight 0.0",
        "--decoder-prototype-weight-final 0.005",
        "--decoder-prototype-datasets CommonVoice",
        "--decoder-prototype-source true",
        "--anti-neutral-weight 0.0",
        "--anti-neutral-weight-final 0.005",
        "--anti-neutral-mode prototype_margin",
        "--anti-neutral-datasets CommonVoice",
        f"--generated-audio-objective-plan {config.objective_plan_path}",
        "--generated-audio-objective-report "
        "results/generated_audio_calibrated_objective_training_report.json",
    ]
    return " \\\n  ".join(parts)


def build_objective_plan(
    gate_summary: Mapping,
    failure_summary: Mapping,
    config: ObjectiveConfig,
) -> Dict[str, object]:
    style_summary = gate_summary.get("style_summary", {})
    ready_styles = set(failure_summary.get("ready_styles", []))
    blocked_styles = blocked_style_lookup(failure_summary)
    candidate_styles = sorted(
        set(style_summary) | ready_styles | set(blocked_styles),
        key=style_sort_key,
    )

    style_plan = {}
    for style in candidate_styles:
        gate_decision = style_summary.get(style, {}).get("style_decision", "")
        style_plan[style] = choose_style_action(
            style,
            gate_decision,
            ready_styles,
            blocked_styles,
            config,
        )

    selected_styles = [
        style for style in candidate_styles if bool(style_plan[style]["selected"])
    ]
    blocked_selected = [
        style for style in candidate_styles if not bool(style_plan[style]["selected"])
    ]
    style_weights = {
        style: float(style_plan.get(style, {}).get("weight", config.blocked_weight))
        for style in STYLE_ORDER
    }
    style_strengths = {
        style: float(style_plan.get(style, {}).get("strength", 0.0))
        for style in STYLE_ORDER
    }
    trainer_overrides = {
        "style_teacher_target_mode": "target_dim",
        "style_teacher_require_label": True,
        "style_teacher_style_weights": format_style_map(style_weights),
        "decoder_prototype_style_weights": format_style_map(style_weights),
        "decoder_prototype_style_strengths": format_style_map(style_strengths),
        "anti_neutral_styles": ",".join(selected_styles),
        "anti_neutral_style_weights": format_style_map(style_weights),
        "anti_neutral_style_strengths": format_style_map(style_strengths),
    }

    return {
        "objective_name": "generated_audio_calibrated_hard_style_repair",
        "reference_condition": failure_summary.get("reference_condition", ""),
        "source_gate": DEFAULT_GATE_JSON,
        "source_failure_targets": DEFAULT_FAILURE_TARGET_JSON,
        "selected_styles": selected_styles,
        "blocked_styles": blocked_selected,
        "style_plan": style_plan,
        "trainer_overrides": trainer_overrides,
        "recommended_command": build_recommended_command(config),
        "notes": [
            "This plan is for training repair, not preset promotion.",
            "A style must have clean generated-audio failures before receiving loss pressure.",
            "Styles blocked by Joe's perceptual review stay diagnostic until new audio wins.",
        ],
    }


def write_json(path: str | Path, payload: Mapping) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: str | Path, plan: Mapping) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["style", "selected", "action", "weight", "strength", "reason"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for style in sorted(plan["style_plan"], key=style_sort_key):
            writer.writerow(plan["style_plan"][style])


def write_markdown(path: str | Path, plan: Mapping) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# Generated-Audio-Calibrated Objective Plan",
        "",
        "This plan converts generated-audio evidence into trainer overrides.",
        "",
        f"Reference condition: `{plan['reference_condition']}`",
        "",
        "## Selected Styles",
        "",
        ", ".join(f"`{style}`" for style in plan["selected_styles"]) or "-",
        "",
        "## Style Plan",
        "",
        "| Style | Selected | Action | Weight | Strength | Reason |",
        "|-------|----------|--------|-------:|---------:|--------|",
    ]
    for style in sorted(plan["style_plan"], key=style_sort_key):
        row = plan["style_plan"][style]
        lines.append(
            f"| `{style}` | `{row['selected']}` | `{row['action']}` | "
            f"`{format_number(row['weight'])}` | `{format_number(row['strength'])}` | "
            f"{row['reason']} |"
        )

    lines.extend([
        "",
        "## Trainer Overrides",
        "",
    ])
    for key, value in plan["trainer_overrides"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend([
        "",
        "## Recommended Command",
        "",
        "```bash",
        plan["recommended_command"],
        "```",
        "",
        "## Interpretation",
        "",
        (
            "- `anger` receives conservative repair pressure because it has clean "
            "failures, but the strength preset did not win perceptually."
        ),
        (
            "- `disgust` receives content-repair pressure because no safe "
            "strength-grid row passed the gate."
        ),
        (
            "- `fear` remains blocked from this training objective because its "
            "clean-failure target supply is not established and Joe heard an "
            "unnatural pitch change in the best metric row."
        ),
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = ObjectiveConfig(
        diagnostic_weight=args.diagnostic_weight,
        content_repair_weight=args.content_repair_weight,
        blocked_weight=args.blocked_weight,
        training_strength=args.training_strength,
        output_checkpoint=args.output_checkpoint,
        objective_plan_path=args.out_json,
        embeddings=args.embeddings,
        style_teacher_checkpoint=args.style_teacher_checkpoint,
    )
    gate_summary = read_json(args.gate_json)
    failure_summary = read_json(args.failure_target_json)
    plan = build_objective_plan(gate_summary, failure_summary, config)
    plan["source_gate"] = args.gate_json
    plan["source_failure_targets"] = args.failure_target_json

    write_json(args.out_json, plan)
    write_csv(args.out_csv, plan)
    write_markdown(args.out_md, plan)

    print(f"Wrote generated-audio objective plan JSON to {args.out_json}")
    print(f"Wrote generated-audio objective plan CSV to {args.out_csv}")
    print(f"Wrote generated-audio objective plan summary to {args.out_md}")


if __name__ == "__main__":
    main()
