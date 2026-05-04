"""
Filter or rebalance scored CommonVoice pseudo labels without re-running the teacher.

This script takes an artifact produced by `annotate_commonvoice_pseudolabels.py`
and records a reusable row-level acceptance decision. Downstream builders can use
that decision directly via `--acceptance-policy artifact_selected`.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch


UNIFIED_STYLES = [
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


def parse_style_floats(raw: str, default: float) -> dict[str, float]:
    values = {style: float(default) for style in UNIFIED_STYLES}
    if not raw.strip():
        return values
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(f"Expected style=value, got: {item}")
        style, value = item.split('=', 1)
        style = style.strip()
        if style not in UNIFIED_STYLES:
            raise ValueError(f"Unknown style: {style}")
        values[style] = float(value.strip())
    return values


def parse_style_ints(raw: str) -> dict[str, int]:
    if not raw.strip():
        return {}
    values = {}
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(f"Expected style=count, got: {item}")
        style, value = item.split('=', 1)
        style = style.strip()
        if style not in UNIFIED_STYLES:
            raise ValueError(f"Unknown style: {style}")
        values[style] = int(value.strip())
    return values


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--input',
        default='embeddings/openvoice_commonvoice_cv500_pseudo.pt',
        help='Annotated CommonVoice artifact from annotate_commonvoice_pseudolabels.py',
    )
    ap.add_argument(
        '--output',
        default='embeddings/openvoice_commonvoice_cv500_pseudo_filtered.pt',
        help='Output artifact with row-level pseudo-label acceptance decisions',
    )
    ap.add_argument(
        '--default-threshold',
        type=float,
        default=0.60,
        help='Default per-style confidence threshold (default: 0.60)',
    )
    ap.add_argument(
        '--style-thresholds',
        default='',
        help='Optional per-style thresholds, e.g. neutral=0.995,sad=0.98,happy=0.92',
    )
    ap.add_argument(
        '--style-caps',
        default='',
        help='Optional per-style caps after thresholding, e.g. neutral=150,sad=120',
    )
    ap.add_argument(
        '--style-targets',
        default='',
        help='Optional per-style selected-row targets, e.g. anger=30,fear=30,happy=80',
    )
    ap.add_argument(
        '--acceptance-policy',
        default='threshold_plus_caps',
        choices=['confidence_only', 'threshold_plus_caps', 'balanced_targets'],
        help='Row-level acceptance policy (default: threshold_plus_caps)',
    )
    return ap.parse_args()


def select_rows(data, thresholds, caps, targets, acceptance_policy):
    pseudo_style = data.get('pseudo_style', [])
    pseudo_confidence = data.get('pseudo_style_confidence', [])
    total_rows = len(pseudo_style)

    selected_style = [None] * total_rows
    selected_mask = [False] * total_rows
    selected_reason = ['unscored'] * total_rows
    selected_confidence = [0.0] * total_rows

    style_to_rows = defaultdict(list)
    threshold_rejected = Counter()
    unmapped_rows = 0
    missing_conf_rows = 0

    for idx, (style, confidence) in enumerate(zip(pseudo_style, pseudo_confidence)):
        if style not in UNIFIED_STYLES:
            selected_reason[idx] = 'unmapped_or_missing_style'
            unmapped_rows += 1
            continue
        if confidence is None:
            selected_reason[idx] = 'missing_confidence'
            missing_conf_rows += 1
            continue
        confidence = float(confidence)
        selected_confidence[idx] = confidence
        if confidence < thresholds.get(style, 0.0):
            selected_reason[idx] = 'below_threshold'
            threshold_rejected[style] += 1
            continue
        style_to_rows[style].append((idx, confidence))

    selected_counts = Counter()
    cap_rejected = Counter()
    target_rejected = Counter()

    for style in UNIFIED_STYLES:
        candidates = sorted(style_to_rows.get(style, []), key=lambda item: item[1], reverse=True)
        if acceptance_policy == 'confidence_only':
            limit = len(candidates)
        elif acceptance_policy == 'threshold_plus_caps':
            limit = caps.get(style, len(candidates))
        elif acceptance_policy == 'balanced_targets':
            limit = targets.get(style, caps.get(style, len(candidates)))
        else:
            raise ValueError(f'Unsupported acceptance policy: {acceptance_policy}')

        for rank, (idx, confidence) in enumerate(candidates):
            if rank < limit:
                selected_style[idx] = style
                selected_mask[idx] = True
                selected_reason[idx] = 'selected'
                selected_confidence[idx] = float(confidence)
                selected_counts[style] += 1
            else:
                if acceptance_policy == 'balanced_targets' and style in targets:
                    selected_reason[idx] = 'target_limited'
                    target_rejected[style] += 1
                else:
                    selected_reason[idx] = 'cap_limited'
                    cap_rejected[style] += 1

    return {
        'selected_style': selected_style,
        'selected_mask': selected_mask,
        'selected_reason': selected_reason,
        'selected_confidence': selected_confidence,
        'threshold_rejected_counts': dict(threshold_rejected),
        'cap_rejected_counts': dict(cap_rejected),
        'target_rejected_counts': dict(target_rejected),
        'selected_counts': dict(selected_counts),
        'unmapped_rows': unmapped_rows,
        'missing_confidence_rows': missing_conf_rows,
    }


def main():
    args = parse_args()
    data = torch.load(args.input, weights_only=False)

    thresholds = parse_style_floats(args.style_thresholds, args.default_threshold)
    caps = parse_style_ints(args.style_caps)
    targets = parse_style_ints(args.style_targets)

    selection = select_rows(data, thresholds, caps, targets, args.acceptance_policy)

    enriched = dict(data)
    enriched['pseudo_style_selected'] = selection['selected_style']
    enriched['pseudo_style_selected_mask'] = torch.tensor(selection['selected_mask'], dtype=torch.bool)
    enriched['pseudo_style_selected_reason'] = selection['selected_reason']
    enriched['pseudo_style_selected_confidence'] = torch.tensor(
        selection['selected_confidence'], dtype=torch.float32
    ).unsqueeze(1)
    enriched['pseudo_style_filter_report'] = {
        'acceptance_policy': args.acceptance_policy,
        'default_threshold': float(args.default_threshold),
        'thresholds': thresholds,
        'style_caps': caps,
        'style_targets': targets,
        'selected_counts': selection['selected_counts'],
        'threshold_rejected_counts': selection['threshold_rejected_counts'],
        'cap_rejected_counts': selection['cap_rejected_counts'],
        'target_rejected_counts': selection['target_rejected_counts'],
        'unmapped_rows': int(selection['unmapped_rows']),
        'missing_confidence_rows': int(selection['missing_confidence_rows']),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(enriched, output_path)

    print(f'Saved filtered CommonVoice artifact to {output_path}')
    print(f'Acceptance policy: {args.acceptance_policy}')
    print('Selected counts:')
    for style, count in sorted(selection['selected_counts'].items()):
        print(f'  {style:8s}: {count}')


if __name__ == '__main__':
    main()
