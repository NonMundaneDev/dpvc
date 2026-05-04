"""
Annotate CommonVoice embedding artifacts with confidence-scored pseudo-style labels.

This uses emotion2vec to assign pseudo labels for the overlapping emotional
styles supported by the controllable OpenVoice pipeline:

  angry -> anger
  disgusted -> disgust
  fearful -> fear
  happy -> happy
  neutral -> neutral
  sad -> sad

Rows with other / <unk> predictions remain unlabeled. The raw mapped label and
confidence are always stored; downstream pretraining can apply its own
confidence threshold without re-running the annotation pass.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch
from funasr import AutoModel
from tqdm import tqdm

from dpvc import utils


E2V_TO_STYLE = {
    'angry': 'anger',
    'disgusted': 'disgust',
    'fearful': 'fear',
    'happy': 'happy',
    'neutral': 'neutral',
    'sad': 'sad',
}


def canonical_label(raw):
    if '/' in raw:
        raw = raw.split('/')[-1]
    return raw.strip().lower()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--embeddings',
        default='embeddings/openvoice_commonvoice_cv500_emb.pt',
        help='Path to CommonVoice embedding artifact',
    )
    ap.add_argument(
        '--output',
        default='embeddings/openvoice_commonvoice_cv500_pseudo.pt',
        help='Output path for enriched artifact',
    )
    ap.add_argument(
        '--model',
        default='iic/emotion2vec_plus_large',
        help='funasr model id (default: iic/emotion2vec_plus_large)',
    )
    ap.add_argument(
        '--teacher-checkpoint',
        default='',
        help='Optional local checkpoint or teacher identifier recorded in the artifact metadata',
    )
    ap.add_argument(
        '--teacher-config',
        default='',
        help='Optional path or note describing the teacher config recorded in the artifact metadata',
    )
    ap.add_argument(
        '--report-threshold',
        type=float,
        default=0.60,
        help='Confidence threshold used only for the printed/saved coverage report (default: 0.60)',
    )
    ap.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Optional limit for smoke runs',
    )
    ap.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='How many ranked teacher predictions to store per row (default: 3)',
    )
    ap.add_argument(
        '--save-style-score-map',
        action='store_true',
        help='Save mapped per-style score dictionaries for later class-aware filtering',
    )
    return ap.parse_args()


def resolve_clip_path(corpus_path, clip_rel):
    clip_rel_path = Path(clip_rel)
    if clip_rel_path.is_absolute():
        return clip_rel_path
    return Path(corpus_path) / clip_rel_path


def main():
    args = parse_args()
    data = torch.load(args.embeddings, weights_only=False)

    clip_paths = data['clip_paths']
    corpus_path = data['corpus_path']
    total_rows = len(clip_paths)
    if args.limit is not None:
        total_rows = min(total_rows, args.limit)

    print(f'Loading {args.model} for pseudo-label annotation...')
    model = AutoModel(model=args.model, hub='hf', disable_update=True)
    print('Model loaded.')

    pseudo_style = [None] * len(clip_paths)
    pseudo_style_confidence = [None] * len(clip_paths)
    pseudo_style_raw_label = [None] * len(clip_paths)
    pseudo_style_topk_labels = [None] * len(clip_paths)
    pseudo_style_topk_scores = [None] * len(clip_paths)
    pseudo_style_score_map = [None] * len(clip_paths)

    raw_counts = Counter()
    mapped_counts = Counter()
    accepted_counts = Counter()
    mapped_score_totals = defaultdict(float)

    for idx, clip_rel in enumerate(tqdm(clip_paths[:total_rows], desc='Pseudo labels')):
        clip_path = resolve_clip_path(corpus_path, clip_rel)
        rec = model.generate(str(clip_path), granularity='utterance', extract_embedding=False)
        row = rec[0]
        labels = [canonical_label(label) for label in row['labels']]
        scores = [float(score) for score in row['scores']]
        top_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        raw_label = labels[top_idx]
        confidence = float(scores[top_idx])
        mapped_style = E2V_TO_STYLE.get(raw_label)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        topk_indexes = ranked[: max(args.top_k, 1)]
        score_map = defaultdict(float)
        for label, score in zip(labels, scores):
            mapped = E2V_TO_STYLE.get(label)
            if mapped is not None:
                score_map[mapped] += float(score)
                mapped_score_totals[mapped] += float(score)

        pseudo_style[idx] = mapped_style
        pseudo_style_confidence[idx] = confidence
        pseudo_style_raw_label[idx] = raw_label
        pseudo_style_topk_labels[idx] = [labels[i] for i in topk_indexes]
        pseudo_style_topk_scores[idx] = [float(scores[i]) for i in topk_indexes]
        if args.save_style_score_map:
            pseudo_style_score_map[idx] = {
                style: float(score_map.get(style, 0.0))
                for style in sorted(E2V_TO_STYLE.values())
                if score_map.get(style, 0.0) > 0
            }

        raw_counts[raw_label] += 1
        if mapped_style:
            mapped_counts[mapped_style] += 1
            if confidence >= args.report_threshold:
                accepted_counts[mapped_style] += 1

    metadata_report = data.get('metadata_report')
    if metadata_report is None:
        metadata_report = utils.build_commonvoice_metadata_report(
            data.get('age', []),
            data.get('gender', []),
            data.get('accent', []),
        )

    pseudo_style_report = {
        'model': args.model,
        'teacher_checkpoint': args.teacher_checkpoint or None,
        'teacher_config': args.teacher_config or None,
        'report_threshold': args.report_threshold,
        'top_k': args.top_k,
        'save_style_score_map': bool(args.save_style_score_map),
        'rows_annotated': total_rows,
        'raw_label_counts': dict(raw_counts),
        'mapped_style_counts': dict(mapped_counts),
        'accepted_style_counts': dict(accepted_counts),
        'mapped_style_score_totals': {
            style: float(score)
            for style, score in sorted(mapped_score_totals.items())
        },
    }

    enriched = dict(data)
    enriched['pseudo_style'] = pseudo_style
    enriched['pseudo_style_confidence'] = pseudo_style_confidence
    enriched['pseudo_style_raw_label'] = pseudo_style_raw_label
    enriched['pseudo_style_topk_labels'] = pseudo_style_topk_labels
    enriched['pseudo_style_topk_scores'] = pseudo_style_topk_scores
    if args.save_style_score_map:
        enriched['pseudo_style_score_map'] = pseudo_style_score_map
    enriched['pseudo_style_source'] = args.model
    enriched['pseudo_style_teacher'] = {
        'model': args.model,
        'teacher_checkpoint': args.teacher_checkpoint or None,
        'teacher_config': args.teacher_config or None,
        'top_k': args.top_k,
        'save_style_score_map': bool(args.save_style_score_map),
    }
    enriched['pseudo_style_report'] = pseudo_style_report
    enriched['metadata_report'] = metadata_report

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(enriched, output_path)

    print(f'Saved pseudo-labeled CommonVoice artifact to {output_path}')
    print(f'Rows annotated: {total_rows}/{len(clip_paths)}')
    print('Accepted pseudo-style counts at report threshold:')
    for style, count in sorted(accepted_counts.items()):
        print(f'  {style:8s}: {count}')


if __name__ == '__main__':
    main()
