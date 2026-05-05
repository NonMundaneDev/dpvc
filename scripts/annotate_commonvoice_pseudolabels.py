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
from tempfile import NamedTemporaryFile
from typing import Iterable

import soundfile as sf
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
        '--batch-size',
        type=int,
        default=8,
        help='Number of clips passed to emotion2vec per model.generate call (default: 8)',
    )
    ap.add_argument(
        '--checkpoint-every',
        type=int,
        default=1000,
        help='Save a resumable checkpoint after this many newly annotated rows (default: 1000)',
    )
    ap.add_argument(
        '--checkpoint-path',
        default=None,
        help='Optional explicit checkpoint path (default: <output>.checkpoint.pt)',
    )
    ap.add_argument(
        '--no-resume',
        action='store_true',
        help='Ignore an existing checkpoint and start annotation from scratch',
    )
    ap.add_argument(
        '--fail-on-error',
        action='store_true',
        help='Abort on clip scoring errors instead of recording the error and continuing',
    )
    ap.add_argument(
        '--stop-when-accepted-targets',
        default='',
        help=(
            'Optional comma-separated accepted-count targets, e.g. '
            'anger=50,fear=50. Stops after a batch once all targets are met.'
        ),
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
    ap.add_argument(
        '--consistency-view',
        default='none',
        choices=['none', 'center_crop'],
        help='Optional second view scored for teacher-consistency agreement (default: none)',
    )
    ap.add_argument(
        '--consistency-crop-frac',
        type=float,
        default=0.8,
        help='Fraction of the clip kept in the center-crop consistency view (default: 0.8)',
    )
    ap.add_argument(
        '--consistency-min-seconds',
        type=float,
        default=1.0,
        help='Minimum duration kept for the consistency crop view (default: 1.0)',
    )
    return ap.parse_args()


def resolve_clip_path(corpus_path, clip_rel):
    clip_rel_path = Path(clip_rel)
    if clip_rel_path.is_absolute():
        return clip_rel_path
    return Path(corpus_path) / clip_rel_path


def score_row(row, top_k):
    labels = [canonical_label(label) for label in row['labels']]
    scores = [float(score) for score in row['scores']]
    top_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    topk_indexes = ranked[: max(top_k, 1)]
    score_map = defaultdict(float)
    for label, score in zip(labels, scores):
        mapped = E2V_TO_STYLE.get(label)
        if mapped is not None:
            score_map[mapped] += float(score)
    return {
        'labels': labels,
        'scores': scores,
        'raw_label': labels[top_idx],
        'confidence': float(scores[top_idx]),
        'mapped_style': E2V_TO_STYLE.get(labels[top_idx]),
        'topk_labels': [labels[i] for i in topk_indexes],
        'topk_scores': [float(scores[i]) for i in topk_indexes],
        'style_score_map': {
            style: float(score_map.get(style, 0.0))
            for style in sorted(E2V_TO_STYLE.values())
            if score_map.get(style, 0.0) > 0
        },
    }


def score_clip(model, clip_path, top_k):
    rec = model.generate(str(clip_path), granularity='utterance', extract_embedding=False)
    return score_row(rec[0], top_k)


def score_clip_batch(model, clip_paths, top_k, batch_size):
    recs = model.generate(
        [str(path) for path in clip_paths],
        granularity='utterance',
        extract_embedding=False,
        batch_size=batch_size,
    )
    if len(recs) != len(clip_paths):
        raise RuntimeError(
            f'emotion2vec returned {len(recs)} rows for {len(clip_paths)} inputs'
        )
    return [score_row(row, top_k) for row in recs]


def build_consistency_view(clip_path, mode, crop_frac, min_seconds):
    if mode == 'none':
        return None
    if mode != 'center_crop':
        raise ValueError(f'Unsupported consistency view: {mode}')
    audio, sample_rate = sf.read(str(clip_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    total = len(audio)
    if total == 0:
        return None
    min_samples = max(int(min_seconds * sample_rate), 1)
    crop_samples = max(int(total * crop_frac), min_samples)
    crop_samples = min(crop_samples, total)
    if crop_samples >= total:
        return None
    start = max((total - crop_samples) // 2, 0)
    end = start + crop_samples
    crop = audio[start:end]
    if len(crop) < min_samples:
        return None
    with NamedTemporaryFile(suffix='.wav', delete=False) as handle:
        temp_path = Path(handle.name)
    sf.write(temp_path, crop, sample_rate)
    return temp_path


def none_list(size):
    return [None] * size


def copy_list(source, key, size):
    values = source.get(key)
    if not isinstance(values, list):
        return none_list(size)
    if len(values) < size:
        return values + [None] * (size - len(values))
    return values[:size]


def output_checkpoint_path(output_path, checkpoint_path):
    if checkpoint_path:
        return Path(checkpoint_path)
    return output_path.with_suffix(output_path.suffix + '.checkpoint.pt')


def parse_style_targets(spec):
    targets = {}
    if not spec:
        return targets
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(
                f'Invalid --stop-when-accepted-targets item {item!r}; expected style=count'
            )
        style, value = item.split('=', 1)
        style = style.strip()
        if style not in set(E2V_TO_STYLE.values()):
            raise ValueError(f'Unknown target style {style!r}')
        targets[style] = int(value)
        if targets[style] < 1:
            raise ValueError(f'Target for {style!r} must be >= 1')
    return targets


def iter_batches(items: Iterable, batch_size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_pseudo_style_report(
    args,
    total_rows,
    pseudo_style,
    pseudo_style_confidence,
    pseudo_style_raw_label,
    pseudo_style_score_map,
    pseudo_style_agrees,
    pseudo_style_error,
):
    raw_counts = Counter()
    mapped_counts = Counter()
    accepted_counts = Counter()
    mapped_score_totals = defaultdict(float)
    agreement_counts = Counter()
    secondary_missing = 0
    rows_annotated = 0
    rows_failed = 0

    for idx in range(total_rows):
        error = pseudo_style_error[idx]
        if error:
            rows_failed += 1
            continue

        confidence = pseudo_style_confidence[idx]
        raw_label = pseudo_style_raw_label[idx]
        mapped_style = pseudo_style[idx]
        if confidence is None or raw_label is None:
            continue

        rows_annotated += 1
        raw_counts[raw_label] += 1
        if mapped_style:
            mapped_counts[mapped_style] += 1
            if confidence >= args.report_threshold:
                accepted_counts[mapped_style] += 1

        score_map = pseudo_style_score_map[idx]
        if isinstance(score_map, dict):
            for style, score in score_map.items():
                mapped_score_totals[style] += float(score)

        agrees = pseudo_style_agrees[idx]
        if agrees is None:
            secondary_missing += 1
        else:
            agreement_counts['agree' if agrees else 'disagree'] += 1

    return {
        'model': args.model,
        'teacher_checkpoint': args.teacher_checkpoint or None,
        'teacher_config': args.teacher_config or None,
        'report_threshold': args.report_threshold,
        'top_k': args.top_k,
        'save_style_score_map': bool(args.save_style_score_map),
        'consistency_view': args.consistency_view,
        'consistency_crop_frac': float(args.consistency_crop_frac),
        'consistency_min_seconds': float(args.consistency_min_seconds),
        'batch_size': int(args.batch_size),
        'checkpoint_every': int(args.checkpoint_every),
        'stop_when_accepted_targets': args.stop_when_accepted_targets or None,
        'rows_requested': total_rows,
        'rows_annotated': rows_annotated,
        'rows_failed': rows_failed,
        'raw_label_counts': dict(raw_counts),
        'mapped_style_counts': dict(mapped_counts),
        'accepted_style_counts': dict(accepted_counts),
        'mapped_style_score_totals': {
            style: float(score)
            for style, score in sorted(mapped_score_totals.items())
        },
        'secondary_view_missing_rows': int(secondary_missing),
        'agreement_counts': dict(agreement_counts),
    }


def build_enriched_artifact(
    data,
    args,
    total_rows,
    metadata_report,
    pseudo_style,
    pseudo_style_confidence,
    pseudo_style_raw_label,
    pseudo_style_topk_labels,
    pseudo_style_topk_scores,
    pseudo_style_score_map,
    pseudo_style_view2,
    pseudo_style_view2_confidence,
    pseudo_style_view2_raw_label,
    pseudo_style_view2_topk_labels,
    pseudo_style_view2_topk_scores,
    pseudo_style_view2_score_map,
    pseudo_style_agrees,
    pseudo_style_error,
):
    pseudo_style_report = build_pseudo_style_report(
        args,
        total_rows,
        pseudo_style,
        pseudo_style_confidence,
        pseudo_style_raw_label,
        pseudo_style_score_map,
        pseudo_style_agrees,
        pseudo_style_error,
    )

    enriched = dict(data)
    enriched['pseudo_style'] = pseudo_style
    enriched['pseudo_style_confidence'] = pseudo_style_confidence
    enriched['pseudo_style_raw_label'] = pseudo_style_raw_label
    enriched['pseudo_style_topk_labels'] = pseudo_style_topk_labels
    enriched['pseudo_style_topk_scores'] = pseudo_style_topk_scores
    if args.save_style_score_map:
        enriched['pseudo_style_score_map'] = pseudo_style_score_map
        enriched['pseudo_style_view2_score_map'] = pseudo_style_view2_score_map
    enriched['pseudo_style_view2'] = pseudo_style_view2
    enriched['pseudo_style_view2_confidence'] = pseudo_style_view2_confidence
    enriched['pseudo_style_view2_raw_label'] = pseudo_style_view2_raw_label
    enriched['pseudo_style_view2_topk_labels'] = pseudo_style_view2_topk_labels
    enriched['pseudo_style_view2_topk_scores'] = pseudo_style_view2_topk_scores
    enriched['pseudo_style_agrees'] = pseudo_style_agrees
    enriched['pseudo_style_error'] = pseudo_style_error
    enriched['pseudo_style_source'] = args.model
    enriched['pseudo_style_teacher'] = {
        'model': args.model,
        'teacher_checkpoint': args.teacher_checkpoint or None,
        'teacher_config': args.teacher_config or None,
        'top_k': args.top_k,
        'save_style_score_map': bool(args.save_style_score_map),
        'consistency_view': args.consistency_view,
        'consistency_crop_frac': float(args.consistency_crop_frac),
        'consistency_min_seconds': float(args.consistency_min_seconds),
        'batch_size': int(args.batch_size),
    }
    enriched['pseudo_style_report'] = pseudo_style_report
    enriched['metadata_report'] = metadata_report
    return enriched


def save_artifact(path, artifact):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError('--batch-size must be >= 1')
    if args.checkpoint_every < 1:
        raise ValueError('--checkpoint-every must be >= 1')
    stop_targets = parse_style_targets(args.stop_when_accepted_targets)

    data = torch.load(args.embeddings, weights_only=False)

    clip_paths = data['clip_paths']
    corpus_path = data['corpus_path']
    total_rows = len(clip_paths)
    if args.limit is not None:
        total_rows = min(total_rows, args.limit)
    output_path = Path(args.output)
    checkpoint_path = output_checkpoint_path(output_path, args.checkpoint_path)

    resume_data = None
    if not args.no_resume and checkpoint_path.exists():
        resume_data = torch.load(checkpoint_path, weights_only=False)
        print(f'Resuming pseudo-label annotation from {checkpoint_path}')

    source = resume_data or {}
    row_count = len(clip_paths)
    pseudo_style = copy_list(source, 'pseudo_style', row_count)
    pseudo_style_confidence = copy_list(source, 'pseudo_style_confidence', row_count)
    pseudo_style_raw_label = copy_list(source, 'pseudo_style_raw_label', row_count)
    pseudo_style_topk_labels = copy_list(source, 'pseudo_style_topk_labels', row_count)
    pseudo_style_topk_scores = copy_list(source, 'pseudo_style_topk_scores', row_count)
    pseudo_style_score_map = copy_list(source, 'pseudo_style_score_map', row_count)
    pseudo_style_view2 = copy_list(source, 'pseudo_style_view2', row_count)
    pseudo_style_view2_confidence = copy_list(source, 'pseudo_style_view2_confidence', row_count)
    pseudo_style_view2_raw_label = copy_list(source, 'pseudo_style_view2_raw_label', row_count)
    pseudo_style_view2_topk_labels = copy_list(source, 'pseudo_style_view2_topk_labels', row_count)
    pseudo_style_view2_topk_scores = copy_list(source, 'pseudo_style_view2_topk_scores', row_count)
    pseudo_style_view2_score_map = copy_list(source, 'pseudo_style_view2_score_map', row_count)
    pseudo_style_agrees = copy_list(source, 'pseudo_style_agrees', row_count)
    pseudo_style_error = copy_list(source, 'pseudo_style_error', row_count)

    pending = [
        idx
        for idx in range(total_rows)
        if pseudo_style_confidence[idx] is None and pseudo_style_error[idx] is None
    ]
    print(f'Rows requested: {total_rows}/{len(clip_paths)}')
    print(f'Rows already annotated: {total_rows - len(pending)}')
    print(f'Rows pending: {len(pending)}')
    if stop_targets:
        print(f'Stop targets at threshold {args.report_threshold}: {stop_targets}')

    metadata_report = data.get('metadata_report')
    if metadata_report is None:
        metadata_report = utils.build_commonvoice_metadata_report(
            data.get('age', []),
            data.get('gender', []),
            data.get('accent', []),
        )

    newly_annotated = 0
    rows_since_checkpoint = 0

    def checkpoint():
        artifact = build_enriched_artifact(
            data,
            args,
            total_rows,
            metadata_report,
            pseudo_style,
            pseudo_style_confidence,
            pseudo_style_raw_label,
            pseudo_style_topk_labels,
            pseudo_style_topk_scores,
            pseudo_style_score_map,
            pseudo_style_view2,
            pseudo_style_view2_confidence,
            pseudo_style_view2_raw_label,
            pseudo_style_view2_topk_labels,
            pseudo_style_view2_topk_scores,
            pseudo_style_view2_score_map,
            pseudo_style_agrees,
            pseudo_style_error,
        )
        save_artifact(checkpoint_path, artifact)
        rows_done = artifact['pseudo_style_report']['rows_annotated']
        rows_failed = artifact['pseudo_style_report']['rows_failed']
        print(f'Checkpoint saved to {checkpoint_path} ({rows_done} annotated, {rows_failed} failed)')

    if pending:
        print(f'Loading {args.model} for pseudo-label annotation...')
        model = AutoModel(
            model=args.model,
            hub='hf',
            disable_update=True,
            disable_pbar=True,
        )
        print('Model loaded.')
    else:
        model = None

    stop_requested = False
    progress = tqdm(total=len(pending), desc='Pseudo labels')
    try:
        for batch_indices in iter_batches(pending, args.batch_size):
            batch_clip_paths = [
                resolve_clip_path(corpus_path, clip_paths[idx])
                for idx in batch_indices
            ]
            try:
                scored_batch = score_clip_batch(
                    model,
                    batch_clip_paths,
                    args.top_k,
                    args.batch_size,
                )
            except Exception as batch_error:
                scored_batch = []
                for idx, clip_path in zip(batch_indices, batch_clip_paths):
                    try:
                        scored_batch.append(score_clip(model, clip_path, args.top_k))
                    except Exception as clip_error:
                        if args.fail_on_error:
                            raise
                        pseudo_style_error[idx] = f'{type(clip_error).__name__}: {clip_error}'
                        scored_batch.append(None)
                if any(row is not None for row in scored_batch):
                    print(f'Batch fallback used after error: {batch_error}')

            for idx, clip_path, primary in zip(batch_indices, batch_clip_paths, scored_batch):
                if primary is None:
                    progress.update(1)
                    rows_since_checkpoint += 1
                    continue

                raw_label = primary['raw_label']
                confidence = primary['confidence']
                mapped_style = primary['mapped_style']

                pseudo_style[idx] = mapped_style
                pseudo_style_confidence[idx] = confidence
                pseudo_style_raw_label[idx] = raw_label
                pseudo_style_topk_labels[idx] = primary['topk_labels']
                pseudo_style_topk_scores[idx] = primary['topk_scores']
                if args.save_style_score_map:
                    pseudo_style_score_map[idx] = primary['style_score_map']

                consistency_path = build_consistency_view(
                    clip_path,
                    args.consistency_view,
                    args.consistency_crop_frac,
                    args.consistency_min_seconds,
                )
                if consistency_path is not None:
                    try:
                        secondary = score_clip(model, consistency_path, args.top_k)
                        pseudo_style_view2[idx] = secondary['mapped_style']
                        pseudo_style_view2_confidence[idx] = secondary['confidence']
                        pseudo_style_view2_raw_label[idx] = secondary['raw_label']
                        pseudo_style_view2_topk_labels[idx] = secondary['topk_labels']
                        pseudo_style_view2_topk_scores[idx] = secondary['topk_scores']
                        if args.save_style_score_map:
                            pseudo_style_view2_score_map[idx] = secondary['style_score_map']
                        agrees = (
                            mapped_style is not None
                            and secondary['mapped_style'] is not None
                            and mapped_style == secondary['mapped_style']
                        )
                        pseudo_style_agrees[idx] = bool(agrees)
                    finally:
                        consistency_path.unlink(missing_ok=True)
                else:
                    pseudo_style_agrees[idx] = None

                newly_annotated += 1
                rows_since_checkpoint += 1
                progress.update(1)

            if rows_since_checkpoint >= args.checkpoint_every:
                checkpoint()
                rows_since_checkpoint = 0

            if stop_targets:
                report = build_pseudo_style_report(
                    args,
                    total_rows,
                    pseudo_style,
                    pseudo_style_confidence,
                    pseudo_style_raw_label,
                    pseudo_style_score_map,
                    pseudo_style_agrees,
                    pseudo_style_error,
                )
                accepted = report['accepted_style_counts']
                if all(accepted.get(style, 0) >= target for style, target in stop_targets.items()):
                    print(f'Stop targets met: {accepted}')
                    stop_requested = True
                    break
            if stop_requested:
                break
    except KeyboardInterrupt:
        print('Interrupted; saving checkpoint before exit...')
        checkpoint()
        raise
    finally:
        progress.close()

    enriched = build_enriched_artifact(
        data,
        args,
        total_rows,
        metadata_report,
        pseudo_style,
        pseudo_style_confidence,
        pseudo_style_raw_label,
        pseudo_style_topk_labels,
        pseudo_style_topk_scores,
        pseudo_style_score_map,
        pseudo_style_view2,
        pseudo_style_view2_confidence,
        pseudo_style_view2_raw_label,
        pseudo_style_view2_topk_labels,
        pseudo_style_view2_topk_scores,
        pseudo_style_view2_score_map,
        pseudo_style_agrees,
        pseudo_style_error,
    )
    save_artifact(output_path, enriched)

    print(f'Saved pseudo-labeled CommonVoice artifact to {output_path}')
    print(f'Rows annotated: {enriched["pseudo_style_report"]["rows_annotated"]}/{len(clip_paths)}')
    print(f'Rows failed: {enriched["pseudo_style_report"]["rows_failed"]}')
    print(f'New rows annotated this run: {newly_annotated}')
    print('Accepted pseudo-style counts at report threshold:')
    for style, count in sorted(enriched['pseudo_style_report']['accepted_style_counts'].items()):
        print(f'  {style:8s}: {count}')


if __name__ == '__main__':
    main()
