"""
External speaker-verifier novelty evaluation for generated voice outputs.

This script mirrors examples/eval_novelty.py, but uses an external speaker
verification backend instead of OpenVoice's native speaker embedding space.
The default production backend is SpeechBrain ECAPA-TDNN trained on VoxCeleb.

Typical usage:

    python scripts/eval_external_speaker_verifier.py \
        --manifest output/mixed_teacher_cvrare_hybrid_style_distill_labeled_warmup_sad_enunc_guard_eval/generation_manifest.jsonl \
        --backend speechbrain-ecapa \
        --derive-proxy-trials \
        --out results/eval_external_speaker_verifier_cvrare_sad_enunc_guard.csv \
        --summary-out results/eval_external_speaker_verifier_cvrare_sad_enunc_guard.md

For paper-facing EER, prefer --trial-csv with independent labeled speaker
verification trials. The --derive-proxy-trials mode is useful for current
generated-output panels, but it is a calibration proxy: it treats same-source
baseline conversions as positives and other-source baselines as negatives.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


DEFAULT_SPEECHBRAIN_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_SPEECHBRAIN_SAVEDIR = "pretrained_models/spkrec-ecapa-voxceleb"


@dataclass(frozen=True)
class Trial:
    enroll_file: str
    test_file: str
    label: int
    trial_type: str


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    a_arr = np.asarray(a, dtype=np.float32).flatten()
    b_arr = np.asarray(b, dtype=np.float32).flatten()
    a_norm = np.linalg.norm(a_arr)
    b_norm = np.linalg.norm(b_arr)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a_arr / a_norm, b_arr / b_norm))


def clean_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def load_audio_mono(path: str, target_sample_rate: int):
    """Load audio with soundfile to avoid torchaudio/torchcodec decoder drift."""

    try:
        import soundfile as sf
        import torch
        import torchaudio
    except ImportError as exc:
        raise SystemExit(
            "Audio loading requires soundfile, torch, and torchaudio."
        ) from exc

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    waveform = torch.from_numpy(mono).unsqueeze(0)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    return waveform


def parse_label(value: str) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "same", "target", "positive", "genuine"}:
        return 1
    if normalized in {"0", "false", "different", "impostor", "negative", "nontarget"}:
        return 0
    raise ValueError(f"Unsupported trial label: {value!r}")


def normalize_record(record: dict) -> dict:
    normalized = dict(record)
    normalized["source_file"] = str(Path(record["source_file"]).expanduser().resolve())
    normalized["generated_file"] = str(Path(record["generated_file"]).expanduser().resolve())
    if "output_file" in normalized:
        normalized.pop("output_file")
    return normalized


def load_manifest(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            rows.append(
                {
                    "source_file": raw["source_file"],
                    "generated_file": raw.get("output_file", raw.get("generated_file")),
                    "source_stem": raw.get("source_stem") or Path(raw["source_file"]).stem,
                    "style": raw.get("style", "unknown"),
                    "style_index": raw.get("style_index", ""),
                    "style_strength": raw.get("style_strength", ""),
                    "noise_level": raw.get("noise_level", ""),
                    "seed": raw.get("seed", ""),
                    "vae_checkpoint": raw.get("vae_checkpoint", ""),
                    "latent_dims": raw.get("latent_dims", ""),
                }
            )
    return [normalize_record(row) for row in rows]


def build_single_record(args: argparse.Namespace) -> list[dict]:
    return [
        normalize_record(
            {
                "source_file": args.source,
                "generated_file": args.generated,
                "source_stem": Path(args.source).stem,
                "style": args.style,
                "style_index": "",
                "style_strength": args.style_strength or "",
                "noise_level": args.noise_level or "",
                "seed": args.seed or "",
                "vae_checkpoint": "",
                "latent_dims": "",
            }
        )
    ]


def build_baseline_map(records: Iterable[dict]) -> dict[str, str]:
    baseline_map = {}
    for row in records:
        if row.get("style") == "baseline":
            baseline_map[row["source_stem"]] = row["generated_file"]
    return baseline_map


def collect_unique_paths(records: Iterable[dict], baseline_map: dict[str, str], trials: Iterable[Trial]) -> list[str]:
    paths = set()
    for row in records:
        paths.add(row["source_file"])
        paths.add(row["generated_file"])
        baseline_file = baseline_map.get(row["source_stem"])
        if baseline_file:
            paths.add(baseline_file)
    for trial in trials:
        paths.add(trial.enroll_file)
        paths.add(trial.test_file)
    return sorted(paths)


def build_proxy_trials(records: Sequence[dict], baseline_map: dict[str, str]) -> list[Trial]:
    """Build EER-style proxy trials from source files and baseline conversions.

    Positives are source vs same-source baseline conversion. Negatives are source
    vs another source's baseline conversion. This is useful for a threshold on
    current generated-output panels, but it is not a substitute for an
    independently labeled speaker-verification trial set.
    """

    source_by_stem = {}
    for row in records:
        source_by_stem.setdefault(row["source_stem"], row["source_file"])

    trials: list[Trial] = []
    for stem in sorted(source_by_stem):
        baseline_file = baseline_map.get(stem)
        if not baseline_file:
            continue
        source_file = source_by_stem[stem]
        trials.append(
            Trial(
                enroll_file=source_file,
                test_file=baseline_file,
                label=1,
                trial_type="same_source_baseline_proxy",
            )
        )
        for other_stem in sorted(baseline_map):
            if other_stem == stem:
                continue
            trials.append(
                Trial(
                    enroll_file=source_file,
                    test_file=baseline_map[other_stem],
                    label=0,
                    trial_type="other_source_baseline_proxy",
                )
            )
    return trials


def load_trial_csv(path: str | Path) -> list[Trial]:
    trials = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"enroll_file", "test_file", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Trial CSV missing required columns: {sorted(missing)}")
        for row in reader:
            trials.append(
                Trial(
                    enroll_file=str(Path(row["enroll_file"]).expanduser().resolve()),
                    test_file=str(Path(row["test_file"]).expanduser().resolve()),
                    label=parse_label(row["label"]),
                    trial_type=row.get("trial_type") or "trial_csv",
                )
            )
    return trials


def compute_eer(scores: Sequence[float], labels: Sequence[int]) -> dict:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if not scores:
        raise ValueError("at least one score is required")

    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    positives = labels_arr == 1
    negatives = labels_arr == 0
    num_pos = int(positives.sum())
    num_neg = int(negatives.sum())
    if num_pos == 0 or num_neg == 0:
        raise ValueError("EER requires at least one positive and one negative trial")

    thresholds = sorted(set(scores_arr.tolist()), reverse=True)
    candidates = [float("inf")] + thresholds + [float("-inf")]
    best = None
    for threshold in candidates:
        accepted = scores_arr >= threshold
        false_accepts = int((accepted & negatives).sum())
        false_rejects = int((~accepted & positives).sum())
        far = false_accepts / num_neg
        frr = false_rejects / num_pos
        eer = (far + frr) / 2.0
        diff = abs(far - frr)
        candidate = {
            "eer": float(eer),
            "threshold": float(threshold),
            "far": float(far),
            "frr": float(frr),
            "num_positive_trials": num_pos,
            "num_negative_trials": num_neg,
            "num_trials": int(len(scores_arr)),
        }
        if best is None or (diff, eer) < (best["_diff"], best["eer"]):
            best = {**candidate, "_diff": diff}

    assert best is not None
    best.pop("_diff", None)
    return best


def optional_mean(values: Sequence[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def parse_optional_float(value: str | float | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def summarize_verifier_rows(rows: Sequence[dict]) -> dict:
    styled_rows = [row for row in rows if row.get("style") != "baseline"]
    accepted_values = [
        int(row["accepted_as_source_at_threshold"])
        for row in rows
        if row.get("accepted_as_source_at_threshold") not in (None, "")
    ]
    styled_accepted_values = [
        int(row["accepted_as_source_at_threshold"])
        for row in styled_rows
        if row.get("accepted_as_source_at_threshold") not in (None, "")
    ]
    styled_gains = [
        parse_optional_float(row.get("external_novelty_gain_vs_baseline"))
        for row in styled_rows
    ]
    styled_gains = [value for value in styled_gains if value is not None]

    return {
        "rows": len(rows),
        "styled_rows": len(styled_rows),
        "accepted_as_source_count": sum(accepted_values),
        "accept_rate": optional_mean(accepted_values),
        "styled_accept_rate": optional_mean(styled_accepted_values),
        "mean_styled_novelty_gain_vs_baseline": optional_mean(styled_gains),
    }


def summarize_by_style(rows: Sequence[dict]) -> list[dict]:
    by_style: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("style") == "baseline":
            continue
        by_style.setdefault(row["style"], []).append(row)

    summaries = []
    for style in sorted(by_style):
        style_rows = by_style[style]
        similarities = [float(row["external_similarity"]) for row in style_rows]
        gains = [
            parse_optional_float(row.get("external_novelty_gain_vs_baseline"))
            for row in style_rows
        ]
        gains = [value for value in gains if value is not None]
        accept_values = [
            int(row["accepted_as_source_at_threshold"])
            for row in style_rows
            if row.get("accepted_as_source_at_threshold") not in (None, "")
        ]
        summaries.append(
            {
                "style": style,
                "rows": len(style_rows),
                "mean_external_similarity": optional_mean(similarities),
                "mean_novelty_gain_vs_baseline": optional_mean(gains),
                "accept_as_source_rate": optional_mean(accept_values),
            }
        )
    return summaries


class SpeechBrainEcapaBackend:
    name = "speechbrain-ecapa"

    def __init__(self, source: str = DEFAULT_SPEECHBRAIN_MODEL, savedir: str = DEFAULT_SPEECHBRAIN_SAVEDIR):
        try:
            import torch
            import torchaudio
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:
            raise SystemExit(
                "speechbrain-ecapa backend requires the optional dependency "
                "`speechbrain`. Install with: pip install '.[speaker-verifier]' "
                "or pip install speechbrain"
            ) from exc

        self.torch = torch
        self.torchaudio = torchaudio
        self.classifier = EncoderClassifier.from_hparams(source=source, savedir=savedir)
        self.sample_rate = 16000

    def embed(self, path: str) -> np.ndarray:
        waveform = load_audio_mono(path, target_sample_rate=self.sample_rate)
        with self.torch.no_grad():
            embedding = self.classifier.encode_batch(waveform)
        return embedding.squeeze().detach().cpu().numpy().astype(np.float32)


def make_backend(args: argparse.Namespace):
    if args.backend == "speechbrain-ecapa":
        return SpeechBrainEcapaBackend(source=args.speechbrain_model, savedir=args.speechbrain_savedir)
    raise SystemExit(f"Unsupported backend: {args.backend}")


def score_trials(trials: Sequence[Trial], embedding_cache: dict[str, np.ndarray]) -> tuple[list[float], list[int]]:
    scores = []
    labels = []
    for trial in trials:
        scores.append(cosine(embedding_cache[trial.enroll_file], embedding_cache[trial.test_file]))
        labels.append(trial.label)
    return scores, labels


def build_output_rows(
    records: Sequence[dict],
    baseline_map: dict[str, str],
    embedding_cache: dict[str, np.ndarray],
    backend_name: str,
    threshold: float | None,
    trial_source: str,
) -> list[dict]:
    rows = []
    for row in records:
        source_emb = embedding_cache[row["source_file"]]
        generated_emb = embedding_cache[row["generated_file"]]
        similarity = cosine(source_emb, generated_emb)
        distance = 1.0 - similarity

        baseline_file = baseline_map.get(row["source_stem"])
        baseline_similarity = None
        baseline_distance = None
        similarity_delta = None
        distance_delta = None
        novelty_gain = None

        if baseline_file:
            baseline_emb = embedding_cache[baseline_file]
            baseline_similarity = cosine(source_emb, baseline_emb)
            baseline_distance = 1.0 - baseline_similarity
            similarity_delta = similarity - baseline_similarity
            distance_delta = distance - baseline_distance
            novelty_gain = baseline_similarity - similarity

        accepted = ""
        if threshold is not None:
            accepted = "1" if similarity >= threshold else "0"

        rows.append(
            {
                "speaker": row["source_stem"],
                "style": row["style"],
                "source_file": row["source_file"],
                "generated_file": row["generated_file"],
                "baseline_file": baseline_file or "",
                "external_backend": backend_name,
                "external_similarity": f"{similarity:.4f}",
                "external_distance": f"{distance:.4f}",
                "external_baseline_similarity": clean_float(baseline_similarity),
                "external_baseline_distance": clean_float(baseline_distance),
                "external_similarity_delta_vs_baseline": clean_float(similarity_delta),
                "external_distance_delta_vs_baseline": clean_float(distance_delta),
                "external_novelty_gain_vs_baseline": clean_float(novelty_gain),
                "accepted_as_source_at_threshold": accepted,
                "threshold": clean_float(threshold),
                "threshold_trial_source": trial_source,
                "noise_level": row.get("noise_level", ""),
                "style_strength": row.get("style_strength", ""),
                "seed": row.get("seed", ""),
                "vae_checkpoint": row.get("vae_checkpoint", ""),
                "latent_dims": row.get("latent_dims", ""),
            }
        )
    return rows


def write_csv(path: str | Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise SystemExit("No output rows to write.")
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: str | Path,
    *,
    args: argparse.Namespace,
    rows: Sequence[dict],
    eer: dict | None,
    trial_source: str,
    trials: Sequence[Trial],
) -> None:
    summary = summarize_verifier_rows(rows)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# External Speaker-Verifier Novelty Summary",
        "",
        f"- Backend: `{args.backend}`",
        f"- Manifest: `{Path(args.manifest).resolve() if args.manifest else ''}`",
        f"- Output CSV: `{Path(args.out).resolve()}`",
        f"- Rows: `{summary['rows']}`",
        f"- Styled rows: `{summary['styled_rows']}`",
        f"- Trial source: `{trial_source}`",
        f"- Trial count: `{len(trials)}`",
    ]
    if eer:
        lines.extend(
            [
                f"- EER: `{eer['eer']:.4f}`",
                f"- Threshold: `{eer['threshold']:.4f}`",
                f"- FAR at threshold: `{eer['far']:.4f}`",
                f"- FRR at threshold: `{eer['frr']:.4f}`",
                f"- Positive trials: `{eer['num_positive_trials']}`",
                f"- Negative trials: `{eer['num_negative_trials']}`",
            ]
        )
    if summary["mean_styled_novelty_gain_vs_baseline"] is not None:
        lines.append(
            "- Mean styled novelty gain vs baseline: "
            f"`{summary['mean_styled_novelty_gain_vs_baseline']:.4f}`"
        )
    if summary["styled_accept_rate"] is not None:
        lines.append(f"- Styled accept-as-source rate: `{summary['styled_accept_rate']:.4f}`")

    by_style = summarize_by_style(rows)
    if by_style:
        lines.extend(
            [
                "",
                "## Per-Style Summary",
                "",
                "| Style | Rows | Mean external similarity | Mean novelty gain vs baseline | Accept-as-source rate |",
                "|-------|------|--------------------------|-------------------------------|-----------------------|",
            ]
        )
        for item in by_style:
            lines.append(
                "| "
                f"`{item['style']}` | "
                f"`{item['rows']}` | "
                f"`{item['mean_external_similarity']:.4f}` | "
                f"`{item['mean_novelty_gain_vs_baseline']:.4f}` | "
                f"`{item['accept_as_source_rate']:.4f}` |"
            )

    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Lower `external_similarity` means the verifier sees the generated voice as",
            "  farther from the source speaker.",
            "- Positive `external_novelty_gain_vs_baseline` means the style output moved",
            "  farther from the source than the same-source baseline conversion did.",
            "- `accepted_as_source_at_threshold` is only populated when an EER threshold",
            "  is available from `--trial-csv` or `--derive-proxy-trials`.",
            "- Proxy trials are diagnostic. Paper-facing EER should use an independent",
            "  labeled trial CSV.",
        ]
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--manifest", help="generation_manifest.jsonl from openvoice_infer_controllable.py")
    source_group.add_argument("--source", help="Single source file for one-off evaluation")
    parser.add_argument("--generated", help="Single generated file for one-off evaluation")
    parser.add_argument("--style", default="single", help="Style label for one-off evaluation")
    parser.add_argument("--style-strength", type=float, default=None)
    parser.add_argument("--noise-level", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prefix", default=None, help="Only process records whose source_stem starts with this prefix")
    parser.add_argument("--backend", default="speechbrain-ecapa", choices=["speechbrain-ecapa"])
    parser.add_argument("--speechbrain-model", default=DEFAULT_SPEECHBRAIN_MODEL)
    parser.add_argument("--speechbrain-savedir", default=DEFAULT_SPEECHBRAIN_SAVEDIR)
    parser.add_argument("--trial-csv", help="CSV with enroll_file,test_file,label columns for EER calibration")
    parser.add_argument(
        "--derive-proxy-trials",
        action="store_true",
        help="Build proxy EER trials from source files and baseline conversions in the manifest",
    )
    parser.add_argument("--out", default="results/eval_external_speaker_verifier.csv")
    parser.add_argument("--summary-out", default=None)
    args = parser.parse_args()

    if args.source and not args.generated:
        parser.error("--generated is required when using --source")
    if args.generated and not args.source:
        parser.error("--source is required when using --generated")
    if args.source and args.derive_proxy_trials:
        parser.error("--derive-proxy-trials requires manifest mode")
    if args.trial_csv and args.derive_proxy_trials:
        parser.error("Use either --trial-csv or --derive-proxy-trials, not both")

    return args


def resolve_records(args: argparse.Namespace) -> list[dict]:
    records = load_manifest(args.manifest) if args.manifest else build_single_record(args)
    if args.prefix:
        records = [row for row in records if row["source_stem"].startswith(args.prefix)]
    if not records:
        raise SystemExit("No records matched the provided inputs.")
    return records


def main() -> None:
    args = parse_args()
    records = resolve_records(args)
    baseline_map = build_baseline_map(records)
    unique_sources = sorted({row["source_stem"] for row in records})

    trials: list[Trial] = []
    trial_source = ""
    if args.trial_csv:
        trials = load_trial_csv(args.trial_csv)
        trial_source = "trial_csv"
    elif args.derive_proxy_trials:
        trials = build_proxy_trials(records, baseline_map)
        trial_source = "derived_proxy_trials"

    print(f"Resolved records      : {len(records)}")
    print(f"Unique sources        : {len(unique_sources)}")
    print(f"Baselines available   : {len(baseline_map)}/{len(unique_sources)}")
    print(f"Verifier backend      : {args.backend}")
    print(f"Trial source          : {trial_source or 'none'}")
    print(f"Trials                : {len(trials)}")

    backend = make_backend(args)
    unique_paths = collect_unique_paths(records, baseline_map, trials)
    print(f"Unique audio files    : {len(unique_paths)}")
    print("Extracting external speaker embeddings...")

    embedding_cache = {}
    for index, path in enumerate(unique_paths, 1):
        print(f"[{index}/{len(unique_paths)}] {path}")
        embedding_cache[path] = backend.embed(path)

    eer = None
    threshold = None
    if trials:
        scores, labels = score_trials(trials, embedding_cache)
        eer = compute_eer(scores, labels)
        threshold = eer["threshold"]

    rows = build_output_rows(
        records=records,
        baseline_map=baseline_map,
        embedding_cache=embedding_cache,
        backend_name=backend.name,
        threshold=threshold,
        trial_source=trial_source,
    )
    write_csv(args.out, rows)
    if args.summary_out:
        write_summary(args.summary_out, args=args, rows=rows, eer=eer, trial_source=trial_source, trials=trials)

    summary = summarize_verifier_rows(rows)
    print("\nSummary")
    print("=" * 60)
    print(f"Rows written          : {summary['rows']}")
    print(f"Styled rows           : {summary['styled_rows']}")
    if eer:
        print(f"EER                   : {eer['eer']:.4f}")
        print(f"Threshold             : {eer['threshold']:.4f}")
    if summary["mean_styled_novelty_gain_vs_baseline"] is not None:
        print(f"Mean styled gain      : {summary['mean_styled_novelty_gain_vs_baseline']:.4f}")
    if summary["styled_accept_rate"] is not None:
        print(f"Styled accept rate    : {summary['styled_accept_rate']:.4f}")
    print(f"CSV written to        : {Path(args.out).resolve()}")
    if args.summary_out:
        print(f"Summary written to    : {Path(args.summary_out).resolve()}")


if __name__ == "__main__":
    main()
