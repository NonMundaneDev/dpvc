#!/usr/bin/env python3
"""Run a listening-first recovery study for Joe's historical metadata VAEs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch

import dpvc
try:
    from scripts.analyze_gender_followup_acoustics import analyze_file
except ModuleNotFoundError:
    from analyze_gender_followup_acoustics import analyze_file


DEFAULT_SOURCES = (
    "male_1_cremad_1003,female_1_cremad_1002,cremad_1004,cremad_1023,cremad_1045"
)
HISTORICAL_PROVENANCE = {
    "joe_age_gender_v1": (
        "git commit `de65862`, `examples/openvoice_vae_features.pt` "
        "(CommonVoice age + gender)"
    ),
    "joe_age_gender_accent_v2": (
        "git commit `8bfb1fe`, `examples/openvoice_vae_features2.pt` "
        "(CommonVoice age + gender + accent)"
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    label: str
    checkpoint: Path
    latent_dims: int


@dataclass(frozen=True)
class ConditionSpec:
    label: str
    attribute: str
    endpoint: str
    strength: float
    latent_dim: int | None
    latent_value: float | None

    @property
    def control_features(self):
        if self.latent_dim is None:
            return None
        return {self.latent_dim: self.latent_value}


def conditions():
    rows = [ConditionSpec("baseline", "baseline", "baseline", 0.0, None, None)]
    for strength in (1.0, 2.0):
        suffix = f"s{int(strength)}"
        rows.extend(
            [
                ConditionSpec(f"male_{suffix}", "gender", "male", strength, 1, -strength),
                ConditionSpec(f"female_{suffix}", "gender", "female", strength, 1, strength),
                ConditionSpec(f"teens_{suffix}", "age", "teens", strength, 0, -strength),
                ConditionSpec(f"sixties_{suffix}", "age", "sixties", strength, 0, strength),
            ]
        )
    return rows


def parse_model(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("models must use LABEL=CHECKPOINT")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("models must use LABEL=CHECKPOINT")
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    latent_dims = int(state["encoder.to_mu.weight"].shape[0])
    return ModelSpec(label.strip(), checkpoint, latent_dims)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        required=True,
        help="Historical model as LABEL=CHECKPOINT; may be repeated.",
    )
    parser.add_argument("--source-dir", default="examples/source_speakers")
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bundle-dir",
        default="results/historical_metadata_recovery_review_bundle_2026-07-15",
    )
    parser.add_argument(
        "--bundle-zip",
        default="results/historical_metadata_recovery_review_bundle_2026-07-15.zip",
    )
    parser.add_argument(
        "--summary-out",
        default="results/historical_metadata_recovery_summary.md",
    )
    parser.add_argument(
        "--acoustic-summary-out",
        default="results/historical_metadata_recovery_acoustic_summary.csv",
    )
    parser.add_argument(
        "--manifest-out",
        default="results/historical_metadata_recovery_generation_manifest.jsonl",
    )
    return parser.parse_args()


def build_anonymizer(wrapper, model):
    config = wrapper.get_vae_config()
    config["checkpoint_path"] = str(model.checkpoint)
    config["latent_dim"] = model.latent_dims
    return dpvc.Anonymizer(wrapper, vae_config=config)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def clean(value):
    return "" if value is None else f"{value:.4f}"


def generate(models, sources, bundle_dir, noise_level, seed):
    wrapper = dpvc.OpenVoiceWrapper()
    condition_specs = conditions()
    rows = []
    for model in models:
        anonymizer = build_anonymizer(wrapper, model)
        for source in sources:
            for condition in condition_specs:
                output = bundle_dir / "audio" / model.label / f"{source.stem}_{condition.label}.wav"
                output.parent.mkdir(parents=True, exist_ok=True)
                print(f"[{model.label}] {source.stem}: {condition.label}")
                anonymizer.anonymize(
                    str(source),
                    str(output),
                    noise_level=noise_level,
                    seed=seed,
                    control_features=condition.control_features,
                )
                stats = analyze_file(output)
                rows.append(
                    {
                        "model": model.label,
                        "checkpoint": str(model.checkpoint),
                        "latent_dims": model.latent_dims,
                        "source_stem": source.stem,
                        "source_file": str(source.resolve()),
                        "condition": condition.label,
                        "attribute": condition.attribute,
                        "endpoint": condition.endpoint,
                        "strength": condition.strength,
                        "latent_dim": "" if condition.latent_dim is None else condition.latent_dim,
                        "latent_value": "" if condition.latent_value is None else condition.latent_value,
                        "output_file": str(output.resolve()),
                        "bundle_audio": output.relative_to(bundle_dir).as_posix(),
                        "median_f0_hz": clean(stats.median_f0_hz),
                        "spectral_centroid_hz": clean(stats.spectral_centroid_hz),
                        "rms_dbfs": clean(stats.rms_dbfs),
                        "duration_sec": clean(stats.duration_sec),
                    }
                )
    return rows


def build_pair_rows(rows):
    index = {
        (row["model"], row["source_stem"], row["attribute"], float(row["strength"]), row["endpoint"]): row
        for row in rows
    }
    pairs = []
    models = sorted({row["model"] for row in rows})
    sources = sorted({row["source_stem"] for row in rows})
    endpoints = {"gender": ("male", "female"), "age": ("teens", "sixties")}
    for model in models:
        for source in sources:
            for attribute, (negative_label, positive_label) in endpoints.items():
                for strength in (1.0, 2.0):
                    negative = index[(model, source, attribute, strength, negative_label)]
                    positive = index[(model, source, attribute, strength, positive_label)]
                    negative_f0 = float(negative["median_f0_hz"]) if negative["median_f0_hz"] else None
                    positive_f0 = float(positive["median_f0_hz"]) if positive["median_f0_hz"] else None
                    negative_centroid = (
                        float(negative["spectral_centroid_hz"])
                        if negative["spectral_centroid_hz"]
                        else None
                    )
                    positive_centroid = (
                        float(positive["spectral_centroid_hz"])
                        if positive["spectral_centroid_hz"]
                        else None
                    )
                    pairs.append(
                        {
                            "model": model,
                            "source_stem": source,
                            "attribute": attribute,
                            "strength": strength,
                            "negative_endpoint": negative_label,
                            "positive_endpoint": positive_label,
                            "negative_audio": negative["bundle_audio"],
                            "positive_audio": positive["bundle_audio"],
                            "negative_f0_hz": clean(negative_f0),
                            "positive_f0_hz": clean(positive_f0),
                            "positive_minus_negative_f0_hz": clean(
                                None if negative_f0 is None or positive_f0 is None else positive_f0 - negative_f0
                            ),
                            "negative_centroid_hz": clean(negative_centroid),
                            "positive_centroid_hz": clean(positive_centroid),
                            "positive_minus_negative_centroid_hz": clean(
                                None
                                if negative_centroid is None or positive_centroid is None
                                else positive_centroid - negative_centroid
                            ),
                        }
                    )
    return pairs


def median(values):
    cleaned = [float(value) for value in values if value not in (None, "")]
    return statistics.median(cleaned) if cleaned else None


def summarize_pairs(pair_rows):
    summaries = []
    keys = sorted({(row["model"], row["attribute"], row["strength"]) for row in pair_rows})
    for model, attribute, strength in keys:
        group = [
            row
            for row in pair_rows
            if row["model"] == model
            and row["attribute"] == attribute
            and row["strength"] == strength
        ]
        f0_values = [row["positive_minus_negative_f0_hz"] for row in group]
        centroid_values = [row["positive_minus_negative_centroid_hz"] for row in group]
        summaries.append(
            {
                "model": model,
                "attribute": attribute,
                "strength": strength,
                "pairs": len(group),
                "positive_endpoint_higher_f0": sum(
                    1 for value in f0_values if value and float(value) > 0
                ),
                "median_positive_minus_negative_f0_hz": clean(median(f0_values)),
                "median_positive_minus_negative_centroid_hz": clean(median(centroid_values)),
            }
        )
    return summaries


def audio(path, label):
    return (
        f'<div class="label">{html.escape(label)}</div>'
        f'<audio controls preload="none" src="{html.escape(path)}"></audio>'
    )


def write_html(path, rows, pair_rows, source_audio):
    baseline = {
        (row["model"], row["source_stem"]): row["bundle_audio"]
        for row in rows
        if row["condition"] == "baseline"
    }
    models = sorted({row["model"] for row in rows})
    sources = sorted({row["source_stem"] for row in rows})
    lines = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
        "<title>Historical metadata recovery</title><style>",
        ":root{--bg:#f4f4f0;--surface:#fff;--ink:#20221f;--muted:#666b63;--line:#d5d8d1;--neg:#315f78;--pos:#8a493f}",
        "*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}",
        "main{max-width:1380px;margin:auto;padding:24px 18px 56px}h1{font-size:28px;margin:0 0 6px;letter-spacing:0}h2{font-size:21px;margin:32px 0 8px;letter-spacing:0}h3{font-size:17px;margin:22px 0 8px;letter-spacing:0}",
        ".meta,.label{color:var(--muted)}.protocol{border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin-top:18px;padding:12px 0}.table{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--surface)}",
        "table{width:100%;min-width:1050px;border-collapse:collapse}th,td{text-align:left;vertical-align:top;padding:9px;border-bottom:1px solid var(--line)}th{font-size:12px;color:var(--muted);background:#eeefea}tr:last-child td{border-bottom:0}.label{font-size:12px;margin-bottom:2px}audio{width:210px;height:34px}.neg{color:var(--neg)}.pos{color:var(--pos)}code{font-size:12px}",
        "</style></head><body><main><h1>Historical metadata recovery</h1>",
        f'<div class="meta">{len(models)} exact historical checkpoints, {len(sources)} source speakers, trained endpoints and old demo extrapolation</div>',
        '<div class="protocol"><strong>Decision gate:</strong> judge whether the paired outputs sound like the named age/gender endpoints, not merely different speakers. Review strength 1 before strength 2.</div>',
    ]
    for model in models:
        lines.append(f"<h2>{html.escape(model)}</h2>")
        for attribute in ("gender", "age"):
            negative_label, positive_label = ("male", "female") if attribute == "gender" else ("teens", "sixties")
            lines.extend(
                [
                    f"<h3>{attribute.title()}</h3><div class=\"table\"><table>",
                    "<thead><tr><th>Source context</th><th>Strength</th>"
                    f'<th class="neg">{negative_label}</th><th class="pos">{positive_label}</th>'
                    "<th>Acoustic delta</th><th>Listening judgment</th></tr></thead><tbody>",
                ]
            )
            for source in sources:
                group = [
                    row
                    for row in pair_rows
                    if row["model"] == model
                    and row["source_stem"] == source
                    and row["attribute"] == attribute
                ]
                for index, pair in enumerate(group):
                    context = ""
                    if index == 0:
                        context = audio(source_audio[source], "source") + audio(
                            baseline[(model, source)], "model baseline"
                        )
                    delta = pair["positive_minus_negative_f0_hz"] or "n/a"
                    lines.append(
                        "<tr>"
                        f"<td>{context}<code>{html.escape(source)}</code></td>"
                        f"<td>{pair['strength']:.0f}</td>"
                        f"<td>{audio(pair['negative_audio'], negative_label)}</td>"
                        f"<td>{audio(pair['positive_audio'], positive_label)}</td>"
                        f"<td>F0: {html.escape(delta)} Hz</td>"
                        "<td>clear / subtle / timbre-only / failed</td></tr>"
                    )
            lines.append("</tbody></table></div>")
    lines.append("</main></body></html>")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_ratings(path, pair_rows):
    rows = []
    for pair in pair_rows:
        rows.append(
            {
                **pair,
                "named_attribute_clear_yes_no_uncertain": "",
                "strength_of_difference_1_5": "",
                "intelligibility_1_5": "",
                "naturalness_1_5": "",
                "speaker_timbre_only_yes_no": "",
                "notes": "",
            }
        )
    write_csv(path, rows)


def write_summary(path, models, sources, summaries):
    lines = [
        "# Historical Metadata-Control Recovery",
        "",
        "This study re-runs Joe's exact March 2026 CommonVoice metadata checkpoints.",
        "It tests the original latent ordering and label polarity rather than the current metadata-control interface.",
        "",
        "## Provenance",
        "",
    ]
    for model in models:
        provenance = HISTORICAL_PROVENANCE.get(
            model.label, f"runtime checkpoint `{model.checkpoint}`"
        )
        lines.append(f"- `{model.label}`: {provenance}; `{model.latent_dims}` latent dimensions")
    lines.extend(
        [
            f"- Sources: `{len(sources)}`",
            "- Historical dimensions: age `0`, gender `1`",
            "- Historical polarity: male `-1`, female `+1`; teens `-1`, sixties `+1`",
            "- Strength `1` is the trained endpoint; strength `2` reproduces the extrapolated old demo setting.",
            "- Noise `0`, seed `42`",
            "",
            "## Acoustic Sanity Check",
            "",
            "| Model | Attribute | Strength | Pairs | Positive endpoint higher F0 | Median F0 delta | Median centroid delta |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summaries:
        lines.append(
            f"| `{row['model']}` | `{row['attribute']}` | {row['strength']:.0f} | "
            f"{row['pairs']} | {row['positive_endpoint_higher_f0']}/{row['pairs']} | "
            f"{row['median_positive_minus_negative_f0_hz'] or 'n/a'} Hz | "
            f"{row['median_positive_minus_negative_centroid_hz'] or 'n/a'} Hz |"
        )
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "- F0 and spectral centroid are sanity checks, not age/gender classifiers.",
            "- Promote a checkpoint only if listeners consistently hear the named endpoint across source speakers while speech remains intelligible and natural.",
            "- If only strength `2` is audible, report that the old demo depended on latent extrapolation and evaluate its quality cost before treating it as a control.",
            "- If differences remain speaker/timbre-only, keep historical metadata control out of the paper claim.",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_zip(bundle_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(bundle_dir.parent))


def main():
    args = parse_args()
    bundle_dir = Path(args.bundle_dir)
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    source_dir = Path(args.source_dir)
    source_names = [item.strip() for item in args.sources.split(",") if item.strip()]
    sources = [source_dir / f"{name}.wav" for name in source_names]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing sources: {missing}")

    source_audio = {}
    for source in sources:
        destination = bundle_dir / "audio" / "source" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_audio[source.stem] = destination.relative_to(bundle_dir).as_posix()

    rows = generate(args.model, sources, bundle_dir, args.noise_level, args.seed)
    pair_rows = build_pair_rows(rows)
    summaries = summarize_pairs(pair_rows)
    write_csv(bundle_dir / "generation_and_acoustics.csv", rows)
    write_csv(bundle_dir / "paired_acoustics.csv", pair_rows)
    acoustic_summary_path = bundle_dir / "acoustic_summary.csv"
    write_csv(acoustic_summary_path, summaries)
    acoustic_summary_out = Path(args.acoustic_summary_out)
    acoustic_summary_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(acoustic_summary_path, acoustic_summary_out)
    write_ratings(bundle_dir / "ratings.csv", pair_rows)
    write_html(bundle_dir / "index.html", rows, pair_rows, source_audio)
    summary_path = bundle_dir / "summary.md"
    write_summary(summary_path, args.model, sources, summaries)
    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(summary_path, summary_out)
    manifest_path = bundle_dir / "generation_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest_out = Path(args.manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, manifest_out)
    (bundle_dir / "README.md").write_text(
        "# Historical Metadata Recovery Review Bundle\n\n"
        "Open `index.html` through a local HTTP server. Listen at strength 1 before strength 2, "
        "then record judgments in `ratings.csv`. Read `quality_summary.md` for WER/MOS results "
        "after running the quality evaluation. All review audio is self-contained here.\n",
        encoding="utf-8",
    )
    write_zip(bundle_dir, Path(args.bundle_zip))
    print(f"Wrote {len(rows)} generated rows to {bundle_dir}")
    print(f"Wrote review zip: {args.bundle_zip}")


if __name__ == "__main__":
    main()
