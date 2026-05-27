"""
Audit Common Voice age/gender metadata for controllable-speaker experiments.

The first metadata-control branch needs to know whether Common Voice has enough
local age/gender labels to justify adding those controls to the VAE. This script
audits the local validated.tsv, optionally restricts to locally available clips,
and optionally audits an extracted OpenVoice embedding artifact.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


AGE_ORDER = [
    "teens",
    "twenties",
    "thirties",
    "fourties",
    "forties",
    "fifties",
    "sixties",
    "seventies",
    "eighties",
    "nineties",
]

AGE_NORMALIZATION = {
    "fourties": "forties",
}

AGE_BIN_MAP = {
    "teens": "young",
    "twenties": "young",
    "thirties": "adult",
    "fourties": "adult",
    "forties": "adult",
    "fifties": "older",
    "sixties": "older",
    "seventies": "older",
    "eighties": "older",
    "nineties": "older",
}

GENDER_NORMALIZATION = {
    "male": "male",
    "male_masculine": "male",
    "female": "female",
    "female_feminine": "female",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit Common Voice metadata for age/gender control"
    )
    parser.add_argument(
        "--corpus-path",
        required=True,
        help="Common Voice language directory containing validated.tsv and clips/",
    )
    parser.add_argument(
        "--validated-name",
        default="validated.tsv",
        help="Validated TSV filename under --corpus-path (default: validated.tsv)",
    )
    parser.add_argument(
        "--embedding-artifact",
        default=None,
        help="Optional extracted OpenVoice .pt artifact to audit for preserved metadata",
    )
    parser.add_argument(
        "--out-md",
        default="results/commonvoice_metadata_controls_audit.md",
        help="Markdown report path",
    )
    parser.add_argument(
        "--out-json",
        default="results/commonvoice_metadata_controls_audit.json",
        help="JSON summary path",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Top value count to include in reports",
    )
    return parser.parse_args()


def is_known(value):
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def clean_value(value):
    if not is_known(value):
        return None
    return str(value).strip()


def normalize_age(value):
    value = clean_value(value)
    if value is None:
        return None
    return AGE_NORMALIZATION.get(value, value)


def age_to_ordinal(value):
    value = normalize_age(value)
    if value is None or value not in AGE_ORDER:
        return None
    normalized_order = [AGE_NORMALIZATION.get(item, item) for item in AGE_ORDER]
    deduped = []
    for item in normalized_order:
        if item not in deduped:
            deduped.append(item)
    idx = deduped.index(value)
    if len(deduped) == 1:
        return 0.0
    return -1.0 + 2.0 * idx / (len(deduped) - 1)


def gender_to_scalar(value):
    value = clean_value(value)
    value = GENDER_NORMALIZATION.get(value, value)
    if value == "female":
        return -1.0
    if value == "male":
        return 1.0
    return None


def summarize_values(values, top_n):
    cleaned = [clean_value(value) for value in values]
    known = [value for value in cleaned if value is not None]
    counts = Counter(known)
    return {
        "total": len(cleaned),
        "known": len(known),
        "missing": len(cleaned) - len(known),
        "unique_known": len(counts),
        "top_values": dict(counts.most_common(top_n)),
    }


def summarize_ready_rows(df):
    ages = [normalize_age(value) for value in df.get("age", [])]
    genders = [
        GENDER_NORMALIZATION.get(clean_value(value), clean_value(value))
        for value in df.get("gender", [])
    ]
    age_ord = [age_to_ordinal(value) for value in ages]
    gender_scalar = [gender_to_scalar(value) for value in genders]
    age_bins = [AGE_BIN_MAP.get(value) if value is not None else None for value in ages]

    ready_age = [value is not None for value in age_ord]
    ready_gender = [value is not None for value in gender_scalar]
    ready_both = [a and g for a, g in zip(ready_age, ready_gender)]

    return {
        "age_control_rows": int(sum(ready_age)),
        "gender_control_rows": int(sum(ready_gender)),
        "age_gender_control_rows": int(sum(ready_both)),
        "age_bin_counts": dict(Counter(value for value in age_bins if value is not None)),
        "recommended_control_dims": {
            "dim_9": "gender_binary_scalar female_feminine/female=-1 male_masculine/male=1",
            "dim_10": "age_ordinal_scalar youngest=-1 oldest=1",
            "dims_11_14": "free speaker/identity capacity for latent_dims=15",
        },
    }


def summarize_dataframe(df, name, top_n):
    if "client_id" not in df.columns:
        raise ValueError(f"{name} is missing client_id")
    if "path" not in df.columns:
        raise ValueError(f"{name} is missing path")

    report = {
        "name": name,
        "rows": int(len(df)),
        "unique_speakers": int(df["client_id"].astype(str).nunique()),
        "age": summarize_values(df["age"], top_n) if "age" in df else None,
        "gender": summarize_values(df["gender"], top_n) if "gender" in df else None,
        "accent": summarize_values(
            df["accents"] if "accents" in df else df["accent"], top_n
        ) if ("accents" in df or "accent" in df) else None,
        "control_readiness": summarize_ready_rows(df),
    }
    if "age" in df and "gender" in df:
        known = df[df["age"].apply(is_known) & df["gender"].apply(is_known)].copy()
        if not known.empty:
            known["age_norm"] = known["age"].apply(normalize_age)
            report["age_gender_pairs_top"] = {
                f"{age}/{gender}": int(count)
                for (age, gender), count in Counter(
                    zip(known["age_norm"], known["gender"])
                ).most_common(top_n)
            }
    return report


def load_embedding_artifact(path):
    import torch

    artifact = torch.load(path, weights_only=True, map_location="cpu")
    rows = {
        "client_id": artifact.get("speaker_ids", []),
        "path": artifact.get("clip_paths", []),
        "age": artifact.get("age", []),
        "gender": artifact.get("gender", []),
        "accent": artifact.get("accent", []),
    }
    lengths = {key: len(value) for key, value in rows.items() if hasattr(value, "__len__")}
    if not lengths or len(set(lengths.values())) != 1:
        raise ValueError(f"Embedding artifact does not preserve per-row metadata: {lengths}")
    return pd.DataFrame(rows), artifact


def filter_to_local_clips(df, clips_dir):
    if not clips_dir.is_dir():
        return df.iloc[0:0].copy(), 0
    available = {path.name for path in clips_dir.rglob("*") if path.is_file()}
    filtered = df[df["path"].astype(str).isin(available)].copy()
    return filtered, len(available)


def pct(value, total):
    if total == 0:
        return "0.0%"
    return f"{100.0 * value / total:.1f}%"


def write_markdown(summary, path):
    lines = [
        "# CommonVoice Metadata Controls Audit",
        "",
        f"Corpus: `{summary['corpus_path']}`",
        f"Validated TSV: `{summary['validated_tsv']}`",
        f"Local clips: `{summary['local_clip_files']}`",
        "",
        "## Recommendation",
        "",
    ]
    local = summary["local_validated_rows"]
    readiness = local["control_readiness"]
    lines.extend([
        "- Add a first-pass metadata-control experiment.",
        "- Use `gender` as one binary scalar control: `female=-1`, `male=1`.",
        "- Use `age` as one ordinal scalar control over known CommonVoice age buckets.",
        "- Keep latent dimensionality at `15` for the first pass: style dims `0-8`, metadata dims `9-10`, free dims `11-14`.",
        "- Train with masked partial-label losses so rows missing age or gender still contribute reconstruction/style information.",
        "",
        "## Local Validated Subset",
        "",
        f"- Rows: `{local['rows']}`",
        f"- Unique speakers: `{local['unique_speakers']}`",
        f"- Age-control rows: `{readiness['age_control_rows']}` ({pct(readiness['age_control_rows'], local['rows'])})",
        f"- Gender-control rows: `{readiness['gender_control_rows']}` ({pct(readiness['gender_control_rows'], local['rows'])})",
        f"- Rows with both age and gender controls: `{readiness['age_gender_control_rows']}` ({pct(readiness['age_gender_control_rows'], local['rows'])})",
        "",
        "### Age Counts",
        "",
    ])
    for value, count in (local.get("age") or {}).get("top_values", {}).items():
        lines.append(f"- `{value}`: `{count}`")
    lines.extend(["", "### Gender Counts", ""])
    for value, count in (local.get("gender") or {}).get("top_values", {}).items():
        lines.append(f"- `{value}`: `{count}`")
    lines.extend(["", "### Age Bin Counts", ""])
    for value, count in readiness["age_bin_counts"].items():
        lines.append(f"- `{value}`: `{count}`")

    if "embedding_artifact" in summary:
        emb = summary["embedding_artifact"]
        emb_ready = emb["control_readiness"]
        lines.extend([
            "",
            "## Extracted Embedding Artifact",
            "",
            f"Artifact: `{summary['embedding_artifact_path']}`",
            f"- Rows: `{emb['rows']}`",
            f"- Unique speakers: `{emb['unique_speakers']}`",
            f"- Age-control rows: `{emb_ready['age_control_rows']}` ({pct(emb_ready['age_control_rows'], emb['rows'])})",
            f"- Gender-control rows: `{emb_ready['gender_control_rows']}` ({pct(emb_ready['gender_control_rows'], emb['rows'])})",
            f"- Rows with both age and gender controls: `{emb_ready['age_gender_control_rows']}` ({pct(emb_ready['age_gender_control_rows'], emb['rows'])})",
        ])
        source_report = summary.get("embedding_metadata_report")
        if source_report:
            lines.extend(["", "### Existing Metadata Report In Artifact", ""])
            for field, report in source_report.items():
                lines.append(
                    f"- `{field}`: known `{report.get('known')}` / `{report.get('total')}`, "
                    f"unique `{report.get('unique_known')}`"
                )

    lines.extend([
        "",
        "## Implementation Implications",
        "",
        "- The raw/extracted CommonVoice artifact already preserves per-row age/gender metadata.",
        "- Mixed artifacts should preserve per-row metadata scalars and masks, not only aggregate CommonVoice metadata.",
        "- Metadata controls need masked label loss so missing age/gender values do not become false zeros.",
        "- The first training target should be conservative: `gender_binary` and `age_ordinal`, not a large one-hot demographic taxonomy.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def main():
    args = parse_args()
    corpus_path = Path(args.corpus_path).expanduser().resolve()
    validated_tsv = corpus_path / args.validated_name
    clips_dir = corpus_path / "clips"

    if not validated_tsv.exists():
        raise FileNotFoundError(validated_tsv)

    full_df = pd.read_csv(validated_tsv, sep="\t", low_memory=False)
    local_df, local_clip_count = filter_to_local_clips(full_df, clips_dir)
    if local_df.empty:
        raise RuntimeError("No validated rows matched local clips")

    summary = {
        "corpus_path": str(corpus_path),
        "validated_tsv": str(validated_tsv),
        "local_clip_files": int(local_clip_count),
        "validated_rows": summarize_dataframe(full_df, "validated_tsv", args.top_n),
        "local_validated_rows": summarize_dataframe(local_df, "local_validated_rows", args.top_n),
    }

    if args.embedding_artifact:
        artifact_path = Path(args.embedding_artifact).expanduser().resolve()
        emb_df, artifact = load_embedding_artifact(artifact_path)
        summary["embedding_artifact_path"] = str(artifact_path)
        summary["embedding_artifact"] = summarize_dataframe(
            emb_df, "embedding_artifact", args.top_n
        )
        if "metadata_report" in artifact:
            summary["embedding_metadata_report"] = artifact["metadata_report"]

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(summary, Path(args.out_md))
    print(f"Wrote JSON audit to {out_json}")
    print(f"Wrote Markdown audit to {args.out_md}")
    local = summary["local_validated_rows"]
    ready = local["control_readiness"]
    print(
        "Local control-ready rows: "
        f"age={ready['age_control_rows']} "
        f"gender={ready['gender_control_rows']} "
        f"both={ready['age_gender_control_rows']}"
    )


if __name__ == "__main__":
    main()
