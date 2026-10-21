"""
Aggregate all K-NHIB benchmark CSV outputs into a clean master dataset
and per-(model, condition) summary tables.

Output: results/aggregated/
  - master_dataset.csv         deduplicated long-format records
  - coverage_matrix.csv        unique items per (model_version, condition)
  - summary_model_condition.csv
  - summary_by_cancer.csv
  - summary_by_class.csv
  - per_class_metrics.csv
  - REPORT.md                  human-readable summary report
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE.parent / "results"
OUT_DIR = RESULTS_DIR / "aggregated"


CLASSES = ["eligible", "ineligible", "undeterminable"]

# Canonical alias for each evaluated model_version.
CANONICAL_MODELS = {
    "claude-opus-4-6":        "claude",
    "claude-sonnet-4-6":      "claude-fast",
    "gpt-5.4-2026-03-05":     "gpt",
    "gpt-5-mini-2025-08-07":  "gpt-fast",
    "gemini-3.1-pro-preview": "gemini",
    "gemini-3-flash-preview": "gemini-fast",
}


def load_all() -> pd.DataFrame:
    files = sorted(glob.glob(str(RESULTS_DIR / "knhib_results_*.csv")))
    dfs = []
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        d["_source_file"] = Path(f).name
        dfs.append(d)
    if not dfs:
        raise ValueError(f"No benchmark result CSV files in {RESULTS_DIR}")
    return pd.concat(dfs, ignore_index=True, sort=False)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Keep latest attempt per explicit episode, never collapse efforts or replicates."""
    required = {"episode_id", "attempt_id", "experiment_id", "run_id", "reasoning_effort",
                "reasoning_mode", "input_hash", "is_dry_run", "timestamp"}
    if required - set(df.columns):
        raise ValueError("Legacy results lack explicit episode/config identity; migrate explicitly")
    if df.empty or df[list(required - {"reasoning_effort"})].isna().any().any():
        raise ValueError("Empty results or incomplete episode identity")
    df = df.copy()
    df["_ts"] = pd.to_datetime(df["timestamp"], errors="raise", utc=True)
    duplicate_attempt = df[df.duplicated("attempt_id", keep=False)]
    if not duplicate_attempt.empty:
        # Copies in a later retry CSV must agree on every substantive field.
        columns = [x for x in df.columns if x not in {"_source_file", "_ts"}]
        if (duplicate_attempt.groupby("attempt_id")[columns].nunique(dropna=False) > 1).any().any():
            raise ValueError("Conflicting copies of one attempt")
    df = df.sort_values(["_ts", "attempt_id"], kind="stable")
    return df.drop_duplicates(subset=["episode_id"], keep="last").drop(columns="_ts")


def compute_metrics(group: pd.DataFrame) -> pd.Series:
    y_true = group["expected"].astype(str).tolist()
    y_pred = group["predicted"].astype(str).tolist()
    n = len(group)
    n_error = int((group["predicted"] == "error").sum())
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, average="macro", zero_division=0
    )
    return pd.Series({
        "n": n,
        "n_error_pred": n_error,
        "accuracy": round(acc, 4),
        "macro_precision": round(prec, 4),
        "macro_recall": round(rec, 4),
        "macro_f1": round(f1, 4),
    })


def per_class_metrics(group: pd.DataFrame) -> pd.DataFrame:
    y_true = group["expected"].astype(str).tolist()
    y_pred = group["predicted"].astype(str).tolist()
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )
    return pd.DataFrame({
        "label":     CLASSES,
        "support":   sup,
        "precision": np.round(prec, 4),
        "recall":    np.round(rec, 4),
        "f1":        np.round(f1, 4),
    })


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading all CSVs...")
    raw = load_all()
    print(f"  raw rows: {len(raw):,}")

    raw.to_csv(OUT_DIR / "all_attempts.csv", index=False)
    print("Selecting final attempts by immutable episode ID (all attempts retained)...")
    master = deduplicate(raw)
    print(f"  unique rows: {len(master):,}")

    # Group by requested model/config/experiment, including failed calls and every replicate.
    dry = master["is_dry_run"].astype(str).str.lower()
    if not dry.isin(["true", "false"]).all():
        raise ValueError("Unknown dry-run classification")
    master["is_main_analysis"] = dry.eq("false")
    master["model_canonical"] = (master["model"].astype(str) + "|" +
        master["reasoning_effort"].fillna("provider_default").astype(str) + "|" +
        master["reasoning_mode"].astype(str) + "|" + master["experiment_id"].astype(str))
    master.to_csv(OUT_DIR / "master_dataset.csv", index=False)
    main_df = master[master["is_main_analysis"]].copy()
    if main_df.empty:
        raise ValueError("No non-placeholder episodes to score")
    print(f"Scoring {len(main_df)} episodes, preserving all replicates and failed predictions")
    replicate = (main_df.groupby(["model_canonical", "condition", "run_id"])
                 .apply(compute_metrics).reset_index())
    replicate.to_csv(OUT_DIR / "summary_by_replicate.csv", index=False)

    # Coverage matrix
    coverage = (main_df.groupby(["model_canonical", "condition"])["id"]
                .nunique().unstack(fill_value=0))
    coverage.to_csv(OUT_DIR / "coverage_matrix.csv")
    print(f"  -> coverage_matrix.csv")

    # Summary: model × condition
    s_mc = (main_df.groupby(["model_canonical", "condition"])
            .apply(compute_metrics).reset_index())
    s_mc.to_csv(OUT_DIR / "summary_model_condition.csv", index=False)

    # Summary: model × condition × cancer_type
    s_can = (main_df.groupby(["model_canonical", "condition", "cancer_type"])
             .apply(compute_metrics).reset_index())
    s_can.to_csv(OUT_DIR / "summary_by_cancer.csv", index=False)

    # Summary: model × condition × class
    s_class = (main_df.groupby(["model_canonical", "condition", "class"])
               .apply(compute_metrics).reset_index())
    s_class.to_csv(OUT_DIR / "summary_by_class.csv", index=False)

    # Per-class metrics
    rows = []
    for (m, c), g in main_df.groupby(["model_canonical", "condition"]):
        cm = per_class_metrics(g)
        cm.insert(0, "condition", c)
        cm.insert(0, "model_canonical", m)
        rows.append(cm)
    pc = pd.concat(rows, ignore_index=True)
    pc.to_csv(OUT_DIR / "per_class_metrics.csv", index=False)

    print(f"  -> summary_model_condition.csv  ({len(s_mc)} rows)")
    print(f"  -> summary_by_cancer.csv        ({len(s_can)} rows)")
    print(f"  -> summary_by_class.csv         ({len(s_class)} rows)")
    print(f"  -> per_class_metrics.csv        ({len(pc)} rows)")

    # REPORT.md
    timestamp_range = (
        f"{master['timestamp'].min()}  ~  {master['timestamp'].max()}"
        if "timestamp" in master.columns else "N/A"
    )
    n_files = master["_source_file"].nunique()

    pivot_acc = (s_mc.pivot(index="model_canonical",
                            columns="condition", values="accuracy")
                 .round(3))
    pivot_f1 = (s_mc.pivot(index="model_canonical",
                           columns="condition", values="macro_f1")
                .round(3))

    report = []
    report.append("# K-NHIB Benchmark — Aggregated Results Report")
    report.append("")
    report.append(f"- Generated from {n_files} CSV files in `results/`")
    report.append(f"- Raw rows: {len(raw):,}  →  Deduplicated: {len(master):,}")
    report.append(f"- Main-analysis rows: {len(main_df):,} (all replicates, explicit experiment identity)")
    report.append(f"- Timestamp range: {timestamp_range}")
    report.append("")
    report.append("## Canonical model mapping")
    report.append("")
    report.append("| model_version | canonical alias |")
    report.append("|---|---|")
    for v, a in CANONICAL_MODELS.items():
        report.append(f"| `{v}` | `{a}` |")
    report.append("")
    report.append("## Coverage (unique items per cell)")
    report.append("")
    report.append(coverage.to_markdown())
    report.append("")
    report.append("## Accuracy — model × condition")
    report.append("")
    report.append("```\n" + pivot_acc.to_string() + "\n```")
    report.append("")
    report.append("## Macro F1 — model × condition")
    report.append("")
    report.append("```\n" + pivot_f1.to_string() + "\n```")
    report.append("")
    report.append("## Files in this aggregation bundle")
    report.append("")
    for p in sorted(OUT_DIR.glob("*.csv")):
        report.append(f"- `{p.name}` ({p.stat().st_size:,} bytes)")
    report.append("")

    (OUT_DIR / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"  -> REPORT.md")
    print("\nDone.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Score explicit K-NHIB experiment episodes")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    RESULTS_DIR = args.results_dir.resolve()
    OUT_DIR = RESULTS_DIR / "aggregated"
    main()
