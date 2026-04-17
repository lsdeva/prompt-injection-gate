"""Evaluate the trained pipeline against held-out test sets.

Produces:
    eval/evaluation_report.json
    Console summary of metrics per dataset and per threshold.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --eval-set eval       # only internal eval set
    python scripts/evaluate.py --threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config(project_root: Path) -> dict:
    with open(project_root / "config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_metrics_at_threshold(y_true: list[int], scores: list[float], threshold: float) -> dict:
    y_pred = [1 if s >= threshold else 0 for s in scores]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "threshold": threshold,
        "precision": round(precision_score(y_true, y_pred, average="binary", zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, average="binary", zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, average="binary", zero_division=0), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "confusion_matrix": cm,
    }


def evaluate_dataset(
    df: pd.DataFrame,
    classifier,
    thresholds: list[float],
    name: str,
) -> dict:
    """Run classifier over df and compute metrics at each threshold."""
    from tqdm import tqdm

    print(f"\n  Evaluating {name} ({len(df):,} rows)…")
    texts = df["text"].tolist()
    y_true = df["label"].tolist()

    scores: list[float] = []
    for text in tqdm(texts, desc=f"  {name}"):
        try:
            result = classifier.classify(text)
            scores.append(result["injection_score"])
        except Exception:
            scores.append(0.0)

    report: dict = {"dataset": name, "n_samples": len(df), "thresholds": []}

    for t in thresholds:
        metrics = compute_metrics_at_threshold(y_true, scores, t)
        report["thresholds"].append(metrics)

    # Per-attack-type breakdown (best threshold = first that maximises F1)
    best_t = max(report["thresholds"], key=lambda x: x["f1"])["threshold"]
    if "attack_type" in df.columns:
        y_pred_best = [1 if s >= best_t else 0 for s in scores]
        attack_breakdown: dict = {}
        for attack_type in df["attack_type"].unique():
            mask = df["attack_type"] == attack_type
            y_t = [y_true[i] for i in range(len(y_true)) if mask.iloc[i]]
            y_p = [y_pred_best[i] for i in range(len(y_pred_best)) if mask.iloc[i]]
            if not y_t:
                continue
            attack_breakdown[attack_type] = {
                "n": len(y_t),
                "f1": round(f1_score(y_t, y_p, average="binary", zero_division=0), 4),
                "precision": round(precision_score(y_t, y_p, average="binary", zero_division=0), 4),
                "recall": round(recall_score(y_t, y_p, average="binary", zero_division=0), 4),
            }
        report["attack_type_breakdown"] = attack_breakdown

    return report


def print_report(report: dict) -> None:
    """Pretty-print a dataset evaluation report."""
    print(f"\n{'='*60}")
    print(f"Dataset: {report['dataset']}  (n={report['n_samples']:,})")
    print(f"{'='*60}")
    print(f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Accuracy':>10}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*10}")
    for t in report["thresholds"]:
        print(
            f"  {t['threshold']:>10.2f}  {t['precision']:>10.4f}  "
            f"{t['recall']:>8.4f}  {t['f1']:>8.4f}  {t['accuracy']:>10.4f}"
        )

    if "attack_type_breakdown" in report:
        print(f"\n  Per-attack-type breakdown (threshold optimised for F1):")
        for atype, m in report["attack_type_breakdown"].items():
            print(f"    {atype:<24}  n={m['n']:>5}  F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")

    cm_data = next(iter(report["thresholds"]), {}).get("confusion_matrix")
    if cm_data:
        print(f"\n  Confusion matrix (threshold={report['thresholds'][0]['threshold']}):")
        print(f"    [[TN={cm_data[0][0]:>5}, FP={cm_data[0][1]:>5}]")
        print(f"     [FN={cm_data[1][0]:>5}, TP={cm_data[1][1]:>5}]]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the prompt injection classifier")
    parser.add_argument(
        "--eval-set",
        choices=["eval", "rogue", "all"],
        default="all",
        help="Which evaluation set to run (default: all)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the single classification threshold to evaluate",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    config = _load_config(project_root)
    eval_cfg = config.get("evaluation", {})
    data_cfg = config.get("data", {})

    processed_dir = project_root / data_cfg.get("processed_dir", "data/processed")
    eval_output_dir = project_root / eval_cfg.get("output_dir", "eval")
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = (
        [args.threshold]
        if args.threshold is not None
        else eval_cfg.get("thresholds", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    )

    # ── Load classifier ───────────────────────────────────────────────
    print("[INFO] Loading classifier…")
    try:
        from src.stage2_classifier import DeBERTaClassifier, ModelNotReadyError

        clf = DeBERTaClassifier()
        clf.load()
        print("[INFO] Classifier loaded.")
    except Exception as exc:
        print(f"[ERROR] Could not load classifier: {exc}")
        print("Run: python scripts/train_classifier.py")
        sys.exit(1)

    # ── Identify eval sets ────────────────────────────────────────────
    eval_sets: dict[str, Path] = {}

    if args.eval_set in ("eval", "all"):
        p = processed_dir / data_cfg.get("eval_file", "eval.parquet")
        if p.exists():
            eval_sets["internal_eval"] = p
        else:
            print(f"[WARN] Eval set not found: {p}")

    if args.eval_set in ("rogue", "all"):
        p = processed_dir / data_cfg.get("rogue_eval_file", "rogue_eval.parquet")
        if p.exists():
            eval_sets["rogue_security_benchmark"] = p
        else:
            print(f"[WARN] Rogue benchmark not found: {p}")

    if not eval_sets:
        print("[ERROR] No evaluation sets found. Run prepare_data.py first.")
        sys.exit(1)

    # ── Run evaluations ───────────────────────────────────────────────
    full_report: dict = {"datasets": []}

    for name, path in eval_sets.items():
        df = pd.read_parquet(path)
        report = evaluate_dataset(df, clf, thresholds, name)
        print_report(report)
        full_report["datasets"].append(report)

    # ── Save report ───────────────────────────────────────────────────
    report_path = eval_output_dir / eval_cfg.get("report_file", "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(full_report, fh, indent=2)

    print(f"\n[DONE] Report saved to {report_path}")


if __name__ == "__main__":
    main()
