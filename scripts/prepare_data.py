"""Normalize, deduplicate, balance, and split all datasets.

Output:
    data/processed/train.parquet
    data/processed/eval.parquet
    data/processed/rogue_eval.parquet

Common schema: {"text": str, "label": int, "source": str, "attack_type": str}
  label:       0 = benign, 1 = injection/jailbreak
  attack_type: benign | direct_override | persona_manipulation |
               encoding_exploit | jailbreak | unknown
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config(project_root: Path) -> dict:
    cfg_path = project_root / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

SCHEMA_COLS = ["text", "label", "source", "attack_type"]


def _make_row(text: str, label: int, source: str, attack_type: str) -> dict:
    return {"text": str(text).strip(), "label": int(label), "source": source, "attack_type": attack_type}


def _validate_not_rogue(source: str) -> None:
    """Hard guard: rogue_benchmark must never enter training data."""
    if "rogue" in source.lower():
        raise ValueError(
            "SAFETY CHECK FAILED: Attempted to include rogue-security benchmark "
            "in training data. This dataset is CC-BY-NC-4.0 and must be used for "
            "evaluation ONLY. Check your prepare_data.py logic."
        )


# ---------------------------------------------------------------------------
# Dataset-specific loaders
# ---------------------------------------------------------------------------

def _load_hf_disk(path: Path):
    """Load a dataset saved with save_to_disk()."""
    from datasets import load_from_disk
    return load_from_disk(str(path))


def load_wildchat(raw_dir: Path, cfg: dict) -> list[dict]:
    """Extract user messages from WildChat-1M conversations (label=0, benign)."""
    ds_path = raw_dir / "wildchat"
    if not ds_path.exists():
        print(f"  [SKIP] WildChat not found at {ds_path}")
        return []

    ds = _load_hf_disk(ds_path)
    print(f"  WildChat splits: {list(ds.keys())}")

    rows: list[dict] = []
    for split_name in ds:
        split = ds[split_name]
        print(f"    {split_name}: columns = {split.column_names}")

        for item in tqdm(split, desc=f"    WildChat/{split_name}", leave=False):
            # 'conversation' is a list of turn dicts with 'role' and 'content'
            conv = item.get("conversation", [])
            for turn in conv:
                if turn.get("role") == "user":
                    text = turn.get("content", "").strip()
                    if len(text) < 5:
                        continue
                    # Filter toxic content if flagged
                    if item.get("toxic", False):
                        continue
                    rows.append(_make_row(text, 0, "wildchat", "benign"))

    print(f"  WildChat: extracted {len(rows):,} user messages")
    return rows


def load_oasst1(raw_dir: Path, cfg: dict) -> list[dict]:
    """Extract prompter messages from oasst1 (label=0, benign)."""
    ds_path = raw_dir / "oasst1"
    if not ds_path.exists():
        print(f"  [SKIP] oasst1 not found at {ds_path}")
        return []

    ds = _load_hf_disk(ds_path)
    print(f"  oasst1 splits: {list(ds.keys())}")

    rows: list[dict] = []
    for split_name in ds:
        split = ds[split_name]
        print(f"    {split_name}: columns = {split.column_names}")

        role_col = cfg.get("role_col", "role")
        text_col = cfg.get("text_col", "text")
        role_filter = cfg.get("role_filter", "prompter")

        for item in tqdm(split, desc=f"    oasst1/{split_name}", leave=False):
            if item.get(role_col) != role_filter:
                continue
            text = str(item.get(text_col, "")).strip()
            if len(text) < 5:
                continue
            rows.append(_make_row(text, 0, "oasst1", "benign"))

    print(f"  oasst1: extracted {len(rows):,} prompter messages")
    return rows


def load_neuralchemy(raw_dir: Path, cfg: dict) -> list[dict]:
    """Load neuralchemy/Prompt-injection-dataset (has native labels)."""
    ds_path = raw_dir / "neuralchemy"
    if not ds_path.exists():
        print(f"  [SKIP] neuralchemy not found at {ds_path}")
        return []

    ds = _load_hf_disk(ds_path)
    print(f"  neuralchemy splits: {list(ds.keys())}")

    rows: list[dict] = []
    for split_name in ds:
        split = ds[split_name]
        print(f"    {split_name}: columns = {split.column_names}")

        # Discover text and label columns
        cols = split.column_names
        text_col = cfg.get("text_col", "text")
        label_col = cfg.get("label_col", "label")

        # Fallback column discovery
        if text_col not in cols:
            candidates = [c for c in cols if "text" in c.lower() or "prompt" in c.lower() or "input" in c.lower()]
            text_col = candidates[0] if candidates else cols[0]
            print(f"    [INFO] Using column '{text_col}' as text")
        if label_col not in cols:
            candidates = [c for c in cols if "label" in c.lower() or "class" in c.lower() or "target" in c.lower()]
            label_col = candidates[0] if candidates else None
            print(f"    [INFO] Using column '{label_col}' as label")

        for item in tqdm(split, desc=f"    neuralchemy/{split_name}", leave=False):
            text = str(item.get(text_col, "")).strip()
            if len(text) < 5:
                continue
            raw_label = item.get(label_col, 0) if label_col else 0
            # Normalise various label formats
            if isinstance(raw_label, str):
                label = 1 if raw_label.lower() in ("1", "injection", "injected", "malicious", "attack") else 0
            else:
                label = int(raw_label)
            attack_type = "direct_override" if label == 1 else "benign"
            rows.append(_make_row(text, label, "neuralchemy", attack_type))

    print(f"  neuralchemy: extracted {len(rows):,} rows")
    return rows


def load_advbench(raw_dir: Path, cfg: dict) -> list[dict]:
    """Load AdvBench — all rows are adversarial (label=1)."""
    ds_path = raw_dir / "advbench"
    if not ds_path.exists():
        print(f"  [SKIP] AdvBench not found at {ds_path}")
        return []

    ds = _load_hf_disk(ds_path)
    print(f"  AdvBench splits: {list(ds.keys())}")

    rows: list[dict] = []
    for split_name in ds:
        split = ds[split_name]
        print(f"    {split_name}: columns = {split.column_names}")

        cols = split.column_names
        text_col = cfg.get("text_col", "prompt")
        if text_col not in cols:
            candidates = [c for c in cols if "prompt" in c.lower() or "text" in c.lower()]
            text_col = candidates[0] if candidates else cols[0]

        for item in tqdm(split, desc=f"    AdvBench/{split_name}", leave=False):
            text = str(item.get(text_col, "")).strip()
            if len(text) < 5:
                continue
            rows.append(_make_row(text, 1, "advbench", "jailbreak"))

    print(f"  AdvBench: extracted {len(rows):,} rows")
    return rows


def load_tensor_trust(raw_dir: Path, cfg: dict) -> list[dict]:
    """Load HumanCompatibleAI/tensor-trust-data — attacks = label 1."""
    ds_path = raw_dir / "tensor_trust"
    if not ds_path.exists():
        print(f"  [SKIP] tensor-trust not found at {ds_path}")
        return []

    ds = _load_hf_disk(ds_path)
    print(f"  tensor-trust splits: {list(ds.keys())}")

    rows: list[dict] = []
    for split_name in ds:
        split = ds[split_name]
        print(f"    {split_name}: columns = {split.column_names}")

        cols = split.column_names
        # Try to find the attack text column
        text_col = cfg.get("text_col", "attack")
        if text_col not in cols:
            candidates = [c for c in cols if "attack" in c.lower() or "prompt" in c.lower() or "text" in c.lower()]
            text_col = candidates[0] if candidates else cols[0]
            print(f"    [INFO] Using column '{text_col}' as text")

        # Determine if split contains attacks or defenses
        is_attack_split = "attack" in split_name.lower()
        label = 1 if is_attack_split else 0
        attack_type = "direct_override" if is_attack_split else "benign"

        for item in tqdm(split, desc=f"    tensor-trust/{split_name}", leave=False):
            text = str(item.get(text_col, "")).strip()
            if len(text) < 5:
                continue
            rows.append(_make_row(text, label, "tensor_trust", attack_type))

    print(f"  tensor-trust: extracted {len(rows):,} rows")
    return rows


def load_rogue_benchmark(raw_dir: Path, cfg: dict) -> list[dict]:
    """Load rogue-security benchmark — FOR EVALUATION ONLY."""
    ds_path = raw_dir / "rogue_benchmark"
    if not ds_path.exists():
        print(f"  [SKIP] rogue-benchmark not found at {ds_path}")
        return []

    ds = _load_hf_disk(ds_path)
    print(f"  rogue-benchmark splits: {list(ds.keys())}")

    rows: list[dict] = []
    for split_name in ds:
        split = ds[split_name]
        print(f"    {split_name}: columns = {split.column_names}")

        cols = split.column_names
        text_col = cfg.get("text_col", "prompt")
        label_col = cfg.get("label_col", "label")

        if text_col not in cols:
            candidates = [c for c in cols if "prompt" in c.lower() or "text" in c.lower()]
            text_col = candidates[0] if candidates else cols[0]
        if label_col not in cols:
            candidates = [c for c in cols if "label" in c.lower()]
            label_col = candidates[0] if candidates else None

        for item in tqdm(split, desc=f"    rogue/{split_name}", leave=False):
            text = str(item.get(text_col, "")).strip()
            if len(text) < 5:
                continue
            raw_label = item.get(label_col, 1) if label_col else 1
            if isinstance(raw_label, str):
                label = 1 if raw_label.lower() in ("1", "injection", "attack", "malicious") else 0
            else:
                label = int(raw_label)
            rows.append(_make_row(text, label, "rogue_benchmark", "unknown"))

    print(f"  rogue-benchmark: extracted {len(rows):,} rows (EVAL ONLY)")
    return rows


# ---------------------------------------------------------------------------
# Deduplication (MinHash LSH)
# ---------------------------------------------------------------------------

def deduplicate(rows: list[dict], threshold: float = 0.85, num_perm: int = 128) -> list[dict]:
    """Remove near-duplicate texts using MinHash LSH."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("  [WARN] datasketch not installed — skipping deduplication")
        return rows

    print(f"  Running MinHash deduplication (threshold={threshold}, n={len(rows):,})…")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    unique: list[dict] = []
    duplicates = 0

    for i, row in enumerate(tqdm(rows, desc="  Dedup")):
        tokens = set(row["text"].lower().split())
        m = MinHash(num_perm=num_perm)
        for token in tokens:
            m.update(token.encode("utf-8"))

        key = f"row_{i}"
        try:
            result = lsh.query(m)
            if result:
                duplicates += 1
                continue
            lsh.insert(key, m)
            unique.append(row)
        except Exception:
            unique.append(row)

    print(f"  Dedup: removed {duplicates:,} duplicates. Remaining: {len(unique):,}")
    return unique


# ---------------------------------------------------------------------------
# Main preparation logic
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare training and evaluation data")
    parser.add_argument("--no-dedup", action="store_true", help="Skip MinHash deduplication")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    config = _load_config(project_root)
    data_cfg = config.get("data", {})
    datasets_cfg = config.get("datasets", {})
    dedup_cfg = config.get("data", {}).get("dedup", {})

    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / data_cfg.get("processed_dir", "data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    # ── Load each dataset ──────────────────────────────────────────────
    print("\n=== Loading datasets ===")

    loaders = {
        "wildchat": (load_wildchat, datasets_cfg.get("wildchat", {})),
        "oasst1": (load_oasst1, datasets_cfg.get("oasst1", {})),
        "neuralchemy": (load_neuralchemy, datasets_cfg.get("neuralchemy", {})),
        "advbench": (load_advbench, datasets_cfg.get("advbench", {})),
        "tensor_trust": (load_tensor_trust, datasets_cfg.get("tensor_trust", {})),
    }

    training_rows: list[dict] = []
    for name, (loader_fn, ds_cfg) in loaders.items():
        if not ds_cfg.get("include_in_training", True):
            print(f"  [SKIP] {name} excluded from training by config")
            continue
        print(f"\n-- {name} --")
        rows = loader_fn(raw_dir, ds_cfg)
        for r in rows:
            _validate_not_rogue(r["source"])  # Safety guard
        training_rows.extend(rows)

    # ── Load rogue benchmark separately ────────────────────────────────
    print("\n-- rogue_benchmark (EVAL ONLY) --")
    rogue_cfg = datasets_cfg.get("rogue_benchmark", {})
    rogue_rows = load_rogue_benchmark(raw_dir, rogue_cfg)

    # ── Deduplication ──────────────────────────────────────────────────
    if not args.no_dedup and dedup_cfg.get("enabled", True):
        print("\n=== Deduplication ===")
        training_rows = deduplicate(
            training_rows,
            threshold=float(dedup_cfg.get("jaccard_threshold", 0.85)),
            num_perm=int(dedup_cfg.get("num_perm", 128)),
        )

    # ── Stats before balancing ─────────────────────────────────────────
    if not training_rows:
        print("\n[ERROR] No training rows collected. Run download_datasets.py first.")
        sys.exit(1)

    df = pd.DataFrame(training_rows, columns=SCHEMA_COLS)
    print(f"\n=== Pre-balance statistics ===")
    print(f"  Total rows : {len(df):,}")
    print(f"  Label dist :\n{df['label'].value_counts().to_string()}")
    print(f"  Source dist:\n{df['source'].value_counts().to_string()}")

    # ── Class balancing: 80:20 benign:injection ────────────────────────
    print("\n=== Balancing to 80:20 (benign:injection) ===")
    benign_ratio = float(data_cfg.get("class_ratio", {}).get("benign", 0.80))

    df_benign = df[df["label"] == 0]
    df_inject = df[df["label"] == 1]
    n_inject = len(df_inject)

    if n_inject == 0:
        print("  [WARN] No injection samples found — skipping balancing")
    else:
        n_benign_target = int(n_inject * benign_ratio / (1 - benign_ratio))
        if len(df_benign) > n_benign_target:
            df_benign = df_benign.sample(n=n_benign_target, random_state=args.seed)
            print(f"  Downsampled benign to {n_benign_target:,} (injection={n_inject:,})")
        df = pd.concat([df_benign, df_inject]).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # ── Train/eval split ───────────────────────────────────────────────
    from sklearn.model_selection import train_test_split

    train_ratio = float(data_cfg.get("train_eval_split", 0.85))
    train_df, eval_df = train_test_split(
        df, test_size=1 - train_ratio, stratify=df["label"], random_state=args.seed
    )
    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────────────
    train_path = processed_dir / data_cfg.get("train_file", "train.parquet")
    eval_path = processed_dir / data_cfg.get("eval_file", "eval.parquet")
    rogue_path = processed_dir / data_cfg.get("rogue_eval_file", "rogue_eval.parquet")

    train_df.to_parquet(train_path, index=False)
    eval_df.to_parquet(eval_path, index=False)
    print(f"\n  Saved train -> {train_path}  ({len(train_df):,} rows)")
    print(f"  Saved eval  -> {eval_path}  ({len(eval_df):,} rows)")

    if rogue_rows:
        rogue_df = pd.DataFrame(rogue_rows, columns=SCHEMA_COLS)
        rogue_df.to_parquet(rogue_path, index=False)
        print(f"  Saved rogue -> {rogue_path}  ({len(rogue_df):,} rows)")

    # ── Final stats ───────────────────────────────────────────────────
    print("\n=== Final statistics ===")
    print(f"  Train set : {len(train_df):,} rows")
    print(f"    Labels  :\n{train_df['label'].value_counts().to_string()}")
    print(f"  Eval set  : {len(eval_df):,} rows")
    print(f"    Labels  :\n{eval_df['label'].value_counts().to_string()}")
    print("\nData preparation complete.")


if __name__ == "__main__":
    main()
