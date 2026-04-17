"""Download all six datasets from Hugging Face to data/raw/.

Usage:
    python scripts/download_datasets.py                 # download all
    python scripts/download_datasets.py --dataset advbench

Notes:
- Gated datasets (AdvBench) require `huggingface-cli login` first.
- Already-downloaded datasets are skipped automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(project_root: Path) -> dict:
    cfg_path = project_root / "config.yaml"
    if not cfg_path.exists():
        print(f"[ERROR] config.yaml not found at {cfg_path}")
        sys.exit(1)
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _is_downloaded(save_dir: Path) -> bool:
    """Return True if the directory exists and contains at least one file."""
    if not save_dir.exists():
        return False
    return any(save_dir.iterdir())


def _print_schema(hf_id: str) -> None:
    """Print the dataset card info: splits and column names."""
    try:
        from datasets import get_dataset_config_names, load_dataset_builder

        try:
            configs = get_dataset_config_names(hf_id)
            config_name = configs[0] if configs else None
        except Exception:
            config_name = None

        builder = load_dataset_builder(hf_id, config_name)
        info = builder.info
        print(f"  Splits  : {list(info.splits.keys()) if info.splits else 'unknown'}")
        if info.features:
            print(f"  Columns : {list(info.features.keys())}")
    except Exception as exc:
        print(f"  (Could not inspect schema: {exc})")


def _download_one(name: str, cfg: dict, project_root: Path, dry_run: bool = False) -> bool:
    """Download a single dataset. Returns True on success."""
    from datasets import load_dataset

    hf_id: str = cfg["hf_id"]
    save_dir: Path = project_root / cfg["save_dir"]

    if _is_downloaded(save_dir):
        print(f"[SKIP] {name} already present at {save_dir}")
        return True

    print(f"\n[INFO] Dataset: {name} ({hf_id})")
    print(f"       License : {cfg.get('license', 'unknown')}")
    _print_schema(hf_id)

    if dry_run:
        print("  (dry-run mode — skipping download)")
        return True

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Saving to {save_dir} …")

    try:
        ds = load_dataset(hf_id, trust_remote_code=True)
    except Exception as exc:
        err_str = str(exc)
        if "gated" in err_str.lower() or "authentication" in err_str.lower() or "401" in err_str:
            print(f"  [AUTH ERROR] Dataset '{hf_id}' is gated.")
            print("  Run:  huggingface-cli login")
            print("  Then accept the dataset terms at https://huggingface.co/datasets/" + hf_id)
            return False
        print(f"  [ERROR] Failed to download {name}: {exc}")
        return False

    ds.save_to_disk(str(save_dir))

    # Print sizes
    for split_name, split_ds in ds.items():
        print(f"    {split_name}: {len(split_ds):,} rows")

    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets for prompt-injection-gate")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Name of a single dataset to download (e.g. advbench). Omit for all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print dataset info without downloading.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    config = _load_config(project_root)
    datasets_cfg: dict = config.get("datasets", {})

    if args.dataset:
        if args.dataset not in datasets_cfg:
            print(f"[ERROR] Unknown dataset '{args.dataset}'. Available: {list(datasets_cfg.keys())}")
            sys.exit(1)
        targets = {args.dataset: datasets_cfg[args.dataset]}
    else:
        targets = datasets_cfg

    successes, failures = [], []

    for name, cfg in targets.items():
        ok = _download_one(name, cfg, project_root, dry_run=args.dry_run)
        if ok:
            successes.append(name)
        else:
            failures.append(name)

    print("\n" + "=" * 60)
    print(f"Done. Succeeded: {successes}")
    if failures:
        print(f"Failed : {failures}")
        print("Check authentication (huggingface-cli login) for gated datasets.")
    print("=" * 60)


if __name__ == "__main__":
    main()
