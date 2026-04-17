"""Fine-tune DeBERTa-v3-base on the prepared prompt-injection dataset.

Usage:
    python scripts/train_classifier.py
    python scripts/train_classifier.py --batch-size 8 --cpu-only
    python scripts/train_classifier.py --resume-from models/injection-classifier/checkpoint-500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config(project_root: Path) -> dict:
    with open(project_root / "config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "f1": f1_score(labels, predictions, average="binary", zero_division=0),
        "precision": precision_score(labels, predictions, average="binary", zero_division=0),
        "recall": recall_score(labels, predictions, average="binary", zero_division=0),
        "accuracy": accuracy_score(labels, predictions),
    }


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class InjectionDataset:
    """Torch Dataset wrapping a parquet file."""

    def __init__(self, path: Path, tokenizer, max_length: int) -> None:
        import torch

        df = pd.read_parquet(path)
        self._texts = df["text"].tolist()
        self._labels = df["label"].tolist()
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self._tokenizer(
            self._texts[idx],
            truncation=True,
            max_length=self._max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": self._labels[idx],
        }
        # token_type_ids only if the tokenizer returns them (DeBERTa-v3 does not)
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze()
        return item


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    project_root = Path(__file__).parent.parent
    config = _load_config(project_root)
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    base_model: str = train_cfg.get("base_model", "microsoft/deberta-v3-base")
    output_dir = project_root / train_cfg.get("output_dir", "models/injection-classifier")
    final_dir = project_root / train_cfg.get("final_dir", "models/injection-classifier-final")
    processed_dir = project_root / data_cfg.get("processed_dir", "data/processed")

    train_path = processed_dir / data_cfg.get("train_file", "train.parquet")
    eval_path = processed_dir / data_cfg.get("eval_file", "eval.parquet")

    if not train_path.exists():
        print(f"[ERROR] Training data not found: {train_path}")
        print("Run: python scripts/prepare_data.py")
        sys.exit(1)

    # Device selection
    if args.cpu_only:
        use_fp16 = False
        print("[INFO] CPU-only mode — FP16 disabled")
    elif args.no_fp16:
        use_fp16 = False
        print("[INFO] FP16 disabled by --no-fp16 flag")
    else:
        cuda_available = torch.cuda.is_available()
        fp16_cfg = train_cfg.get("fp16", "auto")
        if fp16_cfg == "auto":
            # Pascal GPUs (GTX 10xx) don't support FP16 grad scaling well
            use_fp16 = cuda_available and torch.cuda.get_device_capability()[0] >= 7
        else:
            use_fp16 = bool(fp16_cfg)
        print(f"[INFO] CUDA: {'available' if cuda_available else 'not available'}  FP16: {use_fp16}")

    # Hyperparameters (CLI overrides config)
    epochs = args.epochs if args.epochs is not None else int(train_cfg.get("num_train_epochs", 5))
    batch_size = args.batch_size if args.batch_size is not None else int(train_cfg.get("per_device_train_batch_size", 16))
    max_length = args.max_length if args.max_length is not None else int(train_cfg.get("max_length", 512))
    grad_accum = args.gradient_accumulation if args.gradient_accumulation is not None else int(train_cfg.get("gradient_accumulation_steps", 1))

    print(f"[INFO] Model        : {base_model}")
    print(f"[INFO] Epochs       : {epochs}")
    print(f"[INFO] Batch        : {batch_size}  (grad_accum={grad_accum}, effective={batch_size * grad_accum})")
    print(f"[INFO] MaxLen       : {max_length}")
    print(f"[INFO] OutDir       : {output_dir}")
    if args.max_train_samples:
        print(f"[INFO] Max samples  : {args.max_train_samples} (smoke-test mode)")

    # Load tokenizer and model
    print("\n[INFO] Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print("[INFO] Loading model…")
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        id2label={0: "benign", 1: "injection"},
        label2id={"benign": 0, "injection": 1},
    )
    # Ensure float32 — some checkpoints ship as float16 (e.g. DeBERTa-v3).
    # Pascal GPUs (GTX 10xx, compute cap 6.1) can't run FP16 training without
    # grad scaling, which overflows, so always force FP32 before training.
    if next(model.parameters()).dtype != torch.float32:
        model = model.float()
        print("[INFO] Model cast to float32")
    else:
        print(f"[INFO] Model dtype: float32")

    # Datasets
    print("[INFO] Loading datasets…")
    train_ds = InjectionDataset(train_path, tokenizer, max_length)
    eval_ds = InjectionDataset(eval_path, tokenizer, max_length) if eval_path.exists() else None

    # Optionally cap training samples for quick smoke tests
    if args.max_train_samples and args.max_train_samples < len(train_ds):
        from torch.utils.data import Subset
        import random
        indices = random.sample(range(len(train_ds)), args.max_train_samples)
        train_ds = Subset(train_ds, indices)
        print(f"  Train: {len(train_ds):,} samples (capped from full set)")
    else:
        print(f"  Train: {len(train_ds):,} samples")
    if eval_ds:
        print(f"  Eval : {len(eval_ds):,} samples")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute warmup_steps from warmup_ratio since warmup_ratio is deprecated in v5+
    total_steps = (len(train_ds) // batch_size) * epochs // grad_accum
    warmup_steps = max(1, int(total_steps * float(train_cfg.get("warmup_ratio", 0.1))))
    # Eval/save every ~1/3 epoch (min 50, max 500 steps)
    eval_save_steps = max(50, min(500, total_steps // 3))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 32)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        warmup_steps=warmup_steps,
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        fp16=use_fp16,
        eval_strategy=train_cfg.get("evaluation_strategy", "steps"),
        eval_steps=eval_save_steps,
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=eval_save_steps,
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "f1"),
        greater_is_better=True,
        logging_steps=int(train_cfg.get("logging_steps", 100)),
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 0)),
        report_to="none",
        save_total_limit=3,
        use_cpu=args.cpu_only,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    # Resume from checkpoint if specified
    resume_from = args.resume_from
    if resume_from and Path(resume_from).exists():
        print(f"[INFO] Resuming from checkpoint: {resume_from}")

    print("\n[INFO] Starting training…")
    trainer.train(resume_from_checkpoint=resume_from)

    print(f"\n[INFO] Saving final model to {final_dir}…")
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print("\n[DONE] Training complete.")
    print(f"       Final model: {final_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa-v3-base for prompt injection detection")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device train batch size (default from config)")
    parser.add_argument("--max-length", type=int, default=None, help="Max token sequence length (default from config)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (default from config)")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU training even if GPU is available")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to a checkpoint directory to resume from")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Cap training samples for quick smoke tests (e.g. 500)")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 mixed precision (needed for Pascal GPUs like GTX 10xx)")
    parser.add_argument("--gradient-accumulation", type=int, default=None, help="Gradient accumulation steps (use with small batch to keep effective batch size)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
