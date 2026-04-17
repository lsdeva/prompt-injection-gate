"""Stage 2: DeBERTa-v3-base binary classifier inference wrapper."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TypedDict

import yaml


class Stage2Result(TypedDict):
    injection_score: float
    benign_score: float
    latency_ms: float


class ModelNotReadyError(RuntimeError):
    """Raised when the model directory does not exist or is incomplete."""


def _load_config() -> dict:
    here = Path(__file__).parent
    cfg_path = here.parent / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    return {}


class DeBERTaClassifier:
    """Wraps a fine-tuned DeBERTa-v3-base model for prompt-injection scoring.

    Lazy-loads PyTorch/Transformers so that Stage 1 can function without
    them installed.
    """

    def __init__(self, model_dir: str | Path | None = None, device: str = "auto") -> None:
        config = _load_config()
        stage2_cfg = config.get("stage2", {})

        if model_dir is None:
            model_dir = stage2_cfg.get("model_dir", "models/injection-classifier-final")

        # Resolve relative to project root (one level up from src/)
        self._model_dir = Path(model_dir)
        if not self._model_dir.is_absolute():
            self._model_dir = Path(__file__).parent.parent / self._model_dir

        self._max_length: int = int(stage2_cfg.get("max_length", 512))
        self._device_pref: str = device if device != "auto" else stage2_cfg.get("device", "auto")

        self._model = None
        self._tokenizer = None
        self._device = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model and tokenizer from disk.  Raises ModelNotReadyError
        if the model directory does not exist."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not self._model_dir.exists():
            raise ModelNotReadyError(
                f"Model directory not found: {self._model_dir}\n"
                "Run `python scripts/train_classifier.py` first."
            )

        if self._device_pref == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = self._device_pref

        self._device = torch.device(device_str)

        self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(self._model_dir))
        self._model.to(self._device)
        self._model.eval()
        self._loaded = True

        self._warmup()

    def _warmup(self) -> None:
        """Run a dummy inference to populate caches and avoid cold-start latency."""
        if self._loaded:
            self._infer("warmup text")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer(self, text: str) -> tuple[float, float]:
        """Return (injection_score, benign_score) probabilities."""
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze()

        # Convention: label 0 = benign, label 1 = injection
        if probs.dim() == 0:
            # Edge case: single-class output
            benign_score = float(probs)
            injection_score = 1.0 - benign_score
        else:
            benign_score = float(probs[0])
            injection_score = float(probs[1])

        return injection_score, benign_score

    def classify(self, text: str) -> Stage2Result:
        """Classify *text* and return injection/benign scores with latency.

        Raises ModelNotReadyError if load() has not been called successfully.
        """
        if not self._loaded:
            raise ModelNotReadyError("Model not loaded. Call load() first.")

        t0 = time.perf_counter()
        injection_score, benign_score = self._infer(text)
        latency_ms = (time.perf_counter() - t0) * 1000

        return Stage2Result(
            injection_score=round(injection_score, 6),
            benign_score=round(benign_score, 6),
            latency_ms=round(latency_ms, 3),
        )


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_singleton: DeBERTaClassifier | None = None


def get_classifier(model_dir: str | Path | None = None) -> DeBERTaClassifier:
    """Return a module-level singleton classifier, loading it if necessary."""
    global _singleton
    if _singleton is None or not _singleton.is_loaded:
        _singleton = DeBERTaClassifier(model_dir=model_dir)
        _singleton.load()
    return _singleton
