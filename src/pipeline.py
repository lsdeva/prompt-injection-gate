"""Pipeline orchestrator: runs Stage 1 → Stage 2 → Stage 3 in sequence.

Decision logic
--------------
Stage 1 (heuristic):
    matched → BLOCKED immediately (stage=1)

Stage 2 (classifier):
    score >= threshold          → BLOCKED  (stage=2)
    score in [lower, upper)     → run Stage 3 if enabled
    score < lower               → ALLOWED  (stage=2)

Stage 3 (semantic):
    is_injection=True           → BLOCKED  (stage=3)
    is_injection=False          → ALLOWED  (stage=3)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal, TypedDict

import yaml

from .stage1_heuristic import check as stage1_check
from .stage2_classifier import DeBERTaClassifier, ModelNotReadyError, Stage2Result
from .stage3_semantic import SemanticAnalyzer, Stage3Result


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Decision = Literal["ALLOWED", "BLOCKED", "REVIEW"]


class PipelineResult(TypedDict):
    decision: Decision
    stage: int
    score: float | None
    pattern: str | None
    reasoning: str | None
    latency_ms: float


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    here = Path(__file__).parent
    cfg_path = here.parent / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    return {}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class InjectionGate:
    """Three-stage prompt injection classifier pipeline.

    Parameters
    ----------
    threshold:
        Stage 2 score above which input is BLOCKED (default from config).
    ambiguous_lower / ambiguous_upper:
        Score range that triggers Stage 3 analysis.
    stage2_classifier:
        Pre-loaded DeBERTaClassifier instance. Pass None to create lazily
        (model must be trained and present on disk).
    stage3_analyzer:
        Pre-loaded SemanticAnalyzer instance.
    """

    def __init__(
        self,
        threshold: float | None = None,
        ambiguous_lower: float | None = None,
        ambiguous_upper: float | None = None,
        stage2_classifier: DeBERTaClassifier | None = None,
        stage3_analyzer: SemanticAnalyzer | None = None,
    ) -> None:
        config = _load_config()
        s2cfg = config.get("stage2", {})

        self._threshold: float = threshold if threshold is not None else float(s2cfg.get("threshold", 0.65))
        self._ambiguous_lower: float = ambiguous_lower if ambiguous_lower is not None else float(s2cfg.get("ambiguous_lower", 0.40))
        self._ambiguous_upper: float = ambiguous_upper if ambiguous_upper is not None else float(s2cfg.get("ambiguous_upper", 0.70))

        self._classifier: DeBERTaClassifier | None = stage2_classifier
        self._analyzer: SemanticAnalyzer | None = stage3_analyzer
        self._classifier_error: str | None = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Explicitly load Stage 2 and Stage 3 components."""
        self._load_classifier()
        if self._analyzer is None:
            self._analyzer = SemanticAnalyzer()

    def _load_classifier(self) -> None:
        if self._classifier is not None and self._classifier.is_loaded:
            return
        try:
            from .stage2_classifier import DeBERTaClassifier
            clf = DeBERTaClassifier()
            clf.load()
            self._classifier = clf
            self._classifier_error = None
        except ModelNotReadyError as exc:
            self._classifier_error = str(exc)
            self._classifier = None
        except Exception as exc:
            self._classifier_error = f"Unexpected error loading model: {exc}"
            self._classifier = None

    @property
    def classifier_ready(self) -> bool:
        return self._classifier is not None and self._classifier.is_loaded

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, text: str, threshold: float | None = None) -> PipelineResult:
        """Run the full three-stage pipeline and return a PipelineResult."""
        t0 = time.perf_counter()
        effective_threshold = threshold if threshold is not None else self._threshold

        # ── Stage 1 ──────────────────────────────────────────────────
        s1 = stage1_check(text)
        if s1["matched"]:
            return PipelineResult(
                decision="BLOCKED",
                stage=1,
                score=None,
                pattern=s1["pattern_name"],
                reasoning=f"Heuristic pattern matched: {s1['pattern_name']}",
                latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            )

        # ── Stage 2 ──────────────────────────────────────────────────
        if self._classifier is None:
            self._load_classifier()

        if self._classifier is None:
            # Model not trained yet — fail open but report clearly
            return PipelineResult(
                decision="REVIEW",
                stage=2,
                score=None,
                pattern=None,
                reasoning=f"Stage 2 model unavailable: {self._classifier_error}",
                latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            )

        s2: Stage2Result = self._classifier.classify(text)
        score = s2["injection_score"]

        if score >= effective_threshold:
            return PipelineResult(
                decision="BLOCKED",
                stage=2,
                score=score,
                pattern=None,
                reasoning=f"Classifier score {score:.4f} exceeds threshold {effective_threshold:.2f}",
                latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            )

        if score < self._ambiguous_lower:
            return PipelineResult(
                decision="ALLOWED",
                stage=2,
                score=score,
                pattern=None,
                reasoning=f"Classifier score {score:.4f} below ambiguous zone",
                latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            )

        # ── Stage 3 ──────────────────────────────────────────────────
        if self._analyzer is None:
            self._analyzer = SemanticAnalyzer()

        if not self._analyzer.is_enabled:
            # Ambiguous but no Stage 3 — allow with REVIEW flag
            return PipelineResult(
                decision="REVIEW",
                stage=2,
                score=score,
                pattern=None,
                reasoning=f"Score {score:.4f} in ambiguous zone [{self._ambiguous_lower}, {self._ambiguous_upper}); Stage 3 disabled",
                latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            )

        s3: Stage3Result = self._analyzer.analyze(text)

        decision: Decision = "BLOCKED" if s3["is_injection"] else "ALLOWED"
        return PipelineResult(
            decision=decision,
            stage=3,
            score=s3["confidence"],
            pattern=None,
            reasoning=s3["reasoning"],
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_gate: InjectionGate | None = None


def get_gate() -> InjectionGate:
    """Return (and lazily load) the module-level InjectionGate singleton."""
    global _gate
    if _gate is None:
        _gate = InjectionGate()
        _gate.load()
    return _gate
