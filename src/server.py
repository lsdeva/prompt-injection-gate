"""FastAPI server exposing the prompt injection gate pipeline.

Run with:
    uvicorn src.server:app --host 0.0.0.0 --port 8081
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .pipeline import InjectionGate, PipelineResult


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    here = Path(__file__).parent
    cfg_path = here.parent / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    return {}


_CONFIG = _load_config()
_SERVER_CFG = _CONFIG.get("server", {})
_LOG_FILE = Path(_SERVER_CFG.get("log_file", "eval/server_requests.jsonl"))


# ---------------------------------------------------------------------------
# State (shared across requests)
# ---------------------------------------------------------------------------

class _ServerState:
    def __init__(self) -> None:
        self.gate: InjectionGate | None = None
        self.start_time: float = time.time()
        self.total_requests: int = 0
        self.blocked_count: int = 0
        self.allowed_count: int = 0
        self.review_count: int = 0
        self.total_latency_ms: float = 0.0
        self.stage_counts: dict[int, int] = {1: 0, 2: 0, 3: 0}
        self.model_ready: bool = False
        self.model_error: str | None = None


_state = _ServerState()


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated startup/shutdown events)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[server] Loading injection gate pipeline…")
    try:
        gate = InjectionGate()
        gate.load()
        _state.gate = gate
        _state.model_ready = gate.classifier_ready
        if not gate.classifier_ready:
            _state.model_error = (
                "Stage 2 classifier not loaded — model may not be trained yet. "
                "Stage 1 heuristics are active."
            )
            print(f"[server] WARNING: {_state.model_error}")
        else:
            print("[server] Pipeline ready.")
    except Exception as exc:
        _state.model_error = str(exc)
        print(f"[server] ERROR during startup: {exc}")
        _state.gate = InjectionGate()  # Stage 1 still works

    # Ensure log directory exists
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    yield

    # Shutdown
    print("[server] Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Prompt Injection Gate",
    description="Multi-stage prompt injection detection pipeline",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_SERVER_CFG.get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to classify")
    threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Override Stage 2 decision threshold (default from config)",
    )


class ClassifyResponse(BaseModel):
    decision: str
    stage: int
    score: Optional[float]
    pattern: Optional[str]
    reasoning: Optional[str]
    latency_ms: float


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_ready: bool
    model_error: Optional[str]
    uptime_seconds: float


class StatsResponse(BaseModel):
    total_requests: int
    blocked_count: int
    allowed_count: int
    review_count: int
    average_latency_ms: float
    stage_distribution: dict[str, int]
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log_request(text_preview: str, result: PipelineResult) -> None:
    """Append a JSONL log entry for each classification."""
    entry = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "text_preview": text_preview[:120],
        "decision": result["decision"],
        "stage": result["stage"],
        "score": result["score"],
        "pattern": result["pattern"],
        "latency_ms": result["latency_ms"],
    }
    try:
        with open(_LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError:
        pass  # Non-critical — don't fail the request over logging


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest) -> ClassifyResponse:
    """Classify a text input for prompt injection."""
    gate = _state.gate
    if gate is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised.")

    result: PipelineResult = gate.classify(req.text, threshold=req.threshold)

    # Update stats
    _state.total_requests += 1
    _state.total_latency_ms += result["latency_ms"]
    decision = result["decision"]
    if decision == "BLOCKED":
        _state.blocked_count += 1
    elif decision == "ALLOWED":
        _state.allowed_count += 1
    else:
        _state.review_count += 1
    stage = result["stage"]
    _state.stage_counts[stage] = _state.stage_counts.get(stage, 0) + 1

    _log_request(req.text, result)

    return ClassifyResponse(
        decision=result["decision"],
        stage=result["stage"],
        score=result["score"],
        pattern=result["pattern"],
        reasoning=result["reasoning"],
        latency_ms=result["latency_ms"],
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return current health status of the server."""
    return HealthResponse(
        status="ok",
        model_ready=_state.model_ready,
        model_error=_state.model_error,
        uptime_seconds=round(time.time() - _state.start_time, 1),
    )


@app.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    """Return classification statistics since server start."""
    avg = (
        round(_state.total_latency_ms / _state.total_requests, 3)
        if _state.total_requests > 0
        else 0.0
    )
    return StatsResponse(
        total_requests=_state.total_requests,
        blocked_count=_state.blocked_count,
        allowed_count=_state.allowed_count,
        review_count=_state.review_count,
        average_latency_ms=avg,
        stage_distribution={
            f"stage_{k}": v for k, v in _state.stage_counts.items()
        },
        uptime_seconds=round(time.time() - _state.start_time, 1),
    )
