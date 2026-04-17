"""Stage 1: Regex/heuristic pre-filter.

All patterns are compiled at module load time to guarantee sub-2ms latency
per call. No external dependencies required.
"""

from __future__ import annotations

import base64
import re
import time
from pathlib import Path
from typing import TypedDict

import yaml


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Stage1Result(TypedDict):
    matched: bool
    pattern_name: str | None
    latency_ms: float


# ---------------------------------------------------------------------------
# Pattern loading and compilation
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config.yaml relative to the project root."""
    here = Path(__file__).parent
    config_path = here.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    return {}


def _compile_patterns(config: dict) -> list[tuple[str, re.Pattern, bool]]:
    """Return list of (name, compiled_regex, decode_check) tuples."""
    flag_map = {
        "IGNORECASE": re.IGNORECASE,
        "MULTILINE": re.MULTILINE,
        "DOTALL": re.DOTALL,
        "0": 0,
    }
    patterns: list[tuple[str, re.Pattern, bool]] = []
    raw = config.get("stage1", {}).get("patterns", [])
    for entry in raw:
        name: str = entry["name"]
        flags_str: str = str(entry.get("flags", "0"))
        flags = flag_map.get(flags_str, 0)
        compiled = re.compile(entry["regex"], flags)
        decode_check: bool = bool(entry.get("decode_check", False))
        patterns.append((name, compiled, decode_check))
    return patterns


_CONFIG = _load_config()
_PATTERNS: list[tuple[str, re.Pattern, bool]] = _compile_patterns(_CONFIG)

# Fallback patterns used when config is unavailable
_FALLBACK_PATTERNS: list[tuple[str, re.Pattern, bool]] = [
    ("ignore_instructions",
     re.compile(r"ignore\s+(all\s+)?(previous\s+|prior\s+)?instructions", re.IGNORECASE), False),
    ("disregard",
     re.compile(r"disregard\s+(all\s+|prior\s+|above\s+|previous\s+)", re.IGNORECASE), False),
    ("you_are_now",
     re.compile(r"you\s+are\s+now\s+(DAN|in developer mode|jailbroken|unfiltered)", re.IGNORECASE), False),
    ("system_prompt_injection",
     re.compile(r"system\s*prompt\s*[:=]", re.IGNORECASE), False),
    ("base64_decode",
     re.compile(r"base64\s*(decode|encoded|string)", re.IGNORECASE), False),
    ("no_restrictions",
     re.compile(r"pretend\s+you\s+(are|have)\s+no\s+(restrictions|rules|guidelines)", re.IGNORECASE), False),
    ("template_injection_inst",
     re.compile(r"\[INST\]|\[/INST\]"), False),
    ("template_injection_sys",
     re.compile(r"<<SYS>>|<</SYS>>"), False),
    ("do_anything_now",
     re.compile(r"do\s+anything\s+now", re.IGNORECASE), False),
    ("respond_as_unrestricted",
     re.compile(r"respond\s+as\s+(an\s+unrestricted|a\s+completely\s+different)", re.IGNORECASE), False),
    ("jailbreak_dan",
     re.compile(r"\bDAN\b|developer\s+mode\s+enabled|jailbreak\s+mode", re.IGNORECASE), False),
    ("override_directives",
     re.compile(r"(override|bypass|circumvent)\s+(your\s+)?(safety|guidelines|rules|restrictions|directives|training)", re.IGNORECASE), False),
    ("new_instructions",
     re.compile(r"your\s+new\s+(instructions|directive|task|role|purpose)\s+(is|are|will)", re.IGNORECASE), False),
    ("forget_previous",
     re.compile(r"(forget|clear|erase|reset)\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|context|rules)", re.IGNORECASE), False),
    ("base64_payload",
     re.compile(r"[A-Za-z0-9+/]{40,}={0,2}"), True),
]

# Use config patterns if available, else fall back to hardcoded list
ACTIVE_PATTERNS = _PATTERNS if _PATTERNS else _FALLBACK_PATTERNS

# Pre-compiled injection keywords for decoded base64 payloads
_DECODED_INJECTION_RE = re.compile(
    r"(ignore|disregard|override|bypass|jailbreak|DAN|system\s*prompt|new\s+instructions)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Core detection logic
# ---------------------------------------------------------------------------

def _check_base64_payload(text: str) -> bool:
    """Return True if any long base64 token decodes to injection-like content."""
    b64_re = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
    for match in b64_re.finditer(text):
        candidate = match.group(0)
        try:
            decoded = base64.b64decode(candidate + "==").decode("utf-8", errors="ignore")
            if _DECODED_INJECTION_RE.search(decoded):
                return True
        except Exception:
            continue
    return False


def check(text: str) -> Stage1Result:
    """Run all heuristic patterns against *text*.

    Returns a :class:`Stage1Result` with the first match found (or
    ``matched=False`` when none trigger).  Latency is measured wall-clock
    inside this function.
    """
    t0 = time.perf_counter()

    for name, pattern, decode_check in ACTIVE_PATTERNS:
        if pattern.search(text):
            if decode_check and not _check_base64_payload(text):
                continue
            latency_ms = (time.perf_counter() - t0) * 1000
            return Stage1Result(matched=True, pattern_name=name, latency_ms=round(latency_ms, 3))

    latency_ms = (time.perf_counter() - t0) * 1000
    return Stage1Result(matched=False, pattern_name=None, latency_ms=round(latency_ms, 3))


def get_all_pattern_names() -> list[str]:
    """Return list of all registered pattern names (useful for tests)."""
    return [name for name, _, _ in ACTIVE_PATTERNS]
