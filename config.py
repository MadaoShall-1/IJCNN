"""SolverConfig — unified runtime parameters for the full pipeline.

All scoring-sensitive and compute-sensitive decisions are controlled here
so that pipeline behaviour can be tuned after deployment without touching
pipeline code (design §3.2).

Usage::

    from config import SolverConfig
    cfg = SolverConfig()              # all defaults
    cfg = SolverConfig(beam_n=1)      # reduce beam for speed
    cfg = SolverConfig.from_dict({    # load from JSON / env
        "type1_use_z3": False,
        "generate_fol": False,
    })
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ConfidenceThresholds:
    """Per-format confidence floor below which abstain_behavior applies.

    Default 0.0 = always emit an answer.  This is correct for the competition
    scoring model (P1 + P2 + P3, no penalty for wrong answers).  Raise these
    thresholds only if penalty scoring is confirmed at the kick-off workshop.
    """

    mcq: float = 0.0
    yes_no: float = 0.0
    numerical: float = 0.0
    open_ended: float = 0.0

    def for_format(self, fmt: str) -> float:
        """Return threshold for a format string (e.g. 'mcq', 'yes_no')."""
        return getattr(self, fmt.replace("-", "_"), 0.0)


@dataclass
class SolverConfig:
    """Runtime parameters controlling the full pipeline (design §3.2).

    Fields are grouped by concern.  Every field has a sensible default so
    ``SolverConfig()`` is always a valid Phase 0 configuration.
    """

    # ── Scoring / abstain ────────────────────────────────────────────────────
    confidence_threshold: ConfidenceThresholds = field(
        default_factory=ConfidenceThresholds
    )
    # "best_effort": always emit an answer (correct for no-penalty scoring).
    # "blank": omit the answer field — only if penalty scoring is confirmed.
    abstain_behavior: str = "best_effort"

    # ── Type 2 beam / repair ─────────────────────────────────────────────────
    beam_n: int = 3                # candidate formula paths to explore
    step_retry_limit: int = 3      # per-step solver retry attempts
    trace_budget: int = 10         # max total solver invocations per trace
    repair_budget: int = 3         # max FWS-centred repair attempts (Stage 5)

    # ── Pipeline path enablement ─────────────────────────────────────────────
    type1_enabled: bool = True
    type2_enabled: bool = True

    # Phase 1+ Type 1 features (no-ops in Phase 0; activate when adapters ready)
    type1_use_z3: bool = True      # Z3 for Yes/No/Uncertain; falls to LLM if False
    type1_verify: bool = True      # second verifier-adapter pass for MCQ/open-ended

    # ── Optional output fields ───────────────────────────────────────────────
    generate_fol: bool = True
    generate_cot: bool = True
    generate_premises: bool = True
    generate_confidence: bool = True

    # ── Latency / timeout (seconds) ──────────────────────────────────────────
    # Hard cap per request is 60 s (competition Q13).  55 s leaves 5 s margin.
    latency_budget_seconds: float = 55.0
    # Tier 1 (≥12 s): disable beam search and repair loops.
    timeout_tier1_seconds: float = 12.0
    # Tier 2 (≥35 s): additionally skip optional field generation.
    timeout_tier2_seconds: float = 35.0

    # ── Reproducibility ──────────────────────────────────────────────────────
    # Passed as extra_body={"seed": seed} in every vLLM request so that the
    # same input always produces the same output on Public Test Day.
    seed: int = 42

    # ── LoRA adapter paths (declared at vLLM startup) ─────────────────────────
    adapter_physics_solver: str = "adapters/physics-solver"
    adapter_verifier: str = "adapters/verifier"
    adapter_logic_reasoner: str = "adapters/logic-reasoner"
    adapter_response_assembler: str = "adapters/response-assembler"

    # ────────────────────────────────────────────────────────────────────────
    # Factory helpers
    # ────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SolverConfig":
        """Build a SolverConfig from a plain dict (e.g. parsed from JSON).

        Unknown keys are silently ignored so that partial override dicts work.
        The nested ``confidence_threshold`` sub-object is handled automatically.
        """
        kwargs: Dict[str, Any] = {}
        ct_data: Optional[Dict[str, Any]] = None

        for key, value in d.items():
            if key == "confidence_threshold" and isinstance(value, dict):
                ct_data = value
            elif hasattr(cls, key):
                kwargs[key] = value

        cfg = cls(**kwargs)
        if ct_data:
            cfg.confidence_threshold = ConfidenceThresholds(**{
                k: v for k, v in ct_data.items()
                if hasattr(ConfidenceThresholds, k)
            })
        return cfg

    @classmethod
    def from_json(cls, path: str | Path) -> "SolverConfig":
        """Load a SolverConfig from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # ConfidenceThresholds is already flattened by asdict; restore nesting.
        return d

    # ────────────────────────────────────────────────────────────────────────
    # Latency tier helpers (used by pipeline orchestrators)
    # ────────────────────────────────────────────────────────────────────────

    def tier(self, elapsed_seconds: float) -> int:
        """Return the active timeout tier for an elapsed time.

        Returns
        -------
        0  Full pipeline — all features active.
        1  Tier 1 — disable beam search and repair loops.
        2  Tier 2 — also skip optional field generation.
        3  Hard stop — emit best available answer immediately.
        """
        if elapsed_seconds >= self.latency_budget_seconds:
            return 3
        if elapsed_seconds >= self.timeout_tier2_seconds:
            return 2
        if elapsed_seconds >= self.timeout_tier1_seconds:
            return 1
        return 0

    def optional_fields_enabled(self, elapsed_seconds: float) -> bool:
        """Return True if optional fields (fol, cot, premises) should be generated."""
        return self.tier(elapsed_seconds) < 2
