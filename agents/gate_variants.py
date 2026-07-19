"""Gate variants for RouterBench A/B testing (Day 7).

Three variants exercise the cost-quality curve at the gate stage:

    haiku-weighted    — current production gate. Haiku 4.5 + 4-component
                         weighted rubric output. Decomposed scoring.
    haiku-verdict     — Haiku 4.5 with verdict-only output (score + verdict
                         + issues, no component breakdown). Faster, cheaper,
                         less defensible.
    sonnet-verdict    — Sonnet 4.6 with verdict-only output. More expensive,
                         arguably smarter scoring than Haiku, but no
                         decomposition.

The headline experiment is whether the latency/cost premium of the
weighted rubric is justified by score-correlation differences vs the
verdict-only variants.

This module deliberately reuses the QualityGate base class but overrides
the system prompt + model on a per-variant basis. The output schema is
unified (`{score, verdict, routing, issues, llm_latency_ms}`) so the
RouterBench analysis can compare apples to apples.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from agents.base_agent import BaseAgent
from agents.models import HAIKU, SONNET
from agents.quality_gate import (
    GATE_SYSTEM_VERDICT as VERDICT_ONLY_SYSTEM,
    apply_verdict,
    compact_gate_payload,
)


class _BaseGateVariant(BaseAgent):
    """Shared body — subclasses set variant_name + variant_model + variant_system."""
    stage = "gate"
    variant_name: str
    variant_system: str
    max_tokens: int = 1024

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # agent_name encodes the variant so traces + the recent-runs table
        # can disambiguate without an extra column.
        self.agent_name = f"QualityGate.{self.variant_name}"

    def execute(self, run_id: UUID, inputs: dict[str, Any]) -> dict[str, Any]:
        plan = inputs.get("plan")
        code = inputs.get("code")
        tests = inputs.get("tests")
        if not (plan and code and tests):
            raise ValueError(f"{self.variant_name}: missing one of plan/code/tests")

        call = self._generate(
            system=self.variant_system,
            user=json.dumps(compact_gate_payload(plan, code, tests)),
            max_tokens=self.max_tokens,
            input_data={"variant": self.variant_name},
        )

        out = call.parsed or {}
        out["llm_latency_ms"] = call.latency_ms
        out["variant"] = self.variant_name
        out["input_tokens"] = call.usage.get("input", 0)
        out["output_tokens"] = call.usage.get("output", 0)
        out["model"] = call.model
        return apply_verdict(out, self.variant_name)


class GateHaikuWeighted(_BaseGateVariant):
    """Production default — Haiku 4.5 with weighted-rubric component output.
    Lifted directly from agents.quality_gate.QualityGate."""
    variant_name = "haiku-weighted"
    model = HAIKU
    max_tokens = 1024
    # Use the weighted prompt (Day 7 RouterBench A/B baseline). Production
    # default switched to verdict-only after this experiment.
    from agents.quality_gate import GATE_SYSTEM_WEIGHTED as variant_system  # type: ignore[misc]


class GateHaikuVerdict(_BaseGateVariant):
    """Cheaper alternative — same Haiku, no component breakdown."""
    variant_name = "haiku-verdict"
    model = HAIKU
    variant_system = VERDICT_ONLY_SYSTEM
    max_tokens = 512


class GateSonnetVerdict(_BaseGateVariant):
    """Smarter-but-pricier alternative — Sonnet, verdict-only."""
    variant_name = "sonnet-verdict"
    model = SONNET
    variant_system = VERDICT_ONLY_SYSTEM
    max_tokens = 512


VARIANTS: dict[str, type[_BaseGateVariant]] = {
    "haiku-weighted": GateHaikuWeighted,
    "haiku-verdict": GateHaikuVerdict,
    "sonnet-verdict": GateSonnetVerdict,
}
