"""Centralized model IDs + routing table for CodeOrch agents.

The router maps agent role -> model. This is the single source of truth.
Any agent can override its model attribute, but doing so without
documenting the reason is a code smell — the rationale below is what
makes the routing defensible in interviews.

----------------------------------------------------------------------------
Routing rationale (data from RouterBench Day 7, n=39 gate runs across 13
distinct source runs):

    Stage            Model       Why
    -------------    ---------   ----------------------------------------
    orchestrator     Opus 4.7    Lifecycle decisions span the full run.
                                 Worth the premium for routing accuracy
                                 and the run summary that the API returns.

    planner          Sonnet 4.6  Spec -> structured plan needs reasoning;
                                 Haiku tends to under-decompose. Sonnet
                                 hits 1.9s avg, well within < 8s SLI.

    coder            Sonnet 4.6  Code quality > cost on the headline
                                 deliverable. Average 7s on the
                                 RouterBench corpus is acceptable.

    tester           Sonnet 4.6  Test quality directly affects what the
                                 gate scores against. Same reasoning as
                                 Coder; we don't cheap out on the
                                 verifier of the verifier.

    quality_gate     Haiku 4.5   See note below — the answer changed
                                 between Day 5 and Day 7.

    reviewer         Sonnet 4.6  Final approval needs holistic judgment.
                                 Cheaper variants have not been tested
                                 here yet (Day 7 RouterBench was scoped
                                 to gate only).

    documenter       Haiku 4.5   Adding docstrings is mechanical. Haiku
                                 averages 4-6s per call; the < $0.01
                                 cost-per-call ceiling holds.

----------------------------------------------------------------------------
Gate-stage routing decision — the Day 5/7 story:

    Day 5 shipped the gate with a 4-component weighted rubric output
    ({score, components: {correctness, test_coverage, code_quality,
    completeness}, issues}). Latency averaged 6.7s; the < 5s SLI from
    DESIGN.md was missed and we documented the variance with the
    rationale that decomposition was interview-defensible.

    Day 7 RouterBench A/B tested this against verdict-only on Haiku
    (haiku-verdict) and verdict-only on Sonnet (sonnet-verdict) on the
    same 13 source runs. Result:

        variant            score    latency    cost
        haiku-weighted     0.903    7353 ms    $0.00428
        haiku-verdict      0.915    1881 ms    $0.00189
        sonnet-verdict     0.949    2491 ms    $0.00657
                                              (1 JSON-format error)

    Haiku-verdict is faster (3.9x), cheaper (44%), and slightly higher-
    scoring than haiku-weighted. The Day 5 latency variance has an
    empirical answer: there's no quality benefit from the component
    breakdown that justifies the latency tax. Production default
    switches to haiku-verdict.

    The weighted rubric is kept as `agents.gate_variants.GateHaikuWeighted`
    and used selectively on retry/escalate verdicts where decomposed
    scoring helps debug WHY a low score happened.

    Sonnet-verdict produced the highest scores but at 3.5x cost over
    Haiku-verdict, with a JSON-format failure (verdict prose instead of
    structured output) on 1/13 runs. Not safe for production default
    without strict response-format enforcement (tool-use mode).
"""

from __future__ import annotations

# Model IDs current as of sprint start (2026-05). Update via .env override
# if Anthropic ships newer point releases mid-sprint.
OPUS = "claude-opus-4-7"
SONNET = "claude-sonnet-4-6"
HAIKU = "claude-haiku-4-5-20251001"

ROUTING: dict[str, dict[str, str]] = {
    "orchestrator": {"model": OPUS,   "rationale": "lifecycle decisions; routing accuracy worth premium"},
    "planner":      {"model": SONNET, "rationale": "spec decomposition needs reasoning; Haiku under-decomposes"},
    "coder":        {"model": SONNET, "rationale": "code quality > cost on the headline deliverable"},
    "tester":       {"model": SONNET, "rationale": "verifier of the verifier — quality matters"},
    "quality_gate": {"model": HAIKU,  "rationale": "RouterBench Day 7: verdict-only beats weighted on speed+cost+score"},
    "reviewer":     {"model": SONNET, "rationale": "holistic final judgment; cheaper variants untested"},
    "documenter":   {"model": HAIKU,  "rationale": "mechanical docstring addition; < $0.01 cost ceiling holds"},
}
