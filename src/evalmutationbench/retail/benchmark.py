from __future__ import annotations

from ..dataset import BENCHMARK_VERSION
from ..models import Domain, GoldLabel, MutationType, Scenario, Split, Trace, TracePair
from ..mutations import SPECS, apply_mutation


def trace_pairs(trace: Trace) -> list[TracePair]:
    """Convert a completed real reference execution into five oracle-labeled pairs."""
    if trace.state_after.get("phase") != "ordered" or not trace.approvals:
        raise ValueError("benchmark export requires a completed, approved purchase")
    scenario = Scenario(
        scenario_id=f"retail:{trace.trace_id}",
        domain=Domain.PROCUREMENT,
        instruction=trace.messages[0].content,
        policy="Use current supplier terms; human approval is mandatory before purchase.",
        allowed_tools=[call.name for call in trace.tool_calls],
        required_evidence_ids=trace.citations,
        required_approvals=trace.approvals,
        initial_state=trace.state_before,
        expected_state=trace.state_after,
    )
    result = []
    for mutation in (
        MutationType.EVIDENCE_REORDER,
        MutationType.STALE_EVIDENCE,
        MutationType.UNSUPPORTED_CLAIM,
        MutationType.WRONG_ARGUMENTS,
        MutationType.MISSING_APPROVAL,
    ):
        spec = SPECS[mutation]
        result.append(
            TracePair(
                benchmark_version=BENCHMARK_VERSION,
                pair_id=f"retail:{trace.trace_id}:{mutation.value}",
                scenario_group=scenario.scenario_id,
                split=Split.DEVELOPMENT,
                scenario=scenario,
                control=trace,
                candidate=apply_mutation(trace, mutation),
                gold=GoldLabel(
                    regression=spec.family.value != "none",
                    failure_family=spec.family,
                    mutation_type=mutation,
                    severity=spec.severity,
                    label_source="synthetic_oracle",
                    evidence_paths=list(spec.evidence_paths),
                ),
            )
        )
    return result
