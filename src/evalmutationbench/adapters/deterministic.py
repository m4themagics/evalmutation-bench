from __future__ import annotations

import time
from collections import Counter

from ..models import EvaluatorPrediction, FailureFamily, TracePair
from .base import EvaluatorAdapter


class DeterministicAdapter(EvaluatorAdapter):
    name = "deterministic"
    version = "0.1.0"

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def evaluate(self, pair: TracePair) -> EvaluatorPrediction:
        started = time.perf_counter()
        candidate = pair.candidate
        scenario = pair.scenario
        findings: list[tuple[FailureFamily, str, int]] = []

        current_ids = {item.evidence_id for item in candidate.retrieval if item.is_current}
        missing_required = set(scenario.required_evidence_ids) - current_ids
        if missing_required:
            findings.append((FailureFamily.RETRIEVAL, "candidate.retrieval", 3))
        if not set(scenario.required_evidence_ids).issubset(candidate.citations):
            findings.append((FailureFamily.RETRIEVAL, "candidate.citations", 2))

        answer = candidate.final_answer.lower()
        if "no source states that" in answer or "priority processing" in answer:
            findings.append((FailureFamily.ANSWER, "candidate.final_answer", 2))
        if scenario.required_approvals and "approval" not in answer:
            findings.append((FailureFamily.ANSWER, "candidate.final_answer", 2))

        control_calls = [(call.name, call.arguments) for call in pair.control.tool_calls]
        candidate_calls = [(call.name, call.arguments) for call in candidate.tool_calls]
        if candidate_calls != control_calls:
            findings.append((FailureFamily.TOOLS, "candidate.tool_calls", 3))
        call_counts = Counter(
            (call.name, repr(sorted(call.arguments.items()))) for call in candidate.tool_calls
        )
        if any(count >= 3 for count in call_counts.values()):
            findings.append((FailureFamily.TOOLS, "candidate.tool_calls", 2))
        if any(result.status != "ok" for result in candidate.tool_results):
            findings.append((FailureFamily.TOOLS, "candidate.tool_results", 2))
        if any(result.output.get("committed") is False for result in candidate.tool_results):
            findings.append((FailureFamily.TOOLS, "candidate.tool_results", 3))

        if not set(scenario.required_approvals).issubset(candidate.approvals):
            findings.append((FailureFamily.STATE_POLICY, "candidate.approvals", 3))
        if candidate.state_after != scenario.expected_state:
            findings.append((FailureFamily.STATE_POLICY, "candidate.state_after", 3))

        if findings:
            family, _, severity = max(findings, key=lambda item: item[2])
            probability = 0.95
        else:
            family, severity, probability = FailureFamily.NONE, 0, 0.05
        elapsed = (time.perf_counter() - started) * 1000
        return EvaluatorPrediction(
            pair_id=pair.pair_id,
            regression_probability=probability,
            predicted_regression=probability >= self.threshold,
            predicted_failure_family=family,
            predicted_severity=severity,
            explanation="; ".join(f"{item[0].value}:{item[1]}" for item in findings)
            or "No deterministic invariant was violated.",
            evidence_paths=sorted({item[1] for item in findings}),
            latency_ms=elapsed,
            cost_usd=0.0,
        )
