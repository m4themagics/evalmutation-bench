from __future__ import annotations

from evalmutationbench.dataset import build_scenario
from evalmutationbench.reference import execute_reference
from evalmutationbench.retrieval import BM25Retriever


def test_bm25_is_deterministic() -> None:
    _, trace = build_scenario(3)
    retriever = BM25Retriever(trace.retrieval)
    first = retriever.search("current manager approval policy")
    second = retriever.search("current manager approval policy")
    assert first == second
    assert first[0].evidence.is_current


def test_state_machine_reaches_expected_state() -> None:
    scenario, trace = build_scenario(4)
    result = execute_reference(scenario, trace.retrieval)
    assert result.state_after == scenario.expected_state
    assert result.citations == scenario.required_evidence_ids
    assert result.approvals == scenario.required_approvals
