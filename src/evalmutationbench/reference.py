from __future__ import annotations

from typing import Any, TypedDict

from .models import Evidence, Message, Scenario, ToolCall, ToolResult, Trace
from .retrieval import BM25Retriever


def execute_reference(scenario: Scenario, evidence: list[Evidence]) -> Trace:
    ranked = BM25Retriever(evidence).search(scenario.instruction, limit=len(evidence))
    current = [item.evidence for item in ranked if item.evidence.is_current]
    selected = current + [item.evidence for item in ranked if not item.evidence.is_current]
    amount = scenario.initial_state.get("amount")
    entity_id = scenario.initial_state.get("entity_id")
    calls = [
        ToolCall(
            call_id="lookup",
            name=scenario.allowed_tools[0],
            arguments={"entity_id": entity_id},
        ),
        ToolCall(
            call_id="approval",
            name=scenario.allowed_tools[1],
            arguments={"amount": amount},
        ),
        ToolCall(
            call_id="action",
            name=scenario.allowed_tools[2],
            arguments={"entity_id": entity_id, "amount": amount},
        ),
    ]
    return Trace(
        trace_id=f"{scenario.scenario_id}:reference",
        messages=[
            Message(role="user", content=scenario.instruction),
            Message(role="assistant", content="I will verify policy, request approval, and act."),
        ],
        retrieval=selected,
        tool_calls=calls,
        tool_results=[
            ToolResult(call_id="lookup", status="ok", output={"found": True}),
            ToolResult(
                call_id="approval",
                status="ok",
                output={"approval_id": scenario.required_approvals[0]},
            ),
            ToolResult(call_id="action", status="ok", output={"committed": True}),
        ],
        citations=scenario.required_evidence_ids,
        approvals=scenario.required_approvals,
        final_answer="The request was completed after approval under the current policy.",
        state_before=scenario.initial_state,
        state_after=scenario.expected_state,
    )


class AgentState(TypedDict):
    scenario: Scenario
    evidence: list[Evidence]
    trace: Trace | None
    events: list[str]


def build_langgraph_reference() -> Any:
    """Build the optional checkpoint-friendly deterministic reference graph."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("install the reference extra: uv sync --extra reference") from exc

    def retrieve(state: AgentState) -> dict[str, list[str]]:
        return {"events": [*state["events"], "retrieved"]}

    def approve(state: AgentState) -> dict[str, list[str]]:
        return {"events": [*state["events"], "approved"]}

    def act(state: AgentState) -> dict[str, Any]:
        return {
            "events": [*state["events"], "acted"],
            "trace": execute_reference(state["scenario"], state["evidence"]),
        }

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("approve", approve)
    graph.add_node("act", act)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "approve")
    graph.add_edge("approve", "act")
    graph.add_edge("act", END)
    return graph.compile()
