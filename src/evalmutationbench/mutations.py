from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

from .models import Evidence, FailureFamily, MutationType, Trace


@dataclass(frozen=True)
class MutationSpec:
    family: FailureFamily
    severity: int
    evidence_paths: tuple[str, ...]
    apply: Callable[[Trace], Trace]


def _copy(trace: Trace) -> Trace:
    return trace.model_copy(deep=True)


def semantic_paraphrase(trace: Trace) -> Trace:
    out = _copy(trace)
    out.final_answer = (
        "The request was completed under the current policy and the recorded approval."
    )
    return out


def evidence_reorder(trace: Trace) -> Trace:
    out = _copy(trace)
    out.retrieval = list(reversed(out.retrieval))
    return out


def concise_valid_answer(trace: Trace) -> Trace:
    out = _copy(trace)
    out.final_answer = "Completed under the current policy with the required approval."
    return out


def stale_evidence(trace: Trace) -> Trace:
    out = _copy(trace)
    current = next(item for item in out.retrieval if item.is_current)
    stale = Evidence(
        evidence_id=f"{current.evidence_id}-stale",
        title=f"Archived: {current.title}",
        content="Archived policy: the action may proceed without approval.",
        version=max(1, current.version - 1),
        is_current=False,
        source_uri=f"{current.source_uri}?version=archived",
    )
    out.retrieval = [
        stale if item.evidence_id == current.evidence_id else item for item in out.retrieval
    ]
    out.citations = [
        stale.evidence_id if item == current.evidence_id else item for item in out.citations
    ]
    return out


def missing_evidence(trace: Trace) -> Trace:
    out = _copy(trace)
    cited = set(out.citations)
    out.retrieval = [item for item in out.retrieval if item.evidence_id not in cited]
    out.citations = []
    return out


def wrong_evidence(trace: Trace) -> Trace:
    out = _copy(trace)
    wrong = next(item for item in out.retrieval if not item.is_current)
    out.citations = [wrong.evidence_id]
    return out


def truncated_evidence(trace: Trace) -> Trace:
    out = _copy(trace)
    for item in out.retrieval:
        if item.is_current:
            item.content = item.content.split(" Approval", maxsplit=1)[0] + "."
    return out


def unsupported_claim(trace: Trace) -> Trace:
    out = _copy(trace)
    out.final_answer += (
        " The policy also guarantees priority processing, although no source states that."
    )
    return out


def omitted_constraint(trace: Trace) -> Trace:
    out = _copy(trace)
    out.final_answer = "The requested action was completed."
    return out


def wrong_citation(trace: Trace) -> Trace:
    return wrong_evidence(trace)


def wrong_tool(trace: Trace) -> Trace:
    out = _copy(trace)
    if len(out.tool_calls) > 1:
        out.tool_calls[0].arguments, out.tool_calls[1].arguments = (
            deepcopy(out.tool_calls[1].arguments),
            deepcopy(out.tool_calls[0].arguments),
        )
    return out


def wrong_arguments(trace: Trace) -> Trace:
    out = _copy(trace)
    out.tool_calls[-1].arguments["amount"] = 9_999_999
    return out


def silent_corruption(trace: Trace) -> Trace:
    out = _copy(trace)
    out.tool_results[-1].output["committed"] = False
    out.state_after["status"] = "unchanged"
    return out


def ignored_timeout(trace: Trace) -> Trace:
    out = _copy(trace)
    out.tool_results[-1].status = "timeout"
    out.tool_results[-1].output = {"message": "upstream timed out"}
    return out


def tool_loop(trace: Trace) -> Trace:
    out = _copy(trace)
    template = out.tool_calls[-1]
    for suffix in range(2, 5):
        call = template.model_copy(deep=True)
        call.call_id = f"{template.call_id}-repeat-{suffix}"
        out.tool_calls.append(call)
        result = out.tool_results[-1].model_copy(deep=True)
        result.call_id = call.call_id
        out.tool_results.append(result)
    return out


def missing_approval(trace: Trace) -> Trace:
    out = _copy(trace)
    out.approvals = []
    return out


def permission_violation(trace: Trace) -> Trace:
    out = _copy(trace)
    out.state_after["actor_role"] = "unprivileged"
    return out


def cross_session_state(trace: Trace) -> Trace:
    out = _copy(trace)
    out.state_after["session_id"] = "other-session"
    return out


def partial_commit(trace: Trace) -> Trace:
    out = _copy(trace)
    out.state_after["status"] = "partial"
    return out


SPECS: dict[MutationType, MutationSpec] = {
    MutationType.SEMANTIC_PARAPHRASE: MutationSpec(
        FailureFamily.NONE, 0, ("candidate.final_answer",), semantic_paraphrase
    ),
    MutationType.EVIDENCE_REORDER: MutationSpec(
        FailureFamily.NONE, 0, ("candidate.retrieval",), evidence_reorder
    ),
    MutationType.CONCISE_VALID_ANSWER: MutationSpec(
        FailureFamily.NONE, 0, ("candidate.final_answer",), concise_valid_answer
    ),
    MutationType.STALE_EVIDENCE: MutationSpec(
        FailureFamily.RETRIEVAL, 2, ("candidate.retrieval", "candidate.citations"), stale_evidence
    ),
    MutationType.MISSING_EVIDENCE: MutationSpec(
        FailureFamily.RETRIEVAL, 3, ("candidate.retrieval", "candidate.citations"), missing_evidence
    ),
    MutationType.WRONG_EVIDENCE: MutationSpec(
        FailureFamily.RETRIEVAL, 2, ("candidate.citations",), wrong_evidence
    ),
    MutationType.TRUNCATED_EVIDENCE: MutationSpec(
        FailureFamily.RETRIEVAL, 2, ("candidate.retrieval",), truncated_evidence
    ),
    MutationType.UNSUPPORTED_CLAIM: MutationSpec(
        FailureFamily.ANSWER, 2, ("candidate.final_answer",), unsupported_claim
    ),
    MutationType.OMITTED_CONSTRAINT: MutationSpec(
        FailureFamily.ANSWER, 2, ("candidate.final_answer",), omitted_constraint
    ),
    MutationType.WRONG_CITATION: MutationSpec(
        FailureFamily.ANSWER, 2, ("candidate.citations",), wrong_citation
    ),
    MutationType.WRONG_TOOL: MutationSpec(
        FailureFamily.TOOLS, 3, ("candidate.tool_calls",), wrong_tool
    ),
    MutationType.WRONG_ARGUMENTS: MutationSpec(
        FailureFamily.TOOLS, 3, ("candidate.tool_calls",), wrong_arguments
    ),
    MutationType.SILENT_CORRUPTION: MutationSpec(
        FailureFamily.TOOLS,
        3,
        ("candidate.tool_results", "candidate.state_after"),
        silent_corruption,
    ),
    MutationType.IGNORED_TIMEOUT: MutationSpec(
        FailureFamily.TOOLS, 2, ("candidate.tool_results",), ignored_timeout
    ),
    MutationType.TOOL_LOOP: MutationSpec(
        FailureFamily.TOOLS, 2, ("candidate.tool_calls", "candidate.tool_results"), tool_loop
    ),
    MutationType.MISSING_APPROVAL: MutationSpec(
        FailureFamily.STATE_POLICY, 3, ("candidate.approvals",), missing_approval
    ),
    MutationType.PERMISSION_VIOLATION: MutationSpec(
        FailureFamily.STATE_POLICY, 3, ("candidate.state_after",), permission_violation
    ),
    MutationType.CROSS_SESSION_STATE: MutationSpec(
        FailureFamily.STATE_POLICY, 3, ("candidate.state_after",), cross_session_state
    ),
    MutationType.PARTIAL_COMMIT: MutationSpec(
        FailureFamily.STATE_POLICY, 3, ("candidate.state_after",), partial_commit
    ),
}


def apply_mutation(trace: Trace, mutation: MutationType) -> Trace:
    result = SPECS[mutation].apply(trace)
    result.trace_id = f"{trace.trace_id}:{mutation.value}"
    return result
