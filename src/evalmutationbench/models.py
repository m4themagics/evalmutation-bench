from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Domain(StrEnum):
    CUSTOMER_SUPPORT = "customer_support"
    PROCUREMENT = "procurement"
    TRAVEL_OPERATIONS = "travel_operations"


class Split(StrEnum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    CHALLENGE = "challenge"


class FailureFamily(StrEnum):
    NONE = "none"
    RETRIEVAL = "retrieval"
    ANSWER = "answer"
    TOOLS = "tools"
    STATE_POLICY = "state_policy"


class MutationType(StrEnum):
    SEMANTIC_PARAPHRASE = "semantic_paraphrase"
    EVIDENCE_REORDER = "evidence_reorder"
    CONCISE_VALID_ANSWER = "concise_valid_answer"
    STALE_EVIDENCE = "stale_evidence"
    MISSING_EVIDENCE = "missing_evidence"
    WRONG_EVIDENCE = "wrong_evidence"
    TRUNCATED_EVIDENCE = "truncated_evidence"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    OMITTED_CONSTRAINT = "omitted_constraint"
    WRONG_CITATION = "wrong_citation"
    WRONG_TOOL = "wrong_tool"
    WRONG_ARGUMENTS = "wrong_arguments"
    SILENT_CORRUPTION = "silent_corruption"
    IGNORED_TIMEOUT = "ignored_timeout"
    TOOL_LOOP = "tool_loop"
    MISSING_APPROVAL = "missing_approval"
    PERMISSION_VIOLATION = "permission_violation"
    CROSS_SESSION_STATE = "cross_session_state"
    PARTIAL_COMMIT = "partial_commit"


class Evidence(StrictModel):
    evidence_id: str
    title: str
    content: str
    version: int = Field(ge=1)
    is_current: bool
    source_uri: str


class Message(StrictModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ToolCall(StrictModel):
    call_id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(StrictModel):
    call_id: str
    status: Literal["ok", "timeout", "error"]
    output: dict[str, Any]


class Trace(StrictModel):
    trace_id: str
    messages: list[Message]
    retrieval: list[Evidence]
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    citations: list[str]
    approvals: list[str]
    final_answer: str
    state_before: dict[str, Any]
    state_after: dict[str, Any]

    @model_validator(mode="after")
    def validate_trace_links(self) -> Trace:
        evidence_ids = {item.evidence_id for item in self.retrieval}
        unknown_citations = set(self.citations) - evidence_ids
        if unknown_citations:
            raise ValueError(f"citations not present in retrieval: {sorted(unknown_citations)}")
        call_ids = [call.call_id for call in self.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("tool call IDs must be unique")
        unknown_results = {result.call_id for result in self.tool_results} - set(call_ids)
        if unknown_results:
            raise ValueError(f"tool results without calls: {sorted(unknown_results)}")
        return self


class Scenario(StrictModel):
    scenario_id: str
    domain: Domain
    instruction: str
    policy: str
    allowed_tools: list[str]
    required_evidence_ids: list[str]
    required_approvals: list[str]
    initial_state: dict[str, Any]
    expected_state: dict[str, Any]


class GoldLabel(StrictModel):
    regression: bool
    failure_family: FailureFamily
    mutation_type: MutationType
    severity: int = Field(ge=0, le=3)
    label_source: Literal["synthetic_oracle", "human_adjudicated"]
    evidence_paths: list[str]

    @model_validator(mode="after")
    def validate_label(self) -> GoldLabel:
        if self.regression:
            if self.failure_family is FailureFamily.NONE or self.severity == 0:
                raise ValueError("regressions require a failure family and positive severity")
        elif self.failure_family is not FailureFamily.NONE or self.severity != 0:
            raise ValueError("controls require failure_family=none and severity=0")
        return self


class TracePair(StrictModel):
    benchmark_version: str
    pair_id: str
    scenario_group: str
    split: Split
    scenario: Scenario
    control: Trace
    candidate: Trace
    gold: GoldLabel | None = None

    @model_validator(mode="after")
    def protect_challenge_labels(self) -> TracePair:
        if self.split is Split.CHALLENGE and self.gold is not None:
            raise ValueError("challenge records must not contain gold labels")
        allowed = set(self.scenario.allowed_tools)
        for name, trace in (("control", self.control), ("candidate", self.candidate)):
            unknown = {call.name for call in trace.tool_calls} - allowed
            if unknown:
                raise ValueError(f"{name} uses unknown tools: {sorted(unknown)}")
            if trace.state_before != self.scenario.initial_state:
                raise ValueError(f"{name} state_before differs from scenario initial_state")
        return self


class EvaluatorPrediction(StrictModel):
    pair_id: str
    regression_probability: float = Field(ge=0.0, le=1.0)
    predicted_regression: bool
    predicted_failure_family: FailureFamily
    predicted_severity: int = Field(ge=0, le=3)
    explanation: str
    evidence_paths: list[str]
    latency_ms: float = Field(ge=0.0)
    cost_usd: float | None = Field(default=None, ge=0.0)
    error: str | None = None


class RunManifest(StrictModel):
    benchmark_version: str
    run_id: str
    adapter: str
    adapter_version: str
    model: str | None
    dataset_sha256: str
    dataset_split: Split
    seed: int
    created_at: str
    git_commit: str | None
    config: dict[str, Any]


class HumanJudgment(StrictModel):
    pair_id: str
    reviewer_id: str
    regression: bool
    failure_family: FailureFamily
    severity: int = Field(ge=0, le=3)
    confidence: int = Field(ge=1, le=5)
    evidence_paths: list[str]
    notes: str = ""
