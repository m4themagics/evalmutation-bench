from __future__ import annotations

import json
import os
import time
from typing import Any, Literal

from ..models import EvaluatorPrediction, FailureFamily, TracePair
from .base import EvaluatorAdapter

SYSTEM_PROMPT = """You are evaluating whether a candidate RAG/tool-agent trace regressed.
Return JSON only with keys regression_probability, predicted_regression,
predicted_failure_family, predicted_severity, explanation, evidence_paths.
failure_family must be one of none, retrieval, answer, tools, state_policy.
Judge correctness, not style or verbosity. Treat semantically equivalent traces as controls."""


class LLMJudgeAdapter(EvaluatorAdapter):
    version = "0.1.0"

    def __init__(
        self,
        model: str | None = None,
        mode: Literal["pointwise", "pairwise"] = "pairwise",
        api_base: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model or os.getenv("EVALMUT_MODEL")
        if not self.model:
            raise ValueError("set --model or EVALMUT_MODEL for an LLM judge")
        self.mode = mode
        self.api_base = api_base or os.getenv("EVALMUT_API_BASE")
        self.temperature = temperature
        self.name = mode

    def _payload(self, pair: TracePair) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "scenario": pair.scenario.model_dump(mode="json"),
            "candidate": pair.candidate.model_dump(mode="json"),
        }
        if self.mode == "pairwise":
            payload["control"] = pair.control.model_dump(mode="json")
        return payload

    def evaluate(self, pair: TracePair) -> EvaluatorPrediction:
        try:
            from litellm import completion, completion_cost  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("install the llm extra: uv sync --extra llm") from exc

        started = time.perf_counter()
        response = completion(
            model=self.model,
            api_base=self.api_base,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(self._payload(pair), ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        elapsed = (time.perf_counter() - started) * 1000
        content = response.choices[0].message.content
        if not content:
            raise ValueError("judge returned an empty response")
        parsed = json.loads(content)
        try:
            cost = float(completion_cost(completion_response=response))
        except Exception:
            cost = None
        return EvaluatorPrediction(
            pair_id=pair.pair_id,
            regression_probability=parsed["regression_probability"],
            predicted_regression=parsed["predicted_regression"],
            predicted_failure_family=FailureFamily(parsed["predicted_failure_family"]),
            predicted_severity=parsed["predicted_severity"],
            explanation=parsed["explanation"],
            evidence_paths=parsed.get("evidence_paths", []),
            latency_ms=elapsed,
            cost_usd=cost,
        )

    async def aevaluate(self, pair: TracePair) -> EvaluatorPrediction:
        from litellm import acompletion, completion_cost

        started = time.perf_counter()
        response = await acompletion(
            model=self.model,
            api_base=self.api_base,
            temperature=self.temperature,
            num_retries=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(self._payload(pair), ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("judge returned an empty response")
        parsed = json.loads(content)
        try:
            cost = float(completion_cost(completion_response=response))
        except Exception:
            cost = None
        return EvaluatorPrediction(
            pair_id=pair.pair_id,
            regression_probability=parsed["regression_probability"],
            predicted_regression=parsed["predicted_regression"],
            predicted_failure_family=FailureFamily(parsed["predicted_failure_family"]),
            predicted_severity=parsed["predicted_severity"],
            explanation=parsed["explanation"],
            evidence_paths=parsed.get("evidence_paths", []),
            latency_ms=(time.perf_counter() - started) * 1000,
            cost_usd=cost,
        )
