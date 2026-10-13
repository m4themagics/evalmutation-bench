from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace

from evalmutationbench.adapters import DeterministicAdapter
from evalmutationbench.adapters.llm import LLMJudgeAdapter
from evalmutationbench.dataset import build_pairs
from evalmutationbench.execution import evaluate_pairs


def test_native_async_judge_payload_and_cost(monkeypatch) -> None:
    pair = build_pairs()[0]
    prediction = DeterministicAdapter().evaluate(pair)
    content = prediction.model_dump(exclude={"pair_id", "latency_ms", "cost_usd", "error"})
    calls = []

    async def acompletion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(content)))]
        )

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(acompletion=acompletion, completion_cost=lambda **kwargs: 0.003),
    )
    evaluator = LLMJudgeAdapter(model="test/model", mode="pairwise")
    outputs = asyncio.run(evaluate_pairs([pair], evaluator))
    assert outputs[0].error is None and outputs[0].cost_usd == 0.003
    assert calls[0]["num_retries"] == 0
    payload = json.loads(calls[0]["messages"][1]["content"])
    assert set(payload) == {"scenario", "candidate", "control"}
    assert "gold" not in payload and "mutation_type" not in payload
    evaluator = LLMJudgeAdapter(model="test/model", mode="pointwise")
    assert "control" not in evaluator._payload(pair)
