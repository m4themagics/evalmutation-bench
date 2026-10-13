from __future__ import annotations

import asyncio

import pytest

from evalmutationbench.adapters import DeterministicAdapter
from evalmutationbench.dataset import build_pairs
from evalmutationbench.execution import evaluate_pairs
from evalmutationbench.metrics import cluster_bootstrap_ci, score_predictions
from evalmutationbench.models import EvaluatorPrediction, TracePair


def test_concurrency_order_and_timeout_cancellation() -> None:
    pairs = build_pairs()[:8]

    class Fake(DeterministicAdapter):
        active = 0
        peak = 0
        cancelled = False

        async def aevaluate(self, pair: TracePair) -> EvaluatorPrediction:
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                if pair.pair_id == pairs[1].pair_id:
                    try:
                        await asyncio.sleep(1)
                    except asyncio.CancelledError:
                        self.cancelled = True
                        raise
                else:
                    await asyncio.sleep(0.001)
                return self.evaluate(pair)
            finally:
                self.active -= 1

    fake = Fake()
    predictions = asyncio.run(evaluate_pairs(pairs, fake, concurrency=2, timeout_seconds=0.05))
    assert [prediction.pair_id for prediction in predictions] == [pair.pair_id for pair in pairs]
    assert fake.peak == 2 and fake.active == 0 and fake.cancelled
    assert predictions[1].error == "TimeoutError"
    assert predictions[1].cost_usd is None
    assert all(prediction.error is None for i, prediction in enumerate(predictions) if i != 1)
    with pytest.raises(ValueError, match="failed predictions"):
        score_predictions(pairs, predictions)
    with pytest.raises(ValueError, match="failed predictions"):
        cluster_bootstrap_ci(pairs, predictions, samples=2)


def test_error_redaction_and_pair_id_validation() -> None:
    pairs = build_pairs()[:2]

    class Broken(DeterministicAdapter):
        async def aevaluate(self, pair: TracePair) -> EvaluatorPrediction:
            if pair.pair_id == pairs[0].pair_id:
                raise RuntimeError("secret-api-token")
            return self.evaluate(pair).model_copy(update={"pair_id": "wrong"})

    predictions = asyncio.run(evaluate_pairs(pairs, Broken()))
    assert [prediction.error for prediction in predictions] == ["RuntimeError", "ValueError"]
    assert "secret-api-token" not in str(predictions)


def test_batch_cancellation_drains_async_tasks() -> None:
    class Slow(DeterministicAdapter):
        active = 0

        async def aevaluate(self, pair: TracePair) -> EvaluatorPrediction:
            self.active += 1
            try:
                await asyncio.sleep(10)
                return self.evaluate(pair)
            finally:
                self.active -= 1

    async def exercise() -> None:
        adapter = Slow()
        task = asyncio.create_task(evaluate_pairs(build_pairs()[:5], adapter, concurrency=2))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert adapter.active == 0

    asyncio.run(exercise())
