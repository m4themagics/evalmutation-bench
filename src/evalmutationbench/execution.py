from __future__ import annotations

import asyncio
import time

from .adapters.base import EvaluatorAdapter
from .models import EvaluatorPrediction, FailureFamily, TracePair


async def evaluate_pairs(
    pairs: list[TracePair],
    evaluator: EvaluatorAdapter,
    *,
    concurrency: int = 4,
    timeout_seconds: float = 60,
) -> list[EvaluatorPrediction]:
    """Bound work, preserve input order, and keep failures out of valid metrics.

    No implicit retries: a timed-out paid call may have incurred a charge.
    Cancellation propagates to native async adapters. Sync compatibility plugins
    must enforce their own deadlines; Python cannot kill a running thread.
    """
    if not 1 <= concurrency <= 32 or not 0 < timeout_seconds <= 600:
        raise ValueError("concurrency must be 1..32 and timeout must be (0, 600]")
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise ValueError("duplicate pair IDs")
    results: list[EvaluatorPrediction | None] = [None] * len(pairs)
    next_index = 0

    async def worker() -> None:
        nonlocal next_index
        while next_index < len(pairs):
            index = next_index
            next_index += 1
            pair = pairs[index]
            started = time.perf_counter()
            try:
                async with asyncio.timeout(timeout_seconds):
                    prediction = await evaluator.aevaluate(pair)
                if prediction.pair_id != pair.pair_id:
                    raise ValueError("adapter returned the wrong pair ID")
                results[index] = prediction
            except Exception as exc:
                results[index] = EvaluatorPrediction(
                    pair_id=pair.pair_id,
                    regression_probability=0.5,
                    predicted_regression=False,
                    predicted_failure_family=FailureFamily.NONE,
                    predicted_severity=0,
                    explanation="Evaluation failed; this prediction is not scoreable.",
                    evidence_paths=[],
                    latency_ms=(time.perf_counter() - started) * 1000,
                    error=type(exc).__name__,
                )

    async with asyncio.TaskGroup() as group:
        for _ in range(min(concurrency, len(pairs))):
            group.create_task(worker())
    return [prediction for prediction in results if prediction is not None]
