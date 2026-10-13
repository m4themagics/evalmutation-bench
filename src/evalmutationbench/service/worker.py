from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import signal
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..adapters import load_adapter
from ..dataset import BENCHMARK_VERSION
from ..execution import evaluate_pairs
from ..metrics import score_predictions
from ..models import RunManifest, TracePair
from .settings import Settings
from .store import RunStore

logger = logging.getLogger(__name__)


def _snapshot(path: Path, checksum: str) -> list[TracePair]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError("dataset changed after submission")
    return [TracePair.model_validate_json(line) for line in raw.splitlines() if line.strip()]


async def process_one(store: RunStore, settings: Settings) -> bool:
    await store.expire()
    row = await store.claim(settings.run_timeout_seconds + 30)
    if row is None:
        return False
    run_id, token, request = row["run_id"], row["claim_token"], row["request"]
    try:
        async with asyncio.timeout(settings.run_timeout_seconds):
            if request.get("kind") in ("retail_start", "retail_approval"):
                if not settings.retail_enabled:
                    raise ValueError("retail worker is disabled")
                from ..retail.runtime import process_job

                retail_result = await process_job(store, row)
                await store.finish(run_id, token, result=retail_result)
                return True
            path = settings.data_dir / f"{request['split']}.jsonl"
            pairs = await asyncio.to_thread(_snapshot, path, request["dataset_sha256"])
            if not pairs or any(pair.split.value != request["split"] for pair in pairs):
                raise ValueError("invalid dataset split")
            config = {"model": request["model"]} if request["model"] else {}
            evaluator = load_adapter(request["adapter"], config)
            predictions = await evaluate_pairs(
                pairs,
                evaluator,
                concurrency=settings.concurrency,
                timeout_seconds=settings.pair_timeout_seconds,
            )
            errors = sum(prediction.error is not None for prediction in predictions)
            metrics: dict[str, Any] | None = None
            if errors == 0 and all(pair.gold is not None for pair in pairs):
                metrics = await asyncio.to_thread(score_predictions, pairs, predictions)
            result: dict[str, Any] = {
                "predictions": [prediction.model_dump(mode="json") for prediction in predictions],
                "metrics": metrics,
                "errors": errors,
                "execution": {
                    "concurrency": settings.concurrency,
                    "pair_timeout_seconds": settings.pair_timeout_seconds,
                    "adapter_version": evaluator.version,
                },
            }
            result["manifest"] = RunManifest(
                benchmark_version=BENCHMARK_VERSION,
                run_id=run_id,
                adapter=evaluator.name,
                adapter_version=evaluator.version,
                model=request["model"],
                dataset_sha256=request["dataset_sha256"],
                dataset_split=pairs[0].split,
                seed=42,
                created_at=datetime.fromtimestamp(row["created_at"], UTC).isoformat(),
                git_commit=os.getenv("EVALMUT_GIT_COMMIT"),
                config=result["execution"],
            ).model_dump(mode="json")
            await store.finish(
                run_id, token, result=result, error="PairEvaluationFailed" if errors else None
            )
    except asyncio.CancelledError:
        await asyncio.shield(store.finish(run_id, token, error="WorkerCancelled"))
        raise
    except Exception as exc:
        # Raw exception strings may contain provider credentials or trace content.
        if request.get("kind") in ("retail_start", "retail_approval"):
            from ..retail.store import RetailStore

            await RetailStore(store).fail_session(request["session_id"])
        await store.finish(run_id, token, error=type(exc).__name__)
    logger.info("processed run_id=%s", run_id)
    return True


async def serve(settings: Settings) -> None:
    store = RunStore(settings.database_url.get_secret_value())
    try:
        while True:
            if not await process_one(store, settings):
                await asyncio.sleep(1)
    finally:
        await store.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    async def run() -> None:
        task = asyncio.create_task(serve(Settings.from_env()))
        loop = asyncio.get_running_loop()
        # Windows falls back to lease expiry on external termination.
        with suppress(NotImplementedError):
            loop.add_signal_handler(signal.SIGTERM, task.cancel)
        with suppress(asyncio.CancelledError):
            await task

    asyncio.run(run())
