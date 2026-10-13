from __future__ import annotations

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from evalmutationbench.dataset import write_dataset
from evalmutationbench.io import dataset_sha256
from evalmutationbench.service.api import create_app
from evalmutationbench.service.settings import Settings
from evalmutationbench.service.store import IdempotencyConflict, RunStore
from evalmutationbench.service.worker import process_one


def settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}",
        api_token="test-token-" + "x" * 32,
        data_dir=tmp_path / "data",
    )


def test_api_auth_idempotency_and_validation(tmp_path: Path) -> None:
    config = settings(tmp_path)
    write_dataset(config.data_dir)

    async def initialize() -> None:
        store = RunStore(config.database_url.get_secret_value())
        await store.initialize()
        await store.close()

    asyncio.run(initialize())
    headers = {
        "Authorization": f"Bearer {config.api_token.get_secret_value()}",
        "Idempotency-Key": "request-1",
    }
    with TestClient(create_app(config)) as client:
        assert client.get("/health/ready").status_code == 200
        assert client.post("/v1/runs", json={}).status_code == 401
        first = client.post("/v1/runs", headers=headers, json={})
        assert first.status_code == 202
        repeated = client.post("/v1/runs", headers=headers, json={})
        assert repeated.json()["run_id"] == first.json()["run_id"]
        conflict = client.post("/v1/runs", headers=headers, json={"split": "validation"})
        assert conflict.status_code == 409
        assert (
            client.post("/v1/runs", headers=headers, json={"adapter": "arbitrary:code"}).status_code
            == 422
        )
        assert (
            client.post("/v1/runs", headers=headers, json={"split": "../../secrets"}).status_code
            == 422
        )
        assert client.get(f"/v1/runs/{first.json()['run_id']}", headers=headers).status_code == 200
        assert client.get(f"/v1/runs/{uuid4()}", headers=headers).status_code == 404
        assert "claim_token" not in first.json()
        assert "idempotency_key" not in first.json()


def test_worker_results_challenge_and_dataset_change(tmp_path: Path) -> None:
    config = settings(tmp_path)
    write_dataset(config.data_dir)

    async def exercise() -> None:
        store = RunStore(config.database_url.get_secret_value())
        await store.initialize()
        try:
            for split in ("validation", "challenge"):
                request = {
                    "split": split,
                    "adapter": "deterministic",
                    "model": None,
                    "dataset_sha256": dataset_sha256(config.data_dir / f"{split}.jsonl"),
                }
                row = await store.create(split, request)
                assert await process_one(store, config)
                completed = await store.get(row["run_id"])
                assert completed is not None and completed["status"] == "completed"
                assert len(completed["result"]["predictions"]) == 60
                assert (completed["result"]["metrics"] is None) == (split == "challenge")
            changed = await store.create("changed", {**request, "dataset_sha256": "bad"})
            assert await process_one(store, config)
            failed = await store.get(changed["run_id"])
            assert failed is not None and failed["status"] == "failed"
            assert failed["error"] == "ValueError"
            assert not await process_one(store, config)
        finally:
            await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("postgres", [False, True])
def test_transactional_claim_and_expired_owner(tmp_path: Path, postgres: bool) -> None:
    url = (
        os.getenv("EVALMUT_TEST_DATABASE_URL")
        if postgres
        else (f"sqlite+aiosqlite:///{tmp_path / 'claims.db'}")
    )
    if not url:
        pytest.skip("set EVALMUT_TEST_DATABASE_URL for real PostgreSQL integration")

    async def exercise() -> None:
        assert url is not None
        store = RunStore(url)
        await store.initialize()
        try:
            key = str(uuid4())
            created = await asyncio.gather(
                *[store.create(key, {"split": "validation"}) for _ in range(4)]
            )
            assert len({row["run_id"] for row in created}) == 1
            with pytest.raises(IdempotencyConflict):
                await store.create(key, {"split": "challenge"})
            claims = await asyncio.gather(*[store.claim(60) for _ in range(4)])
            owners = [claim for claim in claims if claim is not None]
            assert len(owners) == 1
            owner = owners[0]
            assert not await store.finish(owner["run_id"], "wrong", result={})
            assert await store.finish(owner["run_id"], owner["claim_token"], result={"ok": True})
            assert not await store.finish(owner["run_id"], owner["claim_token"], result={})
            stale = await store.create(str(uuid4()), {"split": "validation"})
            expired = await store.claim(-1)
            assert expired is not None
            await store.expire()
            assert not await store.finish(expired["run_id"], expired["claim_token"], result={})
            row = await store.get(stale["run_id"])
            assert row is not None and row["error"] == "WorkerLeaseExpired"
        finally:
            await store.close()

    asyncio.run(exercise())
