from __future__ import annotations

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient
from sqlalchemy import func, select, update

from evalmutationbench.adapters import DeterministicAdapter
from evalmutationbench.metrics import score_predictions
from evalmutationbench.models import Trace
from evalmutationbench.retail.benchmark import trace_pairs
from evalmutationbench.retail.graph import calculate_proposal
from evalmutationbench.retail.models import ApprovalRequest, Proposal, RetailRequest, StockSnapshot
from evalmutationbench.retail.runtime import initialize
from evalmutationbench.retail.store import RetailConflict, RetailStore
from evalmutationbench.retail.tables import inventory, orders
from evalmutationbench.service.api import create_app
from evalmutationbench.service.settings import Settings
from evalmutationbench.service.store import RunStore
from evalmutationbench.service.worker import process_one


def test_retail_api_contract_and_early_approval(tmp_path: Path) -> None:
    config = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'retail.db'}",
        api_token="x" * 40,
        retail_enabled=True,
    )

    async def setup() -> None:
        store = RunStore(config.database_url.get_secret_value())
        await store.initialize()
        await RetailStore(store).seed()
        await store.close()

    asyncio.run(setup())
    headers = {"Authorization": "Bearer " + "x" * 40, "Idempotency-Key": str(uuid4())}
    with TestClient(create_app(config)) as client:
        assert client.post("/v1/retail/sessions", json={"sku": "COFFEE-001"}).status_code == 401
        first = client.post("/v1/retail/sessions", headers=headers, json={"sku": "COFFEE-001"})
        assert first.status_code == 202
        repeated = client.post("/v1/retail/sessions", headers=headers, json={"sku": "COFFEE-001"})
        assert first.json()["run_id"] == repeated.json()["run_id"]
        session_id = first.json()["request"]["session_id"]
        assert (
            client.post("/v1/retail/sessions", headers=headers, json={"sku": "OTHER"}).status_code
            == 422
        )
        assert (
            client.post("/v1/retail/sessions", headers=headers, json={"sku": "TEA-001"}).status_code
            == 409
        )
        assert (
            client.post(
                f"/v1/retail/sessions/{session_id}/approval",
                headers={**headers, "Idempotency-Key": str(uuid4())},
                json={"approved": True, "proposal_sha256": "0" * 64},
            ).status_code
            == 409
        )
        assert client.get(f"/v1/retail/sessions/{uuid4()}", headers=headers).status_code == 404
        assert (
            client.get(f"/v1/retail/sessions/{session_id}", headers=headers).json()["order"] is None
        )


def test_calculation_and_policy_limits() -> None:
    stock = StockSnapshot(
        sku="COFFEE-001", on_hand=20, sold_last_7_days=70, unit_cost_cents=500, revision=1
    )
    docs = [
        {"chunk_id": "supplier", "rules": {"pack_size": 10}},
        {"chunk_id": "policy", "rules": {"max_units": 500}},
    ]
    proposal = calculate_proposal(stock, 14, docs)
    assert proposal.quantity == 120 and proposal.total_cost_cents == 60000
    assert proposal.checksum() == proposal.checksum()
    with pytest.raises(RetailConflict, match="no purchase"):
        calculate_proposal(stock.model_copy(update={"on_hand": 1000}), 14, docs)
    with pytest.raises(RetailConflict, match="exceeds policy"):
        calculate_proposal(stock.model_copy(update={"sold_last_7_days": 1000}), 14, docs)
    with pytest.raises(RetailConflict, match="exactly one"):
        calculate_proposal(stock, 14, docs[:1])


@pytest.mark.parametrize("decision", [True, False])
def test_postgres_resume_isolation_and_order_replay(decision: bool) -> None:
    url = os.getenv("EVALMUT_TEST_DATABASE_URL")
    if not url:
        pytest.skip("requires a real PostgreSQL checkpoint database")
    config = Settings(database_url=url, api_token="x" * 40, retail_enabled=True)

    async def exercise() -> None:
        store = RunStore(url)
        await store.initialize()
        await initialize(store)
        retail = RetailStore(store)
        first = await retail.start(str(uuid4()), RetailRequest(sku="COFFEE-001"))
        sid = first["request"]["session_id"]
        other = await retail.start(str(uuid4()), RetailRequest(sku="TEA-001"))
        other_sid = other["request"]["session_id"]
        assert await process_one(store, config)
        assert await process_one(store, config)
        pending = await retail.get_session(sid)
        assert pending is not None and pending["phase"] == "awaiting_approval"
        assert pending["order"] is None and pending["decision"] is None
        assert (await retail.get_session(other_sid))["proposal"]["sku"] == "TEA-001"

        # Dispose every client/engine, then build a fresh graph with a new saver.
        await store.close()
        store = RunStore(url)
        retail = RetailStore(store)
        try:
            key = str(uuid4())
            approval = ApprovalRequest(
                approved=decision, proposal_sha256=pending["proposal_sha256"]
            )
            command = await retail.approve(key, sid, approval)
            repeated = await retail.approve(key, sid, approval)
            assert command["run_id"] == repeated["run_id"]
            with pytest.raises(RetailConflict):
                await retail.approve(
                    str(uuid4()),
                    sid,
                    ApprovalRequest(
                        approved=not decision, proposal_sha256=pending["proposal_sha256"]
                    ),
                )
            assert await process_one(store, config)
            completed = await retail.get_session(sid)
            assert completed is not None
            assert completed["phase"] == ("ordered" if decision else "rejected")
            trace = Trace.model_validate(completed["trace"])
            assert trace.trace_id == sid
            assert bool(trace.approvals) == decision
            assert (completed["order"] is not None) == decision
            if decision:
                replayed = await retail.create_order(
                    sid, Proposal.model_validate(pending["proposal"])
                )
                assert replayed == completed["order"]["order_id"]
                pairs = trace_pairs(trace)
                predictions = [DeterministicAdapter().evaluate(pair) for pair in pairs]
                assert len(pairs) == 5
                assert score_predictions(pairs, predictions)["balanced_accuracy"] == 1.0
            async with store.engine.connect() as connection:
                count = (
                    await connection.execute(
                        select(func.count()).select_from(orders).where(orders.c.session_id == sid)
                    )
                ).scalar_one()
            assert count == int(decision)
            isolated = await retail.get_session(other_sid)
            assert isolated["phase"] == "awaiting_approval" and isolated["order"] is None
            competing = await asyncio.gather(
                retail.approve(
                    str(uuid4()),
                    other_sid,
                    ApprovalRequest(approved=True, proposal_sha256=isolated["proposal_sha256"]),
                ),
                retail.approve(
                    str(uuid4()),
                    other_sid,
                    ApprovalRequest(approved=False, proposal_sha256=isolated["proposal_sha256"]),
                ),
                return_exceptions=True,
            )
            assert sum(isinstance(result, dict) for result in competing) == 1
            assert sum(isinstance(result, RetailConflict) for result in competing) == 1
            assert await process_one(store, config)
        finally:
            await store.close()

    asyncio.run(exercise())


def test_inventory_change_prevents_commit() -> None:
    url = os.getenv("EVALMUT_TEST_DATABASE_URL")
    if not url:
        pytest.skip("requires PostgreSQL")
    config = Settings(database_url=url, api_token="x" * 40, retail_enabled=True)

    async def exercise() -> None:
        store = RunStore(url)
        await store.initialize()
        await initialize(store)
        retail = RetailStore(store)
        try:
            started = await retail.start(str(uuid4()), RetailRequest(sku="RICE-001"))
            sid = started["request"]["session_id"]
            assert await process_one(store, config)
            pending = await retail.get_session(sid)
            assert pending is not None
            proposal = Proposal.model_validate(pending["proposal"])
            async with store.engine.begin() as connection:
                await connection.execute(
                    update(inventory)
                    .where(inventory.c.sku == "RICE-001")
                    .values(revision=inventory.c.revision + 1)
                )
            approval = await retail.approve(
                str(uuid4()),
                sid,
                ApprovalRequest(approved=True, proposal_sha256=proposal.checksum()),
            )
            assert await process_one(store, config)
            failed = await store.get(approval["run_id"])
            assert failed is not None and failed["status"] == "failed"
            assert failed["error"] == "RetailConflict"
            assert (await retail.get_session(sid))["order"] is None
            assert (await retail.get_session(sid))["phase"] == "failed"
            with pytest.raises(RetailConflict, match="requires approval"):
                other = await retail.start(str(uuid4()), RetailRequest(sku="COFFEE-001"))
                await retail.create_order(other["request"]["session_id"], proposal)
            # Complete the remaining start command so integration tests leave no queued jobs.
            assert await process_one(store, config)
        finally:
            await store.close()

    asyncio.run(exercise())
