from __future__ import annotations

import time
import uuid
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from ..service.store import RunStore, runs
from .models import ApprovalRequest, Proposal, RetailRequest, StockSnapshot, evidence_checksum
from .tables import documents, inventory, orders, sessions


class RetailConflict(ValueError):
    pass


class RetailStore:
    def __init__(self, store: RunStore) -> None:
        self.store = store

    async def seed(self) -> None:
        # Seed once; never overwrite operator data during subsequent initialization.
        insert = pg_insert if self.store.engine.dialect.name == "postgresql" else sqlite_insert
        async with self.store.engine.begin() as connection:
            for sku, stock, sales, cost in (
                ("COFFEE-001", 20, 70, 500),
                ("TEA-001", 40, 42, 300),
                ("RICE-001", 25, 56, 800),
            ):
                await connection.execute(
                    insert(inventory)
                    .values(
                        sku=sku,
                        on_hand=stock,
                        sold_last_7_days=sales,
                        unit_cost_cents=cost,
                        revision=1,
                    )
                    .on_conflict_do_nothing()
                )
                for kind, rules, content in (
                    (
                        "supplier",
                        {"pack_size": 10},
                        f"{sku} purchase quantities must be multiples of 10 units.",
                    ),
                    (
                        "policy",
                        {"max_units": 500, "approval_required": True},
                        f"{sku} replenishment limit is 500 units. Human approval is required.",
                    ),
                ):
                    await connection.execute(
                        insert(documents)
                        .values(
                            chunk_id=f"{sku}-{kind}-v1",
                            sku=sku,
                            title=f"{sku} {kind} terms",
                            content=content,
                            source_uri=f"retail://{kind}/{sku}/v1",
                            version=1,
                            is_current=True,
                            rules=rules,
                        )
                        .on_conflict_do_nothing()
                    )

    async def start(
        self,
        key: str,
        request: RetailRequest,
        retrieval: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"evalmutationbench:retail:{key}"))
        payload = {
            "kind": "retail_start",
            "session_id": session_id,
            **request.model_dump(),
            "retrieval": retrieval or {"mode": "lexical"},
        }
        return await self.store.create(key, payload)

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        async with self.store.engine.connect() as connection:
            row = (
                (
                    await connection.execute(
                        select(sessions).where(sessions.c.session_id == session_id)
                    )
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                return None
            order = (
                (await connection.execute(select(orders).where(orders.c.session_id == session_id)))
                .mappings()
                .one_or_none()
            )
        return {**dict(row), "order": dict(order) if order else None}

    async def approve(
        self,
        key: str,
        session_id: str,
        approval: ApprovalRequest,
    ) -> dict[str, Any]:
        payload = {"kind": "retail_approval", "session_id": session_id, **approval.model_dump()}
        return await self.store.create(key, payload)

    async def prepare_enqueue(self, connection: Any, run_id: str, request: dict[str, Any]) -> None:
        """Session/decision and queued command share one database transaction."""
        if request["kind"] == "retail_start":
            await connection.execute(
                sessions.insert().values(
                    session_id=request["session_id"],
                    sku=request["sku"],
                    target_days=request["target_days"],
                    phase="queued",
                )
            )
            return
        accepted = (
            await connection.execute(
                update(sessions)
                .where(
                    sessions.c.session_id == request["session_id"],
                    sessions.c.phase == "awaiting_approval",
                    sessions.c.decision.is_(None),
                    sessions.c.proposal_sha256 == request["proposal_sha256"],
                )
                .values(decision=request["approved"], approval_run_id=run_id)
                .returning(sessions.c.session_id)
            )
        ).first()
        if accepted is None:
            raise RetailConflict("session is not awaiting this proposal's approval")

    async def read_stock(self, sku: str) -> StockSnapshot:
        async with self.store.engine.connect() as connection:
            row = (
                (await connection.execute(select(inventory).where(inventory.c.sku == sku)))
                .mappings()
                .one()
            )
        return StockSnapshot.model_validate(dict(row))

    async def read_documents(self, sku: str) -> list[dict[str, Any]]:
        async with self.store.engine.connect() as connection:
            rows = (
                (
                    await connection.execute(
                        select(documents)
                        .where(documents.c.sku == sku, documents.c.is_current.is_(True))
                        .order_by(documents.c.chunk_id)
                    )
                )
                .mappings()
                .all()
            )
        return [dict(row) for row in rows]

    async def retrieval_config(self, session_id: str) -> dict[str, Any]:
        async with self.store.engine.connect() as connection:
            request: dict[str, Any] = (
                await connection.execute(
                    select(runs.c.request).where(
                        runs.c.request["kind"].as_string() == "retail_start",
                        runs.c.request["session_id"].as_string() == session_id,
                    )
                )
            ).scalar_one()
        config: dict[str, Any] = request.get("retrieval", {"mode": "lexical"})
        return config

    async def save_proposal(self, session_id: str, proposal: Proposal) -> None:
        async with self.store.engine.begin() as connection:
            await connection.execute(
                update(sessions)
                .where(sessions.c.session_id == session_id, sessions.c.decision.is_(None))
                .values(
                    phase="awaiting_approval",
                    proposal=proposal.model_dump(),
                    proposal_sha256=proposal.checksum(),
                )
            )

    async def create_order(self, session_id: str, proposal: Proposal) -> str:
        async with self.store.engine.begin() as connection:
            session = (
                (
                    await connection.execute(
                        select(sessions)
                        .where(sessions.c.session_id == session_id)
                        .with_for_update()
                    )
                )
                .mappings()
                .one()
            )
            existing = (
                (await connection.execute(select(orders).where(orders.c.session_id == session_id)))
                .mappings()
                .one_or_none()
            )
            if existing is not None:
                if existing["proposal_sha256"] != proposal.checksum():
                    raise RetailConflict("existing order belongs to a different proposal")
                return str(existing["order_id"])
            if session["decision"] is not True or session["proposal_sha256"] != proposal.checksum():
                raise RetailConflict("purchase requires approval of this exact proposal")
            owner = (
                await connection.execute(
                    select(runs.c.run_id)
                    .where(
                        runs.c.run_id == session["approval_run_id"],
                        runs.c.status == "running",
                        runs.c.lease_until > time.time(),
                    )
                    .with_for_update()
                )
            ).first()
            if owner is None:
                raise RetailConflict("approval execution no longer has an active lease")
            current_docs = (
                (
                    await connection.execute(
                        select(documents)
                        .where(documents.c.sku == proposal.sku, documents.c.is_current.is_(True))
                        .with_for_update()
                    )
                )
                .mappings()
                .all()
            )
            if evidence_checksum([dict(doc) for doc in current_docs]) != proposal.evidence_sha256:
                raise RetailConflict("supplier or policy source changed; request a fresh proposal")
            stock = (
                (
                    await connection.execute(
                        select(inventory).where(inventory.c.sku == proposal.sku).with_for_update()
                    )
                )
                .mappings()
                .one()
            )
            if stock["revision"] != proposal.inventory_revision:
                raise RetailConflict("inventory changed; request a fresh recommendation")
            if session["sku"] != proposal.sku:
                raise RetailConflict("proposal belongs to a different SKU")
            order_id = str(uuid.uuid4())
            await connection.execute(
                orders.insert().values(
                    order_id=order_id,
                    session_id=session_id,
                    sku=proposal.sku,
                    quantity=proposal.quantity,
                    total_cost_cents=proposal.total_cost_cents,
                    proposal_sha256=proposal.checksum(),
                )
            )
            await connection.execute(
                update(sessions).where(sessions.c.session_id == session_id).values(phase="ordered")
            )
        return order_id

    async def save_trace(self, session_id: str, trace: dict[str, Any], phase: str) -> None:
        async with self.store.engine.begin() as connection:
            await connection.execute(
                update(sessions)
                .where(sessions.c.session_id == session_id)
                .values(trace=trace, phase=phase)
            )

    async def fail_session(self, session_id: str) -> None:
        async with self.store.engine.begin() as connection:
            await connection.execute(update(sessions).where(
                sessions.c.session_id == session_id,
                sessions.c.phase.not_in(("ordered", "rejected")),
            ).values(phase="failed"))
