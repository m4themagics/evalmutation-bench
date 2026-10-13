from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    Float,
    Index,
    MetaData,
    String,
    Table,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

metadata = MetaData()
runs = Table(
    "evaluation_runs",
    metadata,
    Column("run_id", String(36), primary_key=True),
    Column("idempotency_key", String(128), nullable=False, unique=True),
    Column("request", JSON, nullable=False),
    Column("status", String(16), nullable=False),
    Column("created_at", Float, nullable=False),
    Column("lease_until", Float),
    Column("claim_token", String(36)),
    Column("result", JSON),
    Column("error", String(64)),
    Index("ix_runs_status_created", "status", "created_at"),
)


class IdempotencyConflict(ValueError):
    pass


class RunStore:
    def __init__(self, database_url: str) -> None:
        self.engine = create_async_engine(database_url, pool_pre_ping=True)

    async def initialize(self) -> None:
        """Explicit initial-schema creation; never run DDL during API startup."""
        from ..retail import tables  # noqa: F401 - register additive tables

        async with self.engine.begin() as connection:
            await connection.run_sync(metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()

    async def healthy(self) -> bool:
        async with self.engine.connect() as connection:
            await connection.execute(select(runs.c.run_id).limit(0))
        return True

    async def create(self, key: str, request: dict[str, Any]) -> dict[str, Any]:
        run_id = str(uuid.uuid4())
        try:
            async with self.engine.begin() as connection:
                await connection.execute(
                    runs.insert().values(
                        run_id=run_id,
                        idempotency_key=key,
                        request=request,
                        status="queued",
                        created_at=time.time(),
                    )
                )
                if request.get("kind") in ("retail_start", "retail_approval"):
                    from ..retail.store import RetailStore

                    await RetailStore(self).prepare_enqueue(connection, run_id, request)
        except IntegrityError:
            async with self.engine.connect() as connection:
                existing = (
                    (await connection.execute(select(runs).where(runs.c.idempotency_key == key)))
                    .mappings()
                    .one_or_none()
                )
            if existing is None:
                raise
            if existing["request"] != request:
                raise IdempotencyConflict("key already belongs to a different request") from None
            return dict(existing)
        created = await self.get(run_id)
        assert created is not None
        return created

    async def get(self, run_id: str) -> dict[str, Any] | None:
        async with self.engine.connect() as connection:
            row = (
                (await connection.execute(select(runs).where(runs.c.run_id == run_id)))
                .mappings()
                .one_or_none()
            )
        return dict(row) if row is not None else None

    async def claim(self, lease_seconds: float) -> dict[str, Any] | None:
        """Atomic compare-and-set; competing workers can never own the same claim."""
        candidate = (
            select(runs.c.run_id)
            .where(runs.c.status == "queued")
            .order_by(runs.c.created_at, runs.c.run_id)
            .limit(1)
            .scalar_subquery()
        )
        async with self.engine.begin() as connection:
            row = (
                (
                    await connection.execute(
                        update(runs)
                        .where(runs.c.run_id == candidate, runs.c.status == "queued")
                        .values(
                            status="running",
                            claim_token=str(uuid.uuid4()),
                            lease_until=time.time() + lease_seconds,
                        )
                        .returning(runs)
                    )
                )
                .mappings()
                .one_or_none()
            )
        return dict(row) if row is not None else None

    async def finish(
        self,
        run_id: str,
        claim_token: str,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        async with self.engine.begin() as connection:
            row = (
                await connection.execute(
                    update(runs)
                    .where(
                        runs.c.run_id == run_id,
                        runs.c.status == "running",
                        runs.c.claim_token == claim_token,
                        runs.c.lease_until > time.time(),
                    )
                    .values(
                        status="failed" if error else "completed",
                        result=result,
                        error=error,
                        lease_until=None,
                    )
                    .returning(runs.c.run_id)
                )
            ).first()
        return row is not None

    async def expire(self) -> None:
        # Never retry abandoned paid runs implicitly: delivery is not exactly-once.
        async with self.engine.begin() as connection:
            await connection.execute(
                update(runs)
                .where(runs.c.status == "running", runs.c.lease_until <= time.time())
                .values(status="failed", error="WorkerLeaseExpired", lease_until=None)
            )


def main() -> None:
    from .settings import Settings

    async def initialize() -> None:
        settings = Settings.from_env()
        store = RunStore(settings.database_url.get_secret_value())
        try:
            await store.initialize()
            if settings.retail_enabled:
                from ..retail.runtime import initialize as initialize_retail

                await initialize_retail(store)
        finally:
            await store.close()

    asyncio.run(initialize())
