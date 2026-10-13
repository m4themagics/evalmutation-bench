from __future__ import annotations

import hashlib
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from sqlalchemy import text
from sqlalchemy.engine import make_url

from ..service.store import RunStore
from .graph import build_graph
from .store import RetailConflict, RetailStore


def checkpoint_url(store: RunStore) -> str:
    return (
        make_url(store.engine.url)
        .set(drivername="postgresql")
        .render_as_string(hide_password=False)
    )


async def initialize(store: RunStore) -> None:
    if store.engine.dialect.name != "postgresql":
        raise ValueError("durable retail execution requires PostgreSQL")
    await RetailStore(store).seed()
    async with AsyncPostgresSaver.from_conn_string(checkpoint_url(store)) as saver:
        await saver.setup()


async def process_job(store: RunStore, row: dict[str, Any]) -> dict[str, Any]:
    if store.engine.dialect.name != "postgresql":
        raise ValueError("durable retail execution requires PostgreSQL")
    request = row["request"]
    session_id = request["session_id"]
    key = int.from_bytes(hashlib.sha256(session_id.encode()).digest()[:8], "big", signed=True)
    retail = RetailStore(store)
    # A PostgreSQL advisory lock serializes writes to one LangGraph thread across processes.
    async with store.engine.connect() as lock_connection:
        await lock_connection.execute(text("SELECT pg_advisory_lock(:key)"), {"key": key})
        try:
            session = await retail.get_session(session_id)
            if session is None:
                raise RetailConflict("session not found")
            async with AsyncPostgresSaver.from_conn_string(checkpoint_url(store)) as saver:
                graph = build_graph(saver, retail, await retail.retrieval_config(session_id))
                config = {"configurable": {"thread_id": session_id}}
                snapshot = await graph.aget_state(config)
                if session["phase"] in ("ordered", "rejected") and not snapshot.next:
                    return {
                        "session_id": session_id,
                        "phase": session["phase"],
                        "order": session["order"],
                        "trace": session["trace"],
                    }
                if request["kind"] == "retail_start":
                    initial = {
                        "session_id": session_id,
                        "sku": request["sku"],
                        "target_days": request["target_days"],
                    }
                    await graph.ainvoke(None if snapshot.values else initial, config)
                else:
                    if session["approval_run_id"] != row["run_id"]:
                        raise RetailConflict("approval command does not own this decision")
                    if not any(task.interrupts for task in snapshot.tasks):
                        await graph.ainvoke(None, config)
                    await graph.ainvoke(
                        Command(
                            resume={
                                "approved": request["approved"],
                                "proposal_sha256": request["proposal_sha256"],
                            }
                        ),
                        config,
                    )
            current = await retail.get_session(session_id)
            assert current is not None
            return {
                "session_id": session_id,
                "phase": current["phase"],
                "proposal": current["proposal"],
                "proposal_sha256": current["proposal_sha256"],
                "order": current["order"],
                "trace": current["trace"],
            }
        finally:
            await lock_connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})
