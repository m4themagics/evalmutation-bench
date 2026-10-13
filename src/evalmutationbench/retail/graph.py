from __future__ import annotations

import asyncio
import math
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..models import Evidence, Message, ToolCall, ToolResult, Trace
from ..retrieval import BM25Retriever, HybridCrossEncoderRetriever
from .models import Proposal, StockSnapshot, evidence_checksum
from .store import RetailConflict, RetailStore


class RetailState(TypedDict, total=False):
    session_id: str
    sku: str
    target_days: int
    stock: dict[str, Any]
    documents: list[dict[str, Any]]
    proposal: dict[str, Any]
    approved: bool
    order_id: str | None
    phase: str
    trace: dict[str, Any]


def calculate_proposal(
    stock: StockSnapshot,
    target_days: int,
    documents: list[dict[str, Any]],
) -> Proposal:
    pack_rules = [doc["rules"]["pack_size"] for doc in documents if "pack_size" in doc["rules"]]
    caps = [doc["rules"]["max_units"] for doc in documents if "max_units" in doc["rules"]]
    if len(pack_rules) != 1 or len(caps) != 1:
        raise RetailConflict("exactly one current supplier and policy source are required")
    pack, cap = int(pack_rules[0]), int(caps[0])
    if pack <= 0 or cap < pack:
        raise RetailConflict("invalid supplier rules")
    target_stock = math.ceil(stock.sold_last_7_days * target_days / 7)
    needed = max(0, target_stock - stock.on_hand)
    if not needed:
        raise RetailConflict("stock already covers this horizon; no purchase is required")
    quantity = math.ceil(needed / pack) * pack
    if quantity > cap:
        raise RetailConflict("required purchase exceeds policy limit")
    return Proposal(
        sku=stock.sku,
        quantity=quantity,
        total_cost_cents=quantity * stock.unit_cost_cents,
        inventory_revision=stock.revision,
        target_days=target_days,
        citations=[doc["chunk_id"] for doc in documents],
        evidence_sha256=evidence_checksum(documents),
        explanation=(
            f"{stock.sold_last_7_days} units sold in 7 days; {target_days}-day target "
            f"is {target_stock}; {stock.on_hand} on hand. Round deficit {needed} "
            f"to supplier packs of {pack}: order {quantity} units. "
            f"Policy cap: {cap}. Human approval is required."
        ),
    )


def build_graph(
    checkpointer: Any,
    store: RetailStore,
    retrieval: dict[str, Any] | None = None,
) -> Any:
    async def read_tools(state: RetailState) -> dict[str, Any]:
        stock = await store.read_stock(state["sku"])
        docs = await store.read_documents(state["sku"])
        # Metadata/current-version filtering happens before retrieval.
        evidence = [
            Evidence(
                evidence_id=doc["chunk_id"],
                title=doc["title"],
                content=doc["content"],
                version=doc["version"],
                is_current=True,
                source_uri=doc["source_uri"],
            )
            for doc in docs
        ]
        query = f"{state['sku']} purchase supplier policy approval"
        if retrieval and retrieval.get("mode") == "hybrid":

            def hybrid_search() -> Any:
                return HybridCrossEncoderRetriever(
                    evidence, retrieval["embedding_model"], retrieval["reranker_model"]
                ).search(query)

            ranked = await asyncio.to_thread(hybrid_search)
        else:
            ranked = BM25Retriever(evidence).search(query)
        by_id = {doc["chunk_id"]: doc for doc in docs}
        return {
            "stock": stock.model_dump(),
            "documents": [by_id[item.evidence.evidence_id] for item in ranked],
        }

    async def propose(state: RetailState) -> dict[str, Any]:
        proposal = calculate_proposal(
            StockSnapshot.model_validate(state["stock"]), state["target_days"], state["documents"]
        )
        await store.save_proposal(state["session_id"], proposal)
        return {"proposal": proposal.model_dump(), "phase": "awaiting_approval"}

    def approve(state: RetailState) -> dict[str, Any]:
        proposal = Proposal.model_validate(state["proposal"])
        decision = interrupt(
            {"proposal": proposal.model_dump(), "proposal_sha256": proposal.checksum()}
        )
        # Never use model text or retrieved text as an approval decision.
        if not isinstance(decision, dict) or type(decision.get("approved")) is not bool:
            raise RetailConflict("invalid approval payload")
        if decision.get("proposal_sha256") != proposal.checksum():
            raise RetailConflict("approval does not match this proposal")
        return {"approved": decision["approved"]}

    async def commit(state: RetailState) -> dict[str, Any]:
        if not state["approved"]:
            return {"phase": "rejected", "order_id": None}
        proposal = Proposal.model_validate(state["proposal"])
        order_id = await store.create_order(state["session_id"], proposal)
        return {"phase": "ordered", "order_id": order_id}

    async def trace(state: RetailState) -> dict[str, Any]:
        proposal = Proposal.model_validate(state["proposal"])
        evidence = [
            Evidence(
                evidence_id=doc["chunk_id"],
                title=doc["title"],
                content=doc["content"],
                version=doc["version"],
                is_current=True,
                source_uri=doc["source_uri"],
            )
            for doc in state["documents"]
        ]
        calls = [
            ToolCall(call_id="inventory", name="get_inventory", arguments={"sku": state["sku"]}),
            ToolCall(
                call_id="sources", name="retrieve_supplier_policy", arguments={"sku": state["sku"]}
            ),
            ToolCall(
                call_id="approval",
                name="human_approval",
                arguments={"proposal_sha256": proposal.checksum()},
            ),
        ]
        results = [
            ToolResult(call_id="inventory", status="ok", output=state["stock"]),
            ToolResult(call_id="sources", status="ok", output={"citations": proposal.citations}),
            ToolResult(call_id="approval", status="ok", output={"approved": state["approved"]}),
        ]
        if state["approved"]:
            calls.append(
                ToolCall(
                    call_id="order",
                    name="create_purchase_order",
                    arguments={"sku": proposal.sku, "quantity": proposal.quantity},
                )
            )
            results.append(
                ToolResult(
                    call_id="order",
                    status="ok",
                    output={"order_id": state["order_id"], "committed": True},
                )
            )
        output = Trace(
            trace_id=state["session_id"],
            messages=[
                Message(
                    role="user",
                    content=f"Replenish {state['sku']} for {state['target_days']} days.",
                ),
                Message(role="assistant", content=proposal.explanation),
            ],
            retrieval=evidence,
            tool_calls=calls,
            tool_results=results,
            citations=proposal.citations,
            approvals=[f"human:{proposal.checksum()}"] if state["approved"] else [],
            final_answer=f"{proposal.explanation} Decision: {state['phase']}.",
            state_before=state["stock"],
            state_after={**state["stock"], "order_id": state["order_id"], "phase": state["phase"]},
        ).model_dump(mode="json")
        await store.save_trace(state["session_id"], output, state["phase"])
        return {"trace": output}

    graph = StateGraph(RetailState)
    for name, node in (
        ("read_tools", read_tools),
        ("propose", propose),
        ("approve", approve),
        ("commit", commit),
        ("trace", trace),
    ):
        graph.add_node(name, node)
    graph.add_edge(START, "read_tools")
    graph.add_edge("read_tools", "propose")
    graph.add_edge("propose", "approve")
    graph.add_edge("approve", "commit")
    graph.add_edge("commit", "trace")
    graph.add_edge("trace", END)
    return graph.compile(checkpointer=checkpointer)
