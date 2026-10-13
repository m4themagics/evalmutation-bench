from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RetailRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sku: Literal["COFFEE-001", "TEA-001", "RICE-001"]
    target_days: int = Field(default=14, ge=1, le=30, strict=True)


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approved: bool = Field(strict=True)
    proposal_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class StockSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sku: str
    on_hand: int = Field(ge=0)
    sold_last_7_days: int = Field(ge=0)
    unit_cost_cents: int = Field(gt=0)
    revision: int = Field(ge=1)


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sku: str
    quantity: int = Field(gt=0, le=10000)
    total_cost_cents: int = Field(gt=0)
    inventory_revision: int = Field(ge=1)
    target_days: int = Field(ge=1, le=30)
    citations: list[str] = Field(min_length=2)
    evidence_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    explanation: str

    def checksum(self) -> str:
        raw = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()


def evidence_checksum(documents: list[dict[str, Any]]) -> str:
    raw = json.dumps(
        sorted(documents, key=lambda doc: doc["chunk_id"]), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(raw).hexdigest()
