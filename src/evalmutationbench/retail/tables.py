from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    ForeignKey,
    Integer,
    String,
    Table,
)

from ..service.store import metadata

inventory = Table(
    "retail_inventory",
    metadata,
    Column("sku", String(32), primary_key=True),
    Column("on_hand", Integer, nullable=False),
    Column("sold_last_7_days", Integer, nullable=False),
    Column("unit_cost_cents", Integer, nullable=False),
    Column("revision", Integer, nullable=False),
    CheckConstraint("on_hand >= 0 AND sold_last_7_days >= 0 AND unit_cost_cents > 0"),
)
documents = Table(
    "retail_documents",
    metadata,
    Column("chunk_id", String(128), primary_key=True),
    Column("sku", String(32), nullable=False),
    Column("title", String(256), nullable=False),
    Column("content", String, nullable=False),
    Column("source_uri", String(256), nullable=False),
    Column("version", Integer, nullable=False),
    Column("is_current", Boolean, nullable=False),
    Column("rules", JSON, nullable=False),
)
sessions = Table(
    "retail_sessions",
    metadata,
    Column("session_id", String(36), primary_key=True),
    Column("sku", String(32), ForeignKey("retail_inventory.sku"), nullable=False),
    Column("target_days", Integer, nullable=False),
    Column("phase", String(32), nullable=False),
    Column("proposal", JSON),
    Column("proposal_sha256", String(64)),
    Column("decision", Boolean),
    Column("approval_run_id", String(36)),
    Column("trace", JSON),
)
orders = Table(
    "retail_purchase_orders",
    metadata,
    Column("order_id", String(36), primary_key=True),
    Column(
        "session_id",
        String(36),
        ForeignKey("retail_sessions.session_id"),
        nullable=False,
        unique=True,
    ),
    Column("sku", String(32), nullable=False),
    Column("quantity", Integer, nullable=False),
    Column("total_cost_cents", Integer, nullable=False),
    Column("proposal_sha256", String(64), nullable=False),
    CheckConstraint("quantity > 0 AND total_cost_cents > 0"),
)
