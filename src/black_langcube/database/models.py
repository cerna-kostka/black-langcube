"""
SQLAlchemy ORM models for the Black LangCube database storage layer.

Tables
------
sessions       — replaces timestamped result folders
graph_outputs  — replaces graph*_output.json per-stage files
node_outputs   — replaces per-node intermediate JSON lines
token_usage    — replaces token accounting entries in event logs

Design decisions
----------------
- UUID primary keys are safe for distributed / parallel sessions.
- ``JSONType`` is ``JSONB`` on PostgreSQL (efficient querying) and ``JSON`` on
  SQLite (test compatibility), selected at import time from DATABASE_URL.
- Every foreign key column carries an explicit ``Index``.
- ``TimestampMixin`` provides ``created_at`` / ``updated_at`` on all tables.
- Cascade deletes ensure that removing a Session removes all child rows.
- ``Session.status`` is validated via ``@validates`` to behave like an enum.
"""

import uuid

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from .config import Base, DATABASE_URL

# ---------------------------------------------------------------------------
# JSON type alias: JSONB on PostgreSQL, plain JSON for SQLite
# ---------------------------------------------------------------------------
_IS_POSTGRES: bool = DATABASE_URL.startswith("postgresql")
JSONType = JSONB if _IS_POSTGRES else JSON

# ---------------------------------------------------------------------------
# UUID type alias: native on PostgreSQL, String on SQLite
# ---------------------------------------------------------------------------
if _IS_POSTGRES:
    UUIDType = PG_UUID(as_uuid=True)
    _uuid_default = uuid.uuid4  # returns uuid.UUID object
else:
    # SQLite stores UUIDs as 36-character strings.
    UUIDType = String(36)
    _uuid_default = lambda: str(uuid.uuid4())  # noqa: E731

# ---------------------------------------------------------------------------
# Valid session statuses
# ---------------------------------------------------------------------------
VALID_SESSION_STATUSES: frozenset[str] = frozenset(
    {"running", "completed", "failed", "cancelled"}
)

# ---------------------------------------------------------------------------
# Timestamp mixin
# ---------------------------------------------------------------------------


class TimestampMixin:
    """Adds created_at and updated_at columns to every inheriting model."""

    created_at: Mapped[DateTime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=True
    )


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------


class Session(TimestampMixin, Base):
    """Top-level session record; replaces timestamped result folders."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(UUIDType, primary_key=True, default=_uuid_default)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, nullable=True)

    # Relationships ----------------------------------------------------------------
    graph_outputs: Mapped[list["GraphOutput"]] = relationship(
        "GraphOutput",
        back_populates="session",
        cascade="all, delete-orphan",
    )
    node_outputs: Mapped[list["NodeOutput"]] = relationship(
        "NodeOutput",
        back_populates="session",
        cascade="all, delete-orphan",
    )
    token_usages: Mapped[list["TokenUsage"]] = relationship(
        "TokenUsage",
        back_populates="session",
        cascade="all, delete-orphan",
    )

    # Indexes ----------------------------------------------------------------------
    __table_args__ = (Index("ix_sessions_status", "status"),)

    @validates("status")
    def validate_status(self, key: str, value: str) -> str:  # noqa: ARG002
        if value not in VALID_SESSION_STATUSES:
            raise ValueError(
                f"Invalid session status '{value}'. "
                f"Must be one of: {sorted(VALID_SESSION_STATUSES)}"
            )
        return value

    def __repr__(self) -> str:
        return f"<Session id={self.id!r} status={self.status!r}>"


class GraphOutput(TimestampMixin, Base):
    """Per-graph-step output; replaces graph*_output.json files."""

    __tablename__ = "graph_outputs"

    id: Mapped[str] = mapped_column(UUIDType, primary_key=True, default=_uuid_default)
    session_id: Mapped[str] = mapped_column(
        UUIDType, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    graph_name: Mapped[str] = mapped_column(String(255), nullable=False)
    step_name: Mapped[str] = mapped_column(String(255), nullable=True)
    data: Mapped[dict] = mapped_column(JSONType, nullable=True)

    # Relationships ----------------------------------------------------------------
    session: Mapped["Session"] = relationship(
        "Session",
        back_populates="graph_outputs",
    )

    # Indexes & constraints --------------------------------------------------------
    __table_args__ = (
        UniqueConstraint(
            "session_id", "graph_name", "step_name", name="uq_graph_output_session_step"
        ),
        Index("ix_graph_outputs_session_id", "session_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<GraphOutput id={self.id!r} graph={self.graph_name!r}"
            f" step={self.step_name!r}>"
        )


class NodeOutput(TimestampMixin, Base):
    """Per-node intermediate output; replaces per-node JSON lines."""

    __tablename__ = "node_outputs"

    id: Mapped[str] = mapped_column(UUIDType, primary_key=True, default=_uuid_default)
    session_id: Mapped[str] = mapped_column(
        UUIDType, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    graph_name: Mapped[str] = mapped_column(String(255), nullable=False)
    node_name: Mapped[str] = mapped_column(String(255), nullable=False)
    data: Mapped[dict] = mapped_column(JSONType, nullable=True)

    # Relationships ----------------------------------------------------------------
    session: Mapped["Session"] = relationship(
        "Session",
        back_populates="node_outputs",
    )

    # Indexes & constraints --------------------------------------------------------
    __table_args__ = (Index("ix_node_outputs_session_id", "session_id"),)

    def __repr__(self) -> str:
        return (
            f"<NodeOutput id={self.id!r} graph={self.graph_name!r}"
            f" node={self.node_name!r}>"
        )


class TokenUsage(TimestampMixin, Base):
    """Token accounting entry; replaces token log lines in event logs."""

    __tablename__ = "token_usage"

    id: Mapped[str] = mapped_column(UUIDType, primary_key=True, default=_uuid_default)
    session_id: Mapped[str] = mapped_column(
        UUIDType, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    graph_name: Mapped[str] = mapped_column(String(255), nullable=False)
    node_name: Mapped[str] = mapped_column(String(255), nullable=True)
    model: Mapped[str] = mapped_column(String(128), nullable=True)
    prompt_tokens: Mapped[int] = mapped_column(nullable=True)
    completion_tokens: Mapped[int] = mapped_column(nullable=True)
    total_tokens: Mapped[int] = mapped_column(nullable=True)
    raw_data: Mapped[dict] = mapped_column(JSONType, nullable=True)

    # Relationships ----------------------------------------------------------------
    session: Mapped["Session"] = relationship(
        "Session",
        back_populates="token_usages",
    )

    # Indexes & constraints --------------------------------------------------------
    __table_args__ = (
        Index("ix_token_usage_session_id", "session_id"),
        Index("ix_token_usage_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<TokenUsage id={self.id!r} graph={self.graph_name!r}"
            f" total={self.total_tokens!r}>"
        )


# Convenience re-export of all models for use in tests and migrations
__all__ = [
    "Base",
    "TimestampMixin",
    "VALID_SESSION_STATUSES",
    "Session",
    "GraphOutput",
    "NodeOutput",
    "TokenUsage",
]
