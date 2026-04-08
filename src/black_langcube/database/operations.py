"""
DatabaseService — async context manager for all database operations.

Usage
-----
    async with DatabaseService() as db:
        session_id = await db.create_session(metadata={"run": "1"})
        await db.save_graph_output(session_id, "graph1", data)

Lifecycle
---------
- ``__aenter__`` opens an ``AsyncSession``.
- ``__aexit__`` commits on clean exit, rolls back on exception, always closes.

Error handling
--------------
Every public write method wraps its body in ``try/except SQLAlchemyError``.
On error it logs a warning and returns ``False`` / ``None`` rather than
re-raising, so callers can decide how to respond without being forced to
handle DB exceptions everywhere.
"""

import logging
import uuid

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from .config import engine
from .models import GraphOutput, NodeOutput, Session, TokenUsage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility: null-byte sanitizer
# ---------------------------------------------------------------------------


def sanitize_json_data(data: object) -> object:
    """Recursively strip PostgreSQL-hostile null bytes (\\u0000) from JSON values.

    PostgreSQL rejects ``\\u0000`` in ``text`` and ``JSONB`` columns. This
    function must be applied before *every* database write — retrofitting it
    later is error-prone.

    Args:
        data: Any JSON-serialisable value (dict, list, str, or scalar).

    Returns:
        A new value of the same shape with null bytes removed from all strings.
    """
    if isinstance(data, dict):
        return {k: sanitize_json_data(v) for k, v in data.items()}
    if isinstance(data, list):
        return [sanitize_json_data(v) for v in data]
    if isinstance(data, str):
        return data.replace("\u0000", "")
    return data


# ---------------------------------------------------------------------------
# DatabaseService
# ---------------------------------------------------------------------------


class DatabaseService:
    """Async context manager that wraps a single ``AsyncSession``.

    All write methods:
    - sanitize data before insertion
    - use ``flush()`` (not ``commit()``) to obtain generated IDs within the
      open transaction
    - return ``True`` on success, ``False`` on ``SQLAlchemyError``

    All read methods:
    - use ``scalar_one_or_none()`` for single-row look-ups
    - return ``None`` on ``SQLAlchemyError``
    """

    def __init__(self) -> None:
        self.session: AsyncSession | None = None

    async def __aenter__(self) -> "DatabaseService":
        self.session = AsyncSession(engine)
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        assert self.session is not None  # noqa: S101 (invariant — set in __aenter__)
        try:
            if exc_type is None:
                await self.session.commit()
            else:
                await self.session.rollback()
        finally:
            await self.session.close()

    # ------------------------------------------------------------------
    # Session operations
    # ------------------------------------------------------------------

    async def create_session(
        self,
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> str | None:
        """Insert a new Session row and return its id.

        Args:
            session_id: Optional explicit UUID string. Generated if omitted.
            metadata: Optional free-form JSON metadata dict.

        Returns:
            The session id string on success, ``None`` on error.
        """
        assert self.session is not None
        try:
            sid = session_id or str(uuid.uuid4())
            obj = Session(
                id=sid,
                metadata_=sanitize_json_data(metadata) if metadata else None,
            )
            self.session.add(obj)
            await self.session.flush()
            return sid
        except SQLAlchemyError as exc:
            logger.error("DB write failed (create_session): %s", exc)
            return None

    async def get_session(self, session_id: str) -> Session | None:
        """Fetch a single Session by primary key.

        Args:
            session_id: The UUID string of the session to retrieve.

        Returns:
            The ``Session`` ORM object, or ``None`` if the session does not
            exist.

        Raises:
            SQLAlchemyError: Re-raised on database errors so callers can
                distinguish a missing session (``None`` return) from a DB
                connectivity or query failure (raised exception).
        """
        assert self.session is not None
        try:
            stmt = select(Session).where(Session.id == session_id)
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            logger.error("DB read failed (get_session): %s", exc)
            raise

    async def update_session_current_node_id(
        self, session_id: str, node_id: int
    ) -> bool:
        """Update the ``current_node_id`` field of an existing Session.

        Args:
            session_id: The UUID string identifying the session.
            node_id: The node that has just completed successfully.

        Returns:
            ``True`` on success, ``False`` when the session is not found or a
            database error occurs.
        """
        assert self.session is not None
        try:
            obj = await self.get_session(session_id)
            if obj is None:
                logger.warning(
                    "update_session_current_node_id: session %r not found", session_id
                )
                return False
            obj.current_node_id = node_id
            self.session.add(obj)
            await self.session.flush()
            return True
        except SQLAlchemyError as exc:
            logger.error("DB write failed (update_session_current_node_id): %s", exc)
            return False

    async def update_session_status(self, session_id: str, status: str) -> bool:
        """Update the status field of an existing Session.

        Args:
            session_id: The UUID string identifying the session.
            status: New status value; must be in ``VALID_SESSION_STATUSES``.

        Returns:
            ``True`` on success, ``False`` when the session is not found or a
            database error occurs.

        Raises:
            ValueError: If *status* is not a member of ``VALID_SESSION_STATUSES``.
                Raised by the ``@validates`` decorator on ``Session.status`` before
                any database write — treat this as a programmer error, not a
                recoverable runtime failure.
        """
        assert self.session is not None
        try:
            obj = await self.get_session(session_id)
            if obj is None:
                logger.warning(
                    "update_session_status: session %r not found", session_id
                )
                return False
            obj.status = status  # @validates fires here
            self.session.add(obj)
            await self.session.flush()
            return True
        except SQLAlchemyError as exc:
            logger.error("DB write failed (update_session_status): %s", exc)
            return False

    # ------------------------------------------------------------------
    # GraphOutput operations
    # ------------------------------------------------------------------

    async def save_graph_output(
        self,
        session_id: str,
        graph_name: str,
        data: object,
        step_name: str | None = None,
    ) -> bool:
        """Insert a GraphOutput row.

        Args:
            session_id: UUID of the owning session.
            graph_name: Name of the graph (e.g. ``"graf1"``).
            data: Arbitrary JSON-serialisable output data.
            step_name: Optional step/stage identifier within the graph.

        Returns:
            ``True`` on success, ``False`` on error.
        """
        assert self.session is not None
        try:
            obj = GraphOutput(
                session_id=session_id,
                graph_name=graph_name,
                step_name=step_name,
                data=sanitize_json_data(data),
            )
            self.session.add(obj)
            await self.session.flush()
            return True
        except SQLAlchemyError as exc:
            logger.error("DB write failed (save_graph_output): %s", exc)
            return False

    # ------------------------------------------------------------------
    # NodeOutput operations
    # ------------------------------------------------------------------

    async def save_node_output(
        self,
        session_id: str,
        graph_name: str,
        node_name: str,
        data: object,
    ) -> bool:
        """Insert a NodeOutput row.

        Args:
            session_id: UUID of the owning session.
            graph_name: Name of the graph containing the node.
            node_name: Name of the node that produced the output.
            data: Arbitrary JSON-serialisable output data.

        Returns:
            ``True`` on success, ``False`` on error.
        """
        assert self.session is not None
        try:
            obj = NodeOutput(
                session_id=session_id,
                graph_name=graph_name,
                node_name=node_name,
                data=sanitize_json_data(data),
            )
            self.session.add(obj)
            await self.session.flush()
            return True
        except SQLAlchemyError as exc:
            logger.error("DB write failed (save_node_output): %s", exc)
            return False

    # ------------------------------------------------------------------
    # TokenUsage operations
    # ------------------------------------------------------------------

    async def save_token_usage(
        self,
        session_id: str,
        graph_name: str,
        node_name: str | None = None,
        model: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        raw_data: object | None = None,
    ) -> bool:
        """Insert a TokenUsage row.

        Args:
            session_id: UUID of the owning session.
            graph_name: Name of the graph that consumed tokens.
            node_name: Optional node name within the graph.
            model: Optional model identifier (e.g. ``"gpt-4o"``).
            prompt_tokens: Number of prompt tokens used.
            completion_tokens: Number of completion tokens used.
            total_tokens: Total token count.
            raw_data: Optional raw token accounting payload.

        Returns:
            ``True`` on success, ``False`` on error.
        """
        assert self.session is not None
        try:
            obj = TokenUsage(
                session_id=session_id,
                graph_name=graph_name,
                node_name=node_name,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                raw_data=sanitize_json_data(raw_data) if raw_data is not None else None,
            )
            self.session.add(obj)
            await self.session.flush()
            return True
        except SQLAlchemyError as exc:
            logger.error("DB write failed (save_token_usage): %s", exc)
            return False
