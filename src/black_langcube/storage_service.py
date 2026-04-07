"""
StorageService — three-mode storage abstraction.

The active mode is controlled by the ``STORAGE_MODE`` environment variable:

| Mode       | Behavior                                                  |
|------------|-----------------------------------------------------------|
| ``file``   | Write/read files only — backward-compatible default       |
| ``database``| Write/read database only                                 |
| ``dual``   | Write to both; read from database first (migration path)  |

Unknown mode values raise ``ValueError`` at construction time (fail fast).

The ``_db_service`` property is lazy: the ``DatabaseService`` is not
instantiated until the first database write, avoiding DB connection cost in
file-only deployments.

Dual-mode degradation
---------------------
In ``dual`` mode a DB write failure is logged and execution falls back to the
file write. If *both* the DB write and the file write fail the original
exception from the file write is re-raised.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

STORAGE_MODE_ENV: str = "STORAGE_MODE"

VALID_STORAGE_MODES: frozenset[str] = frozenset({"file", "database", "dual"})

DEFAULT_STORAGE_MODE: str = "file"


class StorageService:
    """Abstraction over file-based and database-backed output storage.

    Args:
        storage_mode: One of ``"file"``, ``"database"``, or ``"dual"``.
            Reads the ``STORAGE_MODE`` environment variable when omitted.

    Raises:
        ValueError: If the resolved mode is not one of the valid values.
    """

    def __init__(self, storage_mode: str | None = None) -> None:
        resolved = (
            storage_mode
            if storage_mode is not None
            else os.getenv(STORAGE_MODE_ENV, DEFAULT_STORAGE_MODE)
        )
        if resolved not in VALID_STORAGE_MODES:
            raise ValueError(
                f"Invalid storage mode: {resolved!r}. "
                f"Must be one of: {sorted(VALID_STORAGE_MODES)}"
            )
        self._storage_mode: str = resolved
        self.__db_service: Any = None  # lazy; see _db_service property

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def storage_mode(self) -> str:
        """Return the active storage mode."""
        return self._storage_mode

    @property
    def _db_service(self) -> Any:
        """Lazy initializer for DatabaseService.

        The DatabaseService (and therefore the DB engine) is not created until
        the first time a database write is attempted. This avoids paying the
        connection-pool setup cost in file-only deployments.
        """
        if self.__db_service is None:
            # Import here to avoid circular imports and to defer the import
            # of SQLAlchemy (an optional dependency) until actually needed.
            from .database.operations import DatabaseService  # noqa: PLC0415

            self.__db_service = DatabaseService
        return self.__db_service

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    async def save_graph_output(
        self,
        session_id: str,
        graph_name: str,
        data: dict,
        step_name: str | None = None,
        file_path: str | None = None,
    ) -> bool:
        """Persist graph output according to the active storage mode.

        Args:
            session_id: Session UUID string.
            graph_name: Name of the graph that produced the output.
            data: JSON-serialisable output data.
            step_name: Optional step/stage identifier.
            file_path: Path to write the JSON file (required in ``file``/``dual``
                modes unless the caller handles file writing itself).

        Returns:
            ``True`` if *at least one* write succeeded, ``False`` otherwise.
        """
        if self._storage_mode == "file":
            return await self._write_file(file_path, data)

        if self._storage_mode == "database":
            async with self._db_service() as db:
                return await db.save_graph_output(
                    session_id, graph_name, data, step_name=step_name
                )

        # dual
        db_ok = False
        try:
            async with self._db_service() as db:
                db_ok = await db.save_graph_output(
                    session_id, graph_name, data, step_name=step_name
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "StorageService dual-mode DB write failed (save_graph_output): %s", exc
            )

        file_ok = await self._write_file(file_path, data)

        if not db_ok and not file_ok:
            raise RuntimeError(
                f"StorageService: both DB and file writes failed for graph_output"
                f" session={session_id!r} graph={graph_name!r}"
            )
        return True

    async def save_node_output(
        self,
        session_id: str,
        graph_name: str,
        node_name: str,
        data: dict,
        file_path: str | None = None,
    ) -> bool:
        """Persist node output according to the active storage mode.

        Args:
            session_id: Session UUID string.
            graph_name: Name of the owning graph.
            node_name: Name of the node that produced the output.
            data: JSON-serialisable output data.
            file_path: Path to write the JSON file (optional).

        Returns:
            ``True`` if *at least one* write succeeded, ``False`` otherwise.
        """
        if self._storage_mode == "file":
            return await self._write_file(file_path, data)

        if self._storage_mode == "database":
            async with self._db_service() as db:
                return await db.save_node_output(
                    session_id, graph_name, node_name, data
                )

        # dual
        db_ok = False
        try:
            async with self._db_service() as db:
                db_ok = await db.save_node_output(
                    session_id, graph_name, node_name, data
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "StorageService dual-mode DB write failed (save_node_output): %s", exc
            )

        file_ok = await self._write_file(file_path, data)

        if not db_ok and not file_ok:
            raise RuntimeError(
                f"StorageService: both DB and file writes failed for node_output"
                f" session={session_id!r} graph={graph_name!r} node={node_name!r}"
            )
        return True

    async def save_token_usage(
        self,
        session_id: str,
        graph_name: str,
        node_name: str | None = None,
        model: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        raw_data: dict | None = None,
        file_path: str | None = None,
    ) -> bool:
        """Persist token usage according to the active storage mode.

        Args:
            session_id: Session UUID string.
            graph_name: Name of the owning graph.
            node_name: Optional node name.
            model: Optional model identifier.
            prompt_tokens: Number of prompt tokens used.
            completion_tokens: Number of completion tokens used.
            total_tokens: Total token count.
            raw_data: Optional raw token accounting payload.
            file_path: Path to write the JSON file (optional).

        Returns:
            ``True`` if *at least one* write succeeded, ``False`` otherwise.
        """
        if self._storage_mode == "file":
            return await self._write_file(file_path, raw_data or {})

        if self._storage_mode == "database":
            async with self._db_service() as db:
                return await db.save_token_usage(
                    session_id,
                    graph_name,
                    node_name=node_name,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    raw_data=raw_data,
                )

        # dual
        db_ok = False
        try:
            async with self._db_service() as db:
                db_ok = await db.save_token_usage(
                    session_id,
                    graph_name,
                    node_name=node_name,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    raw_data=raw_data,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "StorageService dual-mode DB write failed (save_token_usage): %s", exc
            )

        file_ok = await self._write_file(file_path, raw_data or {})

        if not db_ok and not file_ok:
            raise RuntimeError(
                f"StorageService: both DB and file writes failed for token_usage"
                f" session={session_id!r} graph={graph_name!r}"
            )
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _write_file(file_path: str | None, data: dict) -> bool:
        """Write *data* as JSON to *file_path* asynchronously.

        Returns ``True`` on success, ``False`` when *file_path* is ``None`` (a
        no-op, since the caller did not request file output) or on I/O error.
        """
        if file_path is None:
            return False
        import json  # noqa: PLC0415

        import aiofiles  # noqa: PLC0415

        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            async with aiofiles.open(file_path, "w", encoding="utf-8") as fh:
                await fh.write(json.dumps(data, ensure_ascii=False, indent=2))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("StorageService file write failed: %s", exc)
            return False
