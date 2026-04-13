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

Dual-mode reads
---------------
In ``dual`` mode reads attempt the database first. On any exception the call
falls back to the file system and logs the fallback at ``WARNING`` level.

Session files (file / dual mode)
---------------------------------
Session metadata is stored as JSON under
``{base_dir}/{session_id}/session.json``. The ``base_dir`` is resolved in
priority order: constructor argument → ``STORAGE_BASE_DIR`` environment
variable → current working directory.

Singleton accessor
------------------
``get_storage_service()`` returns a module-level singleton. The instance can
be replaced for testing via ``set_storage_service()``.
"""

import asyncio
import json
import logging
import os
from typing import Any

import aiofiles

logger = logging.getLogger(__name__)

STORAGE_MODE_ENV: str = "STORAGE_MODE"
STORAGE_BASE_DIR_ENV: str = "STORAGE_BASE_DIR"

VALID_STORAGE_MODES: frozenset[str] = frozenset({"file", "database", "dual"})

DEFAULT_STORAGE_MODE: str = "file"

SESSION_FILENAME: str = "session.json"
GRAPH_OUTPUT_SUFFIX: str = "_output.json"

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_storage_service_instance: "StorageService | None" = None


def get_storage_service() -> "StorageService":
    """Return the shared ``StorageService`` singleton.

    Creates the instance on the first call using the default constructor
    (which reads ``STORAGE_MODE`` from the environment). Subsequent calls
    return the same object.

    Returns:
        The module-level ``StorageService`` instance.

    See Also:
        :func:`set_storage_service` — replace the singleton for testing.
    """
    global _storage_service_instance  # noqa: PLW0603
    if _storage_service_instance is None:
        _storage_service_instance = StorageService()
    return _storage_service_instance


def set_storage_service(instance: "StorageService | None") -> None:
    """Replace the module-level ``StorageService`` singleton.

    Intended for use in tests and dependency-injection scenarios. Pass
    ``None`` to reset the singleton so the next call to
    :func:`get_storage_service` creates a fresh instance from the
    environment.

    Args:
        instance: A ``StorageService`` instance (or ``None`` to clear).
    """
    global _storage_service_instance  # noqa: PLW0603
    _storage_service_instance = instance


class StorageService:
    """Abstraction over file-based and database-backed output storage.

    Args:
        storage_mode: One of ``"file"``, ``"database"``, or ``"dual"``.
            Reads the ``STORAGE_MODE`` environment variable when omitted.
        base_dir: Root directory for file-mode session and output files.
            Reads the ``STORAGE_BASE_DIR`` environment variable when omitted;
            falls back to the current working directory. Only used by the
            session lifecycle and output-retrieval methods in ``file`` and
            ``dual`` modes.

    Raises:
        ValueError: If the resolved mode is not one of the valid values.
    """

    def __init__(
        self,
        storage_mode: str | None = None,
        base_dir: str | None = None,
    ) -> None:
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
        self._base_dir: str = (
            base_dir
            if base_dir is not None
            else os.getenv(STORAGE_BASE_DIR_ENV, os.getcwd())
        )

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
    # Session lifecycle methods
    # ------------------------------------------------------------------

    async def create_session(
        self,
        session_id: str,
        metadata: dict | None = None,
    ) -> bool:
        """Create a new session record according to the active storage mode.

        ``file`` mode
            Writes ``{base_dir}/{session_id}/session.json`` containing the
            session id and the provided metadata.

        ``database`` mode
            Delegates to :meth:`DatabaseService.create_session`.

        ``dual`` mode
            Attempts the database write first; falls back to the file write
            on any exception.

        Args:
            session_id: Unique identifier for the new session.
            metadata: Optional free-form metadata dict to store with the
                session.

        Returns:
            ``True`` if at least one write succeeded, ``False`` otherwise.
        """
        if self._storage_mode == "file":
            return await self._write_session_file(
                session_id, {"id": session_id, "metadata": metadata or {}}
            )

        if self._storage_mode == "database":
            async with self._db_service() as db:
                result = await db.create_session(
                    session_id=session_id, metadata=metadata
                )
                return result is not None

        # dual
        db_ok = False
        try:
            async with self._db_service() as db:
                result = await db.create_session(
                    session_id=session_id, metadata=metadata
                )
                db_ok = result is not None
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "StorageService dual-mode DB write failed (create_session): %s", exc
            )

        file_ok = await self._write_session_file(
            session_id, {"id": session_id, "metadata": metadata or {}}
        )
        if not db_ok and not file_ok:
            raise RuntimeError(
                f"StorageService: both DB and file writes failed for create_session"
                f" session={session_id!r}"
            )
        return db_ok or file_ok

    async def get_session(self, session_id: str) -> dict | None:
        """Retrieve session data according to the active storage mode.

        ``file`` mode
            Reads ``{base_dir}/{session_id}/session.json`` and returns the
            parsed dict, or ``None`` if the file does not exist.

        ``database`` mode
            Delegates to :meth:`DatabaseService.get_session` and converts the
            ORM object to a plain dict with keys ``id``, ``status``,
            ``current_node_id``, and ``metadata``.

        ``dual`` mode
            Attempts the database first; falls back to the file system on
            any exception, logging the fallback at ``WARNING`` level.

        Args:
            session_id: The unique session identifier to look up.

        Returns:
            A dict of session data, or ``None`` if the session does not exist.
        """
        if self._storage_mode == "file":
            return await self._read_session_file(session_id)

        if self._storage_mode == "database":
            async with self._db_service() as db:
                obj = await db.get_session(session_id)
                if obj is None:
                    return None
                return self._session_obj_to_dict(obj)

        # dual — DB first, fall back to file
        try:
            async with self._db_service() as db:
                obj = await db.get_session(session_id)
                if obj is not None:
                    return self._session_obj_to_dict(obj)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "StorageService dual-mode DB read failed (get_session), "
                "falling back to file: %s",
                exc,
            )

        return await self._read_session_file(session_id)

    async def update_session(self, session_id: str, data: dict) -> bool:
        """Merge *data* into the session's metadata according to the active mode.

        ``file`` mode
            Reads ``{base_dir}/{session_id}/session.json``, merges *data*
            into the ``metadata`` sub-dict, and writes the file back.
            Returns ``False`` when the session file does not yet exist.

        ``database`` mode
            Delegates to :meth:`DatabaseService.update_session`.

        ``dual`` mode
            Attempts the database write first; falls back to the file write
            on any exception.

        Args:
            session_id: The unique session identifier to update.
            data: Key/value pairs to merge into the session's metadata.

        Returns:
            ``True`` if at least one write succeeded, ``False`` otherwise.
        """
        if self._storage_mode == "file":
            existing = await self._read_session_file(session_id)
            if existing is None:
                return False
            merged_meta = {**existing.get("metadata", {}), **data}
            return await self._write_session_file(
                session_id, {**existing, "metadata": merged_meta}
            )

        if self._storage_mode == "database":
            async with self._db_service() as db:
                return await db.update_session(session_id, data)

        # dual
        db_ok = False
        try:
            async with self._db_service() as db:
                db_ok = await db.update_session(session_id, data)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "StorageService dual-mode DB write failed (update_session): %s", exc
            )

        existing = await self._read_session_file(session_id)
        if existing is None:
            if not db_ok:
                raise RuntimeError(
                    f"StorageService: both DB and file writes failed for update_session"
                    f" session={session_id!r}"
                )
            return db_ok
        merged_meta = {**existing.get("metadata", {}), **data}
        file_ok = await self._write_session_file(
            session_id, {**existing, "metadata": merged_meta}
        )
        if not db_ok and not file_ok:
            raise RuntimeError(
                f"StorageService: both DB and file writes failed for update_session"
                f" session={session_id!r}"
            )
        return db_ok or file_ok

    # ------------------------------------------------------------------
    # Output retrieval methods
    # ------------------------------------------------------------------

    async def get_graph_output(self, session_id: str, graph_name: str) -> dict | None:
        """Retrieve previously saved graph output according to the active mode.

        ``file`` mode
            Reads ``{base_dir}/{session_id}/{graph_name}_output.json`` and
            returns the parsed dict, or ``None`` if the file does not exist.

        ``database`` mode
            Delegates to :meth:`DatabaseService.get_graph_output`.

        ``dual`` mode
            Attempts the database first; falls back to the file system on
            any exception, logging the fallback at ``WARNING`` level.

        Args:
            session_id: The unique session identifier.
            graph_name: Name of the graph whose output to retrieve.

        Returns:
            The output data dict, or ``None`` if no output was found.
        """
        if self._storage_mode == "file":
            return await self._read_graph_output_file(session_id, graph_name)

        if self._storage_mode == "database":
            async with self._db_service() as db:
                return await db.get_graph_output(session_id, graph_name)

        # dual — DB first, fall back to file
        try:
            async with self._db_service() as db:
                result = await db.get_graph_output(session_id, graph_name)
                if result is not None:
                    return result
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "StorageService dual-mode DB read failed (get_graph_output), "
                "falling back to file: %s",
                exc,
            )

        return await self._read_graph_output_file(session_id, graph_name)

    async def get_output_files(self, session_id: str) -> list[str]:
        """List output file paths or identifiers for a session.

        ``file`` mode
            Returns the absolute paths of all ``*.json`` files found under
            ``{base_dir}/{session_id}/``. Returns an empty list when no
            output has been written for the session.

        ``database`` mode
            Returns a list of distinct ``graph_name`` strings for all
            :class:`~black_langcube.database.models.GraphOutput` rows
            belonging to *session_id*. These are logical identifiers, not
            file-system paths.

        ``dual`` mode
            Attempts the database first; falls back to the file system on
            any exception, logging the fallback at ``WARNING`` level.

        Args:
            session_id: The unique session identifier.

        Returns:
            A (possibly empty) list of file paths (``file`` mode) or
            ``graph_name`` strings (``database`` mode).
        """
        if self._storage_mode == "file":
            return await self._list_output_files(session_id)

        if self._storage_mode == "database":
            async with self._db_service() as db:
                return await db.get_output_files(session_id)

        # dual — DB first, fall back to file
        try:
            async with self._db_service() as db:
                result = await db.get_output_files(session_id)
                if result is not None:
                    return result
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "StorageService dual-mode DB read failed (get_output_files), "
                "falling back to file: %s",
                exc,
            )

        return await self._list_output_files(session_id)

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
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            async with aiofiles.open(file_path, "w", encoding="utf-8") as fh:
                await fh.write(json.dumps(data, ensure_ascii=False, indent=2))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("StorageService file write failed: %s", exc)
            return False

    def _session_dir(self, session_id: str) -> str:
        """Return the directory that holds all files for *session_id*."""
        return os.path.join(self._base_dir, session_id)

    def _session_file_path(self, session_id: str) -> str:
        """Return the path to the session metadata JSON file."""
        return os.path.join(self._session_dir(session_id), SESSION_FILENAME)

    def _graph_output_file_path(self, session_id: str, graph_name: str) -> str:
        """Return the path to the graph output JSON file."""
        return os.path.join(
            self._session_dir(session_id),
            f"{graph_name}{GRAPH_OUTPUT_SUFFIX}",
        )

    async def _write_session_file(self, session_id: str, data: dict) -> bool:
        """Write session metadata to ``{base_dir}/{session_id}/session.json``."""
        return await self._write_file(self._session_file_path(session_id), data)

    async def _read_session_file(self, session_id: str) -> dict | None:
        """Read and return the session metadata from disk, or ``None``."""
        path = self._session_file_path(session_id)
        if not os.path.isfile(path):
            return None
        try:
            async with aiofiles.open(path, encoding="utf-8") as fh:
                return json.loads(await fh.read())
        except Exception as exc:  # noqa: BLE001
            logger.error("StorageService session file read failed: %s", exc)
            return None

    async def _read_graph_output_file(
        self, session_id: str, graph_name: str
    ) -> dict | None:
        """Read graph output from ``{base_dir}/{session_id}/{graph_name}_output.json``."""
        path = self._graph_output_file_path(session_id, graph_name)
        if not os.path.isfile(path):
            return None
        try:
            async with aiofiles.open(path, encoding="utf-8") as fh:
                return json.loads(await fh.read())
        except Exception as exc:  # noqa: BLE001
            logger.error("StorageService graph output file read failed: %s", exc)
            return None

    async def _list_output_files(self, session_id: str) -> list[str]:
        """Return absolute paths of all ``*.json`` files under the session dir."""
        session_dir = self._session_dir(session_id)
        if not os.path.isdir(session_dir):
            return []
        try:
            names = await asyncio.to_thread(os.listdir, session_dir)
            return sorted(
                os.path.join(session_dir, name)
                for name in names
                if name.endswith(".json")
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("StorageService output file listing failed: %s", exc)
            return []

    @staticmethod
    def _session_obj_to_dict(obj: Any) -> dict:
        """Convert a SQLAlchemy Session ORM object to a plain dict."""
        return {
            "id": obj.id,
            "status": obj.status,
            "current_node_id": obj.current_node_id,
            "metadata": obj.metadata_ if obj.metadata_ is not None else {},
        }
