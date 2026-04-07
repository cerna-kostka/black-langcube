"""
Tests for the database storage layer (BP-1).

Covers:
- sanitize_json_data(): dicts, lists, nested structures, null-byte strings
- DatabaseService async context manager lifecycle (commit / rollback / close)
- DatabaseService.create_session() and get_session()
- DatabaseService.update_session_status() including invalid-status rejection
- DatabaseService.save_graph_output()
- DatabaseService.save_node_output()
- DatabaseService.save_token_usage()
- All DB tests use in-memory SQLite with StaticPool (via conftest fixtures)
- Per-test rollback is enforced in the test_db_session fixture (conftest.py)
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from black_langcube.database.operations import DatabaseService, sanitize_json_data  # noqa: E402
from black_langcube.database.models import (  # noqa: E402
    Session,
    VALID_SESSION_STATUSES,
)

# ---------------------------------------------------------------------------
# sanitize_json_data — synchronous unit tests (no DB required)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSanitizeJsonData(unittest.TestCase):
    """Unit tests for the null-byte sanitizer utility."""

    def test_plain_string_unchanged(self):
        assert sanitize_json_data("hello") == "hello"

    def test_null_bytes_removed_from_string(self):
        assert sanitize_json_data("a\u0000b\u0000c") == "abc"

    def test_string_of_only_null_bytes(self):
        assert sanitize_json_data("\u0000\u0000") == ""

    def test_integer_passthrough(self):
        assert sanitize_json_data(42) == 42

    def test_none_passthrough(self):
        assert sanitize_json_data(None) is None

    def test_float_passthrough(self):
        assert sanitize_json_data(3.14) == 3.14

    def test_bool_passthrough(self):
        assert sanitize_json_data(True) is True

    def test_dict_values_sanitized(self):
        result = sanitize_json_data({"key": "val\u0000ue"})
        assert result == {"key": "value"}

    def test_dict_keys_untouched(self):
        # Keys are not sanitized (unusual edge case — keys are identifiers)
        result = sanitize_json_data({"k\u0000": "v"})
        assert "k\u0000" in result

    def test_list_items_sanitized(self):
        result = sanitize_json_data(["\u0000a", "b\u0000", "c"])
        assert result == ["a", "b", "c"]

    def test_nested_dict_in_list(self):
        data = [{"x": "a\u0000b"}, {"y": 1}]
        result = sanitize_json_data(data)
        assert result == [{"x": "ab"}, {"y": 1}]

    def test_deeply_nested_structure(self):
        data = {"level1": {"level2": {"level3": "val\u0000ue"}}}
        result = sanitize_json_data(data)
        assert result == {"level1": {"level2": {"level3": "value"}}}

    def test_mixed_types_in_dict(self):
        data = {"a": "str\u0000ing", "b": 42, "c": None, "d": True}
        result = sanitize_json_data(data)
        assert result == {"a": "string", "b": 42, "c": None, "d": True}

    def test_empty_dict_unchanged(self):
        assert sanitize_json_data({}) == {}

    def test_empty_list_unchanged(self):
        assert sanitize_json_data([]) == []

    def test_empty_string_unchanged(self):
        assert sanitize_json_data("") == ""


# ---------------------------------------------------------------------------
# DatabaseService tests — async, use in-memory SQLite from conftest
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestDatabaseServiceLifecycle:
    """Context manager lifecycle: commit on success, rollback on exception."""

    async def test_clean_exit_commits(self, db_engine):
        """A clean __aexit__ should commit the session (no exception inside)."""

        svc = DatabaseService.__new__(DatabaseService)
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()
        svc.session = mock_session

        await svc.__aexit__(None, None, None)

        mock_session.commit.assert_awaited_once()
        mock_session.rollback.assert_not_awaited()
        mock_session.close.assert_awaited_once()

    async def test_exception_exit_rolls_back(self, db_engine):
        """An __aexit__ with an exception should roll back, not commit."""

        svc = DatabaseService.__new__(DatabaseService)
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()
        svc.session = mock_session

        await svc.__aexit__(ValueError, ValueError("boom"), None)

        mock_session.commit.assert_not_awaited()
        mock_session.rollback.assert_awaited_once()
        mock_session.close.assert_awaited_once()


@pytest.mark.integration
@pytest.mark.asyncio
class TestDatabaseServiceCRUD:
    """Full CRUD operations via DatabaseService, using the conftest db fixtures."""

    def _make_db_service(self, session) -> DatabaseService:
        """Create a DatabaseService backed by the provided test session."""
        svc = DatabaseService.__new__(DatabaseService)
        svc.session = session
        return svc

    async def test_create_and_get_session(self, test_db_session):
        svc = self._make_db_service(test_db_session)
        sid = await svc.create_session(metadata={"run": "1"})
        assert sid is not None

        retrieved = await svc.get_session(sid)
        assert retrieved is not None
        assert retrieved.id == sid
        assert retrieved.status == "running"

    async def test_create_session_with_explicit_id(self, test_db_session):
        import uuid

        explicit_id = str(uuid.uuid4())
        svc = self._make_db_service(test_db_session)
        sid = await svc.create_session(session_id=explicit_id)
        assert sid == explicit_id

    async def test_get_session_not_found(self, test_db_session):
        svc = self._make_db_service(test_db_session)
        result = await svc.get_session("nonexistent-id")
        assert result is None

    async def test_update_session_status_valid(self, test_db_session):
        svc = self._make_db_service(test_db_session)
        sid = await svc.create_session()
        ok = await svc.update_session_status(sid, "completed")
        assert ok is True
        updated = await svc.get_session(sid)
        assert updated.status == "completed"

    async def test_update_session_status_invalid_raises(self, test_db_session):
        svc = self._make_db_service(test_db_session)
        sid = await svc.create_session()
        # The @validates decorator raises ValueError for bad status
        with pytest.raises(ValueError, match="Invalid session status"):
            await svc.update_session_status(sid, "INVALID_STATUS")

    async def test_update_session_status_not_found(self, test_db_session):
        svc = self._make_db_service(test_db_session)
        ok = await svc.update_session_status("does-not-exist", "completed")
        assert ok is False

    async def test_save_graph_output(self, test_db_session):
        svc = self._make_db_service(test_db_session)
        sid = await svc.create_session()
        ok = await svc.save_graph_output(
            sid, "graf1", {"result": "ok"}, step_name="step1"
        )
        assert ok is True

    async def test_save_graph_output_sanitizes_null_bytes(self, test_db_session):
        svc = self._make_db_service(test_db_session)
        sid = await svc.create_session()
        ok = await svc.save_graph_output(sid, "graf1", {"text": "a\u0000b"})
        assert ok is True

    async def test_save_node_output(self, test_db_session):
        svc = self._make_db_service(test_db_session)
        sid = await svc.create_session()
        ok = await svc.save_node_output(sid, "graf1", "my_node", {"output": "data"})
        assert ok is True

    async def test_save_token_usage(self, test_db_session):
        svc = self._make_db_service(test_db_session)
        sid = await svc.create_session()
        ok = await svc.save_token_usage(
            sid,
            "graf1",
            node_name="node1",
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            raw_data={"usage": {"total": 150}},
        )
        assert ok is True

    async def test_save_token_usage_with_null_bytes_in_raw_data(self, test_db_session):
        svc = self._make_db_service(test_db_session)
        sid = await svc.create_session()
        ok = await svc.save_token_usage(
            sid,
            "graf1",
            raw_data={"note": "val\u0000ue"},
        )
        assert ok is True


# ---------------------------------------------------------------------------
# Session model — @validates tests (synchronous, no DB needed)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSessionModelValidation(unittest.TestCase):
    """Ensure @validates on Session.status rejects invalid values."""

    def _make_session_obj(self, status: str) -> Session:
        import uuid

        return Session(id=str(uuid.uuid4()), status=status)

    def test_valid_statuses_accepted(self):
        for status in VALID_SESSION_STATUSES:
            obj = self._make_session_obj(status)
            self.assertEqual(obj.status, status)

    def test_invalid_status_raises(self):
        with self.assertRaises(ValueError):
            self._make_session_obj("UNKNOWN")


# ---------------------------------------------------------------------------
# StorageService — three-mode tests (file, database, dual)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStorageServiceModeValidation(unittest.TestCase):
    """StorageService raises ValueError for invalid modes at construction time."""

    def test_file_mode_accepted(self):
        from black_langcube.storage_service import StorageService

        svc = StorageService("file")
        self.assertEqual(svc.storage_mode, "file")

    def test_database_mode_accepted(self):
        from black_langcube.storage_service import StorageService

        svc = StorageService("database")
        self.assertEqual(svc.storage_mode, "database")

    def test_dual_mode_accepted(self):
        from black_langcube.storage_service import StorageService

        svc = StorageService("dual")
        self.assertEqual(svc.storage_mode, "dual")

    def test_invalid_mode_raises_value_error(self):
        from black_langcube.storage_service import StorageService

        with self.assertRaises(ValueError, msg="Invalid storage mode"):
            StorageService("magic")

    def test_empty_mode_raises_value_error(self):
        from black_langcube.storage_service import StorageService

        with self.assertRaises(ValueError):
            StorageService("")

    def test_default_mode_from_env(self):
        import os

        from black_langcube.storage_service import StorageService

        original = os.environ.pop("STORAGE_MODE", None)
        try:
            svc = StorageService()
            self.assertEqual(svc.storage_mode, "file")
        finally:
            if original is not None:
                os.environ["STORAGE_MODE"] = original


@pytest.mark.integration
@pytest.mark.asyncio
class TestStorageServiceFileModeAsync:
    """file mode: writes to file, skips DB."""

    async def test_file_mode_no_file_path_returns_false(self):
        from black_langcube.storage_service import StorageService

        svc = StorageService("file")
        result = await svc.save_graph_output("sid", "g", {"k": "v"}, file_path=None)
        assert result is False

    async def test_file_mode_writes_file(self, tmp_path):
        from black_langcube.storage_service import StorageService

        svc = StorageService("file")
        fp = str(tmp_path / "out" / "output.json")
        result = await svc.save_graph_output(
            "sid", "g", {"hello": "world"}, file_path=fp
        )
        assert result is True
        import json
        from pathlib import Path

        assert json.loads(Path(fp).read_text()) == {"hello": "world"}


@pytest.mark.integration
@pytest.mark.asyncio
class TestStorageServiceDatabaseMode:
    """database mode: writes only to DB."""

    async def test_save_graph_output_database_mode(self, db_engine):
        from black_langcube.database.operations import DatabaseService
        from black_langcube.storage_service import StorageService

        svc = StorageService("database")

        import uuid

        sid = str(uuid.uuid4())

        # Build a context manager that uses the test engine and creates both
        # the owning session row and the graph_output in the same transaction
        # so that FK constraints are satisfied within a single rollback scope.
        class _TestCM:
            def __init__(self):
                self._svc = None

            async def __aenter__(self):
                self._svc = DatabaseService.__new__(DatabaseService)
                self._svc.session = AsyncSession(db_engine, expire_on_commit=False)
                await self._svc.create_session(session_id=sid)
                return self._svc

            async def __aexit__(self, *a):
                await self._svc.session.rollback()
                await self._svc.session.close()

        svc._StorageService__db_service = lambda: _TestCM()

        result = await svc.save_graph_output(sid, "graf1", {"x": 1}, step_name="s1")
        assert result is True


@pytest.mark.integration
@pytest.mark.asyncio
class TestStorageServiceDualMode:
    """dual mode: DB failure falls back to file write without raising."""

    async def test_dual_db_failure_falls_back_to_file(self, tmp_path):
        from black_langcube.storage_service import StorageService

        svc = StorageService("dual")

        # Patch _db_service to raise an exception
        class _FailingCM:
            async def __aenter__(self):
                raise RuntimeError("DB unavailable")

            async def __aexit__(self, *a):
                pass

        svc._StorageService__db_service = lambda: _FailingCM()

        fp = str(tmp_path / "fallback" / "output.json")
        result = await svc.save_graph_output(
            "sid", "g", {"fallback": True}, file_path=fp
        )
        assert result is True
        import json
        from pathlib import Path

        assert json.loads(Path(fp).read_text()) == {"fallback": True}

    async def test_dual_both_fail_raises(self):
        from black_langcube.storage_service import StorageService

        svc = StorageService("dual")

        class _FailingCM:
            async def __aenter__(self):
                raise RuntimeError("DB unavailable")

            async def __aexit__(self, *a):
                pass

        svc._StorageService__db_service = lambda: _FailingCM()

        # No file_path → file write also returns False → RuntimeError raised
        with pytest.raises(RuntimeError, match="both DB and file writes failed"):
            await svc.save_graph_output("sid", "g", {"k": "v"}, file_path=None)
