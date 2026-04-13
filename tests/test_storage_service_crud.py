"""
Tests for StorageService session CRUD and read operations (BP-26).

Covers:
- create_session / get_session / update_session round-trip for each mode
- get_graph_output returns data previously written by save_graph_output
- get_output_files returns expected list of paths / identifiers
- get_storage_service() returns the same singleton on repeated calls
- Singleton can be replaced via set_storage_service() (dependency injection)
- DatabaseService.update_session merges metadata
- DatabaseService.get_graph_output and get_output_files
"""

import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from black_langcube.database.operations import DatabaseService  # noqa: E402
from black_langcube.storage_service import (  # noqa: E402
    StorageService,
    get_storage_service,
    set_storage_service,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db_service(session: AsyncSession) -> DatabaseService:
    """Return a DatabaseService backed by the provided test session."""
    svc = DatabaseService.__new__(DatabaseService)
    svc.session = session
    return svc


def _new_sid() -> str:
    return str(uuid.uuid4())


def _build_db_svc(shared_session: AsyncSession) -> StorageService:
    """Return a StorageService (database mode) that reuses *shared_session*.

    All context-manager calls share the same session so writes are immediately
    visible to subsequent reads within the same test.
    """
    svc = StorageService("database")

    class _SharedSessionCM:
        def __init__(self_inner):
            self_inner._db = DatabaseService.__new__(DatabaseService)
            self_inner._db.session = shared_session

        async def __aenter__(self_inner):
            return self_inner._db

        async def __aexit__(self_inner, *a):
            # Lifecycle controlled by test_db_session fixture.
            pass

    svc._StorageService__db_service = lambda: _SharedSessionCM()
    return svc


# ---------------------------------------------------------------------------
# DatabaseService — update_session
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
class TestDatabaseServiceUpdateSession:
    """DatabaseService.update_session merges data into metadata_."""

    async def test_update_session_creates_metadata_from_empty(self, test_db_session):
        svc = _make_db_service(test_db_session)
        sid = await svc.create_session()
        ok = await svc.update_session(sid, {"key": "value"})
        assert ok is True
        obj = await svc.get_session(sid)
        assert obj.metadata_["key"] == "value"

    async def test_update_session_merges_with_existing_metadata(self, test_db_session):
        svc = _make_db_service(test_db_session)
        sid = await svc.create_session(metadata={"existing": "data"})
        ok = await svc.update_session(sid, {"new_key": "new_val"})
        assert ok is True
        obj = await svc.get_session(sid)
        assert obj.metadata_["existing"] == "data"
        assert obj.metadata_["new_key"] == "new_val"

    async def test_update_session_overwrites_key(self, test_db_session):
        svc = _make_db_service(test_db_session)
        sid = await svc.create_session(metadata={"k": "old"})
        await svc.update_session(sid, {"k": "new"})
        obj = await svc.get_session(sid)
        assert obj.metadata_["k"] == "new"

    async def test_update_session_not_found_returns_false(self, test_db_session):
        svc = _make_db_service(test_db_session)
        ok = await svc.update_session("nonexistent-id", {"k": "v"})
        assert ok is False


# ---------------------------------------------------------------------------
# DatabaseService — get_graph_output
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
class TestDatabaseServiceGetGraphOutput:
    """DatabaseService.get_graph_output retrieves previously saved output."""

    async def test_returns_saved_data(self, test_db_session):
        svc = _make_db_service(test_db_session)
        sid = await svc.create_session()
        await svc.save_graph_output(sid, "graf1", {"result": 42})
        data = await svc.get_graph_output(sid, "graf1")
        assert data == {"result": 42}

    async def test_returns_none_when_not_found(self, test_db_session):
        svc = _make_db_service(test_db_session)
        sid = await svc.create_session()
        data = await svc.get_graph_output(sid, "nonexistent_graph")
        assert data is None

    async def test_returns_none_for_unknown_session(self, test_db_session):
        svc = _make_db_service(test_db_session)
        data = await svc.get_graph_output("unknown-session", "graf1")
        assert data is None


# ---------------------------------------------------------------------------
# DatabaseService — get_output_files
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
class TestDatabaseServiceGetOutputFiles:
    """DatabaseService.get_output_files returns graph_name identifiers."""

    async def test_returns_graph_names_after_save(self, test_db_session):
        svc = _make_db_service(test_db_session)
        sid = await svc.create_session()
        await svc.save_graph_output(sid, "graf1", {"x": 1})
        await svc.save_graph_output(sid, "graf2", {"x": 2})
        files = await svc.get_output_files(sid)
        assert sorted(files) == ["graf1", "graf2"]

    async def test_returns_empty_list_for_session_without_output(self, test_db_session):
        svc = _make_db_service(test_db_session)
        sid = await svc.create_session()
        files = await svc.get_output_files(sid)
        assert files == []

    async def test_deduplicates_graph_names(self, test_db_session):
        """The UniqueConstraint on (session_id, graph_name, step_name) means
        distinct step_names produce separate rows but the same graph_name.
        get_output_files must return graph_name only once per distinct value.

        Here we use two different step_names to create two rows for the same
        graph_name, then verify only one entry is returned.
        """
        svc = _make_db_service(test_db_session)
        sid = await svc.create_session()
        # Two rows with same graph_name but different step_names
        await svc.save_graph_output(sid, "graf1", {"step": 1}, step_name="step1")
        await svc.save_graph_output(sid, "graf1", {"step": 2}, step_name="step2")
        files = await svc.get_output_files(sid)
        assert files == ["graf1"]


# ---------------------------------------------------------------------------
# StorageService — file mode session CRUD round-trip
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
class TestStorageServiceFileModeSessionCRUD:
    """File mode: create / get / update session round-trip using tmp_path."""

    async def test_create_and_get_session(self, tmp_path):
        svc = StorageService("file", base_dir=str(tmp_path))
        sid = _new_sid()
        ok = await svc.create_session(sid, {"run": "1"})
        assert ok is True
        session = await svc.get_session(sid)
        assert session is not None
        assert session["id"] == sid
        assert session["metadata"]["run"] == "1"

    async def test_get_session_not_found_returns_none(self, tmp_path):
        svc = StorageService("file", base_dir=str(tmp_path))
        result = await svc.get_session("does-not-exist")
        assert result is None

    async def test_update_session_merges_metadata(self, tmp_path):
        svc = StorageService("file", base_dir=str(tmp_path))
        sid = _new_sid()
        await svc.create_session(sid, {"initial": "data"})
        ok = await svc.update_session(sid, {"extra": "value"})
        assert ok is True
        session = await svc.get_session(sid)
        assert session["metadata"]["initial"] == "data"
        assert session["metadata"]["extra"] == "value"

    async def test_update_session_not_found_returns_false(self, tmp_path):
        svc = StorageService("file", base_dir=str(tmp_path))
        ok = await svc.update_session("nonexistent", {"k": "v"})
        assert ok is False


# ---------------------------------------------------------------------------
# StorageService — file mode output retrieval
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
class TestStorageServiceFileModeOutputRetrieval:
    """File mode: get_graph_output reads data written by save_graph_output."""

    async def test_get_graph_output_returns_saved_data(self, tmp_path):
        svc = StorageService("file", base_dir=str(tmp_path))
        sid = _new_sid()
        # Write using the convention path that get_graph_output will read
        fp = svc._graph_output_file_path(sid, "graf1")
        await svc.save_graph_output(sid, "graf1", {"answer": 99}, file_path=fp)
        data = await svc.get_graph_output(sid, "graf1")
        assert data == {"answer": 99}

    async def test_get_graph_output_returns_none_when_missing(self, tmp_path):
        svc = StorageService("file", base_dir=str(tmp_path))
        result = await svc.get_graph_output(_new_sid(), "graf1")
        assert result is None

    async def test_get_output_files_returns_paths_after_write(self, tmp_path):
        svc = StorageService("file", base_dir=str(tmp_path))
        sid = _new_sid()
        fp1 = svc._graph_output_file_path(sid, "graf1")
        fp2 = svc._graph_output_file_path(sid, "graf2")
        await svc.save_graph_output(sid, "graf1", {"x": 1}, file_path=fp1)
        await svc.save_graph_output(sid, "graf2", {"x": 2}, file_path=fp2)
        files = await svc.get_output_files(sid)
        assert len(files) == 2
        assert all(f.endswith(".json") for f in files)

    async def test_get_output_files_returns_empty_for_unknown_session(self, tmp_path):
        svc = StorageService("file", base_dir=str(tmp_path))
        files = await svc.get_output_files(_new_sid())
        assert files == []

    async def test_get_output_files_includes_session_json(self, tmp_path):
        svc = StorageService("file", base_dir=str(tmp_path))
        sid = _new_sid()
        await svc.create_session(sid, {"x": 1})
        files = await svc.get_output_files(sid)
        assert any(f.endswith("session.json") for f in files)


# ---------------------------------------------------------------------------
# StorageService — database mode session CRUD round-trip
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
class TestStorageServiceDatabaseModeSessionCRUD:
    """database mode: session CRUD delegated to DatabaseService."""

    async def test_create_and_get_session(self, test_db_session):
        svc = _build_db_svc(test_db_session)
        sid = _new_sid()
        ok = await svc.create_session(sid, {"run": "db"})
        assert ok is True
        session = await svc.get_session(sid)
        assert session is not None
        assert session["id"] == sid
        assert session["metadata"]["run"] == "db"

    async def test_get_session_not_found_returns_none(self, test_db_session):
        svc = _build_db_svc(test_db_session)
        result = await svc.get_session("no-such-session")
        assert result is None

    async def test_update_session_round_trip(self, test_db_session):
        svc = _build_db_svc(test_db_session)
        sid = _new_sid()
        await svc.create_session(sid, {"initial": True})
        ok = await svc.update_session(sid, {"extra": "added"})
        assert ok is True
        session = await svc.get_session(sid)
        assert session["metadata"]["initial"] is True
        assert session["metadata"]["extra"] == "added"


# ---------------------------------------------------------------------------
# StorageService — database mode output retrieval
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
class TestStorageServiceDatabaseModeOutputRetrieval:
    """database mode: get_graph_output and get_output_files."""

    async def test_get_graph_output_returns_saved_data(self, test_db_session):
        svc = _build_db_svc(test_db_session)
        sid = _new_sid()
        await svc.create_session(sid)
        await svc.save_graph_output(sid, "graf1", {"answer": 7})
        data = await svc.get_graph_output(sid, "graf1")
        assert data == {"answer": 7}

    async def test_get_graph_output_none_when_missing(self, test_db_session):
        svc = _build_db_svc(test_db_session)
        sid = _new_sid()
        await svc.create_session(sid)
        result = await svc.get_graph_output(sid, "nonexistent")
        assert result is None

    async def test_get_output_files_returns_graph_names(self, test_db_session):
        svc = _build_db_svc(test_db_session)
        sid = _new_sid()
        await svc.create_session(sid)
        await svc.save_graph_output(sid, "graf1", {"x": 1})
        await svc.save_graph_output(sid, "graf2", {"x": 2})
        files = await svc.get_output_files(sid)
        assert sorted(files) == ["graf1", "graf2"]

    async def test_get_output_files_empty_for_new_session(self, test_db_session):
        svc = _build_db_svc(test_db_session)
        sid = _new_sid()
        await svc.create_session(sid)
        files = await svc.get_output_files(sid)
        assert files == []


# ---------------------------------------------------------------------------
# StorageService — dual mode session CRUD
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
class TestStorageServiceDualModeSessionCRUD:
    """dual mode: DB failure falls back to file for create and update."""

    @staticmethod
    def _failing_cm():
        class _FailingCM:
            async def __aenter__(self):
                raise RuntimeError("DB unavailable")

            async def __aexit__(self, *a):
                pass

        return _FailingCM()

    async def test_create_session_falls_back_to_file(self, tmp_path):
        svc = StorageService("dual", base_dir=str(tmp_path))
        svc._StorageService__db_service = self._failing_cm
        sid = _new_sid()
        ok = await svc.create_session(sid, {"fallback": True})
        assert ok is True
        session = await svc.get_session(sid)
        assert session["metadata"]["fallback"] is True

    async def test_get_session_falls_back_to_file(self, tmp_path):
        svc = StorageService("dual", base_dir=str(tmp_path))
        sid = _new_sid()
        # Write the file directly using file mode
        file_svc = StorageService("file", base_dir=str(tmp_path))
        await file_svc.create_session(sid, {"source": "file"})

        svc._StorageService__db_service = self._failing_cm
        session = await svc.get_session(sid)
        assert session is not None
        assert session["metadata"]["source"] == "file"

    async def test_update_session_falls_back_to_file(self, tmp_path):
        file_svc = StorageService("file", base_dir=str(tmp_path))
        sid = _new_sid()
        await file_svc.create_session(sid, {"orig": 1})

        svc = StorageService("dual", base_dir=str(tmp_path))
        svc._StorageService__db_service = self._failing_cm
        ok = await svc.update_session(sid, {"added": 2})
        assert ok is True
        session = await file_svc.get_session(sid)
        assert session["metadata"]["orig"] == 1
        assert session["metadata"]["added"] == 2


# ---------------------------------------------------------------------------
# StorageService — dual mode output retrieval fallback
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
class TestStorageServiceDualModeOutputRetrieval:
    """dual mode: DB failure falls back to file for get_graph_output / get_output_files."""

    @staticmethod
    def _failing_cm():
        class _FailingCM:
            async def __aenter__(self):
                raise RuntimeError("DB unavailable")

            async def __aexit__(self, *a):
                pass

        return _FailingCM()

    async def test_get_graph_output_falls_back_to_file(self, tmp_path):
        svc = StorageService("dual", base_dir=str(tmp_path))
        sid = _new_sid()
        fp = svc._graph_output_file_path(sid, "graf1")
        await svc.save_graph_output(sid, "graf1", {"val": 5}, file_path=fp)
        svc._StorageService__db_service = self._failing_cm
        data = await svc.get_graph_output(sid, "graf1")
        assert data == {"val": 5}

    async def test_get_output_files_falls_back_to_file(self, tmp_path):
        svc = StorageService("dual", base_dir=str(tmp_path))
        sid = _new_sid()
        fp = svc._graph_output_file_path(sid, "graf1")
        await svc.save_graph_output(sid, "graf1", {"v": 1}, file_path=fp)
        svc._StorageService__db_service = self._failing_cm
        files = await svc.get_output_files(sid)
        assert len(files) >= 1


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetStorageServiceSingleton:
    """get_storage_service() returns the same instance; set_storage_service() replaces it."""

    def setup_method(self):
        # Reset singleton before each test in this class
        set_storage_service(None)

    def teardown_method(self):
        # Clean up after each test
        set_storage_service(None)

    def test_returns_same_instance_on_repeated_calls(self):
        svc1 = get_storage_service()
        svc2 = get_storage_service()
        assert svc1 is svc2

    def test_returns_storage_service_instance(self):
        svc = get_storage_service()
        assert isinstance(svc, StorageService)

    def test_set_storage_service_replaces_singleton(self):
        custom = StorageService("file")
        set_storage_service(custom)
        assert get_storage_service() is custom

    def test_set_storage_service_none_resets_singleton(self):
        original = get_storage_service()
        set_storage_service(None)
        new_instance = get_storage_service()
        assert new_instance is not original

    def test_dependency_injection_scenario(self):
        """Replacing the singleton does not affect callers that cached the old instance."""
        original = get_storage_service()
        replacement = StorageService("file")
        set_storage_service(replacement)
        assert get_storage_service() is replacement
        assert get_storage_service() is not original
