"""
Tests for BP-3: thread session_id through BaseGraph and GraphState.

Covers:
- BaseGraph raises ValueError when both folder_name and session_id are None
- BaseGraph instantiates with only session_id (database mode)
- BaseGraph instantiates with only folder_name (existing file mode)
- intro_info_check() passes for each valid combination and raises for the invalid one
- graph_name is injected into the initial state passed to astream
- get_subfolder() returns None when folder_name is None
- write_events_to_file() returns None when folder_name is None
- GraphState contains a graph_name field
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from black_langcube.graf.graph_base import BaseGraph, GraphState  # noqa: E402


def _make_graph(folder_name=None, session_id=None, user_message="hello"):
    """Return a minimal concrete BaseGraph subclass instance."""

    class _TestGraph(BaseGraph):
        @property
        def workflow_name(self):
            return "test_graph"

    return _TestGraph(
        GraphState, user_message, folder_name=folder_name, session_id=session_id
    )


@pytest.mark.unit
class TestBaseGraphInit(unittest.TestCase):
    """Constructor validation: folder_name / session_id combinations."""

    def test_raises_when_neither_folder_name_nor_session_id(self):
        with self.assertRaises(ValueError):
            _make_graph(folder_name=None, session_id=None)

    def test_instantiates_with_only_folder_name(self):
        g = _make_graph(folder_name="results/session")
        self.assertEqual(g.folder_name, "results/session")
        self.assertIsNone(g.session_id)

    def test_instantiates_with_only_session_id(self):
        uid = "550e8400-e29b-41d4-a716-446655440000"
        g = _make_graph(session_id=uid)
        self.assertEqual(g.session_id, uid)
        self.assertIsNone(g.folder_name)

    def test_instantiates_with_both(self):
        uid = "550e8400-e29b-41d4-a716-446655440000"
        g = _make_graph(folder_name="results/session", session_id=uid)
        self.assertEqual(g.folder_name, "results/session")
        self.assertEqual(g.session_id, uid)

    def test_session_id_stored(self):
        uid = "550e8400-e29b-41d4-a716-446655440001"
        g = _make_graph(session_id=uid)
        self.assertEqual(g.session_id, uid)


@pytest.mark.unit
class TestIntroInfoCheck(unittest.TestCase):
    """intro_info_check validation."""

    def test_passes_with_folder_name_only(self):
        g = _make_graph(folder_name="results/session")
        g.intro_info_check()  # must not raise

    def test_passes_with_session_id_only(self):
        g = _make_graph(session_id="abc-123")
        g.intro_info_check()  # must not raise

    def test_passes_with_both(self):
        g = _make_graph(folder_name="results/session", session_id="abc-123")
        g.intro_info_check()  # must not raise

    def test_raises_when_only_user_message_missing(self):
        g = _make_graph(folder_name="results/session", user_message="")
        with self.assertRaises(ValueError):
            g.intro_info_check()

    def test_raises_when_folder_and_session_both_cleared(self):
        g = _make_graph(folder_name="results/session")
        # Simulate state where both become absent after construction
        g.folder_name = None
        g.session_id = None
        with self.assertRaises(ValueError):
            g.intro_info_check()


@pytest.mark.unit
class TestGetSubfolder(unittest.TestCase):
    """get_subfolder() returns None in database mode."""

    def test_returns_path_when_folder_name_set(self):
        g = _make_graph(folder_name="results/session")
        result = g.get_subfolder()
        self.assertIsNotNone(result)
        self.assertEqual(result, Path("results/session") / "test_graph")

    def test_returns_none_when_folder_name_absent(self):
        g = _make_graph(session_id="abc-123")
        result = g.get_subfolder()
        self.assertIsNone(result)


@pytest.mark.unit
class TestWriteEventsToFile(unittest.IsolatedAsyncioTestCase):
    """write_events_to_file() returns None and skips I/O in database mode."""

    async def test_returns_none_when_no_folder_name(self):
        g = _make_graph(session_id="abc-123")
        result = await g.write_events_to_file([{"some": "event"}], "output.json")
        self.assertIsNone(result)


@pytest.mark.unit
class TestGraphNameInjection(unittest.IsolatedAsyncioTestCase):
    """graph_name is injected into the state passed to astream."""

    async def test_graph_name_in_state_passed_to_astream(self):
        g = _make_graph(folder_name="results/session")

        captured = {}

        async def mock_astream(state, config):
            captured.update(state)
            return
            yield  # make it an async generator

        mock_graph = MagicMock()
        mock_graph.astream = mock_astream
        g._graph = mock_graph

        await g.graph_streaming({"question": "test question"})

        self.assertIn("graph_name", captured)
        self.assertEqual(captured["graph_name"], "test_graph")

    async def test_graph_name_does_not_mutate_original_state(self):
        g = _make_graph(folder_name="results/session")

        original_state = {"question": "test"}

        async def mock_astream(state, config):
            return
            yield

        mock_graph = MagicMock()
        mock_graph.astream = mock_astream
        g._graph = mock_graph

        await g.graph_streaming(original_state)

        self.assertNotIn("graph_name", original_state)


@pytest.mark.unit
class TestGraphStateHasGraphName(unittest.TestCase):
    """GraphState TypedDict includes graph_name field."""

    def test_graph_name_in_graphstate_annotations(self):
        self.assertIn("graph_name", GraphState.__annotations__)

    def test_graph_state_accepts_graph_name(self):
        state: GraphState = {
            "folder_name": "results/session",
            "session_id": "abc-123",
            "graph_name": "test_graph",
        }
        self.assertEqual(state["graph_name"], "test_graph")


if __name__ == "__main__":
    unittest.main()
