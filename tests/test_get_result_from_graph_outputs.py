"""
Tests for get_result_from_graph_outputs_async and
get_simple_result_from_graph_outputs_async (BP-29).

Covers:
- File path taken when subfolder_name is set (storage_service ignored).
- DB path taken when subfolder_name is None and storage_service is provided.
- Error dict returned when both subfolder_name and storage_service are absent.
- Exceptions raised by storage_service propagate to the caller.
- Backward-compatible: existing positional-only callers still work.
"""

import pytest
from unittest.mock import AsyncMock, patch

from black_langcube.helper_modules.get_result_from_graph_outputs import (
    get_result_from_graph_outputs_async,
    get_simple_result_from_graph_outputs_async,
)

_FILE_FUNC = "black_langcube.helper_modules.get_result_from_graph_outputs.get_result_from_graph_outputs"
_SIMPLE_FILE_FUNC = "black_langcube.helper_modules.get_result_from_graph_outputs.get_simple_result_from_graph_outputs"


def _make_storage_service(return_value):
    """Return a mock storage_service whose get_graph_output resolves to *return_value*."""
    svc = AsyncMock()
    svc.get_graph_output = AsyncMock(return_value=return_value)
    return svc


@pytest.mark.unit
class TestGetResultFromGraphOutputsAsync:
    """Unit tests for get_result_from_graph_outputs_async."""

    async def test_file_path_used_when_subfolder_set(self):
        """When subfolder_name is set the file reader is called; storage_service is ignored."""
        expected = {"answer": "from_file"}
        with patch(_FILE_FUNC, return_value=expected) as mock_fn:
            svc = _make_storage_service("should_not_be_returned")
            result = await get_result_from_graph_outputs_async(
                "k1",
                "k2",
                "sk",
                "ssk",
                "some/folder",
                "out.json",
                storage_service=svc,
            )
        mock_fn.assert_called_once_with(
            "k1", "k2", "sk", "ssk", "some/folder", "out.json"
        )
        svc.get_graph_output.assert_not_called()
        assert result == expected

    async def test_file_path_used_without_storage_service(self):
        """File path works normally when storage_service is not supplied."""
        expected = "plain_result"
        with patch(_FILE_FUNC, return_value=expected):
            result = await get_result_from_graph_outputs_async(
                "k1", "k2", "sk", "ssk", "folder", "file.json"
            )
        assert result == expected

    async def test_db_path_taken_when_subfolder_none(self):
        """When subfolder_name is None the storage_service is awaited."""
        db_result = {"data": "from_db"}
        svc = _make_storage_service(db_result)
        with patch(_FILE_FUNC) as mock_fn:
            result = await get_result_from_graph_outputs_async(
                "k1",
                "k2",
                "sk",
                "ssk",
                None,
                "graph.json",
                storage_service=svc,
            )
        mock_fn.assert_not_called()
        svc.get_graph_output.assert_awaited_once_with("graph.json")
        assert result == db_result

    async def test_error_dict_when_no_subfolder_and_no_service(self):
        """Returns an error dict when both subfolder_name and storage_service are absent."""
        result = await get_result_from_graph_outputs_async(
            "k1", "k2", "sk", "ssk", None, "graph.json"
        )
        assert result == {"error": "subfolder_name is not set."}

    async def test_storage_service_error_propagates(self):
        """Exceptions raised by storage_service.get_graph_output bubble up."""
        svc = AsyncMock()
        svc.get_graph_output = AsyncMock(side_effect=RuntimeError("db error"))
        with pytest.raises(RuntimeError, match="db error"):
            await get_result_from_graph_outputs_async(
                "k1",
                "k2",
                "sk",
                "ssk",
                None,
                "graph.json",
                storage_service=svc,
            )

    async def test_backward_compatible_positional_call(self):
        """All-positional call (no storage_service) remains unchanged."""
        expected = "compat_result"
        with patch(_FILE_FUNC, return_value=expected):
            result = await get_result_from_graph_outputs_async(
                "k1", "k2", "sk", "ssk", "folder", "file.json"
            )
        assert result == expected


@pytest.mark.unit
class TestGetSimpleResultFromGraphOutputsAsync:
    """Unit tests for get_simple_result_from_graph_outputs_async."""

    async def test_file_path_used_when_subfolder_set(self):
        """When subfolder_name is set the simple file reader is called; storage_service is ignored."""
        expected = "simple_from_file"
        with patch(_SIMPLE_FILE_FUNC, return_value=expected) as mock_fn:
            svc = _make_storage_service("ignored")
            result = await get_simple_result_from_graph_outputs_async(
                "my_key",
                "some/folder",
                "out.json",
                storage_service=svc,
            )
        mock_fn.assert_called_once_with("my_key", "some/folder", "out.json")
        svc.get_graph_output.assert_not_called()
        assert result == expected

    async def test_file_path_used_without_storage_service(self):
        """Simple file path works normally without storage_service."""
        expected = "plain_simple"
        with patch(_SIMPLE_FILE_FUNC, return_value=expected):
            result = await get_simple_result_from_graph_outputs_async(
                "key", "folder", "file.json"
            )
        assert result == expected

    async def test_db_path_taken_when_subfolder_none(self):
        """When subfolder_name is None the storage_service is awaited."""
        db_result = "from_db_simple"
        svc = _make_storage_service(db_result)
        with patch(_SIMPLE_FILE_FUNC) as mock_fn:
            result = await get_simple_result_from_graph_outputs_async(
                "key",
                None,
                "graph.json",
                storage_service=svc,
            )
        mock_fn.assert_not_called()
        svc.get_graph_output.assert_awaited_once_with("graph.json")
        assert result == db_result

    async def test_error_dict_when_no_subfolder_and_no_service(self):
        """Returns an error dict when both subfolder_name and storage_service are absent."""
        result = await get_simple_result_from_graph_outputs_async(
            "key", None, "graph.json"
        )
        assert result == {"error": "subfolder_name is not set."}

    async def test_storage_service_error_propagates(self):
        """Exceptions raised by storage_service.get_graph_output bubble up."""
        svc = AsyncMock()
        svc.get_graph_output = AsyncMock(side_effect=ValueError("not found"))
        with pytest.raises(ValueError, match="not found"):
            await get_simple_result_from_graph_outputs_async(
                "key",
                None,
                "graph.json",
                storage_service=svc,
            )

    async def test_backward_compatible_positional_call(self):
        """All-positional call (no storage_service) remains unchanged."""
        expected = "compat_simple"
        with patch(_SIMPLE_FILE_FUNC, return_value=expected):
            result = await get_simple_result_from_graph_outputs_async(
                "key", "folder", "file.json"
            )
        assert result == expected
