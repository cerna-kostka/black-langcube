"""
Tests for validate_session_continuity_async (BP-8).

Covers:
- First node (current_node_id == 1) skips validation entirely
- Valid continuation: session.current_node_id == current_node_id - 1, no exception
- Out-of-order call: session.current_node_id != current_node_id - 1, ValueError raised
- Session not found: ValueError raised with descriptive message

All tests use unittest.mock to isolate the validator from the database layer.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from black_langcube.helper_modules.session_validator import (
    validate_session_continuity_async,
)

_PATCH_TARGET = "black_langcube.helper_modules.session_validator.DatabaseService"


def _make_db_mock(current_node_id: int | None) -> AsyncMock:
    """Return a DatabaseService async-context-manager mock.

    If *current_node_id* is ``None`` the mock's ``get_session`` returns
    ``None`` (session not found).
    """
    session_mock = MagicMock()
    session_mock.current_node_id = current_node_id

    db_mock = AsyncMock()
    db_mock.get_session = AsyncMock(
        return_value=None if current_node_id is None else session_mock
    )
    db_mock.__aenter__ = AsyncMock(return_value=db_mock)
    db_mock.__aexit__ = AsyncMock(return_value=None)
    return db_mock


@pytest.mark.unit
class TestValidateSessionContinuityAsync:
    """Unit tests for validate_session_continuity_async."""

    async def test_first_node_skips_validation(self):
        """current_node_id == 1 must bypass the database entirely."""
        with patch(_PATCH_TARGET) as db_class_mock:
            await validate_session_continuity_async("session-abc", 1)
            db_class_mock.assert_not_called()

    async def test_valid_continuation_no_exception(self):
        """Happy path: session at node 1, requesting node 2 — no exception."""
        db_mock = _make_db_mock(current_node_id=1)
        with patch(_PATCH_TARGET, return_value=db_mock):
            await validate_session_continuity_async("session-abc", 2)

    async def test_valid_continuation_mid_pipeline(self):
        """Happy path: session at node 3, requesting node 4 — no exception."""
        db_mock = _make_db_mock(current_node_id=3)
        with patch(_PATCH_TARGET, return_value=db_mock):
            await validate_session_continuity_async("session-abc", 4)

    async def test_out_of_order_raises_value_error(self):
        """Session at node 1 but caller requests node 3 — raises ValueError."""
        db_mock = _make_db_mock(current_node_id=1)
        with patch(_PATCH_TARGET, return_value=db_mock):
            with pytest.raises(ValueError, match="Session session-abc is at node 1"):
                await validate_session_continuity_async("session-abc", 3)

    async def test_out_of_order_error_mentions_requested_node(self):
        """ValueError message must include the requested node number."""
        db_mock = _make_db_mock(current_node_id=2)
        with patch(_PATCH_TARGET, return_value=db_mock):
            with pytest.raises(ValueError, match="requested node 5"):
                await validate_session_continuity_async("session-abc", 5)

    async def test_duplicate_node_raises_value_error(self):
        """Replaying the same node (session at 2, requesting 2) must raise."""
        db_mock = _make_db_mock(current_node_id=2)
        with patch(_PATCH_TARGET, return_value=db_mock):
            with pytest.raises(ValueError):
                await validate_session_continuity_async("session-abc", 2)

    async def test_session_not_found_raises_value_error(self):
        """A missing session must raise ValueError, not return silently."""
        db_mock = _make_db_mock(current_node_id=None)
        with patch(_PATCH_TARGET, return_value=db_mock):
            with pytest.raises(ValueError, match="Session .* not found"):
                await validate_session_continuity_async("missing-session", 2)

    async def test_error_message_contains_session_id(self):
        """The ValueError message must include the session_id for traceability."""
        db_mock = _make_db_mock(current_node_id=1)
        with patch(_PATCH_TARGET, return_value=db_mock):
            with pytest.raises(ValueError, match="my-unique-session-id"):
                await validate_session_continuity_async("my-unique-session-id", 3)
