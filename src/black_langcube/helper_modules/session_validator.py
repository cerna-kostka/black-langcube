"""
Session continuity validator for multi-step pipeline orchestration.

Prevents replay attacks, duplicate processing, and out-of-order API calls
when the library is consumed in a web application context.
"""

from black_langcube.database.operations import DatabaseService

_NOT_FOUND_CODE = 404
_OUT_OF_ORDER_CODE = 409


async def validate_session_continuity_async(
    session_id: str,
    current_node_id: int,
    *,
    raise_on_error: bool = True,
) -> None | dict[str, str | int]:
    """Verify that *current_node_id* is the logically expected next node.

    Skips validation for ``current_node_id == 1`` (the first node has no
    predecessor to check).  For all subsequent nodes the function fetches
    the session record and either raises :exc:`ValueError` or returns a
    structured error dict, depending on *raise_on_error*.

    Args:
        session_id: UUID string identifying the session to validate.
        current_node_id: The node the caller is requesting to execute next.
        raise_on_error: When ``True`` (default) a :exc:`ValueError` is raised
            on any validation failure, preserving the original behaviour.
            When ``False`` the function returns a dict on failure and ``None``
            on success, allowing callers to handle errors without try/except.

    Returns:
        ``None`` when validation passes (or ``current_node_id == 1``).
        When *raise_on_error* is ``False`` and validation fails, returns a
        dict with two keys:

        * ``"error"`` — human-readable description of the failure.
        * ``"code"`` — integer that mirrors an HTTP status code:
          ``404`` when the session is not found,
          ``409`` when the node sequence is out of order.

    Raises:
        ValueError: If the session cannot be found, or if the session's stored
            ``current_node_id`` is not ``current_node_id - 1``.  Only raised
            when *raise_on_error* is ``True`` (the default).
    """
    if current_node_id == 1:
        return None

    async with DatabaseService() as db:
        session = await db.get_session(session_id)

        if session is None:
            msg = f"Session {session_id} not found."
            if raise_on_error:
                raise ValueError(msg)
            return {"error": msg, "code": _NOT_FOUND_CODE}

        if session.current_node_id != current_node_id - 1:
            msg = (
                f"Session {session_id} is at node {session.current_node_id}, "
                f"but caller requested node {current_node_id}."
            )
            if raise_on_error:
                raise ValueError(msg)
            return {"error": msg, "code": _OUT_OF_ORDER_CODE}

    return None
