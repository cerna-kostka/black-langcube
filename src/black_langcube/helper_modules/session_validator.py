"""
Session continuity validator for multi-step pipeline orchestration.

Prevents replay attacks, duplicate processing, and out-of-order API calls
when the library is consumed in a web application context.
"""

from black_langcube.database.operations import DatabaseService


async def validate_session_continuity_async(
    session_id: str, current_node_id: int
) -> None:
    """Verify that *current_node_id* is the logically expected next node.

    Skips validation for ``current_node_id == 1`` (the first node has no
    predecessor to check).  For all subsequent nodes the function fetches
    the session record and raises :exc:`ValueError` if the stored
    ``current_node_id`` is not exactly ``current_node_id - 1``.

    Args:
        session_id: UUID string identifying the session to validate.
        current_node_id: The node the caller is requesting to execute next.

    Raises:
        ValueError: If the session cannot be found, or if the session's stored
            ``current_node_id`` is not ``current_node_id - 1``.
    """
    if current_node_id == 1:
        return

    async with DatabaseService() as db:
        session = await db.get_session(session_id)

        if session is None:
            raise ValueError(f"Session {session_id} not found.")

        if session.current_node_id != current_node_id - 1:
            raise ValueError(
                f"Session {session_id} is at node {session.current_node_id}, "
                f"but caller requested node {current_node_id}."
            )
