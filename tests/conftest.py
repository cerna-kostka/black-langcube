"""
pytest configuration: autouse fixture that resets all circuit breakers before and
after every test so state cannot bleed between test cases.

Database fixtures
-----------------
Set DATABASE_URL to an in-memory SQLite database *before* importing any
black_langcube.database modules so that the config module (which reads the env
var at import time) picks up the in-memory URL.  The fixtures here create the
schema and provide a per-test async session with guaranteed rollback in the
``finally`` block.
"""

import os

# Must be set before importing black_langcube.database so that database/config.py
# reads the in-memory URL at import time.
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import StaticPool

from black_langcube.llm_modules.circuit_breaker import (
    reset_all_circuit_breakers as reset_sync,
)
from black_langcube.llm_modules.circuit_breaker_async import (
    reset_all_circuit_breakers as reset_async,
)

# ---------------------------------------------------------------------------
# Circuit-breaker reset (runs for every test)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_circuit_breakers():
    reset_sync()
    reset_async()
    yield
    reset_sync()
    reset_async()


# ---------------------------------------------------------------------------
# In-memory SQLite engine — shared across the test session
# ---------------------------------------------------------------------------
# StaticPool keeps a single connection alive for the whole process so every
# async session sees the same in-memory database.

_TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="session")
async def db_engine():
    """Create a test-only async engine backed by an in-memory SQLite database."""
    from black_langcube.database.models import Base

    engine = create_async_engine(
        _TEST_DB_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture()
async def db_session(db_engine):
    """Yield a fresh AsyncSession for one test; always roll back in ``finally``."""
    session = AsyncSession(db_engine, expire_on_commit=False)
    try:
        yield session
    finally:
        await session.rollback()
        await session.close()
