"""
Async SQLAlchemy engine and session factory.

Reads DATABASE_URL from the environment and automatically converts it to the
appropriate async dialect URL:
    postgresql://... -> postgresql+asyncpg://...
    sqlite://...     -> sqlite+aiosqlite://...

PostgreSQL connections use a fixed connection pool suitable for production;
SQLite connections skip pool settings because SQLite does not support
connection pooling.
"""

import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./black_langcube.db")

# ---------------------------------------------------------------------------
# Dialect URL rewriting
# ---------------------------------------------------------------------------
_DIALECT_MAP: dict[str, str] = {
    "postgresql://": "postgresql+asyncpg://",
    "postgres://": "postgresql+asyncpg://",
    "sqlite://": "sqlite+aiosqlite://",
}


def _to_async_url(url: str) -> str:
    """Convert a sync SQLAlchemy URL to the corresponding async dialect URL."""
    for sync_prefix, async_prefix in _DIALECT_MAP.items():
        if url.startswith(sync_prefix):
            return async_prefix + url[len(sync_prefix) :]
    return url


_ASYNC_URL: str = _to_async_url(DATABASE_URL)

# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------
_IS_POSTGRES: bool = _ASYNC_URL.startswith("postgresql+asyncpg://")

if _IS_POSTGRES:
    engine = create_async_engine(
        _ASYNC_URL,
        pool_size=10,
        max_overflow=20,
        pool_recycle=3600,
        pool_pre_ping=True,
        echo=False,
    )
else:
    # SQLite does not support connection pooling arguments.
    engine = create_async_engine(
        _ASYNC_URL,
        echo=False,
    )

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------
# expire_on_commit=False prevents lazy-load errors after commit in async context.
async_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)

# ---------------------------------------------------------------------------
# Declarative base — imported by models.py and exposed for migrations
# ---------------------------------------------------------------------------
Base = declarative_base()
