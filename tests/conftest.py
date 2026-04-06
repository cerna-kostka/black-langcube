"""
pytest configuration: autouse fixture that resets all circuit breakers before and
after every test so state cannot bleed between test cases.
"""

import pytest

from black_langcube.llm_modules.circuit_breaker import (
    reset_all_circuit_breakers as reset_sync,
)
from black_langcube.llm_modules.circuit_breaker_async import (
    reset_all_circuit_breakers as reset_async,
)


@pytest.fixture(autouse=True)
def reset_circuit_breakers():
    reset_sync()
    reset_async()
    yield
    reset_sync()
    reset_async()
