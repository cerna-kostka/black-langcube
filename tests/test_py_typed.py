"""
Regression test: assert that py.typed is shipped with the installed package
(PEP 561 compliance).  The marker must be present in the installed
site-packages tree so that mypy / pyright can discover inline type annotations
without requiring separate stub files.
"""

import sys
import unittest
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


@pytest.mark.unit
class TestPyTypedMarker(unittest.TestCase):
    """Ensure py.typed exists in the installed black_langcube package tree."""

    def test_py_typed_exists_via_importlib_resources(self):
        """py.typed is accessible through importlib.resources (installed tree)."""
        import black_langcube

        package_path = Path(black_langcube.__file__).parent
        marker = package_path / "py.typed"
        self.assertTrue(
            marker.exists(),
            f"py.typed marker not found in installed package at {package_path}",
        )

    def test_py_typed_is_empty_file(self):
        """py.typed must be an empty file per PEP 561."""
        import black_langcube

        package_path = Path(black_langcube.__file__).parent
        marker = package_path / "py.typed"
        self.assertTrue(marker.is_file(), "py.typed must be a regular file")
        self.assertEqual(
            marker.stat().st_size,
            0,
            "py.typed must be an empty file (PEP 561 §5)",
        )

    def test_py_typed_exists_in_source_tree(self):
        """py.typed exists in src/black_langcube/ within the repository."""
        source_marker = src_path / "black_langcube" / "py.typed"
        self.assertTrue(
            source_marker.exists(),
            f"py.typed not found in source tree at {source_marker}",
        )
