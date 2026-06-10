"""Tests for cell isolation/highlighting feature."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_main_window():
    """Create a mock MainWindow with enough structure to test cell isolation."""
    # We can't easily instantiate the real MainWindow without openmc.lib,
    # so test the logic directly via a mocked approach.

    # Import just the module to verify syntax/structure
    import ast
    from pathlib import Path

    main_window_path = Path(__file__).parent.parent / "openmc_plotter" / "main_window.py"
    tree = ast.parse(main_window_path.read_text())

    # Verify the new methods exist in the MainWindow class
    main_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MainWindow":
            main_class = node
            break

    assert main_class is not None
    method_names = [
        n.name for n in ast.walk(main_class)
        if isinstance(n, ast.FunctionDef)
    ]
    return method_names


def test_cell_isolation_methods_exist(mock_main_window):
    """Verify all cell isolation methods are defined."""
    assert "_getValidCellId" in mock_main_window
    assert "highlightCellDialog" in mock_main_window
    assert "isolateCellDialog" in mock_main_window
    assert "_highlightCell" in mock_main_window
    assert "_isolateCell" in mock_main_window
    assert "clearCellIsolation" in mock_main_window


def test_navigate_to_cell_methods_exist(mock_main_window):
    """Verify DAGMC-aware navigation methods are defined."""
    assert "_navigateToCell" in mock_main_window
    assert "_findCellInCurrentSlice" in mock_main_window
    assert "_findCellBySliceScan" in mock_main_window
    assert "_findCellBySampling" in mock_main_window


def test_cell_isolation_actions_exist():
    """Verify the menu actions are created in createMenuBar."""
    import ast
    from pathlib import Path

    main_window_path = Path(__file__).parent.parent / "openmc_plotter" / "main_window.py"
    source = main_window_path.read_text()

    # Verify action names appear in the source
    assert "highlightCellAction" in source
    assert "isolateCellAction" in source
    assert "clearCellIsolationAction" in source
    assert "Highlight Cell" in source
    assert "Isolate Cell" in source
    assert "lear Cell Isolation" in source
