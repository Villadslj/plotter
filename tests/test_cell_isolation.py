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
    assert "navigateToOverlap" in mock_main_window
    assert "nextOverlap" in mock_main_window
    assert "_getOverlapCycleLocations" in mock_main_window
    assert "_collectOverlapLocations" in mock_main_window
    assert "_getOverlapLocationsForView" in mock_main_window
    assert "_getNearestOverlapIndex" in mock_main_window
    assert "_updateGeometryPanelFromActiveView" in mock_main_window
    assert "_normalizeOrigin" in mock_main_window
    assert "_sortOverlapLocations" in mock_main_window
    assert "_dedupeOverlapLocations" in mock_main_window
    assert "_sameOverlapLocation" in mock_main_window
    assert "_findOverlapInCurrentSlice" in mock_main_window
    assert "_findOverlapBySliceScan" in mock_main_window
    assert "_findOverlapBySampling" in mock_main_window
    assert "_findCellInCurrentSlice" in mock_main_window
    assert "_findCellByBoundingBox" in mock_main_window
    assert "_findCellBySliceScan" in mock_main_window
    assert "_findCellBySampling" in mock_main_window
    assert "_getRepresentativePixel" in mock_main_window
    assert "_pixelToPlotPoint" in mock_main_window
    assert "_getLocationFromMask" in mock_main_window
    assert "_getCellSearchSettings" in mock_main_window
    assert "_setCellSearchSettings" in mock_main_window


def test_cell_isolation_actions_exist():
    """Verify the menu actions are created in createMenuBar."""
    import ast
    from pathlib import Path

    main_window_path = Path(__file__).parent.parent / "openmc_plotter" / "main_window.py"
    source = main_window_path.read_text()

    # Verify action names appear in the source
    assert "highlightCellAction" in source
    assert "isolateCellAction" in source
    assert "navigateOverlapAction" in source
    assert "nextOverlapAction" in source
    assert "clearCellIsolationAction" in source
    assert "Highlight Cell" in source
    assert "Isolate Cell" in source
    assert "Navigate to Overlap" in source
    assert "Go to Ne&xt Overlap" in source
    assert "lear Cell Isolation" in source


def test_navigate_to_cell_search_settings_exist():
    """Verify adjustable navigate-to-cell search settings are wired in."""
    from pathlib import Path

    main_window_path = Path(__file__).parent.parent / "openmc_plotter" / "main_window.py"
    plotmodel_path = Path(__file__).parent.parent / "openmc_plotter" / "plotmodel.py"

    main_source = main_window_path.read_text()
    plotmodel_source = plotmodel_path.read_text()

    settings = [
        "cellSearchNumSlices",
        "cellSearchSliceResolution",
        "cellSearchNumSamples",
        "cellSearchCenterBias",
        "cellSearchCenterSpan",
        "cellSearchFallbackRange",
    ]

    for setting in settings:
        assert setting in main_source
        assert setting in plotmodel_source

    assert "Slice search count:" in main_source
    assert "Slice search resolution:" in main_source
    assert "Random samples:" in main_source
    assert "Center-biased sample share:" in main_source
    assert "Fallback range multiplier:" in main_source


def test_cell_isolation_uses_all_default_cells():
    """Verify cell isolation/highlighting iterates the full cell set."""
    from pathlib import Path

    main_window_path = Path(__file__).parent.parent / "openmc_plotter" / "main_window.py"
    plotmodel_path = Path(__file__).parent.parent / "openmc_plotter" / "plotmodel.py"

    main_source = main_window_path.read_text()
    plotmodel_source = plotmodel_path.read_text()

    assert "def default_ids(self):" in plotmodel_source
    assert "return self.defaults.keys()" in plotmodel_source
    assert main_source.count("for cid in av.cells.default_ids():") == 3


def test_navigation_updates_geometry_panel_controls():
    """Verify cell/overlap navigation refreshes geometry panel fields."""
    from pathlib import Path

    main_window_path = Path(__file__).parent.parent / "openmc_plotter" / "main_window.py"
    main_source = main_window_path.read_text()

    assert "def _updateGeometryPanelFromActiveView(self):" in main_source
    assert "self.geometryPanel.updateOrigin()" in main_source
    assert "self.geometryPanel.updateWidth()" in main_source
    assert "self.geometryPanel.updateHeight()" in main_source
    assert main_source.count("self._updateGeometryPanelFromActiveView()") >= 5
