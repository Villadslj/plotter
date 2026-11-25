"""Dock panels of the 3D view and the particle tracks

:class:`View3DPanel` holds the settings of the ray-traced 3D view.
:class:`ParticlePanel` lists the lost particles and particle tracks drawn on
top of the 2D and 3D views and replays lost particles to record their tracks.
"""

import math
import os

from PySide6.QtCore import QItemSelectionModel, QSettings, Qt, Signal
from PySide6.QtGui import QColor, QTextCursor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QColorDialog,
                               QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGridLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QMenu, QMessageBox, QPlainTextEdit, QPushButton,
                               QSizePolicy, QSlider, QSpinBox, QTableWidget,
                               QTableWidgetItem, QToolButton, QVBoxLayout,
                               QWidget)

from .custom_widgets import HorizontalLine
from .docks import PlotterPanel
from .particles import KIND_LABELS, ReplayRunner, default_openmc_executable
from .render3d import (ORTHOGRAPHIC, PERSPECTIVE, VIEW_PRESET_LABELS,
                       VIEW_PRESET_ORDER, to_rgb)
from .view3d import capability_message

_EXTENDED_TIP = 'Needs OpenMC with the extended ray tracer API'
_SLIDER_STEPS = 1000
_PRESET_TEXT = {'+x': '+X', '-x': '-X', '+y': '+Y', '-y': '-Y', '+z': '+Z',
                '-z': '-Z', 'iso': 'Iso'}
_QUALITY_ITEMS = (('Fast (half resolution)', 'fast'),
                  ('Standard', 'standard'),
                  ('High (HiDPI)', 'high'))
_LIGHT_ITEMS = (('Headlight', 'headlight'), ('Fixed', 'fixed'))
_EXECUTABLE_KEY = 'particles/openmcExecutable'


def _fmt_vector(v, fmt='{:.6g}'):
    if v is None:
        return ''
    return '(' + ', '.join(fmt.format(float(x)) for x in v) + ')'


def _qcolor(rgb):
    r, g, b = to_rgb(rgb)
    return QColor(r, g, b)


def _form_layout():
    layout = QFormLayout()
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
    layout.setFieldGrowthPolicy(
        QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return layout


def _double_spin(minimum, maximum, step, decimals=2, suffix=''):
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setDecimals(decimals)
    if suffix:
        box.setSuffix(suffix)
    box.setKeyboardTracking(False)
    return box


def _select_data(combo, value):
    index = combo.findData(value)
    if index >= 0:
        combo.setCurrentIndex(index)


class _ColorButton(QPushButton):
    """Button showing a color that opens a color picker when clicked"""

    colorChanged = Signal(object)

    def __init__(self, color=(0, 0, 0), parent=None):
        super().__init__(parent)
        self._color = to_rgb(color)
        self.setFixedWidth(48)
        self._updateStyle()
        self.clicked.connect(self._pick)

    def color(self):
        return self._color

    def setColor(self, color):
        self._color = to_rgb(color)
        self._updateStyle()

    def _updateStyle(self):
        self.setStyleSheet('border-radius: 8px; background-color: rgb{}'.format(
            self._color))

    def _pick(self):
        color = QColorDialog.getColor(_qcolor(self._color), self)
        if color.isValid():
            self.setColor(color.getRgb()[:3])
            self.colorChanged.emit(self._color)


class View3DPanel(PlotterPanel):
    """Settings of the ray-traced 3D view

    Parameters
    ----------
    model : openmc_plotter.plotmodel.PlotModel
        Model providing the 3D view state
    font_metric : PySide6.QtGui.QFontMetrics
        Metrics of the application font
    main_window : openmc_plotter.main_window.MainWindow
        Main window of the plotter
    viewport : openmc_plotter.view3d.Viewport3D
        The 3D view
    parent : QWidget, optional
        Parent widget
    """

    def __init__(self, model, font_metric, main_window, viewport,
                 parent=None):
        super().__init__(model, font_metric, main_window, parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Expanding)
        self.viewport = viewport
        self._updating = False
        self._extendedWidgets = []

        self._createHeader()
        self._createCameraBox()
        self._createShadingBox()
        self._createVisibilityBox()
        self._createEdgeBox()
        self._createDepthBox()
        self._createCutBox()
        self._createPerformanceBox()
        self._createOverlayBox()

        contentLayout = QVBoxLayout()
        contentLayout.setContentsMargins(0, 0, 0, 0)
        for box in (self.cameraGroupBox, self.shadingGroupBox,
                    self.visibilityGroupBox, self.edgeGroupBox,
                    self.depthGroupBox, self.cutGroupBox,
                    self.performanceGroupBox, self.overlayGroupBox):
            contentLayout.addWidget(box)
        self.contentWidget = QWidget()
        self.contentWidget.setLayout(contentLayout)

        self.panelLayout = QVBoxLayout()
        self.panelLayout.addLayout(self.headerLayout)
        self.panelLayout.addWidget(HorizontalLine())
        self.panelLayout.addWidget(self.contentWidget)
        self.panelLayout.addStretch()
        self.setLayout(self.panelLayout)

        viewport.stateChanged.connect(self.update)
        viewport.resultChanged.connect(self._updateRenderInfo)
        viewport.busyChanged.connect(self._updateRenderInfo)

        self._applyCapability()
        self.update()

    @property
    def state(self):
        return self.model.view3d

    # Widgets

    def _createHeader(self):
        self.toggleButton = QPushButton('Show 3D View')
        self.toggleButton.setCheckable(True)
        self.toggleButton.setMinimumHeight(
            int(self.font_metric.height() * 1.6))
        self.toggleButton.setToolTip(
            'Switch between the 2D and 3D views (Ctrl+3)')
        self.toggleButton.clicked.connect(
            lambda checked=False: self.main_window.toggle3DView(bool(checked)))

        self.capabilityLabel = QLabel()
        self.capabilityLabel.setWordWrap(True)

        self.headerLayout = QVBoxLayout()
        self.headerLayout.addWidget(self.toggleButton)
        self.headerLayout.addWidget(self.capabilityLabel)

    def _createCameraBox(self):
        self.projectionBox = QComboBox()
        self.projectionBox.addItem('Perspective', PERSPECTIVE)
        self.projectionBox.addItem('Orthographic', ORTHOGRAPHIC)
        self.projectionBox.setToolTip('Projection of the 3D view (O)')
        self.projectionBox.currentIndexChanged.connect(self._onProjection)
        self._extendedWidgets.append(self.projectionBox)

        self.fovBox = _double_spin(5.0, 120.0, 5.0, decimals=0,
                                   suffix='\u00b0')
        self.fovBox.setToolTip(
            'Horizontal field of view of the perspective projection')
        self.fovBox.valueChanged.connect(self._onFov)

        presetLayout = QGridLayout()
        presetLayout.setContentsMargins(0, 0, 0, 0)
        presetLayout.setSpacing(2)
        self.presetButtons = {}
        for i, name in enumerate(VIEW_PRESET_ORDER):
            button = QToolButton()
            button.setText(_PRESET_TEXT[name])
            button.setToolTip('{} ({})'.format(VIEW_PRESET_LABELS[name],
                                               i + 1))
            button.setSizePolicy(QSizePolicy.Policy.Expanding,
                                 QSizePolicy.Policy.Fixed)
            button.clicked.connect(
                lambda checked=False, n=name: self.viewport.setPreset(n))
            presetLayout.addWidget(button, i // 4, i % 4)
            self.presetButtons[name] = button

        self.fitButton = QPushButton('Fit')
        self.fitButton.setToolTip('Show the whole model (F)')
        self.fitButton.clicked.connect(
            lambda checked=False: self.viewport.fit())
        self.resetButton = QPushButton('Reset')
        self.resetButton.setToolTip('Isometric view of the whole model (R)')
        self.resetButton.clicked.connect(
            lambda checked=False: self.viewport.resetCamera())
        buttonLayout = QHBoxLayout()
        buttonLayout.setContentsMargins(0, 0, 0, 0)
        buttonLayout.addWidget(self.fitButton)
        buttonLayout.addWidget(self.resetButton)

        layout = _form_layout()
        layout.addRow('Projection:', self.projectionBox)
        layout.addRow('Field of view:', self.fovBox)
        layout.addRow('View from:', presetLayout)
        layout.addRow(buttonLayout)
        self.cameraGroupBox = QGroupBox('Camera')
        self.cameraGroupBox.setLayout(layout)

    def _createShadingBox(self):
        self.lightBox = QComboBox()
        for label, value in _LIGHT_ITEMS:
            self.lightBox.addItem(label, value)
        self.lightBox.setToolTip(
            'A headlight moves with the camera, a fixed light stays above '
            'the model')
        self.lightBox.currentIndexChanged.connect(self._onLight)

        self.ambientBox = _double_spin(0.0, 1.0, 0.05)
        self.ambientBox.setToolTip(
            'Fraction of the light that illuminates all surfaces evenly')
        self.ambientBox.valueChanged.connect(self._onAmbient)

        self.backgroundButton = _ColorButton()
        self.backgroundButton.setToolTip('Background color of the 3D view')
        self.backgroundButton.colorChanged.connect(self._onBackground)
        self.backgroundSameCheck = QCheckBox('Same as 2D')
        self.backgroundSameCheck.setToolTip(
            'Use the background color of the 2D view')
        self.backgroundSameCheck.toggled.connect(self._onBackgroundSame)
        backgroundLayout = QHBoxLayout()
        backgroundLayout.setContentsMargins(0, 0, 0, 0)
        backgroundLayout.addWidget(self.backgroundButton)
        backgroundLayout.addWidget(self.backgroundSameCheck)
        backgroundLayout.addStretch()
        self.backgroundWidget = QWidget()
        self.backgroundWidget.setLayout(backgroundLayout)
        self._extendedWidgets.append(self.backgroundWidget)

        layout = _form_layout()
        layout.addRow('Light:', self.lightBox)
        layout.addRow('Ambient light:', self.ambientBox)
        layout.addRow('Background:', self.backgroundWidget)
        self.shadingGroupBox = QGroupBox('Shading')
        self.shadingGroupBox.setLayout(layout)

    def _createVisibilityBox(self):
        self.ghostAlphaBox = _double_spin(0.0, 1.0, 0.05)
        self.ghostAlphaBox.setToolTip(
            'Opacity of ghosted cells and materials and, while highlighting '
            'is active, of those that are not highlighted')
        self.ghostAlphaBox.valueChanged.connect(self._onGhostAlpha)
        self._extendedWidgets.append(self.ghostAlphaBox)

        self.colorsButton = QPushButton('Colors && Opacity...')
        self.colorsButton.setToolTip(
            'Edit the colors and opacities of cells and materials')
        self.colorsButton.clicked.connect(
            lambda checked=False: self.main_window.showColorDialog())

        self.overridesLabel = QLabel()
        self.overridesLabel.setWordWrap(True)
        self.showAllButton = QPushButton('Show All')
        self.showAllButton.setToolTip(
            'Show all cells and materials hidden, ghosted or isolated in the '
            '3D view')
        self.showAllButton.clicked.connect(
            lambda checked=False: self.viewport.showAll())
        overridesLayout = QHBoxLayout()
        overridesLayout.setContentsMargins(0, 0, 0, 0)
        overridesLayout.addWidget(self.overridesLabel, 1)
        overridesLayout.addWidget(self.showAllButton)

        self.lostRaysCheck = QCheckBox('Highlight lost rays')
        self.lostRaysCheck.setToolTip(
            'Draw rays that got lost in the geometry in magenta. Lost rays '
            'point to gaps and overlaps.')
        self.lostRaysCheck.toggled.connect(self._onLostRays)
        self._extendedWidgets.append(self.lostRaysCheck)

        hint = QLabel('Right-click the 3D view to hide, ghost or isolate '
                      'cells and materials.')
        hint.setWordWrap(True)

        layout = _form_layout()
        layout.addRow('Ghost opacity:', self.ghostAlphaBox)
        layout.addRow(self.colorsButton)
        layout.addRow(overridesLayout)
        layout.addRow(self.lostRaysCheck)
        layout.addRow(hint)
        self.visibilityGroupBox = QGroupBox('Visibility')
        self.visibilityGroupBox.setLayout(layout)

    def _createEdgeBox(self):
        self.edgeSensitivityBox = _double_spin(0.0, 1.0, 0.05)
        self.edgeSensitivityBox.setToolTip(
            'Higher values find more edges, e.g. between surfaces at similar '
            'depths')
        self.edgeSensitivityBox.valueChanged.connect(
            lambda value: self._setAndRender('edge_sensitivity',
                                             float(value)))
        self.edgeThicknessBox = QSpinBox()
        self.edgeThicknessBox.setRange(1, 4)
        self.edgeThicknessBox.setSuffix(' px')
        self.edgeThicknessBox.setKeyboardTracking(False)
        self.edgeThicknessBox.valueChanged.connect(
            lambda value: self._setAndRender('edge_thickness', int(value)))
        self.edgeColorButton = _ColorButton()
        self.edgeColorButton.colorChanged.connect(
            lambda color: self._setAndRender('edge_color', tuple(color)))

        layout = _form_layout()
        layout.addRow('Sensitivity:', self.edgeSensitivityBox)
        layout.addRow('Thickness:', self.edgeThicknessBox)
        layout.addRow('Color:', self.edgeColorButton)
        self.edgeGroupBox = QGroupBox('Edges')
        self.edgeGroupBox.setCheckable(True)
        self.edgeGroupBox.setToolTip(
            'Outline the boundaries of cells and materials (E)')
        self.edgeGroupBox.toggled.connect(
            lambda checked: self._setAndRender('edges', bool(checked), True))
        self.edgeGroupBox.setLayout(layout)
        self._extendedWidgets.append(self.edgeGroupBox)

    def _createDepthBox(self):
        self.depthStrengthBox = _double_spin(0.0, 1.0, 0.05)
        self.depthStrengthBox.setToolTip(
            'How much surfaces far from the camera are darkened')
        self.depthStrengthBox.valueChanged.connect(
            lambda value: self._setAndRender('depth_strength', float(value)))

        layout = _form_layout()
        layout.addRow('Strength:', self.depthStrengthBox)
        self.depthGroupBox = QGroupBox('Depth Shading')
        self.depthGroupBox.setCheckable(True)
        self.depthGroupBox.setToolTip(
            'Darken surfaces with their distance from the camera (D)')
        self.depthGroupBox.toggled.connect(
            lambda checked: self._setAndRender('depth_shading', bool(checked),
                                               True))
        self.depthGroupBox.setLayout(layout)
        self._extendedWidgets.append(self.depthGroupBox)

    def _createCutBox(self):
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        self.cutWidgets = {}
        for row, axis in enumerate('xyz'):
            check = QCheckBox(axis.upper())
            check.setToolTip(
                'Cut the model with a plane normal to the {}-axis'.format(axis))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, _SLIDER_STEPS)
            slider.setToolTip('Position of the cut plane in the model')
            spin = _double_spin(-1e9, 1e9, 1.0, decimals=4)
            spin.setToolTip('Position of the cut plane in [cm]')
            flip = QToolButton()
            flip.setText('Flip')
            flip.setCheckable(True)
            flip.setToolTip('Remove the part of the model below the plane '
                            'instead of the part above it')

            check.toggled.connect(
                lambda checked, a=axis: self._onCutEnabled(a, checked))
            slider.valueChanged.connect(
                lambda value, a=axis: self._onCutSlider(a, value))
            slider.sliderReleased.connect(
                lambda a=axis: self._onCutReleased(a))
            spin.valueChanged.connect(
                lambda value, a=axis: self._onCutSpin(a, value))
            flip.toggled.connect(
                lambda checked, a=axis: self._onCutFlip(a, checked))

            grid.addWidget(check, row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(spin, row, 2)
            grid.addWidget(flip, row, 3)
            self.cutWidgets[axis] = (check, slider, spin, flip)
        grid.setColumnStretch(1, 1)

        self.cutFacesCheck = QCheckBox('Show cut faces')
        self.cutFacesCheck.setToolTip(
            'Draw the cut surfaces of cells and materials as solid faces. '
            'Otherwise cut cells are open and their inside is visible.')
        self.cutFacesCheck.toggled.connect(
            lambda checked: self._setAndRender('show_cut_faces',
                                               bool(checked)))
        self.clearCutsButton = QPushButton('Remove All')
        self.clearCutsButton.setToolTip('Disable all cut planes')
        self.clearCutsButton.clicked.connect(
            lambda checked=False: self.viewport.clearCutPlanes())
        bottomLayout = QHBoxLayout()
        bottomLayout.setContentsMargins(0, 0, 0, 0)
        bottomLayout.addWidget(self.cutFacesCheck)
        bottomLayout.addStretch()
        bottomLayout.addWidget(self.clearCutsButton)

        layout = QVBoxLayout()
        layout.addLayout(grid)
        layout.addLayout(bottomLayout)
        self.cutGroupBox = QGroupBox('Cut Planes')
        self.cutGroupBox.setLayout(layout)
        self._extendedWidgets.append(self.cutGroupBox)

    def _createPerformanceBox(self):
        self.qualityBox = QComboBox()
        for label, value in _QUALITY_ITEMS:
            self.qualityBox.addItem(label, value)
        self.qualityBox.setToolTip(
            'Resolution of the final image. While the view changes, previews '
            'are rendered at a lower resolution.')
        self.qualityBox.currentIndexChanged.connect(self._onQuality)

        self.previewBox = QSpinBox()
        self.previewBox.setRange(10, 2000)
        self.previewBox.setSingleStep(10)
        self.previewBox.setSuffix(' ms')
        self.previewBox.setKeyboardTracking(False)
        self.previewBox.setToolTip(
            'Target render time of the previews shown while rotating, '
            'panning or zooming. Lower values make the view more responsive '
            'at a lower resolution.')
        self.previewBox.valueChanged.connect(self._onPreviewTime)

        self.renderInfoLabel = QLabel()
        self.renderInfoLabel.setWordWrap(True)

        layout = _form_layout()
        layout.addRow('Quality:', self.qualityBox)
        layout.addRow('Preview time:', self.previewBox)
        layout.addRow('Last image:', self.renderInfoLabel)
        self.performanceGroupBox = QGroupBox('Performance')
        self.performanceGroupBox.setLayout(layout)

    def _createOverlayBox(self):
        self.tracksCheck = QCheckBox('Particle tracks')
        self.tracksCheck.setToolTip(
            'Draw the loaded particle tracks on top of the 3D view (T)')
        self.tracksCheck.toggled.connect(
            lambda checked: self._onOverlay('show_tracks', checked))
        self.axesCheck = QCheckBox('Axes')
        self.axesCheck.setToolTip('Show the orientation of the x, y and z axes')
        self.axesCheck.toggled.connect(
            lambda checked: self._onOverlay('show_axes', checked))

        layout = QHBoxLayout()
        layout.addWidget(self.tracksCheck)
        layout.addWidget(self.axesCheck)
        layout.addStretch()
        self.overlayGroupBox = QGroupBox('Overlays')
        self.overlayGroupBox.setLayout(layout)

    def _applyCapability(self):
        capability = self.viewport.capability
        available = self.viewport.available
        extended = self.viewport.renderer.extended
        self.capabilityLabel.setText(capability_message(capability))
        self.toggleButton.setEnabled(available)
        self.contentWidget.setEnabled(available)
        for widget in self._extendedWidgets:
            widget.setEnabled(extended)
            if not extended:
                widget.setToolTip(_EXTENDED_TIP)

    # Updates from the model

    def updateMode(self, in3d):
        """Show whether the 3D view is active"""
        self.toggleButton.setChecked(bool(in3d))
        self.toggleButton.setText('Back to 2D View' if in3d
                                  else 'Show 3D View')

    def update(self):
        state = self.state
        cam = state.camera
        self._updating = True
        try:
            _select_data(self.projectionBox, cam.projection)
            self.fovBox.setValue(cam.fov)
            self.fovBox.setEnabled(not cam.is_orthographic)
            _select_data(self.lightBox, state.light)
            self.ambientBox.setValue(state.diffuse_fraction)
            self._updateBackground()
            self.ghostAlphaBox.setValue(state.ghost_alpha)
            self.lostRaysCheck.setChecked(bool(state.highlight_lost_rays))
            self.edgeGroupBox.setChecked(bool(state.edges))
            self.edgeSensitivityBox.setValue(state.edge_sensitivity)
            self.edgeThicknessBox.setValue(int(state.edge_thickness))
            self.edgeColorButton.setColor(state.edge_color)
            self.depthGroupBox.setChecked(bool(state.depth_shading))
            self.depthStrengthBox.setValue(state.depth_strength)
            self.cutFacesCheck.setChecked(bool(state.show_cut_faces))
            _select_data(self.qualityBox, state.quality)
            self.previewBox.setValue(int(round(state.interactive_ms)))
            self.tracksCheck.setChecked(bool(state.show_tracks))
            self.axesCheck.setChecked(bool(state.show_axes))
        finally:
            self._updating = False
        self._updateCutWidgets()
        self._updateOverrides()
        self._updateRenderInfo()

    def _updateBackground(self):
        background = self.state.background
        same = background is None
        self.backgroundSameCheck.setChecked(same)
        if same:
            color = to_rgb(getattr(self.model.currentView, 'domainBackground',
                                   None), default=(50, 50, 50))
        else:
            color = background
        self.backgroundButton.setColor(color)
        self.backgroundButton.setEnabled(not same)

    def _updateOverrides(self):
        state = self.state
        kinds = ('cell', 'material')
        parts = []
        hidden = sum(len(state.hidden[k]) for k in kinds)
        ghosted = sum(len(state.ghosted[k]) for k in kinds)
        if hidden:
            parts.append('{} hidden'.format(hidden))
        if ghosted:
            parts.append('{} ghosted'.format(ghosted))
        for kind in kinds:
            isolated = state.isolated[kind]
            if isolated is not None:
                parts.append('{} {}{} isolated'.format(
                    len(isolated), kind, '' if len(isolated) == 1 else 's'))
        if parts:
            self.overridesLabel.setText(', '.join(parts))
        else:
            self.overridesLabel.setText('All cells and materials shown')
        self.showAllButton.setEnabled(bool(parts))

    def _updateRenderInfo(self, _=None):
        viewport = self.viewport
        result = viewport.result
        lines = []
        if result is not None:
            text = '{}\u00d7{} px, {:.0f} ms'.format(
                result.width, result.height, 1000.0 * result.seconds)
            if not result.final:
                final = viewport.manager.final_scale or 1.0
                text += ', preview {:.0f}%'.format(
                    100.0 * result.scale / final)
            lines.append(text)
            if result.n_lost_rays:
                lines.append('{} lost ray{}'.format(
                    result.n_lost_rays,
                    '' if result.n_lost_rays == 1 else 's'))
        if viewport.manager.is_busy:
            lines.append('Rendering...')
        self.renderInfoLabel.setText('\n'.join(lines) if lines else 'None')

    # Cut planes

    def _cutRange(self, axis):
        i = 'xyz'.index(axis)
        lo = float(self.state.scene_lower_left[i])
        hi = float(self.state.scene_upper_right[i])
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            center = 0.5 * (lo + hi) if math.isfinite(lo + hi) else 0.0
            lo, hi = center - 1.0, center + 1.0
        return lo, hi

    def _sliderToPosition(self, axis, value):
        lo, hi = self._cutRange(axis)
        return lo + (hi - lo) * value / _SLIDER_STEPS

    def _positionToSlider(self, axis, position):
        lo, hi = self._cutRange(axis)
        value = int(round((position - lo) / (hi - lo) * _SLIDER_STEPS))
        return min(max(value, 0), _SLIDER_STEPS)

    def _updateCutWidgets(self):
        updating = self._updating
        self._updating = True
        try:
            any_enabled = False
            for axis, (check, slider, spin, flip) in self.cutWidgets.items():
                plane = self.state.cut_planes[axis]
                position = float(plane['position'])
                lo, hi = self._cutRange(axis)
                span = hi - lo
                check.setChecked(bool(plane['enabled']))
                if not slider.isSliderDown():
                    slider.setValue(self._positionToSlider(axis, position))
                spin.setRange(min(lo - 0.5 * span, position),
                              max(hi + 0.5 * span, position))
                spin.setSingleStep(max(0.01 * span, 1e-4))
                spin.setValue(position)
                flip.setChecked(bool(plane['flip']))
                any_enabled = any_enabled or bool(plane['enabled'])
            self.clearCutsButton.setEnabled(any_enabled)
        finally:
            self._updating = updating

    def _onCutEnabled(self, axis, checked):
        if self._updating:
            return
        plane = self.state.cut_planes[axis]
        plane['enabled'] = bool(checked)
        if checked:
            lo, hi = self._cutRange(axis)
            if not lo <= float(plane['position']) <= hi:
                plane['position'] = 0.5 * (lo + hi)
        self._updateCutWidgets()
        self.viewport.requestRender()

    def _onCutSlider(self, axis, value):
        if self._updating:
            return
        plane = self.state.cut_planes[axis]
        plane['position'] = self._sliderToPosition(axis, value)
        plane['enabled'] = True
        self._updateCutWidgets()
        slider = self.cutWidgets[axis][1]
        self.viewport.requestRender(interactive=slider.isSliderDown())

    def _onCutReleased(self, axis):
        if self.state.cut_planes[axis]['enabled']:
            self.viewport.requestRender()

    def _onCutSpin(self, axis, value):
        if self._updating:
            return
        plane = self.state.cut_planes[axis]
        plane['position'] = float(value)
        plane['enabled'] = True
        self._updateCutWidgets()
        self.viewport.requestRender()

    def _onCutFlip(self, axis, checked):
        if self._updating:
            return
        plane = self.state.cut_planes[axis]
        plane['flip'] = bool(checked)
        if plane['enabled']:
            self.viewport.requestRender()

    # Other settings

    def _setAndRender(self, attribute, value, emit=False):
        if self._updating:
            return
        setattr(self.state, attribute, value)
        self.viewport.settingsChanged()
        if emit:
            self.viewport.stateChanged.emit()

    def _onProjection(self, index):
        if self._updating:
            return
        projection = self.projectionBox.itemData(index)
        if projection and projection != self.state.camera.projection:
            self.viewport.setProjection(projection)

    def _onFov(self, value):
        if self._updating:
            return
        self.state.camera.set_fov(value)
        self.viewport.settingsChanged()

    def _onLight(self, index):
        light = self.lightBox.itemData(index)
        if light:
            self._setAndRender('light', light)

    def _onAmbient(self, value):
        self._setAndRender('diffuse_fraction', float(value))

    def _onBackground(self, color):
        self._setAndRender('background', tuple(color))

    def _onBackgroundSame(self, checked):
        if self._updating:
            return
        self.state.background = None if checked else \
            self.backgroundButton.color()
        self._updating = True
        try:
            self._updateBackground()
        finally:
            self._updating = False
        self.viewport.settingsChanged()

    def _onGhostAlpha(self, value):
        if self._updating:
            return
        self.state.ghost_alpha = float(value)
        self.viewport.domainStylesChanged()

    def _onLostRays(self, checked):
        self._setAndRender('highlight_lost_rays', bool(checked))

    def _onQuality(self, index):
        quality = self.qualityBox.itemData(index)
        if quality:
            self._setAndRender('quality', quality)

    def _onPreviewTime(self, value):
        if self._updating:
            return
        self.state.interactive_ms = float(value)

    def _onOverlay(self, attribute, checked):
        if self._updating:
            return
        setattr(self.state, attribute, bool(checked))
        self.viewport.settingsChanged(False)
        self.viewport.stateChanged.emit()


class ParticlePanel(PlotterPanel):
    """Lost particles and particle tracks shown in the 2D and 3D views

    Parameters
    ----------
    model : openmc_plotter.plotmodel.PlotModel
        Model holding the particles and their display options
    font_metric : PySide6.QtGui.QFontMetrics
        Metrics of the application font
    main_window : openmc_plotter.main_window.MainWindow
        Main window of the plotter
    parent : QWidget, optional
        Parent widget
    """

    COLUMNS = ('Show', 'Color', 'ID', 'Batch', 'Type', 'Kind', 'States',
               'Lost At')
    SHOW_COLUMN = 0
    COLOR_COLUMN = 1

    def __init__(self, model, font_metric, main_window, parent=None):
        super().__init__(model, font_metric, main_window, parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Expanding)
        self._updating = False
        self._cancelling = False

        self.runner = ReplayRunner(self)
        self.runner.finished.connect(self._onReplayFinished)
        self.runner.failed.connect(self._onReplayFailed)
        self.runner.output.connect(self._onReplayOutput)

        self._createFileButtons()
        self._createTable()
        self._createSelectionButtons()
        self._createReplayBox()
        self._createOptionsBox()

        self.panelLayout = QVBoxLayout()
        self.panelLayout.addWidget(self.hintLabel)
        self.panelLayout.addLayout(self.fileLayout)
        self.panelLayout.addWidget(self.table, 3)
        self.panelLayout.addWidget(self.detailsText, 2)
        self.panelLayout.addLayout(self.selectionLayout)
        self.panelLayout.addWidget(self.replayGroupBox)
        self.panelLayout.addWidget(self.optionsGroupBox)
        self.setLayout(self.panelLayout)

        self.update()

    @property
    def particles(self):
        return self.model.particleTracks

    @property
    def viewport(self):
        return getattr(self.main_window, 'viewport3d', None)

    # Widgets

    def _createFileButtons(self):
        self.hintLabel = QLabel(
            'Load the particle_*.h5 files that OpenMC writes for lost '
            'particles, or track files written with openmc -t.')
        self.hintLabel.setWordWrap(True)

        self.openFilesButton = QPushButton('Open Files...')
        self.openFilesButton.setToolTip(
            'Load particle restart or track files (Ctrl+Shift+L)')
        self.openFilesButton.clicked.connect(
            lambda checked=False: self.main_window.openParticleFiles())
        self.openFolderButton = QPushButton('Open Folder...')
        self.openFolderButton.setToolTip(
            'Load all particle_*.h5 files of a folder')
        self.openFolderButton.clicked.connect(
            lambda checked=False: self.main_window.openParticleFolder())
        self.removeButton = QPushButton('Remove')
        self.removeButton.setToolTip('Remove the selected particles')
        self.removeButton.clicked.connect(
            lambda checked=False: self.removeSelected())
        self.clearButton = QPushButton('Clear')
        self.clearButton.setToolTip('Remove all particles')
        self.clearButton.clicked.connect(
            lambda checked=False: self.main_window.clearParticleTracks())

        self.fileLayout = QGridLayout()
        self.fileLayout.addWidget(self.openFilesButton, 0, 0)
        self.fileLayout.addWidget(self.openFolderButton, 0, 1)
        self.fileLayout.addWidget(self.removeButton, 1, 0)
        self.fileLayout.addWidget(self.clearButton, 1, 1)

    def _createTable(self):
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.table.setMinimumHeight(int(self.font_metric.height() * 10))
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._showTableMenu)
        self.table.itemChanged.connect(self._onItemChanged)
        self.table.itemSelectionChanged.connect(self._onSelectionChanged)
        self.table.cellDoubleClicked.connect(self._onCellDoubleClicked)

        self.detailsText = QPlainTextEdit()
        self.detailsText.setReadOnly(True)
        self.detailsText.setMinimumHeight(int(self.font_metric.height() * 6))

    def _createSelectionButtons(self):
        self.show2DButton = QPushButton('Show in 2D')
        self.show2DButton.setToolTip(
            'Center the 2D view on where the selected particle was lost')
        self.show2DButton.clicked.connect(
            lambda checked=False: self._showSelected(False))
        self.show3DButton = QPushButton('Show in 3D')
        self.show3DButton.setToolTip(
            'Center the 3D view on where the selected particle was lost')
        self.show3DButton.clicked.connect(
            lambda checked=False: self._showSelected(True))

        self.cutButton = QPushButton('Cut 3D View')
        self.cutButton.setToolTip(
            'Add a cut plane to the 3D view through where the selected '
            'particle was lost')
        self.cutMenu = QMenu(self.cutButton)
        self.cutMenu.addAction('Facing the Camera').triggered.connect(
            lambda checked=False: self._cutSelected(None))
        for axis in 'xyz':
            self.cutMenu.addAction('Normal to {}'.format(
                axis.upper())).triggered.connect(
                lambda checked=False, a=axis: self._cutSelected(a))
        self.cutButton.setMenu(self.cutMenu)

        self.showOnlyButton = QPushButton('Show Only Selected')
        self.showOnlyButton.clicked.connect(
            lambda checked=False: self.showOnlySelected())
        self.showAllButton = QPushButton('Show All')
        self.showAllButton.clicked.connect(
            lambda checked=False: self.showAll())

        self.selectionLayout = QGridLayout()
        self.selectionLayout.addWidget(self.show2DButton, 0, 0)
        self.selectionLayout.addWidget(self.show3DButton, 0, 1)
        self.selectionLayout.addWidget(self.cutButton, 1, 0, 1, 2)
        self.selectionLayout.addWidget(self.showOnlyButton, 2, 0)
        self.selectionLayout.addWidget(self.showAllButton, 2, 1)

    def _createReplayBox(self):
        self.executableEdit = QLineEdit()
        self.executableEdit.setPlaceholderText(
            default_openmc_executable() or 'openmc')
        self.executableEdit.setToolTip(
            'OpenMC executable used to replay lost particles. Defaults to '
            'openmc on the PATH.')
        self.executableEdit.setText(
            str(QSettings().value(_EXECUTABLE_KEY, '') or ''))
        self.executableEdit.editingFinished.connect(self._saveExecutable)
        self.browseButton = QPushButton('Browse...')
        self.browseButton.clicked.connect(
            lambda checked=False: self._browseExecutable())
        executableLayout = QHBoxLayout()
        executableLayout.setContentsMargins(0, 0, 0, 0)
        executableLayout.addWidget(self.executableEdit, 1)
        executableLayout.addWidget(self.browseButton)

        self.threadsBox = QSpinBox()
        self.threadsBox.setRange(1, max(1, os.cpu_count() or 1))
        self.threadsBox.setToolTip('Number of OpenMP threads of the replay')
        self.threadsBox.valueChanged.connect(self._onThreads)

        self.replayButton = QPushButton('Replay Selected')
        self.replayButton.setToolTip(
            'Transport the selected lost particle again with '
            'openmc -r <file> -t to record its full track')
        self.replayButton.clicked.connect(
            lambda checked=False: self.replaySelected())
        self.cancelButton = QPushButton('Cancel')
        self.cancelButton.clicked.connect(
            lambda checked=False: self.cancelReplay())
        buttonLayout = QHBoxLayout()
        buttonLayout.setContentsMargins(0, 0, 0, 0)
        buttonLayout.addWidget(self.replayButton)
        buttonLayout.addWidget(self.cancelButton)

        self.replayStatusLabel = QLabel()
        self.replayStatusLabel.setWordWrap(True)
        self.replayLog = QPlainTextEdit()
        self.replayLog.setReadOnly(True)
        self.replayLog.setMaximumBlockCount(300)
        self.replayLog.setMinimumHeight(int(self.font_metric.height() * 6))
        self.replayLog.setVisible(False)

        layout = _form_layout()
        layout.addRow('OpenMC:', executableLayout)
        layout.addRow('Threads:', self.threadsBox)
        layout.addRow(buttonLayout)
        layout.addRow(self.replayStatusLabel)
        layout.addRow(self.replayLog)
        self.replayGroupBox = QGroupBox('Replay')
        self.replayGroupBox.setToolTip(
            'Files without the steps before the loss can be replayed to '
            'record the full track of the particle')
        self.replayGroupBox.setLayout(layout)

    def _createOptionsBox(self):
        self.show2DCheck = QCheckBox('2D view')
        self.show2DCheck.toggled.connect(
            lambda checked: self._setOption('show2d', bool(checked)))
        self.show3DCheck = QCheckBox('3D view')
        self.show3DCheck.toggled.connect(self._onShow3D)
        showLayout = QHBoxLayout()
        showLayout.setContentsMargins(0, 0, 0, 0)
        showLayout.addWidget(self.show2DCheck)
        showLayout.addWidget(self.show3DCheck)
        showLayout.addStretch()

        self.markersCheck = QCheckBox('Birth and loss markers')
        self.markersCheck.toggled.connect(
            lambda checked: self._setOption('showMarkers', bool(checked)))
        self.stepsCheck = QCheckBox('Step markers')
        self.stepsCheck.toggled.connect(
            lambda checked: self._setOption('showSteps', bool(checked)))
        self.labelsCheck = QCheckBox('Labels')
        self.labelsCheck.toggled.connect(
            lambda checked: self._setOption('showLabels', bool(checked)))
        self.xrayCheck = QCheckBox('Show hidden parts dashed (3D)')
        self.xrayCheck.setToolTip(
            'Draw the parts of tracks hidden behind the geometry as dashed '
            'lines instead of hiding them')
        self.xrayCheck.toggled.connect(
            lambda checked: self._setOption('xray', bool(checked)))

        self.toleranceBox = _double_spin(0.0, 10000.0, 0.1, decimals=3,
                                         suffix=' cm')
        self.toleranceBox.setSpecialValueText('Auto')
        self.toleranceBox.setToolTip(
            'Tracks closer to the 2D slice than this are drawn solid, other '
            'parts dashed and faded. Auto uses 1% of the view width.')
        self.toleranceBox.valueChanged.connect(
            lambda value: self._setOption('sliceTolerance', float(value)))
        self.lineWidthBox = _double_spin(0.5, 8.0, 0.5, decimals=1,
                                         suffix=' px')
        self.lineWidthBox.valueChanged.connect(
            lambda value: self._setOption('lineWidth', float(value)))

        layout = _form_layout()
        layout.addRow('Show in:', showLayout)
        layout.addRow(self.markersCheck)
        layout.addRow(self.stepsCheck)
        layout.addRow(self.labelsCheck)
        layout.addRow(self.xrayCheck)
        layout.addRow('Slice tolerance:', self.toleranceBox)
        layout.addRow('Line width:', self.lineWidthBox)
        self.optionsGroupBox = QGroupBox('Display')
        self.optionsGroupBox.setLayout(layout)

    # Updates from the model

    def update(self):
        self.updateTable()
        self.updateOptions()

    def updateTable(self):
        selected = self.selectedParticles()
        particles = self.particles
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        self._updating = True
        try:
            self.table.setRowCount(len(particles))
            for row, particle in enumerate(particles):
                tooltip = particle.describe()
                show = QTableWidgetItem()
                show.setFlags(flags | Qt.ItemFlag.ItemIsUserCheckable)
                show.setCheckState(Qt.CheckState.Checked if particle.visible
                                   else Qt.CheckState.Unchecked)
                show.setToolTip('Show the particle in the 2D and 3D views')
                self.table.setItem(row, self.SHOW_COLUMN, show)

                color = QTableWidgetItem()
                color.setFlags(flags)
                color.setData(Qt.ItemDataRole.DecorationRole,
                              _qcolor(particle.color))
                color.setToolTip('Double-click to change the color')
                self.table.setItem(row, self.COLOR_COLUMN, color)

                values = (
                    '?' if particle.particle_id is None
                    else particle.particle_id,
                    '' if particle.batch is None else particle.batch,
                    particle.particle,
                    KIND_LABELS.get(particle.kind, particle.kind),
                    particle.n_states,
                    _fmt_vector(particle.loss_position, '{:.4g}'))
                for column, value in enumerate(values, start=2):
                    item = QTableWidgetItem(str(value))
                    item.setFlags(flags)
                    item.setToolTip(tooltip)
                    self.table.setItem(row, column, item)
        finally:
            self._updating = False
        self._selectParticles(selected)
        self._onSelectionChanged()

    def updateOptions(self):
        options = self.model.particleOptions
        viewport = self.viewport
        available = viewport is not None and viewport.available
        self._updating = True
        try:
            self.show2DCheck.setChecked(bool(options.get('show2d', True)))
            self.show3DCheck.setChecked(bool(self.model.view3d.show_tracks))
            self.markersCheck.setChecked(bool(options.get('showMarkers',
                                                          True)))
            self.stepsCheck.setChecked(bool(options.get('showSteps', False)))
            self.labelsCheck.setChecked(bool(options.get('showLabels', True)))
            self.xrayCheck.setChecked(bool(options.get('xray', True)))
            self.toleranceBox.setValue(float(options.get('sliceTolerance',
                                                         0.0)))
            self.lineWidthBox.setValue(float(options.get('lineWidth', 1.5)))
            self.threadsBox.setValue(int(options.get('threads', 1)))
        finally:
            self._updating = False
        self.show3DCheck.setEnabled(available)
        self.xrayCheck.setEnabled(available and viewport.renderer.extended)

    # Selection

    def selectedRows(self):
        n = len(self.particles)
        rows = {index.row() for index in
                self.table.selectionModel().selectedRows()}
        return sorted(r for r in rows if r < n)

    def selectedParticles(self):
        particles = self.particles
        return [particles[r] for r in self.selectedRows()]

    def _selectParticles(self, particles):
        selection = self.table.selectionModel()
        wanted = {id(p) for p in particles}
        flags = (QItemSelectionModel.SelectionFlag.Select |
                 QItemSelectionModel.SelectionFlag.Rows)
        selection.clearSelection()
        for row, particle in enumerate(self.particles):
            if id(particle) in wanted:
                selection.select(self.table.model().index(row, 0), flags)

    def selectParticle(self, particle):
        """Select a particle in the table and show its details"""
        self._selectParticles([particle])
        for row, p in enumerate(self.particles):
            if p is particle:
                self.table.scrollToItem(self.table.item(row, 0))
                break

    def _onSelectionChanged(self):
        particles = self.selectedParticles()
        if len(particles) == 1:
            self.detailsText.setPlainText(particles[0].describe())
        elif particles:
            self.detailsText.setPlainText(
                '{} particles selected'.format(len(particles)))
        elif self.particles:
            self.detailsText.setPlainText(
                'Select a particle to show its details. Double-click it to '
                'show it in the current view.')
        else:
            self.detailsText.setPlainText('No particles loaded.')
        self._updateButtons()

    def _updateButtons(self):
        particles = self.selectedParticles()
        single = particles[0] if len(particles) == 1 else None
        located = single is not None and single.key_position() is not None
        viewport = self.viewport
        available = viewport is not None and viewport.available
        extended = available and viewport.renderer.extended
        running = self.runner.running

        self.removeButton.setEnabled(bool(particles))
        self.clearButton.setEnabled(bool(self.particles))
        self.show2DButton.setEnabled(located)
        self.show3DButton.setEnabled(located and available)
        self.cutButton.setEnabled(located and extended)
        if not extended and available:
            self.cutButton.setToolTip(_EXTENDED_TIP)
        self.showOnlyButton.setEnabled(bool(particles))
        self.showAllButton.setEnabled(
            any(not p.visible for p in self.particles))
        self.replayButton.setEnabled(
            single is not None and single.can_replay and not running)
        self.cancelButton.setEnabled(running)

    def _onItemChanged(self, item):
        if self._updating or item.column() != self.SHOW_COLUMN:
            return
        row = item.row()
        if 0 <= row < len(self.particles):
            self.particles[row].visible = \
                item.checkState() == Qt.CheckState.Checked
            self._updateButtons()
            self.main_window.particlesChanged(refresh_panel=False)

    def _onCellDoubleClicked(self, row, column):
        if not 0 <= row < len(self.particles):
            return
        particle = self.particles[row]
        if column == self.COLOR_COLUMN:
            self.editColor(particle)
        elif column != self.SHOW_COLUMN and \
                particle.key_position() is not None:
            self.main_window.showParticle(particle)

    def _showTableMenu(self, pos):
        row = self.table.rowAt(pos.y())
        if row < 0:
            return
        if row not in self.selectedRows():
            self._selectParticles([self.particles[row]])
        self._updateButtons()
        particles = self.selectedParticles()

        menu = QMenu(self)
        for button in (self.show2DButton, self.show3DButton):
            action = menu.addAction(button.text())
            action.setEnabled(button.isEnabled())
            action.triggered.connect(
                lambda checked=False, b=button: b.click())
        cut = menu.addMenu('Cut 3D View')
        cut.setEnabled(self.cutButton.isEnabled())
        for action in self.cutMenu.actions():
            cut.addAction(action)
        menu.addSeparator()
        replay = menu.addAction('Replay')
        replay.setEnabled(self.replayButton.isEnabled())
        replay.triggered.connect(lambda checked=False: self.replaySelected())
        color = menu.addAction('Change Color...')
        color.setEnabled(len(particles) == 1)
        color.triggered.connect(
            lambda checked=False: self.editColor(particles[0]))
        menu.addSeparator()
        menu.addAction('Show Only Selected').triggered.connect(
            lambda checked=False: self.showOnlySelected())
        menu.addAction('Remove').triggered.connect(
            lambda checked=False: self.removeSelected())
        menu.exec(self.table.viewport().mapToGlobal(pos))

    # Actions

    def _showSelected(self, in3d):
        particles = self.selectedParticles()
        if particles:
            self.main_window.showParticle(particles[0], in3d=in3d)

    def _cutSelected(self, axis):
        particles = self.selectedParticles()
        if particles:
            self.main_window.cutThroughParticle(particles[0], axis)

    def editColor(self, particle):
        color = QColorDialog.getColor(_qcolor(particle.color), self,
                                      'Track Color')
        if color.isValid():
            particle.color = tuple(color.getRgb()[:3])
            self.main_window.particlesChanged()

    def showOnlySelected(self):
        selected = {id(p) for p in self.selectedParticles()}
        if not selected:
            return
        for particle in self.particles:
            particle.visible = id(particle) in selected
        self.main_window.particlesChanged()

    def showAll(self):
        for particle in self.particles:
            particle.visible = True
        self.main_window.particlesChanged()

    def removeSelected(self):
        removed = {id(p) for p in self.selectedParticles()}
        if not removed:
            return
        if self.runner.running and id(self.runner.particle) in removed:
            self.cancelReplay()
        self.particles[:] = [p for p in self.particles if id(p) not in removed]
        self.main_window.particlesChanged()

    # Options

    def _setOption(self, key, value):
        if self._updating:
            return
        self.model.particleOptions[key] = value
        self.main_window.particlesChanged(refresh_panel=False)

    def _onShow3D(self, checked):
        if self._updating:
            return
        self.model.view3d.show_tracks = bool(checked)
        viewport = self.viewport
        if viewport is not None:
            viewport.settingsChanged(False)
            viewport.stateChanged.emit()

    def _onThreads(self, value):
        if not self._updating:
            self.model.particleOptions['threads'] = int(value)

    # Replay

    def _saveExecutable(self):
        QSettings().setValue(_EXECUTABLE_KEY,
                             self.executableEdit.text().strip())

    def _browseExecutable(self):
        start = self.executableEdit.text().strip() or \
            default_openmc_executable()
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select the OpenMC Executable', start)
        if path:
            self.executableEdit.setText(path)
            self._saveExecutable()

    def replaySelected(self):
        """Replay the selected lost particle to record its track"""
        particles = self.selectedParticles()
        if len(particles) != 1 or not particles[0].can_replay or \
                self.runner.running:
            return
        particle = particles[0]
        self._cancelling = False
        self.replayLog.clear()
        self.replayLog.setVisible(True)
        self.replayStatusLabel.setText('Replaying {}...'.format(
            particle.summary()))
        try:
            self.runner.start(particle,
                              getattr(self.main_window, 'model_path', None),
                              self.executableEdit.text().strip() or None,
                              int(self.model.particleOptions.get('threads',
                                                                 1)))
        except (RuntimeError, ValueError) as e:
            self.replayStatusLabel.setText(str(e))
        self._updateButtons()

    def cancelReplay(self):
        if self.runner.running:
            self._cancelling = True
            self.replayStatusLabel.setText('Cancelling the replay...')
            self.runner.cancel()

    def shutdown(self):
        """Stop a running replay, e.g. when the plotter is closed"""
        self._cancelling = True
        self.runner.shutdown()

    def _onReplayOutput(self, text):
        self.replayLog.moveCursor(QTextCursor.MoveOperation.End)
        self.replayLog.insertPlainText(text)
        self.replayLog.moveCursor(QTextCursor.MoveOperation.End)

    def _onReplayFinished(self, result):
        original = self.runner.particle
        particles = self.particles
        for i, particle in enumerate(particles):
            if particle is original:
                particles[i] = result
                break
        else:
            particles.append(result)
        self.replayStatusLabel.setText(
            'Replayed {}: {} track{} with {} states.'.format(
                result.summary(), len(result.tracks),
                '' if len(result.tracks) == 1 else 's', result.n_states))
        self.main_window.particlesChanged()
        self.selectParticle(result)
        self._updateButtons()

    def _onReplayFailed(self, message):
        cancelled = self._cancelling
        self._cancelling = False
        if cancelled:
            self.replayStatusLabel.setText('The replay was cancelled.')
        else:
            self.replayStatusLabel.setText('The replay failed.')
            QMessageBox.warning(self, 'Replay Failed', message)
        self._updateButtons()
