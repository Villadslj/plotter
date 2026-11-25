"""Qt widgets of the ray-traced 3D view

:class:`Render3DManager` renders the 3D view in a background thread with a
dynamic resolution. While the user interacts with the view, images are
rendered at a reduced resolution chosen so that a frame takes about
:attr:`View3DState.interactive_ms`. Once the view stops changing, the image is
refined up to the final resolution. :class:`Viewport3D` displays the images,
handles the mouse and keyboard and draws particle tracks and an axis triad on
top of the rendered image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import (QEventLoop, QObject, QPointF, QRectF, QSize, Qt,
                            QThread, QTimer, Signal, Slot)
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QApplication, QInputDialog, QMenu, QSizePolicy,
                               QToolTip, QWidget)

from .particles import BIRTH_COLOR, LOSS_COLOR
from .plotmodel import OPENMC_LIB_LOCK, lib_generation
from .render3d import (CAPABILITY_BASIC, CAPABILITY_NONE, LOST_RAY_ID,
                       ORTHOGRAPHIC, PERSPECTIVE, QUALITY_SCALES,
                       VIEW_PRESET_LABELS, VIEW_PRESET_ORDER, Renderer3D,
                       RenderRequest, box_from_2d_view, domain_kind,
                       domain_styles, finite_bounding_box, pixel_footprint,
                       point_visibility, to_rgb, track_overlay)

# Smallest preview resolution relative to the final resolution
MIN_DRAFT_SCALE = 1.0 / 16.0
# Preview resolutions are multiples of this step
SCALE_STEP = 1.0 / 32.0
# Edges are not drawn on previews rendered below this scale
EDGE_MIN_SCALE = 0.35
# Delay before an interactive preview is refined
REFINE_DELAY_MS = 200

_DRAG_THRESHOLD = 4
_HOVER_RADIUS = 8.0
_ARROW_PIXELS = 36.0

_UNAVAILABLE_MESSAGE = (
    'The 3D view needs a version of OpenMC that provides '
    'openmc.lib.SolidRayTracePlot.')
_BASIC_MESSAGE = (
    'Basic ray tracer: orthographic views, cut planes, transparency, edge '
    'highlighting, depth shading and picking need OpenMC with the extended '
    'ray tracer API.')


def _qcolor(rgb, alpha=255):
    r, g, b = (int(c) for c in tuple(rgb)[:3])
    return QColor(r, g, b, int(alpha))


def _fmt_point(p):
    return '({:.6g}, {:.6g}, {:.6g})'.format(*(float(x) for x in p))


class _RenderWorker(QObject):
    """Renders requests in the thread of the :class:`Render3DManager`"""
    finished = Signal(object)
    failed = Signal(object, str)

    def __init__(self, renderer):
        super().__init__()
        self.renderer = renderer

    @Slot(object)
    def render(self, request):
        try:
            result = self.renderer.render(request)
        except Exception as exc:
            self.failed.emit(request, '{}: {}'.format(type(exc).__name__, exc))
        else:
            self.finished.emit(result)


class Render3DManager(QObject):
    """Schedules renders of the 3D view with a dynamic resolution

    One image is rendered at a time. Requests made while an image is being
    rendered replace each other, so that only the latest view is rendered
    next. The time per ray of past renders is used to pick the resolution of
    previews.

    Parameters
    ----------
    renderer : openmc_plotter.render3d.Renderer3D
        Renderer used in the background thread
    make_request : callable
        Called with the scale and whether the image is final when a render
        starts. Returns a :class:`RenderRequest` for the latest state of the
        view or None if there is nothing to render.
    parent : QObject, optional
        Parent object
    """
    resultReady = Signal(object)
    renderFailed = Signal(str)
    busyChanged = Signal(bool)
    _requestRender = Signal(object)

    def __init__(self, renderer, make_request, parent=None):
        super().__init__(parent)
        self.renderer = renderer
        self._make_request = make_request

        # Scale of the final image and time per preview frame
        self.final_scale = 1.0
        self.target_seconds = 0.06
        # Number of rays of an image at scale 1
        self.base_rays = 1

        self._serial = 0
        self._epoch = 0
        self._displayed = -1
        self._in_flight = None
        self._pending = None
        self._refine = []
        self._refine_delay = 0
        self._seconds_per_ray = None
        self._busy = False

        self._refine_timer = QTimer(self)
        self._refine_timer.setSingleShot(True)
        self._refine_timer.timeout.connect(self._refine_next)

        self._thread = QThread()
        self._worker = _RenderWorker(renderer)
        self._worker.moveToThread(self._thread)
        self._thread.finished.connect(self._worker.deleteLater)
        self._requestRender.connect(self._worker.render)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    @property
    def is_busy(self):
        return (self._in_flight is not None or self._pending is not None or
                bool(self._refine))

    @property
    def seconds_per_ray(self):
        return self._seconds_per_ray

    def estimate(self, scale):
        """Estimated time in seconds to render an image at a scale"""
        if self._seconds_per_ray is None:
            return None
        return self._seconds_per_ray * max(1.0, self.base_rays) * scale * scale

    def draft_scale(self):
        """Scale of previews rendered in about :attr:`target_seconds`"""
        final = self.final_scale
        if self._seconds_per_ray is None:
            return min(0.25, final)
        rays = self._seconds_per_ray * max(1.0, self.base_rays)
        scale = math.sqrt(max(self.target_seconds, 1e-3) / rays)
        scale = min(max(scale, MIN_DRAFT_SCALE), final)
        return max(SCALE_STEP, math.floor(scale / SCALE_STEP) * SCALE_STEP)

    def request(self, interactive=False):
        """Render the latest state of the view

        Parameters
        ----------
        interactive : bool
            Whether the view is changing continuously, e.g. while it is being
            dragged. Interactive requests render a preview and refine it once
            no further request was made for a short time.
        """
        self._serial += 1
        self._refine_timer.stop()
        final = self.final_scale
        draft = self.draft_scale()
        if interactive:
            self._refine_delay = REFINE_DELAY_MS
            if draft >= final:
                self._refine = []
                self._submit(final, True)
            else:
                self._refine = self._refine_levels(draft)
                self._submit(draft, False)
        else:
            self._refine_delay = 0
            estimate = self.estimate(final)
            if draft >= final or (estimate is not None and estimate <=
                                  max(0.25, 2.0 * self.target_seconds)):
                self._refine = []
                self._submit(final, True)
            else:
                self._refine = self._refine_levels(draft)
                self._submit(draft, False)
        self._update_busy()

    def invalidate(self):
        """Discard all requests and results that have not been displayed"""
        self._epoch += 1
        self._pending = None
        self._refine = []
        self._refine_timer.stop()
        self._displayed = -1
        self._update_busy()

    def wait_for_idle(self, timeout_ms=None):
        """Process events until the final image has been rendered

        Returns
        -------
        bool
            Whether the manager is idle
        """
        if not self.is_busy:
            return True
        loop = QEventLoop()

        def on_busy(busy):
            if not busy:
                loop.quit()

        self.busyChanged.connect(on_busy)
        timer = None
        if timeout_ms is not None:
            timer = QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(loop.quit)
            timer.start(int(timeout_ms))
        if self.is_busy:
            loop.exec()
        self.busyChanged.disconnect(on_busy)
        if timer is not None:
            timer.stop()
        return not self.is_busy

    def shutdown(self):
        """Stop the render thread after the current image is finished"""
        self.invalidate()
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()

    def _refine_levels(self, draft):
        final = self.final_scale
        levels = []
        half = 0.5 * final
        estimate = self.estimate(final)
        if half > 1.25 * draft and (estimate is None or estimate > 0.6):
            levels.append(half)
        levels.append(final)
        return levels

    def _submit(self, scale, final):
        entry = (self._serial, scale, final)
        if self._in_flight is not None:
            self._pending = entry
        else:
            self._dispatch(entry)

    def _dispatch(self, entry):
        serial, scale, final = entry
        try:
            request = self._make_request(scale, final)
        except Exception as exc:
            self._refine = []
            self.renderFailed.emit(str(exc))
            request = None
        if request is None:
            self._refine = []
            self._update_busy()
            return
        request.request_id = serial
        request.epoch = self._epoch
        request.scale = scale
        request.final = final
        self._in_flight = request
        self._update_busy()
        self._requestRender.emit(request)

    def _continue(self, finished_serial):
        if self._pending is not None:
            entry, self._pending = self._pending, None
            self._dispatch(entry)
        elif finished_serial == self._serial and self._refine:
            if self._refine_delay > 0:
                self._refine_timer.start(self._refine_delay)
            else:
                self._refine_next()
        else:
            self._refine = []
        self._update_busy()

    def _refine_next(self):
        if not self._refine or self._in_flight is not None:
            return
        self._refine_delay = 0
        scale = self._refine.pop(0)
        self._dispatch((self._serial, scale, not self._refine))

    def _update_busy(self):
        busy = self.is_busy
        if busy != self._busy:
            self._busy = busy
            self.busyChanged.emit(busy)

    @Slot(object)
    def _on_finished(self, result):
        self._in_flight = None
        if result.rays > 0 and result.seconds > 0.0:
            spr = result.seconds / result.rays
            if self._seconds_per_ray is None:
                self._seconds_per_ray = spr
            else:
                self._seconds_per_ray = 0.7 * self._seconds_per_ray + 0.3 * spr
        if result.epoch == self._epoch and \
                result.request_id >= self._displayed:
            self._displayed = result.request_id
            self.resultReady.emit(result)
        self._continue(result.request_id)

    @Slot(object, str)
    def _on_failed(self, request, message):
        self._in_flight = None
        self._refine = []
        if request.epoch == self._epoch:
            self.renderFailed.emit(message)
        self._continue(None)


@dataclass
class _TrackOverlay:
    """Particle tracks projected onto a rendered image (image pixels)"""
    # (rgb, [(xy, visible), ...])
    lines: list = field(default_factory=list)
    # (rgb, xy, visible)
    steps: list = field(default_factory=list)
    # (kind, xy, visible) where kind is 'birth' or 'loss'
    markers: list = field(default_factory=list)
    # (xy_start, xy_end, visible)
    arrows: list = field(default_factory=list)
    # (xy, text, rgb)
    labels: list = field(default_factory=list)
    # (xy with shape (N, 2), function of the index returning a tooltip)
    hover: list = field(default_factory=list)


class Viewport3D(QWidget):
    """Interactive ray-traced 3D view of the model

    Mouse
        Left drag rotates, Shift + left, middle or right drag pans, Ctrl +
        left drag and the wheel zoom. Double-clicking a surface makes it the
        center of rotation. A right click opens the context menu.
    Keyboard
        F fits the model, R or Home resets the camera, 1-7 select standard
        views, O toggles the projection, E edges, D depth shading, T particle
        tracks, the arrow keys rotate (pan with Shift) and +/- zoom.

    Parameters
    ----------
    model : openmc_plotter.plotmodel.PlotModel
        Model providing the 3D view state, colors and particle tracks
    main_window : openmc_plotter.main_window.MainWindow
        Main window of the plotter
    parent : QWidget, optional
        Parent widget
    """
    # Emitted when the 3D view state was changed from the view itself
    stateChanged = Signal()
    busyChanged = Signal(bool)
    resultChanged = Signal(object)

    def __init__(self, model, main_window, parent=None, renderer=None):
        super().__init__(parent)
        self.model = model
        self.main_window = main_window

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setMinimumSize(160, 120)

        if renderer is None:
            renderer = Renderer3D(OPENMC_LIB_LOCK, lib_generation)
        self.renderer = renderer
        self.manager = Render3DManager(renderer, self._make_request, self)
        self.manager.resultReady.connect(self._on_result)
        self.manager.renderFailed.connect(self._on_failed)
        self.manager.busyChanged.connect(self._on_busy)

        self.result = None
        self._image = None
        self._error = None
        self._active = False
        self._dirty = True
        self._styles = None
        self._styles_version = 0
        self._see_through_cache = None
        self._overlay_cache = None
        self._particles_version = 0
        self._drag = None

    # Properties

    @property
    def state(self):
        return self.model.view3d

    @property
    def capability(self):
        return self.renderer.capability

    @property
    def available(self):
        return self.renderer.available

    @property
    def is_active(self):
        return self._active

    def sizeHint(self):
        return QSize(800, 600)

    # Hooks called by the main window and panels

    def activate(self):
        """Start rendering, e.g. when the 3D view is shown"""
        self._active = True
        if not self.available:
            self.update()
            return
        if not self.state.camera_initialized:
            self.fit(render=False)
        self.requestRender()

    def deactivate(self):
        """Stop rendering, e.g. when the 2D view is shown"""
        self._active = False
        QToolTip.hideText()

    def domainStylesChanged(self):
        """Colors, opacities or visibility of cells or materials changed"""
        self._styles = None
        self._styles_version += 1
        self._see_through_cache = None
        self.requestRender()

    def settingsChanged(self, render=True):
        """Settings of the 3D view changed"""
        if render:
            self.requestRender()
        else:
            self.update()

    def particlesChanged(self):
        """Particle tracks or their display options changed"""
        self._particles_version += 1
        self.update()

    def modelReloading(self):
        """Called before :mod:`openmc.lib` is re-initialized"""
        self.manager.invalidate()
        self.renderer.invalidate()

    def modelReloaded(self):
        """Called after :mod:`openmc.lib` was re-initialized"""
        cam = self.state.camera
        try:
            ll, ur = self._scene_box()
        except Exception:
            ll = ur = None
        if ll is not None:
            self.state.scene_lower_left = tuple(float(x) for x in ll)
            self.state.scene_upper_right = tuple(float(x) for x in ur)
            cam.scene_center = tuple(float(x) for x in 0.5 * (ll + ur))
            cam.scene_radius = max(
                0.5 * float(np.linalg.norm(np.subtract(ur, ll))), 1e-6)
        self.domainStylesChanged()

    def shutdown(self):
        """Stop the render thread before :mod:`openmc.lib` is finalized"""
        self._active = False
        self.manager.shutdown()

    def waitForRender(self, timeout_ms=None):
        """Wait until the final image of the current view has been rendered"""
        return self.manager.wait_for_idle(timeout_ms)

    def exportImage(self):
        """Image of the view including overlays as a QImage"""
        return self.grab().toImage()

    # Camera operations

    def fit(self, render=True):
        """Show the whole model"""
        ll, ur = self._scene_box()
        state = self.state
        state.scene_lower_left = tuple(float(x) for x in ll)
        state.scene_upper_right = tuple(float(x) for x in ur)
        state.camera.fit(ll, ur, max(1, self.width()), max(1, self.height()))
        state.camera_initialized = True
        if render:
            self._camera_changed()
        self.stateChanged.emit()

    def resetCamera(self):
        """Isometric view of the whole model"""
        self.state.camera.set_preset('iso')
        self.fit()

    def setPreset(self, name):
        """Look at the model from one of the standard directions"""
        self.state.camera.set_preset(name)
        self._camera_changed()
        self.stateChanged.emit()

    def setProjection(self, projection):
        self.state.camera.set_projection(projection)
        self._camera_changed()
        self.stateChanged.emit()

    def lookAt(self, point, width=None):
        """Rotate about a point and optionally zoom to a view width in [cm]"""
        cam = self.state.camera
        cam.center_on(point)
        if width is not None and width > 0.0:
            if cam.is_orthographic:
                cam.ortho_width = float(width)
            else:
                cam.distance = float(width) / (
                    2.0 * math.tan(math.radians(cam.fov) / 2.0))
        self._camera_changed()

    def cutThrough(self, point, axis=None):
        """Enable a cut plane through a point facing the camera

        Parameters
        ----------
        point : iterable of float
            Point on the cut plane
        axis : {'x', 'y', 'z'}, optional
            Normal of the cut plane. Defaults to the axis closest to the
            viewing direction.
        """
        forward = self.state.camera.forward_vec
        if axis is None:
            axis = 'xyz'[int(np.argmax(np.abs(forward)))]
        i = 'xyz'.index(axis)
        plane = self.state.cut_planes[axis]
        plane['enabled'] = True
        plane['position'] = float(point[i])
        # Remove the half of the model that is closer to the camera
        plane['flip'] = bool(forward[i] > 0.0)
        self.requestRender()
        self.stateChanged.emit()

    def _camera_changed(self, interactive=False):
        self.requestRender(interactive)
        self.update()

    # Rendering

    def requestRender(self, interactive=False):
        """Render the view if it is active, otherwise do so once activated"""
        if not self._active or not self.available or not self.isVisible():
            self._dirty = True
            return
        self._dirty = False
        state = self.state
        self.manager.final_scale = QUALITY_SCALES.get(state.quality, 1.0)
        self.manager.target_seconds = max(5.0, float(state.interactive_ms)) \
            / 1000.0
        w, h = self._render_size(1.0)
        if state.camera.is_orthographic and self.renderer.extended:
            self.manager.base_rays = max(w, h) ** 2
        else:
            self.manager.base_rays = w * h
        self.manager.request(interactive)
        self.update()

    def _device_ratio(self):
        if self.state.quality == 'high':
            return max(1.0, float(self.devicePixelRatioF()))
        return 1.0

    def _render_size(self, scale):
        ratio = self._device_ratio() * scale
        return (max(8, int(round(max(1, self.width()) * ratio))),
                max(8, int(round(max(1, self.height()) * ratio))))

    def _background(self):
        state = self.state
        if state.background is not None:
            return to_rgb(state.background)
        return to_rgb(getattr(self.model.currentView, 'domainBackground',
                              None), default=(50, 50, 50))

    def _current_styles(self):
        if self._styles is None:
            self._styles = domain_styles(self.model.currentView, self.state)
        return self._styles

    def _make_request(self, scale, final):
        if not self._active or self.width() <= 0 or self.height() <= 0:
            return None
        state = self.state
        view = self.model.currentView
        width, height = self._render_size(scale)
        return RenderRequest(
            width=width, height=height, camera=state.camera.copy(),
            color_by=domain_kind(view), styles=self._current_styles(),
            styles_version=self._styles_version,
            background=self._background(), clip_box=state.clip_box(),
            show_cut_faces=state.show_cut_faces,
            diffuse_fraction=state.diffuse_fraction, light=state.light,
            edges=(state.edges and self.renderer.extended and
                   scale >= EDGE_MIN_SCALE * self.manager.final_scale),
            edge_sensitivity=state.edge_sensitivity,
            edge_thickness=state.edge_thickness,
            edge_color=to_rgb(state.edge_color),
            depth_shading=state.depth_shading,
            depth_strength=state.depth_strength,
            highlight_lost_rays=state.highlight_lost_rays,
            final=final, scale=scale)

    def _scene_box(self):
        view = self.model.currentView
        fallback = box_from_2d_view(view.origin, view.width, view.height,
                                    view.basis)
        try:
            ll, ur = self.renderer.bounding_box()
        except Exception:
            return fallback
        return finite_bounding_box(ll, ur, *fallback)

    @Slot(object)
    def _on_result(self, result):
        image = np.ascontiguousarray(result.image, dtype=np.uint8)
        h, w = image.shape[:2]
        self._image = QImage(image.tobytes(), w, h, 3 * w,
                             QImage.Format.Format_RGB888).copy()
        self.result = result
        self._error = None
        self.update()
        self.resultChanged.emit(result)

    @Slot(str)
    def _on_failed(self, message):
        self._error = message
        self.main_window.statusBar().showMessage(
            '3D render failed: {}'.format(message), 10000)
        self.update()

    @Slot(bool)
    def _on_busy(self, busy):
        self.busyChanged.emit(busy)
        self.update()

    # Events

    def showEvent(self, event):
        super().showEvent(event)
        if self._active and self._dirty:
            self.requestRender()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._active:
            self.requestRender(interactive=True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), _qcolor(self._background()))
        if not self.available:
            self._paint_message(painter, _UNAVAILABLE_MESSAGE)
            painter.end()
            return

        result = self.result
        if self._image is not None and result is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform,
                                  bool(result.final))
            painter.drawImage(QRectF(self.rect()), self._image)
            overlay = self._current_overlay()
            if overlay is not None:
                self._paint_overlay(painter, overlay, result)
        elif self._error is None:
            self._paint_message(painter, 'Rendering...')

        if self.state.show_axes:
            self._paint_axes(painter)
        self._paint_status(painter)
        painter.end()

    def mousePressEvent(self, event):
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        button = event.button()
        modifiers = event.modifiers()
        if button == Qt.MouseButton.LeftButton:
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                mode = 'zoom'
            elif modifiers & Qt.KeyboardModifier.ShiftModifier:
                mode = 'pan'
            else:
                mode = 'orbit'
        elif button in (Qt.MouseButton.MiddleButton,
                        Qt.MouseButton.RightButton):
            mode = 'pan'
        else:
            super().mousePressEvent(event)
            return
        pos = event.position()
        self._drag = {'mode': mode, 'button': button, 'start': pos,
                      'last': pos, 'moved': False}
        QToolTip.hideText()
        event.accept()

    def mouseMoveEvent(self, event):
        pos = event.position()
        drag = self._drag
        if drag is None:
            self._hover(event)
            return
        if not drag['moved']:
            offset = pos - drag['start']
            if abs(offset.x()) + abs(offset.y()) < _DRAG_THRESHOLD:
                return
            drag['moved'] = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor
                           if drag['mode'] != 'zoom'
                           else Qt.CursorShape.SizeVerCursor)
        dx = pos.x() - drag['last'].x()
        dy = pos.y() - drag['last'].y()
        drag['last'] = pos
        if not self.available:
            return
        cam = self.state.camera
        if drag['mode'] == 'orbit':
            cam.orbit(dx, dy)
        elif drag['mode'] == 'pan':
            cam.pan(dx, dy, max(1, self.width()))
        else:
            start = drag['start']
            cam.zoom(1.01 ** dy, start.x(), start.y(), max(1, self.width()),
                     max(1, self.height()))
        self._camera_changed(interactive=True)
        event.accept()

    def mouseReleaseEvent(self, event):
        drag = self._drag
        if drag is None or event.button() != drag['button']:
            super().mouseReleaseEvent(event)
            return
        self._drag = None
        self.unsetCursor()
        if drag['button'] == Qt.MouseButton.RightButton and not drag['moved']:
            self._show_context_menu(event.position(),
                                    event.globalPosition().toPoint())
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pick = self._pick(event.position())
        if pick is not None and pick.position is not None:
            self.state.camera.center_on(pick.position)
            self._camera_changed()
            self.main_window.statusBar().showMessage(
                'Rotating about {}'.format(_fmt_point(pick.position)), 5000)
        event.accept()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.pixelDelta().y()
        if delta == 0 or not self.available:
            return
        pos = event.position()
        self.state.camera.zoom(0.85 ** (delta / 120.0), pos.x(), pos.y(),
                               max(1, self.width()), max(1, self.height()))
        self._camera_changed(interactive=True)
        event.accept()

    def keyPressEvent(self, event):
        modifiers = event.modifiers()
        if modifiers & (Qt.KeyboardModifier.ControlModifier |
                        Qt.KeyboardModifier.AltModifier |
                        Qt.KeyboardModifier.MetaModifier):
            super().keyPressEvent(event)
            return
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        key = int(event.key())
        K = Qt.Key
        cam = self.state.camera
        state = self.state
        arrows = {int(K.Key_Left): (-1, 0), int(K.Key_Right): (1, 0),
                  int(K.Key_Up): (0, -1), int(K.Key_Down): (0, 1)}
        presets = {int(K.Key_1) + i: name
                   for i, name in enumerate(VIEW_PRESET_ORDER)}

        if key == int(K.Key_F):
            self.fit()
        elif key in (int(K.Key_R), int(K.Key_Home)):
            self.resetCamera()
        elif key in presets:
            self.setPreset(presets[key])
        elif key == int(K.Key_O):
            self.setProjection(PERSPECTIVE if cam.is_orthographic
                               else ORTHOGRAPHIC)
        elif key == int(K.Key_E):
            state.edges = not state.edges
            self.requestRender()
            self.stateChanged.emit()
        elif key == int(K.Key_D):
            state.depth_shading = not state.depth_shading
            self.requestRender()
            self.stateChanged.emit()
        elif key == int(K.Key_T):
            state.show_tracks = not state.show_tracks
            self.update()
            self.stateChanged.emit()
        elif key in arrows:
            dx, dy = arrows[key]
            if shift:
                step = 0.1 * max(1, self.width())
                cam.pan(-dx * step, -dy * step, max(1, self.width()))
            else:
                cam.orbit(dx * 15.0, dy * 15.0, degrees_per_pixel=1.0)
            self._camera_changed()
        elif key in (int(K.Key_Plus), int(K.Key_Equal)):
            cam.zoom(0.8)
            self._camera_changed()
        elif key in (int(K.Key_Minus), int(K.Key_Underscore)):
            cam.zoom(1.25)
            self._camera_changed()
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)

    # Picking

    def _to_image(self, pos):
        result = self.result
        return (pos.x() * result.width / max(1, self.width()),
                pos.y() * result.height / max(1, self.height()))

    def _pick(self, pos):
        if self.result is None:
            return None
        return self.result.pick(*self._to_image(pos))

    def _domain_label(self, kind, domain_id):
        view = self.model.currentView
        domains = view.cells if kind == 'cell' else view.materials
        try:
            name = domains[domain_id].name
        except (KeyError, AttributeError):
            name = ''
        label = '{} {}'.format('Cell' if kind == 'cell' else 'Material',
                               domain_id)
        return '{} ({})'.format(label, name) if name else label

    def _pick_message(self, pick, kind):
        if pick is None:
            return ''
        if pick.domain_id == LOST_RAY_ID:
            return ('Lost ray: the geometry is undefined or overlapping '
                    'along this line of sight')
        if pick.domain_id < 0 or pick.position is None:
            return ''
        return '{} at {}'.format(self._domain_label(kind, pick.domain_id),
                                 _fmt_point(pick.position))

    def _hover(self, event):
        result = self.result
        if result is None or not self._active:
            return
        pos = event.position()
        text = self._hover_text(pos)
        if text:
            QToolTip.showText(event.globalPosition().toPoint(), text, self)
        else:
            QToolTip.hideText()
        message = self._pick_message(self._pick(pos), result.color_by)
        if message:
            self.main_window.statusBar().showMessage(message)
        else:
            self.main_window.statusBar().clearMessage()

    def _hover_text(self, pos):
        overlay = self._current_overlay()
        result = self.result
        if overlay is None or not overlay.hover:
            return None
        sx = self.width() / result.width
        sy = self.height() / result.height
        p = np.array([pos.x(), pos.y()])
        best = None
        for xy, text_fn in overlay.hover:
            d2 = np.sum((xy * (sx, sy) - p) ** 2, axis=1)
            k = int(np.argmin(d2))
            if d2[k] <= _HOVER_RADIUS ** 2 and (best is None or
                                                d2[k] < best[0]):
                best = (d2[k], text_fn, k)
        if best is None:
            return None
        return best[1](best[2])

    # Context menu

    def _show_context_menu(self, pos, global_pos):
        menu = QMenu(self)
        state = self.state
        cam = state.camera
        extended = self.renderer.extended
        pick = self._pick(pos) if self.available else None
        kind = (self.result.color_by if self.result is not None
                else domain_kind(self.model.currentView))
        kind_label = 'Cell' if kind == 'cell' else 'Material'

        if pick is not None and pick.domain_id >= 0:
            title = menu.addAction(self._domain_label(kind, pick.domain_id))
            title.setEnabled(False)
            menu.addSeparator()
            domain_id = pick.domain_id
            menu.addAction('Hide {}'.format(kind_label)).triggered.connect(
                lambda: self._set_visibility('hide', kind, domain_id))
            ghost = menu.addAction('Ghost {}'.format(kind_label))
            ghost.setEnabled(extended)
            ghost.triggered.connect(
                lambda: self._set_visibility('ghost', kind, domain_id))
            menu.addAction('Isolate {}'.format(kind_label)).triggered.connect(
                lambda: self._set_visibility('isolate', kind, domain_id))
            opacity = menu.addAction('Set Opacity...')
            opacity.setEnabled(extended)
            opacity.triggered.connect(
                lambda: self._edit_opacity(kind, domain_id))
            menu.addAction('Edit Color...').triggered.connect(
                lambda: self.main_window.editDomainColor(kind_label,
                                                         domain_id))
            menu.addSeparator()
        elif pick is not None and pick.domain_id == LOST_RAY_ID:
            title = menu.addAction('Lost ray (geometry error)')
            title.setEnabled(False)
            menu.addSeparator()

        if pick is not None and pick.position is not None:
            point = np.array(pick.position)
            menu.addAction('Center Rotation Here').triggered.connect(
                lambda: (cam.center_on(point), self._camera_changed()))
            if hasattr(self.main_window, 'showPointIn2D'):
                menu.addAction('Show in 2D View').triggered.connect(
                    lambda: self.main_window.showPointIn2D(point))
            menu.addAction('Copy Coordinates').triggered.connect(
                lambda: QApplication.clipboard().setText(
                    '{:.8g}, {:.8g}, {:.8g}'.format(*point)))
            cut_menu = menu.addMenu('Cut Plane Here')
            cut_menu.setEnabled(extended)
            for axis in 'xyz':
                value = point['xyz'.index(axis)]
                cut_menu.addAction('{} = {:.6g}'.format(
                    axis.upper(), value)).triggered.connect(
                    lambda checked=False, a=axis: self.cutThrough(point, a))
            menu.addSeparator()

        show_all = menu.addAction('Show All')
        show_all.setEnabled(state.has_visibility_overrides('cell') or
                            state.has_visibility_overrides('material'))
        show_all.triggered.connect(self.showAll)
        if any(p['enabled'] for p in state.cut_planes.values()):
            menu.addAction('Remove Cut Planes').triggered.connect(
                self.clearCutPlanes)
        menu.addSeparator()

        menu.addAction('Fit to Model\tF').triggered.connect(self.fit)
        menu.addAction('Reset Camera\tR').triggered.connect(self.resetCamera)
        view_menu = menu.addMenu('View From')
        for i, name in enumerate(VIEW_PRESET_ORDER):
            view_menu.addAction('{}\t{}'.format(
                VIEW_PRESET_LABELS[name], i + 1)).triggered.connect(
                lambda checked=False, n=name: self.setPreset(n))
        projection_menu = menu.addMenu('Projection')
        for label, value in (('Perspective', PERSPECTIVE),
                             ('Orthographic', ORTHOGRAPHIC)):
            action = projection_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(cam.projection == value)
            action.setEnabled(extended or value == PERSPECTIVE)
            action.triggered.connect(
                lambda checked=False, v=value: self.setProjection(v))
        menu.addSeparator()

        def toggle(label, attribute, render, enabled=True):
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(bool(getattr(state, attribute)))
            action.setEnabled(enabled)

            def on_toggled(checked):
                setattr(state, attribute, bool(checked))
                self.settingsChanged(render)
                self.stateChanged.emit()
            action.toggled.connect(on_toggled)

        toggle('Edges\tE', 'edges', True, extended)
        toggle('Depth Shading\tD', 'depth_shading', True, extended)
        toggle('Particle Tracks\tT', 'show_tracks', False)
        toggle('Axes', 'show_axes', False)
        menu.addSeparator()

        for name in ('copyImageAction', 'saveImageAction', 'toggle3DAction',
                     'dockAction'):
            action = getattr(self.main_window, name, None)
            if action is not None:
                menu.addAction(action)
        menu.exec(global_pos)

    def _set_visibility(self, operation, kind, domain_id):
        getattr(self.state, operation)(kind, {domain_id})
        self.domainStylesChanged()
        self.stateChanged.emit()

    def showAll(self):
        """Clear hidden, ghosted and isolated domains of the 3D view"""
        self.state.show_all()
        self.domainStylesChanged()
        self.stateChanged.emit()

    def clearCutPlanes(self):
        for plane in self.state.cut_planes.values():
            plane['enabled'] = False
        self.requestRender()
        self.stateChanged.emit()

    def _edit_opacity(self, kind, domain_id):
        view = self.model.activeView
        domains = view.cells if kind == 'cell' else view.materials
        try:
            current = float(getattr(domains[domain_id], 'opacity', 1.0))
        except KeyError:
            return
        value, ok = QInputDialog.getDouble(
            self, 'Set Opacity',
            '{} opacity in the 3D view\n(0 = invisible, 1 = opaque):'.format(
                self._domain_label(kind, domain_id)),
            current, 0.0, 1.0, 2)
        if ok:
            domains.set_opacity(domain_id, value)
            self.main_window.applyChanges()

    # Particle tracks

    def _see_through(self, result):
        """Pixels showing partially transparent domains"""
        if result.ids is None:
            return None
        cache = self._see_through_cache
        if cache is not None and cache[0] is result:
            return cache[1]
        styles = self._current_styles()
        transparent = [i for i, (_, alpha) in styles.items()
                       if 0.0 < alpha < 1.0]
        mask = None
        if transparent:
            mask = np.isin(result.ids, np.asarray(transparent,
                                                  dtype=result.ids.dtype))
        self._see_through_cache = (result, mask)
        return mask

    def _current_overlay(self):
        result = self.result
        if result is None or not self.state.show_tracks:
            return None
        options = getattr(self.model, 'particleOptions', {}) or {}
        particles = getattr(self.model, 'particleTracks', None)
        if not particles:
            return None
        key = (self._particles_version, self._styles_version,
               tuple(sorted(options.items())), self.width(), self.height())
        cache = self._overlay_cache
        if cache is not None and cache[0] is result and cache[1] == key:
            return cache[2]
        overlay = self._build_overlay(result, particles, options)
        self._overlay_cache = (result, key, overlay)
        return overlay

    def _build_overlay(self, result, particles, options):
        overlay = _TrackOverlay()
        cam = result.camera
        w, h, depth = result.width, result.height, result.depth
        see_through = self._see_through(result)
        show_markers = options.get('showMarkers', True)
        show_steps = options.get('showSteps', False)
        show_labels = options.get('showLabels', True)
        marker_hover = []
        state_hover = []

        def project(point):
            return point_visibility(cam, w, h, depth, [point], see_through)

        for p in particles:
            if not p.visible:
                continue
            rgb = to_rgb(p.color)
            for ti, track in enumerate(p.tracks):
                positions = track.positions
                if positions is None or len(positions) == 0:
                    continue
                if len(positions) >= 2:
                    overlay.lines.append((rgb, track_overlay(
                        cam, w, h, depth, positions, see_through)))
                xy, valid, visible = point_visibility(cam, w, h, depth,
                                                      positions, see_through)
                index = np.flatnonzero(valid)
                if index.size == 0:
                    continue
                if show_steps:
                    overlay.steps.append((rgb, xy[index], visible[index]))
                state_hover.append((xy[index], lambda k, p=p, ti=ti, i=index:
                                    p.state_description(ti, int(i[k]))))

            if show_markers:
                if p.birth is not None:
                    xy, valid, visible = project(p.birth)
                    if valid[0]:
                        overlay.markers.append(('birth', xy[0],
                                                bool(visible[0])))
                        marker_hover.append((xy[:1], lambda k, p=p: (
                            '{}\nSource position: {}'.format(
                                p.summary(), _fmt_point(p.birth)))))
                loss = p.loss_position
                if loss is not None:
                    xy, valid, visible = project(loss)
                    if valid[0]:
                        overlay.markers.append(('loss', xy[0],
                                                bool(visible[0])))
                        marker_hover.append((xy[:1],
                                             lambda k, p=p: p.describe()))
                        arrow = self._loss_arrow(cam, w, h, loss,
                                                 p.loss_direction)
                        if arrow is not None:
                            overlay.arrows.append((xy[0], arrow,
                                                   bool(visible[0])))

            if show_labels:
                anchor = p.key_position()
                if anchor is not None:
                    xy, valid, _ = project(anchor)
                    if valid[0]:
                        text = ('#{}'.format(p.particle_id)
                                if p.particle_id is not None else p.label)
                        overlay.labels.append((xy[0], text, rgb))

        # Markers take precedence over track states at the same position
        overlay.hover = marker_hover + state_hover
        return overlay

    def _loss_arrow(self, cam, w, h, loss, direction):
        """End of an arrow showing the direction of a lost particle"""
        if direction is None:
            return None
        u = np.asarray(direction, dtype=float).reshape(3)
        norm = float(np.linalg.norm(u))
        if not np.isfinite(norm) or norm <= 0.0:
            return None
        _, d, valid = cam.project([loss], w, h)
        if not valid[0]:
            return None
        # The arrow is about _ARROW_PIXELS long on the screen
        length = _ARROW_PIXELS * float(pixel_footprint(
            abs(float(d[0])), cam, max(1, self.width())))
        tip, _, tip_valid = cam.project([np.asarray(loss) + u / norm * length],
                                        w, h)
        if not tip_valid[0] or not np.all(np.isfinite(tip[0])):
            return None
        return tip[0]

    # Painting

    def _paint_overlay(self, painter, overlay, result):
        options = getattr(self.model, 'particleOptions', {}) or {}
        xray = options.get('xray', True)
        width = max(0.5, float(options.get('lineWidth', 1.5)))
        sx = self.width() / result.width
        sy = self.height() / result.height

        def point(xy):
            return QPointF(float(xy[0]) * sx, float(xy[1]) * sy)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for rgb, runs in overlay.lines:
            for xy, visible in runs:
                if not visible and not xray:
                    continue
                pen = QPen(_qcolor(rgb, 255 if visible else 115),
                           width if visible else max(1.0, 0.8 * width))
                pen.setStyle(Qt.PenStyle.SolidLine if visible
                             else Qt.PenStyle.DashLine)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)
                painter.drawPolyline(QPolygonF([point(p) for p in xy]))

        painter.setPen(Qt.PenStyle.NoPen)
        for rgb, xy, visible in overlay.steps:
            for p, vis in zip(xy, visible):
                if not vis and not xray:
                    continue
                painter.setBrush(_qcolor(rgb, 255 if vis else 110))
                painter.drawEllipse(point(p), 2.5, 2.5)

        for start, end, visible in overlay.arrows:
            if not visible and not xray:
                continue
            self._paint_arrow(painter, point(start), point(end),
                              _qcolor(LOSS_COLOR, 255 if visible else 140))

        for kind, xy, visible in overlay.markers:
            if not visible and not xray:
                continue
            alpha = 255 if visible else 150
            center = point(xy)
            if kind == 'birth':
                painter.setPen(QPen(_qcolor((255, 255, 255), alpha), 3.5))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(center, 5.5, 5.5)
                painter.setPen(QPen(_qcolor(BIRTH_COLOR, alpha), 2.0))
                painter.setBrush(_qcolor(BIRTH_COLOR, alpha // 3))
                painter.drawEllipse(center, 5.5, 5.5)
            else:
                r = 6.0
                for color, pen_width in (((255, 255, 255), 4.5),
                                         (LOSS_COLOR, 2.5)):
                    pen = QPen(_qcolor(color, alpha), pen_width)
                    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    painter.setPen(pen)
                    painter.drawLine(center + QPointF(-r, -r),
                                     center + QPointF(r, r))
                    painter.drawLine(center + QPointF(-r, r),
                                     center + QPointF(r, -r))

        for xy, text, rgb in overlay.labels:
            self._paint_label(painter, point(xy) + QPointF(9.0, -9.0), text,
                              rgb)
        painter.restore()

    @staticmethod
    def _paint_arrow(painter, start, end, color):
        pen = QPen(color, 2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(start, end)
        delta = end - start
        length = math.hypot(delta.x(), delta.y())
        if length < 1e-6:
            return
        ux, uy = delta.x() / length, delta.y() / length
        size = min(9.0, 0.5 * length)
        for sign in (1.0, -1.0):
            # Rotate the reversed direction by +-25 degrees
            c, s = math.cos(math.radians(25.0)), sign * math.sin(
                math.radians(25.0))
            hx = -(ux * c - uy * s) * size
            hy = -(ux * s + uy * c) * size
            painter.drawLine(end, end + QPointF(hx, hy))

    @staticmethod
    def _paint_label(painter, anchor, text, rgb):
        metrics = painter.fontMetrics()
        rect = QRectF(metrics.boundingRect(text)).adjusted(-3, -1, 3, 1)
        rect.moveBottomLeft(anchor)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 200))
        painter.drawRoundedRect(rect, 3, 3)
        painter.setPen(_qcolor(rgb).darker(140))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_axes(self, painter):
        cam = self.state.camera
        r, u, f = cam.right_vec, cam.up_vec, cam.forward_vec
        origin = QPointF(44.0, self.height() - 44.0)
        length = 28.0
        axes = []
        for name, color in (('X', (220, 50, 50)), ('Y', (40, 170, 40)),
                            ('Z', (60, 110, 235))):
            e = np.eye(3)['XYZ'.index(name)]
            axes.append((float(e @ f), name, float(e @ r), -float(e @ u),
                         color))
        # Draw the axes pointing away from the viewer first
        axes.sort(key=lambda a: -a[0])

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 70))
        painter.drawEllipse(origin, length + 14.0, length + 14.0)
        for _, name, dx, dy, color in axes:
            end = origin + QPointF(dx * length, dy * length)
            pen = QPen(_qcolor(color), 2.5)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(origin, end)
            label = origin + QPointF(dx * (length + 9.0), dy * (length + 9.0))
            painter.drawText(QRectF(label.x() - 7.0, label.y() - 8.0, 14.0,
                                    16.0), Qt.AlignmentFlag.AlignCenter, name)
        painter.restore()

    def _paint_message(self, painter, text):
        painter.save()
        painter.setPen(QColor(230, 230, 230))
        rect = QRectF(self.rect()).adjusted(20, 20, -20, -20)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter |
                         Qt.TextFlag.TextWordWrap, text)
        painter.restore()

    def _paint_status(self, painter):
        if not self.available:
            return
        lines = []
        result = self.result
        cam = self.state.camera
        projection = 'Orthographic' if cam.is_orthographic else 'Perspective'
        if result is not None:
            text = '{} | {}\u00d7{} px | {:.0f} ms'.format(
                projection, result.width, result.height,
                1000.0 * result.seconds)
            final = self.manager.final_scale or 1.0
            if not result.final:
                text += ' | preview {:.0f}%'.format(
                    100.0 * result.scale / final)
            lines.append((text, QColor(240, 240, 240)))
            if result.n_lost_rays:
                lines.append(('{} lost rays (highlighted in magenta) - check '
                              'the geometry for gaps or overlaps'.format(
                                  result.n_lost_rays), QColor(255, 120, 255)))
        if self.manager.is_busy:
            lines.append(('Rendering...', QColor(240, 240, 240)))
        if self.capability == CAPABILITY_BASIC:
            lines.append((_BASIC_MESSAGE, QColor(255, 220, 120)))
        if self._error:
            lines.append(('Render failed: {}'.format(self._error),
                          QColor(255, 130, 130)))
        if not lines:
            return

        painter.save()
        metrics = painter.fontMetrics()
        line_height = metrics.height()
        max_width = max(100, self.width() - 16)
        widths = [min(max_width, metrics.horizontalAdvance(t))
                  for t, _ in lines]
        box = QRectF(6.0, 6.0, max(widths) + 10.0,
                     line_height * len(lines) + 6.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 130))
        painter.drawRoundedRect(box, 4, 4)
        y = box.top() + 3.0
        for text, color in lines:
            painter.setPen(color)
            elided = metrics.elidedText(text, Qt.TextElideMode.ElideRight,
                                        max_width)
            painter.drawText(QRectF(box.left() + 5.0, y, max_width,
                                    line_height),
                             Qt.AlignmentFlag.AlignLeft |
                             Qt.AlignmentFlag.AlignVCenter, elided)
            y += line_height
        painter.restore()


def capability_message(capability):
    """Explanation of a ray tracer capability for tooltips"""
    if capability == CAPABILITY_NONE:
        return _UNAVAILABLE_MESSAGE
    if capability == CAPABILITY_BASIC:
        return _BASIC_MESSAGE
    return ('Extended ray tracer: all features of the 3D view are '
            'available.')
