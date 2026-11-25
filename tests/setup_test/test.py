import filecmp
import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from openmc_plotter import render3d
from openmc_plotter.main_window import MainWindow, _openmcReload
from openmc_plotter.particles import particle_option_defaults
from openmc_plotter.view3d import capability_message

MODEL_DIR = Path(__file__).resolve().parent

@pytest.fixture
def run_in_tmpdir(tmpdir):
    orig = tmpdir.chdir()
    try:
        yield
    finally:
        orig.chdir()

def test_window(tmpdir, qtbot):
    orig = tmpdir.chdir()
    mw = MainWindow(model_path=orig)
    _openmcReload(model_path=orig)
    mw.loadGui()

    try:
        assert mw.waitForPlotIdle(60000)
        mw.saveImage(tmpdir / 'test.png')

        qtbot.addWidget(mw)
    finally:
        orig.chdir()

    filecmp.cmp(orig / 'ref.png', tmpdir / 'test.png')

    mw.close()

def test_batch_image(tmpdir, qtbot):
    orig = tmpdir.chdir()

    # move view file into tmpdir
    shutil.copy2(orig / 'test.pltvw', tmpdir)
    shutil.copy2(orig / 'test1.pltvw', tmpdir)

    _openmcReload(model_path=orig)

    mw = MainWindow(model_path=orig)
    mw.loadGui()

    try:
        mw.saveBatchImage('test.pltvw')
        qtbot.addWidget(mw)

        mw.saveBatchImage('test1.pltvw')
        qtbot.addWidget(mw)
    finally:
        orig.chdir()

    filecmp.cmp(orig / 'ref.png', tmpdir / 'test.png')
    filecmp.cmp(orig / 'ref1.png', tmpdir / 'test1.png')

    mw.close()

def test_copy_image_to_clipboard(tmpdir, monkeypatch, qtbot):
    orig = tmpdir.chdir()
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    mw = MainWindow(model_path=orig)
    _openmcReload(model_path=orig)
    mw.loadGui()
    qtbot.addWidget(mw)
    mw.show()

    class FakeClipboard:
        def __init__(self):
            self.image = None

        def setImage(self, image):
            self.image = image

    fake_clipboard = FakeClipboard()
    monkeypatch.setattr(QtGui.QGuiApplication,
                        'clipboard',
                        staticmethod(lambda: fake_clipboard))

    try:
        assert mw.waitForPlotIdle(60000)
        mw.model.currentView.domainVisible = False
        mw.plotIm.updatePixmap()
        assert mw.copyImageToClipboard()
    finally:
        orig.chdir()

    assert fake_clipboard.image is not None
    assert not fake_clipboard.image.isNull()
    assert fake_clipboard.image.hasAlphaChannel()

    canvas_width, canvas_height = mw.plotIm.get_width_height()
    expected_width = round(
        min(mw.frame.viewport().width(), mw.plotIm.width())
        * canvas_width
        / mw.plotIm.width()
    )
    expected_height = round(
        min(mw.frame.viewport().height(), mw.plotIm.height())
        * canvas_height
        / mw.plotIm.height()
    )
    assert fake_clipboard.image.width() == pytest.approx(expected_width, abs=1)
    assert fake_clipboard.image.height() == pytest.approx(expected_height, abs=1)

    center = fake_clipboard.image.pixelColor(fake_clipboard.image.width() // 2,
                                             fake_clipboard.image.height() // 2)
    assert center.alpha() == 0

    mw.close()


# 3D view and lost particle tracks

LOSS_POSITION = (3.0, 1.0, 0.0)


def write_lost_particle(h5py, path):
    """Write a restart file like OpenMC does for a lost particle"""
    with h5py.File(str(path), 'w') as f:
        f.attrs['filetype'] = np.bytes_('particle restart')
        f['current_batch'] = 1
        f['generations_per_batch'] = 1
        f['current_generation'] = 1
        f['n_particles'] = 100
        f['run_mode'] = np.bytes_('fixed source')
        f['id'] = 5
        f['type'] = 2112
        f['weight'] = 1.0
        f['energy'] = 2.0e6
        f['xyz'] = [0.0, 0.0, 0.0]
        f['uvw'] = [1.0, 0.0, 0.0]
        f['time'] = 0.0
        lost = f.create_group('lost')
        lost['message'] = np.bytes_('No cells found.')
        lost['xyz'] = LOSS_POSITION
        lost['uvw'] = [1.0, 0.0, 0.0]
        lost['energy'] = 1.0e6
        lost['time'] = 1.0e-9
        lost['weight'] = 1.0
        lost['xyz_last'] = [1.0, 1.0, 0.0]
        lost['uvw_last'] = [1.0, 0.0, 0.0]
        lost['cell_last_id'] = np.array([2], dtype=np.int32)


def open_window(qtbot, use_settings_pkl=False, show=True):
    """Open the plotter for the test model and wait for the first 2D plot"""
    mw = MainWindow(model_path=MODEL_DIR)
    _openmcReload(model_path=MODEL_DIR)
    mw.loadGui(use_settings_pkl=use_settings_pkl)
    qtbot.addWidget(mw)
    if show:
        mw.show()
    # loadGui requests the first plot from the event loop
    qtbot.waitUntil(lambda: mw.plotIm.ax is not None, timeout=60000)
    assert mw.waitForPlotIdle(60000)
    return mw


def unexpected_message_box(parent, title, text, *args, **kwargs):
    pytest.fail('Unexpected message box "{}": {}'.format(title, text))


def drag(widget, dx, dy, steps=6):
    """Drag across a widget with the left mouse button"""
    left = QtCore.Qt.MouseButton.LeftButton
    no_button = QtCore.Qt.MouseButton.NoButton
    x0, y0 = widget.width() / 2.0, widget.height() / 2.0

    def send(kind, x, y, button, buttons):
        event = QtGui.QMouseEvent(kind, QtCore.QPointF(x, y), button, buttons,
                                  QtCore.Qt.KeyboardModifier.NoModifier)
        QtWidgets.QApplication.sendEvent(widget, event)

    send(QtCore.QEvent.Type.MouseButtonPress, x0, y0, left, left)
    for i in range(1, steps + 1):
        send(QtCore.QEvent.Type.MouseMove, x0 + dx * i / steps,
             y0 + dy * i / steps, no_button, left)
    send(QtCore.QEvent.Type.MouseButtonRelease, x0 + dx, y0 + dy, left,
         no_button)


def assert_render_changes(viewport, change):
    """Check that changing a setting of the 3D view changes its image"""
    before = viewport.result.image.copy()
    change()
    assert viewport.waitForRender(60000)
    after = viewport.result.image
    assert after.shape == before.shape
    assert not np.array_equal(after, before)


def assert_rays_hit_model(result):
    """Check that the camera of a rendered image casts the rays of OpenMC

    The first visible surface of every pixel must lie on one of the cylinders
    of the test model or on the plane y = 5.
    """
    depth = result.depth
    rows, cols = np.nonzero(np.isfinite(depth))
    assert rows.size > 0.1 * depth.size
    for iy, ix in zip(rows[::7], cols[::7]):
        origin, direction = result.camera.pixel_ray(
            ix + 0.5, iy + 0.5, result.width, result.height)
        x, y, _ = origin + float(depth[iy, ix]) * direction
        radius = np.hypot(x, y)
        distance = min(abs(radius - 2.0), abs(radius - 5.0),
                       abs(radius - 10.0), abs(y - 5.0))
        assert distance < 1e-3, (ix, iy, x, y)


def test_particle_tracks_and_3d_view(tmpdir, qtbot, monkeypatch):
    h5py = pytest.importorskip('h5py')
    path = Path(tmpdir) / 'particle_1_5.h5'
    write_lost_particle(h5py, path)
    # A file that cannot be read would block the test with a message box
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning',
                        staticmethod(unexpected_message_box))
    monkeypatch.chdir(tmpdir)

    mw = open_window(qtbot)
    try:
        # 2D view
        [particle] = mw.addParticles([str(path)])
        assert particle.particle_id == 5
        assert np.allclose(particle.loss_position, LOSS_POSITION)
        assert mw.particlePanel.table.rowCount() == 1
        assert mw.plotIm._track_hover
        # Files that are already loaded are skipped
        assert mw.addParticles([str(path)]) == []
        assert len(mw.model.particleTracks) == 1

        mw.showParticle(particle, in3d=False)
        assert not mw.in3D
        assert mw.waitForPlotIdle(60000)
        assert np.allclose(mw.model.currentView.origin, LOSS_POSITION)
        assert mw.plotIm._track_hover
        assert '#5' in [text.get_text() for text in mw.plotIm.ax.texts]
        mw.saveImage(str(Path(tmpdir) / 'tracks_2d.png'))
        assert (Path(tmpdir) / 'tracks_2d.png').is_file()

        viewport = mw.viewport3d
        if not viewport.available:
            # Without a ray tracer, particles can only be shown in 2D
            mw.showParticle(particle, in3d=True)
            assert not mw.in3D
            return

        # 3D view
        mw.showParticle(particle, in3d=True)
        assert mw.in3D
        assert viewport.waitForRender(60000)
        result = viewport.result
        assert result is not None and result.final
        assert result.image.shape == (result.height, result.width, 3)
        overlay = viewport._current_overlay()
        assert overlay is not None
        assert any(runs for _, runs in overlay.lines)
        assert {'birth', 'loss'} <= {kind for kind, _, _ in overlay.markers}
        assert [text for _, text, _ in overlay.labels] == ['#5']

        # Dragging with the left mouse button orbits the camera
        forward = mw.model.view3d.camera.forward
        drag(viewport, 60, 0)
        assert not np.allclose(mw.model.view3d.camera.forward, forward)
        assert viewport.waitForRender(60000)
        assert viewport.result.final
        assert np.allclose(viewport.result.camera.forward,
                           mw.model.view3d.camera.forward)

        if viewport.renderer.extended:
            viewport.resetCamera()
            assert viewport.waitForRender(60000)
            for projection in (render3d.ORTHOGRAPHIC, render3d.PERSPECTIVE):
                viewport.setProjection(projection)
                assert viewport.waitForRender(60000)
                assert viewport.result.camera.projection == projection
                assert_rays_hit_model(viewport.result)
            panel = mw.view3DPanel
            assert_render_changes(
                viewport, lambda: panel.edgeGroupBox.setChecked(False))
            assert not mw.model.view3d.edges
            assert_render_changes(
                viewport, lambda: panel.depthGroupBox.setChecked(False))
            assert not mw.model.view3d.depth_shading
            assert_render_changes(
                viewport, lambda: panel.cutWidgets['z'][0].setChecked(True))
            assert mw.model.view3d.clip_box() is not None

        mw.saveImage(str(Path(tmpdir) / 'tracks_3d.png'))
        assert (Path(tmpdir) / 'tracks_3d.png').is_file()
        assert not viewport.exportImage().isNull()

        assert not mw.toggle3DView(False)
        assert mw.plotIm._track_hover
    finally:
        mw.close()


def test_3d_view_fallbacks(tmpdir, qtbot, monkeypatch):
    monkeypatch.chdir(tmpdir)
    mw = open_window(qtbot)
    viewport = mw.viewport3d
    try:
        if viewport.available:
            # Without the extended ray tracer API, images are rendered in
            # perspective and without depth or domain IDs
            monkeypatch.setattr(viewport.renderer, 'capability',
                                render3d.CAPABILITY_BASIC)
            viewport.setProjection(render3d.ORTHOGRAPHIC)
            assert mw.toggle3DView(True)
            assert viewport.waitForRender(60000)
            result = viewport.result
            assert result is not None and result.final
            assert result.capability == render3d.CAPABILITY_BASIC
            assert result.depth is None and result.ids is None
            assert not result.camera.is_orthographic
            assert result.image.shape == (result.height, result.width, 3)
            assert not mw.toggle3DView(False)

        # Without a ray tracer, the 2D view is kept
        monkeypatch.setattr(viewport.renderer, 'capability',
                            render3d.CAPABILITY_NONE)
        assert not mw.toggle3DView(True)
        assert not mw.in3D
        assert not mw.toggle3DAction.isEnabled()
        assert mw.statusBar().currentMessage() == \
            capability_message(render3d.CAPABILITY_NONE)
    finally:
        mw.close()


def test_settings_without_3d_view(tmpdir, qtbot, monkeypatch):
    monkeypatch.chdir(tmpdir)
    settings_pkl = MODEL_DIR / 'plot_settings.pkl'

    mw = open_window(qtbot, show=False)
    mw.model.view3d.edges = False
    mw.model.particleOptions['lineWidth'] = 3.0
    mw.close()
    with settings_pkl.open('rb') as f:
        data = pickle.load(f)
    assert not data['view3d']['edges']
    assert data['particles']['lineWidth'] == 3.0

    # The 3D view settings and particle options are restored
    mw = open_window(qtbot, use_settings_pkl=True, show=False)
    assert not mw.model.view3d.edges
    assert mw.model.particleOptions['lineWidth'] == 3.0
    mw.close()

    # Settings saved by earlier versions of the plotter have neither
    del data['view3d'], data['particles']
    with settings_pkl.open('wb') as f:
        pickle.dump(data, f)
    mw = open_window(qtbot, use_settings_pkl=True, show=False)
    assert mw.model.view3d.edges
    assert mw.model.particleOptions == particle_option_defaults()
    mw.close()
