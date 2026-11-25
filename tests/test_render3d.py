"""Tests for the camera, settings and image processing of the 3D view

These run without OpenMC's ray tracer. Rendered images are replaced by
synthetic depth and domain ID buffers.
"""

import math
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from openmc_plotter import render3d as rd

PROJECTIONS = (rd.PERSPECTIVE, rd.ORTHOGRAPHIC)
SIZES = ((80, 50), (50, 80))


def oblique_camera(projection):
    return rd.Camera3D(target=(1.0, 2.0, 3.0), distance=40.0,
                       forward=(-1.0, 2.0, -0.5), up=(0.0, 0.0, 1.0),
                       projection=projection, fov=50.0, ortho_width=30.0,
                       scene_center=(0.0, 0.0, 0.0), scene_radius=20.0)


def top_camera():
    """Orthographic camera looking down on a 40 x 40 image

    The point (x, y, z) is at display position (2x + 20.5, 20.5 - 2y) and
    depth 21 - z.
    """
    return rd.Camera3D(forward=(0.0, 0.0, -1.0), up=(0.0, 1.0, 0.0),
                       projection=rd.ORTHOGRAPHIC, ortho_width=20.0,
                       scene_radius=10.0)


def surface_left_half():
    """Depth buffer of top_camera() with a surface at z = 0 where x < 0"""
    depth = np.full((40, 40), np.inf)
    depth[:, :20] = 21.0
    return depth


# Camera

@pytest.mark.parametrize('projection', PROJECTIONS)
@pytest.mark.parametrize('width, height', SIZES)
def test_pixel_ray_inverts_projection(projection, width, height):
    cam = oblique_camera(projection)
    rng = np.random.default_rng(1)
    for x, y in rng.uniform(0.0, 1.0, (20, 2)) * (width, height):
        origin, direction = cam.pixel_ray(x, y, width, height)
        assert np.linalg.norm(direction) == pytest.approx(1.0)
        for t in (0.5, 10.0, 200.0):
            xy, depth, valid = cam.project([origin + t * direction],
                                           width, height)
            assert valid[0]
            np.testing.assert_allclose(xy[0], (x, y), atol=1e-8)
            # Depths are distances along the ray like the depth buffer
            assert depth[0] == pytest.approx(t)

    xy, _, _ = cam.project([cam.target], width, height)
    assert np.all(np.abs(xy[0] - (width / 2, height / 2)) <= 0.5 + 1e-9)


@pytest.mark.parametrize('projection', PROJECTIONS)
@pytest.mark.parametrize('width, height', SIZES)
@pytest.mark.parametrize('preset', rd.VIEW_PRESET_ORDER)
def test_fit_shows_whole_box(projection, width, height, preset):
    cam = rd.Camera3D(projection=projection)
    cam.set_preset(preset)
    cam.fit((-30.0, -5.0, 0.0), (10.0, 5.0, 100.0), width, height)

    corners = [(x, y, z) for x in (-30.0, 10.0) for y in (-5.0, 5.0)
               for z in (0.0, 100.0)]
    xy, _, valid = cam.project(corners, width, height)
    assert valid.all()
    assert np.all((xy >= 0.0) & (xy <= (width, height)))
    np.testing.assert_allclose(cam.target, (-10.0, 0.0, 50.0))


def test_orbit_keeps_camera_upright():
    cam = rd.Camera3D()
    rng = np.random.default_rng(2)
    for dx, dy in rng.uniform(-500.0, 500.0, (200, 2)):
        cam.orbit(dx, dy)
        f, u = cam.forward_vec, cam.up_vec
        # OpenMC's ray tracer fails if the directions are parallel
        assert np.linalg.norm(f) == pytest.approx(1.0)
        assert np.linalg.norm(u) == pytest.approx(1.0)
        assert abs(np.dot(f, u)) < 1e-9
        assert u[2] >= -1e-9

    # Tilting stops when looking straight up or down
    cam.set_preset('+x')
    cam.orbit(0.0, -1000.0)
    assert cam.elevation == pytest.approx(90.0)
    cam.orbit(0.0, -1000.0)
    assert cam.elevation == pytest.approx(90.0)
    cam.orbit(0.0, 2000.0)
    assert cam.elevation == pytest.approx(-90.0)


@pytest.mark.parametrize('projection', PROJECTIONS)
def test_zoom_keeps_point_below_cursor(projection):
    cam = oblique_camera(projection)
    x, y, width, height = 10.5, 30.5, 80, 50
    origin, direction = cam.pixel_ray(x, y, width, height)
    f = cam.forward_vec
    t = np.dot(np.array(cam.target) - origin, f) / np.dot(direction, f)
    point = origin + t * direction
    view_width = cam.view_width()

    cam.zoom(0.5, x, y, width, height)
    assert cam.view_width() == pytest.approx(0.5 * view_width)
    xy, _, _ = cam.project([point], width, height)
    np.testing.assert_allclose(xy[0], (x, y), atol=1e-8)


def test_projection_switch_keeps_view_width():
    cam = rd.Camera3D(distance=40.0, fov=30.0)
    width = cam.view_width()
    cam.set_projection(rd.ORTHOGRAPHIC)
    assert cam.view_width() == pytest.approx(width)
    cam.set_projection(rd.PERSPECTIVE)
    assert cam.view_width() == pytest.approx(width)
    assert cam.distance == pytest.approx(40.0)
    with pytest.raises(ValueError):
        cam.set_projection('fisheye')


# Settings

def test_view_state_round_trips_through_builtins():
    state = rd.View3DState()
    state.camera.fit((-1.0, -2.0, -3.0), (4.0, 5.0, 6.0), 64, 48)
    state.camera.orbit(30.0, -20.0)
    state.camera.set_projection(rd.ORTHOGRAPHIC)
    state.cut_planes['z'].update(enabled=True, position=1.5, flip=True)
    state.hide('cell', {1, 2})
    state.isolate('material', {7})
    state.edges = False

    data = state.to_dict()
    # Settings files can be read without this module
    pickled = pickle.dumps(data)
    assert b'numpy' not in pickled and b'openmc_plotter' not in pickled

    restored = rd.View3DState.from_dict(pickle.loads(pickled)).to_dict()
    camera, restored_camera = data.pop('camera'), restored.pop('camera')
    assert restored == data
    assert restored_camera.keys() == camera.keys()
    for key, value in camera.items():
        if isinstance(value, tuple):
            np.testing.assert_allclose(restored_camera[key], value,
                                       atol=1e-12)
        else:
            assert restored_camera[key] == value


def test_view_state_from_partial_or_unknown_settings():
    state = rd.View3DState.from_dict({
        'edges': False, 'from_a_newer_version': 1,
        'cut_planes': {'x': {'enabled': True}}, 'hidden': {'cell': [3]},
    })
    assert state.edges is False
    assert not hasattr(state, 'from_a_newer_version')
    assert state.cut_planes['x'] == {'enabled': True, 'position': 0.0,
                                     'flip': False}
    assert state.hidden == {'cell': {3}, 'material': set()}
    assert rd.View3DState.from_dict(None).to_dict() == \
        rd.View3DState().to_dict()

    # Pickles of older versions lack newer settings
    old = rd.View3DState()
    old.edges = False
    del old.ghost_alpha, old.isolated
    restored = pickle.loads(pickle.dumps(old))
    assert restored.edges is False
    assert restored.ghost_alpha == rd.View3DState().ghost_alpha
    assert restored.isolated == {'cell': None, 'material': None}


def test_clip_box_from_cut_planes():
    state = rd.View3DState()
    assert state.clip_box() is None
    state.cut_planes['z'].update(enabled=True, position=2.0)
    state.cut_planes['x'].update(enabled=True, position=-1.0, flip=True)
    state.cut_planes['y'].update(position=5.0)
    assert state.clip_box() == ((-1.0, -math.inf, -math.inf),
                                (math.inf, math.inf, 2.0))


# Domain styles

def domain(color=(10, 20, 30), masked=False, highlight=False, opacity=1.0):
    return SimpleNamespace(color=color, masked=masked, highlight=highlight,
                           opacity=opacity)


def test_domain_styles_follow_2d_view_and_3d_visibility():
    view = SimpleNamespace(
        colorby='cell', materials={}, masking=True, highlighting=False,
        cells={1: domain(color=(255, 0, 0)), 2: domain(masked=True),
               3: domain(highlight=True), 4: domain(opacity=0.5),
               -1: domain()})

    def alphas(state=None):
        return {i: a for i, (_, a) in rd.domain_styles(view, state).items()}

    assert rd.domain_styles(view)[1] == ((255, 0, 0), 1.0)
    assert alphas() == {1: 1.0, 2: 0.0, 3: 1.0, 4: 0.5}
    # Domains that are not highlighted are ghosted
    view.highlighting = True
    assert alphas() == pytest.approx({1: 0.15, 2: 0.0, 3: 1.0, 4: 0.075})

    view.masking = view.highlighting = False
    state = rd.View3DState()
    state.hide('cell', {1})
    state.ghost('cell', {4})
    assert alphas(state) == pytest.approx({1: 0.0, 2: 1.0, 3: 1.0, 4: 0.075})
    state.isolate('cell', {1, 3})
    assert alphas(state) == pytest.approx({1: 1.0, 2: 0.0, 3: 1.0, 4: 0.0})
    state.show_all()
    assert alphas(state) == pytest.approx({1: 1.0, 2: 1.0, 3: 1.0, 4: 0.5})

    view.colorby = 'material'
    view.materials = {7: domain(color=(0, 0, 255))}
    assert rd.domain_styles(view, state) == {7: ((0, 0, 255), 1.0)}


def test_domain_styles_with_plotter_domains():
    plotmodel = pytest.importorskip('openmc_plotter.plotmodel')
    defaults = {i: plotmodel.DomainView(i, 'cell', (i, i, i))
                for i in (1, 2, 3)}
    cells = plotmodel.DomainViewDict(defaults)
    cells.set_opacity(2, 0.25)
    cells.set_masked(3, True)
    view = SimpleNamespace(colorby='cell', cells=cells, materials={},
                           masking=True, highlighting=False)
    assert rd.domain_styles(view) == {1: ((1, 1, 1), 1.0),
                                      2: ((2, 2, 2), 0.25),
                                      3: ((3, 3, 3), 0.0)}

    # Settings saved by older versions have no opacity
    old = plotmodel.DomainView(5, 'old', (1, 2, 3))
    del old.opacity
    assert pickle.loads(pickle.dumps(old)).opacity == 1.0


# Image processing

def test_edges_mark_nearer_side_of_domain_changes():
    ids = np.ones((4, 6), dtype=np.int32)
    ids[:, 3:] = 2
    depth = np.where(ids == 1, 10.0, 20.0)
    expected = np.zeros(ids.shape, dtype=bool)
    expected[:, 2] = True
    np.testing.assert_array_equal(rd.detect_edges(depth, ids), expected)
    np.testing.assert_array_equal(
        rd.detect_edges(depth[:, ::-1], ids[:, ::-1]), expected[:, ::-1])


def test_edges_at_depth_steps_but_not_on_slopes():
    ids = np.ones((5, 8), dtype=np.int32)
    columns = np.arange(8) * np.ones((5, 1))
    step = np.where(columns < 4, 10.0, 20.0)
    expected = np.zeros(ids.shape, dtype=bool)
    expected[:, 3] = True
    # Depth steps are only found with the size of a pixel
    assert not rd.detect_edges(step, ids).any()
    np.testing.assert_array_equal(
        rd.detect_edges(step, ids, footprint=0.1), expected)

    small_step = np.where(columns < 4, 10.0, 11.5)
    np.testing.assert_array_equal(rd.detect_edges(
        small_step, ids, footprint=0.1, sensitivity=1.0), expected)
    assert not rd.detect_edges(small_step, ids, footprint=0.1,
                               sensitivity=0.0).any()

    slope = 10.0 + 3.0 * columns
    assert not rd.detect_edges(slope, ids, footprint=0.1,
                               sensitivity=1.0).any()


def test_edges_outline_objects_and_creases():
    ids = np.full((7, 7), rd.MISS_ID, dtype=np.int32)
    depth = np.full((7, 7), np.inf)
    ids[2:5, 2:5] = 5
    depth[2:5, 2:5] = 10.0
    expected = np.zeros(ids.shape, dtype=bool)
    expected[2:5, 2:5] = True
    expected[3, 3] = False
    np.testing.assert_array_equal(
        rd.detect_edges(depth, ids, footprint=0.1), expected)

    ids = np.ones((4, 6), dtype=np.int32)
    normals = np.zeros((4, 6, 3))
    normals[:, :3] = (0.0, 0.0, 1.0)
    normals[:, 3:] = (1.0, 0.0, 0.0)
    expected = np.zeros(ids.shape, dtype=bool)
    expected[:, 2] = True
    np.testing.assert_array_equal(
        rd.detect_edges(np.full(ids.shape, 10.0), ids, normals), expected)


def test_depth_shading_fades_distant_surfaces():
    image = np.full((1, 4, 3), 200, dtype=np.uint8)
    depth = np.array([[10.0, 20.0, 30.0, np.inf]])
    shaded = rd.depth_shade(image, depth, (0, 0, 0), strength=1.0)
    np.testing.assert_array_equal(shaded[0, :, 0], [200, 130, 60, 200])
    assert rd.depth_shade(image, depth, (0, 0, 0), strength=0.0) is image


def test_postprocess_highlights_lost_rays():
    cam = top_camera()
    image = np.full((3, 3, 3), 100, dtype=np.uint8)
    ids = np.ones((3, 3), dtype=np.int32)
    ids[1, 1] = rd.LOST_RAY_ID
    out = rd.postprocess(image, np.full((3, 3), 10.0), ids, None, cam, 3,
                         edges=False, depth_shading=False)
    np.testing.assert_array_equal(out[1, 1], (201, 35, 201))
    out[1, 1] = 100
    assert np.all(out == 100)
    # The basic ray tracer has no buffers to post-process
    assert rd.postprocess(image, None, None, None, cam, 3) is image


# Overlays and picking

def test_point_visibility_uses_depth_buffer():
    cam = top_camera()
    depth = surface_left_half()
    points = [(-5.0, 0.0, 0.0),   # on the surface
              (-5.0, 0.0, -5.0),  # below the surface
              (5.0, 0.0, -5.0),   # nothing in front of it
              (-5.0, 0.0, 3.0)]   # above the surface
    xy, valid, visible = rd.point_visibility(cam, 40, 40, depth, points)
    np.testing.assert_allclose(xy[0], (10.5, 20.5))
    assert valid.all()
    assert visible.tolist() == [True, False, True, True]

    see_through = np.ones(depth.shape, dtype=bool)
    _, _, visible = rd.point_visibility(cam, 40, 40, depth, points,
                                        see_through)
    assert visible.all()


def test_track_overlay_splits_where_geometry_hides_track():
    cam = top_camera()
    depth = surface_left_half()
    below = [(-8.0, 0.0, -5.0), (8.0, 0.0, -5.0)]
    runs = rd.track_overlay(cam, 40, 40, depth, below)
    assert [visible for _, visible in runs] == [False, True]
    (hidden, _), (shown, _) = runs
    np.testing.assert_allclose(hidden[[0, -1]], [(4.5, 20.5), (20.5, 20.5)])
    np.testing.assert_allclose(shown[[0, -1]], [(20.5, 20.5), (36.5, 20.5)])

    def visibility(points, depth, see_through=None):
        return [v for _, v in rd.track_overlay(cam, 40, 40, depth, points,
                                               see_through)]

    assert visibility([(-8.0, 0.0, 5.0), (8.0, 0.0, 5.0)], depth) == [True]
    assert visibility(below, depth, np.ones(depth.shape, bool)) == [True]
    assert visibility(below, None) == [True]


def test_track_overlay_clips_segments_behind_camera():
    # The camera is at z = 10 and its near plane at z = 9.9
    cam = rd.Camera3D(distance=10.0, forward=(0.0, 0.0, -1.0),
                      up=(0.0, 1.0, 0.0), projection=rd.PERSPECTIVE,
                      fov=90.0)
    [(xy, visible)] = rd.track_overlay(cam, 40, 40, None,
                                       [(-1.0, 0.0, 0.0), (3.0, 0.0, 20.0)])
    assert visible and np.all(np.isfinite(xy))
    ends, _, _ = cam.project([(-1.0, 0.0, 0.0), (0.98, 0.0, 9.9)], 40, 40)
    np.testing.assert_allclose(xy[[0, -1]], ends)
    assert rd.track_overlay(cam, 40, 40, None,
                            [(0.0, 0.0, 15.0), (1.0, 0.0, 20.0)]) == []


def test_pick_finds_domain_and_position():
    depth = surface_left_half()
    ids = np.where(np.isfinite(depth), 3, rd.MISS_ID).astype(np.int32)
    result = rd.RenderResult(np.zeros((40, 40, 3), np.uint8), depth, ids,
                             top_camera(), 40, 40, 'cell',
                             rd.CAPABILITY_EXTENDED)
    hit = result.pick(10.7, 10.2)
    assert (hit.domain_id, hit.depth) == (3, 21.0)
    np.testing.assert_allclose(hit.position, (-5.0, 5.0, 0.0))
    miss = result.pick(30.5, 9.5)
    assert miss.domain_id == rd.MISS_ID and miss.position is None
    assert result.pick(-0.5, 9.5) is None and result.pick(40.5, 9.5) is None
    result.depth = result.ids = None
    assert result.pick(10.5, 9.5) is None


# Helpers

def test_raytrace_capability_detection():
    members = ('render', 'set_domain_styles', 'clip_box',
               'orthographic_width', 'show_cut_faces', 'background')

    def lib(*names):
        return SimpleNamespace(SolidRayTracePlot=type(
            'SolidRayTracePlot', (), dict.fromkeys(names)))

    assert rd.raytrace_capability(SimpleNamespace()) == rd.CAPABILITY_NONE
    assert rd.raytrace_capability(lib()) == rd.CAPABILITY_BASIC
    assert rd.raytrace_capability(lib(*members[1:])) == rd.CAPABILITY_BASIC
    assert rd.raytrace_capability(lib(*members)) == rd.CAPABILITY_EXTENDED


def test_bounding_box_helpers():
    ll, ur = rd.finite_bounding_box((-np.inf, 0.0, 5.0), (np.inf, 10.0, 1.0),
                                    (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    np.testing.assert_array_equal(ll, (-1.0, 0.0, -1.0))
    np.testing.assert_array_equal(ur, (1.0, 10.0, 1.0))

    ll, ur = rd.box_from_2d_view((1.0, 2.0, 3.0), 10.0, 4.0, 'xz')
    np.testing.assert_array_equal(ll, (-4.0, -3.0, 1.0))
    np.testing.assert_array_equal(ur, (6.0, 7.0, 5.0))
