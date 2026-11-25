"""Ray-traced 3D views of OpenMC models.

The 3D view is rendered by OpenMC's built-in Phong ray tracer
(:class:`openmc.lib.SolidRayTracePlot`). Rays are traced through the CSG
geometry itself, so the image is exact rather than a tessellation of the
geometry. This module contains everything that does not depend on Qt:

* :class:`Camera3D` holds the camera, implements rotate/pan/zoom and the
  projection used by the ray tracer, so that overlays (e.g. particle tracks)
  and picking line up with the rendered pixels.
* :class:`View3DState` holds the user's settings for the 3D view.
* :class:`Renderer3D` turns a :class:`RenderRequest` into a
  :class:`RenderResult` and post-processes the image (edge highlighting and
  depth-based shading).
* :func:`track_overlay` and :func:`point_visibility` project particle tracks
  onto a rendered image and use its depth buffer to find the parts of the
  tracks that are hidden by the geometry.

The level of support is detected at runtime, see :func:`raytrace_capability`.
OpenMC builds with the extended ray tracer API (``SolidRayTracePlot.render``
and friends) support orthographic projections, cut planes, partially
transparent domains and per-pixel depth, domain IDs and normals. The per-pixel
buffers are used for edge highlighting, depth shading, picking and hiding
overlays behind geometry. With the basic API only perspective images of
opaque domains are available.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

CAPABILITY_NONE = 'none'
CAPABILITY_BASIC = 'basic'
CAPABILITY_EXTENDED = 'extended'

COLOR_BY_MATERIAL = 0
COLOR_BY_CELL = 1

# Domain IDs reported by the extended ray tracer for pixels without a hit
MISS_ID = -1
LOST_RAY_ID = -2

PERSPECTIVE = 'perspective'
ORTHOGRAPHIC = 'orthographic'

LOST_RAY_COLOR = (255, 0, 255)

_EXTENDED_MEMBERS = ('render', 'set_domain_styles', 'clip_box',
                     'orthographic_width', 'show_cut_faces', 'background')

_WORLD_UP = np.array([0.0, 0.0, 1.0])

# Camera orientations as (forward, up). Names refer to the side of the model
# that the camera looks from.
VIEW_PRESETS = {
    '+x': ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    '-x': ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    '+y': ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    '-y': ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    '+z': ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    '-z': ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    'iso': ((-1.0, 1.0, -1.0), (-1.0, 1.0, 2.0)),
}
VIEW_PRESET_ORDER = ('+x', '-x', '+y', '-y', '+z', '-z', 'iso')
VIEW_PRESET_LABELS = {
    '+x': 'From +X',
    '-x': 'From -X',
    '+y': 'From +Y',
    '-y': 'From -Y',
    '+z': 'From +Z (top)',
    '-z': 'From -Z (bottom)',
    'iso': 'Isometric',
}

# Final render scale for each quality level. 'high' additionally renders at
# the device pixel ratio of HiDPI screens.
QUALITY_SCALES = {'fast': 0.5, 'standard': 1.0, 'high': 1.0}


def raytrace_capability(lib=None):
    """Determine how much of the 3D view the installed OpenMC supports.

    Parameters
    ----------
    lib : module, optional
        Module providing ``SolidRayTracePlot``. Defaults to
        :mod:`openmc.lib`.

    Returns
    -------
    str
        :data:`CAPABILITY_EXTENDED`, :data:`CAPABILITY_BASIC` or
        :data:`CAPABILITY_NONE`
    """
    if lib is None:
        try:
            import openmc.lib as lib
        except Exception:
            return CAPABILITY_NONE
    cls = getattr(lib, 'SolidRayTracePlot', None)
    if cls is None:
        return CAPABILITY_NONE
    if all(hasattr(cls, member) for member in _EXTENDED_MEMBERS):
        return CAPABILITY_EXTENDED
    return CAPABILITY_BASIC


def _vec(v):
    return np.asarray(v, dtype=float).reshape(3)


def _tuple(v):
    return tuple(float(x) for x in np.asarray(v, dtype=float).reshape(3))


def _normalize(v):
    v = _vec(v)
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError('Cannot normalize a vector of length {}'.format(norm))
    return v / norm


def _elevation(forward):
    """Angle in radians between a direction and the xy-plane

    Unlike asin(forward[2]), this stays accurate near the poles.
    """
    return math.atan2(forward[2], math.hypot(forward[0], forward[1]))


def _rotate(v, axis, angle):
    """Rotate a vector about an axis by an angle in radians (Rodrigues)"""
    k = _normalize(axis)
    v = _vec(v)
    c, s = math.cos(angle), math.sin(angle)
    return v * c + np.cross(k, v) * s + k * float(np.dot(k, v)) * (1.0 - c)


def _perpendicular_up(forward, up):
    """Unit vector perpendicular to forward that is closest to up"""
    f = _normalize(forward)
    u = _vec(up)
    u = u - float(np.dot(u, f)) * f
    norm = float(np.linalg.norm(u))
    if norm < 1e-9:
        # up is parallel to forward; use the axis least aligned with forward
        axis = np.eye(3)[int(np.argmin(np.abs(f)))]
        u = axis - float(np.dot(axis, f)) * f
        norm = float(np.linalg.norm(u))
    return u / norm


def finite_bounding_box(lower_left, upper_right, fallback_lower_left,
                        fallback_upper_right):
    """Replace infinite or invalid components of a bounding box

    Parameters
    ----------
    lower_left, upper_right : iterable of float
        Bounding box, e.g. from :func:`openmc.lib.global_bounding_box`
    fallback_lower_left, fallback_upper_right : iterable of float
        Finite box used for components that are not finite

    Returns
    -------
    tuple of numpy.ndarray
        Finite lower-left and upper-right corners
    """
    ll = np.array(lower_left, dtype=float).reshape(3)
    ur = np.array(upper_right, dtype=float).reshape(3)
    fll = np.array(fallback_lower_left, dtype=float).reshape(3)
    fur = np.array(fallback_upper_right, dtype=float).reshape(3)
    bad = ~(np.isfinite(ll) & np.isfinite(ur) & (ur >= ll))
    ll[bad] = fll[bad]
    ur[bad] = fur[bad]
    return ll, ur


def box_from_2d_view(origin, width, height, basis):
    """Bounding box covering the visible region of a 2D slice view

    The extent along the slice axis is the larger of the width and height.
    """
    half = max(float(width), float(height)) / 2.0
    half_extents = np.full(3, half)
    half_extents['xyz'.index(basis[0])] = float(width) / 2.0
    half_extents['xyz'.index(basis[1])] = float(height) / 2.0
    center = _vec(origin)
    return center - half_extents, center + half_extents


class Camera3D:
    """Camera of the 3D view

    The camera orbits a target point. It always keeps an up direction that is
    perpendicular to the viewing direction so that it can be passed to the
    ray tracer, which fails if the two are parallel.

    The projection methods reproduce the pixel-to-ray mapping of OpenMC's
    ray tracer. Display coordinates are in pixels of the rendered image with
    the origin at the top-left corner of the image, and pixel ``(i, j)``
    covering ``[i, i+1) x [j, j+1)``.

    Parameters
    ----------
    target : iterable of float
        Point the camera looks at and orbits around
    distance : float
        Distance from the camera to the target for perspective projections
    forward : iterable of float
        Viewing direction
    up : iterable of float
        Approximate up direction of the image
    projection : {'perspective', 'orthographic'}
        Projection type
    fov : float
        Horizontal field of view in degrees for perspective projections
    ortho_width : float
        Width of the view in [cm] for orthographic projections
    scene_center : iterable of float
        Center of the model's bounding sphere
    scene_radius : float
        Radius of the model's bounding sphere
    """

    def __init__(self, target=(0.0, 0.0, 0.0), distance=100.0,
                 forward=VIEW_PRESETS['iso'][0], up=VIEW_PRESETS['iso'][1],
                 projection=PERSPECTIVE, fov=40.0, ortho_width=100.0,
                 scene_center=(0.0, 0.0, 0.0), scene_radius=50.0):
        self.target = _tuple(target)
        self.distance = float(distance)
        self.projection = projection
        self.fov = float(fov)
        self.ortho_width = float(ortho_width)
        self.scene_center = _tuple(scene_center)
        self.scene_radius = float(scene_radius)
        self.forward = (0.0, 1.0, 0.0)
        self.up = (0.0, 0.0, 1.0)
        self.set_orientation(forward, up)

    def __setstate__(self, state):
        self.__init__()
        self.__dict__.update(state)

    def __eq__(self, other):
        if not isinstance(other, Camera3D):
            return NotImplemented
        return self.__dict__ == other.__dict__

    __hash__ = None

    def __repr__(self):
        return ('Camera3D(target={}, distance={:.6g}, forward={}, up={}, '
                'projection={!r}, fov={:.4g}, ortho_width={:.6g})'.format(
                    self.target, self.distance, self.forward, self.up,
                    self.projection, self.fov, self.ortho_width))

    def copy(self):
        """Return a copy of the camera"""
        new = Camera3D.__new__(Camera3D)
        # All attributes are immutable
        new.__dict__.update(self.__dict__)
        return new

    # Orientation

    @property
    def is_orthographic(self):
        return self.projection == ORTHOGRAPHIC

    @property
    def forward_vec(self):
        return np.array(self.forward)

    @property
    def up_vec(self):
        return np.array(self.up)

    @property
    def right_vec(self):
        return _normalize(np.cross(self.forward_vec, self.up_vec))

    def set_orientation(self, forward, up):
        """Set the viewing direction and the approximate up direction"""
        f = _normalize(forward)
        self.forward = _tuple(f)
        self.up = _tuple(_perpendicular_up(f, up))

    def set_preset(self, name):
        """Look at the target from one of :data:`VIEW_PRESETS`"""
        forward, up = VIEW_PRESETS[name]
        self.set_orientation(forward, up)

    @property
    def elevation(self):
        """Angle between the viewing direction and the xy-plane in degrees"""
        return math.degrees(_elevation(self.forward))

    # Position

    def ortho_eye_distance(self):
        """Distance of the eye from the target for orthographic projections

        The eye is placed outside of the bounding sphere of the model so that
        no part of the model is cut off by the image plane.
        """
        offset = float(np.linalg.norm(np.subtract(self.target,
                                                  self.scene_center)))
        return 2.0 * self.scene_radius + offset + 1.0

    @property
    def eye(self):
        """Position of the camera"""
        target = np.array(self.target)
        if self.is_orthographic:
            return target - self.forward_vec * self.ortho_eye_distance()
        return target - self.forward_vec * self.distance

    def view_width(self):
        """Width of the view in [cm] at the target"""
        if self.is_orthographic:
            return self.ortho_width
        return 2.0 * self.distance * math.tan(math.radians(self.fov) / 2.0)

    def pixel_size(self, width):
        """Size of a pixel in [cm] at the target for an image width"""
        return self.view_width() / max(1, int(width))

    def light_position(self, mode='headlight'):
        """Position of the light source

        Parameters
        ----------
        mode : {'headlight', 'fixed'}
            A headlight follows the camera and illuminates the model from
            slightly above and to the left of the camera. A fixed light is
            located above the model.
        """
        if mode == 'fixed':
            direction = _normalize((1.0, -2.0, 3.0))
            return (np.array(self.scene_center) +
                    direction * 4.0 * max(self.scene_radius, 1e-6))
        eye = self.eye
        dist = float(np.linalg.norm(np.array(self.target) - eye))
        return eye + (self.up_vec * 0.35 - self.right_vec * 0.25) * dist

    # Interaction

    def orbit(self, dx, dy, degrees_per_pixel=0.4):
        """Rotate the camera about the target like a turntable

        Horizontal motion rotates about the global z-axis, vertical motion
        tilts the camera. The tilt stops when looking straight up or down.

        Parameters
        ----------
        dx, dy : float
            Mouse motion in pixels. Dragging moves the model with the mouse.
        degrees_per_pixel : float
            Rotation per pixel of mouse motion
        """
        f, u = self.forward_vec, self.up_vec
        yaw = math.radians(-dx * degrees_per_pixel)
        if yaw:
            f = _rotate(f, _WORLD_UP, yaw)
            u = _rotate(u, _WORLD_UP, yaw)
        pitch = math.radians(-dy * degrees_per_pixel)
        if pitch:
            # Cameras that are upside down tilt freely. The tolerance keeps
            # rounding errors at the poles from disabling the limit.
            if u[2] >= -1e-9:
                elevation = _elevation(f)
                new_elevation = max(-math.pi / 2,
                                    min(math.pi / 2, elevation + pitch))
                pitch = new_elevation - elevation
            right = _normalize(np.cross(f, u))
            f = _rotate(f, right, pitch)
            u = _rotate(u, right, pitch)
        self.set_orientation(f, u)

    def pan(self, dx, dy, width):
        """Move the camera parallel to the image plane

        Parameters
        ----------
        dx, dy : float
            Mouse motion in pixels. The model moves with the mouse.
        width : int
            Width of the view in pixels
        """
        size = self.pixel_size(width)
        target = (np.array(self.target) - self.right_vec * dx * size +
                  self.up_vec * dy * size)
        self.target = _tuple(target)

    def zoom(self, factor, x=None, y=None, width=None, height=None):
        """Zoom by a factor, optionally keeping the point under the cursor

        Parameters
        ----------
        factor : float
            Ratio of the new and old view widths; values below one zoom in.
        x, y : float, optional
            Cursor position in pixels. The point on the target plane below
            the cursor stays below the cursor.
        width, height : int, optional
            Size of the view in pixels. Required if a cursor position is given.
        """
        factor = float(factor)
        if not np.isfinite(factor) or factor <= 0.0:
            return
        offset = np.zeros(3)
        if x is not None and y is not None and width and height:
            origin, direction = self.pixel_ray(x, y, width, height)
            f = self.forward_vec
            target = np.array(self.target)
            denom = float(np.dot(direction, f))
            if denom > 1e-9:
                t = float(np.dot(target - origin, f)) / denom
                offset = origin + t * direction - target
        scale = max(self.scene_radius, 1e-6)
        if self.is_orthographic:
            new_width = min(max(self.ortho_width * factor, 1e-7 * scale),
                            1e7 * scale)
            factor = new_width / self.ortho_width
            self.ortho_width = new_width
        else:
            new_distance = min(max(self.distance * factor, 1e-7 * scale),
                               1e7 * scale)
            factor = new_distance / self.distance
            self.distance = new_distance
        self.target = _tuple(np.array(self.target) + offset * (1.0 - factor))

    def center_on(self, point):
        """Orbit about a point without moving the view direction or zoom

        In a perspective view, the camera keeps the same distance to the
        plane through the point, so the image only shifts.
        """
        point = _vec(point)
        if not self.is_orthographic:
            depth = float(np.dot(point - self.eye, self.forward_vec))
            if depth > 1e-6 * max(self.scene_radius, 1e-6):
                self.distance = depth
        self.target = _tuple(point)

    def set_projection(self, projection):
        """Switch the projection, keeping the size of the view at the target"""
        if projection not in (PERSPECTIVE, ORTHOGRAPHIC):
            raise ValueError('Unknown projection: {}'.format(projection))
        if projection == self.projection:
            return
        tan_half = math.tan(math.radians(self.fov) / 2.0)
        if projection == ORTHOGRAPHIC:
            self.ortho_width = 2.0 * self.distance * tan_half
        else:
            self.distance = self.ortho_width / (2.0 * tan_half)
        self.projection = projection

    def set_fov(self, fov):
        """Change the field of view, keeping the size of the view at the target"""
        fov = min(max(float(fov), 1.0), 150.0)
        width = self.view_width()
        self.fov = fov
        if not self.is_orthographic:
            self.distance = width / (2.0 * math.tan(math.radians(fov) / 2.0))

    def fit(self, lower_left, upper_right, width, height, margin=1.05):
        """Look at the center of a box from a distance that shows all of it

        Parameters
        ----------
        lower_left, upper_right : iterable of float
            Finite corners of the box
        width, height : int
            Size of the view in pixels
        margin : float
            Extra space around the box relative to its size
        """
        ll, ur = _vec(lower_left), _vec(upper_right)
        center = 0.5 * (ll + ur)
        radius = 0.5 * float(np.linalg.norm(ur - ll))
        if not np.isfinite(radius) or radius <= 0.0:
            radius = 1.0
        width = max(1, int(width))
        height = max(1, int(height))
        self.scene_center = _tuple(center)
        self.scene_radius = radius
        self.target = _tuple(center)

        half_h = math.radians(self.fov) / 2.0
        half_v = math.atan(math.tan(half_h) * height / width)
        self.distance = radius / math.sin(min(half_h, half_v)) * margin
        self.ortho_width = 2.0 * radius * max(1.0, width / height) * margin

    # Projection

    def project(self, points, width, height):
        """Project points onto an image rendered with this camera

        Parameters
        ----------
        points : array_like
            Points with shape (N, 3)
        width, height : int
            Size of the image in pixels

        Returns
        -------
        xy : numpy.ndarray
            Display coordinates with shape (N, 2)
        depth : numpy.ndarray
            Distance along the camera ray, comparable to the depth buffer
        valid : numpy.ndarray
            Whether each point is in front of the camera
        """
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        f, u, r = self.forward_vec, self.up_vec, self.right_vec
        rel = points - self.eye
        along = rel @ f
        width = max(1, int(width))
        height = max(1, int(height))
        if self.is_orthographic:
            # Pixels are square and span ortho_width horizontally
            scale = width / self.ortho_width
            x = (rel @ r) * scale + width / 2.0 + 0.5
            y = height / 2.0 - (rel @ u) * scale + 0.5
            depth = along
            valid = np.isfinite(along)
        else:
            tan2 = 2.0 * math.tan(math.radians(self.fov) / 2.0)
            valid = along > 1e-12
            safe = np.where(valid, along, 1.0)
            h = width * (0.5 + (rel @ r) / safe / tan2)
            v = height / 2.0 - width * (rel @ u) / safe / tan2
            x = h + 0.5
            y = v + 0.5
            depth = np.linalg.norm(rel, axis=1)
        return np.column_stack([x, y]), depth, valid

    def pixel_ray(self, x, y, width, height):
        """Origin and unit direction of the ray through a display position

        This is the inverse of :meth:`project`. Pixel centers are at
        half-integer display coordinates.
        """
        f, u, r = self.forward_vec, self.up_vec, self.right_vec
        eye = self.eye
        width = max(1, int(width))
        height = max(1, int(height))
        if self.is_orthographic:
            pixel = self.ortho_width / width
            a = (x - 0.5 - width / 2.0) * pixel
            b = (height / 2.0 - (y - 0.5)) * pixel
            return eye + a * r + b * u, f
        tan2 = 2.0 * math.tan(math.radians(self.fov) / 2.0)
        a = (x - 0.5 - width / 2.0) * tan2 / width
        b = (height / 2.0 - (y - 0.5)) * tan2 / width
        return eye, _normalize(f + a * r + b * u)


def _default_cut_planes():
    return {axis: {'enabled': False, 'position': 0.0, 'flip': False}
            for axis in 'xyz'}


class View3DState:
    """Settings of the 3D view

    Instances are pickled with the plotter settings and saved views, so new
    attributes must be given defaults in :meth:`__init__`.
    """

    def __init__(self):
        self.camera = Camera3D()
        # Whether the camera has been fitted to the model
        self.camera_initialized = False
        self.scene_lower_left = (-50.0, -50.0, -50.0)
        self.scene_upper_right = (50.0, 50.0, 50.0)

        # Shading
        self.light = 'headlight'
        self.diffuse_fraction = 0.35
        self.background = None  # None uses the 2D background color

        # Post-processing
        self.edges = True
        self.edge_sensitivity = 0.5
        self.edge_thickness = 1
        self.edge_color = (0, 0, 0)
        self.depth_shading = True
        self.depth_strength = 0.5
        self.highlight_lost_rays = True

        # Cut planes, which are converted to the ray tracer's clip box
        self.cut_planes = _default_cut_planes()
        self.show_cut_faces = True

        # Opacity applied to ghosted domains and, when highlighting is
        # active, to domains that are not highlighted
        self.ghost_alpha = 0.15
        # 3D-only visibility, by domain kind ('cell' or 'material')
        self.hidden = {'cell': set(), 'material': set()}
        self.ghosted = {'cell': set(), 'material': set()}
        self.isolated = {'cell': None, 'material': None}

        # Rendering performance
        self.quality = 'standard'
        self.interactive_ms = 60.0

        # Overlays
        self.show_tracks = True
        self.show_axes = True

    def __setstate__(self, state):
        self.__init__()
        self.__dict__.update(state)
        self._normalize()

    def _normalize(self):
        planes = _default_cut_planes()
        for axis, plane in dict(self.cut_planes).items():
            if axis in planes and isinstance(plane, dict):
                planes[axis].update(plane)
        self.cut_planes = planes
        for name in ('hidden', 'ghosted'):
            value = getattr(self, name)
            setattr(self, name, {kind: set(value.get(kind) or ())
                                 for kind in ('cell', 'material')})
        isolated = self.isolated
        self.isolated = {kind: (None if isolated.get(kind) is None
                                else set(isolated[kind]))
                         for kind in ('cell', 'material')}

    def to_dict(self):
        """Settings as builtin types

        Settings files and saved views store this dictionary rather than the
        object, so that they can be read by versions of the plotter without
        this module.
        """
        data = copy.deepcopy({k: v for k, v in self.__dict__.items()
                              if k != 'camera'})
        data['camera'] = dict(self.camera.__dict__)
        return data

    @classmethod
    def from_dict(cls, data):
        """Create settings from :meth:`to_dict`, ignoring unknown keys"""
        state = cls()
        if not isinstance(data, dict):
            return state
        defaults = state.__dict__
        for key, value in data.items():
            if key != 'camera' and key in defaults:
                setattr(state, key, copy.deepcopy(value))
        camera = data.get('camera')
        if isinstance(camera, dict):
            cam = Camera3D()
            for key, value in camera.items():
                if key in cam.__dict__:
                    setattr(cam, key, value)
            cam.target = _tuple(cam.target)
            cam.scene_center = _tuple(cam.scene_center)
            for key in ('distance', 'fov', 'ortho_width', 'scene_radius'):
                setattr(cam, key, float(getattr(cam, key)))
            if cam.projection not in (PERSPECTIVE, ORTHOGRAPHIC):
                cam.projection = PERSPECTIVE
            cam.set_orientation(cam.forward, cam.up)
            state.camera = cam
        state.scene_lower_left = _tuple(state.scene_lower_left)
        state.scene_upper_right = _tuple(state.scene_upper_right)
        state._normalize()
        return state

    def clip_box(self):
        """Clip box for the enabled cut planes or None

        A cut plane removes the part of the model above its position along
        its axis, or below its position if it is flipped.
        """
        lower = [-math.inf] * 3
        upper = [math.inf] * 3
        enabled = False
        for i, axis in enumerate('xyz'):
            plane = self.cut_planes[axis]
            if not plane['enabled']:
                continue
            enabled = True
            if plane['flip']:
                lower[i] = float(plane['position'])
            else:
                upper[i] = float(plane['position'])
        if not enabled:
            return None
        return tuple(lower), tuple(upper)

    def hide(self, kind, ids):
        self.hidden[kind].update(ids)
        self.ghosted[kind].difference_update(ids)

    def ghost(self, kind, ids):
        self.ghosted[kind].update(ids)
        self.hidden[kind].difference_update(ids)

    def isolate(self, kind, ids):
        self.isolated[kind] = set(ids)
        self.hidden[kind].difference_update(ids)
        self.ghosted[kind].difference_update(ids)

    def show_all(self, kind=None):
        kinds = (kind,) if kind else ('cell', 'material')
        for k in kinds:
            self.hidden[k].clear()
            self.ghosted[k].clear()
            self.isolated[k] = None

    def has_visibility_overrides(self, kind):
        return bool(self.hidden[kind] or self.ghosted[kind] or
                    self.isolated[kind] is not None)


def domain_kind(view):
    """Domain type shown in 3D for a 2D view's color-by setting"""
    return 'cell' if getattr(view, 'colorby', 'material') == 'cell' else 'material'


def to_rgb(color, default=(200, 200, 200)):
    """Convert an RGB tuple or SVG color name to a tuple of int"""
    if color is None:
        return tuple(default)
    if isinstance(color, str):
        import openmc.plots
        return tuple(int(c) for c in
                     openmc.plots._SVG_COLORS[color.lower()][:3])
    return tuple(min(255, max(0, int(c))) for c in tuple(color)[:3])


def domain_styles(view, state=None):
    """Color and opacity of every cell or material for the 3D view

    The colors, masking and highlighting of the 2D view are reused: masked
    domains are hidden and, when highlighting is active, domains that are not
    highlighted are ghosted. The 3D-only visibility of ``state`` is applied on
    top of that.

    Parameters
    ----------
    view : openmc_plotter.plotmodel.PlotView
        View providing colors, opacities, masking and highlighting
    state : View3DState, optional
        3D view settings

    Returns
    -------
    dict
        Mapping of domain ID to ((r, g, b), alpha)
    """
    kind = domain_kind(view)
    domains = view.cells if kind == 'cell' else view.materials
    defaults = getattr(domains, 'defaults', domains)
    ids = [i for i in defaults.keys() if int(i) >= 0]

    ghost_alpha = state.ghost_alpha if state is not None else 0.15
    hidden = state.hidden[kind] if state is not None else set()
    ghosted = state.ghosted[kind] if state is not None else set()
    isolated = state.isolated[kind] if state is not None else None

    masking = bool(getattr(view, 'masking', False))
    highlighting = bool(getattr(view, 'highlighting', False))
    domain_list = [domains[i] for i in ids]
    any_highlight = highlighting and any(d.highlight for d in domain_list)

    styles = {}
    for i, domain in zip(ids, domain_list):
        alpha = float(getattr(domain, 'opacity', 1.0))
        if (masking and domain.masked) or i in hidden or \
           (isolated is not None and i not in isolated):
            alpha = 0.0
        elif (any_highlight and not domain.highlight) or i in ghosted:
            alpha *= ghost_alpha
        styles[int(i)] = (to_rgb(domain.color), min(1.0, max(0.0, alpha)))
    return styles


@dataclass
class RenderRequest:
    """Everything needed to render one image of the 3D view"""
    width: int
    height: int
    camera: Camera3D
    color_by: str
    styles: Dict[int, Tuple[Tuple[int, int, int], float]]
    styles_version: int = 0
    background: Tuple[int, int, int] = (50, 50, 50)
    clip_box: Optional[tuple] = None
    show_cut_faces: bool = True
    diffuse_fraction: float = 0.35
    light: str = 'headlight'
    edges: bool = True
    edge_sensitivity: float = 0.5
    edge_thickness: int = 1
    edge_color: Tuple[int, int, int] = (0, 0, 0)
    depth_shading: bool = True
    depth_strength: float = 0.5
    highlight_lost_rays: bool = True
    final: bool = True
    scale: float = 1.0
    request_id: int = 0
    epoch: int = 0


@dataclass
class PickResult:
    """Domain and position below a pixel of a rendered image"""
    domain_id: int
    position: Optional[np.ndarray]
    depth: float


@dataclass
class RenderResult:
    """Rendered image and per-pixel buffers of the 3D view

    The depth and IDs are None unless the extended ray tracer API is
    available. ``camera``, ``width`` and ``height`` describe the projection
    of the image, see :meth:`Camera3D.project`.
    """
    image: np.ndarray
    depth: Optional[np.ndarray]
    ids: Optional[np.ndarray]
    camera: Camera3D
    width: int
    height: int
    color_by: str
    capability: str
    final: bool = True
    scale: float = 1.0
    request_id: int = 0
    epoch: int = 0
    seconds: float = 0.0
    rays: int = 0
    n_lost_rays: int = 0

    def pick(self, x, y):
        """Domain and position below a display position of the image

        Returns None if the position is outside the image or no per-pixel
        buffers are available.
        """
        if self.ids is None or self.depth is None:
            return None
        ix, iy = int(math.floor(x)), int(math.floor(y))
        if not (0 <= ix < self.width and 0 <= iy < self.height):
            return None
        domain_id = int(self.ids[iy, ix])
        depth = float(self.depth[iy, ix])
        position = None
        if np.isfinite(depth):
            origin, direction = self.camera.pixel_ray(
                ix + 0.5, iy + 0.5, self.width, self.height)
            position = origin + depth * direction
        return PickResult(domain_id, position, depth)


class Renderer3D:
    """Renders the 3D view with :class:`openmc.lib.SolidRayTracePlot`

    The ray-traced plot is created on first use and re-created whenever the
    generation reported by ``generation_fn`` changes, e.g. after the model
    has been reloaded. Calls into :mod:`openmc.lib` are made while holding
    ``lock`` so that they do not overlap with other users of the library;
    post-processing happens outside of the lock.

    Parameters
    ----------
    lock : threading.RLock, optional
        Lock serializing calls into :mod:`openmc.lib`
    generation_fn : callable, optional
        Returns a value that changes whenever :mod:`openmc.lib` is
        re-initialized
    lib : module, optional
        Module providing ``SolidRayTracePlot``. Defaults to
        :mod:`openmc.lib`.
    """

    def __init__(self, lock=None, generation_fn: Optional[Callable] = None,
                 lib=None):
        if lib is None:
            try:
                import openmc.lib as lib
            except Exception:
                lib = None
        self._lib = lib
        self.lock = lock if lock is not None else threading.RLock()
        self._generation_fn = generation_fn or (lambda: 0)
        self.capability = raytrace_capability(lib) if lib is not None \
            else CAPABILITY_NONE
        self._plot = None
        self._generation = None
        self._color_by = None
        self._styles = {}
        self._styles_version = None

    @property
    def available(self):
        return self.capability != CAPABILITY_NONE

    @property
    def extended(self):
        return self.capability == CAPABILITY_EXTENDED

    def invalidate(self):
        """Forget the ray-traced plot, e.g. before :mod:`openmc.lib` is
        finalized"""
        with self.lock:
            self._plot = None
            self._generation = None
            self._color_by = None
            self._styles = {}
            self._styles_version = None

    def bounding_box(self):
        """Global bounding box of the model, possibly with infinite values"""
        with self.lock:
            return self._lib.global_bounding_box()

    def _ensure_plot(self, color_by):
        generation = self._generation_fn()
        if self._plot is None or generation != self._generation:
            self._plot = self._lib.SolidRayTracePlot()
            self._generation = generation
            self._color_by = None
        if color_by != self._color_by:
            plot = self._plot
            plot.color_by = (COLOR_BY_CELL if color_by == 'cell'
                             else COLOR_BY_MATERIAL)
            plot.set_default_colors()
            plot.set_all_opaque()
            self._color_by = color_by
            self._styles = {}
            self._styles_version = None
        return self._plot

    def _apply_styles(self, plot, styles, version):
        if version is not None and version == self._styles_version:
            return
        changed = {i: s for i, s in styles.items() if self._styles.get(i) != s}
        if changed:
            ids = list(changed)
            applied = False
            if self.extended:
                try:
                    plot.set_domain_styles(
                        ids, [changed[i][0] for i in ids],
                        [changed[i][1] for i in ids])
                    applied = True
                except Exception:
                    # Fall back to individual domains so that one unknown ID
                    # does not prevent the others from being styled
                    applied = False
            if not applied:
                for i in ids:
                    color, alpha = changed[i]
                    try:
                        if self.extended:
                            plot.set_domain_styles([i], [color], [alpha])
                        else:
                            plot.set_color(i, color)
                            plot.set_visibility(i, alpha > 0.0)
                    except Exception:
                        continue
            self._styles.update(changed)
        self._styles_version = version

    def render(self, request: RenderRequest) -> RenderResult:
        """Render an image for a request

        Raises
        ------
        RuntimeError
            If the installed OpenMC has no ray tracer
        """
        if not self.available:
            raise RuntimeError('The installed version of OpenMC does not '
                               'provide openmc.lib.SolidRayTracePlot.')
        camera = request.camera
        if camera.is_orthographic and not self.extended:
            camera = camera.copy()
            camera.set_projection(PERSPECTIVE)
        width = max(1, int(request.width))
        height = max(1, int(request.height))

        depth = ids = normals = None
        with self.lock:
            # Time spent waiting for the lock is not part of the render time
            start = time.perf_counter()
            plot = self._ensure_plot(request.color_by)
            self._apply_styles(plot, request.styles, request.styles_version)
            plot.pixels = (width, height)
            plot.camera_position = _tuple(camera.eye)
            plot.look_at = camera.target
            plot.up = camera.up
            plot.fov = camera.fov
            plot.light_position = _tuple(camera.light_position(request.light))
            plot.diffuse_fraction = min(1.0, max(0.0,
                                                 request.diffuse_fraction))
            if self.extended:
                plot.orthographic_width = (camera.ortho_width
                                           if camera.is_orthographic else None)
                plot.clip_box = request.clip_box
                plot.show_cut_faces = request.show_cut_faces
                plot.background = to_rgb(request.background)
                result = plot.render(depth=True, ids=True,
                                     normals=request.edges)
                image, depth, ids, normals = result
            else:
                plot.update_view()
                image = plot.create_image()

        n_lost = int(np.count_nonzero(ids == LOST_RAY_ID)) if ids is not None \
            else 0
        image = postprocess(image, depth, ids, normals, camera, width,
                            background=to_rgb(request.background),
                            edges=request.edges,
                            edge_sensitivity=request.edge_sensitivity,
                            edge_thickness=request.edge_thickness,
                            edge_color=request.edge_color,
                            depth_shading=request.depth_shading,
                            depth_strength=request.depth_strength,
                            highlight_lost_rays=request.highlight_lost_rays)
        seconds = time.perf_counter() - start

        return RenderResult(image=image, depth=depth, ids=ids, camera=camera,
                            width=width, height=height,
                            color_by=request.color_by,
                            capability=self.capability, final=request.final,
                            scale=request.scale, request_id=request.request_id,
                            epoch=request.epoch, seconds=seconds,
                            rays=width * height,
                            n_lost_rays=n_lost)


# Post-processing

def _neighbor_pairs(array, axis):
    if axis == 0:
        return array[:-1], array[1:]
    return array[:, :-1], array[:, 1:]


def _mark_nearer(edges, pairs, depth, axis):
    """Mark the nearer pixel of each pair of neighbors flagged in pairs"""
    near_a, near_b = _neighbor_pairs(depth, axis)
    a = pairs & (near_a <= near_b)
    b = pairs & (near_b < near_a)
    if axis == 0:
        edges[:-1] |= a
        edges[1:] |= b
    else:
        edges[:, :-1] |= a
        edges[:, 1:] |= b


def _filled_depth(depth):
    """Depth with misses replaced by a large finite value"""
    finite = np.isfinite(depth)
    far = float(depth[finite].max()) * 2.0 + 1.0 if finite.any() else 1.0
    return np.where(finite, depth, far).astype(np.float64), finite


def pixel_footprint(depth, camera, width):
    """Size in [cm] of the area covered by a pixel at a depth"""
    if camera.is_orthographic:
        return camera.ortho_width / max(1, int(width))
    tan2 = 2.0 * math.tan(math.radians(camera.fov) / 2.0)
    return depth * tan2 / max(1, int(width))


def detect_edges(depth, ids, normals=None, footprint=None, sensitivity=0.5):
    """Find edges in the per-pixel buffers of a rendered image

    Edges are drawn where the visible domain changes, where the depth is
    discontinuous and, if normals are given, where the surface normal changes
    abruptly. Only the pixel nearer to the camera of each pair of pixels on
    either side of an edge is marked, which keeps edges one pixel wide.

    Parameters
    ----------
    depth : numpy.ndarray
        Distance along the camera rays, infinite for misses
    ids : numpy.ndarray
        Domain IDs of the visible surfaces
    normals : numpy.ndarray, optional
        Surface normals with shape (height, width, 3), zero for misses
    footprint : float or numpy.ndarray, optional
        Size of a pixel in [cm] at the depth of each pixel. Required to
        detect depth discontinuities.
    sensitivity : float
        Value between 0 and 1. Higher values find more edges.

    Returns
    -------
    numpy.ndarray
        Boolean mask of edge pixels
    """
    sensitivity = min(1.0, max(0.0, float(sensitivity)))
    edges = np.zeros(ids.shape, dtype=bool)
    d, finite = _filled_depth(depth)

    for axis in (0, 1):
        id_a, id_b = _neighbor_pairs(ids, axis)
        _mark_nearer(edges, id_a != id_b, d, axis)

    if footprint is not None:
        # A positive second difference means a pixel is nearer than the
        # average of its neighbors, i.e. on the near side of a step
        factor = 20.0 - 18.0 * sensitivity
        threshold = factor * np.broadcast_to(np.asarray(footprint, float),
                                             d.shape)
        if d.shape[1] > 2:
            d2 = d[:, :-2] + d[:, 2:] - 2.0 * d[:, 1:-1]
            edges[:, 1:-1] |= (d2 > threshold[:, 1:-1]) & finite[:, 1:-1]
        if d.shape[0] > 2:
            d2 = d[:-2] + d[2:] - 2.0 * d[1:-1]
            edges[1:-1] |= (d2 > threshold[1:-1]) & finite[1:-1]

    if normals is not None:
        cos_max = math.cos(math.radians(60.0 - 45.0 * sensitivity))
        valid = np.any(normals != 0.0, axis=-1)
        for axis in (0, 1):
            n_a, n_b = _neighbor_pairs(normals, axis)
            v_a, v_b = _neighbor_pairs(valid, axis)
            dot = np.einsum('ijk,ijk->ij', n_a, n_b)
            _mark_nearer(edges, v_a & v_b & (dot < cos_max), d, axis)
    return edges


def dilate(mask, iterations=1):
    """Grow a boolean mask by a number of pixels"""
    out = mask
    for _ in range(max(0, int(iterations))):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def apply_edges(image, mask, color=(0, 0, 0), opacity=0.85):
    """Blend an edge color into the masked pixels of an image"""
    if not mask.any():
        return image
    out = image.copy()
    blended = (image[mask].astype(np.float32) * (1.0 - opacity) +
               np.asarray(color, np.float32) * opacity)
    out[mask] = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
    return out


def depth_shade(image, depth, background, strength=0.5, max_fog=0.7):
    """Blend distant surfaces toward the background color

    Depths are normalized between the 2nd and 98th percentiles of the visible
    depths, so the effect adapts to the part of the model in view.
    """
    strength = min(1.0, max(0.0, float(strength)))
    finite = np.isfinite(depth)
    if strength <= 0.0 or np.count_nonzero(finite) < 2:
        return image
    lo, hi = np.percentile(depth[finite], [2.0, 98.0])
    if not hi > lo:
        return image
    t = np.zeros(depth.shape, dtype=np.float32)
    t[finite] = np.clip((depth[finite] - lo) / (hi - lo), 0.0, 1.0)
    fog = (max_fog * strength * t)[..., None]
    out = (image.astype(np.float32) * (1.0 - fog) +
           np.asarray(background, np.float32) * fog)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def postprocess(image, depth, ids, normals, camera, width, *,
                background=(50, 50, 50), edges=True, edge_sensitivity=0.5,
                edge_thickness=1, edge_color=(0, 0, 0), depth_shading=True,
                depth_strength=0.5, highlight_lost_rays=True):
    """Apply depth shading, lost-ray highlighting and edges to an image

    Effects that need the depth and ID buffers of the extended ray tracer are
    skipped when the buffers are None.
    """
    if depth_shading and depth is not None:
        image = depth_shade(image, depth, background, depth_strength)
    if highlight_lost_rays and ids is not None:
        lost = ids == LOST_RAY_ID
        if lost.any():
            image = apply_edges(image, lost, LOST_RAY_COLOR, 0.65)
    if edges and ids is not None and depth is not None:
        footprint = pixel_footprint(
            np.where(np.isfinite(depth), depth, 0.0), camera, width)
        mask = detect_edges(depth, ids, normals, footprint, edge_sensitivity)
        mask = dilate(mask, int(edge_thickness) - 1)
        image = apply_edges(image, mask, edge_color)
    return image


# Overlays

def occlusion_tolerance(depth, camera, width):
    """Depth difference below which a point counts as lying on a surface

    Points of particle tracks often lie exactly on surfaces, e.g. where a
    particle crosses a surface or got lost, so they are compared to the depth
    buffer with a tolerance of a few pixel footprints.
    """
    depth = np.abs(np.asarray(depth, dtype=float))
    return 1e-3 * depth + 2.0 * pixel_footprint(depth, camera, width)


def point_visibility(camera, width, height, depth, points, see_through=None):
    """Project points onto an image and test whether geometry hides them

    Parameters
    ----------
    camera : Camera3D
        Camera of the image
    width, height : int
        Size of the image in pixels
    depth : numpy.ndarray or None
        Depth buffer of the image. Without one, no point is hidden.
    points : array_like
        Points with shape (N, 3)
    see_through : numpy.ndarray, optional
        Boolean mask of pixels whose nearest surface does not hide what is
        behind it, e.g. because it is partially transparent

    Returns
    -------
    xy : numpy.ndarray
        Display coordinates with shape (N, 2)
    valid : numpy.ndarray
        Whether each point is in front of the camera
    visible : numpy.ndarray
        Whether each point is in front of the camera and not hidden
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    xy, d, valid = camera.project(points, width, height)
    valid = valid & np.all(np.isfinite(xy), axis=1)
    visible = valid.copy()
    if depth is None or not valid.any():
        return xy, valid, visible

    rows, cols = depth.shape[:2]
    ix = np.full(len(points), -1, dtype=np.int64)
    iy = np.full(len(points), -1, dtype=np.int64)
    ix[valid] = np.floor(np.clip(xy[valid, 0], -1.0, cols)).astype(np.int64)
    iy[valid] = np.floor(np.clip(xy[valid, 1], -1.0, rows)).astype(np.int64)
    inside = valid & (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows)
    if not inside.any():
        return xy, valid, visible

    buffer = depth[iy[inside], ix[inside]].astype(float)
    d_in = d[inside]
    hidden = np.isfinite(buffer) & \
        (d_in > buffer + occlusion_tolerance(d_in, camera, width))
    if see_through is not None:
        hidden &= ~see_through[iy[inside], ix[inside]]
    visible[inside] = ~hidden
    return xy, valid, visible


def track_overlay(camera, width, height, depth, points, see_through=None,
                  px_step=6.0, max_subdivisions=64):
    """Project a polyline onto an image and split it by visibility

    Each segment is divided into pieces a few pixels long and the midpoint of
    each piece is tested against the depth buffer, so that tracks entering
    and leaving the geometry are drawn correctly. In perspective views,
    segments are clipped to a plane in front of the camera.

    Parameters
    ----------
    camera : Camera3D
        Camera of the image
    width, height : int
        Size of the image in pixels
    depth : numpy.ndarray or None
        Depth buffer of the image. Without one, the whole track is visible.
    points : array_like
        Vertices of the polyline with shape (N, 3)
    see_through : numpy.ndarray, optional
        Boolean mask of pixels that do not hide what is behind them
    px_step : float
        Approximate length of the pieces in pixels
    max_subdivisions : int
        Maximum number of pieces per segment

    Returns
    -------
    list of tuple
        Runs of connected pieces as (xy, visible) where xy holds display
        coordinates with shape (M, 2), M >= 2
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) < 2:
        return []
    p0, p1 = pts[:-1], pts[1:]
    keep = np.all(np.isfinite(p0), axis=1) & np.all(np.isfinite(p1), axis=1)
    clip0 = np.zeros(len(p0), dtype=bool)
    clip1 = np.zeros(len(p0), dtype=bool)
    if not camera.is_orthographic:
        eye, f = camera.eye, camera.forward_vec
        near = 1e-2 * camera.distance
        with np.errstate(invalid='ignore'):
            a0 = (p0 - eye) @ f
            a1 = (p1 - eye) @ f
        keep &= ~((a0 < near) & (a1 < near))
        clip0 = keep & (a0 < near)
        clip1 = keep & (a1 < near)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(clip0 | clip1, (near - a0) / (a1 - a0), 0.0)
            # At most one end of a kept segment is behind the near plane
            clipped = pts[:-1] + t[:, None] * (pts[1:] - pts[:-1])
        p0 = np.where(clip0[:, None], clipped, p0)
        p1 = np.where(clip1[:, None], clipped, p1)

    segments = np.flatnonzero(keep)
    if len(segments) == 0:
        return []
    a, b = p0[segments], p1[segments]

    ends_a, _, _ = camera.project(a, width, height)
    ends_b, _, _ = camera.project(b, width, height)
    length = np.hypot(*(ends_b - ends_a).T)
    length = np.where(np.isfinite(length), length, 0.0)
    n = np.clip(np.ceil(length / max(px_step, 1e-6)), 1,
                max(1, int(max_subdivisions))).astype(np.int64)

    # Pieces of all segments
    seg = np.repeat(np.arange(len(segments)), n)
    first = np.repeat(np.cumsum(n) - n, n)
    k = np.arange(len(seg)) - first
    ta = (k / n[seg])[:, None]
    tb = ((k + 1) / n[seg])[:, None]
    delta = b[seg] - a[seg]
    qa = a[seg] + ta * delta
    qb = a[seg] + tb * delta
    xy_a, _, _ = camera.project(qa, width, height)
    xy_b, _, _ = camera.project(qb, width, height)
    _, _, visible = point_visibility(camera, width, height, depth,
                                     0.5 * (qa + qb), see_through)

    # A run ends where the visibility changes or the polyline is broken
    breaks = np.zeros(len(seg), dtype=bool)
    breaks[0] = True
    breaks[1:] = visible[1:] != visible[:-1]
    seg_start = np.zeros(len(seg), dtype=bool)
    seg_start[np.cumsum(n) - n] = True
    original = segments[seg]
    connected = np.zeros(len(seg), dtype=bool)
    connected[1:] = ((original[1:] == original[:-1] + 1) &
                     ~clip1[original[:-1]] & ~clip0[original[1:]])
    breaks |= seg_start & ~connected

    runs = []
    edges = np.r_[np.flatnonzero(breaks), len(seg)]
    for s, e in zip(edges[:-1], edges[1:]):
        xy = np.vstack([xy_a[s:s + 1], xy_b[s:e]])
        if np.all(np.isfinite(xy)):
            runs.append((xy, bool(visible[s])))
    return runs
