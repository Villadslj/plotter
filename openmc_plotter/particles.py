"""Particle tracks for debugging lost particles

When OpenMC loses a particle, e.g. because of a gap or overlap in the
geometry, it writes a particle restart file named
``particle_<batch>_<id>.h5``. This module reads those files, as well as track
files (``tracks.h5``) written with ``openmc -t``, into
:class:`ParticleTrackData` objects that the plotter draws on top of the 2D and
3D views.

Restart files written by OpenMC builds that support
:attr:`openmc.Settings.lost_particle_history` contain the state of the
particle when it was lost and its most recent steps. Older files only contain
the source site of the particle. Those particles can be replayed with
``openmc -r <file> -t`` (see :class:`ReplayRunner`), which transports the
particle again and records its full track.
"""

from __future__ import annotations

import math
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# Particle types are stored as PDG numbers in current files and as indices
# into OpenMC's particle type enumeration in older files
PARTICLE_TYPES = {
    0: 'neutron', 1: 'photon', 2: 'electron', 3: 'positron',
    2112: 'neutron', 22: 'photon', 11: 'electron', -11: 'positron',
}

KIND_HISTORY = 'lost-history'
KIND_LOST_STATE = 'lost-state'
KIND_SOURCE = 'source-only'
KIND_REPLAYED = 'replayed'
KIND_TRACK_FILE = 'track-file'

KIND_LABELS = {
    KIND_HISTORY: 'Lost (with history)',
    KIND_LOST_STATE: 'Lost (final step)',
    KIND_SOURCE: 'Lost (source only)',
    KIND_REPLAYED: 'Lost (replayed)',
    KIND_TRACK_FILE: 'Track',
}

# Colors of particle tracks, cycled in the order particles are loaded
TRACK_PALETTE = (
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (23, 190, 207), (188, 189, 34),
    (127, 127, 127), (214, 39, 40),
)

BIRTH_COLOR = (0, 170, 0)
LOSS_COLOR = (230, 0, 0)

# Display options for particle tracks and their defaults
PARTICLE_OPTION_DEFAULTS = {
    # Tracks in the 3D view are toggled with View3DState.show_tracks
    'show2d': True,
    'showSteps': False,
    'showMarkers': True,
    'showLabels': True,
    # Half-thickness in [cm] of the slab around a 2D slice in which tracks
    # are drawn solid. Zero selects 1% of the view width.
    'sliceTolerance': 0.0,
    'lineWidth': 1.5,
    # Draw parts of 3D tracks hidden behind geometry dashed instead of
    # hiding them
    'xray': True,
    'threads': 1,
}

_RESTART_NAME = re.compile(r'particle_(\d+)_(\d+)\.h5$')
_TRACK_NAME = re.compile(r'track_(\d+)_(\d+)_(\d+)$')


def particle_name(code):
    """Name of a particle type stored in a restart or track file"""
    try:
        code = int(np.asarray(code).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return str(code)
    return PARTICLE_TYPES.get(code, str(code))


def particle_option_defaults():
    return dict(PARTICLE_OPTION_DEFAULTS)


def _decode(value):
    if value is None:
        return ''
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ''
        value = value.reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode(errors='replace').rstrip('\x00').strip()
    return str(value).strip()


def _read(group, name, default=None):
    if name not in group:
        return default
    return group[name][()]


def _vector(value):
    if value is None:
        return None
    value = np.asarray(value, dtype=float).reshape(-1)
    return value[:3].copy() if value.size >= 3 else None


def _scalar(value, default=math.nan):
    if value is None:
        return default
    try:
        return float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return default


def _fmt_vector(v, fmt='{:.6g}'):
    if v is None:
        return 'n/a'
    return '(' + ', '.join(fmt.format(x) for x in v) + ')'


@dataclass
class Track:
    """Consecutive states of one particle, e.g. a primary or a secondary"""
    particle: str
    positions: np.ndarray
    directions: Optional[np.ndarray] = None
    energies: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    cell_ids: Optional[np.ndarray] = None
    cell_instances: Optional[np.ndarray] = None
    material_ids: Optional[np.ndarray] = None

    def __len__(self):
        return len(self.positions)

    @classmethod
    def from_states(cls, states, particle):
        """Create a track from an array of OpenMC's TrackState type"""
        names = states.dtype.names or ()

        def xyz(name):
            if name not in names:
                return None
            v = states[name]
            return np.column_stack([v['x'], v['y'], v['z']]).astype(float)

        def column(name, dtype=float):
            return states[name].astype(dtype) if name in names else None

        return cls(particle=particle, positions=xyz('r'),
                   directions=xyz('u'), energies=column('E'),
                   times=column('time'), weights=column('wgt'),
                   cell_ids=column('cell_id', int),
                   cell_instances=column('cell_instance', int),
                   material_ids=column('material_id', int))

    @classmethod
    def segment(cls, start, end, particle):
        """Track consisting of a single straight segment"""
        return cls(particle=particle,
                   positions=np.array([start, end], dtype=float))


@dataclass
class LostState:
    """State of a particle when it was lost"""
    message: str
    position: Optional[np.ndarray]
    direction: Optional[np.ndarray] = None
    energy: float = math.nan
    time: float = math.nan
    weight: float = math.nan
    last_position: Optional[np.ndarray] = None
    last_direction: Optional[np.ndarray] = None
    last_cells: List[int] = field(default_factory=list)


@dataclass
class ParticleTrackData:
    """A particle loaded from a restart or track file"""
    label: str
    kind: str
    path: Optional[str] = None
    particle_id: Optional[int] = None
    batch: Optional[int] = None
    generation: Optional[int] = None
    particle: str = 'neutron'
    run_mode: str = ''
    tracks: List[Track] = field(default_factory=list)
    birth: Optional[np.ndarray] = None
    birth_direction: Optional[np.ndarray] = None
    birth_energy: float = math.nan
    lost: Optional[LostState] = None
    restart_path: Optional[str] = None
    visible: bool = True
    color: tuple = TRACK_PALETTE[0]
    notes: List[str] = field(default_factory=list)

    @property
    def can_replay(self):
        return self.restart_path is not None

    @property
    def n_states(self):
        return sum(len(t) for t in self.tracks)

    @property
    def loss_position(self):
        if self.lost is not None and self.lost.position is not None:
            return self.lost.position
        return None

    @property
    def loss_direction(self):
        if self.lost is not None:
            return self.lost.direction
        return None

    def key_position(self):
        """Most relevant position: where the particle was lost or born"""
        if self.loss_position is not None:
            return self.loss_position
        if self.tracks and len(self.tracks[-1]):
            return self.tracks[-1].positions[-1]
        return self.birth

    def all_positions(self):
        """All positions of the particle as an array with shape (N, 3)"""
        points = [t.positions for t in self.tracks if len(t)]
        for p in (self.birth, self.loss_position):
            if p is not None:
                points.append(np.reshape(p, (1, 3)))
        if not points:
            return np.zeros((0, 3))
        return np.vstack(points)

    def summary(self):
        """One line describing the particle"""
        text = '{} {}'.format(self.particle.capitalize(),
                              self.particle_id if self.particle_id is not None
                              else '?')
        if self.batch is not None:
            text += ' (batch {})'.format(self.batch)
        return text

    def describe(self):
        """Multi-line description used in tooltips and the details pane"""
        lines = [self.summary(), 'Kind: {}'.format(
            KIND_LABELS.get(self.kind, self.kind))]
        if self.path:
            lines.append('File: {}'.format(self.path))
        if self.run_mode:
            lines.append('Run mode: {}'.format(self.run_mode))
        if self.generation is not None:
            lines.append('Generation: {}'.format(self.generation))
        if self.birth is not None:
            lines.append('Source position: {}'.format(_fmt_vector(self.birth)))
            lines.append('Source direction: {}'.format(
                _fmt_vector(self.birth_direction)))
            if np.isfinite(self.birth_energy):
                lines.append('Source energy: {:.6g} eV'.format(
                    self.birth_energy))
        lost = self.lost
        if lost is not None:
            if lost.message:
                lines.append('Lost: {}'.format(lost.message))
            lines.append('Lost at: {}'.format(_fmt_vector(lost.position)))
            lines.append('Direction: {}'.format(_fmt_vector(lost.direction)))
            if np.isfinite(lost.energy):
                lines.append('Energy: {:.6g} eV'.format(lost.energy))
            if np.isfinite(lost.time):
                lines.append('Time: {:.6g} s'.format(lost.time))
            if lost.last_position is not None:
                lines.append('Previous position: {}'.format(
                    _fmt_vector(lost.last_position)))
            if lost.last_cells:
                lines.append('Last cells (by level): {}'.format(
                    ', '.join(str(c) for c in lost.last_cells)))
        if self.tracks:
            lines.append('Tracks: {} ({} states)'.format(len(self.tracks),
                                                         self.n_states))
        lines.extend(self.notes)
        return '\n'.join(lines)

    def state_description(self, track_index, state_index):
        """Description of one recorded state of a track"""
        track = self.tracks[track_index]
        lines = ['{} – {} track {}, state {}/{}'.format(
            self.summary(), track.particle, track_index + 1, state_index + 1,
            len(track))]
        lines.append('Position: {}'.format(
            _fmt_vector(track.positions[state_index])))
        if track.directions is not None:
            lines.append('Direction: {}'.format(
                _fmt_vector(track.directions[state_index])))
        if track.energies is not None:
            lines.append('Energy: {:.6g} eV'.format(
                track.energies[state_index]))
        if track.cell_ids is not None:
            lines.append('Cell: {} (instance {})'.format(
                track.cell_ids[state_index],
                track.cell_instances[state_index]
                if track.cell_instances is not None else '?'))
        if track.material_ids is not None:
            lines.append('Material: {}'.format(
                track.material_ids[state_index]))
        return '\n'.join(lines)


def read_track_dataset(dataset):
    """Read the tracks stored in one dataset of a track or restart file

    The dataset is an array of OpenMC's TrackState type with ``offsets`` and
    ``particles`` attributes that split it into tracks.
    """
    states = dataset[()]
    n = len(states)
    offsets = np.asarray(dataset.attrs.get('offsets', [0, n]),
                         dtype=int).reshape(-1)
    particles = np.asarray(dataset.attrs.get('particles', []),
                           dtype=int).reshape(-1)
    if offsets.size == 0 or offsets[-1] != n:
        offsets = np.append(offsets, n)
    tracks = []
    for i in range(offsets.size - 1):
        chunk = states[offsets[i]:offsets[i + 1]]
        if len(chunk) == 0:
            continue
        ptype = particle_name(particles[i]) if i < particles.size \
            else 'unknown'
        tracks.append(Track.from_states(chunk, ptype))
    return tracks


def read_particle_restart(path):
    """Read a particle restart file

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``particle_<batch>_<id>.h5`` file

    Returns
    -------
    ParticleTrackData
    """
    import h5py

    path = Path(path)
    with h5py.File(path, 'r') as f:
        filetype = _decode(f.attrs.get('filetype'))
        if filetype != 'particle restart':
            raise ValueError('{} is not a particle restart file.'.format(path))
        particle_id = int(_scalar(_read(f, 'id'), -1))
        ptype = particle_name(_read(f, 'type', 0))
        data = ParticleTrackData(
            label=path.name, kind=KIND_SOURCE, path=str(path),
            particle_id=particle_id,
            batch=int(_scalar(_read(f, 'current_batch'), 0)),
            generation=int(_scalar(_read(f, 'current_generation'), 1)),
            particle=ptype,
            run_mode=_decode(_read(f, 'run_mode')),
            birth=_vector(_read(f, 'xyz')),
            birth_direction=_vector(_read(f, 'uvw')),
            birth_energy=_scalar(_read(f, 'energy')),
            restart_path=str(path),
        )

        if 'lost' in f:
            group = f['lost']
            cells = _read(group, 'cell_last_id')
            data.lost = LostState(
                message=_decode(_read(group, 'message')),
                position=_vector(_read(group, 'xyz')),
                direction=_vector(_read(group, 'uvw')),
                energy=_scalar(_read(group, 'energy')),
                time=_scalar(_read(group, 'time')),
                weight=_scalar(_read(group, 'weight')),
                last_position=_vector(_read(group, 'xyz_last')),
                last_direction=_vector(_read(group, 'uvw_last')),
                last_cells=[int(c) for c in np.asarray(
                    cells if cells is not None else [], dtype=int).reshape(-1)],
            )
            if 'history' in group:
                data.tracks = read_track_dataset(group['history'])
            if data.tracks:
                data.kind = KIND_HISTORY
            else:
                data.kind = KIND_LOST_STATE
                lost = data.lost
                if lost.position is not None and lost.last_position is not None:
                    data.tracks = [Track.segment(lost.last_position,
                                                 lost.position, ptype)]
        else:
            data.notes.append(
                'The file only contains the source site of the particle. Use '
                '"Replay" to record its track.')
    return data


def read_track_file(path):
    """Read all particles of a track file written with ``openmc -t``

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``tracks.h5`` file

    Returns
    -------
    list of ParticleTrackData
    """
    import h5py

    path = Path(path)
    result = []
    with h5py.File(path, 'r') as f:
        filetype = _decode(f.attrs.get('filetype'))
        if filetype != 'track':
            raise ValueError('{} is not a track file.'.format(path))

        def sort_key(name):
            match = _TRACK_NAME.match(name)
            return tuple(int(x) for x in match.groups()) if match else ()

        names = [name for name in f if _TRACK_NAME.match(name)]
        for name in sorted(names, key=sort_key):
            batch, generation, particle_id = sort_key(name)
            tracks = read_track_dataset(f[name])
            if not tracks:
                continue
            first = tracks[0]
            result.append(ParticleTrackData(
                label='{}:{}'.format(path.name, name), kind=KIND_TRACK_FILE,
                path=str(path), particle_id=particle_id, batch=batch,
                generation=generation, particle=first.particle,
                tracks=tracks, birth=first.positions[0].copy(),
                birth_direction=(first.directions[0].copy()
                                 if first.directions is not None else None),
                birth_energy=(float(first.energies[0])
                              if first.energies is not None else math.nan),
            ))
    return result


def file_type(path):
    """Value of the ``filetype`` attribute of an HDF5 file"""
    import h5py
    with h5py.File(path, 'r') as f:
        return _decode(f.attrs.get('filetype'))


def load_particle_file(path):
    """Read a particle restart or track file

    Returns
    -------
    list of ParticleTrackData
    """
    kind = file_type(path)
    if kind == 'particle restart':
        return [read_particle_restart(path)]
    if kind == 'track':
        return read_track_file(path)
    raise ValueError('{} is neither a particle restart file nor a track file '
                     '(filetype "{}").'.format(path, kind))


def _restart_sort_key(path):
    match = _RESTART_NAME.search(Path(path).name)
    if match:
        return (0, int(match.group(1)), int(match.group(2)), '')
    return (1, 0, 0, Path(path).name)


def scan_directory(directory):
    """Paths of the particle restart files in a directory, in batch order"""
    paths = Path(directory).glob('particle_*.h5')
    return sorted((p for p in paths if p.is_file()), key=_restart_sort_key)


def assign_colors(particles, start=0):
    """Give each particle a color from :data:`TRACK_PALETTE`"""
    for i, particle in enumerate(particles):
        particle.color = TRACK_PALETTE[(start + i) % len(TRACK_PALETTE)]


def focus_width(particle, axes, reference=None):
    """Width of a view that shows the last steps of a particle

    Parameters
    ----------
    particle : ParticleTrackData
        Particle to show. The view is centered on
        :meth:`ParticleTrackData.key_position`.
    axes : sequence of int
        Indices of the coordinates spanned by the view, e.g. ``(0, 1)`` for a
        view of the xy-plane
    reference : float, optional
        Size of the geometry in [cm]. The width is kept between 1% and 200%
        of it.

    Returns
    -------
    float
        Width of the view in [cm]
    """
    try:
        reference = float(reference)
    except (TypeError, ValueError):
        reference = math.nan
    valid_reference = math.isfinite(reference) and reference > 0.0

    extent = 0.0
    center = particle.key_position()
    if center is not None:
        axes = list(axes)
        center = np.asarray(center, dtype=float).reshape(-1)[axes]
        nearby = []
        if particle.tracks and len(particle.tracks[-1]):
            nearby.append(particle.tracks[-1].positions[-10:])
        if particle.lost is not None and \
                particle.lost.last_position is not None:
            nearby.append(np.reshape(particle.lost.last_position, (1, 3)))
        if nearby:
            distance = np.abs(np.vstack(nearby)[:, axes] - center)
            distance = distance[np.isfinite(distance)]
            if distance.size:
                extent = float(distance.max())

    if not (math.isfinite(extent) and extent > 0.0):
        return 0.1 * reference if valid_reference else 10.0
    width = 2.4 * extent
    if valid_reference:
        width = min(max(width, 0.01 * reference), 2.0 * reference)
    return width


def slab_runs(points, axis, lower, upper):
    """Split a polyline into the parts inside and outside of a slab

    The 2D view draws the parts of a track close to the slice plane solid and
    the other parts dashed. Segments that cross the slab are split where they
    enter and leave it.

    Parameters
    ----------
    points : numpy.ndarray
        Vertices of the polyline with shape (N, 3)
    axis : int
        Index of the coordinate normal to the slab
    lower, upper : float
        Bounds of the slab along the axis

    Returns
    -------
    list of (bool, numpy.ndarray)
        For each run of consecutive parts, whether it is inside the slab and
        its vertices with shape (M, 3). Segments with non-finite vertices are
        left out.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) < 2:
        return []
    p0 = points[:-1]
    # Segments with non-finite vertices produce NaNs below and are skipped
    with np.errstate(divide='ignore', invalid='ignore'):
        delta = points[1:] - p0
        s0 = p0[:, axis]
        ds = delta[:, axis]
        ta = (lower - s0) / ds
        tb = (upper - s0) / ds
        # Parameter interval [t0, t1] of each segment inside the slab. It is
        # empty if t1 <= t0.
        t0 = np.clip(np.minimum(ta, tb), 0.0, 1.0)
        t1 = np.clip(np.maximum(ta, tb), 0.0, 1.0)
        flat = ds == 0.0
        inside = (s0 >= lower) & (s0 <= upper)
    t0 = np.where(flat, np.where(inside, 0.0, 1.0), t0)
    t1 = np.where(flat, 1.0, t1)
    finite = np.all(np.isfinite(points), axis=1)
    valid = finite[:-1] & finite[1:] & np.isfinite(t0) & np.isfinite(t1)

    runs = []
    current = None
    for i in range(len(p0)):
        if not valid[i]:
            current = None
            continue
        a, b = float(t0[i]), float(t1[i])
        for is_inside, start, end in ((False, 0.0, a), (True, a, b),
                                      (False, b, 1.0)):
            if end <= start:
                continue
            if current is None or current[0] != is_inside:
                current = (is_inside, [p0[i] + start * delta[i]])
                runs.append(current)
            current[1].append(p0[i] + end * delta[i])
    return [(is_inside, np.array(vertices)) for is_inside, vertices in runs]


def default_openmc_executable():
    return shutil.which('openmc') or ''


def replay_arguments(restart_path, model_path=None, threads=1):
    """Command line arguments to replay a lost particle with tracking"""
    args = ['-r', str(Path(restart_path).resolve()), '-t']
    if threads:
        args += ['-s', str(max(1, int(threads)))]
    if model_path is not None:
        args.append(str(Path(model_path).resolve()))
    return args


def replayed_particle(original, tracks_path):
    """Combine a lost particle with the track recorded by a replay

    Parameters
    ----------
    original : ParticleTrackData
        Particle read from the restart file that was replayed
    tracks_path : str or pathlib.Path
        Track file written by the replay

    Returns
    -------
    ParticleTrackData
    """
    replays = read_track_file(tracks_path)
    if not replays:
        raise ValueError('The replay did not record any track.')
    replay = next((r for r in replays
                   if r.particle_id == original.particle_id), replays[0])

    lost = original.lost
    notes = []
    if lost is None and replay.tracks:
        # Without the loss state of newer files, the end of the replayed
        # track is where the particle was lost
        last = replay.tracks[-1]
        lost = LostState(message='', position=last.positions[-1].copy(),
                         direction=(last.directions[-1].copy()
                                    if last.directions is not None else None))
        notes.append('The particle was lost at the end of the replayed track.')
    elif lost is not None and lost.position is not None:
        ends = np.array([t.positions[-1] for t in replay.tracks if len(t)])
        if ends.size:
            gap = float(np.min(np.linalg.norm(ends - lost.position, axis=1)))
            tolerance = 1e-6 * max(1.0, float(np.max(np.abs(lost.position))))
            if gap > tolerance:
                notes.append('Warning: the replayed track ends {:.4g} cm from '
                             'where the particle was lost. The model may have '
                             'changed since the file was written.'.format(gap))

    return ParticleTrackData(
        label=original.label, kind=KIND_REPLAYED, path=original.path,
        particle_id=original.particle_id, batch=original.batch,
        generation=original.generation, particle=original.particle,
        run_mode=original.run_mode, tracks=replay.tracks,
        birth=original.birth, birth_direction=original.birth_direction,
        birth_energy=original.birth_energy, lost=lost,
        restart_path=original.restart_path, visible=original.visible,
        color=original.color, notes=notes)


try:
    from PySide6.QtCore import QObject, QProcess, Signal
except ImportError:  # pragma: no cover - the plotter always has PySide6
    QObject = object
    QProcess = Signal = None


class ReplayRunner(QObject):
    """Replays a lost particle with ``openmc -r <file> -t``

    The replay runs in a temporary directory. Once it finishes, the recorded
    track is combined with the original particle and emitted with
    :attr:`finished`.
    """

    if Signal is not None:
        finished = Signal(object)
        failed = Signal(str)
        output = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process = None
        self._directory = None
        self._particle = None
        self._log = []
        self._cancelled = False

    @property
    def running(self):
        return (self._process is not None and
                self._process.state() != QProcess.NotRunning)

    @property
    def particle(self):
        return self._particle

    def start(self, particle, model_path, executable=None, threads=1):
        """Start replaying a particle

        Parameters
        ----------
        particle : ParticleTrackData
            Particle read from a restart file
        model_path : str or pathlib.Path
            Model directory or XML file the particle was lost in
        executable : str, optional
            OpenMC executable. Defaults to ``openmc`` on the PATH.
        threads : int
            Number of OpenMP threads
        """
        if self.running:
            raise RuntimeError('A particle is already being replayed.')
        if not particle.can_replay:
            raise ValueError('Only particles read from restart files can be '
                             'replayed.')
        executable = executable or default_openmc_executable()
        if not executable:
            self.failed.emit('The OpenMC executable could not be found. Set '
                             'its path in the Particles panel.')
            return

        self._directory = tempfile.mkdtemp(prefix='openmc_plotter_replay_')
        self._particle = particle
        self._log = []
        self._cancelled = False

        process = QProcess(self)
        process.setWorkingDirectory(self._directory)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)
        self._process = process
        process.start(executable, replay_arguments(
            particle.restart_path, model_path, threads))

    def cancel(self):
        if self.running:
            self._cancelled = True
            self._process.kill()

    def shutdown(self, timeout_ms=2000):
        """Stop a running replay and remove its temporary directory"""
        process = self._process
        if process is not None and process.state() != QProcess.NotRunning:
            self._cancelled = True
            process.kill()
            # Delivers finished, which cleans up, unless it times out
            process.waitForFinished(timeout_ms)
        self._cleanup()

    def _read_output(self):
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardOutput()).decode(
            errors='replace')
        if text:
            self._log.append(text)
            self.output.emit(text)

    def _log_tail(self, n=15):
        lines = ''.join(self._log).splitlines()
        return '\n'.join(lines[-n:])

    def _cleanup(self):
        if self._process is not None:
            self._process.deleteLater()
        self._process = None
        if self._directory is not None:
            shutil.rmtree(self._directory, ignore_errors=True)
        self._directory = None

    def _on_error(self, error):
        # Signals of a process that has already been cleaned up are ignored
        if error != QProcess.FailedToStart or self._process is None:
            return
        message = 'Could not start OpenMC: {}'.format(
            self._process.errorString())
        self._cleanup()
        self.failed.emit(message)

    def _on_finished(self, exit_code, exit_status=None):
        if self._process is None:
            return
        self._read_output()
        if self._cancelled:
            self._cleanup()
            self.failed.emit('The replay was cancelled.')
            return
        tracks_path = Path(self._directory) / 'tracks.h5'
        try:
            if not tracks_path.exists():
                raise RuntimeError('OpenMC did not write a track file.')
            result = replayed_particle(self._particle, tracks_path)
        except Exception as e:
            message = 'Replay failed (exit code {}): {}'.format(exit_code, e)
            tail = self._log_tail()
            if tail:
                message += '\n\nOpenMC output:\n' + tail
            self._cleanup()
            self.failed.emit(message)
            return
        if exit_code != 0:
            result.notes.append('OpenMC exited with code {}.'.format(
                exit_code))
        self._cleanup()
        self.finished.emit(result)
