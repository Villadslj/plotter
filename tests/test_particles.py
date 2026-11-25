"""Tests for reading lost particle files and preparing their tracks"""

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from openmc_plotter import particles as pt

# OpenMC's TrackState HDF5 type
TRACK_STATE = np.dtype([
    ('r', [('x', '<f8'), ('y', '<f8'), ('z', '<f8')]),
    ('u', [('x', '<f8'), ('y', '<f8'), ('z', '<f8')]),
    ('E', '<f8'), ('time', '<f8'), ('wgt', '<f8'),
    ('cell_id', '<i4'), ('cell_instance', '<i4'), ('material_id', '<i4'),
])

NEUTRON = 2112
PHOTON = 22

LOSS_POSITION = (4.0, 1.0, 0.5)
LAST_POSITION = (3.0, 1.0, 0.5)
PRIMARY = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 1.0, 0.5), LAST_POSITION]
SECONDARY = [(1.0, 0.0, 0.0), (1.0, 2.0, 0.0)]


@pytest.fixture
def h5py():
    return pytest.importorskip('h5py')


def track_states(positions):
    positions = np.asarray(positions, dtype=float)
    states = np.zeros(len(positions), dtype=TRACK_STATE)
    for i, name in enumerate('xyz'):
        states['r'][name] = positions[:, i]
    states['u']['x'] = 1.0
    states['E'] = np.linspace(2.0e6, 1.0e6, len(positions))
    states['wgt'] = 1.0
    states['cell_id'] = 1
    states['material_id'] = 2
    return states


def write_tracks(group, name, tracks):
    """Write (PDG number, positions) pairs like OpenMC's track output"""
    states = np.concatenate([track_states(p) for _, p in tracks])
    offsets = np.cumsum([0] + [len(p) for _, p in tracks])
    dset = group.create_dataset(name, data=states)
    dset.attrs['n_particles'] = len(tracks)
    dset.attrs['offsets'] = offsets.astype(np.int32)
    dset.attrs['particles'] = np.array([c for c, _ in tracks], dtype=np.int32)


def write_restart(h5py, path, lost=True, history=None, particle_id=7,
                  batch=3):
    """Write a particle restart file like OpenMC does for a lost particle

    Without ``lost``, the file looks like one written by older versions of
    OpenMC. ``history`` holds the steps recorded with
    ``Settings.lost_particle_history`` as (PDG number, positions) pairs.
    """
    with h5py.File(path, 'w') as f:
        f.attrs['filetype'] = np.bytes_('particle restart')
        f['current_batch'] = batch
        f['generations_per_batch'] = 1
        f['current_generation'] = 1
        f['n_particles'] = 100
        f['run_mode'] = np.bytes_('fixed source')
        f['id'] = particle_id
        f['type'] = NEUTRON
        f['weight'] = 1.0
        f['energy'] = 2.0e6
        f['xyz'] = [0.0, 0.0, 0.0]
        f['uvw'] = [1.0, 0.0, 0.0]
        f['time'] = 0.0
        if not lost:
            return
        group = f.create_group('lost')
        group['message'] = np.bytes_('No cells found.')
        group['xyz'] = LOSS_POSITION
        group['uvw'] = [0.0, 1.0, 0.0]
        group['energy'] = 1.0e6
        group['time'] = 1.0e-8
        group['weight'] = 1.0
        group['xyz_last'] = LAST_POSITION
        group['uvw_last'] = [1.0, 0.0, 0.0]
        group['cell_last_id'] = np.array([10, 20], dtype=np.int32)
        if history:
            write_tracks(group, 'history', history)


def write_track_file(h5py, path, particles):
    """Write a track file like ``openmc -t``

    ``particles`` maps (batch, generation, id) to (PDG number, positions)
    pairs.
    """
    with h5py.File(path, 'w') as f:
        f.attrs['filetype'] = np.bytes_('track')
        for (batch, gen, pid), tracks in particles.items():
            write_tracks(f, 'track_{}_{}_{}'.format(batch, gen, pid), tracks)


def check_runs(runs, expected):
    assert [inside for inside, _ in runs] == [inside for inside, _ in expected]
    for (_, vertices), (_, reference) in zip(runs, expected):
        np.testing.assert_allclose(vertices, reference, atol=1e-12)


# Reading files

def test_restart_file_with_history(h5py, tmp_path):
    path = tmp_path / 'particle_3_7.h5'
    write_restart(h5py, path, history=[(PHOTON, SECONDARY), (NEUTRON, PRIMARY)])

    [p] = pt.load_particle_file(path)
    assert p.kind == pt.KIND_HISTORY
    assert p.label == 'particle_3_7.h5'
    assert (p.particle_id, p.batch, p.particle) == (7, 3, 'neutron')
    assert p.can_replay
    assert [t.particle for t in p.tracks] == ['photon', 'neutron']
    assert [len(t) for t in p.tracks] == [2, 4]
    np.testing.assert_allclose(p.tracks[1].positions, PRIMARY)
    np.testing.assert_allclose(p.tracks[1].energies[[0, -1]], [2.0e6, 1.0e6])
    np.testing.assert_allclose(p.birth, (0.0, 0.0, 0.0))
    np.testing.assert_allclose(p.loss_position, LOSS_POSITION)
    assert p.lost.last_cells == [10, 20]
    # Track states plus the source and loss positions
    assert p.all_positions().shape == (8, 3)

    text = p.describe()
    assert 'Lost: No cells found.' in text
    assert 'Last cells (by level): 10, 20' in text
    assert 'Tracks: 2 (6 states)' in text
    state = p.state_description(1, 3)
    assert 'state 4/4' in state
    assert 'Position: (3, 1, 0.5)' in state
    assert 'Cell: 1 (instance 0)' in state


def test_restart_file_without_history_shows_final_step(h5py, tmp_path):
    path = tmp_path / 'particle_3_7.h5'
    write_restart(h5py, path)

    p = pt.read_particle_restart(path)
    assert p.kind == pt.KIND_LOST_STATE
    [track] = p.tracks
    np.testing.assert_allclose(track.positions, [LAST_POSITION, LOSS_POSITION])
    np.testing.assert_allclose(p.key_position(), LOSS_POSITION)


def test_restart_file_of_older_openmc_only_has_source_site(h5py, tmp_path):
    path = tmp_path / 'particle_3_7.h5'
    write_restart(h5py, path, lost=False)

    p = pt.read_particle_restart(path)
    assert p.kind == pt.KIND_SOURCE
    assert p.lost is None and p.tracks == []
    np.testing.assert_allclose(p.key_position(), (0.0, 0.0, 0.0))
    assert any('Replay' in note for note in p.notes)


def test_track_file_particles_in_numeric_order(h5py, tmp_path):
    path = tmp_path / 'tracks.h5'
    write_track_file(h5py, path, {
        (1, 1, 10): [(NEUTRON, PRIMARY)],
        (1, 1, 2): [(NEUTRON, PRIMARY[:2]), (PHOTON, SECONDARY)],
    })

    particles = pt.load_particle_file(path)
    assert [p.particle_id for p in particles] == [2, 10]
    first = particles[0]
    assert first.kind == pt.KIND_TRACK_FILE
    assert first.label == 'tracks.h5:track_1_1_2'
    assert [t.particle for t in first.tracks] == ['neutron', 'photon']
    np.testing.assert_allclose(first.birth, PRIMARY[0])
    assert first.birth_energy == pytest.approx(2.0e6)
    assert not first.can_replay


def test_load_rejects_other_hdf5_files(h5py, tmp_path):
    path = tmp_path / 'statepoint.10.h5'
    with h5py.File(path, 'w') as f:
        f.attrs['filetype'] = np.bytes_('statepoint')
    with pytest.raises(ValueError, match='neither'):
        pt.load_particle_file(path)


def test_scan_directory_sorts_by_batch_and_id(tmp_path):
    for name in ('particle_10_3.h5', 'particle_2_15.h5', 'particle_2_4.h5',
                 'particle_restart.h5', 'tracks.h5'):
        (tmp_path / name).touch()
    (tmp_path / 'particle_1_1.h5').mkdir()

    names = [p.name for p in pt.scan_directory(tmp_path)]
    assert names == ['particle_2_4.h5', 'particle_2_15.h5',
                     'particle_10_3.h5', 'particle_restart.h5']


# Tracks in the 2D view

def test_slab_runs_split_segments_crossing_the_slab():
    check_runs(pt.slab_runs([(0, 0, -2), (0, 0, 2)], 2, -1.0, 1.0), [
        (False, [(0, 0, -2), (0, 0, -1)]),
        (True, [(0, 0, -1), (0, 0, 1)]),
        (False, [(0, 0, 1), (0, 0, 2)]),
    ])
    check_runs(pt.slab_runs([(0, 0, 2), (0, 0, -2)], 2, -1.0, 1.0), [
        (False, [(0, 0, 2), (0, 0, 1)]),
        (True, [(0, 0, 1), (0, 0, -1)]),
        (False, [(0, 0, -1), (0, 0, -2)]),
    ])


def test_slab_runs_merge_consecutive_parts():
    points = [(0, 0, 3), (1, 0, 3), (1, 0, 0.5), (2, 0, 0.5), (3, 0, 0),
              (4, 0, 5)]
    check_runs(pt.slab_runs(points, 2, -1.0, 1.0), [
        (False, [(0, 0, 3), (1, 0, 3), (1, 0, 1)]),
        (True, [(1, 0, 1), (1, 0, 0.5), (2, 0, 0.5), (3, 0, 0),
                (3.2, 0, 1)]),
        (False, [(3.2, 0, 1), (4, 0, 5)]),
    ])


def test_slab_runs_skip_non_finite_vertices():
    nan, inf = float('nan'), float('inf')
    points = [(0, 0, 0), (1, 0, 0), (nan, nan, nan), (inf, inf, inf),
              (inf, inf, inf), (2, 0, 0), (3, 0, 0)]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        runs = pt.slab_runs(points, 2, -1.0, 1.0)
    check_runs(runs, [
        (True, [(0, 0, 0), (1, 0, 0)]),
        (True, [(2, 0, 0), (3, 0, 0)]),
    ])


def lost_particle(position, last_position=None, track=None):
    p = pt.ParticleTrackData(label='p', kind=pt.KIND_LOST_STATE)
    p.lost = pt.LostState(
        message='', position=np.asarray(position, dtype=float),
        last_position=(None if last_position is None
                       else np.asarray(last_position, dtype=float)))
    if track is not None:
        p.tracks = [pt.Track('neutron', np.asarray(track, dtype=float))]
    return p


def test_focus_width_frames_the_last_steps():
    p = lost_particle(LOSS_POSITION, LAST_POSITION)
    # The previous position is 1 cm from where the particle was lost
    assert pt.focus_width(p, (0, 1)) == pytest.approx(2.4)
    # Between 1% and 200% of the size of the geometry
    assert pt.focus_width(p, (0, 1), reference=1000.0) == pytest.approx(10.0)
    assert pt.focus_width(p, (0, 1), reference=1.0) == pytest.approx(2.0)
    # Both positions coincide in the yz-plane
    assert pt.focus_width(p, (1, 2)) == pytest.approx(10.0)
    assert pt.focus_width(p, (1, 2), reference=50.0) == pytest.approx(5.0)

    # Only the last ten states of a long track are framed
    track = [(float(x), 0.0, 0.0) for x in range(12)]
    p = lost_particle(track[-1], track=track)
    assert pt.focus_width(p, (0, 2)) == pytest.approx(2.4 * 9)


# Replaying particles

def test_replayed_particle_keeps_identity_and_checks_end(h5py, tmp_path):
    restart = tmp_path / 'particle_3_7.h5'
    write_restart(h5py, restart)
    original = pt.read_particle_restart(restart)
    original.color = (1, 2, 3)
    tracks = tmp_path / 'tracks.h5'

    write_track_file(h5py, tracks, {
        (3, 1, 6): [(NEUTRON, [(9.0, 9.0, 9.0), (8.0, 8.0, 8.0)])],
        (3, 1, 7): [(NEUTRON, [(0.0, 0.0, 0.0), LAST_POSITION,
                               LOSS_POSITION])],
    })
    replayed = pt.replayed_particle(original, tracks)
    assert replayed.kind == pt.KIND_REPLAYED
    assert (replayed.label, replayed.color) == ('particle_3_7.h5', (1, 2, 3))
    assert replayed.can_replay
    np.testing.assert_allclose(replayed.tracks[0].positions[-1],
                               LOSS_POSITION)
    assert replayed.notes == []

    # A replay ending elsewhere means that the model has changed
    write_track_file(h5py, tracks, {
        (3, 1, 7): [(NEUTRON, [(0.0, 0.0, 0.0), (3.5, 1.0, 0.5)])],
    })
    [note] = pt.replayed_particle(original, tracks).notes
    assert note.startswith('Warning') and '0.5 cm' in note

    write_track_file(h5py, tracks, {})
    with pytest.raises(ValueError, match='did not record'):
        pt.replayed_particle(original, tracks)


def test_replay_of_source_only_particle_ends_at_loss(h5py, tmp_path):
    restart = tmp_path / 'particle_1_2.h5'
    write_restart(h5py, restart, lost=False, particle_id=2, batch=1)
    tracks = tmp_path / 'tracks.h5'
    write_track_file(h5py, tracks, {(1, 1, 2): [(NEUTRON, PRIMARY)]})

    replayed = pt.replayed_particle(pt.read_particle_restart(restart), tracks)
    np.testing.assert_allclose(replayed.loss_position, PRIMARY[-1])
    np.testing.assert_allclose(replayed.loss_direction, (1.0, 0.0, 0.0))
    assert replayed.notes


def fake_openmc(tmp_path, body):
    """Executable standing in for OpenMC that runs some Python code"""
    script = tmp_path / 'openmc'
    script.write_text(
        '#!{}\nimport os, shutil, sys\nprint("cwd=" + os.getcwd())\n'
        'print("args=" + " ".join(sys.argv[1:]))\n{}\n'.format(
            sys.executable, body))
    script.chmod(0o755)
    return str(script)


@pytest.mark.skipif(sys.platform == 'win32',
                    reason='The fake OpenMC executable is a script')
def test_replay_runner_combines_recorded_track(h5py, qtbot, tmp_path):
    restart = tmp_path / 'particle_3_7.h5'
    write_restart(h5py, restart)
    recorded = tmp_path / 'recorded.h5'
    write_track_file(h5py, recorded, {
        (3, 1, 7): [(NEUTRON, [(0.0, 0.0, 0.0), LOSS_POSITION])]})
    executable = fake_openmc(
        tmp_path, 'shutil.copy({!r}, "tracks.h5")'.format(str(recorded)))

    runner = pt.ReplayRunner()
    output = []
    runner.output.connect(output.append)
    with qtbot.waitSignal(runner.finished, timeout=30000) as blocker:
        runner.start(pt.read_particle_restart(restart), tmp_path, executable)

    replayed = blocker.args[0]
    assert replayed.kind == pt.KIND_REPLAYED
    np.testing.assert_allclose(replayed.tracks[0].positions[-1],
                               LOSS_POSITION)
    assert not runner.running

    log = ''.join(output)
    assert 'args=-r {} -t -s 1 {}'.format(restart.resolve(),
                                          tmp_path.resolve()) in log
    # The replay ran in a temporary directory that has been removed
    directory = Path(re.search(r'cwd=(.*)', log).group(1).strip())
    assert directory != tmp_path and not directory.exists()


@pytest.mark.skipif(sys.platform == 'win32',
                    reason='The fake OpenMC executable is a script')
@pytest.mark.parametrize('body, expected', [
    ('print("ERROR: No geometry file found."); sys.exit(3)',
     ['exit code 3', 'did not write a track file',
      'ERROR: No geometry file found.']),
    (None, ['Could not start OpenMC']),
])
def test_replay_runner_reports_failures(h5py, qtbot, tmp_path, body,
                                        expected):
    restart = tmp_path / 'particle_3_7.h5'
    write_restart(h5py, restart)
    if body is None:
        executable = str(tmp_path / 'missing' / 'openmc')
    else:
        executable = fake_openmc(tmp_path, body)

    runner = pt.ReplayRunner()
    with qtbot.waitSignal(runner.failed, timeout=30000) as blocker:
        runner.start(pt.read_particle_restart(restart), tmp_path, executable)

    message = blocker.args[0]
    for text in expected:
        assert text in message
    assert not runner.running
