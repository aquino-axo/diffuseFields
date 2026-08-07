"""
Unit tests for run_plot_cpsd_diagonal (validation-comparison plotting).

Implements only the correctness checks approved during planning:
1. Validation diagonal extraction + frequency slicing.
2. Relative-L2 per-location error metric (value + ranking).
3. Coordinate alignment integrity (order preserved, no dedup; duplicate
   faces and out-of-tolerance matches raise).
4. Box-vs-band switchover at BAND_FREQ_THRESHOLD.
5. Config validation (box/error require validation; validation requires
   coordinates; .mat validation requires a variable name).
6. Independent validation frequency grid: the diagonal loads unsliced and
   its length is checked against the supplied grid; the point-by-point
   comparison kinds and an index-based solution axis are both rejected;
   the `lines` CSV is long format and carries each series' own grid.

Plot rendering itself is smoke-tested (saves without error) rather than
pixel-asserted.
"""

import csv
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import run_plot_cpsd_diagonal as rp
from run_plot_cpsd_diagonal import (
    BAND_FREQ_THRESHOLD,
    box_render_mode,
    db_error,
    db_error_stats,
    load_validation_diagonal,
    plot_validation_db,
    relative_l2_error,
    resolve_selection,
    select_per_sensor_order,
    to_db,
    validate_config,
)


def _out_dir(tmp_path):
    """
    Resolve the directory a test may write to.

    pytest ignores parameters that carry a default value, so `tmp_path=None`
    is what these tests receive under pytest; falling back to '.' would litter
    the repository root with fixtures. Allocate a real temporary directory
    instead.
    """
    if tmp_path is not None:
        return Path(tmp_path)
    return Path(tempfile.mkdtemp(prefix='test_plot_cpsd_diagonal_'))


def _hermitian_stack(n_loc, n_freq, rng):
    """Random (n_loc, n_loc, n_freq) Hermitian-per-frequency complex array."""
    A = (rng.standard_normal((n_loc, n_loc, n_freq))
         + 1j * rng.standard_normal((n_loc, n_loc, n_freq)))
    return A + np.conj(np.transpose(A, (1, 0, 2)))


def test_validation_extraction_and_slicing(tmp_path=None):
    """diag(.).real extracted correctly and sliced to freq_indices."""
    print("Test 1: validation diagonal extraction + freq slicing...")
    rng = np.random.default_rng(0)
    n_loc, n_freq_full = 4, 7
    arr = _hermitian_stack(n_loc, n_freq_full, rng)

    out = _out_dir(tmp_path)
    npy = out / 'val.npy'
    np.save(npy, arr)

    freq_indices = [0, 2, 5]
    val_diag = load_validation_diagonal(str(npy), None, freq_indices)

    # Diagonal of a Hermitian matrix is real; compare to a direct computation.
    expected_full = np.real(
        np.stack([np.diag(arr[:, :, f]) for f in range(n_freq_full)], axis=1)
    )
    expected = expected_full[:, freq_indices]
    assert val_diag.shape == (n_loc, len(freq_indices))
    assert np.allclose(val_diag, expected)
    assert np.isrealobj(val_diag)

    # Out-of-range frequency index must raise.
    try:
        load_validation_diagonal(str(npy), None, [0, n_freq_full])
        raise AssertionError("expected ValueError for out-of-range freq index")
    except ValueError:
        pass

    # Non-square / non-3D input must raise.
    np.save(out / 'bad.npy', arr[:, :2, :])
    try:
        load_validation_diagonal(str(out / 'bad.npy'), None, [0])
        raise AssertionError("expected ValueError for non-square validation")
    except ValueError:
        pass
    print("  ok")


def test_relative_l2_error():
    """Per-location relative-L2 matches analytic value; ranking is correct."""
    print("Test 2: relative-L2 error metric...")
    # Location 0: solution == validation -> error 0.
    # Location 1: solution = 1.1 * validation -> error 0.1 exactly.
    # Location 2: solution = 1.5 * validation -> error 0.5 exactly.
    val = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ])
    sol = np.array([
        [1.0, 2.0, 3.0],
        [1.1, 2.2, 3.3],
        [1.5, 3.0, 4.5],
    ])
    err = relative_l2_error(sol, val)
    assert np.allclose(err, [0.0, 0.1, 0.5]), err

    order = np.argsort(err)[::-1]  # worst first
    assert list(order) == [2, 1, 0]

    # Zero-validation row yields inf (guarded division), not a crash.
    err2 = relative_l2_error(np.array([[1.0, 1.0]]), np.array([[0.0, 0.0]]))
    assert np.isinf(err2[0])
    print("  ok")


@contextmanager
def _fake_interpolator(centroids):
    """Patch ExodusSideInterpolator to return preset centroids."""
    class _Fake:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_sideset_face_centroids(self, sideset_id):
            return np.asarray(centroids, dtype=float)

    original = rp.ExodusSideInterpolator
    rp.ExodusSideInterpolator = _Fake
    try:
        yield
    finally:
        rp.ExodusSideInterpolator = original


def test_coordinate_alignment_integrity():
    """Validation mode preserves order, forbids dedup, enforces tolerance."""
    print("Test 3: coordinate alignment integrity...")
    centroids = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ]

    # Distinct coordinates: order preserved, faces in coordinate order.
    cfg = {
        'input': {'exodus_file': 'x', 'sideset_id': 1},
        'selection': {
            'indices': None,
            'coordinates': [[2.0, 0, 0], [0.0, 0, 0]],
            'match_tolerance': None,
        },
    }
    with _fake_interpolator(centroids):
        chosen = resolve_selection(cfg, n_entries=3, validation_mode=True)
    assert [idx for idx, _ in chosen] == [2, 0], chosen

    # Two coordinates resolving to the same face must raise in validation mode.
    cfg_dup = {
        'input': {'exodus_file': 'x', 'sideset_id': 1},
        'selection': {
            'indices': None,
            'coordinates': [[0.1, 0, 0], [-0.1, 0, 0]],  # both -> face 0
            'match_tolerance': None,
        },
    }
    with _fake_interpolator(centroids):
        try:
            resolve_selection(cfg_dup, n_entries=3, validation_mode=True)
            raise AssertionError("expected ValueError for duplicate face")
        except ValueError:
            pass

    # Out-of-tolerance match must raise.
    cfg_tol = {
        'input': {'exodus_file': 'x', 'sideset_id': 1},
        'selection': {
            'indices': None,
            'coordinates': [[10.0, 0, 0]],  # 8 units from nearest face
            'match_tolerance': 0.5,
        },
    }
    with _fake_interpolator(centroids):
        try:
            resolve_selection(cfg_tol, n_entries=3, validation_mode=True)
            raise AssertionError("expected ValueError for tolerance breach")
        except ValueError:
            pass
    print("  ok")


def test_box_band_switchover():
    """box_render_mode flips at BAND_FREQ_THRESHOLD."""
    print("Test 4: box vs band switchover...")
    assert box_render_mode(1) == 'boxes'
    assert box_render_mode(BAND_FREQ_THRESHOLD) == 'boxes'
    assert box_render_mode(BAND_FREQ_THRESHOLD + 1) == 'bands'
    print("  ok")


def _base_config(**overrides):
    cfg = {
        'input': {'diagonal_npy_path': 'd.npy', 'sidecar_json_path': 's.json'},
        'selection': {'indices': [0]},
        'plot': {},
        'output': {},
    }
    for section, vals in overrides.items():
        cfg.setdefault(section, {}).update(vals)
    return cfg


def test_config_validation(tmp_path=None):
    """Required-input rules for validation comparison."""
    print("Test 5: config validation...")
    out = _out_dir(tmp_path)
    diag = out / 'd.npy'
    side = out / 's.json'
    np.save(diag, np.ones((3, 2)))
    side.write_text('{"mode": "diagonal", "freq_indices": [0, 1]}')

    # box without validation -> error.
    cfg = _base_config(
        input={'diagonal_npy_path': str(diag), 'sidecar_json_path': str(side)},
        plot={'kind': 'box'},
        selection={'indices': [0], 'coordinates': None},
    )
    try:
        validate_config(cfg)
        raise AssertionError("expected ValueError: box requires validation")
    except ValueError:
        pass

    # validation set but no coordinates -> error.
    val = out / 'v.npy'
    np.save(val, np.ones((2, 2, 2)))
    cfg = _base_config(
        input={
            'diagonal_npy_path': str(diag),
            'sidecar_json_path': str(side),
            'validation_path': str(val),
        },
        plot={'kind': 'lines'},
        selection={'indices': [0], 'coordinates': None},
    )
    try:
        validate_config(cfg)
        raise AssertionError("expected ValueError: validation needs coords")
    except ValueError:
        pass

    # .mat validation without a variable name -> error.
    valmat = out / 'v.mat'
    valmat.write_bytes(b'')  # existence is enough; var check precedes load
    cfg = _base_config(
        input={
            'diagonal_npy_path': str(diag),
            'sidecar_json_path': str(side),
            'validation_path': str(valmat),
            'validation_var': None,
        },
        plot={'kind': 'lines'},
        selection={
            'indices': None,
            'coordinates': [[0, 0, 0]],
        },
        # exodus_file/sideset_id absent -> would also fail, but validation_var
        # check fires first.
    )
    try:
        validate_config(cfg)
        raise AssertionError("expected ValueError: .mat needs validation_var")
    except ValueError:
        pass
    print("  ok")


def test_db_conversion_and_floor():
    """to_db matches 10log10(S/ref); non-positive samples floored, counted."""
    print("Test 6: dB conversion + relative floor...")
    S = np.array([[100.0, 10.0, 1.0, 0.0, -5.0]])  # row peak = 100
    floor_rel = 1e-3  # floor = 0.1 -> 10log10(0.1) = -10 dB
    L, n_clamped = to_db(S, ref=1.0, floor_rel=floor_rel)
    assert np.allclose(L[0, :3], [20.0, 10.0, 0.0]), L
    assert np.allclose(L[0, 3:], [-10.0, -10.0]), L  # 0.0 and -5.0 -> floor
    assert n_clamped == 2
    assert np.all(np.isfinite(L))  # never -inf

    # Reference scaling shifts every level by -10log10(ref).
    L10, _ = to_db(S, ref=10.0, floor_rel=floor_rel)
    assert np.allclose(L10, L - 10.0)
    print("  ok")


def test_signed_db_error():
    """dL = 10log10(S_meas/S_comp) = level(meas) - level(comp), signed."""
    print("Test 7: signed dB error...")
    comp = np.array([[1.0, 2.0, 4.0]])
    meas = np.array([[2.0, 2.0, 2.0]])  # meas/comp = 2, 1, 0.5
    dL, _ = db_error(meas, comp, floor_rel=1e-12)
    assert np.allclose(dL, 10.0 * np.log10([2.0, 1.0, 0.5])), dL

    # Identical to the difference of the two dB levels.
    l_meas, _ = to_db(meas, 1.0, 1e-12)
    l_comp, _ = to_db(comp, 1.0, 1e-12)
    assert np.allclose(dL, l_meas - l_comp)

    # Sign: measured > computed -> positive; measured < computed -> negative.
    assert dL[0, 0] > 0
    assert dL[0, 2] < 0
    print("  ok")


def test_pooled_and_per_sensor_stats():
    """db_error_stats pools |dL|; per-sensor reduces over frequency only."""
    print("Test 8: pooled + per-sensor error statistics...")
    dL = np.array([
        [1.0, -3.0, 2.0],   # |.| = 1, 3, 2
        [-0.5, 0.5, 4.0],   # |.| = .5, .5, 4
    ])
    max_abs, med_abs = db_error_stats(dL)
    # pooled |.| sorted = [.5, .5, 1, 2, 3, 4] -> max 4, median 1.5
    assert np.isclose(max_abs, 4.0)
    assert np.isclose(med_abs, 1.5)

    per_sensor_max = np.max(np.abs(dL), axis=1)
    assert np.allclose(per_sensor_max, [3.0, 4.0])
    print("  ok")


def _db_config(fig_path, top_n=None, per_sensor=True, save_csv=False):
    return {
        'plot': {
            'kind': ['validation_db'], 'log_scale': True,
            'title': 'test', 'ylabel': r'$S_{ii}$', 'xlabel': None,
            'figsize': [6, 5], 'ylim': None, 'xlim': None,
            'db_ref': 1.0, 'db_floor': 1e-12,
        },
        'output': {
            'figure_path': str(fig_path), 'figure_format': 'png',
            'dpi': 60, 'save_selection_csv': save_csv, 'top_n': top_n,
            'per_sensor': per_sensor,
        },
    }


def test_per_sensor_cap_and_naming(tmp_path=None):
    """top_n caps per-sensor figures worst-first; files named sensor_<idx>."""
    print("Test 9: per-sensor cap + naming...")
    out = _out_dir(tmp_path)
    freq = np.array([1.0, 2.0, 3.0])
    comp = np.ones((3, 3))
    meas = np.array([
        [1.05, 1.05, 1.05],  # smallest error
        [1.3, 1.3, 1.3],     # medium
        [3.0, 3.0, 3.0],     # largest
    ])

    # Ordering + cap logic (worst-first, capped, skipped count).
    dL_all, _ = db_error(meas, comp, 1e-12)
    order, n_skipped = select_per_sensor_order(dL_all, top_n=2)
    assert list(order) == [2, 1], order
    assert n_skipped == 1

    # End-to-end file emission with face indices 10/11/12.
    selection = [(10, 'node 10'), (11, 'node 11'), (12, 'node 12')]
    fig_path = out / 'all.png'
    config = _db_config(fig_path, top_n=2, per_sensor=True)
    plot_validation_db(comp, meas, freq, selection, 'Frequency [Hz]',
                       config, fig_path)

    assert fig_path.exists()  # combined figure
    subdir = fig_path.parent / 'per_sensor'
    written = sorted(p.name for p in subdir.glob('sensor_*.png'))
    # worst two: selection[2]=idx12, selection[1]=idx11; idx10 skipped.
    assert written == ['sensor_11.png', 'sensor_12.png'], written
    print("  ok")


def test_config_validation_db(tmp_path=None):
    """validation_db requires validation; db_ref/db_floor must be positive."""
    print("Test 10: validation_db config validation...")
    out = _out_dir(tmp_path)
    diag = out / 'd.npy'
    side = out / 's.json'
    np.save(diag, np.ones((3, 2)))
    side.write_text('{"mode": "diagonal", "freq_indices": [0, 1]}')

    base_input = {'diagonal_npy_path': str(diag), 'sidecar_json_path': str(side)}

    # validation_db without a validation file -> error.
    cfg = _base_config(
        input=dict(base_input),
        plot={'kind': 'validation_db'},
        selection={'indices': [0], 'coordinates': None},
    )
    try:
        validate_config(cfg)
        raise AssertionError("expected ValueError: validation_db needs val")
    except ValueError:
        pass

    # Non-positive db_ref / db_floor -> error (isolated via 'lines' kind).
    for bad in ({'db_ref': 0.0}, {'db_floor': -1e-9}):
        cfg = _base_config(
            input=dict(base_input),
            plot={'kind': 'lines', **bad},
            selection={'indices': [0]},
        )
        try:
            validate_config(cfg)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass
    print("  ok")


# ---------------------------------------------------------------------------
# Independent validation frequency grid
# ---------------------------------------------------------------------------

# Sideset face centroids used by the independent-grid end-to-end tests; the
# first two are selected by coordinate, in that order.
_CENTROIDS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]


def _val_with_diagonal(diag: np.ndarray) -> np.ndarray:
    """(n_loc, n_loc, n_freq) Hermitian array with the given real diagonal."""
    n_loc, n_freq = diag.shape
    arr = np.zeros((n_loc, n_loc, n_freq), dtype=complex)
    for f in range(n_freq):
        arr[:, :, f] = np.diag(diag[:, f].astype(float))
    return arr


def _indep_grid_config(out, n_freq_sol, n_freq_val, freq_spec,
                       kind='lines', save_csv=True, sol_frequencies=True):
    """Write solution/sidecar/validation files and build a matching config."""
    import json

    n_faces, n_loc = len(_CENTROIDS), 2
    rng = np.random.default_rng(7)

    diag = np.abs(rng.standard_normal((n_faces, n_freq_sol))) + 0.5
    np.save(out / 'd_indep.npy', diag)

    sidecar = {'mode': 'diagonal', 'freq_indices': list(range(n_freq_sol))}
    if sol_frequencies:
        sidecar['frequencies'] = [100.0 * (i + 1) for i in range(n_freq_sol)]
    (out / 's_indep.json').write_text(json.dumps(sidecar))

    val_diag = np.abs(rng.standard_normal((n_loc, n_freq_val))) + 0.5
    np.save(out / 'v_indep.npy', _val_with_diagonal(val_diag))

    exo = out / 'mesh.e'
    exo.write_bytes(b'')  # existence only; the interpolator is faked

    return {
        'input': {
            'diagonal_npy_path': str(out / 'd_indep.npy'),
            'sidecar_json_path': str(out / 's_indep.json'),
            'exodus_file': str(exo),
            'sideset_id': 1,
            'validation_path': str(out / 'v_indep.npy'),
            'validation_var': None,
            'validation_frequencies': freq_spec,
        },
        'selection': {
            'indices': None,
            'coordinates': [_CENTROIDS[0], _CENTROIDS[1]],
            'match_tolerance': None,
        },
        'plot': {'kind': kind, 'log_scale': True, 'figsize': [6, 4]},
        'output': {
            'figure_path': str(out / 'indep.png'),
            'dpi': 60,
            'save_selection_csv': save_csv,
        },
    }


def test_independent_grid_loads_unsliced(tmp_path=None):
    """freq_indices=None keeps every validation frequency; length is checked."""
    print("Test 11: independent grid loads unsliced...")
    out = _out_dir(tmp_path)
    rng = np.random.default_rng(11)
    n_loc, n_freq_val = 3, 9
    arr = _hermitian_stack(n_loc, n_freq_val, rng)
    npy = out / 'val_indep.npy'
    np.save(npy, arr)

    val_diag = load_validation_diagonal(str(npy), None, None)

    expected = np.real(
        np.stack([np.diag(arr[:, :, f]) for f in range(n_freq_val)], axis=1)
    )
    assert val_diag.shape == (n_loc, n_freq_val), val_diag.shape
    assert np.allclose(val_diag, expected)

    # A frequency vector whose length disagrees with the validation array is
    # rejected: 5 frequencies against 9 stored.
    cfg = _indep_grid_config(out, n_freq_sol=4, n_freq_val=9,
                             freq_spec=[10.0, 20.0, 30.0, 40.0, 50.0])
    with _fake_interpolator(_CENTROIDS):
        try:
            rp.run(validate_config(cfg))
            raise AssertionError("expected ValueError for length mismatch")
        except ValueError as e:
            assert '9' in str(e) and '5' in str(e), str(e)
    print("  ok")


def test_config_rejects_comparison_kinds_on_independent_grid(tmp_path=None):
    """box/error difference the two spectra and need a shared grid.

    validation_db does not: it degrades to the dB overlay panel alone.
    """
    print("Test 12: difference kinds rejected on independent grid...")
    out = _out_dir(tmp_path)
    f_val = [150.0, 250.0, 350.0]

    for kind in ('box', 'error'):
        cfg = _indep_grid_config(out, 4, 3, f_val, kind=kind)
        try:
            validate_config(cfg)
            raise AssertionError(f"expected ValueError for kind '{kind}'")
        except ValueError as e:
            msg = str(e)
            assert 'validation_frequencies' in msg, msg
            # The offending-kinds list must name exactly this kind, so
            # validation_db is never branded an offender.
            assert f"plot kinds ['{kind}']" in msg, msg

    # validation_db is accepted on an independent grid.
    cfg = _indep_grid_config(out, 4, 3, f_val, kind='validation_db')
    assert validate_config(cfg)['plot']['kind'] == ['validation_db']

    # ... including alongside lines.
    cfg = _indep_grid_config(out, 4, 3, f_val, kind=['lines', 'validation_db'])
    assert validate_config(cfg)['plot']['kind'] == ['lines', 'validation_db']

    # A mixed list is rejected too, naming only the offending kinds.
    cfg = _indep_grid_config(out, 4, 3, f_val, kind=['lines', 'error'])
    try:
        validate_config(cfg)
        raise AssertionError("expected ValueError for ['lines', 'error']")
    except ValueError as e:
        assert 'error' in str(e), str(e)

    # 'lines' alone is accepted.
    cfg = _indep_grid_config(out, 4, 3, f_val, kind='lines')
    assert validate_config(cfg)['plot']['kind'] == ['lines']

    # validation_frequencies without a validation file is a configuration
    # error, not a silently ignored field.
    cfg = _indep_grid_config(out, 4, 3, f_val, kind='lines')
    cfg['input']['validation_path'] = None
    try:
        validate_config(cfg)
        raise AssertionError("expected ValueError: freqs without validation")
    except ValueError as e:
        assert 'validation_path' in str(e), str(e)
    print("  ok")


def test_config_rejects_index_axis_with_independent_grid(tmp_path=None):
    """An index-based solution axis cannot be overlaid with Hz."""
    print("Test 13: index axis rejected on independent grid...")
    out = _out_dir(tmp_path)
    # n_freq_val > n_freq_sol, so the shared-grid slicing path would succeed
    # here; only the index-axis guard can make this raise.
    f_val = [150.0, 250.0, 350.0, 450.0, 550.0, 650.0]
    cfg = _indep_grid_config(out, 4, 6, f_val, sol_frequencies=False)
    with _fake_interpolator(_CENTROIDS):
        try:
            rp.run(validate_config(cfg))
            raise AssertionError("expected ValueError for index frequency axis")
        except ValueError as e:
            msg = str(e)
            assert 'validation_frequencies' in msg, msg
            assert 'index' in msg, msg
    print("  ok")


def test_lines_long_format_csv(tmp_path=None):
    """The lines CSV is long format and carries each series' own grid."""
    print("Test 14: lines long-format CSV...")
    out = _out_dir(tmp_path)
    n_freq_sol, n_freq_val, n_loc = 4, 6, 2
    f_sol = [100.0, 200.0, 300.0, 400.0]
    f_val = [150.0, 250.0, 350.0, 450.0, 550.0, 650.0]

    cfg = _indep_grid_config(out, n_freq_sol, n_freq_val, f_val)
    with _fake_interpolator(_CENTROIDS):
        rp.run(validate_config(cfg))

    fig_path = Path(cfg['output']['figure_path'])
    assert fig_path.exists()

    with open(fig_path.with_suffix('.csv'), newline='') as f:
        rows = list(csv.reader(f))

    assert rows[0] == ['series', 'frequency', 'index', 'label', 'value'], rows[0]
    body = rows[1:]
    assert len(body) == n_loc * (n_freq_sol + n_freq_val), len(body)

    sol = [r for r in body if r[0] == 'solution']
    val = [r for r in body if r[0] == 'validation']
    assert len(sol) == n_loc * n_freq_sol
    assert len(val) == n_loc * n_freq_val

    # Each series reports its own frequency grid, not the other's.
    assert sorted({float(r[1]) for r in sol}) == f_sol
    assert sorted({float(r[1]) for r in val}) == f_val

    # Both selected faces appear in each series.
    assert sorted({int(r[2]) for r in sol}) == [0, 1]
    assert sorted({int(r[2]) for r in val}) == [0, 1]
    assert all(float(r[4]) > 0 for r in body)

    # Without a validation file the same long layout holds, solution only.
    cfg2 = _indep_grid_config(out, n_freq_sol, n_freq_val, None)
    cfg2['input']['validation_path'] = None
    cfg2['input']['validation_frequencies'] = None
    cfg2['selection'] = {'indices': [0, 1], 'coordinates': None,
                         'match_tolerance': None}
    cfg2['output']['figure_path'] = str(out / 'sol_only.png')
    rp.run(validate_config(cfg2))
    with open((out / 'sol_only.png').with_suffix('.csv'), newline='') as f:
        rows2 = list(csv.reader(f))
    assert rows2[0] == ['series', 'frequency', 'index', 'label', 'value']
    assert len(rows2) - 1 == n_loc * n_freq_sol
    assert {r[0] for r in rows2[1:]} == {'solution'}
    assert sorted({float(r[1]) for r in rows2[1:]}) == f_sol
    print("  ok")


# ---------------------------------------------------------------------------
# validation_db on an independent grid: overlay panel only
# ---------------------------------------------------------------------------


def test_validation_db_independent_grid_single_panel():
    """Independent grid -> one axes and no dL stats; shared grid -> two."""
    print("Test 15: validation_db single-panel on independent grid...")
    f_sol = np.array([100.0, 200.0, 300.0, 400.0])
    f_val = np.array([150.0, 250.0, 350.0])
    sol = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])
    val = np.array([[1.5, 2.5, 3.5], [2.5, 3.5, 4.5]])
    selection = [(0, 'node 0'), (1, 'node 1')]
    plot_cfg = _db_config(Path('unused.png'))['plot']

    # Independent grid: single panel, no stats, each series on its own x.
    fig, stats, _nc = rp._render_db_figure(
        sol, val, f_sol, selection, 'Frequency [Hz]', plot_cfg,
        title='t', show_sensor_legend=True, val_freq_axis=f_val,
    )
    assert len(fig.axes) == 1, [ax.get_ylabel() for ax in fig.axes]
    assert stats is None, stats
    xs = [ln.get_xdata() for ln in fig.axes[0].lines if len(ln.get_xdata())]

    def _drawn(target):
        return any(len(x) == len(target) and np.allclose(x, target)
                   for x in xs)

    assert _drawn(f_sol), 'solution drawn on its own grid'
    assert _drawn(f_val), 'validation drawn on its own grid'
    plt.close(fig)

    # Shared grid: unchanged two-panel figure with real stats.
    fig2, stats2, _nc2 = rp._render_db_figure(
        sol, sol * 1.5, f_sol, selection, 'Frequency [Hz]', plot_cfg,
        title='t', show_sensor_legend=True,
    )
    assert len(fig2.axes) == 2, len(fig2.axes)
    assert stats2 is not None and np.isclose(stats2[0], 10 * np.log10(1.5))
    plt.close(fig2)
    print("  ok")


def _indep_db_run(out, top_n=None, per_sensor=True, save_csv=True):
    """Run validation_db end-to-end on an independent grid."""
    f_val = [150.0, 250.0, 350.0, 450.0, 550.0, 650.0]
    cfg = _indep_grid_config(out, 4, 6, f_val, kind='validation_db',
                             save_csv=save_csv)
    cfg['output']['figure_path'] = str(out / 'db.png')
    cfg['output']['top_n'] = top_n
    cfg['output']['per_sensor'] = per_sensor
    cfg['plot']['db_ref'] = 1.0
    cfg['plot']['db_floor'] = 1e-12
    with _fake_interpolator(_CENTROIDS):
        rp.run(validate_config(cfg))
    return cfg, len(f_val)


def test_validation_db_independent_grid_csv(tmp_path=None):
    """Long-format dB CSV; no error-stats file when there is no dL."""
    print("Test 16: validation_db independent-grid CSV...")
    out = _out_dir(tmp_path)
    cfg, n_freq_val = _indep_db_run(out, per_sensor=False)
    n_freq_sol, n_loc = 4, 2

    fig_path = Path(cfg['output']['figure_path'])
    assert fig_path.exists()

    with open(fig_path.with_suffix('.csv'), newline='') as f:
        rows = list(csv.reader(f))
    assert rows[0] == ['series', 'frequency', 'index', 'label', 'level_db'], \
        rows[0]
    body = rows[1:]
    assert len(body) == n_loc * (n_freq_sol + n_freq_val), len(body)
    sol_f = sorted({float(r[1]) for r in body if r[0] == 'computed'})
    val_f = sorted({float(r[1]) for r in body if r[0] == 'measured'})
    assert sol_f == [100.0, 200.0, 300.0, 400.0], sol_f
    assert val_f == [150.0, 250.0, 350.0, 450.0, 550.0, 650.0], val_f

    # No dL exists, so no error summary may be written.
    stats_path = fig_path.with_name(f'{fig_path.stem}_error_stats.csv')
    assert not stats_path.exists(), 'error stats must not be written'
    print("  ok")


def test_per_sensor_order_falls_back_to_selection_order(tmp_path=None):
    """Without dL there is nothing to rank by: selection order, capped."""
    print("Test 17: per-sensor order falls back to selection order...")
    out = _out_dir(tmp_path)
    _indep_db_run(out, top_n=1, per_sensor=True, save_csv=False)

    subdir = out / 'per_sensor'
    written = sorted(p.name for p in subdir.glob('sensor_*.png'))
    # Selection is coordinates 0 then 1 -> faces 0, 1. Capped at 1 => face 0,
    # which is the FIRST selected, not the worst (no error metric exists).
    assert written == ['sensor_0.png'], written
    print("  ok")


def test_legend_sits_outside_axes():
    """Legends are anchored right of the axes so they stop crowding curves."""
    print("Test 18: legend outside the axes...")
    freq = np.array([100.0, 200.0, 300.0])
    sol = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    selection = [(0, 'node 0'), (1, 'node 1')]

    cfg = {
        'plot': {'kind': ['lines'], 'log_scale': True, 'title': 't',
                 'ylabel': 'S', 'xlabel': None, 'figsize': [6, 4],
                 'ylim': None, 'xlim': None, 'db_ref': 1.0,
                 'db_floor': 1e-12},
        'output': {'figure_path': 'x.png', 'figure_format': 'png', 'dpi': 60,
                   'save_selection_csv': False, 'top_n': None,
                   'per_sensor': False},
    }

    def _anchor_x(ax):
        leg = ax.get_legend()
        assert leg is not None, 'no legend drawn'
        return leg.get_bbox_to_anchor().transformed(ax.transAxes.inverted()).x0

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label='node 0')
    rp._legend_outside(ax)
    assert _anchor_x(ax) > 1.0, _anchor_x(ax)
    plt.close(fig)

    # The dB overlay legend is anchored outside too.
    fig2, _stats, _nc = rp._render_db_figure(
        sol, sol * 1.2, freq, selection, 'Frequency [Hz]', cfg['plot'],
        title='t', show_sensor_legend=True,
    )
    assert _anchor_x(fig2.axes[0]) > 1.0
    plt.close(fig2)
    print("  ok")


# ---------------------------------------------------------------------------
# envelope: min / energetic-mean / max across sensors, in dB
# ---------------------------------------------------------------------------


def test_envelope_statistics():
    """Band is the true min/max; the centre line is the ENERGETIC mean."""
    print("Test 19: envelope statistics...")
    # Two sensors, two frequencies. Column 0 spans 1 -> 100 (0 -> 20 dB);
    # column 1 is uniform at 4 (both sensors 6.0206 dB).
    S = np.array([[1.0, 4.0],
                  [100.0, 4.0]])
    stats = rp.envelope_stats(S, db_ref=1.0, db_floor=1e-12)

    assert np.allclose(stats.lo, [0.0, 10 * np.log10(4)]), stats.lo
    assert np.allclose(stats.hi, [20.0, 10 * np.log10(4)]), stats.hi

    # Energetic mean: average the POWERS, then convert.
    assert np.allclose(stats.mean,
                       10 * np.log10([50.5, 4.0])), stats.mean

    # The distinguishing property: averaging powers is strictly louder than
    # averaging dB values whenever the sensors differ. A naive
    # mean(to_db(...)) implementation gives 10 dB here and fails.
    mean_of_db = np.mean(10 * np.log10(S), axis=0)
    assert stats.mean[0] > mean_of_db[0] + 5.0, (stats.mean[0], mean_of_db[0])
    # Uniform column: every definition agrees.
    assert np.isclose(stats.mean[1], mean_of_db[1])

    # The band always encloses its centre line.
    assert np.all(stats.lo <= stats.mean + 1e-12)
    assert np.all(stats.mean <= stats.hi + 1e-12)
    print("  ok")


def test_envelope_independent_grid_csv(tmp_path=None):
    """Two bands on their own grids, one axes, long-format CSV."""
    print("Test 20: envelope on an independent grid...")
    out = _out_dir(tmp_path)
    n_freq_sol, n_freq_val = 4, 6
    f_val = [150.0, 250.0, 350.0, 450.0, 550.0, 650.0]

    cfg = _indep_grid_config(out, n_freq_sol, n_freq_val, f_val,
                             kind='envelope')
    cfg['output']['figure_path'] = str(out / 'env.png')
    with _fake_interpolator(_CENTROIDS):
        rp.run(validate_config(cfg))

    fig_path = Path(cfg['output']['figure_path'])
    assert fig_path.exists()

    with open(fig_path.with_suffix('.csv'), newline='') as f:
        rows = list(csv.reader(f))
    assert rows[0] == ['series', 'frequency', 'statistic', 'level_db'], rows[0]
    body = rows[1:]
    # 3 statistics per frequency per series.
    assert len(body) == 3 * (n_freq_sol + n_freq_val), len(body)
    assert {r[2] for r in body} == {'min', 'mean', 'max'}
    sol_f = sorted({float(r[1]) for r in body if r[0] == 'computed'})
    val_f = sorted({float(r[1]) for r in body if r[0] == 'measured'})
    assert sol_f == [100.0, 200.0, 300.0, 400.0], sol_f
    assert val_f == f_val, val_f

    # min <= mean <= max at every frequency of every series.
    by_key = {(r[0], r[1], r[2]): float(r[3]) for r in body}
    for series, freq in {(r[0], r[1]) for r in body}:
        lo = by_key[(series, freq, 'min')]
        mid = by_key[(series, freq, 'mean')]
        hi = by_key[(series, freq, 'max')]
        assert lo <= mid + 1e-9 <= hi + 1e-9, (series, freq, lo, mid, hi)
    print("  ok")


def test_config_envelope_allowed_on_independent_grid(tmp_path=None):
    """envelope needs no pairing and no validation file."""
    print("Test 21: envelope config acceptance...")
    out = _out_dir(tmp_path)
    f_val = [150.0, 250.0, 350.0]

    # Accepted alongside an independent validation grid.
    cfg = _indep_grid_config(out, 4, 3, f_val, kind='envelope')
    assert validate_config(cfg)['plot']['kind'] == ['envelope']

    # Accepted with no validation data at all: the computed band alone.
    cfg = _indep_grid_config(out, 4, 3, None, kind='envelope')
    cfg['input']['validation_path'] = None
    cfg['input']['validation_frequencies'] = None
    cfg['selection'] = {'indices': [0, 1, 2], 'coordinates': None,
                        'match_tolerance': None}
    cfg['output']['figure_path'] = str(out / 'env_solo.png')
    validated = validate_config(cfg)
    assert validated['plot']['kind'] == ['envelope']
    rp.run(validated)
    assert (out / 'env_solo.png').exists()

    with open((out / 'env_solo.csv'), newline='') as f:
        rows = list(csv.reader(f))
    assert {r[0] for r in rows[1:]} == {'computed'}
    assert len(rows) - 1 == 3 * 4
    print("  ok")


def run_all_tests() -> bool:
    print("=" * 60)
    print("Running CPSD Diagonal Plotting Tests")
    print("=" * 60)
    tests = [
        test_validation_extraction_and_slicing,
        test_relative_l2_error,
        test_coordinate_alignment_integrity,
        test_box_band_switchover,
        test_config_validation,
        test_db_conversion_and_floor,
        test_signed_db_error,
        test_pooled_and_per_sensor_stats,
        test_per_sensor_cap_and_naming,
        test_config_validation_db,
        test_independent_grid_loads_unsliced,
        test_config_rejects_comparison_kinds_on_independent_grid,
        test_config_rejects_index_axis_with_independent_grid,
        test_lines_long_format_csv,
        test_validation_db_independent_grid_single_panel,
        test_validation_db_independent_grid_csv,
        test_per_sensor_order_falls_back_to_selection_order,
        test_legend_sits_outside_axes,
        test_envelope_statistics,
        test_envelope_independent_grid_csv,
        test_config_envelope_allowed_on_independent_grid,
    ]
    passed = failed = 0
    for t in tests:
        try:
            with tempfile.TemporaryDirectory() as td:
                # Tests that write files accept an optional tmp_path kwarg.
                try:
                    t(tmp_path=td)
                except TypeError:
                    t()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


if __name__ == '__main__':
    sys.exit(0 if run_all_tests() else 1)
