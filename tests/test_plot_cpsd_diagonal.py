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

Plot rendering itself is smoke-tested (saves without error) rather than
pixel-asserted.
"""

import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless
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

    out = Path(tmp_path) if tmp_path else Path('.')
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
    out = Path(tmp_path) if tmp_path else Path('.')
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
    out = Path(tmp_path) if tmp_path else Path('.')
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
    out = Path(tmp_path) if tmp_path else Path('.')
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


def run_all_tests() -> bool:
    import tempfile

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
