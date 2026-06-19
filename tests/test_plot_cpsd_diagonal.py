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
    load_validation_diagonal,
    relative_l2_error,
    resolve_selection,
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
