"""
Unit tests for frequency_spec (shared frequency-vector spec parsing).

Implements only the correctness check approved during planning:
1. The three spec forms (`.npy`/`.mat` path, inline list, {min, step, max}
   dict) resolve to the same array; the dict form's endpoint handling is
   pinned explicitly; and the `run_cpsd_inverse.parse_frequencies` wrapper
   keeps its pre-refactor return type and error wording.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from frequency_spec import load_frequency_spec, parse_frequency_spec


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
    return Path(tempfile.mkdtemp(prefix='test_frequency_spec_'))


def test_frequency_spec_forms(tmp_path=None):
    """All spec forms agree; dict endpoints pinned; wrapper unchanged."""
    print("Test 1: frequency spec forms...")
    out = _out_dir(tmp_path)
    expected = np.array([10.0, 15.0, 20.0, 25.0, 30.0])

    # --- the three forms agree ------------------------------------------
    npy = out / 'f_val.npy'
    np.save(npy, expected)
    mat = out / 'f_val.mat'
    savemat(mat, {'f_val': expected})

    from_list = load_frequency_spec(expected.tolist(), 'input.f')
    from_dict = load_frequency_spec({'min': 10, 'step': 5, 'max': 30}, 'input.f')
    from_npy = load_frequency_spec(str(npy), 'input.f')
    from_mat = load_frequency_spec(str(mat), 'input.f')

    for got, name in ((from_list, 'list'), (from_dict, 'dict'),
                      (from_npy, 'npy'), (from_mat, 'mat')):
        assert got.shape == expected.shape, (name, got.shape)
        assert np.allclose(got, expected), (name, got)
        assert got.dtype == np.float64, (name, got.dtype)

    # A (1, n) / (n, 1) MATLAB row/column vector squeezes to 1-D.
    savemat(out / 'f_row.mat', {'f': expected.reshape(1, -1)})
    assert np.allclose(load_frequency_spec(str(out / 'f_row.mat'), 'input.f'),
                       expected)

    # --- dict endpoint handling, pinned explicitly ----------------------
    # max lands exactly on a step -> included.
    assert np.allclose(
        parse_frequency_spec({'min': 10, 'step': 5, 'max': 30}, 'input.f'),
        [10, 15, 20, 25, 30],
    )
    # max does not land on a step -> last sample stays below it.
    assert np.allclose(
        parse_frequency_spec({'min': 10, 'step': 7, 'max': 30}, 'input.f'),
        [10, 17, 24],
    )
    # min == max -> single sample.
    assert np.allclose(
        parse_frequency_spec({'min': 5, 'step': 1, 'max': 5}, 'input.f'), [5.0]
    )

    # --- errors name the caller's field ---------------------------------
    bad_specs = [
        [],                                   # empty list
        [10.0, -1.0],                         # non-positive entry
        {'min': 10, 'step': 5},               # missing 'max'
        {'min': 10, 'step': 5, 'max': 1},     # max < min
        {'min': 0, 'step': 5, 'max': 30},     # non-positive min
        42,                                   # wrong type
    ]
    for spec in bad_specs:
        try:
            parse_frequency_spec(spec, 'input.validation_frequencies')
            raise AssertionError(f"expected ValueError for spec {spec!r}")
        except ValueError as e:
            assert 'input.validation_frequencies' in str(e), str(e)

    # A .mat holding more than one variable is ambiguous -> raise, and say
    # which keys were found.
    savemat(out / 'two.mat', {'a': expected, 'b': expected})
    try:
        load_frequency_spec(str(out / 'two.mat'), 'input.f')
        raise AssertionError("expected ValueError for ambiguous .mat")
    except ValueError as e:
        assert 'a' in str(e) and 'b' in str(e), str(e)

    # A 2-D array that cannot squeeze to 1-D -> raise.
    np.save(out / 'f_2d.npy', np.ones((3, 4)))
    try:
        load_frequency_spec(str(out / 'f_2d.npy'), 'input.f')
        raise AssertionError("expected ValueError for 2-D frequency array")
    except ValueError:
        pass

    # --- the run_cpsd_inverse wrapper keeps its contract ----------------
    import run_cpsd_inverse as rci

    got = rci.parse_frequencies({'min': 10, 'step': 5, 'max': 30})
    assert isinstance(got, list) and all(isinstance(f, float) for f in got)
    assert np.allclose(got, expected)
    try:
        rci.parse_frequencies(42)
        raise AssertionError("expected ValueError from parse_frequencies")
    except ValueError as e:
        assert 'physics.frequencies' in str(e), str(e)
    print("  ok")


def run_all_tests() -> bool:
    print("=" * 60)
    print("Running Frequency Spec Tests")
    print("=" * 60)
    tests = [test_frequency_spec_forms]
    passed = failed = 0
    for t in tests:
        try:
            with tempfile.TemporaryDirectory() as td:
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
