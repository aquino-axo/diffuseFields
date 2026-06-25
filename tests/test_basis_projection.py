"""
Tests for the BasisProjection per-frequency residual computation.

Follows the repository's manual run_all_tests() pattern (no pytest).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from basis_projector import BasisProjection


def _random_complex(rng, shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def test_data_in_span_residual_zero():
    """Test 1: data built from basis columns -> residual ~ 0 at all frequencies."""
    print("Test 1: data inside basis span -> residual ~ 0...")
    rng = np.random.default_rng(0)
    ndof, npws_b, npws_d, nfreq = 40, 8, 5, 4

    basis = _random_complex(rng, (ndof, npws_b, nfreq))
    data = np.empty((ndof, npws_d, nfreq), dtype=np.complex128)
    for i in range(nfreq):
        # Each data column is a random linear combination of basis columns.
        coeffs = _random_complex(rng, (npws_b, npws_d))
        data[:, :, i] = basis[:, :, i] @ coeffs

    result = BasisProjection(basis, data).project()
    np.testing.assert_array_less(result["relative_residual"], 1e-10)
    print("  PASSED")


def test_data_orthogonal_residual_one():
    """Test 2: data orthogonal to col(basis) -> residual ~ 1."""
    print("Test 2: data orthogonal to basis -> residual ~ 1...")
    rng = np.random.default_rng(1)
    half, npws_b, npws_d, nfreq = 20, 6, 4, 3
    ndof = 2 * half

    # Basis lives entirely in the first block of coordinates, data in the second:
    # the two column spaces are orthogonal, so no part of the data is captured.
    basis = np.zeros((ndof, npws_b, nfreq), dtype=np.complex128)
    data = np.zeros((ndof, npws_d, nfreq), dtype=np.complex128)
    for i in range(nfreq):
        basis[:half, :, i] = _random_complex(rng, (half, npws_b))
        data[half:, :, i] = _random_complex(rng, (half, npws_d))

    result = BasisProjection(basis, data).project()
    np.testing.assert_allclose(result["relative_residual"], 1.0, atol=1e-12)
    print("  PASSED")


def test_known_analytic_case():
    """Test 3: closed-form projection onto an axis-aligned subspace."""
    print("Test 3: known analytic residual...")
    # Basis column space = span{e_x} in R^3. Data column = (3, 4, 0)^T.
    # Projection keeps (3,0,0); residual = ||(0,4,0)|| / ||(3,4,0)|| = 4/5.
    basis = np.array([[1.0], [0.0], [0.0]]).reshape(3, 1, 1)
    data = np.array([[3.0], [4.0], [0.0]]).reshape(3, 1, 1)

    result = BasisProjection(basis, data).project()
    np.testing.assert_allclose(result["relative_residual"][0], 0.8, atol=1e-12)
    assert result["basis_rank"][0] == 1, result["basis_rank"][0]
    print("  PASSED")


def test_rank_tolerance():
    """Test 4: near-zero singular value excluded from the numerical rank."""
    print("Test 4: rtol controls basis numerical rank...")
    ndof, nfreq = 10, 1
    # Two well-scaled orthogonal columns plus one tiny-norm column.
    col0 = np.zeros(ndof); col0[0] = 1.0
    col1 = np.zeros(ndof); col1[1] = 1.0
    col_tiny = np.zeros(ndof); col_tiny[2] = 1e-9
    basis = np.stack([col0, col1, col_tiny], axis=1).reshape(ndof, 3, nfreq)
    data = np.zeros((ndof, 1, nfreq)); data[0, 0, 0] = 1.0
    data = data.astype(np.complex128)

    # Loose tolerance drops the tiny column (rank 2); tight tolerance keeps it (rank 3).
    rank_loose = BasisProjection(basis, data, rtol=1e-6).project()["basis_rank"][0]
    rank_tight = BasisProjection(basis, data, rtol=1e-12).project()["basis_rank"][0]
    assert rank_loose == 2, rank_loose
    assert rank_tight == 3, rank_tight
    print("  PASSED")


def test_input_validation():
    """Test 5: shape mismatches and non-3D inputs raise ValueError."""
    print("Test 5: input validation...")
    good = np.zeros((10, 4, 3), dtype=np.complex128)

    # Mismatched ndof.
    try:
        BasisProjection(good, np.zeros((8, 4, 3), dtype=np.complex128))
        raise AssertionError("expected ValueError for mismatched ndof")
    except ValueError:
        pass

    # Mismatched nfreq.
    try:
        BasisProjection(good, np.zeros((10, 4, 2), dtype=np.complex128))
        raise AssertionError("expected ValueError for mismatched nfreq")
    except ValueError:
        pass

    # Non-3D input.
    try:
        BasisProjection(np.zeros((10, 4)), good)
        raise AssertionError("expected ValueError for non-3D basis")
    except ValueError:
        pass
    print("  PASSED")


def run_all_tests():
    print("=" * 60)
    print("Running Basis Projection Tests")
    print("=" * 60)

    tests = [
        test_data_in_span_residual_zero,
        test_data_orthogonal_residual_one,
        test_known_analytic_case,
        test_rank_tolerance,
        test_input_validation,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
