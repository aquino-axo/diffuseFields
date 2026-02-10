"""
Unit tests for pressure field interpolation.

Tests verify:
1. Linear function interpolation accuracy
2. Identical grid interpolation (should be exact)
3. Complex field interpolation correctness
4. Batch interpolation consistency
5. Input validation
6. Eigenvector batch interpolation
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pressure_interpolator import PressureFieldInterpolator


def create_test_cone_coordinates(n_points=100, height=1.0, half_angle=np.pi/6, seed=42):
    """Create synthetic cone surface coordinates for testing."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n_points)
    z = rng.uniform(0, height, n_points)
    r = (height - z) * np.tan(half_angle)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), z])


def test_linear_function_interpolation():
    """
    Test 1: Linear function should be interpolated with good accuracy.

    For f(x,y,z) = a*x + b*y + c*z + d (affine function),
    RBF with thin_plate_spline should reproduce it accurately.
    """
    print("Test 1: Linear function interpolation accuracy...")

    source_coords = create_test_cone_coordinates(n_points=200, seed=42)
    target_coords = create_test_cone_coordinates(n_points=50, seed=123)

    def affine_func(coords):
        return 2*coords[:, 0] + 3*coords[:, 1] - coords[:, 2] + 5

    source_field = affine_func(source_coords) + 1j * (-affine_func(source_coords) + 1)
    expected = affine_func(target_coords) + 1j * (-affine_func(target_coords) + 1)

    interpolator = PressureFieldInterpolator(source_coords, target_coords)
    result = interpolator.interpolate(source_field)

    # RBF should be accurate for smooth functions
    np.testing.assert_allclose(
        result, expected, rtol=1e-6,
        err_msg="Linear function interpolation should be accurate"
    )

    print("  PASSED")


def test_identical_grid_interpolation():
    """
    Test 2: Interpolation on identical source/target should be exact.

    This test verifies the RBF fix - with the old LinearNDInterpolator,
    identical grids would show ~7% false extrapolation due to convex hull
    boundary issues.
    """
    print("Test 2: Identical grid interpolation (RBF fix verification)...")

    coords = create_test_cone_coordinates(n_points=100, seed=42)

    rng = np.random.default_rng(123)
    source_field = (rng.standard_normal(100) +
                    1j * rng.standard_normal(100)).astype(np.complex128)

    # Interpolate from coords to same coords
    interpolator = PressureFieldInterpolator(coords, coords)
    result = interpolator.interpolate(source_field)

    # Should be exact (within numerical precision)
    np.testing.assert_allclose(
        result, source_field, rtol=1e-10,
        err_msg="Identical grid interpolation should be exact"
    )

    # RBF should report 0 extrapolation
    info = interpolator.get_extrapolation_info()
    assert info['n_extrapolated'] == 0, (
        f"Expected 0 extrapolated for identical grid, got {info['n_extrapolated']}"
    )

    print("  PASSED")


def test_complex_field_interpolation():
    """
    Test 3: Complex field components should be interpolated independently.

    Verifies that real and imaginary parts are handled correctly.
    """
    print("Test 3: Complex field interpolation correctness...")

    source_coords = create_test_cone_coordinates(n_points=150, seed=42)
    # Use a subset of source coords as targets
    target_coords = source_coords[:50].copy()

    rng = np.random.default_rng(123)
    source_field = (rng.standard_normal(150) +
                    1j * rng.standard_normal(150)).astype(np.complex128)

    interpolator = PressureFieldInterpolator(source_coords, target_coords)
    result = interpolator.interpolate(source_field)

    # At source points that are also target points, values should match exactly
    expected = source_field[:50]

    np.testing.assert_allclose(
        result, expected, rtol=1e-10,
        err_msg="Interpolation at source points should return source values"
    )

    print("  PASSED")


def test_batch_interpolation():
    """
    Test 4: Batch interpolation should give same results as individual.

    Verifies (n_source, n_fields) input shape handling.
    """
    print("Test 4: Batch interpolation consistency...")

    source_coords = create_test_cone_coordinates(n_points=100, seed=42)
    target_coords = create_test_cone_coordinates(n_points=30, seed=789)

    rng = np.random.default_rng(456)
    n_fields = 5
    source_fields = (rng.standard_normal((100, n_fields)) +
                     1j * rng.standard_normal((100, n_fields)))

    interpolator = PressureFieldInterpolator(source_coords, target_coords)

    # Batch interpolation
    batch_result = interpolator.interpolate(source_fields)

    # Individual interpolation
    individual_results = np.zeros((30, n_fields), dtype=np.complex128)
    for i in range(n_fields):
        individual_results[:, i] = interpolator.interpolate(source_fields[:, i])

    # Check shape
    assert batch_result.shape == (30, n_fields), (
        f"Expected shape (30, {n_fields}), got {batch_result.shape}"
    )

    # Check values match
    np.testing.assert_allclose(
        batch_result, individual_results, rtol=1e-10,
        err_msg="Batch interpolation should match individual interpolation"
    )

    print("  PASSED")


def test_input_validation():
    """
    Test 5: Input validation catches errors.
    """
    print("Test 5: Input validation...")

    valid_coords = create_test_cone_coordinates(n_points=50, seed=42)

    # Test wrong coordinate dimensions
    wrong_shape = np.random.rand(50, 2)  # Should be (n, 3)
    try:
        PressureFieldInterpolator(wrong_shape, valid_coords)
        assert False, "Should have raised ValueError for wrong shape"
    except ValueError as e:
        assert "3" in str(e) or "shape" in str(e).lower()

    # Test wrong pressure field size
    interpolator = PressureFieldInterpolator(valid_coords, valid_coords[:20])
    wrong_size_field = np.ones(30)  # Should be 50
    try:
        interpolator.interpolate(wrong_size_field)
        assert False, "Should have raised ValueError for wrong field size"
    except ValueError as e:
        assert "50" in str(e) or "rows" in str(e).lower()

    print("  PASSED")


def test_eigenvector_batch_interpolation():
    """
    Test 6: Eigenvector-like batch interpolation preserves structure.

    Simulates the eigenvector interpolation workflow where multiple
    complex eigenvectors (columns) are interpolated together.
    """
    print("Test 6: Eigenvector batch interpolation...")

    # Simulate eigendata structure: (ndof, n_modes)
    n_source = 100
    n_target = 80
    n_modes = 5

    source_coords = create_test_cone_coordinates(n_points=n_source, seed=42)
    target_coords = create_test_cone_coordinates(n_points=n_target, seed=123)

    rng = np.random.default_rng(789)

    # Create mock eigenvectors (complex, each column is a mode)
    eigenvectors = (rng.standard_normal((n_source, n_modes)) +
                    1j * rng.standard_normal((n_source, n_modes)))

    interpolator = PressureFieldInterpolator(source_coords, target_coords)
    result = interpolator.interpolate(eigenvectors)

    # Verify output shape matches expected (n_target, n_modes)
    assert result.shape == (n_target, n_modes), (
        f"Expected shape ({n_target}, {n_modes}), got {result.shape}"
    )

    # Verify dtype is complex
    assert result.dtype == np.complex128, (
        f"Expected dtype complex128, got {result.dtype}"
    )

    # Verify extrapolation info is available (RBF always returns 0)
    info = interpolator.get_extrapolation_info()
    assert info['n_extrapolated'] == 0, "RBF should have 0 extrapolated points"
    assert info['extrapolation_ratio'] == 0.0

    print("  PASSED")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Pressure Field Interpolator Tests (RBF)")
    print("=" * 60)

    tests = [
        test_linear_function_interpolation,
        test_identical_grid_interpolation,
        test_complex_field_interpolation,
        test_batch_interpolation,
        test_input_validation,
        test_eigenvector_batch_interpolation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
