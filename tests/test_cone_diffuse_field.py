"""
Unit tests for cone diffuse field analysis.

Tests verify:
1. Incident field matrix D computation
2. Covariance matrix-vector product correctness
3. Eigenvalue computation accuracy
4. Variance ratio criterion
5. Eigenvector orthonormality
6. Input validation
7. Randomized eigensolver utility
8. Wavenumber computation
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from randomized_eigensolver import RandomizedEigensolver, compute_eigenvalues_for_variance
from cone_diffuse_field import ConeDiffuseField


def create_test_cone_data(ndof=50, npws=100, nfreqs=3):
    """Create synthetic test data for cone analysis."""
    rng = np.random.default_rng(42)

    # Random transfer matrix (complex) - scattered field
    T = (rng.standard_normal((ndof, npws, nfreqs)) +
         1j * rng.standard_normal((ndof, npws, nfreqs)))

    # Random coordinates on a cone surface
    theta = rng.uniform(0, 2 * np.pi, ndof)
    z = rng.uniform(0, 1, ndof)
    r = 0.5 * (1 - z)  # Cone shape: radius decreases with height
    coords = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])

    # Random unit directions for plane waves
    phi = rng.uniform(0, 2 * np.pi, npws)
    cos_theta = rng.uniform(-1, 1, npws)
    sin_theta = np.sqrt(1 - cos_theta**2)
    directions = np.column_stack([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta
    ])

    # Frequencies
    frequencies = np.array([100.0, 200.0, 300.0])[:nfreqs]

    cone_geometry = {
        'half_angle': np.pi / 6,  # 30 degrees
        'height': 1.0
    }

    return T, coords, directions, frequencies, cone_geometry


def test_incident_field_matrix():
    """
    Test 1: Verify incident field matrix D computation.

    D[i,j] = exp(i * k * d_j · x_i)
    """
    print("Test 1: Incident field matrix D computation...")

    ndof, npws, nfreqs = 20, 30, 1
    T, coords, directions, frequencies, cone_geom = create_test_cone_data(ndof, npws, nfreqs)

    c = 343.0
    cone = ConeDiffuseField(T, coords, directions, frequencies, c, 1.0, cone_geom)

    freq_idx = 0
    k = cone.wavenumbers[freq_idx]

    # Compute D using class method
    D_computed = cone._compute_incident_field_matrix(freq_idx)

    # Compute D directly
    D_expected = np.zeros((ndof, npws), dtype=np.complex128)
    for i in range(ndof):
        for j in range(npws):
            dot_product = np.dot(directions[j], coords[i])
            D_expected[i, j] = np.exp(1j * k * dot_product)

    np.testing.assert_allclose(
        D_computed, D_expected, rtol=1e-10,
        err_msg="Incident field matrix D computation mismatch"
    )

    # Check dimensions
    assert D_computed.shape == (ndof, npws), f"D shape mismatch: {D_computed.shape}"

    print("  PASSED")


def test_covariance_matvec_correctness():
    """
    Test 2: Verify C @ v matches explicit (Po² * H @ H^H) @ v for small problem.
    """
    print("Test 2: Covariance matvec correctness...")

    ndof, npws, nfreqs = 20, 30, 2
    T, coords, directions, frequencies, cone_geom = create_test_cone_data(ndof, npws, nfreqs)

    Po = 1.5
    c = 343.0
    cone = ConeDiffuseField(T, coords, directions, frequencies, c, Po, cone_geom)

    rng = np.random.default_rng(42)

    for freq_idx in range(nfreqs):
        # Get total field matrix H = D + T
        H = cone._compute_total_field_matrix(freq_idx)

        # Explicit covariance matrix
        C_explicit = Po**2 * H @ H.conj().T

        # Test with random vector
        v = rng.standard_normal(ndof) + 1j * rng.standard_normal(ndof)

        # Matrix-free result
        matvec_result = cone._covariance_matvec(freq_idx, v)

        # Explicit result
        explicit_result = C_explicit @ v

        np.testing.assert_allclose(
            matvec_result, explicit_result, rtol=1e-10,
            err_msg=f"Matvec mismatch at freq_idx={freq_idx}"
        )

    print("  PASSED")


def test_eigenvalue_accuracy():
    """
    Test 3: Compare direct and randomized eigensolvers against explicit computation.
    """
    print("Test 3: Eigenvalue computation accuracy (both solvers)...")

    ndof, npws, nfreqs = 25, 40, 1
    T, coords, directions, frequencies, cone_geom = create_test_cone_data(ndof, npws, nfreqs)

    Po = 1.0
    c = 343.0
    cone = ConeDiffuseField(T, coords, directions, frequencies, c, Po, cone_geom)

    freq_idx = 0

    # Get total field matrix and compute explicit covariance
    H = cone._compute_total_field_matrix(freq_idx)
    C_explicit = Po**2 * H @ H.conj().T
    C_explicit = (C_explicit + C_explicit.conj().T) / 2  # Ensure Hermitian

    # Explicit eigenvalues
    eigvals_exact, _ = np.linalg.eigh(C_explicit)
    eigvals_exact = eigvals_exact[::-1]  # Descending order

    n_components = 10

    # Direct solver eigenvalues (returns ALL eigenvalues, truncates eigenvectors)
    eigvals_direct, eigvecs_direct = cone.compute_covariance_eigenvalues(
        freq_idx=freq_idx,
        n_components=n_components,
        solver='direct'
    )

    # Direct solver should return all eigenvalues matching exact
    np.testing.assert_allclose(
        eigvals_direct, eigvals_exact[:len(eigvals_direct)],
        rtol=1e-10,
        err_msg="Direct eigenvalues differ from exact"
    )

    # Verify eigenvectors are truncated to n_components
    assert eigvecs_direct.shape[1] == n_components, (
        f"Expected {n_components} eigenvectors, got {eigvecs_direct.shape[1]}"
    )

    # Randomized solver eigenvalues
    eigvals_rand, eigvecs_rand = cone.compute_covariance_eigenvalues(
        freq_idx=freq_idx,
        n_components=n_components,
        solver='randomized',
        random_state=42
    )

    # Randomized solver should be close (within 5%) for top eigenvalues
    np.testing.assert_allclose(
        eigvals_rand[:n_components], eigvals_exact[:n_components],
        rtol=0.05,
        err_msg="Randomized eigenvalues differ from exact"
    )

    print("  PASSED")


def test_variance_ratio_criterion():
    """
    Test 4: Verify correct number of eigenvalues selected to capture specified variance.
    Tests both direct and randomized solvers.
    """
    print("Test 4: Variance ratio criterion (both solvers)...")

    ndof, npws, nfreqs = 30, 50, 1
    T, coords, directions, frequencies, cone_geom = create_test_cone_data(ndof, npws, nfreqs)

    Po = 1.0
    c = 343.0
    cone = ConeDiffuseField(T, coords, directions, frequencies, c, Po, cone_geom)

    freq_idx = 0
    var_ratio = 0.90

    # Test direct solver
    eigenvalues_direct, _ = cone.compute_covariance_eigenvalues(
        freq_idx=freq_idx,
        var_ratio=var_ratio,
        solver='direct'
    )
    _, cumulative_direct = cone.get_variance_explained()
    assert cumulative_direct[-1] >= var_ratio, (
        f"Direct solver: Variance captured ({cumulative_direct[-1]:.4f}) is less than "
        f"requested ({var_ratio})"
    )

    # Test randomized solver
    eigenvalues_rand, _ = cone.compute_covariance_eigenvalues(
        freq_idx=freq_idx,
        var_ratio=var_ratio,
        solver='randomized',
        random_state=42
    )
    _, cumulative_rand = cone.get_variance_explained()
    assert cumulative_rand[-1] >= var_ratio, (
        f"Randomized solver: Variance captured ({cumulative_rand[-1]:.4f}) is less than "
        f"requested ({var_ratio})"
    )

    print(f"  PASSED (direct: {cumulative_direct[-1]:.4f}, randomized: {cumulative_rand[-1]:.4f})")


def test_eigenvector_orthonormality():
    """
    Test 5: Verify Phi^H @ Phi ≈ I for computed eigenvectors.
    """
    print("Test 5: Eigenvector orthonormality...")

    ndof, npws, nfreqs = 40, 60, 1
    T, coords, directions, frequencies, cone_geom = create_test_cone_data(ndof, npws, nfreqs)

    Po = 1.0
    c = 343.0
    cone = ConeDiffuseField(T, coords, directions, frequencies, c, Po, cone_geom)

    n_components = 15
    _, eigenvectors = cone.compute_covariance_eigenvalues(
        freq_idx=0,
        n_components=n_components,
        random_state=42
    )

    # Check orthonormality: Phi^H @ Phi should be identity
    gram = eigenvectors.conj().T @ eigenvectors

    np.testing.assert_allclose(
        gram, np.eye(n_components), atol=1e-10,
        err_msg="Eigenvectors are not orthonormal"
    )

    print("  PASSED")


def test_input_validation():
    """Test 6: Input validation catches errors."""
    print("Test 6: Input validation...")

    T, coords, directions, frequencies, cone_geom = create_test_cone_data()
    c = 343.0

    # Test invalid amplitude
    try:
        ConeDiffuseField(T, coords, directions, frequencies, c, -1.0, cone_geom)
        assert False, "Should have raised ValueError for negative amplitude"
    except ValueError as e:
        assert "amplitude must be positive" in str(e)

    # Test invalid speed of sound
    try:
        ConeDiffuseField(T, coords, directions, frequencies, -1.0, 1.0, cone_geom)
        assert False, "Should have raised ValueError for negative speed of sound"
    except ValueError as e:
        assert "speed_of_sound must be positive" in str(e)

    # Test missing cone geometry keys
    incomplete_geom = {'half_angle': 0.5}  # missing 'height'
    try:
        ConeDiffuseField(T, coords, directions, frequencies, c, 1.0, incomplete_geom)
        assert False, "Should have raised ValueError for missing keys"
    except ValueError as e:
        assert "missing keys" in str(e)

    # Test wrong frequencies shape
    wrong_freqs = np.array([100.0, 200.0])  # Wrong number
    try:
        ConeDiffuseField(T, coords, directions, wrong_freqs, c, 1.0, cone_geom)
        assert False, "Should have raised ValueError for wrong frequencies shape"
    except ValueError as e:
        assert "frequencies must have shape" in str(e)

    print("  PASSED")


def test_randomized_eigensolver():
    """Test 7: Standalone RandomizedEigensolver utility."""
    print("Test 7: RandomizedEigensolver utility...")

    n = 50
    rng = np.random.default_rng(42)

    # Create a random Hermitian PSD matrix
    A_half = rng.standard_normal((n, 30)) + 1j * rng.standard_normal((n, 30))
    A = A_half @ A_half.conj().T

    def matvec(v):
        return A @ v

    solver = RandomizedEigensolver(
        matvec_fn=matvec,
        matrix_size=n,
        n_components=10,
        random_state=42
    )

    eigenvalues, eigenvectors = solver.compute()

    # Compare with numpy
    eigvals_exact, _ = np.linalg.eigh(A)
    eigvals_exact = eigvals_exact[::-1]

    np.testing.assert_allclose(
        eigenvalues, eigvals_exact[:10], rtol=0.05
    )

    print("  PASSED")


def test_wavenumber_computation():
    """Test 8: Wavenumbers are computed correctly."""
    print("Test 8: Wavenumber computation...")

    ndof, npws, nfreqs = 20, 30, 3
    T, coords, directions, frequencies, cone_geom = create_test_cone_data(ndof, npws, nfreqs)

    c = 343.0
    cone = ConeDiffuseField(T, coords, directions, frequencies, c, 1.0, cone_geom)

    # Expected wavenumbers: k = 2πf/c
    expected_k = 2 * np.pi * frequencies / c

    np.testing.assert_allclose(
        cone.wavenumbers, expected_k, rtol=1e-10,
        err_msg="Wavenumber computation mismatch"
    )

    print("  PASSED")


def test_all_freqs_eigenvalue_accuracy():
    """
    Test 9: Verify all-frequencies SVD eigenvalues match explicit covariance.

    C_all = Po^2 * H_all @ H_all^H where H_all = [H_0 | H_1 | ... | H_{nf-1}]
    """
    print("Test 9: All-frequencies eigenvalue accuracy...")

    ndof, npws, nfreqs = 25, 40, 3
    T, coords, directions, frequencies, cone_geom = create_test_cone_data(ndof, npws, nfreqs)

    Po = 1.0
    c = 343.0
    cone = ConeDiffuseField(T, coords, directions, frequencies, c, Po, cone_geom)

    # Explicit: stack H matrices, form covariance, eigendecompose
    H_all = np.hstack([cone._compute_total_field_matrix(i) for i in range(nfreqs)])
    C_all = Po**2 * H_all @ H_all.conj().T
    C_all = (C_all + C_all.conj().T) / 2  # Ensure Hermitian
    eigvals_exact, _ = np.linalg.eigh(C_all)
    eigvals_exact = eigvals_exact[::-1]  # Descending order

    # Method under test
    eigenvalues, eigenvectors = cone.compute_covariance_eigenvalues_all_freqs(
        var_ratio=0.99
    )

    np.testing.assert_allclose(
        eigenvalues, eigvals_exact[:len(eigenvalues)],
        rtol=1e-10,
        err_msg="All-frequencies eigenvalues differ from exact"
    )

    print("  PASSED")


def test_all_freqs_eigenvector_orthonormality():
    """
    Test 10: Verify eigenvectors from all-frequencies SVD are orthonormal.
    """
    print("Test 10: All-frequencies eigenvector orthonormality...")

    ndof, npws, nfreqs = 40, 60, 3
    T, coords, directions, frequencies, cone_geom = create_test_cone_data(ndof, npws, nfreqs)

    cone = ConeDiffuseField(T, coords, directions, frequencies, 343.0, 1.0, cone_geom)

    _, eigenvectors = cone.compute_covariance_eigenvalues_all_freqs(
        n_components=15
    )

    gram = eigenvectors.conj().T @ eigenvectors

    np.testing.assert_allclose(
        gram, np.eye(15), atol=1e-10,
        err_msg="All-frequencies eigenvectors are not orthonormal"
    )

    print("  PASSED")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Cone Diffuse Field Tests (Total Field Formulation)")
    print("=" * 60)

    tests = [
        test_incident_field_matrix,
        test_covariance_matvec_correctness,
        test_eigenvalue_accuracy,
        test_variance_ratio_criterion,
        test_eigenvector_orthonormality,
        test_input_validation,
        test_randomized_eigensolver,
        test_wavenumber_computation,
        test_all_freqs_eigenvalue_accuracy,
        test_all_freqs_eigenvector_orthonormality,
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


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
