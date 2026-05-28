"""
Unit tests for the cpsd_inverse module.

Implements only the correctness checks approved during planning:
1. Recovery on synthetic data with several (m, n) shapes (m >= n).
2. n=1 scalar reduction: s = |t|^2 g / (|t|^4 + alpha).
3. Row-index subset: apply_row_subset slices T_r and G symmetrically.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cpsd_inverse import CPSDInverseSolver
from run_cpsd_inverse import apply_row_subset


def _random_psd(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random Hermitian PSD matrix of shape (n, n)."""
    A = (rng.standard_normal((n, n + 2))
         + 1j * rng.standard_normal((n, n + 2)))
    M = A @ A.conj().T
    return 0.5 * (M + M.conj().T)


def test_recovery_synthetic():
    """
    Pick S_r_true PSD, form G = T_r S_r_true T_r^h, invert with tiny alpha,
    expect S_r ~ S_r_true. Requires m >= n for unique recovery.
    """
    print("Test 1: synthetic recovery (multiple shapes)...")
    rng = np.random.default_rng(0)

    cases = [
        # (n_sensors=m, n_pod=n, n_freq)
        (10, 5, 2),
        (20, 8, 1),
        (12, 12, 3),
    ]
    for m, n, nf in cases:
        T_r = (rng.standard_normal((m, n, nf))
               + 1j * rng.standard_normal((m, n, nf)))
        solver = CPSDInverseSolver(T_r)

        for f_idx in range(nf):
            S_true = _random_psd(n, rng)
            Tf = T_r[:, :, f_idx]
            G = Tf @ S_true @ Tf.conj().T
            G = 0.5 * (G + G.conj().T)

            # Make alpha tiny relative to sigma_max^4 of T_r:
            s_max = np.linalg.svd(Tf, compute_uv=False).max()
            alpha = 1e-14 * s_max ** 4

            S_rec, res = solver.solve_single_freq(
                f_idx, G, np.array([alpha])
            )
            S_rec = S_rec[:, :, 0]

            rel_err = (
                np.linalg.norm(S_rec - S_true, 'fro')
                / np.linalg.norm(S_true, 'fro')
            )
            assert rel_err < 1e-5, (
                f"(m={m}, n={n}, f_idx={f_idx}): "
                f"recovery relative error {rel_err:.2e} exceeds 1e-5"
            )
            assert res[0] < 1e-6, (
                f"(m={m}, n={n}, f_idx={f_idx}): "
                f"residual {res[0]:.2e} exceeds 1e-6"
            )
    print("  PASSED")


def test_scalar_reduction():
    """
    For T_r shape (1, 1, 1) with scalar t and real-positive g, the
    PSD-preserving closed form (eqs. 35-36) reduces to

        S_r = g / (|t| + alpha)^2.

    Derivation: reduced SVD gives sigma = |t|, X = t/|t|, V = 1; the PSD
    square root of [[g]] is sqrt(g), so K = conj(t) sqrt(g) / (|t|(|t|+alpha))
    and S_r = |K|^2 = g/(|t|+alpha)^2.
    """
    print("Test 2: n=1 scalar reduction...")
    rng = np.random.default_rng(1)

    triples = [
        (1.0 + 0.0j, 2.5, 1e-6),
        (0.7 - 0.3j, 1.0, 1e-3),
        (rng.standard_normal() + 1j * rng.standard_normal(), 4.2, 5e-2),
        (2.0 + 1.5j, 9.0, 1.0),
        (0.1j, 1.0, 1e-8),
    ]
    for t, g, alpha in triples:
        T_r = np.array([[[t]]], dtype=np.complex128)
        solver = CPSDInverseSolver(T_r)

        G = np.array([[g + 0j]], dtype=np.complex128)
        S_rec, _ = solver.solve_single_freq(0, G, np.array([alpha]))
        s_rec = S_rec[0, 0, 0]

        expected = g / (abs(t) + alpha) ** 2
        np.testing.assert_allclose(
            s_rec, expected, rtol=1e-12, atol=1e-14,
            err_msg=f"scalar mismatch for (t={t}, g={g}, alpha={alpha})"
        )
    print("  PASSED")


def test_apply_row_subset():
    """
    apply_row_subset must return T_r[I,:,:] and G[I,I,:] for the chosen
    index set, with all frequency slices preserved. Exercises a
    non-contiguous, unordered, integer-valued index set.
    """
    print("Test 3: apply_row_subset slicing...")
    rng = np.random.default_rng(2)

    m, n, nf = 9, 4, 3
    T_r = (rng.standard_normal((m, n, nf))
           + 1j * rng.standard_normal((m, n, nf)))
    G = (rng.standard_normal((m, m, nf))
         + 1j * rng.standard_normal((m, m, nf)))
    # Hermitize G per frequency (matches the experimental-CPSD convention).
    for f in range(nf):
        G[:, :, f] = 0.5 * (G[:, :, f] + G[:, :, f].conj().T)

    row_idx = np.array([7, 1, 4, 0, 5], dtype=np.int64)  # unordered subset
    T_r_sub, G_sub = apply_row_subset(T_r, G, row_idx)

    assert T_r_sub.shape == (row_idx.size, n, nf)
    assert G_sub.shape == (row_idx.size, row_idx.size, nf)

    for f in range(nf):
        np.testing.assert_array_equal(T_r_sub[:, :, f], T_r[row_idx, :, f])
        # Symmetric (rows AND cols) subset of G.
        expected_G = G[np.ix_(row_idx, row_idx, [f])][:, :, 0]
        np.testing.assert_array_equal(G_sub[:, :, f], expected_G)
    print("  PASSED")


def run_all_tests() -> bool:
    print("=" * 60)
    print("Running CPSD Inverse Solver Tests")
    print("=" * 60)
    tests = [
        test_recovery_synthetic,
        test_scalar_reduction,
        test_apply_row_subset,
    ]
    passed = failed = 0
    for t in tests:
        try:
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
