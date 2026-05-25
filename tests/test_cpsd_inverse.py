"""
Unit tests for the cpsd_inverse module.

Implements only the two correctness checks approved during planning:
1. Recovery on synthetic data with several (m, n) shapes (m >= n).
2. n=1 scalar reduction: s = |t|^2 g / (|t|^4 + alpha).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cpsd_inverse import CPSDInverseSolver


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
    For T_r shape (1, 1, 1), scalar real g, alpha >= 0, the closed-form
    reduces to s = |t|^2 g / (|t|^4 + alpha).
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

        expected = (abs(t) ** 2 * g) / (abs(t) ** 4 + alpha)
        np.testing.assert_allclose(
            s_rec, expected, rtol=1e-12, atol=1e-14,
            err_msg=f"scalar mismatch for (t={t}, g={g}, alpha={alpha})"
        )
    print("  PASSED")


def run_all_tests() -> bool:
    print("=" * 60)
    print("Running CPSD Inverse Solver Tests")
    print("=" * 60)
    tests = [test_recovery_synthetic, test_scalar_reduction]
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
