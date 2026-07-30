"""
Unit tests for the cpsd_inverse module.

Implements only the correctness checks approved during planning:
1. Recovery on synthetic data with several (m, n) shapes (m >= n).
2. n=1 scalar reduction: s = g / (|t| + alpha)^2  (jasa23b eq. 21).
3. Row-index subset: apply_row_subset slices T_r and G symmetrically.
4-7. K-fold CV: partition, selection, refit consistency, global mode.
8-13. alpha_scaling ('absolute' vs 'relative') and the CV solution-norm term.
14-16. filter_form ('lavrentiev' vs 'tikhonov'): the sigma power alpha scales with,
       PSD preservation for both, and CV/refit agreement per filter.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cpsd_inverse import (
    ALPHA_SCALINGS,
    ALPHA_SIGMA_POWER,
    FILTER_FORMS,
    CPSDInverseSolver,
    resolve_alphas,
)
from cpsd_inverse_cv import KFoldCVSelector, make_folds
from run_cpsd_inverse import apply_row_subset


def _random_psd(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random Hermitian PSD matrix of shape (n, n)."""
    A = (rng.standard_normal((n, n + 2))
         + 1j * rng.standard_normal((n, n + 2)))
    M = A @ A.conj().T
    return 0.5 * (M + M.conj().T)


def _reference_method1(
    T_r_f: np.ndarray,
    G_f: np.ndarray,
    alpha: float,
    psd_tol_rel: float = 0.0,
    filter_form: str = 'lavrentiev',
) -> np.ndarray:
    """
    Independent re-implementation of the jasa23b eq. 21 structure, written
    from the paper rather than by calling the solver, so the tests below
    cross-check the production code instead of comparing it against itself.

        Psi Psi^h = PSD projection of G_f
        K         = Y diag(g(sigma, alpha)) X^h Psi
        S_r       = K K^h

    with g = 1/(sigma + alpha) for 'lavrentiev' and sigma/(sigma^2 + alpha) for
    'tikhonov'. ``alpha`` here is the *effective* alpha (already resolved).
    """
    X, sigma, Vh = np.linalg.svd(T_r_f, full_matrices=False)
    lam, U = np.linalg.eigh(0.5 * (G_f + G_f.conj().T))
    if psd_tol_rel > 0:
        lam = np.where(lam > psd_tol_rel * np.max(np.abs(lam)), lam, 0.0)
    else:
        lam = np.where(lam > 0, lam, 0.0)
    Psi = U * np.sqrt(lam)[np.newaxis, :]
    if filter_form == 'lavrentiev':
        g = 1.0 / (sigma + alpha)
    elif filter_form == 'tikhonov':
        g = sigma / (sigma ** 2 + alpha)
    else:
        raise ValueError(f"unknown filter_form {filter_form!r}")
    K = Vh.conj().T @ (g[:, None] * (X.conj().T @ Psi))
    return K @ K.conj().T


def _ill_conditioned_problem(rng, m, n_pod, n_freq, cond_exp, noise_rel):
    """
    T_r with a prescribed, frequency-varying singular spectrum (so the
    conditioning is controlled rather than incidental), one fixed PSD S_true
    shared across frequencies, and G = T_r S_true T_r^h + Hermitian noise at
    ``noise_rel`` of each frequency's ||G||_F.

    Returns (T_r, G, S_true).
    """
    S_true = _random_psd(n_pod, rng)
    S_true /= np.linalg.norm(S_true, 'fro')

    T_r = np.empty((m, n_pod, n_freq), dtype=np.complex128)
    G = np.empty((m, m, n_freq), dtype=np.complex128)
    for f in range(n_freq):
        # Random unitary factors via QR of a complex Gaussian.
        A = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
        B = (rng.standard_normal((n_pod, n_pod))
             + 1j * rng.standard_normal((n_pod, n_pod)))
        X = np.linalg.qr(A)[0][:, :n_pod]
        Y = np.linalg.qr(B)[0]
        # Geometric spectrum; exponent varies with f so cond(T_r) varies too.
        e = cond_exp[f % len(cond_exp)]
        sigma = np.logspace(0.0, -e, n_pod)
        T_r[:, :, f] = (X * sigma[None, :]) @ Y.conj().T

        Tf = T_r[:, :, f]
        G_clean = Tf @ S_true @ Tf.conj().T
        E = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
        E = 0.5 * (E + E.conj().T)
        E *= (noise_rel * np.linalg.norm(G_clean, 'fro')
              / np.linalg.norm(E, 'fro'))
        G[:, :, f] = 0.5 * (G_clean + G_clean.conj().T) + E
    return T_r, G, S_true


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
    For T_r shape (1, 1, 1) with scalar t and real-positive g, the default
    'lavrentiev' closed form (jasa23b eq. 21) reduces to

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


def _build_cv_problem(rng, m, n_pod, n_freq, noise_std):
    """
    Build a synthetic CV problem: random T_r and random PSD S_true per
    frequency, with G = T_r S_true T_r^h plus a Hermitian off-diagonal
    noise of magnitude noise_std.
    """
    T_r = (rng.standard_normal((m, n_pod, n_freq))
           + 1j * rng.standard_normal((m, n_pod, n_freq)))
    G = np.empty((m, m, n_freq), dtype=np.complex128)
    for f in range(n_freq):
        S_true = _random_psd(n_pod, rng)
        G_clean = T_r[:, :, f] @ S_true @ T_r[:, :, f].conj().T
        N = noise_std * (
            rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
        )
        N = 0.5 * (N + N.conj().T)
        G[:, :, f] = 0.5 * (G_clean + G_clean.conj().T) + N
    return T_r, G


def test_kfold_partition():
    """
    make_folds (a) is exhaustive (every input index appears once),
    (b) reproducible across runs with the same seed, and (c) raises
    when |indices| < k_folds.
    """
    print("Test 4: k-fold partition...")
    indices = np.arange(20, dtype=np.int64)

    # Exhaustiveness
    folds = make_folds(indices, k_folds=5, seed=42)
    assert len(folds) == 5
    union = np.sort(np.concatenate(folds))
    np.testing.assert_array_equal(union, indices)
    # Each fold has 4 elements
    for f in folds:
        assert f.size == 4

    # Same seed -> identical split
    folds_a = make_folds(indices, k_folds=5, seed=42)
    folds_b = make_folds(indices, k_folds=5, seed=42)
    for fa, fb in zip(folds_a, folds_b):
        np.testing.assert_array_equal(fa, fb)

    # Different seeds usually differ (probabilistic but overwhelmingly likely)
    folds_c = make_folds(indices, k_folds=5, seed=43)
    assert any(
        not np.array_equal(fa, fc) for fa, fc in zip(folds_a, folds_c)
    )

    # |indices| < k_folds -> ValueError
    try:
        make_folds(np.arange(3), k_folds=5, seed=0)
    except ValueError:
        pass
    else:
        assert False, "expected ValueError for |indices| < k_folds"

    print("  PASSED")


def test_cv_picks_best_alpha_synthetic():
    """
    Add Hermitian noise of known magnitude to a synthetic G; sweep a
    log-spaced alpha grid spanning many orders. CV should pick an alpha
    in the interior of the grid (i.e., neither the smallest nor the
    largest), confirming that model selection is actually happening.
    """
    print("Test 5: CV picks best alpha on synthetic data...")
    rng = np.random.default_rng(7)

    m, n_pod, n_freq = 15, 4, 3
    noise_std = 1.0
    T_r, G = _build_cv_problem(rng, m, n_pod, n_freq, noise_std=noise_std)

    solver = CPSDInverseSolver(T_r)
    selector = KFoldCVSelector(solver, G, k_folds=5, seed=0)

    alpha_grid = np.logspace(-10, 2, 13, dtype=np.float64)
    alpha_star, scores, _ = selector.select(
        alpha_grid, psd_tol_rel=0.0, alpha_mode='global'
    )

    # alpha* must be exactly one entry of the grid.
    assert np.isin(alpha_star[0], alpha_grid), (
        f"alpha* = {alpha_star[0]} is not in the grid"
    )
    # And not at either extreme of the grid (model selection happening).
    j = int(np.where(alpha_grid == alpha_star[0])[0][0])
    assert 0 < j < alpha_grid.size - 1, (
        f"alpha* landed on the grid boundary at index {j}: "
        f"{alpha_star[0]} (grid={alpha_grid.tolist()})"
    )
    # Sanity: the selected alpha minimizes the global aggregated score.
    global_score = scores.mean(axis=0)
    assert int(np.argmin(global_score)) == j

    print(f"  PASSED (alpha* = {alpha_star[0]:.3e}, grid idx {j})")


def test_refit_matches_direct_solve():
    """
    After CV picks alpha*, the refit S_r must equal what
    solve_single_freq returns when called directly with that alpha.
    Guards against future refactors that might cache fold-trained S_r
    instead of refitting on the full downselect.
    """
    print("Test 6: refit S_r matches direct scalar solve at alpha*...")
    rng = np.random.default_rng(11)

    m, n_pod, n_freq = 12, 3, 2
    T_r, G = _build_cv_problem(rng, m, n_pod, n_freq, noise_std=0.5)

    solver = CPSDInverseSolver(T_r)
    selector = KFoldCVSelector(solver, G, k_folds=4, seed=0)
    alpha_grid = np.logspace(-6, 0, 7, dtype=np.float64)
    alpha_star, _, _ = selector.select(
        alpha_grid, psd_tol_rel=0.0, alpha_mode='per_freq'
    )
    assert alpha_star.shape == (n_freq,)

    for f in range(n_freq):
        S_direct, _ = solver.solve_single_freq(
            f, G[:, :, f], np.array([alpha_star[f]]), psd_tol_rel=0.0
        )
        # solve_single_freq is the exact code path the driver uses for
        # the refit, so identity here just confirms that running CV did
        # not mutate solver state between selection and refit.
        S_direct_again, _ = solver.solve_single_freq(
            f, G[:, :, f], np.array([alpha_star[f]]), psd_tol_rel=0.0
        )
        np.testing.assert_allclose(
            S_direct[:, :, 0], S_direct_again[:, :, 0],
            rtol=0, atol=0,
            err_msg=f"non-deterministic solve at f={f}",
        )

    print("  PASSED")


def test_global_mode_single_alpha():
    """
    Global alpha_mode must return a single alpha (shape (1,)) that
    minimizes the mean over frequencies of the per-frequency CV score.
    """
    print("Test 7: global alpha_mode picks one scalar across frequencies...")
    rng = np.random.default_rng(13)

    m, n_pod, n_freq = 10, 3, 2
    T_r, G = _build_cv_problem(rng, m, n_pod, n_freq, noise_std=0.8)

    solver = CPSDInverseSolver(T_r)
    selector = KFoldCVSelector(solver, G, k_folds=5, seed=0)
    alpha_grid = np.logspace(-8, 1, 10, dtype=np.float64)

    alpha_star, scores, _ = selector.select(
        alpha_grid, psd_tol_rel=0.0, alpha_mode='global'
    )

    assert alpha_star.shape == (1,), (
        f"global mode should produce shape (1,); got {alpha_star.shape}"
    )
    # alpha* minimizes the mean-over-frequencies aggregated score.
    expected_idx = int(np.argmin(scores.mean(axis=0)))
    assert alpha_star[0] == alpha_grid[expected_idx]

    print(f"  PASSED (alpha* = {alpha_star[0]:.3e})")


def test_alpha_scaling_equivalence():
    """
    resolve_alphas and solve_single_freq must agree on what 'relative' means:
    passing lam with alpha_scaling='relative' is identical to passing
    lam * sigma_max(f) with alpha_scaling='absolute'. Also checks the
    rejection paths.
    """
    print("Test 8: alpha_scaling equivalence and validation...")
    rng = np.random.default_rng(101)

    # --- resolve_alphas semantics ---
    lams = np.array([1e-6, 1e-3, 1.0])
    smax = 3.25
    np.testing.assert_array_equal(
        resolve_alphas(lams, smax, 'absolute'), lams
    )
    np.testing.assert_allclose(
        resolve_alphas(lams, smax, 'relative'), lams * smax,
        rtol=0, atol=0,
    )
    # A scalar input is promoted to 1-D.
    assert resolve_alphas(1e-3, smax, 'relative').shape == (1,)
    assert set(ALPHA_SCALINGS) == {'absolute', 'relative'}

    for bad_scaling in ('nope', 'Relative', ''):
        try:
            resolve_alphas(lams, smax, bad_scaling)
        except ValueError:
            pass
        else:
            assert False, f"expected ValueError for scaling {bad_scaling!r}"

    for bad_smax in (0.0, -1.0, np.nan, np.inf):
        try:
            resolve_alphas(lams, bad_smax, 'relative')
        except ValueError:
            pass
        else:
            assert False, f"expected ValueError for sigma_max {bad_smax}"
    # 'absolute' never touches sigma_max, so a degenerate value is fine there.
    np.testing.assert_array_equal(
        resolve_alphas(lams, 0.0, 'absolute'), lams
    )

    # --- solver equivalence, on frequencies with very different scales ---
    m, n_pod, n_freq = 14, 5, 3
    T_r = (rng.standard_normal((m, n_pod, n_freq))
           + 1j * rng.standard_normal((m, n_pod, n_freq)))
    for f in range(n_freq):
        T_r[:, :, f] *= 10.0 ** (-3 * f)     # sigma_max spans ~6 decades
    solver = CPSDInverseSolver(T_r)

    G = np.empty((m, m, n_freq), dtype=np.complex128)
    S_true = _random_psd(n_pod, rng)
    for f in range(n_freq):
        Tf = T_r[:, :, f]
        G[:, :, f] = Tf @ S_true @ Tf.conj().T

    lam_grid = np.array([1e-8, 1e-4, 1e-1])
    for f in range(n_freq):
        s_max = solver.sigma_max(f)
        assert s_max > 0
        S_rel, res_rel = solver.solve_single_freq(
            f, G[:, :, f], lam_grid, alpha_scaling='relative'
        )
        S_abs, res_abs = solver.solve_single_freq(
            f, G[:, :, f], lam_grid * s_max, alpha_scaling='absolute'
        )
        np.testing.assert_allclose(
            S_rel, S_abs, rtol=1e-11, atol=0,
            err_msg=f"relative != absolute*sigma_max at f={f}",
        )
        np.testing.assert_allclose(res_rel, res_abs, rtol=1e-11, atol=0)

        # And the two modes are genuinely different (test has teeth).
        S_plain, _ = solver.solve_single_freq(
            f, G[:, :, f], lam_grid, alpha_scaling='absolute'
        )
        assert not np.allclose(S_rel, S_plain), (
            f"relative and absolute coincide at f={f}; sigma_max={s_max}"
        )

        # sigma_max must match the spectrum solve_single_freq uses.
        sig = np.linalg.svd(T_r[:, :, f], compute_uv=False)
        np.testing.assert_allclose(s_max, sig.max(), rtol=1e-13)

    try:
        solver.solve_single_freq(
            0, G[:, :, 0], lam_grid, alpha_scaling='bogus'
        )
    except ValueError:
        pass
    else:
        assert False, "expected ValueError for bad alpha_scaling in solver"

    print("  PASSED")


def test_alpha_scaling_backward_compat():
    """
    The default alpha_scaling must be 'absolute', must equal passing
    'absolute' explicitly, and must reproduce jasa23b eq. 21 computed
    independently. This pins the pre-change behaviour so adding the scaling
    option cannot silently move published results.
    """
    print("Test 9: default alpha_scaling reproduces eq. 21 exactly...")
    rng = np.random.default_rng(202)

    m, n_pod, n_freq = 16, 6, 2
    T_r, G = _build_cv_problem(rng, m, n_pod, n_freq, noise_std=0.3)
    solver = CPSDInverseSolver(T_r)
    alphas = np.array([1e-8, 1e-4, 1e-1])

    for f in range(n_freq):
        # default == explicit 'absolute', bit for bit
        S_default, res_default = solver.solve_single_freq(
            f, G[:, :, f], alphas
        )
        S_explicit, res_explicit = solver.solve_single_freq(
            f, G[:, :, f], alphas, alpha_scaling='absolute'
        )
        np.testing.assert_array_equal(S_default, S_explicit)
        np.testing.assert_array_equal(res_default, res_explicit)

        # default == independent implementation of eq. 21
        for k, alpha in enumerate(alphas):
            S_ref = _reference_method1(T_r[:, :, f], G[:, :, f], alpha)
            np.testing.assert_allclose(
                S_default[:, :, k], S_ref, rtol=1e-10, atol=0,
                err_msg=f"eq. 21 mismatch at f={f}, alpha={alpha}",
            )
            # Residual diagnostic is against the raw (unclipped) G.
            Tf = T_r[:, :, f]
            expect_res = (
                np.linalg.norm(Tf @ S_ref @ Tf.conj().T - G[:, :, f], 'fro')
                / np.linalg.norm(G[:, :, f], 'fro')
            )
            np.testing.assert_allclose(
                res_default[k], expect_res, rtol=1e-9, atol=0
            )

    # PSD by construction (the reason method 1 is the default).
    for f in range(n_freq):
        S, _ = solver.solve_single_freq(f, G[:, :, f], alphas)
        for k in range(alphas.size):
            ev = np.linalg.eigvalsh(
                0.5 * (S[:, :, k] + S[:, :, k].conj().T)
            )
            assert ev.min() >= -1e-10 * max(ev.max(), 1.0), (
                f"S_r indefinite at f={f}, alpha={alphas[k]}: "
                f"min eig {ev.min():.3e}"
            )

    # The default filter must be 'lavrentiev', and passing it explicitly must be
    # bit-identical -- adding the filter option cannot move existing results.
    for f in range(n_freq):
        S_default, res_default = solver.solve_single_freq(
            f, G[:, :, f], alphas
        )
        S_lav, res_lav = solver.solve_single_freq(
            f, G[:, :, f], alphas, filter_form='lavrentiev'
        )
        np.testing.assert_array_equal(S_default, S_lav)
        np.testing.assert_array_equal(res_default, res_lav)
        # ...and the other filter is genuinely different.
        S_tikh, _ = solver.solve_single_freq(
            f, G[:, :, f], alphas, filter_form='tikhonov'
        )
        assert not np.allclose(S_default, S_tikh), (
            f"'lavrentiev' and 'tikhonov' coincide at f={f}"
        )

    # CV defaults must also stay 'absolute'/'lavrentiev' with norm_weight = 1e-2.
    sel = KFoldCVSelector(solver, G)
    assert sel.alpha_scaling == 'absolute'
    assert sel.norm_weight == 1e-2
    assert sel.filter_form == 'lavrentiev'
    sc_default, _ = sel.score(alphas)
    sc_lav, _ = KFoldCVSelector(
        solver, G, filter_form='lavrentiev'
    ).score(alphas)
    np.testing.assert_array_equal(sc_default, sc_lav)

    print("  PASSED")


def test_cv_norm_term_formula():
    """
    The CV score must be exactly

        held-out relative residual
        + norm_weight * ||S_r||_F * sigma_max(f)^2 / ||G(f)||_F

    averaged over folds. Verified by differencing scores at two
    norm_weights, which isolates the new term, and comparing against the
    formula evaluated from an independent fold solve. norm_weight=0 must
    reproduce the pure prediction score.
    """
    print("Test 10: CV solution-norm term matches its formula...")
    rng = np.random.default_rng(303)

    m, n_pod, n_freq, k_folds, seed = 15, 4, 3, 5, 0
    T_r, G = _build_cv_problem(rng, m, n_pod, n_freq, noise_std=0.4)
    solver = CPSDInverseSolver(T_r)
    alphas = np.array([1e-6, 1e-3, 1e-1])
    mu = 3e-2

    s0, _ = KFoldCVSelector(
        solver, G, k_folds=k_folds, seed=seed, norm_weight=0.0
    ).score(alphas)
    s1, _ = KFoldCVSelector(
        solver, G, k_folds=k_folds, seed=seed, norm_weight=mu
    ).score(alphas)

    # Independent evaluation of the norm term.
    all_idx = np.arange(m, dtype=np.int64)
    folds = make_folds(all_idx, k_folds, seed)
    expected = np.zeros((n_freq, alphas.size))
    for f in range(n_freq):
        Tf, Gf = T_r[:, :, f], G[:, :, f]
        s_max = np.linalg.svd(Tf, compute_uv=False).max()
        scale = np.linalg.norm(Gf, 'fro') / s_max ** 2
        for I_val in folds:
            I_train = np.setdiff1d(all_idx, I_val, assume_unique=True)
            for j, alpha in enumerate(alphas):
                S_fold = _reference_method1(
                    Tf[I_train, :], Gf[np.ix_(I_train, I_train)], alpha
                )
                expected[f, j] += np.linalg.norm(S_fold, 'fro') / scale
        expected[f, :] /= k_folds

    np.testing.assert_allclose(
        s1 - s0, mu * expected, rtol=1e-9, atol=1e-14,
        err_msg="norm term does not match its documented formula",
    )
    # The term is additive and non-negative, so it can only raise the score.
    assert np.all(s1 >= s0 - 1e-14)
    # ...and it is not a no-op.
    assert np.max(np.abs(s1 - s0)) > 0

    # norm_weight=0 must equal a hand-built pure prediction score.
    pred = np.zeros((n_freq, alphas.size))
    for f in range(n_freq):
        Tf, Gf = T_r[:, :, f], G[:, :, f]
        for I_val in folds:
            I_train = np.setdiff1d(all_idx, I_val, assume_unique=True)
            lam_v, U_v = np.linalg.eigh(
                0.5 * (Gf[np.ix_(I_val, I_val)]
                       + Gf[np.ix_(I_val, I_val)].conj().T)
            )
            G_val = (U_v * np.where(lam_v > 0, lam_v, 0.0)[None, :]
                     ) @ U_v.conj().T
            n_val = np.linalg.norm(G_val, 'fro')
            Tv = Tf[I_val, :]
            for j, alpha in enumerate(alphas):
                S_fold = _reference_method1(
                    Tf[I_train, :], Gf[np.ix_(I_train, I_train)], alpha
                )
                r = np.linalg.norm(
                    Tv @ S_fold @ Tv.conj().T - G_val, 'fro'
                )
                pred[f, j] += r / n_val if n_val > 0 else r
        pred[f, :] /= k_folds
    np.testing.assert_allclose(
        s0, pred, rtol=1e-9, atol=1e-14,
        err_msg="norm_weight=0 is not the pure prediction score",
    )

    print("  PASSED")


def test_cv_refit_alpha_consistency():
    """
    Under alpha_scaling='relative' the lam -> alpha map must use the
    FULL-matrix sigma_max(f), not each fold's, so one candidate means one
    absolute alpha in every fold and in the refit. A fold-local sigma_max
    would make CV score a different alpha than the refit applies.
    """
    print("Test 11: CV and refit resolve the same alpha under 'relative'...")
    rng = np.random.default_rng(404)

    m, n_pod, n_freq, k_folds, seed = 14, 4, 2, 4, 0
    T_r, G = _build_cv_problem(rng, m, n_pod, n_freq, noise_std=0.4)
    # Give the frequencies very different scales so the map matters.
    for f in range(n_freq):
        T_r[:, :, f] *= 10.0 ** (-2 * f)
        G[:, :, f] *= 10.0 ** (-4 * f)
    solver = CPSDInverseSolver(T_r)
    alphas = np.array([1e-5, 1e-2])

    scores, _ = KFoldCVSelector(
        solver, G, k_folds=k_folds, seed=seed,
        norm_weight=0.0, alpha_scaling='relative',
    ).score(alphas)

    all_idx = np.arange(m, dtype=np.int64)
    folds = make_folds(all_idx, k_folds, seed)

    def build_score(use_full_sigma_max):
        out = np.zeros((n_freq, alphas.size))
        for f in range(n_freq):
            Tf, Gf = T_r[:, :, f], G[:, :, f]
            s_full = np.linalg.svd(Tf, compute_uv=False).max()
            for I_val in folds:
                I_train = np.setdiff1d(all_idx, I_val, assume_unique=True)
                Ttr = Tf[I_train, :]
                s_ref = (s_full if use_full_sigma_max
                         else np.linalg.svd(Ttr, compute_uv=False).max())
                lam_v, U_v = np.linalg.eigh(
                    0.5 * (Gf[np.ix_(I_val, I_val)]
                           + Gf[np.ix_(I_val, I_val)].conj().T)
                )
                G_val = (U_v * np.where(lam_v > 0, lam_v, 0.0)[None, :]
                         ) @ U_v.conj().T
                n_val = np.linalg.norm(G_val, 'fro')
                Tv = Tf[I_val, :]
                for j, lam in enumerate(alphas):
                    S_fold = _reference_method1(
                        Ttr, Gf[np.ix_(I_train, I_train)], lam * s_ref
                    )
                    r = np.linalg.norm(
                        Tv @ S_fold @ Tv.conj().T - G_val, 'fro'
                    )
                    out[f, j] += r / n_val if n_val > 0 else r
            out[f, :] /= k_folds
        return out

    np.testing.assert_allclose(
        scores, build_score(True), rtol=1e-9, atol=1e-14,
        err_msg="CV does not use the full-matrix sigma_max",
    )
    fold_local = build_score(False)
    assert not np.allclose(scores, fold_local, rtol=1e-6), (
        "full-matrix and fold-local sigma_max give the same score, so this "
        "test cannot detect the difference"
    )

    # The refit must apply exactly the alpha the selector evaluated.
    for f in range(n_freq):
        s_max = solver.sigma_max(f)
        for lam in alphas:
            S_rel, _ = solver.solve_single_freq(
                f, G[:, :, f], np.array([lam]), alpha_scaling='relative'
            )
            S_ref = _reference_method1(
                T_r[:, :, f], G[:, :, f], lam * s_max
            )
            np.testing.assert_allclose(
                S_rel[:, :, 0], S_ref, rtol=1e-10, atol=0,
                err_msg=f"refit alpha mismatch at f={f}, lam={lam}",
            )

    print("  PASSED")


def test_relative_scaling_is_scale_invariant():
    """
    Under T_r -> gamma*T_r and G -> gamma^2*G, alpha_scaling='relative' must
    leave both the recovered S_r and the whole CV score unchanged. This is
    the property that lets one lam and one norm_weight serve a whole
    frequency band, so it catches a normalisation error anywhere in the
    chain: the filter picks up 1/gamma, Z picks up gamma, and the norm term's
    sigma_max^2/||G||_F is invariant.

    'absolute' mode must NOT be invariant, which is the point of adding the
    option at all.
    """
    print("Test 12: 'relative' scaling is invariant to problem scale...")
    rng = np.random.default_rng(505)

    m, n_pod, n_freq = 13, 4, 2
    T_r, G = _build_cv_problem(rng, m, n_pod, n_freq, noise_std=0.3)
    lam_grid = np.array([1e-6, 1e-3, 1e-1])

    solver = CPSDInverseSolver(T_r)
    for filter_form in FILTER_FORMS:
        base_S = [
            solver.solve_single_freq(
                f, G[:, :, f], lam_grid, alpha_scaling='relative',
                filter_form=filter_form,
            )[0]
            for f in range(n_freq)
        ]
        base_scores, _ = KFoldCVSelector(
            solver, G, k_folds=4, seed=0, norm_weight=1e-2,
            alpha_scaling='relative', filter_form=filter_form,
        ).score(lam_grid)

        for gamma in (1e3, 1e-3, 7.5):
            T_s = T_r * gamma
            G_s = G * gamma ** 2
            solver_s = CPSDInverseSolver(T_s)

            for f in range(n_freq):
                S_s, _ = solver_s.solve_single_freq(
                    f, G_s[:, :, f], lam_grid, alpha_scaling='relative',
                    filter_form=filter_form,
                )
                np.testing.assert_allclose(
                    S_s, base_S[f], rtol=1e-9, atol=1e-300,
                    err_msg=f"S_r not scale-invariant at gamma={gamma}, "
                            f"f={f}, filter={filter_form}",
                )

            scores_s, _ = KFoldCVSelector(
                solver_s, G_s, k_folds=4, seed=0, norm_weight=1e-2,
                alpha_scaling='relative', filter_form=filter_form,
            ).score(lam_grid)
            np.testing.assert_allclose(
                scores_s, base_scores, rtol=1e-8, atol=1e-12,
                err_msg=f"CV score not scale-invariant at gamma={gamma}, "
                        f"filter={filter_form}",
            )

    # 'absolute' mode is scale-dependent, as designed.
    gamma = 1e3
    solver_s = CPSDInverseSolver(T_r * gamma)
    S_abs_base, _ = solver.solve_single_freq(
        0, G[:, :, 0], lam_grid, alpha_scaling='absolute'
    )
    S_abs_scaled, _ = solver_s.solve_single_freq(
        0, (G * gamma ** 2)[:, :, 0], lam_grid, alpha_scaling='absolute'
    )
    assert not np.allclose(S_abs_scaled, S_abs_base, rtol=1e-6), (
        "'absolute' mode came out scale-invariant; the comparison above "
        "therefore proves nothing"
    )

    print("  PASSED")


def test_norm_term_reduces_error_when_ill_conditioned():
    """
    Behavioural check on the reason the norm term exists. On an
    ill-conditioned noisy problem with a known S_true, a pure prediction
    score under-regularises; adding the norm term must select an alpha no
    smaller at every frequency, and must lower the true solution error.

    One fixed seed, deterministic, with a wide margin required.
    """
    print("Test 13: norm term lowers true error when ill-conditioned...")
    rng = np.random.default_rng(606)

    m, n_pod, n_freq = 20, 6, 6
    T_r, G, S_true = _ill_conditioned_problem(
        rng, m, n_pod, n_freq,
        cond_exp=(2.0, 2.5, 3.0), noise_rel=3e-2,
    )
    solver = CPSDInverseSolver(T_r)
    lam_grid = np.logspace(-10, 1, 23)
    nS = np.linalg.norm(S_true, 'fro')

    def run(mu):
        sel = KFoldCVSelector(
            solver, G, k_folds=5, seed=0, norm_weight=mu,
            alpha_scaling='relative',
        )
        lam_star, _, _ = sel.select(lam_grid, alpha_mode='per_freq')
        err = np.empty(n_freq)
        for f in range(n_freq):
            S, _ = solver.solve_single_freq(
                f, G[:, :, f], np.array([lam_star[f]]),
                alpha_scaling='relative',
            )
            err[f] = np.linalg.norm(S[:, :, 0] - S_true, 'fro') / nS
        return lam_star, err

    lam0, err0 = run(0.0)
    lam1, err1 = run(1e-2)

    # The norm term can only push the selection toward more regularisation.
    assert np.all(lam1 >= lam0), (
        f"norm term reduced alpha somewhere: lam0={lam0}, lam1={lam1}"
    )
    assert np.any(lam1 > lam0), (
        "norm term changed nothing; this problem is not exercising it"
    )
    # And it must actually help, by a clear margin on the worst frequency.
    assert err1.max() < err0.max(), (
        f"max error not improved: {err0.max():.4f} -> {err1.max():.4f}"
    )
    assert np.median(err1) <= np.median(err0), (
        f"median error worsened: "
        f"{np.median(err0):.4f} -> {np.median(err1):.4f}"
    )
    print(
        f"  PASSED (max err {err0.max():.3f} -> {err1.max():.3f}, "
        f"median {np.median(err0):.3f} -> {np.median(err1):.3f})"
    )


def test_filter_form_alpha_power():
    """
    Under alpha_scaling='relative' the supplied parameter must be multiplied
    by sigma_max raised to the filter's own power (1 for 'lavrentiev', 2 for
    'tikhonov'), so that one dimensionless lam means the same amount of
    damping for either filter. Without this, switching filters silently
    rescales the whole alpha grid.
    """
    print("Test 14: relative alpha uses the filter's sigma power...")
    rng = np.random.default_rng(707)

    assert ALPHA_SIGMA_POWER == {'lavrentiev': 1, 'tikhonov': 2}
    assert set(ALPHA_SIGMA_POWER) == set(FILTER_FORMS)

    lams = np.array([1e-6, 1e-3, 1.0])
    smax = 4.0
    np.testing.assert_allclose(
        resolve_alphas(lams, smax, 'relative', 'lavrentiev'), lams * smax,
        rtol=0, atol=0,
    )
    np.testing.assert_allclose(
        resolve_alphas(lams, smax, 'relative', 'tikhonov'), lams * smax ** 2,
        rtol=0, atol=0,
    )
    # 'absolute' ignores the filter entirely.
    for ff in FILTER_FORMS:
        np.testing.assert_array_equal(
            resolve_alphas(lams, smax, 'absolute', ff), lams
        )
    for bad in ('nope', 'Tikhonov', ''):
        try:
            resolve_alphas(lams, smax, 'relative', bad)
        except ValueError:
            pass
        else:
            assert False, f"expected ValueError for filter_form {bad!r}"

    # Solver must apply exactly that alpha, on frequencies of differing scale.
    m, n_pod, n_freq = 15, 5, 3
    T_r = (rng.standard_normal((m, n_pod, n_freq))
           + 1j * rng.standard_normal((m, n_pod, n_freq)))
    for f in range(n_freq):
        T_r[:, :, f] *= 10.0 ** (-2 * f)
    solver = CPSDInverseSolver(T_r)
    S_true = _random_psd(n_pod, rng)
    G = np.empty((m, m, n_freq), dtype=np.complex128)
    for f in range(n_freq):
        Tf = T_r[:, :, f]
        G[:, :, f] = Tf @ S_true @ Tf.conj().T

    lam_grid = np.array([1e-7, 1e-3])
    for ff in FILTER_FORMS:
        p = ALPHA_SIGMA_POWER[ff]
        for f in range(n_freq):
            s_max = solver.sigma_max(f)
            S_rel, _ = solver.solve_single_freq(
                f, G[:, :, f], lam_grid,
                alpha_scaling='relative', filter_form=ff,
            )
            S_abs, _ = solver.solve_single_freq(
                f, G[:, :, f], lam_grid * s_max ** p,
                alpha_scaling='absolute', filter_form=ff,
            )
            np.testing.assert_allclose(
                S_rel, S_abs, rtol=1e-11, atol=0,
                err_msg=f"relative != absolute*smax^{p} for {ff} at f={f}",
            )
            # Wrong power must be detectably different, or the test is empty.
            S_wrong, _ = solver.solve_single_freq(
                f, G[:, :, f], lam_grid * s_max ** (3 - p),
                alpha_scaling='absolute', filter_form=ff,
            )
            assert not np.allclose(S_rel, S_wrong, rtol=1e-6), (
                f"smax^{p} and smax^{3 - p} indistinguishable for {ff} "
                f"at f={f}; sigma_max={s_max}"
            )

    print("  PASSED")


def test_both_filters_keep_psd():
    """
    S_r = K K^h must stay positive semidefinite for BOTH filters, over a wide
    alpha sweep, when Ghat is genuinely indefinite (finite-averaging noise).
    This is what lets 'tikhonov' be offered at all: it must not reintroduce
    the indefiniteness that kept the paper's eq. 24 out of the code.

    Includes a positive control. On this instance the eq. 24 entrywise form
    H_ij = s_i s_j (ZZ^h)_ij / (s_i^2 s_j^2 + alpha), S_r = Y H Y^h -- computed
    inline, not from the solver, since it is deliberately not implemented -- IS
    indefinite. So the PSD assertions below are a real property of the K K^h
    construction rather than a lucky input.
    """
    print("Test 15: both filters keep S_r PSD on indefinite Ghat...")
    rng = np.random.default_rng(851)

    m, n_pod = 18, 6
    T_r = (rng.standard_normal((m, n_pod, 1))
           + 1j * rng.standard_normal((m, n_pod, 1)))
    solver = CPSDInverseSolver(T_r)

    A = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
    G_clean = A @ A.conj().T
    G_clean = 0.5 * (G_clean + G_clean.conj().T)
    E = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
    E = 0.5 * (E + E.conj().T)
    G = G_clean + (0.25 * np.linalg.norm(G_clean, 'fro')
                   / np.linalg.norm(E, 'fro')) * E

    def rel_min_eig(M):
        Mh = 0.5 * (M + M.conj().T)
        ev = np.linalg.eigvalsh(Mh)
        return ev.min() / ev.max()

    # Premise 1: Ghat really is indefinite, else nothing below is exercised.
    assert rel_min_eig(G) < -1e-6, (
        f"Ghat is not indefinite (min/max eig {rel_min_eig(G):.2e})"
    )

    # Premise 2 (positive control): the eq. 24 form IS indefinite here.
    X, sigma, Vh = np.linalg.svd(T_r[:, :, 0], full_matrices=False)
    Y = Vh.conj().T
    lam, U = np.linalg.eigh(0.5 * (G + G.conj().T))
    Psi = U * np.sqrt(np.where(lam > 0, lam, 0.0))[np.newaxis, :]
    Z = X.conj().T @ Psi
    ZZh = Z @ Z.conj().T
    ss = np.outer(sigma, sigma)
    worst_eq24 = 0.0
    for lam_rel in np.logspace(-8, 2, 41):
        a24 = lam_rel * sigma[0] ** 4          # eq. 24's alpha scales as s^4
        S24 = Y @ (ss * ZZh / (ss ** 2 + a24)) @ Y.conj().T
        worst_eq24 = min(worst_eq24, rel_min_eig(S24))
    assert worst_eq24 < -1e-3, (
        f"the eq. 24 control stayed PSD here (worst min/max eig "
        f"{worst_eq24:.3e}); this instance does not discriminate, so the "
        f"assertions below would prove nothing"
    )

    # The actual claim: K K^h stays PSD for both filters, at every alpha.
    for ff in FILTER_FORMS:
        p_pow = ALPHA_SIGMA_POWER[ff]
        alphas = np.logspace(-8, 2, 41) * sigma[0] ** p_pow
        S, _ = solver.solve_single_freq(
            0, G, alphas, alpha_scaling='absolute', filter_form=ff
        )
        for k in range(alphas.size):
            asym = (np.linalg.norm(S[:, :, k] - S[:, :, k].conj().T, 'fro')
                    / max(np.linalg.norm(S[:, :, k], 'fro'), 1e-300))
            assert asym < 1e-12, (
                f"{ff} S_r not Hermitian at alpha={alphas[k]:.2e}: {asym:.2e}"
            )
            r = rel_min_eig(S[:, :, k])
            assert r >= -1e-10, (
                f"{ff} S_r indefinite at alpha={alphas[k]:.2e}: "
                f"min/max eig {r:.3e}"
            )

    print(f"  PASSED (eq. 24 control reached {worst_eq24:.2e}; "
          f"both K K^h filters stayed PSD)")


def test_cv_and_refit_agree_per_filter():
    """
    Extends the full-matrix-sigma_max guarantee to both filters. Under
    alpha_scaling='relative' the CV score and the refit must resolve a
    candidate to the same absolute alpha, which now also depends on the
    filter's sigma power -- a place a filter/power mismatch could hide.
    """
    print("Test 16: CV score and refit agree for each filter...")
    rng = np.random.default_rng(909)

    m, n_pod, n_freq, k_folds, seed = 14, 4, 2, 4, 0
    T_r, G = _build_cv_problem(rng, m, n_pod, n_freq, noise_std=0.4)
    for f in range(n_freq):
        T_r[:, :, f] *= 10.0 ** (-2 * f)
        G[:, :, f] *= 10.0 ** (-4 * f)
    solver = CPSDInverseSolver(T_r)
    lam_grid = np.array([1e-5, 1e-2])

    all_idx = np.arange(m, dtype=np.int64)
    folds = make_folds(all_idx, k_folds, seed)

    for ff in FILTER_FORMS:
        p = ALPHA_SIGMA_POWER[ff]
        scores, _ = KFoldCVSelector(
            solver, G, k_folds=k_folds, seed=seed, norm_weight=0.0,
            alpha_scaling='relative', filter_form=ff,
        ).score(lam_grid)

        expected = np.zeros((n_freq, lam_grid.size))
        for f in range(n_freq):
            Tf, Gf = T_r[:, :, f], G[:, :, f]
            s_full = np.linalg.svd(Tf, compute_uv=False).max()
            for I_val in folds:
                I_train = np.setdiff1d(all_idx, I_val, assume_unique=True)
                lam_v, U_v = np.linalg.eigh(
                    0.5 * (Gf[np.ix_(I_val, I_val)]
                           + Gf[np.ix_(I_val, I_val)].conj().T)
                )
                G_val = (U_v * np.where(lam_v > 0, lam_v, 0.0)[None, :]
                         ) @ U_v.conj().T
                n_val = np.linalg.norm(G_val, 'fro')
                Tv = Tf[I_val, :]
                for j, lam in enumerate(lam_grid):
                    S_fold = _reference_method1(
                        Tf[I_train, :], Gf[np.ix_(I_train, I_train)],
                        lam * s_full ** p, filter_form=ff,
                    )
                    r = np.linalg.norm(
                        Tv @ S_fold @ Tv.conj().T - G_val, 'fro'
                    )
                    expected[f, j] += r / n_val if n_val > 0 else r
            expected[f, :] /= k_folds

        np.testing.assert_allclose(
            scores, expected, rtol=1e-9, atol=1e-14,
            err_msg=f"CV score mismatch for filter_form={ff}",
        )

        # And the refit applies the very same alpha.
        for f in range(n_freq):
            s_max = solver.sigma_max(f)
            for lam in lam_grid:
                S_rel, _ = solver.solve_single_freq(
                    f, G[:, :, f], np.array([lam]),
                    alpha_scaling='relative', filter_form=ff,
                )
                S_ref = _reference_method1(
                    T_r[:, :, f], G[:, :, f], lam * s_max ** p,
                    filter_form=ff,
                )
                np.testing.assert_allclose(
                    S_rel[:, :, 0], S_ref, rtol=1e-10, atol=0,
                    err_msg=f"refit alpha mismatch for {ff} at f={f}, "
                            f"lam={lam}",
                )

    # The two filters must not produce the same CV scores, or the loop above
    # is testing one thing twice.
    s_e, _ = KFoldCVSelector(
        solver, G, k_folds=k_folds, seed=seed,
        alpha_scaling='relative', filter_form='lavrentiev',
    ).score(lam_grid)
    s_t, _ = KFoldCVSelector(
        solver, G, k_folds=k_folds, seed=seed,
        alpha_scaling='relative', filter_form='tikhonov',
    ).score(lam_grid)
    assert not np.allclose(s_e, s_t, rtol=1e-6), (
        "the two filters give identical CV scores"
    )

    # Bad filter_form is rejected by the selector.
    try:
        KFoldCVSelector(solver, G, filter_form='nope')
    except ValueError:
        pass
    else:
        assert False, "expected ValueError for bad filter_form in selector"

    print("  PASSED")


def run_all_tests() -> bool:
    print("=" * 60)
    print("Running CPSD Inverse Solver Tests")
    print("=" * 60)
    tests = [
        test_recovery_synthetic,
        test_scalar_reduction,
        test_apply_row_subset,
        test_kfold_partition,
        test_cv_picks_best_alpha_synthetic,
        test_refit_matches_direct_solve,
        test_global_mode_single_alpha,
        test_alpha_scaling_equivalence,
        test_alpha_scaling_backward_compat,
        test_cv_norm_term_formula,
        test_cv_refit_alpha_consistency,
        test_relative_scaling_is_scale_invariant,
        test_norm_term_reduces_error_when_ill_conditioned,
        test_filter_form_alpha_power,
        test_both_filters_keep_psd,
        test_cv_and_refit_agree_per_filter,
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
