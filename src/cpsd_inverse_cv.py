"""
K-fold cross-validation for selecting the regularization parameter in the
CPSD inverse problem.

Operates on the (T_r, G) pair the solver was built with -- i.e., already
restricted to the user's row-index downselect if one was applied
upstream. Folds split the full sensor axis of ``solver.T_r`` and ``G``
into ``k_folds`` blocks after a seeded random shuffle.

For each frequency f, fold k, and candidate alpha:

  I_train = I \\ I_fold_k,  I_val = I_fold_k
  S_r  = closed-form K K^h (jasa23b eq. 21 form, filter per ``filter_form``)
         on (T_r[I_train,:,f], G[I_train, I_train, f])
  Ghat = T_r[I_val,:,f] S_r T_r[I_val,:,f]^h

  score(f, alpha, k) = ||Ghat - PSD_clip(G[I_val,I_val,f])||_F
                       / ||PSD_clip(G[I_val,I_val,f])||_F
                     + norm_weight * ||S_r||_F * sigma_max(f)^2 / ||G(f)||_F

Both training and validation blocks are Hermitized and PSD-clipped with
the same ``psd_tol_rel`` for consistency.

Why the second term
-------------------
A pure held-out prediction score is a data-fit criterion, and at
ill-conditioned frequencies it is nearly blind to the directions the
ill-conditioning amplifies: many very different S_r reproduce almost the same
sensor CPSD. Empirically it then under-regularizes by about a decade exactly
where cond(T_r) is worst -- near resonances -- because the CPSD forward map
S -> T_r S T_r^h has condition number cond(T_r)^2. Adding an explicit
solution-norm term restores the information the prediction score lacks.

The term must be added to the *held-out* residual, never to the training
residual. S_r(alpha) already minimizes (training fit) + alpha * (penalty), so
minimizing (training fit) + mu * (penalty) over the family {S_r(alpha)} returns
alpha = mu identically -- a circular criterion that just echoes the constant
supplied. Scoring the fit on sensors the fold did not see breaks that fixed
point. (This is also why an L-curve takes the corner of
(log||S_r||, log||residual||) rather than a weighted sum: curvature is
invariant to the weighting.)

The ``sigma_max(f)^2 / ||G(f)||_F`` factor makes the term dimensionless --
||S_r|| scales like ||G|| / sigma_max^2 -- so one ``norm_weight`` applies
across a whole frequency band.
"""

from typing import List, Optional, Tuple

import numpy as np

from cpsd_inverse import (
    ALPHA_SCALINGS,
    FILTER_FORMS,
    CPSDInverseSolver,
    resolve_alphas,
    spectral_filter,
)


def make_folds(
    indices: np.ndarray, k_folds: int, seed: int
) -> List[np.ndarray]:
    """
    Shuffle ``indices`` with ``numpy.default_rng(seed)`` and split into
    ``k_folds`` contiguous blocks.

    The concatenation of the returned arrays is a permutation of
    ``indices``; each input index appears exactly once.
    """
    idx = np.asarray(indices, dtype=np.int64).copy()
    if idx.ndim != 1:
        raise ValueError(f"indices must be 1D, got shape {idx.shape}")
    if not isinstance(k_folds, int) or k_folds < 2:
        raise ValueError(f"k_folds must be an integer >= 2, got {k_folds}")
    if idx.size < k_folds:
        raise ValueError(
            f"need at least k_folds={k_folds} indices to form folds, "
            f"got {idx.size}"
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    return [arr.copy() for arr in np.array_split(idx, k_folds)]


def _hermitize_clip(G: np.ndarray, tol_rel: float) -> np.ndarray:
    """Hermitian PSD projection of G, returned as a full matrix."""
    Gh = 0.5 * (G + G.conj().T)
    lam, U = np.linalg.eigh(Gh)
    if tol_rel > 0:
        cutoff = tol_rel * np.max(np.abs(lam))
        lam = np.where(lam > cutoff, lam, 0.0)
    else:
        lam = np.where(lam > 0, lam, 0.0)
    return (U * lam[np.newaxis, :]) @ U.conj().T


def _solve_for_alphas(
    T_r_train: np.ndarray,
    G_train: np.ndarray,
    alphas: np.ndarray,
    psd_tol_rel: float,
    filter_form: str = 'lavrentiev',
) -> np.ndarray:
    """
    Per-frequency closed-form S_r for a set of alphas on a training subset.

    Mirrors CPSDInverseSolver.solve_single_freq but skips the
    training-data residual computation since CV scores on held-out data.

    ``alphas`` here are *effective* alphas -- already passed through
    :func:`resolve_alphas` by the caller using the full-matrix sigma_max, so
    that a given candidate maps to the same absolute alpha in every fold and
    in the final refit.
    """
    X, sigma, Vh = np.linalg.svd(T_r_train, full_matrices=False)
    Y = Vh.conj().T

    Gh = 0.5 * (G_train + G_train.conj().T)
    lam, U = np.linalg.eigh(Gh)
    if psd_tol_rel > 0:
        cutoff = psd_tol_rel * np.max(np.abs(lam))
        lam = np.where(lam > cutoff, lam, 0.0)
    else:
        lam = np.where(lam > 0, lam, 0.0)
    psi = U * np.sqrt(lam)[np.newaxis, :]
    Z = X.conj().T @ psi

    n_pod = Y.shape[0]
    S_r_out = np.empty((n_pod, n_pod, alphas.size), dtype=np.complex128)
    for k, alpha in enumerate(alphas):
        g = spectral_filter(sigma, alpha, filter_form)
        K = Y @ (g[:, None] * Z)
        S_r_out[:, :, k] = K @ K.conj().T
    return S_r_out


class KFoldCVSelector:
    """
    K-fold cross-validation for selecting the Tikhonov regularization
    parameter for a pre-built :class:`CPSDInverseSolver`.

    Parameters
    ----------
    solver : CPSDInverseSolver
        Built with the already-downselected T_r.
    G : ndarray, shape (n_sensors, n_sensors, n_freq), complex
        Already-downselected sensor CPSD, aligned with ``solver.T_r``.
    k_folds : int
        Number of folds; default 5. Must be >= 2 and <= solver.n_sensors.
    seed : int
        Seed for the fold shuffle; default 0. Same seed gives the same
        partition.
    save_fold_scores : bool
        If True, :meth:`score` and :meth:`select` also return the per-fold
        score array of shape (n_freq, n_alpha, k_folds). Default False
        (saves memory).
    norm_weight : float
        Dimensionless weight mu on the solution-norm term of the score (see
        the module docstring). ``0.0`` reproduces the pure held-out prediction
        score. Default 1e-2; useful values run roughly 1e-3 to 1e-1, above
        which the selection over-regularizes.
    alpha_scaling : {'absolute', 'relative'}
        How to interpret the candidate parameters passed to :meth:`score` and
        :meth:`select`; see :func:`cpsd_inverse.resolve_alphas`. Must match
        what is later handed to
        :meth:`CPSDInverseSolver.solve_single_freq` for the refit.
    filter_form : {'lavrentiev', 'tikhonov'}
        Spectral filter; see :func:`cpsd_inverse.spectral_filter`. Must also
        match the refit, since it sets both the filter and (under
        ``alpha_scaling='relative'``) the power of sigma_max that alpha
        scales with.
    """

    def __init__(
        self,
        solver: CPSDInverseSolver,
        G: np.ndarray,
        k_folds: int = 5,
        seed: int = 0,
        save_fold_scores: bool = False,
        norm_weight: float = 1e-2,
        alpha_scaling: str = 'absolute',
        filter_form: str = 'lavrentiev',
    ):
        if not isinstance(solver, CPSDInverseSolver):
            raise TypeError("solver must be a CPSDInverseSolver instance")
        G_arr = np.asarray(G, dtype=np.complex128)
        if (
            G_arr.ndim != 3
            or G_arr.shape[:2] != (solver.n_sensors, solver.n_sensors)
            or G_arr.shape[2] != solver.n_freq
        ):
            raise ValueError(
                f"G must have shape ({solver.n_sensors}, "
                f"{solver.n_sensors}, {solver.n_freq}) to match solver, "
                f"got {G_arr.shape}"
            )
        if not isinstance(k_folds, int) or k_folds < 2:
            raise ValueError(
                f"k_folds must be an integer >= 2, got {k_folds}"
            )
        if solver.n_sensors < k_folds:
            raise ValueError(
                f"need at least k_folds={k_folds} sensor rows, got "
                f"{solver.n_sensors}; reduce cv.k_folds or extend the "
                f"row-index subset"
            )
        if not isinstance(seed, int):
            raise ValueError(f"seed must be an int, got {seed!r}")
        if not isinstance(save_fold_scores, bool):
            raise ValueError(
                f"save_fold_scores must be a bool, got {save_fold_scores!r}"
            )
        if (isinstance(norm_weight, bool)
                or not isinstance(norm_weight, (int, float))
                or not np.isfinite(norm_weight)
                or norm_weight < 0):
            raise ValueError(
                f"norm_weight must be a finite non-negative number, "
                f"got {norm_weight!r}"
            )
        if alpha_scaling not in ALPHA_SCALINGS:
            raise ValueError(
                f"alpha_scaling must be one of {ALPHA_SCALINGS}, "
                f"got {alpha_scaling!r}"
            )
        if filter_form not in FILTER_FORMS:
            raise ValueError(
                f"filter_form must be one of {FILTER_FORMS}, "
                f"got {filter_form!r}"
            )

        self.solver = solver
        self.G = G_arr
        self.k_folds = k_folds
        self.seed = seed
        self.save_fold_scores = save_fold_scores
        self.norm_weight = float(norm_weight)
        self.alpha_scaling = alpha_scaling
        self.filter_form = filter_form

    def score(
        self, alphas: np.ndarray, psd_tol_rel: float = 0.0
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Mean-over-folds CV score per frequency and per alpha.

        The score is the held-out relative prediction residual plus
        ``norm_weight`` times the dimensionless solution norm; see the module
        docstring for the formula and for why the norm term belongs on the
        held-out residual rather than the training one.

        Parameters
        ----------
        alphas : array_like, shape (n_alpha,)
            Candidate regularization values, non-negative. Interpreted per
            ``self.alpha_scaling``.
        psd_tol_rel : float
            Relative threshold for clipping eigenvalues in the PSD
            projection of both training and validation blocks.

        Returns
        -------
        scores : ndarray, shape (n_freq, n_alpha)
        fold_scores : ndarray or None
            Shape (n_freq, n_alpha, k_folds) when ``save_fold_scores`` is
            True; otherwise ``None``.
        """
        alphas = np.atleast_1d(np.asarray(alphas, dtype=np.float64))
        if np.any(alphas < 0):
            raise ValueError("alphas must be non-negative")
        if not isinstance(psd_tol_rel, (int, float)) or psd_tol_rel < 0:
            raise ValueError(
                f"psd_tol_rel must be a non-negative number, "
                f"got {psd_tol_rel}"
            )

        n_sensors = self.solver.n_sensors
        n_freq = self.solver.n_freq
        all_idx = np.arange(n_sensors, dtype=np.int64)
        folds = make_folds(all_idx, self.k_folds, self.seed)

        scores = np.zeros((n_freq, alphas.size), dtype=np.float64)
        fold_scores = (
            np.zeros(
                (n_freq, alphas.size, self.k_folds), dtype=np.float64
            )
            if self.save_fold_scores else None
        )

        for f in range(n_freq):
            T_r_f = self.solver.T_r[:, :, f]
            G_f = self.G[:, :, f]

            # Full-matrix spectrum: fixes the lam -> alpha map for this
            # frequency so every fold and the later refit share one alpha.
            sigma_max_f = float(
                np.linalg.svd(T_r_f, compute_uv=False).max()
            )
            alphas_eff = resolve_alphas(
                alphas, sigma_max_f, self.alpha_scaling, self.filter_form
            )

            # Dimensionless scale for ||S_r||_F, since ||S_r|| ~ ||G||/smax^2.
            norm_scale = 0.0
            if self.norm_weight > 0:
                G_f_fro = np.linalg.norm(G_f, 'fro')
                if G_f_fro > 0 and sigma_max_f > 0:
                    norm_scale = G_f_fro / sigma_max_f ** 2

            for k_idx, I_val in enumerate(folds):
                I_train = np.setdiff1d(
                    all_idx, I_val, assume_unique=True
                )

                G_train = G_f[np.ix_(I_train, I_train)]
                G_val_clipped = _hermitize_clip(
                    G_f[np.ix_(I_val, I_val)], psd_tol_rel
                )

                S_r_alphas = _solve_for_alphas(
                    T_r_f[I_train, :], G_train, alphas_eff, psd_tol_rel,
                    self.filter_form,
                )

                T_r_val = T_r_f[I_val, :]
                G_val_fro = np.linalg.norm(G_val_clipped, 'fro')

                for a_idx in range(alphas.size):
                    S_r = S_r_alphas[:, :, a_idx]
                    G_pred = T_r_val @ S_r @ T_r_val.conj().T
                    res_abs = np.linalg.norm(
                        G_pred - G_val_clipped, 'fro'
                    )
                    s = res_abs / G_val_fro if G_val_fro > 0 else res_abs
                    # Solution-norm term on the HELD-OUT score (see module
                    # docstring: adding it to the training fit is circular).
                    if norm_scale > 0:
                        s += self.norm_weight * (
                            np.linalg.norm(S_r, 'fro') / norm_scale
                        )
                    scores[f, a_idx] += s
                    if fold_scores is not None:
                        fold_scores[f, a_idx, k_idx] = s

        scores /= self.k_folds
        return scores, fold_scores

    def select(
        self,
        alphas: np.ndarray,
        psd_tol_rel: float = 0.0,
        alpha_mode: str = 'global',
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Run CV and return the selected alpha(s).

        Parameters
        ----------
        alphas : array_like, shape (n_alpha,)
            Candidate regularization values.
        psd_tol_rel : float
        alpha_mode : {'global', 'per_freq'}
            ``'global'`` aggregates the per-frequency CV scores via a mean
            and picks a single scalar alpha. ``'per_freq'`` picks the
            argmin per frequency independently.

        Returns
        -------
        alpha_star : ndarray
            Shape ``(n_freq,)`` for ``'per_freq'``; shape ``(1,)`` for
            ``'global'``.
        scores : ndarray, shape (n_freq, n_alpha)
        fold_scores : ndarray or None
        """
        if alpha_mode not in ('per_freq', 'global'):
            raise ValueError(
                f"alpha_mode must be 'per_freq' or 'global', "
                f"got {alpha_mode!r}"
            )
        alphas = np.atleast_1d(np.asarray(alphas, dtype=np.float64))
        scores, fold_scores = self.score(alphas, psd_tol_rel=psd_tol_rel)

        if alpha_mode == 'per_freq':
            best_idx = np.argmin(scores, axis=1)
            alpha_star = alphas[best_idx]
        else:
            global_scores = scores.mean(axis=0)
            best_idx = int(np.argmin(global_scores))
            alpha_star = np.array([alphas[best_idx]], dtype=np.float64)

        return alpha_star, scores, fold_scores
