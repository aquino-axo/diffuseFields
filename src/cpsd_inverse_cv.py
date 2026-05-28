"""
K-fold cross-validation for selecting the Tikhonov regularization
parameter in the CPSD inverse problem.

Operates on the (T_r, G) pair the solver was built with -- i.e., already
restricted to the user's row-index downselect if one was applied
upstream. Folds split the full sensor axis of ``solver.T_r`` and ``G``
into ``k_folds`` blocks after a seeded random shuffle.

For each frequency f, fold k, and candidate alpha:

  I_train = I \\ I_fold_k,  I_val = I_fold_k
  S_r  = closed-form (eqs. 35-36) on (T_r[I_train,:,f], G[I_train, I_train, f])
  Ghat = T_r[I_val,:,f] S_r T_r[I_val,:,f]^h
  score(f, alpha, k) = ||Ghat - PSD_clip(G[I_val, I_val, f])||_F
                       / ||PSD_clip(G[I_val, I_val, f])||_F

Both training and validation blocks are Hermitized and PSD-clipped with
the same ``psd_tol_rel`` for consistency.
"""

from typing import List, Optional, Tuple

import numpy as np

from cpsd_inverse import CPSDInverseSolver


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
) -> np.ndarray:
    """
    Per-frequency closed-form S_r for a set of alphas on a training subset.

    Mirrors CPSDInverseSolver.solve_single_freq but skips the
    training-data residual computation since CV scores on held-out data.
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
        inv_filter = 1.0 / (sigma + alpha)
        K = Y @ (inv_filter[:, None] * Z)
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
    """

    def __init__(
        self,
        solver: CPSDInverseSolver,
        G: np.ndarray,
        k_folds: int = 5,
        seed: int = 0,
        save_fold_scores: bool = False,
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

        self.solver = solver
        self.G = G_arr
        self.k_folds = k_folds
        self.seed = seed
        self.save_fold_scores = save_fold_scores

    def score(
        self, alphas: np.ndarray, psd_tol_rel: float = 0.0
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Mean-over-folds CV score per frequency and per alpha.

        Parameters
        ----------
        alphas : array_like, shape (n_alpha,)
            Candidate regularization values, non-negative.
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

            for k_idx, I_val in enumerate(folds):
                I_train = np.setdiff1d(
                    all_idx, I_val, assume_unique=True
                )

                G_train = G_f[np.ix_(I_train, I_train)]
                G_val_clipped = _hermitize_clip(
                    G_f[np.ix_(I_val, I_val)], psd_tol_rel
                )

                S_r_alphas = _solve_for_alphas(
                    T_r_f[I_train, :], G_train, alphas, psd_tol_rel
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
