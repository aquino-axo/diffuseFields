"""
Reduced-basis Tikhonov inversion for cross-power spectral densities (CPSDs).

POD reduction:

  S    = Phi @ S_r @ Phi^h    (Phi in C^{N x n})
  T_r  = T @ Phi              (reduced transfer matrix, T_r in C^{m x n})

The target CPSD is factored G_hat ~ Psi Psi^h via PSD square root.
Following Aquino & Bonnet, "Active Design of Diffuse Acoustic Fields in
Enclosures" (eqs. 34--36), we regularize the per-direction least-squares
problem

  s_q(alpha) = argmin_u (1/2) || T_r u - phi_q ||_2^2 + alpha || u ||_2^2,

whose closed-form solution via the reduced SVD T_r = X Sigma Y^h is

  s_q(alpha) = Y (Sigma + alpha I)^{-1} X^h phi_q,

and summing over the columns phi_q of Psi gives

  S_r(alpha) = K K^h,   K := Y (Sigma + alpha I)^{-1} Z,   Z = X^h Psi.

S_r is positive semidefinite by construction for any alpha >= 0 (cf.
Remark 3 of the reference). This replaces the earlier entrywise filter
H_ij = sigma_i sigma_j (Z Z^h)_ij / (sigma_i^2 sigma_j^2 + alpha)
(reference eq. 43), which is Hermitian but not PSD-preserving when
alpha > 0; both formulations converge to the same minimum-norm solution
as alpha -> 0.
"""

from typing import Optional, Tuple

import numpy as np


class CPSDInverseSolver:
    """
    Solve the regularized reduced-basis CPSD inverse problem.

    Parameters
    ----------
    reduced_transfer_matrix : ndarray
        Reduced transfer matrix T_r = T @ Phi, shape (n_sensors, n_pod, n_freq).
    pod_basis : ndarray, optional
        POD basis Phi of shape (N, n_pod). Required only for full-space
        reconstruction; pass None if you only need S_r.

    Attributes
    ----------
    n_sensors, n_pod, n_freq : int
        Dimensions taken from the reduced transfer matrix.
    N_full : int or None
        Full-space dimension (rows of Phi) when pod_basis was supplied.
    """

    def __init__(
        self,
        reduced_transfer_matrix: np.ndarray,
        pod_basis: Optional[np.ndarray] = None,
    ):
        T_r = np.asarray(reduced_transfer_matrix, dtype=np.complex128)
        if T_r.ndim != 3:
            raise ValueError(
                f"reduced_transfer_matrix must be 3D "
                f"(n_sensors, n_pod, n_freq), got shape {T_r.shape}"
            )
        self.T_r = T_r
        self.n_sensors, self.n_pod, self.n_freq = T_r.shape

        if pod_basis is not None:
            phi = np.asarray(pod_basis, dtype=np.complex128)
            if phi.ndim != 2 or phi.shape[1] != self.n_pod:
                raise ValueError(
                    f"pod_basis must have shape (N, {self.n_pod}) to match "
                    f"the transfer matrix, got {phi.shape}"
                )
            self.pod_basis = phi
            self.N_full = phi.shape[0]
        else:
            self.pod_basis = None
            self.N_full = None

    @staticmethod
    def _psd_project(G: np.ndarray, tol_rel: float = 0.0) -> np.ndarray:
        """
        Hermitize G and PSD-project it via eigen-decomposition.

        Returns Psi such that Psi @ Psi^h equals the PSD projection of G
        (negative eigenvalues clipped to zero; optionally clipped below
        tol_rel * max(|lambda|)).
        """
        Gh = 0.5 * (G + G.conj().T)
        lam, U = np.linalg.eigh(Gh)
        if tol_rel > 0:
            cutoff = tol_rel * np.max(np.abs(lam))
            lam = np.where(lam > cutoff, lam, 0.0)
        else:
            lam = np.where(lam > 0, lam, 0.0)
        # Scale each column of U by sqrt(lam_j) so that Psi @ Psi^h = U L U^h.
        return U * np.sqrt(lam)[np.newaxis, :]

    def solve_single_freq(
        self,
        freq_idx: int,
        G: np.ndarray,
        alphas: np.ndarray,
        psd_tol_rel: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the inverse problem at one frequency for one or more alphas.

        Parameters
        ----------
        freq_idx : int
            Index along the frequency axis of T_r.
        G : ndarray, shape (n_sensors, n_sensors)
            Experimental CPSD at this frequency. Hermitized internally.
        alphas : array_like, shape (n_alpha,)
            Non-negative regularization parameters to evaluate. The SVD and
            H_hat are computed once and reused across all alphas.
        psd_tol_rel : float
            Relative threshold for clipping G's eigenvalues to zero before
            forming the PSD square root.

        Returns
        -------
        S_r : ndarray, shape (n_pod, n_pod, n_alpha)
            Recovered reduced CPSD per alpha.
        residuals_rel : ndarray, shape (n_alpha,)
            Relative Frobenius residual
            || T_r S_r T_r^h - G ||_F / ||G||_F (zero denominator falls
            back to absolute Frobenius norm).
        """
        if not 0 <= freq_idx < self.n_freq:
            raise ValueError(
                f"freq_idx must be in [0, {self.n_freq}), got {freq_idx}"
            )
        if G.shape != (self.n_sensors, self.n_sensors):
            raise ValueError(
                f"G must have shape ({self.n_sensors}, {self.n_sensors}), "
                f"got {G.shape}"
            )

        alphas = np.atleast_1d(np.asarray(alphas, dtype=np.float64))
        if np.any(alphas < 0):
            raise ValueError("alphas must be non-negative")

        T_r = self.T_r[:, :, freq_idx]

        # Reduced SVD: T_r = X Sigma Vh, with r = min(m, n_pod).
        X, sigma, Vh = np.linalg.svd(T_r, full_matrices=False)
        Y = Vh.conj().T  # (n_pod, r)

        # PSD square root: Psi Psi^h is the PSD projection of G.
        psi = self._psd_project(G, tol_rel=psd_tol_rel)  # (m, m)
        Z = X.conj().T @ psi                              # (r, m)

        G_fro = np.linalg.norm(G, 'fro')
        S_r_out = np.empty(
            (self.n_pod, self.n_pod, alphas.size), dtype=np.complex128
        )
        residuals_rel = np.empty(alphas.size, dtype=np.float64)

        for k, alpha in enumerate(alphas):
            # K = Y (Sigma + alpha I)^{-1} Z, S_r = K K^h  (eqs. 35-36).
            inv_filter = 1.0 / (sigma + alpha)             # (r,)
            K = Y @ (inv_filter[:, None] * Z)              # (n_pod, m)
            S_r = K @ K.conj().T                           # PSD by construction
            S_r_out[:, :, k] = S_r

            # n_sensors is typically small, so direct residual is cheap.
            G_model = T_r @ S_r @ T_r.conj().T
            res_abs = np.linalg.norm(G_model - G, 'fro')
            residuals_rel[k] = res_abs / G_fro if G_fro > 0 else res_abs

        return S_r_out, residuals_rel

    def reconstruct_full_cpsd(
        self,
        S_r: np.ndarray,
        diagonal_only: bool = False,
    ) -> np.ndarray:
        """
        Lift reduced CPSD to the full space: S* = Phi @ S_r @ Phi^h.

        Parameters
        ----------
        S_r : ndarray, shape (n_pod, n_pod)
            Reduced CPSD at a single frequency.
        diagonal_only : bool, optional
            If True, return only diag(S*) of shape (N,) (real-valued) instead
            of the full (N, N) matrix.

        Returns
        -------
        S_full : ndarray
            (N, N) complex CPSD by default, or real (N,) diagonal when
            diagonal_only is True.
        """
        if self.pod_basis is None:
            raise RuntimeError(
                "pod_basis was not provided; cannot reconstruct full CPSD."
            )
        return lift_to_full_space(S_r, self.pod_basis, diagonal_only)


def lift_to_full_space(
    S_r: np.ndarray,
    pod_basis: np.ndarray,
    diagonal_only: bool = False,
) -> np.ndarray:
    """
    Module-level helper for S* = Phi @ S_r @ Phi^h.

    Parameters
    ----------
    S_r : ndarray, shape (n_pod, n_pod)
    pod_basis : ndarray, shape (N, n_pod)
    diagonal_only : bool
        If True, return only the real-valued diagonal of S*, shape (N,).

    Returns
    -------
    ndarray
        Full CPSD (N, N) or its diagonal (N,) when diagonal_only.
    """
    phi = np.asarray(pod_basis)
    S_r = np.asarray(S_r)
    if diagonal_only:
        tmp = phi @ S_r                                   # (N, n_pod)
        diag = np.einsum('ij,ij->i', tmp, phi.conj())     # (N,)
        return diag.real
    return phi @ S_r @ phi.conj().T
