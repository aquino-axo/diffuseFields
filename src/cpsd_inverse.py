"""
Reduced-basis regularized inversion for cross-power spectral densities (CPSDs).

POD reduction:

  S    = Phi @ S_r @ Phi^h    (Phi in C^{N x n})
  T_r  = T @ Phi              (reduced transfer matrix, T_r in C^{m x n})

The target CPSD is factored G_hat ~ Psi Psi^h via PSD square root. Following
the *first* regularization method of Aquino & Bonnet, "Active design of
diffuse acoustic fields in enclosures", JASA 2023 (eqs. 19--21, Remark 2), we
solve one regularized problem per column phi_q of Psi and sum the outer
products. With the reduced SVD T_r = X Sigma Y^h and Z := X^h Psi,

  S_r(alpha) = K K^h,   K := Y diag(g(sigma_i, alpha)) Z,

for a diagonal spectral filter g. S_r is positive semidefinite by construction
for any alpha >= 0 and any g, being a sum of outer products s_q s_q^h --
unlike the paper's second method H_ij = sigma_i sigma_j (Z Z^h)_ij /
(sigma_i^2 sigma_j^2 + alpha) (eq. 24), which can come out indefinite for
alpha > 0.

Two filters are available via ``filter_form`` (see :func:`spectral_filter`):

  'lavrentiev'      g = 1/(sigma + alpha)          alpha ~ sigma^1   [default]
  'tikhonov'  g = sigma/(sigma^2 + alpha)    alpha ~ sigma^2

Both are monotone, both converge to the minimum-norm solution as alpha -> 0,
and both keep S_r PSD. They differ in which problem they solve:

- 'lavrentiev' reproduces eq. (21)/Remark 2 exactly as printed, and hence the
  paper's published numerical results. It is *not* the minimizer of the
  Tikhonov functional in eq. (19): it is the solution of the Lavrentiev-type
  equation ((T_r^h T_r)^{1/2} + alpha I) u = Q^h phi_q, where T_r = Q P is the
  polar decomposition (Q = X Y^h, P = (T_r^h T_r)^{1/2}). Equivalently it
  minimizes (1/2) u^h P u - Re(u^h Q^h phi_q) + (alpha/2)||u||^2.
- 'tikhonov' is the minimizer of the least-squares functional eq. (19)
  actually states, and matches the thin-QR route of Remark 4 (QR of
  [T_r; sqrt(alpha) I]), which 'lavrentiev' does not.

Convention for 'tikhonov': alpha is that of

  min_u  ||T_r u - phi_q||^2 + alpha ||u||^2

(equivalently (1/2)||.||^2 + (alpha/2)||u||^2), giving g = sigma/(sigma^2 +
alpha). This is the standard convention and the one Remark 4's QR route
implies. Eq. (19) as printed uses (1/2)||.||^2 + alpha||u||^2, whose minimizer
is sigma/(sigma^2 + 2 alpha) -- so the paper's eq. (19) and its Remark 4
differ from each other by a factor of 2 in alpha. Mapping:
alpha_here = 2 * alpha_eq19.

Because sigma_max(T_r) varies by orders of magnitude across a band containing
resonances, a single absolute alpha applies very different damping at
different frequencies. Prefer alpha_scaling='relative', which multiplies the
supplied value by sigma_max(f) raised to the filter's alpha power, so the
supplied number is dimensionless either way; see :func:`resolve_alphas`.
"""

from typing import Optional, Tuple

import numpy as np

#: Supported meanings for the user-supplied regularization parameter.
ALPHA_SCALINGS = ('absolute', 'relative')

#: Supported diagonal spectral filters.
FILTER_FORMS = ('lavrentiev', 'tikhonov')

#: Power of sigma_max that alpha scales with, per filter. Used by
#: alpha_scaling='relative' so the supplied parameter is dimensionless.
ALPHA_SIGMA_POWER = {'lavrentiev': 1, 'tikhonov': 2}


def spectral_filter(
    sigma: np.ndarray,
    alpha: float,
    filter_form: str = 'lavrentiev',
) -> np.ndarray:
    """
    Diagonal filter g applied as K = Y diag(g) X^h Psi.

    Parameters
    ----------
    sigma : ndarray, shape (r,)
        Singular values of T_r at this frequency, descending.
    alpha : float
        Effective (already resolved) regularization parameter.
    filter_form : {'lavrentiev', 'tikhonov'}
        ``'lavrentiev'`` gives ``1/(sigma + alpha)`` -- eq. (21)/Remark 2 as printed
        in the paper. ``'tikhonov'`` gives ``sigma/(sigma**2 + alpha)`` -- the
        minimizer of the least-squares functional of eq. (19), in the
        convention ``||T_r u - phi||^2 + alpha||u||^2``. See the module
        docstring for the factor-of-2 mapping to eq. (19)'s own alpha.

    Returns
    -------
    ndarray, shape (r,)
        The filter values. Both forms are monotone increasing in sigma and
        tend to 1/sigma as alpha -> 0.
    """
    if filter_form == 'lavrentiev':
        return 1.0 / (sigma + alpha)
    if filter_form == 'tikhonov':
        return sigma / (sigma ** 2 + alpha)
    raise ValueError(
        f"filter_form must be one of {FILTER_FORMS}, got {filter_form!r}"
    )


def resolve_alphas(
    alphas: np.ndarray,
    sigma_max: float,
    alpha_scaling: str = 'absolute',
    filter_form: str = 'lavrentiev',
) -> np.ndarray:
    """
    Map user-supplied regularization parameters to the alpha actually applied.

    Parameters
    ----------
    alphas : array_like, shape (n_alpha,)
        Non-negative parameters as given in the configuration.
    sigma_max : float
        Largest singular value of the *full* T_r at this frequency. Using the
        full-matrix value (not a CV fold's) keeps a given ``lam`` mapped to one
        and the same absolute alpha in both the CV folds and the final refit.
    alpha_scaling : {'absolute', 'relative'}
        ``'absolute'`` returns ``alphas`` unchanged; they then carry the
        filter's own units (sigma for ``'lavrentiev'``, sigma^2 for ``'tikhonov'``).
        ``'relative'`` returns ``alphas * sigma_max ** p`` with
        ``p = ALPHA_SIGMA_POWER[filter_form]``, making the supplied numbers
        dimensionless and comparable across frequencies for either filter.
    filter_form : {'lavrentiev', 'tikhonov'}
        Selects the power ``p``; ignored when ``alpha_scaling='absolute'``.

    Returns
    -------
    ndarray, shape (n_alpha,)
        The effective alphas to pass to :func:`spectral_filter`.
    """
    alphas = np.atleast_1d(np.asarray(alphas, dtype=np.float64))
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
    if alpha_scaling == 'absolute':
        return alphas
    if not np.isfinite(sigma_max) or sigma_max <= 0:
        raise ValueError(
            f"alpha_scaling='relative' needs a finite positive sigma_max, "
            f"got {sigma_max}"
        )
    return alphas * float(sigma_max) ** ALPHA_SIGMA_POWER[filter_form]


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

    def sigma_max(self, freq_idx: int) -> float:
        """
        Largest singular value of T_r at ``freq_idx``.

        Exposed so callers can report, or themselves resolve, the effective
        alpha implied by ``alpha_scaling='relative'``.
        """
        if not 0 <= freq_idx < self.n_freq:
            raise ValueError(
                f"freq_idx must be in [0, {self.n_freq}), got {freq_idx}"
            )
        s = np.linalg.svd(self.T_r[:, :, freq_idx], compute_uv=False)
        return float(s.max())

    def solve_single_freq(
        self,
        freq_idx: int,
        G: np.ndarray,
        alphas: np.ndarray,
        psd_tol_rel: float = 0.0,
        alpha_scaling: str = 'absolute',
        filter_form: str = 'lavrentiev',
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
            H_hat are computed once and reused across all alphas. Interpreted
            according to ``alpha_scaling``.
        psd_tol_rel : float
            Relative threshold for clipping G's eigenvalues to zero before
            forming the PSD square root.
        alpha_scaling : {'absolute', 'relative'}
            How to interpret ``alphas``; see :func:`resolve_alphas`.
            ``'relative'`` multiplies them by sigma_max(T_r) at this
            frequency, raised to the filter's alpha power, which is what makes
            one parameter value comparable across a band containing
            resonances.
        filter_form : {'lavrentiev', 'tikhonov'}
            Diagonal spectral filter; see :func:`spectral_filter`. Default
            ``'lavrentiev'`` reproduces the paper as printed.

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

        # Resolve lam -> alpha once the spectrum is known.
        alphas_eff = resolve_alphas(
            alphas, sigma[0], alpha_scaling, filter_form
        )

        # PSD square root: Psi Psi^h is the PSD projection of G.
        psi = self._psd_project(G, tol_rel=psd_tol_rel)  # (m, m)
        Z = X.conj().T @ psi                              # (r, m)

        G_fro = np.linalg.norm(G, 'fro')
        S_r_out = np.empty(
            (self.n_pod, self.n_pod, alphas.size), dtype=np.complex128
        )
        residuals_rel = np.empty(alphas.size, dtype=np.float64)

        for k, alpha in enumerate(alphas_eff):
            # K = Y diag(g(sigma)) Z, S_r = K K^h  (eq. 21 form).
            g = spectral_filter(sigma, alpha, filter_form)  # (r,)
            K = Y @ (g[:, None] * Z)                        # (n_pod, m)
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
