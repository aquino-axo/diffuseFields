"""
Per-frequency basis-projection residual computation.

Given a frequency-independent "basis" matrix of shape (ndof, npws_basis) and a
per-frequency "data" transfer matrix of shape (ndof, npws_data, nfreq), this
module projects the data columns onto the column space of the basis at each
frequency and reports the relative residual of that best (least-squares)
approximation.

Each per-frequency data slice ``data[:, :, i]`` is an (ndof x npws_data) matrix
whose columns are pressure fields over the dofs (see
``cone_diffuse_field.ConeDiffuseField._compute_total_field_matrix`` for the same
slicing convention). The basis column space is a single subspace of C^ndof that
does not change with frequency, so the basis SVD is computed once and reused for
every data frequency. The basis and data must share the same ndof (rows); the
number of plane waves (columns) may differ.
"""

from typing import Any, Dict

import numpy as np


class BasisProjection:
    """Orthogonal projection of per-frequency data onto a fixed basis column space.

    The basis is frequency-independent. With ``B = basis`` (ndof x npws_basis)
    and ``D = data[:, :, i]`` (ndof x npws_data) at frequency ``i``:

    - Orthonormalize the basis columns once via a thin SVD and keep the columns
      whose singular values exceed ``rtol * s[0]`` (numerical rank), giving an
      orthonormal ``Q`` (ndof x rank).
    - The best approximation of the data in ``col(B)`` is the orthogonal
      projection ``D_hat = Q @ (Q^H @ D)``.
    - The relative residual is the Frobenius norm ratio
      ``||D - D_hat||_F / ||D||_F``.

    Parameters
    ----------
    basis : ndarray
        Frequency-independent basis matrix, shape (ndof, npws_basis).
    data : ndarray
        Data transfer matrix, shape (ndof, npws_data, nfreq). A 2D array
        (ndof, npws_data) is accepted as a single frequency (nfreq = 1).
    rtol : float, optional
        Relative singular-value threshold for the numerical rank of the basis.
        Default 1e-12.
    """

    def __init__(self, basis: np.ndarray, data: np.ndarray, rtol: float = 1e-12):
        basis = np.asarray(basis)
        data = np.asarray(data)

        if basis.ndim != 2:
            raise ValueError(
                f"basis must be 2D (ndof, npws_basis) and frequency-independent, "
                f"got shape {basis.shape}"
            )
        # A 2D data array is a single frequency: promote to (ndof, npws_data, 1).
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        if data.ndim != 3:
            raise ValueError(
                f"data must be 2D (ndof, npws_data) for a single frequency or "
                f"3D (ndof, npws_data, nfreq), got shape {data.shape}"
            )
        if basis.shape[0] != data.shape[0]:
            raise ValueError(
                f"basis and data must share ndof (rows): basis has "
                f"{basis.shape[0]}, data has {data.shape[0]}. The basis column "
                f"space lives in C^ndof, so the row dimensions must match."
            )
        if rtol < 0:
            raise ValueError(f"rtol must be non-negative, got {rtol}")

        # Cast to complex so real-valued inputs are handled uniformly.
        self.basis = basis.astype(np.complex128, copy=False)
        self.data = data.astype(np.complex128, copy=False)
        self.rtol = float(rtol)

        self.ndof = basis.shape[0]
        self.npws_basis = basis.shape[1]
        self.npws_data = data.shape[1]
        self.nfreq = data.shape[2]

    def project(self) -> Dict[str, Any]:
        """Compute the per-frequency projection residual.

        Returns
        -------
        dict
            Keys:
            ``relative_residual`` : (nfreq,) float array, ``||D - D_hat||_F / ||D||_F``.
            ``basis_rank``        : int, numerical rank of the (fixed) basis.
            ``data_norm``         : (nfreq,) float array, ``||D||_F`` per frequency.
            ``nfreq, ndof, npws_basis, npws_data`` : ints.
        """
        # Orthonormal basis for col(B) via thin SVD, truncated to numerical rank.
        # Computed once because the basis does not depend on frequency.
        U, s, _ = np.linalg.svd(self.basis, full_matrices=False)
        if s.size == 0 or s[0] == 0.0:
            rank = 0
        else:
            rank = int(np.count_nonzero(s > self.rtol * s[0]))
        Q = U[:, :rank]
        Qh = Q.conj().T

        relative_residual = np.empty(self.nfreq, dtype=float)
        data_norm = np.empty(self.nfreq, dtype=float)

        for i in range(self.nfreq):
            D = self.data[:, :, i]

            # Orthogonal projection of D onto col(B): D_hat = Q (Q^H D).
            D_hat = Q @ (Qh @ D)
            residual_norm = np.linalg.norm(D - D_hat)
            d_norm = np.linalg.norm(D)

            data_norm[i] = d_norm
            # Guard against an all-zero data slice (||D||_F == 0).
            relative_residual[i] = residual_norm / d_norm if d_norm > 0 else np.nan

        return {
            "relative_residual": relative_residual,
            "basis_rank": rank,
            "data_norm": data_norm,
            "nfreq": self.nfreq,
            "ndof": self.ndof,
            "npws_basis": self.npws_basis,
            "npws_data": self.npws_data,
        }
