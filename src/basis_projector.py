"""
Per-frequency basis-projection residual computation.

Given two transfer matrices of shape (ndof, npws, nfreq) -- one treated as a
"basis" and one as "data" -- this module projects the data columns onto the
column space of the basis at each frequency and reports the relative residual
of that best (least-squares) approximation.

Each per-frequency slice ``M[:, :, i]`` is an (ndof x npws) matrix whose columns
are pressure fields over the dofs (see
``cone_diffuse_field.ConeDiffuseField._compute_total_field_matrix`` for the same
slicing convention). The basis column space is a subspace of C^ndof, so the
basis and data matrices must share the same ndof (rows) and the same number of
frequencies; the number of plane waves (columns) may differ.
"""

from typing import Any, Dict

import numpy as np


class BasisProjection:
    """Orthogonal projection of data columns onto a basis column space.

    For each frequency ``i`` with ``B = basis[:, :, i]`` and ``D = data[:, :, i]``:

    - Orthonormalize the basis columns via a thin SVD and keep the columns whose
      singular values exceed ``rtol * s[0]`` (numerical rank), giving an
      orthonormal ``Q`` (ndof x rank).
    - The best approximation of the data in ``col(B)`` is the orthogonal
      projection ``D_hat = Q @ (Q^H @ D)``.
    - The relative residual is the Frobenius norm ratio
      ``||D - D_hat||_F / ||D||_F``.

    Parameters
    ----------
    basis : ndarray
        Basis transfer matrix, shape (ndof, npws_basis, nfreq).
    data : ndarray
        Data transfer matrix, shape (ndof, npws_data, nfreq).
    rtol : float, optional
        Relative singular-value threshold for the numerical rank of the basis
        at each frequency. Default 1e-12.
    """

    def __init__(self, basis: np.ndarray, data: np.ndarray, rtol: float = 1e-12):
        basis = np.asarray(basis)
        data = np.asarray(data)

        if basis.ndim != 3:
            raise ValueError(
                f"basis must be 3D (ndof, npws, nfreq), got shape {basis.shape}"
            )
        if data.ndim != 3:
            raise ValueError(
                f"data must be 3D (ndof, npws, nfreq), got shape {data.shape}"
            )
        if basis.shape[0] != data.shape[0]:
            raise ValueError(
                f"basis and data must share ndof (rows): basis has "
                f"{basis.shape[0]}, data has {data.shape[0]}. The basis column "
                f"space lives in C^ndof, so the row dimensions must match."
            )
        if basis.shape[2] != data.shape[2]:
            raise ValueError(
                f"basis and data must share the number of frequencies: basis "
                f"has {basis.shape[2]}, data has {data.shape[2]}"
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
        self.nfreq = basis.shape[2]

    def project(self) -> Dict[str, Any]:
        """Compute the per-frequency projection residual.

        Returns
        -------
        dict
            Keys:
            ``relative_residual`` : (nfreq,) float array, ``||D - D_hat||_F / ||D||_F``.
            ``basis_rank``        : (nfreq,) int array, numerical rank of the basis.
            ``data_norm``         : (nfreq,) float array, ``||D||_F``.
            ``nfreq, ndof, npws_basis, npws_data`` : ints.
        """
        relative_residual = np.empty(self.nfreq, dtype=float)
        basis_rank = np.empty(self.nfreq, dtype=int)
        data_norm = np.empty(self.nfreq, dtype=float)

        for i in range(self.nfreq):
            B = self.basis[:, :, i]
            D = self.data[:, :, i]

            # Orthonormal basis for col(B) via thin SVD, truncated to numerical rank.
            U, s, _ = np.linalg.svd(B, full_matrices=False)
            if s.size == 0 or s[0] == 0.0:
                rank = 0
            else:
                rank = int(np.count_nonzero(s > self.rtol * s[0]))
            Q = U[:, :rank]

            # Orthogonal projection of D onto col(B): D_hat = Q (Q^H D).
            D_hat = Q @ (Q.conj().T @ D)
            residual_norm = np.linalg.norm(D - D_hat)
            d_norm = np.linalg.norm(D)

            basis_rank[i] = rank
            data_norm[i] = d_norm
            # Guard against an all-zero data slice (||D||_F == 0).
            relative_residual[i] = residual_norm / d_norm if d_norm > 0 else np.nan

        return {
            "relative_residual": relative_residual,
            "basis_rank": basis_rank,
            "data_norm": data_norm,
            "nfreq": self.nfreq,
            "ndof": self.ndof,
            "npws_basis": self.npws_basis,
            "npws_data": self.npws_data,
        }
