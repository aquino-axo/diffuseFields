"""
Randomized eigensolver for matrix-free eigenvalue computation.

This module provides a reusable randomized SVD-based eigensolver that computes
the top eigenvalues and eigenvectors of a Hermitian positive semi-definite matrix
without explicitly forming the matrix.
"""

import numpy as np
from scipy.linalg import qr
from typing import Callable, Tuple, Optional


class RandomizedEigensolver:
    """
    Matrix-free randomized eigenvalue decomposition for Hermitian matrices.

    Uses randomized SVD algorithm to compute top eigenvalues and eigenvectors
    of a Hermitian positive semi-definite matrix defined only through its
    matrix-vector product.

    Parameters
    ----------
    matvec_fn : callable
        Function that computes the matrix-vector product A @ v.
        Should accept array of shape (n,) or (n, k) and return same shape.
    matrix_size : int
        Dimension of the square matrix (n x n).
    n_components : int
        Number of eigenvalues/eigenvectors to compute.
    n_oversamples : int, optional
        Additional random vectors for improved accuracy, default 10.
    n_power_iter : int, optional
        Number of power iterations for improved accuracy, default 2.
    random_state : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> def matvec(v):
    ...     return A @ v  # A is Hermitian PSD
    >>> solver = RandomizedEigensolver(matvec, n=1000, n_components=10)
    >>> eigenvalues, eigenvectors = solver.compute()
    """

    def __init__(
        self,
        matvec_fn: Callable[[np.ndarray], np.ndarray],
        matrix_size: int,
        n_components: int,
        n_oversamples: int = 10,
        n_power_iter: int = 2,
        random_state: Optional[int] = None
    ):
        self.matvec_fn = matvec_fn
        self.matrix_size = matrix_size
        self.n_components = n_components
        self.n_oversamples = n_oversamples
        self.n_power_iter = n_power_iter
        self.rng = np.random.default_rng(random_state)

    def compute(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute top eigenvalues and eigenvectors.

        Returns
        -------
        eigenvalues : ndarray
            Top eigenvalues in descending order, shape (n_components,).
        eigenvectors : ndarray
            Corresponding eigenvectors as columns, shape (n, n_components).
            Eigenvectors are normalized to unit norm.
        """
        n = self.matrix_size
        k = self.n_components + self.n_oversamples

        # Stage 1: Find approximate range of the matrix
        # Generate random test matrix (complex Gaussian)
        Omega = (
            self.rng.standard_normal((n, k)) +
            1j * self.rng.standard_normal((n, k))
        ) / np.sqrt(2)

        # Form Y = A @ Omega
        Y = self.matvec_fn(Omega)

        # Power iteration to improve range approximation
        for _ in range(self.n_power_iter):
            # Orthonormalize for numerical stability
            Y, _ = qr(Y, mode='economic')
            # Apply A twice (A is Hermitian, so A² has same eigenvectors)
            Y = self.matvec_fn(Y)
            Y = self.matvec_fn(Y)

        # Orthonormal basis for range of Y
        Q, _ = qr(Y, mode='economic')

        # Stage 2: Form small projected matrix and compute its eigendecomposition
        # B = Q^H @ A @ Q is a k×k matrix
        AQ = self.matvec_fn(Q)
        B = Q.conj().T @ AQ

        # Force exact Hermitian symmetry
        B = (B + B.conj().T) / 2

        # Eigendecomposition of small matrix
        eigvals_B, eigvecs_B = np.linalg.eigh(B)

        # Sort in descending order
        idx = np.argsort(eigvals_B)[::-1]
        eigvals_B = eigvals_B[idx]
        eigvecs_B = eigvecs_B[:, idx]

        # Stage 3: Recover eigenvectors of original matrix
        eigenvalues = eigvals_B[:self.n_components]
        eigenvectors = Q @ eigvecs_B[:, :self.n_components]

        return eigenvalues, eigenvectors


def compute_eigenvalues_for_variance(
    matvec_fn: Callable[[np.ndarray], np.ndarray],
    matrix_size: int,
    var_ratio: float,
    initial_components: int = 10,
    max_components: int = 100,
    n_oversamples: int = 10,
    n_power_iter: int = 2,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute eigenvalues until specified variance ratio is captured.

    Iteratively computes eigenvalues, increasing the number until the
    cumulative variance explained exceeds the target ratio.

    Parameters
    ----------
    matvec_fn : callable
        Function that computes the matrix-vector product A @ v.
    matrix_size : int
        Dimension of the square matrix.
    var_ratio : float
        Target variance ratio to capture (0 < var_ratio <= 1).
    initial_components : int, optional
        Initial number of eigenvalues to compute, default 10.
    max_components : int, optional
        Maximum number of eigenvalues to compute, default 100.
    n_oversamples : int, optional
        Additional random vectors for accuracy, default 10.
    n_power_iter : int, optional
        Number of power iterations, default 2.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    eigenvalues : ndarray
        Eigenvalues capturing the specified variance.
    eigenvectors : ndarray
        Corresponding eigenvectors as columns.
    actual_variance : float
        Actual variance ratio captured.
    """
    n_components = initial_components

    while n_components <= max_components:
        solver = RandomizedEigensolver(
            matvec_fn=matvec_fn,
            matrix_size=matrix_size,
            n_components=n_components,
            n_oversamples=n_oversamples,
            n_power_iter=n_power_iter,
            random_state=random_state
        )
        eigenvalues, eigenvectors = solver.compute()

        # Estimate total variance from computed eigenvalues
        # For randomized methods, we use the sum of computed eigenvalues
        # as an approximation (assuming remaining eigenvalues are small)
        cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)

        # Find number of eigenvalues needed
        n_needed = np.searchsorted(cumulative_variance, var_ratio) + 1

        if n_needed <= n_components:
            # We have enough eigenvalues
            return (
                eigenvalues[:n_needed],
                eigenvectors[:, :n_needed],
                cumulative_variance[n_needed - 1]
            )

        # Need more eigenvalues, double the count
        n_components = min(n_components * 2, max_components)

    # Return all computed eigenvalues if max reached
    return eigenvalues, eigenvectors, cumulative_variance[-1]
