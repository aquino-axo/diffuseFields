"""
Cone surface diffuse field analysis.

Computes the cross-power spectral density (CPSD) of the total pressure field
on a conical structure surface produced by a diffuse acoustic field excitation.

The total field is p_t = p_inc + p_scat, where:
- p_inc is the incident plane wave field
- p_scat is the scattered field from the transfer matrix
"""

import numpy as np
from typing import Dict, Tuple, Optional, TYPE_CHECKING
from functools import partial

if TYPE_CHECKING:
    from cone_visualizer import ConeVisualizer

from randomized_eigensolver import RandomizedEigensolver, compute_eigenvalues_for_variance


class ConeDiffuseField:
    """
    Computes total field covariance on cone surface from diffuse field excitation.

    The total field is p_t = p_inc + p_scat = (D + T) @ a, where:
    - D is the incident field matrix: D[i,j] = exp(i * k * d_j · x_i)
    - T is the scattered field transfer matrix
    - a is the vector of plane wave amplitudes with random phases

    For a diffuse field with uncorrelated plane waves, the covariance is:
        C_pp = Po² * H @ H^H, where H = D + T

    Eigendecomposition is computed via SVD of H for efficiency:
        H = U @ Σ @ V^H  =>  eigenvalues = Po² * σ²,  eigenvectors = U

    Parameters
    ----------
    transfer_matrix : ndarray
        Transfer matrix of shape (ndof, npws, nfreqs) giving the scattered
        pressure on the cone surface for unit-amplitude plane waves.
    coordinates : ndarray
        Node coordinates on cone surface, shape (ndof, 3).
    directions : ndarray
        Plane wave unit direction vectors, shape (npws, 3).
    frequencies : ndarray
        Array of frequencies in Hz, shape (nfreqs,).
    speed_of_sound : float
        Speed of sound c for computing wavenumber k = 2πf/c.
    amplitude : float
        Constant amplitude Po of the plane waves.
    cone_geometry : dict
        Cone geometry parameters with keys:
        - 'half_angle': half-angle of the cone in radians
        - 'height': height of the cone

    Attributes
    ----------
    ndof : int
        Number of degrees of freedom (nodes on cone surface).
    npws : int
        Number of plane waves.
    nfreqs : int
        Number of frequencies.
    wavenumbers : ndarray
        Wavenumbers k = 2πf/c for each frequency.
    eigenvalues : ndarray or None
        All eigenvalues after calling compute_covariance_eigenvalues.
    eigenvectors : ndarray or None
        Retained eigenvectors after calling compute_covariance_eigenvalues.
    """

    def __init__(
        self,
        transfer_matrix: np.ndarray,
        coordinates: np.ndarray,
        directions: np.ndarray,
        frequencies: np.ndarray,
        speed_of_sound: float,
        amplitude: float,
        cone_geometry: Dict[str, float]
    ):
        self.transfer_matrix = np.asarray(transfer_matrix, dtype=np.complex128)
        self.coordinates = np.asarray(coordinates, dtype=np.float64)
        self.directions = np.asarray(directions, dtype=np.float64)
        self.frequencies = np.asarray(frequencies, dtype=np.float64)
        self.speed_of_sound = speed_of_sound
        self.amplitude = amplitude
        self.cone_geometry = dict(cone_geometry)  # Make a copy

        # Validate inputs
        self._validate_inputs()

        # Compute base_diameter if not provided
        if 'base_diameter' not in self.cone_geometry:
            self.cone_geometry['base_diameter'] = (
                2 * self.cone_geometry['height'] *
                np.tan(self.cone_geometry['half_angle'])
            )

        # Extract dimensions
        self.ndof = self.transfer_matrix.shape[0]
        self.npws = self.transfer_matrix.shape[1]
        self.nfreqs = self.transfer_matrix.shape[2]

        # Compute wavenumbers
        self.wavenumbers = 2 * np.pi * self.frequencies / self.speed_of_sound

        # Results storage
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        self._current_freq_idx: Optional[int] = None
        self._n_components_kept: Optional[int] = None

    def _validate_inputs(self) -> None:
        """Validate input dimensions and parameters."""
        if self.transfer_matrix.ndim != 3:
            raise ValueError(
                f"transfer_matrix must be 3D (ndof, npws, nfreqs), "
                f"got shape {self.transfer_matrix.shape}"
            )

        ndof = self.transfer_matrix.shape[0]
        npws = self.transfer_matrix.shape[1]
        nfreqs = self.transfer_matrix.shape[2]

        if self.coordinates.shape != (ndof, 3):
            raise ValueError(
                f"coordinates must have shape ({ndof}, 3), "
                f"got {self.coordinates.shape}"
            )

        if self.directions.shape != (npws, 3):
            raise ValueError(
                f"directions must have shape ({npws}, 3), "
                f"got {self.directions.shape}"
            )

        if self.frequencies.ndim != 1 or len(self.frequencies) != nfreqs:
            raise ValueError(
                f"frequencies must have shape ({nfreqs},), "
                f"got {self.frequencies.shape}"
            )

        if self.speed_of_sound <= 0:
            raise ValueError(
                f"speed_of_sound must be positive, got {self.speed_of_sound}"
            )

        if self.amplitude <= 0:
            raise ValueError(f"amplitude must be positive, got {self.amplitude}")

        required_keys = {'half_angle', 'height'}
        missing = required_keys - set(self.cone_geometry.keys())
        if missing:
            raise ValueError(f"cone_geometry missing keys: {missing}")

    def _compute_incident_field_matrix(self, freq_idx: int) -> np.ndarray:
        """
        Compute the incident field matrix D for a given frequency.

        D[i,j] = exp(i * k * d_j · x_i)

        Parameters
        ----------
        freq_idx : int
            Frequency index.

        Returns
        -------
        ndarray
            Incident field matrix of shape (ndof, npws).
        """
        k = self.wavenumbers[freq_idx]

        # Compute d_j · x_i for all i, j
        # coordinates: (ndof, 3), directions: (npws, 3)
        # Result: (ndof, npws)
        dot_products = self.coordinates @ self.directions.T

        # D[i,j] = exp(i * k * d_j · x_i)
        D = np.exp(1j * k * dot_products)

        return D

    def _compute_total_field_matrix(self, freq_idx: int) -> np.ndarray:
        """
        Compute the total field transfer matrix H = D + T.

        Parameters
        ----------
        freq_idx : int
            Frequency index.

        Returns
        -------
        ndarray
            Total field matrix of shape (ndof, npws).
        """
        D = self._compute_incident_field_matrix(freq_idx)
        T = self.transfer_matrix[:, :, freq_idx]
        return D + T

    def _covariance_matvec(self, freq_idx: int, v: np.ndarray) -> np.ndarray:
        """
        Compute the analytical covariance matrix-vector product.

        Computes C @ v = Po² * H @ (H^H @ v) without forming the full matrix,
        where H = D + T is the total field transfer matrix.

        Parameters
        ----------
        freq_idx : int
            Frequency index.
        v : ndarray
            Vector of shape (ndof,) or matrix of shape (ndof, k).

        Returns
        -------
        ndarray
            Result C @ v with same shape as v.
        """
        H = self._compute_total_field_matrix(freq_idx)  # (ndof, npws)
        Po_squared = self.amplitude ** 2

        # C @ v = Po² * H @ (H^H @ v)
        if v.ndim == 1:
            w = H.conj().T @ v      # (npws,)
            result = H @ w           # (ndof,)
        else:
            w = H.conj().T @ v      # (npws, k)
            result = H @ w           # (ndof, k)

        return Po_squared * result

    def compute_covariance_eigenvalues(
        self,
        freq_idx: int,
        var_ratio: float = 0.99,
        n_components: Optional[int] = None,
        solver: str = 'direct',
        n_oversamples: int = 10,
        n_power_iter: int = 2,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues of the total field pressure covariance matrix.

        Parameters
        ----------
        freq_idx : int
            Index into the frequency dimension of the transfer matrix.
        var_ratio : float, optional
            Minimum variance ratio to capture (0 < var_ratio <= 1), default 0.99.
            Used to determine number of eigenvectors to retain.
        n_components : int, optional
            Number of eigenvectors to return. If None, retains eigenvectors
            capturing var_ratio of total variance.
        solver : str, optional
            Solver to use: 'direct' (default) uses numpy SVD of H.
            'randomized' uses matrix-free randomized SVD (better for very large
            problems where even H cannot fit in memory).
        n_oversamples : int, optional
            Additional random vectors for randomized solver accuracy, default 10.
        n_power_iter : int, optional
            Number of power iterations for randomized solver, default 2.
        random_state : int, optional
            Random seed for reproducibility (randomized solver only).

        Returns
        -------
        eigenvalues : ndarray
            All eigenvalues in descending order (length = min(ndof, npws)).
        eigenvectors : ndarray
            Eigenvectors as columns, shape (ndof, n_components_kept).
            The number of eigenvectors is determined by var_ratio or n_components.
        """
        if not 0 < var_ratio <= 1:
            raise ValueError(f"var_ratio must be in (0, 1], got {var_ratio}")

        if not 0 <= freq_idx < self.nfreqs:
            raise ValueError(
                f"freq_idx must be in [0, {self.nfreqs}), got {freq_idx}"
            )

        if solver not in ['direct', 'randomized']:
            raise ValueError(f"solver must be 'direct' or 'randomized', got '{solver}'")

        if solver == 'direct':
            eigenvalues, eigenvectors = self._compute_eigenvalues_direct(
                freq_idx, var_ratio, n_components
            )
        else:
            eigenvalues, eigenvectors = self._compute_eigenvalues_randomized(
                freq_idx, var_ratio, n_components, n_oversamples, n_power_iter, random_state
            )

        # Store results
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self._current_freq_idx = freq_idx

        return eigenvalues, eigenvectors

    def compute_covariance_eigenvalues_all_freqs(
        self,
        freq_indices: Optional[list] = None,
        var_ratio: float = 0.99,
        n_components: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues of the covariance using all frequencies as snapshots.

        Builds the stacked total field matrix H_all = [H_0 | H_1 | ... | H_{nf-1}]
        of shape (ndof, npws * nf), then computes SVD. The covariance is:
            C_all = Po^2 * H_all @ H_all^H

        This captures the dominant spatial modes across all frequencies simultaneously.

        Parameters
        ----------
        freq_indices : list of int, optional
            Which frequency indices to include. If None, uses all frequencies.
        var_ratio : float, optional
            Minimum variance ratio to capture (0 < var_ratio <= 1), default 0.99.
            Used to determine number of eigenvectors to retain.
        n_components : int, optional
            Number of eigenvectors to return. If provided, overrides var_ratio.

        Returns
        -------
        eigenvalues : ndarray
            All eigenvalues in descending order, length min(ndof, npws * nf).
        eigenvectors : ndarray
            Retained eigenvectors as columns, shape (ndof, n_kept).
        """
        if not 0 < var_ratio <= 1:
            raise ValueError(f"var_ratio must be in (0, 1], got {var_ratio}")

        if freq_indices is None:
            freq_indices = list(range(self.nfreqs))

        for idx in freq_indices:
            if not 0 <= idx < self.nfreqs:
                raise ValueError(
                    f"freq_idx must be in [0, {self.nfreqs}), got {idx}"
                )

        # Stack total field matrices across all frequencies
        H_blocks = [self._compute_total_field_matrix(i) for i in freq_indices]
        H_all = np.hstack(H_blocks)  # (ndof, npws * nf)

        Po_squared = self.amplitude ** 2

        # SVD: H_all = U @ S @ V^H
        U, singular_values, _ = np.linalg.svd(H_all, full_matrices=False)

        # Eigenvalues of C_all = Po^2 * H_all @ H_all^H are Po^2 * sigma^2
        eigenvalues = Po_squared * singular_values ** 2

        eigenvectors = U

        # Determine number of eigenvectors to keep
        if n_components is not None:
            n_keep = min(n_components, len(eigenvalues))
        else:
            cumulative = np.cumsum(eigenvalues) / np.sum(eigenvalues)
            n_keep = np.searchsorted(cumulative, var_ratio) + 1
            n_keep = min(n_keep, len(eigenvalues))

        self._n_components_kept = n_keep

        # Store results
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors[:, :n_keep]
        self._current_freq_idx = None  # Signals multi-frequency result

        return eigenvalues, eigenvectors[:, :n_keep]

    def _compute_eigenvalues_direct(
        self,
        freq_idx: int,
        var_ratio: float,
        n_components: Optional[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues via SVD. Returns all eigenvalues, truncated eigenvectors.

        This is more efficient than eigendecomposition of C when npws < ndof:
        SVD of H (ndof × npws) is O(ndof × npws²) vs O(ndof³) for eigh(C).
        """
        H = self._compute_total_field_matrix(freq_idx)
        Po_squared = self.amplitude ** 2

        # Compute SVD: H = U @ S @ V^H
        # The eigenvalues of C = H @ H^H are σ²
        # The eigenvectors of C are the left singular vectors U
        U, singular_values, _ = np.linalg.svd(H, full_matrices=False)

        # Eigenvalues are Po² * σ² (already in descending order from SVD)
        eigenvalues = Po_squared * singular_values ** 2

        # Left singular vectors are the eigenvectors
        eigenvectors = U

        # Determine number of eigenvectors to keep based on variance ratio or n_components
        if n_components is not None:
            n_keep = min(n_components, len(eigenvalues))
        else:
            # Find number needed to capture var_ratio
            cumulative = np.cumsum(eigenvalues) / np.sum(eigenvalues)
            n_keep = np.searchsorted(cumulative, var_ratio) + 1
            n_keep = min(n_keep, len(eigenvalues))

        # Store the cutoff index for plotting
        self._n_components_kept = n_keep

        # Return ALL eigenvalues but only truncated eigenvectors
        return eigenvalues, eigenvectors[:, :n_keep]

    def _compute_eigenvalues_randomized(
        self,
        freq_idx: int,
        var_ratio: float,
        n_components: Optional[int],
        n_oversamples: int,
        n_power_iter: int,
        random_state: Optional[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues using randomized SVD (matrix-free).
        """
        matvec_fn = partial(self._covariance_matvec, freq_idx)

        if n_components is not None:
            solver = RandomizedEigensolver(
                matvec_fn=matvec_fn,
                matrix_size=self.ndof,
                n_components=n_components,
                n_oversamples=n_oversamples,
                n_power_iter=n_power_iter,
                random_state=random_state
            )
            eigenvalues, eigenvectors = solver.compute()
        else:
            eigenvalues, eigenvectors, _ = compute_eigenvalues_for_variance(
                matvec_fn=matvec_fn,
                matrix_size=self.ndof,
                var_ratio=var_ratio,
                n_oversamples=n_oversamples,
                n_power_iter=n_power_iter,
                random_state=random_state
            )

        return eigenvalues, eigenvectors

    def get_variance_explained(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the variance explained by each eigenvalue.

        Returns
        -------
        eigenvalues : ndarray
            The computed eigenvalues.
        cumulative_variance : ndarray
            Cumulative variance explained (fraction from 0 to 1).

        Raises
        ------
        RuntimeError
            If eigenvalues have not been computed yet.
        """
        if self.eigenvalues is None:
            raise RuntimeError(
                "Eigenvalues not computed. Call compute_covariance_eigenvalues first."
            )

        total = np.sum(self.eigenvalues)
        cumulative_variance = np.cumsum(self.eigenvalues) / total

        return self.eigenvalues, cumulative_variance

    def get_visualizer(self) -> 'ConeVisualizer':
        """
        Get a visualizer for this cone's geometry.

        Returns
        -------
        ConeVisualizer
            A visualizer instance configured with this cone's coordinates
            and geometry.
        """
        from cone_visualizer import ConeVisualizer
        return ConeVisualizer(self.coordinates, self.cone_geometry)
