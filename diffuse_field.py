"""
DiffuseField3D: A class for modeling 3D diffuse acoustic fields.

This module implements plane wave superposition to simulate diffuse acoustic fields
and compute spatial correlations, validating against the analytical sinc(kr) correlation.

Mathematical Background:
    A diffuse field is modeled as a superposition of N plane waves with:
    - Fixed uniformly distributed directions on the unit sphere
    - Random phases (regenerated per realization) to approximate ensemble statistics

    Pressure field: P(x) = (1/√N) Σₙ exp(i(k·dₙ·x + φₙ))
    Analytical correlation: ρ(r) = sinc(kr) = sin(kr)/(kr)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr


class DiffuseField3D:
    """
    A class to model 3D diffuse acoustic fields using plane wave superposition.

    The class computes the centered correlation C(x₀, x) = E[P(x₀)P*(x)] where x₀
    is the center of the domain, and validates against the analytical sinc(kr) function.

    Parameters
    ----------
    f_min : float
        Minimum frequency [Hz]
    f_max : float
        Maximum frequency [Hz]
    n_freq : int
        Number of frequency steps (linear spacing)
    c : float, optional
        Speed of sound [m/s], default 343.0
    n_waves : int, optional
        Number of plane waves per realization, default 200
    n_realizations : int, optional
        Number of realizations for ensemble average, default 500

    Attributes
    ----------
    frequencies : ndarray
        Linear array of frequencies from f_min to f_max
    wavenumbers : ndarray
        Wavenumbers k = 2πf/c for each frequency
    lambda_min : float
        Minimum wavelength (at f_max)
    lambda_max : float
        Maximum wavelength (at f_min)
    box_size : float
        Domain size L = 2λ_max (to capture 2 sinc periods)
    grid_spacing : float
        Grid spacing dx = λ_min/8 (4× Nyquist at f_max)
    grid_points : ndarray
        (M, 3) array of grid point coordinates
    directions : ndarray
        (N, 3) array of fixed plane wave directions
    """

    def __init__(
        self,
        f_min: float,
        f_max: float,
        n_freq: int,
        c: float = 343.0,
        n_waves: int = 200,
        n_realizations: int = 500
    ):
        # Validate inputs
        if f_min <= 0:
            raise ValueError("f_min must be positive")
        if f_max <= 0:
            raise ValueError("f_max must be positive")
        if f_min > f_max:
            raise ValueError("f_min must be less than or equal to f_max")
        if n_freq < 1:
            raise ValueError("n_freq must be at least 1")
        if f_min == f_max and n_freq > 1:
            raise ValueError("n_freq must be 1 when f_min == f_max")
        if c <= 0:
            raise ValueError("Speed of sound must be positive")
        if n_waves < 1:
            raise ValueError("n_waves must be at least 1")
        if n_realizations < 1:
            raise ValueError("n_realizations must be at least 1")

        # Store parameters
        self.f_min = f_min
        self.f_max = f_max
        self.n_freq = n_freq
        self.c = c
        self.n_waves = n_waves
        self.n_realizations = n_realizations

        # Compute frequencies and wavenumbers
        if f_min == f_max:
            self.frequencies = np.array([f_min])
        else:
            self.frequencies = np.linspace(f_min, f_max, n_freq)
        self.wavenumbers = 2 * np.pi * self.frequencies / c

        # Compute wavelength bounds
        self.lambda_min = c / f_max  # Shortest wavelength (highest frequency)
        self.lambda_max = c / f_min  # Longest wavelength (lowest frequency)

        # Domain sizing: 2 sinc periods at the frequency
        self.box_size = 2 * self.lambda_max

        # Grid spacing: 4× Nyquist at the frequency
        self.grid_spacing = self.lambda_min / 8

        # Generate grid and plane wave directions
        self.grid_points, self.grid_shape, self.coords_1d = self._generate_grid()
        self.directions = self._generate_plane_wave_directions(n_waves)

        # Find index of center point (origin)
        self.center_idx = self._find_center_index()

    def _generate_grid(self) -> tuple:
        """
        Generate a uniform Cartesian grid within the cubic domain.

        Returns
        -------
        grid_points : ndarray
            (M, 3) array of grid point coordinates
        grid_shape : tuple
            (nx, ny, nz) shape of the 3D grid
        coords_1d : ndarray
            1D coordinate array used for each axis
        """
        n_side = int(np.ceil(self.box_size / self.grid_spacing)) + 1

        # Create 1D coordinate array
        coords_1d = np.linspace(-self.box_size / 2, self.box_size / 2, n_side)

        # Create 3D meshgrid
        X, Y, Z = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')

        # Store shape for later reshaping
        grid_shape = X.shape

        # Reshape to (M, 3) array
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        return grid_points, grid_shape, coords_1d

    def _find_center_index(self) -> int:
        """Find the index of the grid point closest to the origin."""
        distances = np.linalg.norm(self.grid_points, axis=1)
        return np.argmin(distances)

    def _generate_plane_wave_directions(self, n: int) -> np.ndarray:
        """
        Generate uniformly distributed unit vectors on the unit sphere.

        Uses the correct formula for uniform distribution:
            θ = arccos(1 - 2u₁)
            φ = 2πu₂
        where u₁, u₂ ~ Uniform[0, 1)

        Parameters
        ----------
        n : int
            Number of direction vectors to generate

        Returns
        -------
        ndarray
            (n, 3) array of unit direction vectors
        """
        u1 = np.random.rand(n)
        u2 = np.random.rand(n)

        theta = np.arccos(1 - 2 * u1)  # Polar angle [0, π]
        phi = 2 * np.pi * u2           # Azimuthal angle [0, 2π)

        # Convert to Cartesian coordinates
        directions = np.column_stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

        return directions

    def _compute_pressure_field(self, k: float) -> np.ndarray:
        """
        Compute a single realization of the pressure field.

        Uses fixed directions (stored in self.directions) and generates
        random phases for this realization.

        P(x) = (1/√N) Σₙ exp(i(k·dₙ·x + φₙ))

        Parameters
        ----------
        k : float
            Wavenumber

        Returns
        -------
        ndarray
            Complex pressure field of shape (M,)
        """
        # Random phases for this realization
        phases = 2 * np.pi * np.random.rand(self.n_waves)

        # Compute phase arguments: (M, N) = (M, 3) @ (3, N)
        # k * (d · x) for all grid points and all directions
        phase_arg = k * (self.grid_points @ self.directions.T) + phases[np.newaxis, :]

        # Sum over plane waves with normalization
        pressure = np.sum(np.exp(1j * phase_arg), axis=1) / np.sqrt(self.n_waves)

        return pressure

    def compute_centered_correlation(self, freq_idx: int) -> np.ndarray:
        """
        Compute the centered correlation C(x₀, x) = E[P(x₀)P*(x)].

        The correlation is computed with respect to the center of the domain (x₀ = 0).

        Parameters
        ----------
        freq_idx : int
            Index into the frequencies array

        Returns
        -------
        ndarray
            Complex correlation array of shape (M,), same as grid_points
        """
        k = self.wavenumbers[freq_idx]
        M = len(self.grid_points)

        # Initialize correlation array
        C = np.zeros(M, dtype=complex)

        # Accumulate over realizations
        for _ in range(self.n_realizations):
            P = self._compute_pressure_field(k)
            P_center = P[self.center_idx]
            C += P_center * np.conj(P)

        # Average over realizations
        C /= self.n_realizations

        return C

    def analytical_correlation(self, r: np.ndarray, k: float) -> np.ndarray:
        """
        Compute the analytical diffuse field correlation.

        ρ(r) = sinc(kr) = sin(kr)/(kr)

        Parameters
        ----------
        r : ndarray
            Separation distances from center
        k : float
            Wavenumber

        Returns
        -------
        ndarray
            Analytical correlation values
        """
        kr = k * r
        # Handle kr = 0 case (sinc(0) = 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            rho = np.where(kr == 0, 1.0, np.sin(kr) / kr)
        return rho

    def compute_normalized_mse(self, freq_idx: int) -> float:
        """
        Compute normalized mean squared error between computed and analytical correlation.

        NMSE = Σ|C_comp - C_anal|² / Σ|C_anal|²

        Parameters
        ----------
        freq_idx : int
            Index into the frequencies array

        Returns
        -------
        float
            Normalized MSE
        """
        k = self.wavenumbers[freq_idx]

        # Compute centered correlation
        C_computed = self.compute_centered_correlation(freq_idx)

        # Compute distances from center
        r = np.linalg.norm(self.grid_points, axis=1)

        # Compute analytical correlation
        C_analytical = self.analytical_correlation(r, k)

        # Use real part of computed correlation for comparison
        C_computed_real = np.real(C_computed)

        # Compute NMSE
        numerator = np.sum((C_computed_real - C_analytical) ** 2)
        denominator = np.sum(C_analytical ** 2)

        if denominator == 0:
            return np.inf

        return numerator / denominator

    def compute_all_frequencies(self) -> dict:
        """
        Compute correlations and NMSE for all frequencies.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'frequencies': array of frequencies
            - 'nmse': array of NMSE values
        """
        results = {
            'frequencies': self.frequencies.copy(),
            'nmse': np.zeros(self.n_freq)
        }

        for i, f in enumerate(self.frequencies):
            print(f"Processing frequency {i+1}/{self.n_freq}: {f:.1f} Hz")
            results['nmse'][i] = self.compute_normalized_mse(i)

        return results

    def plot_centered_correlation(
        self,
        freq_idx: int,
        plane: str = 'Z',
        ax: plt.Axes = None
    ) -> plt.Axes:
        """
        Plot the centered correlation on a specified plane through the origin.

        Parameters
        ----------
        freq_idx : int
            Index into the frequencies array
        plane : str, optional
            Plane to plot: 'X' (yz-plane), 'Y' (xz-plane), or 'Z' (xy-plane).
            Default is 'Z'.
        ax : plt.Axes, optional
            Axes to plot on. If None, creates new figure.

        Returns
        -------
        plt.Axes
            The axes with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        k = self.wavenumbers[freq_idx]
        f = self.frequencies[freq_idx]

        # Compute centered correlation
        C = self.compute_centered_correlation(freq_idx)
        C_real = np.real(C)

        # Reshape to 3D grid
        C_3d = C_real.reshape(self.grid_shape)

        # Get the middle index for slicing
        mid_idx = len(self.coords_1d) // 2

        # Select the appropriate plane
        plane = plane.upper()
        if plane == 'Z':
            C_plane = C_3d[:, :, mid_idx]
            xlabel, ylabel = 'x [m]', 'y [m]'
            extent = [self.coords_1d[0], self.coords_1d[-1],
                      self.coords_1d[0], self.coords_1d[-1]]
        elif plane == 'Y':
            C_plane = C_3d[:, mid_idx, :]
            xlabel, ylabel = 'x [m]', 'z [m]'
            extent = [self.coords_1d[0], self.coords_1d[-1],
                      self.coords_1d[0], self.coords_1d[-1]]
        elif plane == 'X':
            C_plane = C_3d[mid_idx, :, :]
            xlabel, ylabel = 'y [m]', 'z [m]'
            extent = [self.coords_1d[0], self.coords_1d[-1],
                      self.coords_1d[0], self.coords_1d[-1]]
        else:
            raise ValueError(f"plane must be 'X', 'Y', or 'Z', got '{plane}'")

        # Compute NMSE for this frequency
        r = np.linalg.norm(self.grid_points, axis=1)
        C_analytical = self.analytical_correlation(r, k)
        nmse = np.sum((C_real - C_analytical) ** 2) / np.sum(C_analytical ** 2)

        # Plot
        im = ax.imshow(
            C_plane.T,
            extent=extent,
            origin='lower',
            cmap='RdBu_r',
            vmin=-1,
            vmax=1,
            aspect='equal'
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Centered Correlation at f = {f:.1f} Hz ({plane}-plane)\nNMSE = {nmse:.2e}')

        plt.colorbar(im, ax=ax, label=r'$\Re[C(x_0, x)]$')

        return ax

    def plot_correlation_comparison(
        self,
        freq_idx: int,
        plane: str = 'Z',
        fig: plt.Figure = None
    ) -> plt.Figure:
        """
        Plot computed vs analytical correlation side by side.

        Parameters
        ----------
        freq_idx : int
            Index into the frequencies array
        plane : str, optional
            Plane to plot: 'X', 'Y', or 'Z'. Default is 'Z'.
        fig : plt.Figure, optional
            Figure to plot on. If None, creates new figure.

        Returns
        -------
        plt.Figure
            The figure with the plots
        """
        if fig is None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        else:
            axes = fig.subplots(1, 3)

        k = self.wavenumbers[freq_idx]
        f = self.frequencies[freq_idx]

        # Compute centered correlation
        C = self.compute_centered_correlation(freq_idx)
        C_real = np.real(C)

        # Compute analytical correlation
        r = np.linalg.norm(self.grid_points, axis=1)
        C_analytical = self.analytical_correlation(r, k)

        # Reshape to 3D
        C_computed_3d = C_real.reshape(self.grid_shape)
        C_analytical_3d = C_analytical.reshape(self.grid_shape)

        # Get the middle index for slicing
        mid_idx = len(self.coords_1d) // 2

        # Select the appropriate plane
        plane = plane.upper()
        if plane == 'Z':
            C_comp_plane = C_computed_3d[:, :, mid_idx]
            C_anal_plane = C_analytical_3d[:, :, mid_idx]
            xlabel, ylabel = 'x [m]', 'y [m]'
        elif plane == 'Y':
            C_comp_plane = C_computed_3d[:, mid_idx, :]
            C_anal_plane = C_analytical_3d[:, mid_idx, :]
            xlabel, ylabel = 'x [m]', 'z [m]'
        elif plane == 'X':
            C_comp_plane = C_computed_3d[mid_idx, :, :]
            C_anal_plane = C_analytical_3d[mid_idx, :, :]
            xlabel, ylabel = 'y [m]', 'z [m]'
        else:
            raise ValueError(f"plane must be 'X', 'Y', or 'Z', got '{plane}'")

        extent = [self.coords_1d[0], self.coords_1d[-1],
                  self.coords_1d[0], self.coords_1d[-1]]

        # Compute error
        error_plane = C_comp_plane - C_anal_plane

        # Plot computed
        im0 = axes[0].imshow(
            C_comp_plane.T, extent=extent, origin='lower',
            cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal'
        )
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        axes[0].set_title('Computed')
        plt.colorbar(im0, ax=axes[0])

        # Plot analytical
        im1 = axes[1].imshow(
            C_anal_plane.T, extent=extent, origin='lower',
            cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal'
        )
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel(ylabel)
        axes[1].set_title('Analytical sinc(kr)')
        plt.colorbar(im1, ax=axes[1])

        # Plot error
        max_err = np.max(np.abs(error_plane))
        im2 = axes[2].imshow(
            error_plane.T, extent=extent, origin='lower',
            cmap='RdBu_r', vmin=-max_err, vmax=max_err, aspect='equal'
        )
        axes[2].set_xlabel(xlabel)
        axes[2].set_ylabel(ylabel)
        axes[2].set_title('Error')
        plt.colorbar(im2, ax=axes[2])

        # Compute NMSE
        nmse = np.sum((C_real - C_analytical) ** 2) / np.sum(C_analytical ** 2)
        fig.suptitle(f'Centered Correlation at f = {f:.1f} Hz ({plane}-plane), NMSE = {nmse:.2e}')
        fig.tight_layout()

        return fig

    def plot_nmse_vs_frequency(self, ax: plt.Axes = None) -> plt.Axes:
        """
        Plot NMSE across all frequencies.

        Parameters
        ----------
        ax : plt.Axes, optional
            Axes to plot on. If None, creates new figure.

        Returns
        -------
        plt.Axes
            The axes with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Compute all frequencies
        results = self.compute_all_frequencies()

        ax.semilogy(results['frequencies'], results['nmse'], 'bo-', linewidth=2, markersize=6)

        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Normalized MSE')
        ax.set_title('NMSE vs Frequency')
        ax.grid(True, alpha=0.3, which='both')

        return ax

    def plot_radial_profile(
        self,
        freq_idx: int,
        ax: plt.Axes = None
    ) -> plt.Axes:
        """
        Plot the radial profile of computed vs analytical correlation.

        Parameters
        ----------
        freq_idx : int
            Index into the frequencies array
        ax : plt.Axes, optional
            Axes to plot on. If None, creates new figure.

        Returns
        -------
        plt.Axes
            The axes with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        k = self.wavenumbers[freq_idx]
        f = self.frequencies[freq_idx]

        # Compute centered correlation
        C = self.compute_centered_correlation(freq_idx)
        C_real = np.real(C)

        # Compute distances from center
        r = np.linalg.norm(self.grid_points, axis=1)

        # Compute analytical correlation
        C_analytical = self.analytical_correlation(r, k)

        # Sort by distance for plotting
        sort_idx = np.argsort(r)
        r_sorted = r[sort_idx]
        C_comp_sorted = C_real[sort_idx]
        C_anal_sorted = C_analytical[sort_idx]

        # Plot
        ax.plot(r_sorted, C_anal_sorted, 'b-', linewidth=2, label='Analytical sinc(kr)')
        ax.plot(r_sorted, C_comp_sorted, 'r.', markersize=1, alpha=0.3, label='Computed')

        ax.set_xlabel('Distance r [m]')
        ax.set_ylabel('Correlation')
        ax.set_title(f'Radial Correlation Profile at f = {f:.1f} Hz')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        return ax

    # =========================================================================
    # Matrix-free eigenvalue computation methods
    # =========================================================================

    def _collect_realizations(self, freq_idx: int) -> np.ndarray:
        """
        Collect pressure field realizations for eigenvalue computation.

        Parameters
        ----------
        freq_idx : int
            Index into the frequencies array

        Returns
        -------
        ndarray
            P_realizations of shape (M, R) where each column is a realization
        """
        k = self.wavenumbers[freq_idx]
        M = len(self.grid_points)
        R = self.n_realizations

        P_realizations = np.zeros((M, R), dtype=np.complex128)

        for r in range(R):
            P_realizations[:, r] = self._compute_pressure_field(k)

        return P_realizations

    def _covariance_matvec(
        self,
        P_realizations: np.ndarray,
        v: np.ndarray
    ) -> np.ndarray:
        """
        Matrix-free matrix-vector product C @ v.

        Computes C @ v = (1/R) * sum_r P_r * (P_r^H @ v) without forming C.

        Parameters
        ----------
        P_realizations : ndarray
            Pressure field realizations of shape (M, R)
        v : ndarray
            Vector to multiply, shape (M,) or (M, k)

        Returns
        -------
        ndarray
            Result C @ v with same shape as v
        """
        R = P_realizations.shape[1]

        if v.ndim == 1:
            # Single vector: C @ v = (1/R) * P @ (P^H @ v)
            alphas = P_realizations.conj().T @ v  # (R,)
            result = P_realizations @ alphas       # (M,)
        else:
            # Matrix: C @ V = (1/R) * P @ (P^H @ V)
            alphas = P_realizations.conj().T @ v  # (R, k)
            result = P_realizations @ alphas       # (M, k)

        return result / R

    def compute_covariance_eigenvalues(
        self,
        freq_idx: int,
        n_components: int = 10,
        n_oversamples: int = 10,
        n_power_iter: int = 2,
        random_state: int = None
    ) -> tuple:
        """
        Compute top eigenvalues and eigenvectors of covariance matrix C = E[PP^H].

        Uses randomized SVD algorithm for matrix-free computation. The covariance
        matrix is never explicitly formed.

        Parameters
        ----------
        freq_idx : int
            Index into the frequencies array
        n_components : int, optional
            Number of eigenvalues/eigenvectors to compute, default 10
        n_oversamples : int, optional
            Additional random vectors for improved accuracy, default 10
        n_power_iter : int, optional
            Number of power iterations for improved accuracy, default 2
        random_state : int, optional
            Random seed for reproducibility

        Returns
        -------
        eigenvalues : ndarray
            Top eigenvalues in descending order, shape (n_components,)
        eigenvectors : ndarray
            Corresponding eigenvectors as columns, shape (M, n_components)
        """
        rng = np.random.default_rng(random_state)
        M = len(self.grid_points)
        k = n_components + n_oversamples

        print(f"Collecting {self.n_realizations} realizations...")
        P_realizations = self._collect_realizations(freq_idx)

        print(f"Computing {n_components} eigenvalues using randomized SVD...")

        # Stage 1: Find approximate range of C
        # Generate random test matrix (complex Gaussian)
        Omega = (rng.standard_normal((M, k)) +
                 1j * rng.standard_normal((M, k))) / np.sqrt(2)

        # Form Y = C @ Omega
        Y = self._covariance_matvec(P_realizations, Omega)

        # Power iteration to improve range approximation
        for i in range(n_power_iter):
            # Orthonormalize for numerical stability
            Y, _ = qr(Y, mode='economic')
            # Apply C twice (C is Hermitian, so C² has same eigenvectors)
            Y = self._covariance_matvec(P_realizations, Y)
            Y = self._covariance_matvec(P_realizations, Y)

        # Orthonormal basis for range of Y
        Q, _ = qr(Y, mode='economic')

        # Stage 2: Form small projected matrix and compute its eigendecomposition
        # B = Q^H @ C @ Q is a k×k matrix
        CQ = self._covariance_matvec(P_realizations, Q)
        B = Q.conj().T @ CQ

        # Force exact Hermitian symmetry
        B = (B + B.conj().T) / 2

        # Eigendecomposition of small matrix
        eigvals_B, eigvecs_B = np.linalg.eigh(B)

        # Sort in descending order
        idx = np.argsort(eigvals_B)[::-1]
        eigvals_B = eigvals_B[idx]
        eigvecs_B = eigvecs_B[:, idx]

        # Stage 3: Recover eigenvectors of original matrix
        eigenvalues = eigvals_B[:n_components]
        eigenvectors = Q @ eigvecs_B[:, :n_components]

        return eigenvalues, eigenvectors

    def plot_eigenvalue_decay(
        self,
        freq_idx: int,
        n_components: int = 20,
        ax: plt.Axes = None,
        **kwargs
    ) -> plt.Axes:
        """
        Plot eigenvalue decay of the covariance matrix.

        Parameters
        ----------
        freq_idx : int
            Index into the frequencies array
        n_components : int, optional
            Number of eigenvalues to compute, default 20
        ax : plt.Axes, optional
            Axes to plot on. If None, creates new figure.
        **kwargs
            Additional arguments passed to compute_covariance_eigenvalues

        Returns
        -------
        plt.Axes
            The axes with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        f = self.frequencies[freq_idx]

        # Compute eigenvalues
        eigenvalues, _ = self.compute_covariance_eigenvalues(
            freq_idx, n_components=n_components, **kwargs
        )

        # Normalize by largest eigenvalue
        eigenvalues_norm = eigenvalues / eigenvalues[0]

        # Plot
        indices = np.arange(1, len(eigenvalues) + 1)
        ax.semilogy(indices, eigenvalues_norm, 'bo-', linewidth=2, markersize=8)

        ax.set_xlabel('Eigenvalue Index')
        ax.set_ylabel(r'Normalized Eigenvalue $\lambda_i / \lambda_1$')
        ax.set_title(f'Covariance Eigenvalue Decay at f = {f:.1f} Hz')
        ax.grid(True, alpha=0.3, which='both')

        # Add cumulative energy
        cumulative = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        ax2 = ax.twinx()
        ax2.plot(indices, cumulative, 'r--', linewidth=2, label='Cumulative energy')
        ax2.set_ylabel('Cumulative Energy Fraction', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim([0, 1.05])

        # Find 99% energy threshold
        idx_99 = np.searchsorted(cumulative, 0.99) + 1
        ax.axvline(x=idx_99, color='g', linestyle=':', alpha=0.7,
                   label=f'99% energy at n={idx_99}')
        ax.legend(loc='center right')

        return ax

    def plot_eigenvectors(
        self,
        eigenvectors: np.ndarray,
        eigenvalues: np.ndarray,
        n_vectors: int = 4,
        plane: str = 'Z',
        component: str = 'magnitude'
    ) -> plt.Figure:
        """
        Visualize eigenvector spatial patterns on 2D slices through the domain center.

        Parameters
        ----------
        eigenvectors : ndarray
            Eigenvectors as columns, shape (M, n_components)
        eigenvalues : ndarray
            Corresponding eigenvalues, shape (n_components,)
        n_vectors : int, optional
            Number of eigenvectors to visualize, default 4
        plane : str, optional
            Plane to slice: 'X', 'Y', or 'Z' (default 'Z')
        component : str, optional
            What to plot: 'magnitude', 'real', 'imag', or 'phase'

        Returns
        -------
        plt.Figure
            Figure with eigenvector subplots
        """
        plane = plane.upper()
        n_vectors = min(n_vectors, eigenvectors.shape[1])

        # Determine grid layout
        n_cols = min(4, n_vectors)
        n_rows = (n_vectors + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        if n_vectors == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Get slice indices at center
        nx, ny, nz = self.grid_shape
        center_idx = {'X': nx // 2, 'Y': ny // 2, 'Z': nz // 2}

        # Get coordinate extent (same for all axes since grid is cubic)
        coord_min = self.coords_1d[0]
        coord_max = self.coords_1d[-1]

        for idx in range(n_vectors):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            # Reshape eigenvector to 3D grid
            evec = eigenvectors[:, idx].reshape(self.grid_shape)

            # Extract appropriate slice
            if plane == 'X':
                slice_2d = evec[center_idx['X'], :, :]
                xlabel, ylabel = 'Y [m]', 'Z [m]'
                extent = [coord_min, coord_max, coord_min, coord_max]
            elif plane == 'Y':
                slice_2d = evec[:, center_idx['Y'], :]
                xlabel, ylabel = 'X [m]', 'Z [m]'
                extent = [coord_min, coord_max, coord_min, coord_max]
            else:  # Z
                slice_2d = evec[:, :, center_idx['Z']]
                xlabel, ylabel = 'X [m]', 'Y [m]'
                extent = [coord_min, coord_max, coord_min, coord_max]

            # Compute requested component
            if component == 'magnitude':
                data = np.abs(slice_2d)
                cmap = 'viridis'
            elif component == 'real':
                data = np.real(slice_2d)
                cmap = 'RdBu_r'
            elif component == 'imag':
                data = np.imag(slice_2d)
                cmap = 'RdBu_r'
            elif component == 'phase':
                data = np.angle(slice_2d)
                cmap = 'hsv'
            else:
                raise ValueError(f"Unknown component: {component}")

            # Plot
            im = ax.imshow(
                data.T,
                origin='lower',
                extent=extent,
                aspect='equal',
                cmap=cmap
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'Mode {idx + 1} (λ={eigenvalues[idx]:.2e})')
            plt.colorbar(im, ax=ax, shrink=0.8)

        # Hide unused subplots
        for idx in range(n_vectors, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)

        fig.suptitle(f'Covariance Eigenvectors ({component.capitalize()}, {plane}-slice)',
                     fontsize=12, y=1.02)
        fig.tight_layout()

        return fig

    def __repr__(self) -> str:
        return (
            f"DiffuseField3D(\n"
            f"  f_min={self.f_min} Hz, f_max={self.f_max} Hz, n_freq={self.n_freq}\n"
            f"  c={self.c} m/s, n_waves={self.n_waves}, n_realizations={self.n_realizations}\n"
            f"  box_size={self.box_size:.3f} m, grid_spacing={self.grid_spacing:.4f} m\n"
            f"  grid_points={len(self.grid_points)}, grid_shape={self.grid_shape}\n"
            f")"
        )
