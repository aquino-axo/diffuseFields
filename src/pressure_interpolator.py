"""
Pressure field interpolation between surfaces.

Interpolates complex-valued pressure fields from a source surface
to a target surface using RBF (Radial Basis Function) interpolation.
"""

import numpy as np
from typing import Dict, Any, Optional
from scipy.interpolate import RBFInterpolator


class PressureFieldInterpolator:
    """
    Interpolates pressure fields from source to target surface.

    Uses scipy.interpolate.RBFInterpolator which performs kernel-based
    interpolation without relying on convex hull boundaries.
    Complex values are handled by interpolating real and imaginary parts separately.

    Parameters
    ----------
    source_coords : ndarray
        Source surface coordinates, shape (n_source, 3).
    target_coords : ndarray
        Target surface coordinates, shape (n_target, 3).
    kernel : str, optional
        RBF kernel type. Options: 'thin_plate_spline' (default), 'multiquadric',
        'inverse_multiquadric', 'inverse_quadratic', 'gaussian', 'linear', 'cubic'.
    smoothing : float, optional
        Smoothing parameter. 0.0 (default) means exact interpolation at source points.

    Attributes
    ----------
    n_source : int
        Number of source points.
    n_target : int
        Number of target points.
    extrapolation_mask : ndarray or None
        Boolean mask (always all False for RBF since it can extrapolate).
        Available after calling interpolate().
    """

    def __init__(
        self,
        source_coords: np.ndarray,
        target_coords: np.ndarray,
        kernel: str = 'thin_plate_spline',
        smoothing: float = 0.0
    ):
        self.source_coords = np.asarray(source_coords, dtype=np.float64)
        self.target_coords = np.asarray(target_coords, dtype=np.float64)
        self._kernel = kernel
        self._smoothing = smoothing

        self._validate_inputs()

        self.n_source = self.source_coords.shape[0]
        self.n_target = self.target_coords.shape[0]
        self.extrapolation_mask: Optional[np.ndarray] = None

    def _validate_inputs(self) -> None:
        """Validate input coordinates dimensions and types."""
        if self.source_coords.ndim != 2 or self.source_coords.shape[1] != 3:
            raise ValueError(
                f"source_coords must have shape (n_source, 3), "
                f"got {self.source_coords.shape}"
            )
        if self.target_coords.ndim != 2 or self.target_coords.shape[1] != 3:
            raise ValueError(
                f"target_coords must have shape (n_target, 3), "
                f"got {self.target_coords.shape}"
            )
        if self.source_coords.shape[0] < 1:
            raise ValueError(
                "source_coords must have at least 1 point"
            )

    def interpolate(self, pressure_fields: np.ndarray) -> np.ndarray:
        """
        Interpolate pressure field(s) from source to target coordinates.

        Parameters
        ----------
        pressure_fields : ndarray
            Complex pressure field(s) at source coordinates.
            Shape (n_source,) for single field or (n_source, n_fields) for batch.

        Returns
        -------
        interpolated : ndarray
            Interpolated pressure field(s) at target coordinates.
            Shape (n_target,) or (n_target, n_fields).
        """
        pressure_fields = np.asarray(pressure_fields, dtype=np.complex128)

        # Validate shape
        if pressure_fields.shape[0] != self.n_source:
            raise ValueError(
                f"pressure_fields must have {self.n_source} rows, "
                f"got {pressure_fields.shape[0]}"
            )

        # Handle 1D vs 2D input
        squeeze_output = False
        if pressure_fields.ndim == 1:
            pressure_fields = pressure_fields.reshape(-1, 1)
            squeeze_output = True

        n_fields = pressure_fields.shape[1]
        result = np.zeros((self.n_target, n_fields), dtype=np.complex128)

        # Interpolate each field using RBF
        for i in range(n_fields):
            real_part = np.real(pressure_fields[:, i])
            imag_part = np.imag(pressure_fields[:, i])

            # Create RBF interpolators
            rbf_real = RBFInterpolator(
                self.source_coords,
                real_part,
                kernel=self._kernel,
                smoothing=self._smoothing
            )
            rbf_imag = RBFInterpolator(
                self.source_coords,
                imag_part,
                kernel=self._kernel,
                smoothing=self._smoothing
            )

            # Interpolate
            result[:, i] = rbf_real(self.target_coords) + 1j * rbf_imag(self.target_coords)

        # RBF always returns values (no convex hull limitation)
        # Set extrapolation_mask to all False
        self.extrapolation_mask = np.zeros(self.n_target, dtype=bool)

        if squeeze_output:
            return result.squeeze()
        return result

    def get_extrapolation_info(self) -> Dict[str, Any]:
        """
        Get information about extrapolation.

        Note: RBF interpolation can extrapolate to any point, so
        n_extrapolated will always be 0.

        Returns
        -------
        info : dict
            Dictionary with keys:
            - 'n_extrapolated': always 0 for RBF
            - 'extrapolation_ratio': always 0.0 for RBF
            - 'extrapolation_mask': boolean mask (all False)

        Raises
        ------
        RuntimeError
            If called before interpolate().
        """
        if self.extrapolation_mask is None:
            raise RuntimeError(
                "Extrapolation info not available. Call interpolate() first."
            )

        n_extrapolated = np.sum(self.extrapolation_mask)
        return {
            'n_extrapolated': int(n_extrapolated),
            'extrapolation_ratio': float(n_extrapolated / self.n_target),
            'extrapolation_mask': self.extrapolation_mask.copy()
        }

    @staticmethod
    def from_files(
        source_coords_path: str,
        target_coords_path: str,
        kernel: str = 'thin_plate_spline',
        smoothing: float = 0.0
    ) -> 'PressureFieldInterpolator':
        """
        Create interpolator from .npy coordinate files.

        Parameters
        ----------
        source_coords_path : str
            Path to source coordinates .npy file.
        target_coords_path : str
            Path to target coordinates .npy file.
        kernel : str, optional
            RBF kernel type.
        smoothing : float, optional
            Smoothing parameter.

        Returns
        -------
        PressureFieldInterpolator
            Configured interpolator instance.
        """
        source_coords = np.load(source_coords_path)
        target_coords = np.load(target_coords_path)
        return PressureFieldInterpolator(
            source_coords, target_coords, kernel=kernel, smoothing=smoothing
        )
