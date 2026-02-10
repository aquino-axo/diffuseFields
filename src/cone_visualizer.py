"""
Visualization class for cone surface pressure fields.

Provides 3D plotting capabilities for pressure fields on conical surfaces,
supporting magnitude, real, imaginary, and phase components.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Optional, List, Union


class ConeVisualizer:
    """
    Visualization class for cone surface pressure fields.

    This class handles all plotting and visualization for pressure fields
    on conical surfaces, decoupled from the computation logic.

    Parameters
    ----------
    coordinates : ndarray
        Node coordinates on cone surface, shape (ndof, 3).
    cone_geometry : dict, optional
        Cone geometry parameters with keys:
        - 'half_angle': half-angle of the cone in radians
        - 'height': height of the cone
        The base_diameter is computed as 2 * height * tan(half_angle).
        If None, geometry metadata is not stored.

    Attributes
    ----------
    ndof : int
        Number of nodes on the cone surface.
    """

    def __init__(
        self,
        coordinates: np.ndarray,
        cone_geometry: Optional[Dict[str, float]] = None
    ):
        self.coordinates = np.asarray(coordinates, dtype=np.float64)
        self.cone_geometry = dict(cone_geometry) if cone_geometry else None

        self._validate_inputs()

        # Compute base_diameter if geometry provided and not already present
        if self.cone_geometry and 'base_diameter' not in self.cone_geometry:
            self.cone_geometry['base_diameter'] = (
                2 * self.cone_geometry['height'] *
                np.tan(self.cone_geometry['half_angle'])
            )

        self.ndof = self.coordinates.shape[0]

    def _validate_inputs(self) -> None:
        """Validate input dimensions and parameters."""
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 3:
            raise ValueError(
                f"coordinates must have shape (ndof, 3), "
                f"got {self.coordinates.shape}"
            )

        if self.cone_geometry:
            required_keys = {'half_angle', 'height'}
            missing = required_keys - set(self.cone_geometry.keys())
            if missing:
                raise ValueError(f"cone_geometry missing keys: {missing}")

    def _extract_component(
        self,
        values: np.ndarray,
        component: str
    ) -> tuple:
        """
        Extract the specified component from complex values.

        Parameters
        ----------
        values : ndarray
            Complex-valued array.
        component : str
            Component to extract: 'magnitude', 'real', 'imag', or 'phase'.

        Returns
        -------
        data : ndarray
            Extracted component values.
        label : str
            Label for the colorbar.
        """
        if component == 'magnitude':
            data = np.abs(values)
            label = 'Magnitude'
        elif component == 'real':
            data = np.real(values)
            label = 'Real part'
        elif component == 'imag':
            data = np.imag(values)
            label = 'Imaginary part'
        elif component == 'phase':
            data = np.angle(values)
            label = 'Phase (rad)'
        else:
            raise ValueError(
                f"component must be 'magnitude', 'real', 'imag', or 'phase', "
                f"got '{component}'"
            )
        return data, label

    def _set_equal_aspect(self, ax: Axes3D) -> None:
        """Set equal aspect ratio for 3D axes."""
        x = self.coordinates[:, 0]
        y = self.coordinates[:, 1]
        z = self.coordinates[:, 2]

        max_range = np.array([
            x.max() - x.min(),
            y.max() - y.min(),
            z.max() - z.min()
        ]).max() / 2.0

        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    def plot_pressure_field(
        self,
        pressure: np.ndarray,
        component: str = 'magnitude',
        ax: Optional[Axes3D] = None,
        cmap: str = 'viridis',
        title: Optional[str] = None
    ) -> Axes3D:
        """
        Plot a pressure field on the 3D cone surface.

        Parameters
        ----------
        pressure : ndarray
            Complex pressure values at each node, shape (ndof,).
        component : str, optional
            Component to plot: 'magnitude', 'real', 'imag', or 'phase'.
            Default is 'magnitude'.
        ax : Axes3D, optional
            3D matplotlib axes to plot on. If None, creates new figure.
        cmap : str, optional
            Colormap name, default 'viridis'.
        title : str, optional
            Plot title. If None, no title is set.

        Returns
        -------
        Axes3D
            The 3D axes with the plot.
        """
        pressure = np.asarray(pressure)
        if pressure.shape[0] != self.ndof:
            raise ValueError(
                f"pressure must have {self.ndof} values, got {pressure.shape[0]}"
            )

        # Extract component
        values, label = self._extract_component(pressure, component)

        # Create 3D axes if needed
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        # Extract coordinates
        x = self.coordinates[:, 0]
        y = self.coordinates[:, 1]
        z = self.coordinates[:, 2]

        # Scatter plot colored by pressure values
        scatter = ax.scatter(x, y, z, c=values, cmap=cmap, s=20)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label(label)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if title is not None:
            ax.set_title(title)

        # Set equal aspect ratio
        self._set_equal_aspect(ax)

        return ax

    def plot_multiple_fields(
        self,
        pressures: np.ndarray,
        labels: Optional[List[str]] = None,
        component: str = 'magnitude',
        cmap: str = 'viridis'
    ) -> plt.Figure:
        """
        Plot multiple pressure fields in a grid layout.

        Parameters
        ----------
        pressures : ndarray
            Pressure fields as columns, shape (ndof, n_fields).
        labels : list of str, optional
            Labels for each field. If None, uses "Field 1", "Field 2", etc.
        component : str, optional
            Component to plot: 'magnitude', 'real', 'imag', or 'phase'.
        cmap : str, optional
            Colormap name.

        Returns
        -------
        plt.Figure
            The figure with the plots.
        """
        pressures = np.asarray(pressures)
        if pressures.ndim == 1:
            pressures = pressures.reshape(-1, 1)

        n_fields = pressures.shape[1]

        if labels is None:
            labels = [f'Field {i + 1}' for i in range(n_fields)]

        # Determine grid layout
        ncols = min(2, n_fields)
        nrows = (n_fields + ncols - 1) // ncols

        fig = plt.figure(figsize=(6 * ncols, 5 * nrows))

        for i in range(n_fields):
            ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
            self.plot_pressure_field(
                pressures[:, i],
                component=component,
                ax=ax,
                cmap=cmap,
                title=labels[i]
            )

        plt.tight_layout()
        return fig

    def plot_variance_explained(
        self,
        eigenvalues: np.ndarray,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        n_components_kept: Optional[int] = None
    ) -> plt.Axes:
        """
        Plot variance explained versus number of eigenvalues.

        Parameters
        ----------
        eigenvalues : ndarray
            Eigenvalues in descending order (all eigenvalues).
        ax : plt.Axes, optional
            Axes to plot on. If None, creates new figure.
        title : str, optional
            Plot title. If None, uses default.
        n_components_kept : int, optional
            Number of eigenvalues/eigenvectors retained. If provided,
            a vertical line marks the cutoff threshold.

        Returns
        -------
        plt.Axes
            The axes with the plot.
        """
        eigenvalues = np.asarray(eigenvalues)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        indices = np.arange(1, len(eigenvalues) + 1)

        # Compute cumulative variance with respect to ALL eigenvalues
        total = np.sum(eigenvalues)
        cumulative_variance = np.cumsum(eigenvalues) / total

        # Normalized eigenvalues
        eigenvalues_norm = eigenvalues / eigenvalues[0]

        # Plot eigenvalue decay
        ax.semilogy(
            indices, eigenvalues_norm, 'bo-', linewidth=2, markersize=6,
            label='Normalized eigenvalue'
        )

        # Add cumulative variance on secondary axis
        ax2 = ax.twinx()
        ax2.plot(
            indices, cumulative_variance, 'r--', linewidth=2,
            label='Cumulative variance'
        )
        ax2.set_ylabel('Cumulative Variance Explained', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim([0, 1.05])

        # Add vertical line at cutoff threshold if provided
        if n_components_kept is not None and n_components_kept < len(eigenvalues):
            var_at_cutoff = cumulative_variance[n_components_kept - 1]
            ax.axvline(
                x=n_components_kept, color='green', linestyle=':', linewidth=2,
                label=f'Cutoff (n={n_components_kept}, var={var_at_cutoff:.2%})'
            )

        ax.set_xlabel('Eigenvalue Index')
        ax.set_ylabel(r'Normalized Eigenvalue $\lambda_i / \lambda_1$', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.grid(True, alpha=0.3, which='both')

        if title is None:
            title = 'Variance Explained'
        ax.set_title(title)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')

        return ax
