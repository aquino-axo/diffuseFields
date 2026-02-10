"""
Total pressure field computation from incident and scattered fields.

Computes total pressure field as sum of incident plane waves and scattered field:
    P_total = P_incident + P_scattered

The incident field at position x is:
    P_inc(x) = Σ_j A_j * exp(i * k * d_j · x)

where:
    A_j = complex amplitude for plane wave j
    k = wavenumber = 2πf/c
    d_j = unit direction vector for plane wave j
    x = position vector
"""

import numpy as np
from typing import Optional

from directions_parser import DirectionsParser
from amplitudes_parser import AmplitudesParser


class TotalPressureField:
    """
    Computes total pressure field from incident and scattered components.

    Parameters
    ----------
    coordinates : ndarray, shape (n_nodes, 3)
        Node coordinates where field is computed.
    directions : ndarray, shape (n_pws, 3)
        Unit direction vectors for each plane wave.
    amplitudes : ndarray, shape (n_pws,)
        Complex velocity amplitudes for each plane wave (m/s).
    frequency : float
        Frequency in Hz.
    speed_of_sound : float, optional
        Speed of sound in m/s. Default 343.0.
    density : float, optional
        Fluid density in kg/m³. Default 1.21 (air at 20°C).

    Examples
    --------
    >>> field = TotalPressureField(coords, directions, amplitudes, 1000.0)
    >>> P_inc = field.compute_incident_field()
    >>> P_total = field.compute_total_field(scat_real, scat_imag)
    """

    def __init__(
        self,
        coordinates: np.ndarray,
        directions: np.ndarray,
        amplitudes: np.ndarray,
        frequency: float,
        speed_of_sound: float = 343.0,
        density: float = 1.21
    ):
        self.coordinates = np.asarray(coordinates)
        self.directions = np.asarray(directions)
        self.amplitudes = np.asarray(amplitudes, dtype=np.complex128)
        self.frequency = frequency
        self.speed_of_sound = speed_of_sound
        self.density = density

        # Validate shapes
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 3:
            raise ValueError(
                f"coordinates must have shape (n_nodes, 3), "
                f"got {self.coordinates.shape}"
            )
        if self.directions.ndim != 2 or self.directions.shape[1] != 3:
            raise ValueError(
                f"directions must have shape (n_pws, 3), "
                f"got {self.directions.shape}"
            )
        if self.amplitudes.ndim != 1:
            raise ValueError(
                f"amplitudes must be 1D, got shape {self.amplitudes.shape}"
            )
        if len(self.directions) != len(self.amplitudes):
            raise ValueError(
                f"Number of directions ({len(self.directions)}) must match "
                f"number of amplitudes ({len(self.amplitudes)})"
            )

    @property
    def wavenumber(self) -> float:
        """Return wavenumber k = 2*pi*f/c."""
        return 2 * np.pi * self.frequency / self.speed_of_sound

    @property
    def impedance(self) -> float:
        """Return characteristic acoustic impedance rho*c (Pa*s/m)."""
        return self.density * self.speed_of_sound

    @property
    def n_nodes(self) -> int:
        """Return number of nodes."""
        return len(self.coordinates)

    @property
    def n_plane_waves(self) -> int:
        """Return number of plane waves."""
        return len(self.directions)

    def compute_incident_field(self) -> np.ndarray:
        """
        Compute incident pressure field at all nodes.

        The incident field is computed as:
            P_inc[i] = rho*c * sum_j [ v_j * exp(i * k * d_j · x_i) ]

        where v_j are velocity amplitudes and rho*c is the impedance.

        Returns
        -------
        P_inc : ndarray, shape (n_nodes,), complex
            Incident pressure field at each node (Pa).
        """
        k = self.wavenumber
        # Compute dot products: (n_nodes, n_pws)
        dot_products = self.coordinates @ self.directions.T
        # Phase factors: exp(i * k * d·x)
        phase = np.exp(1j * k * dot_products)
        # Sum over plane waves: (n_nodes, n_pws) @ (n_pws,) -> (n_nodes,)
        # Multiply by impedance to convert velocity to pressure
        P_inc = self.impedance * (phase @ self.amplitudes)
        return P_inc

    def compute_total_field(
        self,
        scattered_real: np.ndarray,
        scattered_imag: np.ndarray
    ) -> np.ndarray:
        """
        Compute total field = incident + scattered.

        Parameters
        ----------
        scattered_real : ndarray, shape (n_nodes,)
            Real part of scattered field.
        scattered_imag : ndarray, shape (n_nodes,)
            Imaginary part of scattered field.

        Returns
        -------
        P_total : ndarray, shape (n_nodes,), complex
            Total pressure field at each node.
        """
        scattered_real = np.asarray(scattered_real)
        scattered_imag = np.asarray(scattered_imag)

        if scattered_real.shape != (self.n_nodes,):
            raise ValueError(
                f"scattered_real must have shape ({self.n_nodes},), "
                f"got {scattered_real.shape}"
            )
        if scattered_imag.shape != (self.n_nodes,):
            raise ValueError(
                f"scattered_imag must have shape ({self.n_nodes},), "
                f"got {scattered_imag.shape}"
            )

        P_scat = scattered_real + 1j * scattered_imag
        P_inc = self.compute_incident_field()

        return P_inc + P_scat

    def set_frequency(self, frequency: float) -> None:
        """
        Update the frequency for field computation.

        Parameters
        ----------
        frequency : float
            New frequency in Hz.
        """
        self.frequency = frequency

    @classmethod
    def from_parsed_files(
        cls,
        coordinates: np.ndarray,
        directions_parser: DirectionsParser,
        amplitudes_parser: AmplitudesParser,
        frequency: float,
        speed_of_sound: float = 343.0,
        density: float = 1.21
    ) -> 'TotalPressureField':
        """
        Factory method to create from parsed file data.

        Parameters
        ----------
        coordinates : ndarray, shape (n_nodes, 3)
            Node coordinates.
        directions_parser : DirectionsParser
            Parser with plane wave directions.
        amplitudes_parser : AmplitudesParser
            Parser with complex amplitudes.
        frequency : float
            Frequency in Hz.
        speed_of_sound : float, optional
            Speed of sound in m/s. Default 343.0.
        density : float, optional
            Fluid density in kg/m³. Default 1.21 (air).

        Returns
        -------
        TotalPressureField
            Configured field calculator.
        """
        directions = directions_parser.get_directions()
        amplitudes = amplitudes_parser.get_complex_amplitudes()

        if len(directions) != len(amplitudes):
            raise ValueError(
                f"Number of directions ({len(directions)}) from directions "
                f"file does not match number of amplitudes ({len(amplitudes)}) "
                f"from amplitudes file"
            )

        return cls(
            coordinates=coordinates,
            directions=directions,
            amplitudes=amplitudes,
            frequency=frequency,
            speed_of_sound=speed_of_sound,
            density=density
        )
