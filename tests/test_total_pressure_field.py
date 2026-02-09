"""Tests for TotalPressureField class."""

import numpy as np
import pytest
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from total_pressure_field import TotalPressureField


class TestTotalPressureField:
    """Tests for TotalPressureField."""

    def test_single_plane_wave(self):
        """Test incident field for a single plane wave."""
        # Single node at origin
        coords = np.array([[0.0, 0.0, 0.0]])
        # Single plane wave in +x direction
        directions = np.array([[1.0, 0.0, 0.0]])
        # Unit amplitude
        amplitudes = np.array([1.0 + 0j])
        frequency = 1000.0
        c = 343.0

        field = TotalPressureField(
            coords, directions, amplitudes, frequency, c
        )

        P_inc = field.compute_incident_field()

        # At origin, d·x = 0, so P_inc = A * exp(0) = A = 1
        np.testing.assert_almost_equal(P_inc[0], 1.0 + 0j)

    def test_plane_wave_at_distance(self):
        """Test incident field at a distance from origin."""
        # Node at x = wavelength/4
        c = 343.0
        f = 1000.0
        wavelength = c / f
        x = wavelength / 4

        coords = np.array([[x, 0.0, 0.0]])
        directions = np.array([[1.0, 0.0, 0.0]])
        amplitudes = np.array([1.0 + 0j])

        field = TotalPressureField(coords, directions, amplitudes, f, c)
        P_inc = field.compute_incident_field()

        # At x = lambda/4, k*x = 2*pi/lambda * lambda/4 = pi/2
        # P_inc = exp(i*pi/2) = i
        expected = np.exp(1j * np.pi / 2)
        np.testing.assert_almost_equal(P_inc[0], expected)

    def test_wavenumber_computation(self):
        """Test that wavenumber is computed correctly."""
        coords = np.array([[0.0, 0.0, 0.0]])
        directions = np.array([[1.0, 0.0, 0.0]])
        amplitudes = np.array([1.0 + 0j])
        frequency = 1000.0
        c = 343.0

        field = TotalPressureField(
            coords, directions, amplitudes, frequency, c
        )

        expected_k = 2 * np.pi * frequency / c
        np.testing.assert_almost_equal(field.wavenumber, expected_k)

    def test_total_field_addition(self):
        """Test that total field = incident + scattered."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        directions = np.array([[1.0, 0.0, 0.0]])
        amplitudes = np.array([1.0 + 0j])

        field = TotalPressureField(coords, directions, amplitudes, 1000.0, 343.0)

        P_inc = field.compute_incident_field()

        # Arbitrary scattered field
        scat_real = np.array([0.5, -0.3])
        scat_imag = np.array([0.2, 0.7])

        P_total = field.compute_total_field(scat_real, scat_imag)

        # Check addition
        P_scat = scat_real + 1j * scat_imag
        expected = P_inc + P_scat
        np.testing.assert_array_almost_equal(P_total, expected)

    def test_multiple_plane_waves_superposition(self):
        """Test superposition of multiple plane waves."""
        # Node at origin
        coords = np.array([[0.0, 0.0, 0.0]])

        # Two plane waves with different directions
        directions = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        # Different amplitudes
        amplitudes = np.array([1.0 + 0j, 0.5 + 0.5j])

        field = TotalPressureField(coords, directions, amplitudes, 1000.0, 343.0)
        P_inc = field.compute_incident_field()

        # At origin, all exp(ik*d·x) = exp(0) = 1
        # P_inc = sum(amplitudes) = 1.0 + (0.5 + 0.5j) = 1.5 + 0.5j
        expected = 1.5 + 0.5j
        np.testing.assert_almost_equal(P_inc[0], expected)

    def test_set_frequency(self):
        """Test updating frequency."""
        coords = np.array([[0.0, 0.0, 0.0]])
        directions = np.array([[1.0, 0.0, 0.0]])
        amplitudes = np.array([1.0 + 0j])

        field = TotalPressureField(coords, directions, amplitudes, 1000.0, 343.0)

        k1 = field.wavenumber
        field.set_frequency(2000.0)
        k2 = field.wavenumber

        # k should double when frequency doubles
        np.testing.assert_almost_equal(k2, 2 * k1)

    def test_shape_validation(self):
        """Test that invalid shapes raise errors."""
        coords = np.array([[0.0, 0.0, 0.0]])
        directions = np.array([[1.0, 0.0, 0.0]])
        amplitudes = np.array([1.0 + 0j])

        field = TotalPressureField(coords, directions, amplitudes, 1000.0, 343.0)

        # Wrong shape for scattered fields
        wrong_real = np.array([0.5, 0.3])  # Should be (1,) not (2,)
        wrong_imag = np.array([0.2])

        with pytest.raises(ValueError, match="scattered_real must have shape"):
            field.compute_total_field(wrong_real, wrong_imag)

    def test_mismatched_directions_amplitudes(self):
        """Test that mismatched counts raise error."""
        coords = np.array([[0.0, 0.0, 0.0]])
        directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # 2 directions
        amplitudes = np.array([1.0 + 0j])  # 1 amplitude

        with pytest.raises(ValueError, match="Number of directions"):
            TotalPressureField(coords, directions, amplitudes, 1000.0, 343.0)

    def test_multiple_nodes(self):
        """Test computation on multiple nodes."""
        # Grid of nodes
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        xx, yy = np.meshgrid(x, y)
        coords = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(25)])

        directions = np.array([[1.0, 0.0, 0.0]])
        amplitudes = np.array([1.0 + 0j])

        field = TotalPressureField(coords, directions, amplitudes, 1000.0, 343.0)

        assert field.n_nodes == 25
        assert field.n_plane_waves == 1

        P_inc = field.compute_incident_field()
        assert P_inc.shape == (25,)
        assert P_inc.dtype == np.complex128
