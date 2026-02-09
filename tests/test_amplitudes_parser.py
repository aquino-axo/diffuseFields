"""Tests for AmplitudesParser class."""

import numpy as np
import pytest
import tempfile
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amplitudes_parser import AmplitudesParser, LoadEntry


class TestAmplitudesParser:
    """Tests for AmplitudesParser."""

    def test_parse_single_amplitude_pair(self):
        """Test parsing a single real/imaginary amplitude pair."""
        content = """    sideset 3
    acoustic_vel = -0.984113
    scale = {vscale_a}
    function = 1

    sideset 3
    iacoustic_vel = 0.177546
    scale = {vscale_a}
    function = 2
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            f.flush()
            temp_path = f.name

        try:
            parser = AmplitudesParser(temp_path)
            entries = parser.parse()

            assert len(entries) == 2
            assert parser.n_entries == 2

            # Check first entry (real)
            assert entries[0].sideset_id == 3
            assert entries[0].amplitude_type == 'acoustic_vel'
            assert entries[0].function_id == 1
            np.testing.assert_almost_equal(entries[0].value, -0.984113)

            # Check second entry (imaginary)
            assert entries[1].sideset_id == 3
            assert entries[1].amplitude_type == 'iacoustic_vel'
            assert entries[1].function_id == 2
            np.testing.assert_almost_equal(entries[1].value, 0.177546)
        finally:
            os.unlink(temp_path)

    def test_get_complex_amplitudes(self):
        """Test complex amplitude pairing."""
        content = """    sideset 3
    acoustic_vel = -0.984113
    scale = {vscale_a}
    function = 1

    sideset 3
    iacoustic_vel = 0.177546
    scale = {vscale_a}
    function = 2

    sideset 3
    acoustic_vel = 0.5
    scale = {vscale_a}
    function = 3

    sideset 3
    iacoustic_vel = -0.8
    scale = {vscale_a}
    function = 4
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            f.flush()
            temp_path = f.name

        try:
            parser = AmplitudesParser(temp_path)
            amplitudes = parser.get_complex_amplitudes()

            assert len(amplitudes) == 2
            assert parser.n_plane_waves == 2
            assert amplitudes.dtype == np.complex128

            # Check first complex amplitude
            expected_1 = complex(-0.984113, 0.177546)
            np.testing.assert_almost_equal(amplitudes[0], expected_1)

            # Check second complex amplitude
            expected_2 = complex(0.5, -0.8)
            np.testing.assert_almost_equal(amplitudes[1], expected_2)
        finally:
            os.unlink(temp_path)

    def test_sideset_consistency(self):
        """Test that sideset ID is consistent across entries."""
        content = """    sideset 3
    acoustic_vel = 0.5
    function = 1

    sideset 3
    iacoustic_vel = 0.5
    function = 2
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            f.flush()
            temp_path = f.name

        try:
            parser = AmplitudesParser(temp_path)
            sideset_id = parser.get_sideset_id()
            assert sideset_id == 3
        finally:
            os.unlink(temp_path)

    def test_sideset_inconsistency_raises(self):
        """Test that inconsistent sideset IDs raise error."""
        content = """    sideset 3
    acoustic_vel = 0.5
    function = 1

    sideset 5
    iacoustic_vel = 0.5
    function = 2
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            f.flush()
            temp_path = f.name

        try:
            parser = AmplitudesParser(temp_path)
            with pytest.raises(ValueError, match="Inconsistent sideset IDs"):
                parser.get_sideset_id()
        finally:
            os.unlink(temp_path)

    def test_amplitude_magnitude(self):
        """Test that amplitudes have expected magnitudes."""
        content = """    sideset 3
    acoustic_vel = 0.6
    function = 1

    sideset 3
    iacoustic_vel = 0.8
    function = 2
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            f.flush()
            temp_path = f.name

        try:
            parser = AmplitudesParser(temp_path)
            amplitudes = parser.get_complex_amplitudes()

            # |0.6 + 0.8i| = 1.0
            np.testing.assert_almost_equal(np.abs(amplitudes[0]), 1.0)
        finally:
            os.unlink(temp_path)

    def test_parse_real_file(self):
        """Test parsing the actual loads.txt file if it exists."""
        real_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'loads.txt'
        )

        if not os.path.exists(real_path):
            pytest.skip("Real loads.txt not available")

        parser = AmplitudesParser(real_path)
        amplitudes = parser.get_complex_amplitudes()

        # Should have 100 plane waves (200 entries / 2)
        assert len(amplitudes) == 100

        # All amplitudes should be finite
        assert np.all(np.isfinite(amplitudes))

        # Check sideset ID is consistent
        sideset_id = parser.get_sideset_id()
        assert sideset_id == 3
