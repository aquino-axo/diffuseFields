"""Tests for DirectionsParser class."""

import numpy as np
import pytest
import tempfile
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from directions_parser import DirectionsParser, FunctionEntry


class TestDirectionsParser:
    """Tests for DirectionsParser."""

    def test_parse_single_function_pair(self):
        """Test parsing a single real/imaginary function pair."""
        content = """Function 1
    type plane_wave_freq
    Material "air"
    Direction -0.855707 0.319087 0.407369
    Origin 0 0 0
END

Function 2
    type iplane_wave_freq
    Material "air"
    Direction -0.855707 0.319087 0.407369
    Origin 0 0 0
END
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            f.flush()
            temp_path = f.name

        try:
            parser = DirectionsParser(temp_path)
            entries = parser.parse()

            assert len(entries) == 2

            # Check first entry (real)
            assert entries[0].function_id == 1
            assert entries[0].func_type == 'plane_wave_freq'
            assert entries[0].material == 'air'
            np.testing.assert_array_almost_equal(
                entries[0].direction,
                [-0.855707, 0.319087, 0.407369]
            )

            # Check second entry (imaginary)
            assert entries[1].function_id == 2
            assert entries[1].func_type == 'iplane_wave_freq'
        finally:
            os.unlink(temp_path)

    def test_parse_multiple_function_pairs(self):
        """Test parsing multiple plane wave directions."""
        content = """Function 1
    type plane_wave_freq
    Direction 1.0 0.0 0.0
    Origin 0 0 0
END

Function 2
    type iplane_wave_freq
    Direction 1.0 0.0 0.0
    Origin 0 0 0
END

Function 3
    type plane_wave_freq
    Direction 0.0 1.0 0.0
    Origin 0 0 0
END

Function 4
    type iplane_wave_freq
    Direction 0.0 1.0 0.0
    Origin 0 0 0
END

Function 5
    type plane_wave_freq
    Direction 0.0 0.0 1.0
    Origin 0 0 0
END

Function 6
    type iplane_wave_freq
    Direction 0.0 0.0 1.0
    Origin 0 0 0
END
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            f.flush()
            temp_path = f.name

        try:
            parser = DirectionsParser(temp_path)
            entries = parser.parse()

            assert len(entries) == 6
            assert parser.n_functions == 6
        finally:
            os.unlink(temp_path)

    def test_get_unique_directions(self):
        """Test that unique directions returns n_pws = n_functions / 2."""
        content = """Function 1
    type plane_wave_freq
    Direction 1.0 0.0 0.0
    Origin 0 0 0
END

Function 2
    type iplane_wave_freq
    Direction 1.0 0.0 0.0
    Origin 0 0 0
END

Function 3
    type plane_wave_freq
    Direction 0.0 1.0 0.0
    Origin 0 0 0
END

Function 4
    type iplane_wave_freq
    Direction 0.0 1.0 0.0
    Origin 0 0 0
END
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            f.flush()
            temp_path = f.name

        try:
            parser = DirectionsParser(temp_path)
            directions = parser.get_directions()

            # Should have 2 unique directions (from 4 functions)
            assert directions.shape == (2, 3)
            assert parser.n_plane_waves == 2

            # Check direction values
            np.testing.assert_array_almost_equal(
                directions[0], [1.0, 0.0, 0.0]
            )
            np.testing.assert_array_almost_equal(
                directions[1], [0.0, 1.0, 0.0]
            )
        finally:
            os.unlink(temp_path)

    def test_function_to_pw_map(self):
        """Test mapping from function IDs to plane wave indices."""
        content = """Function 1
    type plane_wave_freq
    Direction 1.0 0.0 0.0
    Origin 0 0 0
END

Function 2
    type iplane_wave_freq
    Direction 1.0 0.0 0.0
    Origin 0 0 0
END

Function 3
    type plane_wave_freq
    Direction 0.0 1.0 0.0
    Origin 0 0 0
END

Function 4
    type iplane_wave_freq
    Direction 0.0 1.0 0.0
    Origin 0 0 0
END
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            f.flush()
            temp_path = f.name

        try:
            parser = DirectionsParser(temp_path)
            mapping = parser.get_function_to_pw_map()

            # Functions 1,2 -> plane wave 0
            # Functions 3,4 -> plane wave 1
            assert mapping[1] == 0
            assert mapping[2] == 0
            assert mapping[3] == 1
            assert mapping[4] == 1
        finally:
            os.unlink(temp_path)

    def test_parse_real_file(self):
        """Test parsing the actual functions.txt file if it exists."""
        real_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'functions.txt'
        )

        if not os.path.exists(real_path):
            pytest.skip("Real functions.txt not available")

        parser = DirectionsParser(real_path)
        directions = parser.get_directions()

        # Should have 100 plane waves (200 functions / 2)
        assert directions.shape[0] == 100
        assert directions.shape[1] == 3

        # Directions should be approximately unit vectors
        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(100), decimal=5)
