"""
Parser for plane wave directions from functions.txt format.

This module provides functionality to parse plane wave direction specifications
from a custom text format used in acoustic simulations.

File format:
    Function 1
        type plane_wave_freq
        Material "air"
        Direction -0.855707 0.319087 0.407369
        Origin 0 0 0
    END

Functions come in pairs:
- Odd functions (1, 3, 5, ...) have type `plane_wave_freq` (real component)
- Even functions (2, 4, 6, ...) have type `iplane_wave_freq` (imaginary component)
- Paired functions share the same direction vector
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import re


@dataclass
class FunctionEntry:
    """Represents a single function entry from the directions file."""
    function_id: int
    func_type: str  # 'plane_wave_freq' or 'iplane_wave_freq'
    direction: np.ndarray  # (3,) unit vector
    origin: np.ndarray  # (3,)
    material: Optional[str] = None


class DirectionsParser:
    """
    Parser for plane wave directions from functions.txt format.

    Parameters
    ----------
    filepath : str
        Path to the functions.txt file.

    Examples
    --------
    >>> parser = DirectionsParser("data/functions.txt")
    >>> directions = parser.get_directions()
    >>> print(directions.shape)  # (n_pws, 3)
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._entries: Optional[List[FunctionEntry]] = None
        self._directions: Optional[np.ndarray] = None

    def parse(self) -> List[FunctionEntry]:
        """
        Parse the functions file and return all function entries.

        Returns
        -------
        entries : List[FunctionEntry]
            List of all function entries in the file.
        """
        if self._entries is not None:
            return self._entries

        entries = []

        with open(self.filepath, 'r') as f:
            content = f.read()

        # Split into function blocks
        # Pattern matches "Function N" through "END"
        pattern = r'Function\s+(\d+)(.*?)END'
        matches = re.findall(pattern, content, re.DOTALL)

        for func_id_str, block in matches:
            func_id = int(func_id_str)

            # Parse type
            type_match = re.search(r'type\s+(\w+)', block)
            func_type = type_match.group(1) if type_match else None

            # Parse direction
            dir_match = re.search(
                r'Direction\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)',
                block
            )
            if dir_match:
                direction = np.array([
                    float(dir_match.group(1)),
                    float(dir_match.group(2)),
                    float(dir_match.group(3))
                ])
            else:
                raise ValueError(
                    f"Function {func_id}: Direction not found"
                )

            # Parse origin
            origin_match = re.search(
                r'Origin\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)',
                block
            )
            if origin_match:
                origin = np.array([
                    float(origin_match.group(1)),
                    float(origin_match.group(2)),
                    float(origin_match.group(3))
                ])
            else:
                origin = np.zeros(3)

            # Parse material
            mat_match = re.search(r'Material\s+"([^"]+)"', block)
            material = mat_match.group(1) if mat_match else None

            entry = FunctionEntry(
                function_id=func_id,
                func_type=func_type,
                direction=direction,
                origin=origin,
                material=material
            )
            entries.append(entry)

        self._entries = entries
        return entries

    def get_directions(self) -> np.ndarray:
        """
        Get unique plane wave directions.

        Since functions come in pairs (real/imaginary) sharing the same
        direction, this returns only unique directions from odd-numbered
        functions.

        Returns
        -------
        directions : ndarray, shape (n_pws, 3)
            Unique direction vectors, one per plane wave.
        """
        if self._directions is not None:
            return self._directions

        entries = self.parse()

        # Extract directions from odd-numbered functions (real components)
        # which share directions with their even-numbered pairs
        directions = []
        for entry in entries:
            if entry.function_id % 2 == 1:  # Odd function = real component
                directions.append(entry.direction)

        self._directions = np.array(directions)
        return self._directions

    def get_function_to_pw_map(self) -> Dict[int, int]:
        """
        Get mapping from function ID to plane wave index.

        Returns
        -------
        mapping : Dict[int, int]
            Maps function ID (1, 2, 3, ...) to plane wave index (0, 1, 1, ...).
            Functions 1 and 2 both map to plane wave 0, functions 3 and 4 to
            plane wave 1, etc.
        """
        entries = self.parse()
        mapping = {}
        for entry in entries:
            # Function pair (2k-1, 2k) maps to plane wave index k-1
            pw_index = (entry.function_id - 1) // 2
            mapping[entry.function_id] = pw_index
        return mapping

    @property
    def n_plane_waves(self) -> int:
        """Return the number of unique plane waves."""
        return len(self.get_directions())

    @property
    def n_functions(self) -> int:
        """Return the total number of functions."""
        return len(self.parse())
