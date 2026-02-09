"""
Parser for plane wave amplitudes from loads.txt format.

This module provides functionality to parse complex amplitude specifications
from a custom text format used in acoustic simulations.

File format:
    sideset 3
     acoustic_vel = -0.984113
     scale = {vscale_a}
     function = 1

    sideset 3
     iacoustic_vel = 0.177546
     scale = {vscale_a}
     function = 2

- `acoustic_vel` (odd function) = real part of amplitude
- `iacoustic_vel` (even function) = imaginary part of amplitude
- Complex amplitude: A = acoustic_vel + i*iacoustic_vel
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import re


@dataclass
class LoadEntry:
    """Represents a single load entry from the amplitudes file."""
    sideset_id: int
    amplitude_type: str  # 'acoustic_vel' or 'iacoustic_vel'
    value: float
    function_id: int
    scale: Optional[str] = None


class AmplitudesParser:
    """
    Parser for plane wave amplitudes from loads.txt format.

    Parameters
    ----------
    filepath : str
        Path to the loads.txt file.

    Examples
    --------
    >>> parser = AmplitudesParser("data/loads.txt")
    >>> amplitudes = parser.get_complex_amplitudes()
    >>> print(amplitudes.shape)  # (n_pws,)
    >>> print(amplitudes.dtype)  # complex128
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._entries: Optional[List[LoadEntry]] = None
        self._amplitudes: Optional[np.ndarray] = None
        self._sideset_id: Optional[int] = None

    def parse(self) -> List[LoadEntry]:
        """
        Parse the loads file and return all load entries.

        Returns
        -------
        entries : List[LoadEntry]
            List of all load entries in the file, sorted by function_id.
        """
        if self._entries is not None:
            return self._entries

        entries = []

        with open(self.filepath, 'r') as f:
            content = f.read()

        # Split into blocks by "sideset" keyword
        # Each block starts with "sideset N" and contains the load definition
        blocks = re.split(r'(?=\s*sideset\s+\d+)', content)

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # Parse sideset ID
            sideset_match = re.search(r'sideset\s+(\d+)', block)
            if not sideset_match:
                continue
            sideset_id = int(sideset_match.group(1))

            # Parse amplitude type and value
            # Check for iacoustic_vel FIRST (more specific, avoids substring match)
            imag_match = re.search(r'iacoustic_vel\s*=\s*([-\d.]+)', block)
            # Check for acoustic_vel (real) - use word boundary to avoid matching iacoustic_vel
            real_match = re.search(r'(?<![i])acoustic_vel\s*=\s*([-\d.]+)', block)

            if imag_match:
                amp_type = 'iacoustic_vel'
                value = float(imag_match.group(1))
            elif real_match:
                amp_type = 'acoustic_vel'
                value = float(real_match.group(1))
            else:
                continue  # Skip blocks without amplitude data

            # Parse function ID
            func_match = re.search(r'function\s*=\s*(\d+)', block)
            if not func_match:
                raise ValueError(
                    f"Load entry missing function ID in block: {block[:50]}..."
                )
            function_id = int(func_match.group(1))

            # Parse scale (optional, may contain template variable)
            scale_match = re.search(r'scale\s*=\s*(\S+)', block)
            scale = scale_match.group(1) if scale_match else None

            entry = LoadEntry(
                sideset_id=sideset_id,
                amplitude_type=amp_type,
                value=value,
                function_id=function_id,
                scale=scale
            )
            entries.append(entry)

        # Sort by function_id for consistent pairing
        entries.sort(key=lambda e: e.function_id)

        self._entries = entries
        return entries

    def get_complex_amplitudes(self) -> np.ndarray:
        """
        Get complex amplitudes for all plane waves.

        Pairs acoustic_vel (real, odd functions) with iacoustic_vel
        (imaginary, even functions) to form complex amplitudes.

        Returns
        -------
        amplitudes : ndarray, shape (n_pws,), dtype=complex128
            Complex amplitudes, one per plane wave.
        """
        if self._amplitudes is not None:
            return self._amplitudes

        entries = self.parse()

        # Separate into real and imaginary entries
        real_entries = {e.function_id: e for e in entries
                        if e.amplitude_type == 'acoustic_vel'}
        imag_entries = {e.function_id: e for e in entries
                        if e.amplitude_type == 'iacoustic_vel'}

        # Pair by function IDs: (1,2), (3,4), (5,6), ...
        amplitudes = []
        n_pairs = len(real_entries)

        for i in range(n_pairs):
            real_func_id = 2 * i + 1  # 1, 3, 5, ...
            imag_func_id = 2 * i + 2  # 2, 4, 6, ...

            if real_func_id not in real_entries:
                raise ValueError(
                    f"Missing real amplitude for function {real_func_id}"
                )
            if imag_func_id not in imag_entries:
                raise ValueError(
                    f"Missing imaginary amplitude for function {imag_func_id}"
                )

            real_val = real_entries[real_func_id].value
            imag_val = imag_entries[imag_func_id].value

            amplitudes.append(complex(real_val, imag_val))

        self._amplitudes = np.array(amplitudes, dtype=np.complex128)
        return self._amplitudes

    def get_sideset_id(self) -> int:
        """
        Get the sideset ID from the file.

        Returns
        -------
        sideset_id : int
            The sideset ID (assumes all entries use the same sideset).

        Raises
        ------
        ValueError
            If entries reference different sidesets.
        """
        if self._sideset_id is not None:
            return self._sideset_id

        entries = self.parse()
        if not entries:
            raise ValueError("No load entries found in file")

        sideset_ids = set(e.sideset_id for e in entries)
        if len(sideset_ids) > 1:
            raise ValueError(
                f"Inconsistent sideset IDs in file: {sideset_ids}"
            )

        self._sideset_id = entries[0].sideset_id
        return self._sideset_id

    @property
    def n_plane_waves(self) -> int:
        """Return the number of plane waves."""
        return len(self.get_complex_amplitudes())

    @property
    def n_entries(self) -> int:
        """Return the total number of load entries."""
        return len(self.parse())
