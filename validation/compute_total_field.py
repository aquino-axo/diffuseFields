"""
Compute total pressure field H = D + T from transfer matrix and directions.

The incident field D is computed as D[i,j] = exp(i * k * d_j . x_i)
where k is the wavenumber, d_j is the plane wave direction, and x_i is the node coordinate.
"""

import numpy as np
from pathlib import Path
import json
import argparse


def compute_incident_field(
    coordinates: np.ndarray,
    directions: np.ndarray,
    wavenumber: float
) -> np.ndarray:
    """
    Compute incident field matrix D.

    D[i,j] = exp(i * k * d_j . x_i)

    Parameters
    ----------
    coordinates : ndarray
        Node coordinates, shape (ndof, 3).
    directions : ndarray
        Plane wave directions (unit vectors), shape (n_waves, 3).
    wavenumber : float
        Wavenumber k = 2*pi*f/c.

    Returns
    -------
    D : ndarray
        Incident field matrix, shape (ndof, n_waves).
    """
    dot_products = coordinates @ directions.T
    return np.exp(1j * wavenumber * dot_products)


def compute_total_field(
    transfer_matrix: np.ndarray,
    coordinates: np.ndarray,
    directions: np.ndarray,
    frequencies: np.ndarray,
    speed_of_sound: float
) -> np.ndarray:
    """
    Compute total pressure field H = D + T.

    Parameters
    ----------
    transfer_matrix : ndarray
        Scattered field transfer matrix T, shape (ndof, n_waves, nfreqs).
    coordinates : ndarray
        Node coordinates, shape (ndof, 3).
    directions : ndarray
        Plane wave directions (unit vectors), shape (n_waves, 3).
    frequencies : ndarray
        Frequency array in Hz.
    speed_of_sound : float
        Speed of sound in m/s.

    Returns
    -------
    H : ndarray
        Total field matrix, shape (ndof, n_waves, nfreqs).
    """
    ndof, n_waves, nfreqs = transfer_matrix.shape
    wavenumbers = 2 * np.pi * frequencies / speed_of_sound

    H = np.zeros_like(transfer_matrix)
    for freq_idx in range(nfreqs):
        D = compute_incident_field(coordinates, directions, wavenumbers[freq_idx])
        H[:, :, freq_idx] = D + transfer_matrix[:, :, freq_idx]

    return H


def main():
    parser = argparse.ArgumentParser(
        description="Compute total pressure field H = D + T"
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to config JSON (same format as cone analysis config)"
    )
    parser.add_argument(
        "--transfer-matrix", "-t",
        help="Path to transfer matrix T (.npy). If not provided, uses config input.transfer_matrix_path"
    )
    parser.add_argument(
        "--directions", "-d",
        help="Path to plane wave directions (.npy). If not provided, uses config input.directions_path"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output path for total field H (.npy)"
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # Get paths from config or command line
    config_dir = Path(args.config).parent

    if args.transfer_matrix:
        transfer_matrix_path = args.transfer_matrix
    else:
        transfer_matrix_path = config_dir / config['input']['transfer_matrix_path']

    if args.directions:
        directions_path = args.directions
    else:
        directions_path = config_dir / config['input']['directions_path']

    coordinates_path = config_dir / config['input']['coordinates_path']

    # Load data
    T = np.load(transfer_matrix_path)
    coordinates = np.load(coordinates_path)
    directions = np.load(directions_path)

    # Build frequency array
    freq_cfg = config['physics']['frequencies']
    frequencies = np.arange(
        freq_cfg['min'],
        freq_cfg['max'] + freq_cfg['step'],
        freq_cfg['step']
    )
    speed_of_sound = config['physics']['speed_of_sound']

    print(f"Transfer matrix: {transfer_matrix_path}")
    print(f"  Shape: {T.shape}")
    print(f"Coordinates: {coordinates_path}")
    print(f"  Shape: {coordinates.shape}")
    print(f"Directions: {directions_path}")
    print(f"  Shape: {directions.shape}")
    print(f"Frequencies: {frequencies[0]:.0f} - {frequencies[-1]:.0f} Hz ({len(frequencies)} points)")
    print(f"Speed of sound: {speed_of_sound} m/s")

    # Compute total field
    H = compute_total_field(T, coordinates, directions, frequencies, speed_of_sound)

    print(f"Total field H shape: {H.shape}")

    # Save
    np.save(args.output, H)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
