"""
Script to visualize columns of the transfer matrix on the cone surface.

Each column of the transfer matrix represents the scattered pressure response
at all surface nodes due to a unit-amplitude plane wave from a specific direction.

Usage:
    python visualize_transfer_matrix.py
    python visualize_transfer_matrix.py --freq-idx 0 --pw-indices 0 1 2 3
    python visualize_transfer_matrix.py --freq-idx 10 --pw-indices 0 --component real
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from cone_visualizer import ConeVisualizer


def load_data(data_dir: str = 'data'):
    """Load transfer matrix, coordinates, and directions."""
    data_path = Path(data_dir)

    T = np.load(data_path / 'Tmatrix.npy')
    coords = np.load(data_path / 'coordinates.npy')
    directions = np.load(data_path / 'directions.npy')

    return T, coords, directions


def visualize_single_column(
    T: np.ndarray,
    coords: np.ndarray,
    directions: np.ndarray,
    freq_idx: int,
    pw_idx: int,
    component: str = 'magnitude',
    cone_geometry: dict = None
):
    """
    Visualize a single column of the transfer matrix.

    Parameters
    ----------
    T : ndarray
        Transfer matrix of shape (ndof, npws, nfreqs).
    coords : ndarray
        Node coordinates, shape (ndof, 3).
    directions : ndarray
        Plane wave directions, shape (npws, 3).
    freq_idx : int
        Frequency index.
    pw_idx : int
        Plane wave index (column to visualize).
    component : str
        Component to plot: 'magnitude', 'real', 'imag', or 'phase'.
    cone_geometry : dict, optional
        Cone geometry parameters.
    """
    if cone_geometry is None:
        cone_geometry = {
            'half_angle': 0.244346,  # 14 degrees in radians
            'height': 0.5
        }

    visualizer = ConeVisualizer(coords, cone_geometry)

    # Extract the column
    pressure = T[:, pw_idx, freq_idx]

    # Get direction info
    d = directions[pw_idx]

    title = (
        f'Transfer Matrix Column {pw_idx}\n'
        f'Frequency index: {freq_idx}, '
        f'Direction: ({d[0]:.2f}, {d[1]:.2f}, {d[2]:.2f})'
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    visualizer.plot_pressure_field(pressure, component=component, ax=ax, title=title)

    return fig


def visualize_multiple_columns(
    T: np.ndarray,
    coords: np.ndarray,
    directions: np.ndarray,
    freq_idx: int,
    pw_indices: list,
    component: str = 'magnitude',
    cone_geometry: dict = None
):
    """
    Visualize multiple columns of the transfer matrix in a grid.

    Parameters
    ----------
    T : ndarray
        Transfer matrix of shape (ndof, npws, nfreqs).
    coords : ndarray
        Node coordinates, shape (ndof, 3).
    directions : ndarray
        Plane wave directions, shape (npws, 3).
    freq_idx : int
        Frequency index.
    pw_indices : list
        List of plane wave indices (columns to visualize).
    component : str
        Component to plot: 'magnitude', 'real', 'imag', or 'phase'.
    cone_geometry : dict, optional
        Cone geometry parameters.
    """
    if cone_geometry is None:
        cone_geometry = {
            'half_angle': 0.244346,  # 14 degrees in radians
            'height': 0.5
        }

    visualizer = ConeVisualizer(coords, cone_geometry)

    # Extract the columns as pressure fields
    pressures = T[:, pw_indices, freq_idx]

    # Create labels with direction info
    labels = []
    for pw_idx in pw_indices:
        d = directions[pw_idx]
        labels.append(
            f'PW {pw_idx}: d=({d[0]:.2f}, {d[1]:.2f}, {d[2]:.2f})'
        )

    fig = visualizer.plot_multiple_fields(
        pressures,
        labels=labels,
        component=component
    )

    fig.suptitle(f'Transfer Matrix Columns (Frequency index: {freq_idx})', y=1.02)

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize columns of the transfer matrix'
    )
    parser.add_argument(
        '--data-dir', '-d',
        default='data',
        help='Directory containing the .npy files (default: data)'
    )
    parser.add_argument(
        '--freq-idx', '-f',
        type=int,
        default=0,
        help='Frequency index (default: 0)'
    )
    parser.add_argument(
        '--pw-indices', '-p',
        type=int,
        nargs='+',
        default=[0, 1, 2, 3],
        help='Plane wave indices to visualize (default: 0 1 2 3)'
    )
    parser.add_argument(
        '--component', '-c',
        choices=['magnitude', 'real', 'imag', 'phase'],
        default='magnitude',
        help='Component to plot (default: magnitude)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file path (if not specified, shows interactive plot)'
    )
    parser.add_argument(
        '--single',
        action='store_true',
        help='Plot single column (uses first pw-index only)'
    )

    args = parser.parse_args()

    print(f"Loading data from: {args.data_dir}")
    T, coords, directions = load_data(args.data_dir)

    print(f"Transfer matrix shape: {T.shape}")
    print(f"  - {T.shape[0]} DOFs")
    print(f"  - {T.shape[1]} plane waves")
    print(f"  - {T.shape[2]} frequencies")
    print(f"Coordinates shape: {coords.shape}")
    print(f"Directions shape: {directions.shape}")

    # Validate indices
    if args.freq_idx < 0 or args.freq_idx >= T.shape[2]:
        print(f"Error: freq_idx must be in [0, {T.shape[2]})")
        return 1

    for pw_idx in args.pw_indices:
        if pw_idx < 0 or pw_idx >= T.shape[1]:
            print(f"Error: pw_index {pw_idx} must be in [0, {T.shape[1]})")
            return 1

    if args.single:
        print(f"\nVisualizing single column: pw_index={args.pw_indices[0]}, freq_idx={args.freq_idx}")
        fig = visualize_single_column(
            T, coords, directions,
            freq_idx=args.freq_idx,
            pw_idx=args.pw_indices[0],
            component=args.component
        )
    else:
        print(f"\nVisualizing columns: pw_indices={args.pw_indices}, freq_idx={args.freq_idx}")
        fig = visualize_multiple_columns(
            T, coords, directions,
            freq_idx=args.freq_idx,
            pw_indices=args.pw_indices,
            component=args.component
        )

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved to: {args.output}")
        plt.close(fig)
    else:
        plt.show()

    return 0


if __name__ == '__main__':
    exit(main())
