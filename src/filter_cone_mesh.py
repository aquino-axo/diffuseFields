"""Filter cone mesh to exclude base disk points."""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def filter_cone_surface(
    coordinates_path: str,
    transfer_matrix_path: str,
    output_dir: str = "data",
    base_z: float = -0.1778,
    tolerance: float = 1e-6
):
    """
    Filter out base disk points from cone mesh.

    The cone mesh includes a filled base disk at z = base_z. This function
    creates new data files containing only the conical surface points.

    Parameters
    ----------
    coordinates_path : str
        Path to coordinates.npy (shape: n_total, 3)
    transfer_matrix_path : str
        Path to Tmatrix.npy (shape: n_total, n_pws, n_freqs)
    output_dir : str
        Directory for filtered output files
    base_z : float
        Z-coordinate of base disk (default: -0.1778 m)
    tolerance : float
        Tolerance for z-coordinate comparison
    """
    coords = np.load(coordinates_path)
    T = np.load(transfer_matrix_path)

    print(f"Loaded coordinates: {coords.shape}")
    print(f"Loaded transfer matrix: {T.shape}")

    # Create mask for cone surface (exclude base)
    z = coords[:, 2]
    cone_mask = ~np.isclose(z, base_z, atol=tolerance)

    # Filter
    coords_cone = coords[cone_mask]
    T_cone = T[cone_mask, :, :]

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coords_path = output_dir / "coordinates_cone_only.npy"
    T_path = output_dir / "Tmatrix_cone_only.npy"

    np.save(coords_path, coords_cone)
    np.save(T_path, T_cone)

    # Summary
    n_base = (~cone_mask).sum()
    n_cone = cone_mask.sum()
    print(f"\nFiltering summary:")
    print(f"  Original points: {coords.shape[0]}")
    print(f"  Base disk points: {n_base} (z = {base_z})")
    print(f"  Cone surface points: {n_cone}")
    print(f"\nFiltered coordinates shape: {coords_cone.shape}")
    print(f"Filtered transfer matrix shape: {T_cone.shape}")
    print(f"\nCone surface z-range: [{coords_cone[:, 2].min():.4f}, {coords_cone[:, 2].max():.4f}]")
    print(f"\nSaved to:")
    print(f"  {coords_path}")
    print(f"  {T_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter cone mesh to exclude base disk points."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to JSON config file (optional)"
    )
    parser.add_argument(
        "--coordinates", "-c",
        default="data/coordinates.npy",
        help="Path to coordinates file (default: data/coordinates.npy)"
    )
    parser.add_argument(
        "--transfer-matrix", "-t",
        default="data/Tmatrix.npy",
        help="Path to transfer matrix file (default: data/Tmatrix.npy)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="data",
        help="Output directory (default: data)"
    )
    parser.add_argument(
        "--base-z", "-z",
        type=float,
        default=-0.1778,
        help="Z-coordinate of base disk to exclude (default: -0.1778)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for z-coordinate comparison (default: 1e-6)"
    )

    args = parser.parse_args()

    # If config file provided, load it and override defaults
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)

        coordinates_path = config.get("coordinates_path", args.coordinates)
        transfer_matrix_path = config.get("transfer_matrix_path", args.transfer_matrix)
        output_dir = config.get("output_dir", args.output_dir)
        base_z = config.get("base_z", args.base_z)
        tolerance = config.get("tolerance", args.tolerance)
    else:
        coordinates_path = args.coordinates
        transfer_matrix_path = args.transfer_matrix
        output_dir = args.output_dir
        base_z = args.base_z
        tolerance = args.tolerance

    print(f"Base z-coordinate: {base_z}")
    print(f"Tolerance: {tolerance}")

    filter_cone_surface(
        coordinates_path=coordinates_path,
        transfer_matrix_path=transfer_matrix_path,
        output_dir=output_dir,
        base_z=base_z,
        tolerance=tolerance
    )


if __name__ == "__main__":
    main()
