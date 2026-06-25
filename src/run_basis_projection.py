"""
Driver script for per-frequency basis-projection residual analysis.

Given two transfer matrices of the same form as ``data/Tmatrix_cone_only.npy``
(shape ``(ndof, npws, nfreq)``), one is treated as a "basis" and the other as
"data". The data columns are projected onto the column space of the basis at
each frequency, and the relative residual of the best (least-squares)
approximation is reported and plotted versus frequency.

Usage:
    python run_basis_projection.py BASIS DATA
    python run_basis_projection.py basis.npy data.npy --output-dir results_projection
    python run_basis_projection.py basis.mat data.mat --basis-var H --data-var H

The basis column space lives in C^ndof, so both matrices must share the same
number of rows (ndof) and the same number of frequencies. The number of plane
waves (columns) may differ.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from basis_projector import BasisProjection


def _load_mat_var(path: str, var: str = None) -> np.ndarray:
    """Read a named variable from a MATLAB .mat file.

    If ``var`` is None, auto-detect the single non-metadata variable; if there
    is more than one candidate, raise listing the available keys.
    """
    mat = loadmat(path)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    if var is None:
        if len(keys) == 1:
            var = keys[0]
        else:
            raise KeyError(
                f"{path} contains multiple variables {keys}; specify which one "
                f"to use with the corresponding --*-var option."
            )
    if var not in mat:
        raise KeyError(
            f"variable '{var}' not found in {path}; available keys: {keys}"
        )
    return np.asarray(mat[var])


def load_matrix(path: str, var: str = None) -> np.ndarray:
    """Load a transfer matrix from a .npy or .mat file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mat":
        return _load_mat_var(path, var)
    if ext == ".npy":
        return np.load(path)
    raise ValueError(
        f"unsupported file extension '{ext}' for {path}; expected .npy or .mat"
    )


def resolve_frequencies(spec: str, nfreq: int) -> tuple:
    """Resolve x-axis frequency values and label.

    ``spec`` may be a path to a 1-D .npy file, a comma-separated list of values,
    or None (use the frequency index 0..nfreq-1).
    """
    if spec is None:
        return np.arange(nfreq, dtype=float), "Frequency index"

    if os.path.splitext(spec)[1].lower() == ".npy":
        freqs = np.asarray(np.load(spec), dtype=float).ravel()
    else:
        freqs = np.array([float(v) for v in spec.split(",")], dtype=float)

    if freqs.size != nfreq:
        raise ValueError(
            f"--frequencies provides {freqs.size} values but the matrices have "
            f"{nfreq} frequencies"
        )
    return freqs, "Frequency (Hz)"


def save_report(path: str, args, result: dict, frequencies: np.ndarray) -> None:
    """Write the JSON projection report (metadata + per-frequency arrays + stats)."""
    rel = result["relative_residual"]
    finite = rel[np.isfinite(rel)]
    if finite.size > 0:
        worst_idx = int(np.nanargmax(rel))
        summary = {
            "mean_relative_residual": float(np.mean(finite)),
            "min_relative_residual": float(np.min(finite)),
            "max_relative_residual": float(np.max(finite)),
            "worst_frequency": float(frequencies[worst_idx]),
            "worst_frequency_index": worst_idx,
        }
    else:
        summary = {
            "mean_relative_residual": None,
            "min_relative_residual": None,
            "max_relative_residual": None,
            "worst_frequency": None,
            "worst_frequency_index": None,
        }

    report = {
        "metadata": {
            "basis_path": os.path.abspath(args.basis),
            "data_path": os.path.abspath(args.data),
            "ndof": int(result["ndof"]),
            "npws_basis": int(result["npws_basis"]),
            "npws_data": int(result["npws_data"]),
            "nfreq": int(result["nfreq"]),
            "rtol": float(args.rtol),
        },
        "per_frequency": {
            "frequencies": [float(f) for f in frequencies],
            "relative_residual": [
                float(v) if np.isfinite(v) else None for v in rel
            ],
            "basis_rank": [int(r) for r in result["basis_rank"]],
        },
        "summary": summary,
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def save_csv(path: str, result: dict, frequencies: np.ndarray) -> None:
    """Write per-frequency residuals as CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frequency", "relative_residual", "basis_rank"])
        for freq, res, rank in zip(
            frequencies, result["relative_residual"], result["basis_rank"]
        ):
            writer.writerow([float(freq), float(res), int(rank)])


def save_plot(path: str, result: dict, frequencies: np.ndarray,
              xlabel: str, fmt: str) -> None:
    """Plot relative residual versus frequency."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        frequencies,
        result["relative_residual"],
        marker="o",
        color="tab:blue",
        label="Relative residual",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Relative residual  $\|D - \hat{D}\|_F / \|D\|_F$")
    ax.set_title("Best-approximation residual of data onto basis column space")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Project data transfer-matrix columns onto a basis column space "
            "per frequency and report/plot the relative residual."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_basis_projection.py basis.npy data.npy
  python run_basis_projection.py basis.npy data.npy --output-dir results_projection
  python run_basis_projection.py basis.mat data.mat --basis-var H --data-var H \\
      --frequencies freqs.npy

Both matrices must be 3D (ndof, npws, nfreq) and share ndof and nfreq.
        """,
    )
    parser.add_argument("basis", help="Path to the basis matrix (.npy or .mat)")
    parser.add_argument("data", help="Path to the data matrix (.npy or .mat)")
    parser.add_argument(
        "--output-dir", default="results_projection",
        help="Output directory (default: results_projection)",
    )
    parser.add_argument(
        "--basis-var", default=None,
        help="Variable name inside a .mat basis file (auto-detected if single var)",
    )
    parser.add_argument(
        "--data-var", default=None,
        help="Variable name inside a .mat data file (auto-detected if single var)",
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-12,
        help="Relative singular-value threshold for basis numerical rank "
             "(default: 1e-12)",
    )
    parser.add_argument(
        "--frequencies", default=None,
        help="Path to a 1-D .npy of frequencies or a comma-separated list, "
             "used to label the x-axis (default: frequency index)",
    )
    parser.add_argument(
        "--figure-format", default="png", choices=["png", "pdf", "svg", "eps"],
        help="Plot file format (default: png)",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip plot generation",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Basis-Projection Residual Analysis")
    print("=" * 60)

    # Load
    try:
        basis = load_matrix(args.basis, args.basis_var)
        data = load_matrix(args.data, args.data_var)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error loading inputs: {e}")
        sys.exit(1)

    print(f"Basis matrix: {args.basis}  shape {basis.shape}")
    print(f"Data matrix:  {args.data}  shape {data.shape}")

    # Compute
    try:
        projector = BasisProjection(basis, data, rtol=args.rtol)
        frequencies, xlabel = resolve_frequencies(args.frequencies, projector.nfreq)
        result = projector.project()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    rel = result["relative_residual"]
    finite = rel[np.isfinite(rel)]
    print(f"Frequencies: {projector.nfreq}")
    if finite.size > 0:
        print(
            f"Relative residual: mean={np.mean(finite):.3e} "
            f"min={np.min(finite):.3e} max={np.max(finite):.3e}"
        )

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "projection_report.json")
    csv_path = os.path.join(args.output_dir, "relative_residual.csv")
    save_report(report_path, args, result, frequencies)
    save_csv(csv_path, result, frequencies)
    print(f"Saved: {report_path}")
    print(f"Saved: {csv_path}")

    if not args.no_plots:
        plot_path = os.path.join(
            args.output_dir, f"relative_residual_vs_frequency.{args.figure_format}"
        )
        save_plot(plot_path, result, frequencies, xlabel, args.figure_format)
        print(f"Saved: {plot_path}")

    print("Done.")


if __name__ == "__main__":
    main()
