"""
Driver script for plotting selected diagonal entries of the reconstructed
CPSD as functions of frequency.

Consumes the diagonal `.npy` file (shape `(N, n_freq)`, real) produced by
`run_reconstruct_full_cpsd.py` in `mode='diagonal'`, together with its
sidecar JSON. Entries to plot can be specified by integer indices into the
sideset-face dimension and/or by physical (x, y, z) coordinates, in which
case the nearest sideset face centroid is used.

Usage:
    python run_plot_cpsd_diagonal.py config_plot_cpsd_diagonal.json
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from exodus_side_interpolator import ExodusSideInterpolator


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return json.load(f)


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    defaults: Dict[str, Dict[str, Any]] = {
        'input': {
            'diagonal_npy_path': None,
            'sidecar_json_path': None,
            'exodus_file': None,
            'sideset_id': None,
        },
        'selection': {
            'indices': None,
            'coordinates': None,
        },
        'plot': {
            'log_scale': True,
            'title': 'CPSD diagonal vs frequency',
            'ylabel': r'$S_{ii}$',
            'xlabel': None,
            'figsize': [9, 5],
            'ylim': None,
            'xlim': None,
        },
        'output': {
            'figure_path': 'results_cpsd_inverse/diagonal_vs_frequency.png',
            'figure_format': 'png',
            'dpi': 150,
            'save_selection_csv': False,
        },
    }
    for section, section_defaults in defaults.items():
        if section not in config:
            config[section] = dict(section_defaults)
        else:
            for key, value in section_defaults.items():
                config[section].setdefault(key, value)

    inp = config['input']
    if inp['diagonal_npy_path'] is None:
        raise ValueError("input.diagonal_npy_path is required")
    if not os.path.exists(inp['diagonal_npy_path']):
        raise FileNotFoundError(
            f"input.diagonal_npy_path not found: {inp['diagonal_npy_path']}"
        )
    if inp['sidecar_json_path'] is None:
        guess = Path(inp['diagonal_npy_path']).with_suffix('.json')
        if guess.exists():
            inp['sidecar_json_path'] = str(guess)
        else:
            raise ValueError(
                "input.sidecar_json_path is required (no matching sidecar "
                f"found at {guess})"
            )
    if not os.path.exists(inp['sidecar_json_path']):
        raise FileNotFoundError(
            f"input.sidecar_json_path not found: {inp['sidecar_json_path']}"
        )

    plot_cfg = config['plot']
    for key in ('ylim', 'xlim'):
        val = plot_cfg.get(key)
        if val is None:
            continue
        if (
            not isinstance(val, (list, tuple))
            or len(val) != 2
            or not all(isinstance(v, (int, float)) for v in val)
            or val[0] >= val[1]
        ):
            raise ValueError(
                f"plot.{key} must be null or a [min, max] pair with min < max"
            )

    sel = config['selection']
    if sel['indices'] is None and sel['coordinates'] is None:
        raise ValueError(
            "selection requires at least one of 'indices' or 'coordinates' "
            "(use 'indices': 'all' to plot every entry)"
        )

    if isinstance(sel['indices'], str):
        if sel['indices'].lower() != 'all':
            raise ValueError(
                "selection.indices must be a list of ints or the string 'all'"
            )
    elif sel['indices'] is not None:
        for v in sel['indices']:
            if not isinstance(v, int) or v < 0:
                raise ValueError(
                    "selection.indices entries must be non-negative ints"
                )

    if sel['coordinates'] is not None:
        coords = np.asarray(sel['coordinates'], dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                "selection.coordinates must be a list of [x, y, z] triples"
            )
        if inp['exodus_file'] is None or inp['sideset_id'] is None:
            raise ValueError(
                "input.exodus_file and input.sideset_id are required when "
                "selection.coordinates is provided"
            )
        if not os.path.exists(inp['exodus_file']):
            raise FileNotFoundError(
                f"input.exodus_file not found: {inp['exodus_file']}"
            )
        if not isinstance(inp['sideset_id'], int):
            raise ValueError("input.sideset_id must be an integer")

    return config


def load_diagonal_data(
    diagonal_path: str, sidecar_path: str
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load the diagonal array and resolve the frequency axis.

    Returns
    -------
    diag : ndarray, shape (N, n_freq), real
    freq_axis : ndarray, shape (n_freq,)
        Physical frequencies if available in the sidecar; otherwise the
        frequency indices.
    xlabel : str
    """
    diag = np.load(diagonal_path)
    if diag.ndim != 2:
        raise ValueError(
            f"Diagonal array must be 2D (N, n_freq); got shape {diag.shape}"
        )

    with open(sidecar_path, 'r') as f:
        sidecar = json.load(f)

    if sidecar.get('mode', 'diagonal') != 'diagonal':
        raise ValueError(
            f"Sidecar reports mode={sidecar.get('mode')!r}; expected "
            f"'diagonal'. Re-run run_reconstruct_full_cpsd.py with "
            f"reconstruction.mode='diagonal'."
        )

    freq_indices = sidecar.get('freq_indices')
    if freq_indices is None or len(freq_indices) != diag.shape[1]:
        raise ValueError(
            "Sidecar 'freq_indices' is missing or does not match the "
            "diagonal array's frequency dimension."
        )

    frequencies = sidecar.get('frequencies')
    if frequencies is not None and len(frequencies) == diag.shape[1]:
        freq_axis = np.asarray(frequencies, dtype=float)
        xlabel = 'Frequency [Hz]'
    else:
        freq_axis = np.asarray(freq_indices, dtype=float)
        xlabel = 'Frequency index'

    return diag, freq_axis, xlabel


def resolve_selection(
    config: Dict[str, Any], n_entries: int
) -> List[Tuple[int, str]]:
    """
    Convert config selection into a list of (index, label) pairs.

    Coordinate-based entries are resolved to the nearest sideset face
    centroid via a brute-force nearest-neighbour search (n_faces is small).
    Indices are deduplicated while preserving first occurrence order.
    """
    sel = config['selection']
    inp = config['input']

    chosen: List[Tuple[int, str]] = []
    seen: set = set()

    def _add(idx: int, label: str) -> None:
        if idx in seen:
            return
        if not 0 <= idx < n_entries:
            raise IndexError(
                f"Selected index {idx} out of range [0, {n_entries})"
            )
        seen.add(idx)
        chosen.append((idx, label))

    if isinstance(sel['indices'], str) and sel['indices'].lower() == 'all':
        for i in range(n_entries):
            _add(i, f'node {i}')
        return chosen

    if sel['indices'] is not None:
        for i in sel['indices']:
            _add(int(i), f'node {i}')

    if sel['coordinates'] is not None:
        coords = np.asarray(sel['coordinates'], dtype=float)
        with ExodusSideInterpolator(inp['exodus_file'], mode='r') as db:
            centroids = db.get_sideset_face_centroids(inp['sideset_id'])
        if centroids.shape[0] != n_entries:
            raise ValueError(
                f"Sideset {inp['sideset_id']} has {centroids.shape[0]} faces "
                f"but diagonal array has {n_entries} entries; cannot match "
                f"coordinate selection."
            )
        for target in coords:
            diffs = centroids - target[None, :]
            distances = np.linalg.norm(diffs, axis=1)
            nearest = int(np.argmin(distances))
            centroid = centroids[nearest]
            label = (
                f'node {nearest} '
                f'(target=[{target[0]:.3g},{target[1]:.3g},{target[2]:.3g}], '
                f'd={distances[nearest]:.3g})'
            )
            _add(nearest, label)
            print(
                f"  coord {target.tolist()} -> face {nearest} at "
                f"{centroid.tolist()} (distance={distances[nearest]:.4g})"
            )

    return chosen


def plot_diagonal(
    diag: np.ndarray,
    freq_axis: np.ndarray,
    selection: List[Tuple[int, str]],
    xlabel: str,
    config: Dict[str, Any],
) -> None:
    plot_cfg = config['plot']
    out_cfg = config['output']

    fig, ax = plt.subplots(figsize=tuple(plot_cfg['figsize']))
    for idx, label in selection:
        values = diag[idx, :]
        if plot_cfg['log_scale']:
            ax.semilogy(freq_axis, values, 'o-', label=label)
        else:
            ax.plot(freq_axis, values, 'o-', label=label)

    ax.set_xlabel(plot_cfg['xlabel'] or xlabel)
    ax.set_ylabel(plot_cfg['ylabel'])
    ax.set_title(plot_cfg['title'])
    if plot_cfg['ylim'] is not None:
        ax.set_ylim(plot_cfg['ylim'])
    if plot_cfg['xlim'] is not None:
        ax.set_xlim(plot_cfg['xlim'])
    ax.grid(True, alpha=0.3)
    if len(selection) <= 20:
        ax.legend(loc='best', fontsize=8)
    fig.tight_layout()

    fig_path = Path(out_cfg['figure_path'])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    if not fig_path.suffix:
        fig_path = fig_path.with_suffix(f".{out_cfg['figure_format']}")
    fig.savefig(fig_path, dpi=out_cfg['dpi'])
    plt.close(fig)
    print(f"Saved plot to {fig_path}")

    if out_cfg['save_selection_csv']:
        csv_path = fig_path.with_suffix('.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [xlabel] + [f'idx_{idx}' for idx, _ in selection]
            writer.writerow(header)
            for k, freq in enumerate(freq_axis):
                row = [float(freq)] + [
                    float(diag[idx, k]) for idx, _ in selection
                ]
                writer.writerow(row)
        print(f"Saved CSV to {csv_path}")


def run(config: Dict[str, Any]) -> None:
    inp = config['input']
    diag, freq_axis, xlabel = load_diagonal_data(
        inp['diagonal_npy_path'], inp['sidecar_json_path']
    )
    print(
        f"Loaded diagonal: shape={diag.shape}, "
        f"freq range=[{freq_axis.min():.4g}, {freq_axis.max():.4g}]"
    )

    selection = resolve_selection(config, diag.shape[0])
    if not selection:
        raise ValueError("Selection resolved to an empty set of indices.")
    print(f"Plotting {len(selection)} diagonal entries.")

    plot_diagonal(diag, freq_axis, selection, xlabel, config)


def main():
    parser = argparse.ArgumentParser(
        description='Plot selected CPSD diagonal entries vs frequency.'
    )
    parser.add_argument(
        'config_file',
        nargs='?',
        default='config_plot_cpsd_diagonal.json',
        help='Path to configuration JSON file '
             '(default: config_plot_cpsd_diagonal.json)',
    )
    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        print(f"Error: Configuration file '{args.config_file}' not found.")
        return 1

    print(f"Loading configuration from: {args.config_file}")
    config = validate_config(load_config(args.config_file))
    run(config)
    return 0


if __name__ == '__main__':
    exit(main())
