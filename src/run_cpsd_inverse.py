"""
Driver script for reduced-basis CPSD Tikhonov inversion.

Reads a JSON configuration that points to a reduced transfer matrix .npy
(shape n_sensors x n_pod x n_freq), a POD basis .npy (shape N x n_pod), and
an experimental CPSD .mat file (shape n_sensors x n_sensors x n_freq), then
solves the per-frequency regularized inverse problem described in
DiffuseFields_Inversion.pdf and saves S_r per frequency plus diagnostics.

Usage:
    python run_cpsd_inverse.py config_cpsd_inverse.json
    python run_cpsd_inverse.py  # default config_cpsd_inverse.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from cpsd_inverse import CPSDInverseSolver


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def parse_frequencies(freq_config: Any) -> List[float]:
    """Parse a list or a {min, step, max} dict into a list of frequencies."""
    if isinstance(freq_config, list):
        if len(freq_config) == 0:
            raise ValueError("physics.frequencies list cannot be empty")
        for f in freq_config:
            if not isinstance(f, (int, float)) or f <= 0:
                raise ValueError(
                    f"all frequencies must be positive numbers, got {f}"
                )
        return [float(f) for f in freq_config]

    if isinstance(freq_config, dict):
        for key in ('min', 'step', 'max'):
            if key not in freq_config:
                raise ValueError(
                    f"physics.frequencies dict missing key '{key}'"
                )
        f_min = freq_config['min']
        f_step = freq_config['step']
        f_max = freq_config['max']
        if f_min <= 0 or f_step <= 0 or f_max <= 0:
            raise ValueError("physics.frequencies values must be positive")
        if f_max < f_min:
            raise ValueError("physics.frequencies.max must be >= min")
        freqs = np.arange(f_min, f_max + f_step * 0.001, f_step)
        freqs = freqs[freqs <= f_max + 1e-10]
        return freqs.tolist()

    raise ValueError(
        f"physics.frequencies must be a list or a dict with min/step/max, "
        f"got {type(freq_config).__name__}"
    )


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the config and fill in defaults."""
    defaults: Dict[str, Dict[str, Any]] = {
        'input': {
            'transfer_matrix_path': None,
            'transfer_matrix_var': None,   # required only for .mat input
            'transfer_matrix_scale': 1.0,  # gamma: T_r is multiplied by this
            'pod_basis_path': None,
            'experimental_cpsd_path': None,
            'experimental_cpsd_var': None,
            # Optional row-index subset: select a subset of T_r rows (and the
            # corresponding rows/columns of G) to use in the inverse problem.
            'row_indices_path': None,      # .mat file, optional
            'row_indices_var': None,       # variable name inside the .mat
            'row_indices_one_based': True, # MATLAB 1-based by default
        },
        'physics': {
            'frequencies': None,  # optional metadata; alignment is by index
        },
        'regularization': {
            'alpha': None,          # scalar applied to all frequencies
            'alpha_sweep': None,    # list applied to every frequency
            'psd_tol_rel': 0.0,
        },
        'output': {
            'output_dir': 'results_cpsd_inverse',
            'save_figures': True,
            'figure_format': 'png',
        },
    }
    for section, section_defaults in defaults.items():
        if section not in config:
            config[section] = dict(section_defaults)
        else:
            for key, value in section_defaults.items():
                config[section].setdefault(key, value)

    inp = config['input']
    for key in (
        'transfer_matrix_path', 'pod_basis_path',
        'experimental_cpsd_path', 'experimental_cpsd_var',
    ):
        if inp[key] is None:
            raise ValueError(f"input.{key} is required")
    for key in ('transfer_matrix_path', 'pod_basis_path',
                'experimental_cpsd_path'):
        if not os.path.exists(inp[key]):
            raise FileNotFoundError(f"{key}: {inp[key]} not found")

    # .mat transfer matrices need a variable name; .npy does not.
    tm_ext = os.path.splitext(inp['transfer_matrix_path'])[1].lower()
    if tm_ext not in ('.npy', '.mat'):
        raise ValueError(
            f"transfer_matrix_path must be a .npy or .mat file, "
            f"got '{tm_ext}'"
        )
    if tm_ext == '.mat' and inp['transfer_matrix_var'] is None:
        raise ValueError(
            "input.transfer_matrix_var is required when "
            "transfer_matrix_path is a .mat file"
        )

    # Scaling constant gamma applied to T_r before solving.
    scale = inp['transfer_matrix_scale']
    if not isinstance(scale, (int, float)) or scale == 0 or not np.isfinite(scale):
        raise ValueError(
            f"input.transfer_matrix_scale must be a finite non-zero number, "
            f"got {scale}"
        )

    # Optional row-index subset.
    if inp['row_indices_path'] is not None:
        if not os.path.exists(inp['row_indices_path']):
            raise FileNotFoundError(
                f"row_indices_path: {inp['row_indices_path']} not found"
            )
        ri_ext = os.path.splitext(inp['row_indices_path'])[1].lower()
        if ri_ext != '.mat':
            raise ValueError(
                f"input.row_indices_path must be a .mat file, got '{ri_ext}'"
            )
        if inp['row_indices_var'] is None:
            raise ValueError(
                "input.row_indices_var is required when "
                "input.row_indices_path is set"
            )
        if not isinstance(inp['row_indices_one_based'], bool):
            raise ValueError(
                "input.row_indices_one_based must be a boolean, "
                f"got {inp['row_indices_one_based']!r}"
            )
    elif inp['row_indices_var'] is not None:
        raise ValueError(
            "input.row_indices_var is set but input.row_indices_path is not"
        )

    reg = config['regularization']
    if reg['alpha'] is None and reg['alpha_sweep'] is None:
        raise ValueError(
            "Provide regularization.alpha (scalar) or "
            "regularization.alpha_sweep (list); both are missing"
        )
    if reg['alpha'] is not None and reg['alpha_sweep'] is not None:
        raise ValueError(
            "Provide only one of regularization.alpha or "
            "regularization.alpha_sweep"
        )
    if reg['alpha'] is not None:
        if not isinstance(reg['alpha'], (int, float)) or reg['alpha'] < 0:
            raise ValueError(
                f"regularization.alpha must be a non-negative number, "
                f"got {reg['alpha']}"
            )
    if reg['alpha_sweep'] is not None:
        if (not isinstance(reg['alpha_sweep'], list)
                or len(reg['alpha_sweep']) == 0):
            raise ValueError(
                "regularization.alpha_sweep must be a non-empty list"
            )
        for a in reg['alpha_sweep']:
            if not isinstance(a, (int, float)) or a < 0:
                raise ValueError(
                    f"all entries of regularization.alpha_sweep must be "
                    f"non-negative numbers, got {a}"
                )
    if not isinstance(reg['psd_tol_rel'], (int, float)) or reg['psd_tol_rel'] < 0:
        raise ValueError(
            f"regularization.psd_tol_rel must be a non-negative number, "
            f"got {reg['psd_tol_rel']}"
        )

    if config['physics']['frequencies'] is not None:
        config['physics']['frequencies'] = parse_frequencies(
            config['physics']['frequencies']
        )

    out = config['output']
    if out['figure_format'] not in ('png', 'pdf', 'svg', 'eps'):
        raise ValueError(
            f"output.figure_format must be png/pdf/svg/eps, "
            f"got {out['figure_format']}"
        )

    return config


def _load_mat_var(path: str, var: str) -> np.ndarray:
    """Read a single named variable from a MATLAB .mat file."""
    mat = loadmat(path)
    if var not in mat:
        keys = [k for k in mat.keys() if not k.startswith('__')]
        raise KeyError(
            f"variable '{var}' not found in {path}; available keys: {keys}"
        )
    return np.asarray(mat[var])


def load_inputs(
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load reduced transfer matrix, POD basis, and experimental CPSD."""
    inp = config['input']

    tm_ext = os.path.splitext(inp['transfer_matrix_path'])[1].lower()
    if tm_ext == '.mat':
        T_r = _load_mat_var(
            inp['transfer_matrix_path'], inp['transfer_matrix_var']
        )
    else:
        T_r = np.load(inp['transfer_matrix_path'])

    phi = np.load(inp['pod_basis_path'])

    G = _load_mat_var(
        inp['experimental_cpsd_path'], inp['experimental_cpsd_var']
    )
    return T_r, phi, G


def load_row_indices(
    config: Dict[str, Any],
    n_sensors: int,
) -> Optional[np.ndarray]:
    """
    Load and validate the optional row-index subset.

    Returns a 1D int64 array of 0-based, unique, in-range row indices, or
    None if no subset was configured.
    """
    inp = config['input']
    if inp['row_indices_path'] is None:
        return None

    raw = _load_mat_var(inp['row_indices_path'], inp['row_indices_var'])
    idx = np.asarray(raw).squeeze()
    if idx.ndim != 1:
        raise ValueError(
            f"row indices must be 1D (got shape {raw.shape} after squeeze)"
        )
    if idx.size == 0:
        raise ValueError("row indices is empty")
    if not np.issubdtype(idx.dtype, np.number):
        raise ValueError(
            f"row indices must be numeric, got dtype {idx.dtype}"
        )

    idx_int = idx.astype(np.int64)
    if not np.array_equal(idx_int, idx):
        raise ValueError("row indices must be integer-valued")

    if inp['row_indices_one_based']:
        idx_int = idx_int - 1

    if idx_int.min() < 0 or idx_int.max() >= n_sensors:
        raise ValueError(
            f"row indices out of range: must lie in "
            f"[{0 if not inp['row_indices_one_based'] else 1}, "
            f"{n_sensors - 1 if not inp['row_indices_one_based'] else n_sensors}], "
            f"got min={idx.min()}, max={idx.max()} "
            f"(one_based={inp['row_indices_one_based']})"
        )
    if np.unique(idx_int).size != idx_int.size:
        raise ValueError("row indices must be unique")
    return idx_int


def apply_row_subset(
    T_r: np.ndarray,
    G: np.ndarray,
    row_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Restrict T_r and G to the given row index set.

    T_r' = T_r[row_idx, :, :]  and  G' = G[row_idx, :, :][:, row_idx, :].
    """
    T_r_sub = T_r[row_idx, :, :]
    G_sub = G[np.ix_(row_idx, row_idx, np.arange(G.shape[2]))]
    return T_r_sub, G_sub


def validate_shapes(
    T_r: np.ndarray,
    phi: np.ndarray,
    G: np.ndarray,
    frequencies: Optional[List[float]],
) -> None:
    """Cross-check that the three inputs share a consistent set of dims."""
    if T_r.ndim != 3:
        raise ValueError(
            f"transfer matrix must be 3D "
            f"(n_sensors, n_pod, n_freq), got shape {T_r.shape}"
        )
    n_sensors, n_pod, n_freq = T_r.shape

    if phi.ndim != 2 or phi.shape[1] != n_pod:
        raise ValueError(
            f"POD basis must have shape (N, {n_pod}) to match the transfer "
            f"matrix; got {phi.shape}"
        )

    if (G.ndim != 3
            or G.shape[:2] != (n_sensors, n_sensors)
            or G.shape[2] != n_freq):
        raise ValueError(
            f"experimental CPSD must have shape ({n_sensors}, "
            f"{n_sensors}, {n_freq}), got {G.shape}"
        )

    if frequencies is not None and len(frequencies) != n_freq:
        raise ValueError(
            f"physics.frequencies has {len(frequencies)} entries but the "
            f"transfer matrix declares {n_freq} frequencies"
        )


def run_inversion(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load inputs, solve per-frequency, return results in a dict."""
    print("Loading inputs...")
    T_r, phi, G = load_inputs(config)
    frequencies = config['physics']['frequencies']
    validate_shapes(T_r, phi, G, frequencies)

    scale = float(config['input']['transfer_matrix_scale'])
    if scale != 1.0:
        T_r = T_r * scale
        print(f"  Applied transfer_matrix_scale = {scale} to T_r")

    n_sensors_full = T_r.shape[0]
    row_idx = load_row_indices(config, n_sensors_full)
    if row_idx is not None:
        T_r, G = apply_row_subset(T_r, G, row_idx)
        print(
            f"  Row-subset applied: kept {row_idx.size} of "
            f"{n_sensors_full} rows from "
            f"{config['input']['row_indices_path']}"
            f" (one_based={config['input']['row_indices_one_based']})"
        )

    n_sensors, n_pod, n_freq = T_r.shape
    print(f"  T_r shape: {T_r.shape}  (n_sensors, n_pod, n_freq)")
    print(f"  Phi shape: {phi.shape}  (N, n_pod)")
    print(f"  G_hat shape: {G.shape}")
    if frequencies is not None:
        print(
            f"  Frequencies: {frequencies[0]:.2f} ... "
            f"{frequencies[-1]:.2f} Hz ({n_freq})"
        )
    else:
        print(f"  Frequencies: aligned by index across files")

    reg = config['regularization']
    if reg['alpha'] is not None:
        alphas = np.array([reg['alpha']], dtype=np.float64)
        sweep_mode = False
    else:
        alphas = np.array(reg['alpha_sweep'], dtype=np.float64)
        sweep_mode = True
    print(
        f"  Regularization: {'sweep' if sweep_mode else 'scalar'} "
        f"with alphas = {alphas.tolist()}"
    )
    print(f"  PSD eigenvalue clip threshold (relative): {reg['psd_tol_rel']}")

    solver = CPSDInverseSolver(T_r, pod_basis=phi)

    results: Dict[str, Any] = {
        'frequencies': frequencies,
        'alphas': alphas,
        'sweep_mode': sweep_mode,
        'S_r': [],
        'residuals_rel': [],
        'row_indices': None if row_idx is None else row_idx.tolist(),
        'n_sensors_full': int(n_sensors_full),
    }
    for f_idx in range(n_freq):
        S_r, res = solver.solve_single_freq(
            f_idx, G[:, :, f_idx], alphas,
            psd_tol_rel=reg['psd_tol_rel'],
        )
        results['S_r'].append(S_r)
        results['residuals_rel'].append(res)

        if frequencies is not None:
            tag = f"f = {frequencies[f_idx]:8.2f} Hz (idx {f_idx})"
        else:
            tag = f"freq idx {f_idx}"
        if sweep_mode:
            summary = ', '.join(
                f"alpha={a:.2e}:res={r:.3e}" for a, r in zip(alphas, res)
            )
            print(f"  {tag} -> {summary}")
        else:
            print(f"  {tag} -> relative residual {res[0]:.3e}")

    return results


def save_results(results: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Save per-frequency .npz files and a summary.json."""
    out_dir = Path(config['output']['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    frequencies = results['frequencies']
    alphas = results['alphas']
    sweep_mode = results['sweep_mode']

    for f_idx, (S_r, res) in enumerate(
        zip(results['S_r'], results['residuals_rel'])
    ):
        if sweep_mode:
            payload = {
                'S_r': S_r,
                'alphas': alphas,
                'residuals_rel': res,
            }
        else:
            payload = {
                'S_r': S_r[:, :, 0],
                'alpha': float(alphas[0]),
                'residual_rel': float(res[0]),
            }
        if frequencies is not None:
            payload['frequency'] = float(frequencies[f_idx])
        np.savez(out_dir / f'cpsd_inverse_freq{f_idx}.npz', **payload)

    summary: Dict[str, Any] = {
        'n_freq': len(results['residuals_rel']),
        'n_pod': int(results['S_r'][0].shape[0]),
        'sweep_mode': sweep_mode,
        'alphas': alphas.tolist(),
        'pod_basis_path': config['input']['pod_basis_path'],
        'transfer_matrix_path': config['input']['transfer_matrix_path'],
        'transfer_matrix_scale': float(config['input']['transfer_matrix_scale']),
        'experimental_cpsd_path': config['input']['experimental_cpsd_path'],
        'experimental_cpsd_var': config['input']['experimental_cpsd_var'],
        'residual_rel': [r.tolist() for r in results['residuals_rel']],
        'n_sensors_full': results['n_sensors_full'],
        'row_indices_path': config['input']['row_indices_path'],
        'row_indices_var': config['input']['row_indices_var'],
        'row_indices_one_based': config['input']['row_indices_one_based'],
        'row_indices': results['row_indices'],
    }
    if frequencies is not None:
        summary['frequencies'] = list(frequencies)

    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved per-frequency .npz files and summary.json to {out_dir}")


def generate_plots(results: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Generate the residual-vs-frequency diagnostic plot."""
    out_cfg = config['output']
    if not out_cfg['save_figures']:
        return
    out_dir = Path(out_cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    alphas = results['alphas']
    if results['frequencies'] is not None:
        x = np.asarray(results['frequencies'])
        xlabel = 'Frequency [Hz]'
    else:
        x = np.arange(len(results['residuals_rel']))
        xlabel = 'Frequency index'

    res_array = np.array(results['residuals_rel'])  # (n_freq, n_alpha)

    fig, ax = plt.subplots(figsize=(8, 5))
    for k, alpha in enumerate(alphas):
        ax.semilogy(x, res_array[:, k], 'o-', label=f'alpha = {alpha:.2e}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$\|T_r S_r T_r^h - \hat G\|_F / \|\hat G\|_F$')
    ax.set_title('CPSD inversion: relative data-fit residual')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig_path = out_dir / f'residual_vs_frequency.{out_cfg["figure_format"]}'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved residual plot to {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Reduced-basis CPSD Tikhonov inversion driver'
    )
    parser.add_argument(
        'config_file',
        nargs='?',
        default='config_cpsd_inverse.json',
        help='Path to configuration JSON file '
             '(default: config_cpsd_inverse.json)'
    )
    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        print(f"Error: Configuration file '{args.config_file}' not found.")
        return 1

    print(f"Loading configuration from: {args.config_file}")
    config = validate_config(load_config(args.config_file))

    print("\n" + "=" * 60)
    print("CPSD Inverse Problem")
    print("=" * 60)
    results = run_inversion(config)

    print("\nSaving results...")
    save_results(results, config)

    print("\nGenerating plots...")
    generate_plots(results, config)

    print("\nInversion complete.")
    return 0


if __name__ == '__main__':
    exit(main())
