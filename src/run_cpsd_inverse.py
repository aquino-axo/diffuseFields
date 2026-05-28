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
from cpsd_inverse_cv import KFoldCVSelector


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
        'cv': {
            'enabled': False,
            'k_folds': 5,
            'alpha_mode': 'global',     # 'global' or 'per_freq'
            'seed': 0,
            'save_fold_scores': False,
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

    cv = config['cv']
    if not isinstance(cv['enabled'], bool):
        raise ValueError(f"cv.enabled must be a bool, got {cv['enabled']!r}")
    if cv['enabled']:
        if not isinstance(cv['k_folds'], int) or cv['k_folds'] < 2:
            raise ValueError(
                f"cv.k_folds must be an integer >= 2, got {cv['k_folds']}"
            )
        if cv['alpha_mode'] not in ('per_freq', 'global'):
            raise ValueError(
                f"cv.alpha_mode must be 'per_freq' or 'global', "
                f"got {cv['alpha_mode']!r}"
            )
        if not isinstance(cv['seed'], int):
            raise ValueError(f"cv.seed must be an int, got {cv['seed']!r}")
        if not isinstance(cv['save_fold_scores'], bool):
            raise ValueError(
                f"cv.save_fold_scores must be a bool, "
                f"got {cv['save_fold_scores']!r}"
            )
        if reg['alpha'] is not None:
            raise ValueError(
                "cv.enabled=true is incompatible with regularization.alpha "
                "(scalar); supply regularization.alpha_sweep as the CV grid"
            )
        if reg['alpha_sweep'] is None:
            raise ValueError(
                "cv.enabled=true requires regularization.alpha_sweep "
                "(the CV candidate grid)"
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
    cv_cfg = config['cv']
    cv_enabled = cv_cfg['enabled']

    if cv_enabled:
        alphas = np.array(reg['alpha_sweep'], dtype=np.float64)
        sweep_mode = False  # CV refit produces one alpha per frequency
    elif reg['alpha'] is not None:
        alphas = np.array([reg['alpha']], dtype=np.float64)
        sweep_mode = False
    else:
        alphas = np.array(reg['alpha_sweep'], dtype=np.float64)
        sweep_mode = True
    if cv_enabled:
        print(
            f"  Regularization: CV mode (alpha_mode="
            f"{cv_cfg['alpha_mode']!r}, k_folds={cv_cfg['k_folds']}, "
            f"seed={cv_cfg['seed']}) over alpha grid "
            f"{alphas.tolist()}"
        )
    else:
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
        'alphas_per_freq': None,
        'cv': None,
    }

    if cv_enabled:
        selector = KFoldCVSelector(
            solver, G,
            k_folds=cv_cfg['k_folds'],
            seed=cv_cfg['seed'],
            save_fold_scores=cv_cfg['save_fold_scores'],
        )
        print(
            f"  Running CV: {selector.k_folds} folds x "
            f"{solver.n_freq} frequencies x {alphas.size} alphas "
            f"(sensors per fold ~= {solver.n_sensors // selector.k_folds})"
        )
        alpha_star, cv_scores, cv_fold_scores = selector.select(
            alphas, psd_tol_rel=reg['psd_tol_rel'],
            alpha_mode=cv_cfg['alpha_mode'],
        )
        if cv_cfg['alpha_mode'] == 'global':
            alpha_global = float(alpha_star[0])
            print(f"  CV (global) selected alpha = {alpha_global:.3e}")
            alphas_per_freq = np.full(
                n_freq, alpha_global, dtype=np.float64
            )
        else:
            print(
                f"  CV (per-frequency) alpha range: "
                f"[{alpha_star.min():.3e}, {alpha_star.max():.3e}]"
            )
            alphas_per_freq = alpha_star.astype(np.float64)
        results['alphas_per_freq'] = alphas_per_freq.tolist()
        results['cv'] = {
            'enabled': True,
            'k_folds': cv_cfg['k_folds'],
            'alpha_mode': cv_cfg['alpha_mode'],
            'seed': cv_cfg['seed'],
            'alpha_grid': alphas.tolist(),
            'alpha_star': alpha_star.tolist(),
            'scores': cv_scores,
            'fold_scores': cv_fold_scores,
        }

    for f_idx in range(n_freq):
        if cv_enabled:
            alphas_this = np.array(
                [results['alphas_per_freq'][f_idx]], dtype=np.float64
            )
        else:
            alphas_this = alphas
        S_r, res = solver.solve_single_freq(
            f_idx, G[:, :, f_idx], alphas_this,
            psd_tol_rel=reg['psd_tol_rel'],
        )
        results['S_r'].append(S_r)
        results['residuals_rel'].append(res)

        if frequencies is not None:
            tag = f"f = {frequencies[f_idx]:8.2f} Hz (idx {f_idx})"
        else:
            tag = f"freq idx {f_idx}"
        if cv_enabled:
            print(
                f"  {tag} -> alpha*={alphas_this[0]:.2e}, "
                f"refit residual {res[0]:.3e}"
            )
        elif sweep_mode:
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

    alphas_per_freq = results.get('alphas_per_freq')
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
            if alphas_per_freq is not None:
                alpha_value = float(alphas_per_freq[f_idx])
            else:
                alpha_value = float(alphas[0])
            payload = {
                'S_r': S_r[:, :, 0],
                'alpha': alpha_value,
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
    if alphas_per_freq is not None:
        summary['alphas_per_freq'] = list(alphas_per_freq)

    cv_info = results.get('cv')
    if cv_info is not None:
        summary['cv'] = {
            'enabled': cv_info['enabled'],
            'k_folds': cv_info['k_folds'],
            'alpha_mode': cv_info['alpha_mode'],
            'seed': cv_info['seed'],
            'alpha_grid': cv_info['alpha_grid'],
            'alpha_star': cv_info['alpha_star'],
        }

        cv_payload: Dict[str, Any] = {
            'alphas': np.asarray(cv_info['alpha_grid'], dtype=np.float64),
            'scores': cv_info['scores'],
            'alpha_star': np.asarray(
                cv_info['alpha_star'], dtype=np.float64
            ),
            'alpha_mode': cv_info['alpha_mode'],
            'k_folds': cv_info['k_folds'],
            'seed': cv_info['seed'],
        }
        if cv_info['fold_scores'] is not None:
            cv_payload['fold_scores'] = cv_info['fold_scores']
        np.savez(out_dir / 'cv_results.npz', **cv_payload)

    if frequencies is not None:
        summary['frequencies'] = list(frequencies)

    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    if cv_info is not None:
        print(
            f"Saved per-frequency .npz files, cv_results.npz, and "
            f"summary.json to {out_dir}"
        )
    else:
        print(
            f"Saved per-frequency .npz files and summary.json to {out_dir}"
        )


def generate_plots(results: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Generate the residual-vs-frequency plot and (when CV ran) the CV
    score-vs-alpha plot and the CV score heatmap.
    """
    out_cfg = config['output']
    if not out_cfg['save_figures']:
        return
    out_dir = Path(out_cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_format = out_cfg['figure_format']
    alphas = results['alphas']
    sweep_mode = results['sweep_mode']
    cv_info = results.get('cv')

    if results['frequencies'] is not None:
        x = np.asarray(results['frequencies'])
        xlabel = 'Frequency [Hz]'
    else:
        x = np.arange(len(results['residuals_rel']))
        xlabel = 'Frequency index'

    # ---- residual_vs_frequency.{fmt}: post-refit (or per-alpha) residual ----
    res_array = np.array(results['residuals_rel'])  # (n_freq, n_alpha)
    fig, ax = plt.subplots(figsize=(8, 5))
    if sweep_mode:
        for k, alpha in enumerate(alphas):
            ax.semilogy(x, res_array[:, k], 'o-', label=f'alpha = {alpha:.2e}')
    elif cv_info is not None:
        ax.semilogy(
            x, res_array[:, 0], 'o-',
            label=f"alpha* (CV, {cv_info['alpha_mode']})"
        )
    else:
        ax.semilogy(x, res_array[:, 0], 'o-', label=f'alpha = {alphas[0]:.2e}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$\|T_r S_r T_r^h - \hat G\|_F / \|\hat G\|_F$')
    ax.set_title('CPSD inversion: relative data-fit residual')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_path = out_dir / f'residual_vs_frequency.{fig_format}'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved residual plot to {fig_path}")

    if cv_info is None:
        return

    _generate_cv_plots(cv_info, x, xlabel, out_dir, fig_format)


def _generate_cv_plots(
    cv_info: Dict[str, Any],
    x: np.ndarray,
    xlabel: str,
    out_dir: Path,
    fig_format: str,
) -> None:
    """Write cv_score_vs_alpha and cv_score_heatmap."""
    from matplotlib.colors import LogNorm

    cv_scores = np.asarray(cv_info['scores'])         # (n_freq, n_alpha)
    cv_alphas = np.asarray(cv_info['alpha_grid'], dtype=np.float64)
    n_freq = cv_scores.shape[0]
    global_score = cv_scores.mean(axis=0)             # (n_alpha,)
    best_per_f = np.argmin(cv_scores, axis=1)         # (n_freq,)

    # ---- cv_score_vs_alpha.{fmt}: one line per frequency + global mean ----
    fig, ax = plt.subplots(figsize=(8, 5))
    if n_freq > 20:
        ax.loglog(cv_alphas, cv_scores.T, '-', alpha=0.25, color='C0')
    else:
        for f in range(n_freq):
            if x.dtype.kind == 'f':
                lab = f'{x[f]:.0f} Hz'
            else:
                lab = f'idx {int(x[f])}'
            ax.loglog(cv_alphas, cv_scores[f, :], 'o-', label=lab)
    ax.loglog(
        cv_alphas, global_score, 'k-',
        linewidth=2, label='mean over f',
    )
    if cv_info['alpha_mode'] == 'global':
        i = int(np.argmin(global_score))
        ax.axvline(
            cv_alphas[i], color='r', linestyle='--',
            label=f'alpha* = {cv_alphas[i]:.2e}',
        )
    else:
        ax.scatter(
            cv_alphas[best_per_f],
            cv_scores[np.arange(n_freq), best_per_f],
            marker='*', s=70, color='red', zorder=5,
            label='alpha*(f)',
        )
    ax.set_xlabel('alpha')
    ax.set_ylabel('CV score (mean over folds)')
    ax.set_title(f"CV score vs alpha (mode={cv_info['alpha_mode']})")
    ax.grid(True, which='both', alpha=0.3)
    if n_freq <= 20 or cv_info['alpha_mode'] == 'global':
        ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig_path = out_dir / f'cv_score_vs_alpha.{fig_format}'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved CV score-vs-alpha plot to {fig_path}")

    # ---- cv_score_heatmap.{fmt}: pcolormesh on (alpha, frequency) ----
    fig, ax = plt.subplots(figsize=(9, 5))
    eps = 0.0
    if np.any(cv_scores > 0):
        eps = np.min(cv_scores[cv_scores > 0]) * 1e-3
    scores_pos = np.maximum(cv_scores, eps)
    try:
        norm = LogNorm(vmin=scores_pos.min(), vmax=scores_pos.max())
        im = ax.pcolormesh(
            cv_alphas, x, scores_pos, shading='nearest', norm=norm
        )
    except (ValueError, TypeError):
        im = ax.pcolormesh(cv_alphas, x, cv_scores, shading='nearest')
    ax.set_xscale('log')
    ax.set_xlabel('alpha')
    ax.set_ylabel(xlabel)
    ax.set_title('CV score (mean over folds)')
    plt.colorbar(im, ax=ax, label='score')
    ax.plot(
        cv_alphas[best_per_f], x, 'r*', markersize=8, label='alpha*(f)'
    )
    if cv_info['alpha_mode'] == 'global':
        alpha_star = float(np.asarray(cv_info['alpha_star'])[0])
        ax.axvline(
            alpha_star, color='white', linewidth=2, linestyle='--',
            label=f'alpha* = {alpha_star:.2e}',
        )
    ax.legend(loc='best')
    fig.tight_layout()
    fig_path = out_dir / f'cv_score_heatmap.{fig_format}'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved CV score heatmap to {fig_path}")


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
