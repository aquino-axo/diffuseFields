"""
Driver script for plotting the diagonal of the reconstructed (uplifted) CPSD
as a function of frequency, optionally compared against a validation data set.

Consumes the diagonal `.npy` file (shape `(N, n_freq)`, real) produced by
`run_reconstruct_full_cpsd.py` in `mode='diagonal'`, together with its
sidecar JSON. Entries to plot can be specified by integer indices into the
sideset-face dimension and/or by physical (x, y, z) coordinates, in which
case the nearest sideset face centroid is used.

Three plot kinds are supported via `plot.kind` (a string or a list):

  * ``"lines"`` - per-location autopower vs frequency. The inverse solution is
    drawn solid and the validation data dashed, sharing one colour per
    location. Without a validation file this reproduces the original
    solution-only line plot.
  * ``"box"``   - at each frequency, the distribution of the diagonal values
    *across the selected locations*, shown as side-by-side boxes (solution vs
    validation). Box = 25-75th percentile, whiskers = 5th/95th percentile.
    When the number of frequencies exceeds ``BAND_FREQ_THRESHOLD`` the plot
    automatically switches to median + shaded percentile bands.
  * ``"error"`` - per-location relative-L2 error of the solution autopower
    spectrum against the validation spectrum, sorted worst -> best as a bar
    chart. ``output.top_n`` optionally caps the number of bars.

The ``box`` and ``error`` kinds require a validation file. A validation file
requires the selection to be given as ``coordinates`` (validation row k is
aligned to the k-th coordinate).

Usage:
    python run_plot_cpsd_diagonal.py config_plot_cpsd_diagonal.json
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# Above this many frequencies, the box kind renders percentile bands instead
# of discrete side-by-side boxes (which would be unreadable).
BAND_FREQ_THRESHOLD = 40

VALID_KINDS = ('lines', 'box', 'error')

# Imported lazily (and patchable in tests) so the module loads without the
# optional `exodusii` dependency when coordinate selection is not used.
ExodusSideInterpolator = None


def _get_interpolator():
    global ExodusSideInterpolator
    if ExodusSideInterpolator is None:
        from exodus_side_interpolator import ExodusSideInterpolator as _Cls
        ExodusSideInterpolator = _Cls
    return ExodusSideInterpolator


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
            'validation_path': None,
            'validation_var': None,
        },
        'selection': {
            'indices': None,
            'coordinates': None,
            'match_tolerance': None,
        },
        'plot': {
            'kind': 'lines',
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
            'top_n': None,
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

    # --- validation input -------------------------------------------------
    has_validation = inp['validation_path'] is not None
    if has_validation:
        if not os.path.exists(inp['validation_path']):
            raise FileNotFoundError(
                f"input.validation_path not found: {inp['validation_path']}"
            )
        val_ext = os.path.splitext(inp['validation_path'])[1].lower()
        if val_ext not in ('.npy', '.mat'):
            raise ValueError(
                "input.validation_path must be a .npy or .mat file, got "
                f"'{val_ext}'"
            )
        if val_ext == '.mat' and inp['validation_var'] is None:
            raise ValueError(
                "input.validation_var is required when input.validation_path "
                "is a .mat file"
            )

    # --- plot kinds -------------------------------------------------------
    kind = config['plot']['kind']
    kinds = [kind] if isinstance(kind, str) else list(kind)
    if not kinds:
        raise ValueError("plot.kind must name at least one kind")
    for k in kinds:
        if k not in VALID_KINDS:
            raise ValueError(
                f"plot.kind entries must be in {VALID_KINDS}; got '{k}'"
            )
    config['plot']['kind'] = kinds  # normalise to list

    needs_validation = [k for k in kinds if k in ('box', 'error')]
    if needs_validation and not has_validation:
        raise ValueError(
            f"plot kinds {needs_validation} require input.validation_path"
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

    out_cfg = config['output']
    if out_cfg['top_n'] is not None:
        if not isinstance(out_cfg['top_n'], int) or out_cfg['top_n'] <= 0:
            raise ValueError("output.top_n must be null or a positive int")

    sel = config['selection']
    if sel['match_tolerance'] is not None:
        if (
            not isinstance(sel['match_tolerance'], (int, float))
            or sel['match_tolerance'] <= 0
        ):
            raise ValueError(
                "selection.match_tolerance must be null or a positive number"
            )

    if sel['indices'] is None and sel['coordinates'] is None:
        raise ValueError(
            "selection requires at least one of 'indices' or 'coordinates' "
            "(use 'indices': 'all' to plot every entry)"
        )

    # Validation alignment is by coordinate order, so coordinates are required.
    if has_validation and sel['coordinates'] is None:
        raise ValueError(
            "selection.coordinates is required when input.validation_path is "
            "set (validation row k is aligned to the k-th coordinate)"
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
) -> Tuple[np.ndarray, np.ndarray, str, List[int]]:
    """
    Load the diagonal array and resolve the frequency axis.

    Returns
    -------
    diag : ndarray, shape (N, n_freq), real
    freq_axis : ndarray, shape (n_freq,)
        Physical frequencies if available in the sidecar; otherwise the
        frequency indices.
    xlabel : str
    freq_indices : list of int
        The reconstructed frequency indices recorded in the sidecar (used to
        align an external validation data set indexed by the full frequency
        set).
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

    return diag, freq_axis, xlabel, [int(i) for i in freq_indices]


def _load_array(path: str, var: Optional[str]) -> np.ndarray:
    """Load an ndarray from a .npy file or a named variable in a .mat file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path)
    mat = loadmat(path)
    if var not in mat:
        keys = [k for k in mat.keys() if not k.startswith('__')]
        raise KeyError(
            f"variable '{var}' not found in {path}; available keys: {keys}"
        )
    return np.asarray(mat[var])


def load_validation_diagonal(
    path: str, var: Optional[str], freq_indices: List[int]
) -> np.ndarray:
    """
    Load the validation full CPSD ``(n_loc, n_loc, n_freq_full)`` complex,
    extract its real diagonal, and slice it to the reconstructed frequency
    indices.

    Returns
    -------
    val_diag : ndarray, shape (n_loc, n_freq), real
        Row k is the autopower spectrum of validation location k at the
        reconstructed frequencies.
    """
    arr = _load_array(path, var)
    if arr.ndim != 3 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            "Validation CPSD must have shape (n_loc, n_loc, n_freq_full); got "
            f"{arr.shape}"
        )

    n_freq_full = arr.shape[2]
    max_idx = max(freq_indices)
    if max_idx >= n_freq_full:
        raise ValueError(
            f"Validation has {n_freq_full} frequencies but the solution "
            f"references frequency index {max_idx}; validation must span the "
            "full frequency set used in the inversion."
        )

    # diagonal over the first two axes -> (n_freq_full, n_loc); make (n_loc, f)
    diag_full = np.diagonal(arr, axis1=0, axis2=1).real.T
    return diag_full[:, freq_indices]


def resolve_selection(
    config: Dict[str, Any],
    n_entries: int,
    validation_mode: bool = False,
) -> List[Tuple[int, str]]:
    """
    Convert config selection into a list of (face_index, label) pairs.

    Coordinate-based entries are resolved to the nearest sideset face centroid
    via a brute-force nearest-neighbour search (n_faces is small).

    In ``validation_mode`` the per-coordinate ordering must be preserved (row k
    of the validation data aligns to coordinate k), so indices are *not*
    deduplicated: two coordinates resolving to the same face raise an error,
    and a nearest-match distance exceeding ``selection.match_tolerance`` (when
    set) also raises.
    """
    sel = config['selection']
    inp = config['input']

    chosen: List[Tuple[int, str]] = []
    seen: set = set()

    def _add(idx: int, label: str) -> None:
        if idx in seen:
            if validation_mode:
                raise ValueError(
                    f"Two validation coordinates resolve to the same sideset "
                    f"face {idx}; alignment would be ambiguous. Provide "
                    f"distinct coordinates."
                )
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
        tol = sel.get('match_tolerance')
        with _get_interpolator()(inp['exodus_file'], mode='r') as db:
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
            dist = float(distances[nearest])
            if tol is not None and dist > tol:
                raise ValueError(
                    f"Coordinate {target.tolist()} nearest sideset face is "
                    f"{dist:.4g} away, exceeding selection.match_tolerance="
                    f"{tol}."
                )
            centroid = centroids[nearest]
            label = (
                f'node {nearest} '
                f'(target=[{target[0]:.3g},{target[1]:.3g},{target[2]:.3g}], '
                f'd={dist:.3g})'
            )
            _add(nearest, label)
            print(
                f"  coord {target.tolist()} -> face {nearest} at "
                f"{centroid.tolist()} (distance={dist:.4g})"
            )

    return chosen


def _kind_path(out_cfg: Dict[str, Any], kind: str, n_kinds: int) -> Path:
    """Resolve the output path for a plot kind, suffixing when >1 kind."""
    fig_path = Path(out_cfg['figure_path'])
    if not fig_path.suffix:
        fig_path = fig_path.with_suffix(f".{out_cfg['figure_format']}")
    if n_kinds > 1:
        fig_path = fig_path.with_name(
            f"{fig_path.stem}_{kind}{fig_path.suffix}"
        )
    return fig_path


def _save_fig(fig, fig_path: Path, dpi: int) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved plot to {fig_path}")


def plot_lines(
    sol_sel: np.ndarray,
    val_sel: Optional[np.ndarray],
    freq_axis: np.ndarray,
    selection: List[Tuple[int, str]],
    xlabel: str,
    config: Dict[str, Any],
    fig_path: Path,
) -> None:
    """Per-location autopower vs frequency: solution solid, validation dashed."""
    plot_cfg = config['plot']
    out_cfg = config['output']
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)

    fig, ax = plt.subplots(figsize=tuple(plot_cfg['figsize']))
    plotter = ax.semilogy if plot_cfg['log_scale'] else ax.plot
    for k, (idx, label) in enumerate(selection):
        color = prop_cycle[k % len(prop_cycle)] if prop_cycle else None
        plotter(freq_axis, sol_sel[k, :], '-', color=color, label=label)
        if val_sel is not None:
            plotter(freq_axis, val_sel[k, :], '--', color=color)

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
    if val_sel is not None:
        # Linestyle key independent of the per-location colour legend.
        ax.plot([], [], 'k-', label='solution')
        ax.plot([], [], 'k--', label='validation')
        ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    _save_fig(fig, fig_path, out_cfg['dpi'])

    if out_cfg['save_selection_csv']:
        csv_path = fig_path.with_suffix('.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [xlabel]
            for idx, _ in selection:
                header.append(f'sol_idx_{idx}')
                if val_sel is not None:
                    header.append(f'val_idx_{idx}')
            writer.writerow(header)
            for j, freq in enumerate(freq_axis):
                row = [float(freq)]
                for k, (idx, _) in enumerate(selection):
                    row.append(float(sol_sel[k, j]))
                    if val_sel is not None:
                        row.append(float(val_sel[k, j]))
                writer.writerow(row)
        print(f"Saved CSV to {csv_path}")


# Percentiles used for the box / band statistics.
P_LO, P_BOX_LO, P_MED, P_BOX_HI, P_HI = 5, 25, 50, 75, 95


def box_render_mode(n_freq: int) -> str:
    """'boxes' for few frequencies, 'bands' once it would be unreadable."""
    return 'boxes' if n_freq <= BAND_FREQ_THRESHOLD else 'bands'


def plot_box(
    sol_sel: np.ndarray,
    val_sel: np.ndarray,
    freq_axis: np.ndarray,
    xlabel: str,
    config: Dict[str, Any],
    fig_path: Path,
) -> None:
    """Distribution across selected locations at each frequency (sol vs val)."""
    plot_cfg = config['plot']
    out_cfg = config['output']
    n_freq = sol_sel.shape[1]

    sol_color, val_color = 'tab:blue', 'tab:orange'
    fig, ax = plt.subplots(figsize=tuple(plot_cfg['figsize']))

    if box_render_mode(n_freq) == 'boxes':
        # Side-by-side discrete boxes on categorical positions.
        positions = np.arange(n_freq)
        width = 0.35
        whis = (P_LO, P_HI)
        for data, off, color, name in (
            (val_sel, -width / 2, val_color, 'validation'),
            (sol_sel, +width / 2, sol_color, 'solution'),
        ):
            bp = ax.boxplot(
                [data[:, j] for j in range(n_freq)],
                positions=positions + off,
                widths=width,
                whis=whis,
                showfliers=False,
                patch_artist=True,
                manage_ticks=False,
            )
            for box in bp['boxes']:
                box.set(facecolor=color, alpha=0.5)
            for med in bp['medians']:
                med.set(color='black')
            bp['boxes'][0].set_label(name)
        ax.set_xticks(positions)
        ax.set_xticklabels([f'{f:g}' for f in freq_axis], rotation=45,
                           ha='right', fontsize=7)
    else:
        # Median + shaded percentile bands.
        for data, color, name in (
            (val_sel, val_color, 'validation'),
            (sol_sel, sol_color, 'solution'),
        ):
            p = np.percentile(
                data, [P_LO, P_BOX_LO, P_MED, P_BOX_HI, P_HI], axis=0
            )
            ax.plot(freq_axis, p[2], '-', color=color, label=name)
            ax.fill_between(freq_axis, p[1], p[3], color=color, alpha=0.30)
            ax.fill_between(freq_axis, p[0], p[4], color=color, alpha=0.15)

    if plot_cfg['log_scale']:
        ax.set_yscale('log')
    ax.set_xlabel(plot_cfg['xlabel'] or xlabel)
    ax.set_ylabel(plot_cfg['ylabel'])
    ax.set_title(plot_cfg['title'])
    if plot_cfg['ylim'] is not None:
        ax.set_ylim(plot_cfg['ylim'])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    _save_fig(fig, fig_path, out_cfg['dpi'])

    if out_cfg['save_selection_csv']:
        csv_path = fig_path.with_suffix('.csv')
        pct = [P_LO, P_BOX_LO, P_MED, P_BOX_HI, P_HI]
        sol_p = np.percentile(sol_sel, pct, axis=0)
        val_p = np.percentile(val_sel, pct, axis=0)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [xlabel]
            for series in ('val', 'sol'):
                header += [f'{series}_p{p}' for p in pct]
            writer.writerow(header)
            for j, freq in enumerate(freq_axis):
                row = [float(freq)]
                row += [float(v) for v in val_p[:, j]]
                row += [float(v) for v in sol_p[:, j]]
                writer.writerow(row)
        print(f"Saved CSV to {csv_path}")


def relative_l2_error(
    sol_sel: np.ndarray, val_sel: np.ndarray
) -> np.ndarray:
    """Per-location relative-L2 error over frequency: ||sol-val|| / ||val||."""
    num = np.linalg.norm(sol_sel - val_sel, axis=1)
    den = np.linalg.norm(val_sel, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        err = np.where(den > 0, num / den, np.inf)
    return err


def plot_error(
    sol_sel: np.ndarray,
    val_sel: np.ndarray,
    selection: List[Tuple[int, str]],
    config: Dict[str, Any],
    fig_path: Path,
) -> None:
    """Per-location relative-L2 error, sorted worst -> best, as a bar chart."""
    plot_cfg = config['plot']
    out_cfg = config['output']

    err = relative_l2_error(sol_sel, val_sel)
    order = np.argsort(err)[::-1]  # worst (largest) first
    top_n = out_cfg['top_n']
    if top_n is not None and top_n < len(order):
        print(
            f"  error: showing worst {top_n} of {len(order)} locations "
            f"(output.top_n)"
        )
        order = order[:top_n]

    face_idx = [selection[k][0] for k in order]
    values = err[order]
    labels = [str(selection[k][0]) for k in order]

    fig, ax = plt.subplots(figsize=tuple(plot_cfg['figsize']))
    positions = np.arange(len(order))
    ax.bar(positions, values, color='tab:red', alpha=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlabel('Sideset face index (worst -> best)')
    ax.set_ylabel(r'Relative $L_2$ error  $\|S^{sol}-S^{val}\|_2/\|S^{val}\|_2$')
    ax.set_title(plot_cfg['title'])
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, fig_path, out_cfg['dpi'])

    if out_cfg['save_selection_csv']:
        csv_path = fig_path.with_suffix('.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['rank', 'face_index', 'relative_l2_error'])
            for rank, (fi, v) in enumerate(zip(face_idx, values)):
                writer.writerow([rank, int(fi), float(v)])
        print(f"Saved CSV to {csv_path}")


def run(config: Dict[str, Any]) -> None:
    inp = config['input']
    kinds = config['plot']['kind']

    diag, freq_axis, xlabel, freq_indices = load_diagonal_data(
        inp['diagonal_npy_path'], inp['sidecar_json_path']
    )
    print(
        f"Loaded diagonal: shape={diag.shape}, "
        f"freq range=[{freq_axis.min():.4g}, {freq_axis.max():.4g}]"
    )

    has_validation = inp['validation_path'] is not None

    selection = resolve_selection(
        config, diag.shape[0], validation_mode=has_validation
    )
    if not selection:
        raise ValueError("Selection resolved to an empty set of indices.")
    print(f"Selected {len(selection)} diagonal entries.")

    # Solution autopower spectra for the selected faces, in selection order.
    sel_idx = [idx for idx, _ in selection]
    sol_sel = diag[sel_idx, :]

    val_sel = None
    if has_validation:
        val_diag = load_validation_diagonal(
            inp['validation_path'], inp['validation_var'], freq_indices
        )
        if val_diag.shape[0] != len(selection):
            raise ValueError(
                f"Validation has {val_diag.shape[0]} locations but "
                f"{len(selection)} coordinates were selected; row k of the "
                "validation data must align with coordinate k."
            )
        if val_diag.shape[1] != sol_sel.shape[1]:
            raise ValueError(
                f"Validation frequency count {val_diag.shape[1]} does not "
                f"match solution {sol_sel.shape[1]} after slicing."
            )
        val_sel = val_diag
        print(f"Loaded validation diagonal: shape={val_diag.shape}")

    n_kinds = len(kinds)
    for kind in kinds:
        fig_path = _kind_path(config['output'], kind, n_kinds)
        if kind == 'lines':
            plot_lines(sol_sel, val_sel, freq_axis, selection, xlabel,
                       config, fig_path)
        elif kind == 'box':
            plot_box(sol_sel, val_sel, freq_axis, xlabel, config, fig_path)
        elif kind == 'error':
            plot_error(sol_sel, val_sel, selection, config, fig_path)


def main():
    parser = argparse.ArgumentParser(
        description='Plot CPSD diagonal entries vs frequency, optionally '
                    'compared against a validation data set.'
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
