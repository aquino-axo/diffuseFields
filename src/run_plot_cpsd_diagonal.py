"""
Driver script for plotting the diagonal of the reconstructed (uplifted) CPSD
as a function of frequency, optionally compared against a validation data set.

Consumes the diagonal `.npy` file (shape `(N, n_freq)`, real) produced by
`run_reconstruct_full_cpsd.py` in `mode='diagonal'`, together with its
sidecar JSON. Entries to plot can be specified by integer indices into the
sideset-face dimension and/or by physical (x, y, z) coordinates, in which
case the nearest sideset face centroid is used.

Five plot kinds are supported via `plot.kind` (a string or a list):

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
  * ``"validation_db"`` - stacked two-panel dB comparison. The top panel shows
    the autopower level ``L = 10*log10(S_ii / db_ref)`` for the computed
    (solid) and measured (dashed) spectra; the bottom panel shows the signed
    level error ``dL = 10*log10(S_meas / S_comp)`` per location with a
    highlight box reporting ``max|dL|`` and ``median|dL|`` (the reference
    cancels in the error, so it needs none). A combined "all sensors" figure
    is written to ``output.figure_path``; per-sensor figures go to a
    ``per_sensor/`` subdirectory (``output.per_sensor``), capped worst-first
    by ``output.top_n``. On an independent frequency grid only the top panel
    is drawn (see below), and ``output.top_n`` then caps in selection order.
  * ``"envelope"`` - min-max spread *across all selected sensors* at each
    frequency, in dB, as a shaded band per series (computed and measured) with
    the **energetic mean** ``10*log10(mean_i(S_ii)/db_ref)`` as its centre
    line. The statistic is taken within each series, so this kind needs no
    paired frequencies and works on an independent grid. Without a validation
    file only the computed band is drawn. Aggregate by construction, so
    ``output.per_sensor`` and ``output.top_n`` do not apply.

The ``box``, ``error`` and ``validation_db`` kinds require a validation file.
A validation file requires the selection to be given as ``coordinates``
(validation row k is aligned to the k-th coordinate).

The validation data's frequency axis is handled in one of two modes:

  * **Shared grid** (``input.validation_frequencies`` unset, the default). The
    validation array spans the full frequency set used in the inversion and is
    sliced with the sidecar's ``freq_indices``. All four kinds are available.
  * **Independent grid** (``input.validation_frequencies`` set to a path, an
    inline list, or ``{min, step, max}``). The validation data carries its own
    frequency vector and is not sliced; each series is drawn against its own
    frequencies. ``lines``, ``envelope`` and ``validation_db`` all work here --
    the last rendering its dB overlay panel alone, since the level error
    ``dL`` needs paired frequencies. ``box`` and ``error`` difference the two
    spectra outright and are rejected, rather than papered over by
    interpolating the measurement onto the solution's grid.

Usage:
    python run_plot_cpsd_diagonal.py config_plot_cpsd_diagonal.json
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from frequency_spec import load_frequency_spec

# Above this many frequencies, the box kind renders percentile bands instead
# of discrete side-by-side boxes (which would be unreadable).
BAND_FREQ_THRESHOLD = 40

VALID_KINDS = ('lines', 'box', 'error', 'validation_db', 'envelope')

# Kinds that cannot be drawn without a validation data set. `lines` and
# `envelope` are absent: both degrade to the solution series alone.
REQUIRES_VALIDATION_KINDS = ('box', 'error', 'validation_db')

# Kinds that *difference* the two spectra and so need both on one grid: the
# relative-L2 error, and the per-frequency distributions drawn side by side on
# categorical positions. `validation_db` is deliberately absent — on an
# independent grid it degrades to its dB overlay panel, which needs no pairing.
DIFFERENCE_KINDS = ('box', 'error')

# Frequency-axis labels; also used to tell the two axis kinds apart.
FREQ_HZ_LABEL = 'Frequency [Hz]'
FREQ_INDEX_LABEL = 'Frequency index'

# Long-format headers, used when the solution and the validation data may
# sample different frequencies so a shared frequency column is ill-defined.
LINES_CSV_HEADER = ('series', 'frequency', 'index', 'label', 'value')
DB_CSV_HEADER = ('series', 'frequency', 'index', 'label', 'level_db')
ENVELOPE_CSV_HEADER = ('series', 'frequency', 'statistic', 'level_db')

# On the combined validation_db overlay, show the per-sensor colour legend only
# when the sensor count is at or below this; above it, only the computed/
# measured linestyle key is drawn (colours still convey spread).
COMBINED_LEGEND_MAX = 10

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
            'validation_frequencies': None,
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
            'db_ref': 1.0,
            'db_floor': 1e-12,
        },
        'output': {
            'figure_path': 'results_cpsd_inverse/diagonal_vs_frequency.png',
            'figure_format': 'png',
            'dpi': 150,
            'save_selection_csv': False,
            'top_n': None,
            'per_sensor': True,
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
    elif inp['validation_frequencies'] is not None:
        raise ValueError(
            "input.validation_frequencies was supplied without "
            "input.validation_path; there is no validation data to place on "
            "that grid"
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

    needs_validation = [k for k in kinds if k in REQUIRES_VALIDATION_KINDS]
    if needs_validation and not has_validation:
        raise ValueError(
            f"plot kinds {needs_validation} require input.validation_path"
        )
    needs_one_grid = [k for k in kinds if k in DIFFERENCE_KINDS]
    if needs_one_grid and inp['validation_frequencies'] is not None:
        raise ValueError(
            f"plot kinds {needs_one_grid} difference the solution against the "
            f"validation data, so both must lie on one frequency grid, but "
            f"input.validation_frequencies was supplied (independent grid). "
            f"Drop those kinds ('lines' and 'validation_db' both work on "
            f"independent grids), or drop input.validation_frequencies."
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

    for key in ('db_ref', 'db_floor'):
        val = plot_cfg.get(key)
        if not isinstance(val, (int, float)) or isinstance(val, bool) \
                or val <= 0:
            raise ValueError(f"plot.{key} must be a positive number")

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
        xlabel = FREQ_HZ_LABEL
    else:
        freq_axis = np.asarray(freq_indices, dtype=float)
        xlabel = FREQ_INDEX_LABEL

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
    path: str, var: Optional[str], freq_indices: Optional[List[int]]
) -> np.ndarray:
    """
    Load the validation full CPSD ``(n_loc, n_loc, n_freq_full)`` complex and
    extract its real diagonal.

    With ``freq_indices`` given (shared-grid mode) the diagonal is sliced to
    those reconstructed frequency indices. With ``freq_indices=None``
    (independent-grid mode) the validation data carries its own frequency
    vector, so every stored frequency is returned unsliced.

    Returns
    -------
    val_diag : ndarray, shape (n_loc, n_freq), real
        Row k is the autopower spectrum of validation location k.
    """
    arr = _load_array(path, var)
    if arr.ndim != 3 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            "Validation CPSD must have shape (n_loc, n_loc, n_freq_full); got "
            f"{arr.shape}"
        )

    # diagonal over the first two axes -> (n_freq_full, n_loc); make (n_loc, f)
    diag_full = np.diagonal(arr, axis1=0, axis2=1).real.T
    if freq_indices is None:
        return diag_full

    n_freq_full = arr.shape[2]
    max_idx = max(freq_indices)
    if max_idx >= n_freq_full:
        raise ValueError(
            f"Validation has {n_freq_full} frequencies but the solution "
            f"references frequency index {max_idx}; validation must span the "
            "full frequency set used in the inversion."
        )
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


def _legend_outside(ax, **kwargs):
    """
    Place the legend just right of the axes instead of on top of the data.

    With several sensors -- each contributing a colour entry, plus the
    solid/dashed source key -- an in-axes legend covers the curves it is
    labelling. Anchoring outside keeps the data area clear at any sensor count.
    Figures are saved with ``bbox_inches='tight'`` so the legend is not clipped.
    """
    kwargs.setdefault('fontsize', 8)
    return ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                     borderaxespad=0.0, **kwargs)


def _save_fig(fig, fig_path: Path, dpi: int) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches='tight' so a legend anchored outside the axes is included.
    fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
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
    val_freq_axis: Optional[np.ndarray] = None,
) -> None:
    """
    Per-location autopower vs frequency: solution solid, validation dashed.

    ``val_freq_axis`` is the validation data's own frequency vector; ``None``
    means it shares the solution's grid.
    """
    plot_cfg = config['plot']
    out_cfg = config['output']
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)
    val_freq = freq_axis if val_freq_axis is None else val_freq_axis

    fig, ax = plt.subplots(figsize=tuple(plot_cfg['figsize']))
    plotter = ax.semilogy if plot_cfg['log_scale'] else ax.plot
    for k, (idx, label) in enumerate(selection):
        color = prop_cycle[k % len(prop_cycle)] if prop_cycle else None
        plotter(freq_axis, sol_sel[k, :], '-', color=color, label=label)
        if val_sel is not None:
            plotter(val_freq, val_sel[k, :], '--', color=color)

    ax.set_xlabel(plot_cfg['xlabel'] or xlabel)
    ax.set_ylabel(plot_cfg['ylabel'])
    ax.set_title(plot_cfg['title'])
    if plot_cfg['ylim'] is not None:
        ax.set_ylim(plot_cfg['ylim'])
    if plot_cfg['xlim'] is not None:
        ax.set_xlim(plot_cfg['xlim'])
    ax.grid(True, alpha=0.3)
    if val_sel is not None:
        # Linestyle key independent of the per-location colour legend.
        ax.plot([], [], 'k-', label='solution')
        ax.plot([], [], 'k--', label='validation')
    if val_sel is not None or len(selection) <= 20:
        _legend_outside(ax)
    fig.tight_layout()
    _save_fig(fig, fig_path, out_cfg['dpi'])

    if out_cfg['save_selection_csv']:
        csv_path = fig_path.with_suffix('.csv')
        # Long format: each series carries its own frequency grid, so a shared
        # frequency column would not be well defined.
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(LINES_CSV_HEADER)
            series = [('solution', sol_sel, freq_axis)]
            if val_sel is not None:
                series.append(('validation', val_sel, val_freq))
            for name, data, axis in series:
                for k, (idx, label) in enumerate(selection):
                    for j, freq in enumerate(axis):
                        writer.writerow(
                            [name, float(freq), idx, label, float(data[k, j])]
                        )
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
    _legend_outside(ax)
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


# --------------------------------------------------------------------------
# dB helpers, shared by the validation_db and envelope kinds
# --------------------------------------------------------------------------

def to_db(values: np.ndarray, ref: float, floor_rel: float
          ) -> Tuple[np.ndarray, int]:
    """
    Convert power values ``(n_loc, n_freq)`` to level ``10*log10(S/ref)`` in dB.

    Each location (row) is floored at ``floor_rel * row_peak`` before the log
    so that non-positive or vanishing entries yield a finite, bounded-below
    level instead of ``-inf``. Returns the levels and the number of samples
    that hit the floor.
    """
    values = np.asarray(values, dtype=float)
    peaks = np.max(values, axis=1, keepdims=True)
    floors = np.maximum(floor_rel * peaks, np.finfo(float).tiny)
    n_clamped = int(np.count_nonzero(values < floors))
    clamped = np.maximum(values, floors)
    return 10.0 * np.log10(clamped / ref), n_clamped


def db_error(meas: np.ndarray, comp: np.ndarray, floor_rel: float
             ) -> Tuple[np.ndarray, int]:
    """
    Signed level error ``dL = 10*log10(S_meas / S_comp)`` in dB, i.e.
    ``10*log10(S_meas) - 10*log10(S_comp)``. The reference cancels, so this is
    computed with ref = 1. Both operands are floored as in :func:`to_db`.
    Returns the error and the total number of floored samples.
    """
    l_meas, n1 = to_db(meas, 1.0, floor_rel)
    l_comp, n2 = to_db(comp, 1.0, floor_rel)
    return l_meas - l_comp, n1 + n2


def db_error_stats(dL: np.ndarray) -> Tuple[float, float]:
    """``(max|dL|, median|dL|)`` over all samples of ``dL`` (pooled)."""
    a = np.abs(np.asarray(dL, dtype=float))
    return float(np.max(a)), float(np.median(a))


def select_per_sensor_order(dL_all: np.ndarray, top_n: Optional[int]
                            ) -> Tuple[np.ndarray, int]:
    """
    Order sensors worst-error first by per-sensor ``max|dL|`` and apply the
    ``top_n`` cap. Returns the (possibly truncated) index order and the number
    of sensors skipped by the cap.
    """
    per_sensor_max = np.max(np.abs(dL_all), axis=1)
    order = np.argsort(per_sensor_max)[::-1]
    n_total = len(order)
    if top_n is not None and top_n < n_total:
        return order[:top_n], n_total - top_n
    return order, 0


def _draw_db_overlay(
    ax,
    l_comp: np.ndarray,
    l_meas: np.ndarray,
    sol_freq: np.ndarray,
    val_freq: np.ndarray,
    selection: List[Tuple[int, str]],
    plot_cfg: Dict[str, Any],
    title: str,
    show_sensor_legend: bool,
    single_sensor: bool,
) -> None:
    """
    Draw the computed-vs-measured level overlay.

    ``sol_freq`` and ``val_freq`` are per-series, so the two may be sampled on
    different grids; this panel never pairs them.
    """
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)

    if single_sensor:
        # One sensor: distinguish source by colour and linestyle.
        ax.plot(sol_freq, l_comp[0], '-', color='tab:blue', label='computed')
        ax.plot(val_freq, l_meas[0], '--', color='tab:orange', label='measured')
    else:
        # Many sensors: colour = sensor, linestyle = source.
        for k, (idx, label) in enumerate(selection):
            color = prop_cycle[k % len(prop_cycle)] if prop_cycle else None
            ax.plot(sol_freq, l_comp[k], '-', color=color,
                    label=(label if show_sensor_legend else None))
            ax.plot(val_freq, l_meas[k], '--', color=color)
        # Linestyle key independent of the per-sensor colour legend.
        ax.plot([], [], 'k-', label='computed')
        ax.plot([], [], 'k--', label='measured')

    ax.set_ylabel(f"{plot_cfg['ylabel']} level [dB re {plot_cfg['db_ref']:g}]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _legend_outside(ax)
    if plot_cfg['ylim'] is not None:
        ax.set_ylim(plot_cfg['ylim'])


def _draw_db_error(
    ax,
    dL: np.ndarray,
    freq_axis: np.ndarray,
    selection: List[Tuple[int, str]],
    plot_cfg: Dict[str, Any],
    stats: Tuple[float, float],
    single_sensor: bool,
) -> None:
    """Draw the signed level-error panel and its max/median highlight box."""
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)

    if single_sensor:
        ax.plot(freq_axis, dL[0], '-', color='tab:red')
    else:
        for k in range(len(selection)):
            color = prop_cycle[k % len(prop_cycle)] if prop_cycle else None
            ax.plot(freq_axis, dL[k], '-', color=color)

    ax.axhline(0.0, color='k', lw=0.8, alpha=0.6)
    ax.set_ylabel(r'$S_{ii}$ error (measured $-$ computed) [dB]')
    ax.grid(True, alpha=0.3)
    max_abs, med_abs = stats
    box_text = (f"max |error| = {max_abs:.2f} dB\n"
                f"median |error| = {med_abs:.2f} dB")
    ax.text(0.98, 0.95, box_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))


def _render_db_figure(
    sol: np.ndarray,
    val: np.ndarray,
    freq_axis: np.ndarray,
    selection: List[Tuple[int, str]],
    xlabel: str,
    plot_cfg: Dict[str, Any],
    title: str,
    show_sensor_legend: bool,
    single_sensor: bool = False,
    val_freq_axis: Optional[np.ndarray] = None,
):
    """
    Draw the measured-vs-computed dB figure.

    On a shared grid this is the stacked overlay (top) + signed level error
    (bottom) pair. When ``val_freq_axis`` is given the two spectra sample
    different frequencies, so the level error is undefined: the figure is the
    overlay panel alone and ``db_error`` is never called.

    Returns ``(fig, stats, n_clamped)`` where ``stats`` is
    ``(max_abs, median_abs)`` on a shared grid and ``None`` otherwise.
    """
    db_ref = plot_cfg['db_ref']
    db_floor = plot_cfg['db_floor']
    l_comp, nc1 = to_db(sol, db_ref, db_floor)
    l_meas, nc2 = to_db(val, db_ref, db_floor)
    independent_grid = val_freq_axis is not None
    val_freq = freq_axis if val_freq_axis is None else val_freq_axis

    if independent_grid:
        fig, ax_top = plt.subplots(figsize=tuple(plot_cfg['figsize']))
        ax_bottom = ax_top  # the overlay carries the x-axis labelling
        stats = None
    else:
        fig, (ax_top, ax_err) = plt.subplots(
            2, 1, sharex=True, figsize=tuple(plot_cfg['figsize'])
        )
        ax_bottom = ax_err
        dL, _ = db_error(val, sol, db_floor)
        stats = db_error_stats(dL)

    _draw_db_overlay(ax_top, l_comp, l_meas, freq_axis, val_freq, selection,
                     plot_cfg, title, show_sensor_legend, single_sensor)
    if not independent_grid:
        _draw_db_error(ax_err, dL, freq_axis, selection, plot_cfg, stats,
                       single_sensor)

    ax_bottom.set_xlabel(plot_cfg['xlabel'] or xlabel)
    if plot_cfg['xlim'] is not None:
        ax_bottom.set_xlim(plot_cfg['xlim'])

    fig.tight_layout()
    return fig, stats, nc1 + nc2


def _save_db_csv_long(
    fig_path: Path,
    sol_freq: np.ndarray,
    val_freq: np.ndarray,
    selection: List[Tuple[int, str]],
    sol_sel: np.ndarray,
    val_sel: np.ndarray,
    plot_cfg: Dict[str, Any],
) -> None:
    """
    Write the dB levels in long format, one row per (series, location, freq).

    Used on an independent grid, where the two series sample different
    frequencies so a shared frequency column is not well defined. No error
    summary accompanies it: there is no level error to summarise.
    """
    db_ref = plot_cfg['db_ref']
    db_floor = plot_cfg['db_floor']
    l_comp, _ = to_db(sol_sel, db_ref, db_floor)
    l_meas, _ = to_db(val_sel, db_ref, db_floor)

    csv_path = fig_path.with_suffix('.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(DB_CSV_HEADER)
        for name, levels, axis in (('computed', l_comp, sol_freq),
                                   ('measured', l_meas, val_freq)):
            for k, (idx, label) in enumerate(selection):
                for j, freq in enumerate(axis):
                    writer.writerow(
                        [name, float(freq), idx, label, float(levels[k, j])]
                    )
    print(f"Saved CSV to {csv_path}")


def _save_db_csv(
    fig_path: Path,
    freq_axis: np.ndarray,
    xlabel: str,
    selection: List[Tuple[int, str]],
    sol_sel: np.ndarray,
    val_sel: np.ndarray,
    plot_cfg: Dict[str, Any],
) -> None:
    """Write per-frequency dB columns and a per-sensor + pooled error summary."""
    db_ref = plot_cfg['db_ref']
    db_floor = plot_cfg['db_floor']
    l_comp, _ = to_db(sol_sel, db_ref, db_floor)
    l_meas, _ = to_db(val_sel, db_ref, db_floor)
    dL, _ = db_error(val_sel, sol_sel, db_floor)

    csv_path = fig_path.with_suffix('.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [xlabel]
        for idx, _ in selection:
            header += [f'Lcomp_idx_{idx}', f'Lmeas_idx_{idx}', f'dL_idx_{idx}']
        writer.writerow(header)
        for j, freq in enumerate(freq_axis):
            row = [float(freq)]
            for k in range(len(selection)):
                row += [float(l_comp[k, j]), float(l_meas[k, j]),
                        float(dL[k, j])]
            writer.writerow(row)
    print(f"Saved CSV to {csv_path}")

    stats_path = fig_path.with_name(f"{fig_path.stem}_error_stats.csv")
    abs_dL = np.abs(dL)
    per_max = np.max(abs_dL, axis=1)
    per_med = np.median(abs_dL, axis=1)
    with open(stats_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['face_index', 'max_abs_dB', 'median_abs_dB'])
        for k, (idx, _) in enumerate(selection):
            writer.writerow([int(idx), float(per_max[k]), float(per_med[k])])
        writer.writerow(['POOLED', float(np.max(abs_dL)),
                         float(np.median(abs_dL))])
    print(f"Saved error summary to {stats_path}")


def plot_validation_db(
    sol_sel: np.ndarray,
    val_sel: np.ndarray,
    freq_axis: np.ndarray,
    selection: List[Tuple[int, str]],
    xlabel: str,
    config: Dict[str, Any],
    fig_path: Path,
    val_freq_axis: Optional[np.ndarray] = None,
) -> None:
    """
    Combined + per-sensor dB figures (measured vs computed).

    ``val_freq_axis`` is the validation data's own frequency vector; ``None``
    means it shares the solution's grid. On an independent grid the level
    error is undefined, so each figure is the dB overlay panel alone.
    """
    plot_cfg = config['plot']
    out_cfg = config['output']
    db_floor = plot_cfg['db_floor']
    independent_grid = val_freq_axis is not None

    # --- combined "all sensors" figure ------------------------------------
    fig, stats, n_clamped = _render_db_figure(
        sol_sel, val_sel, freq_axis, selection, xlabel, plot_cfg,
        title=plot_cfg['title'],
        show_sensor_legend=(len(selection) <= COMBINED_LEGEND_MAX),
        val_freq_axis=val_freq_axis,
    )
    if n_clamped:
        print(f"  db: clamped {n_clamped} non-positive/near-zero S_ii "
              f"sample(s) to the relative floor (plot.db_floor={db_floor:g})")
    if stats is None:
        print("  db: independent frequency grids -> overlay panel only; the "
              "level error dL is undefined without paired frequencies")
    else:
        print(f"  combined dB error (pooled): max|dL|={stats[0]:.3f} dB, "
              f"median|dL|={stats[1]:.3f} dB")
    _save_fig(fig, fig_path, out_cfg['dpi'])

    if out_cfg['save_selection_csv']:
        if independent_grid:
            _save_db_csv_long(fig_path, freq_axis, val_freq_axis, selection,
                              sol_sel, val_sel, plot_cfg)
        else:
            _save_db_csv(fig_path, freq_axis, xlabel, selection, sol_sel,
                         val_sel, plot_cfg)

    # --- per-sensor figures -----------------------------------------------
    if not out_cfg['per_sensor']:
        return
    if independent_grid:
        # No level error exists, so there is nothing to rank by: fall back to
        # selection order and say so, lest a capped set read as "the worst N".
        order = np.arange(len(selection))
        n_skipped = 0
        top_n = out_cfg['top_n']
        if top_n is not None and top_n < len(order):
            order, n_skipped = order[:top_n], len(order) - top_n
        if n_skipped:
            print(f"  per-sensor: writing the FIRST {len(order)} of "
                  f"{len(order) + n_skipped} figures in selection order "
                  f"(output.top_n); these are not ranked worst-first, because "
                  f"no level error is defined on independent grids")
    else:
        dL_all, _ = db_error(val_sel, sol_sel, db_floor)
        order, n_skipped = select_per_sensor_order(dL_all, out_cfg['top_n'])
        if n_skipped:
            print(f"  per-sensor: writing worst {len(order)} of "
                  f"{len(order) + n_skipped} figures (output.top_n); "
                  f"skipped {n_skipped}")
    subdir = fig_path.parent / 'per_sensor'
    for k in order:
        idx = selection[k][0]
        sub_path = subdir / f"sensor_{idx}{fig_path.suffix}"
        fig_k, _stats, _nc = _render_db_figure(
            sol_sel[k:k + 1], val_sel[k:k + 1], freq_axis, [selection[k]],
            xlabel, plot_cfg,
            title=f"Sensor {idx}: measured vs computed",
            show_sensor_legend=False, single_sensor=True,
            val_freq_axis=val_freq_axis,
        )
        _save_fig(fig_k, sub_path, out_cfg['dpi'])
    if len(order):
        print(f"  wrote {len(order)} per-sensor figure(s) to {subdir}")


# --------------------------------------------------------------------------
# envelope: min / energetic-mean / max across sensors, in dB
# --------------------------------------------------------------------------

class EnvelopeStats(NamedTuple):
    """Per-frequency spread across sensors, in dB. Each field is (n_freq,)."""
    lo: np.ndarray        # quietest sensor
    mean: np.ndarray      # energetic (power) mean over sensors
    hi: np.ndarray        # loudest sensor
    n_clamped: int


def envelope_stats(S: np.ndarray, db_ref: float,
                   db_floor: float) -> EnvelopeStats:
    """
    Spread of ``S (n_loc, n_freq)`` across locations, at each frequency, in dB.

    ``lo``/``hi`` are the quietest and loudest sensor. Because ``10*log10`` is
    monotone, taking them before or after the conversion is equivalent.

    ``mean`` is the **energetic** mean -- the powers are averaged and *then*
    converted, ``10*log10(mean_i(S_i)/ref)``. That is the spatial average of
    the field's energy, and the convention in acoustic qualification. Averaging
    the dB values instead would give the geometric mean of the powers, which is
    always lower and carries no energy interpretation.
    """
    levels, n_clamped = to_db(S, db_ref, db_floor)
    mean_power = np.mean(np.asarray(S, dtype=float), axis=0, keepdims=True)
    mean_level, n_clamped_mean = to_db(mean_power, db_ref, db_floor)
    return EnvelopeStats(
        lo=levels.min(axis=0),
        mean=mean_level[0],
        hi=levels.max(axis=0),
        n_clamped=n_clamped + n_clamped_mean,
    )


def plot_envelope(
    sol_sel: np.ndarray,
    val_sel: Optional[np.ndarray],
    freq_axis: np.ndarray,
    xlabel: str,
    config: Dict[str, Any],
    fig_path: Path,
    val_freq_axis: Optional[np.ndarray] = None,
) -> None:
    """
    Min-max envelope across all selected sensors, computed vs measured, in dB.

    The statistic is taken *within* each series, so the two may be sampled on
    different frequency grids; ``val_freq_axis`` gives the validation data its
    own. Without a validation set only the computed band is drawn.
    """
    plot_cfg = config['plot']
    out_cfg = config['output']
    db_ref, db_floor = plot_cfg['db_ref'], plot_cfg['db_floor']
    val_freq = freq_axis if val_freq_axis is None else val_freq_axis

    n_sensors = sol_sel.shape[0]
    if n_sensors < 2:
        print(f"  envelope: only {n_sensors} sensor selected, so the band "
              f"collapses onto its centre line")

    series = [('computed', sol_sel, freq_axis, 'tab:blue', '-')]
    if val_sel is not None:
        series.append(('measured', val_sel, val_freq, 'tab:orange', '--'))

    fig, ax = plt.subplots(figsize=tuple(plot_cfg['figsize']))
    computed: List[Tuple[str, np.ndarray, EnvelopeStats]] = []
    total_clamped = 0
    for name, data, axis, color, style in series:
        st = envelope_stats(data, db_ref, db_floor)
        total_clamped += st.n_clamped
        computed.append((name, axis, st))
        ax.fill_between(axis, st.lo, st.hi, color=color, alpha=0.25,
                        label=f'{name} min-max')
        ax.plot(axis, st.lo, style, color=color, lw=0.8, alpha=0.8)
        ax.plot(axis, st.hi, style, color=color, lw=0.8, alpha=0.8)
        ax.plot(axis, st.mean, style, color=color, lw=2.0,
                label=f'{name} energetic mean')

    if total_clamped:
        print(f"  envelope: clamped {total_clamped} non-positive/near-zero "
              f"sample(s) to the relative floor (plot.db_floor={db_floor:g})")

    ax.set_xlabel(plot_cfg['xlabel'] or xlabel)
    ax.set_ylabel(f"{plot_cfg['ylabel']} level [dB re {db_ref:g}]")
    ax.set_title(plot_cfg['title'])
    if plot_cfg['ylim'] is not None:
        ax.set_ylim(plot_cfg['ylim'])
    if plot_cfg['xlim'] is not None:
        ax.set_xlim(plot_cfg['xlim'])
    ax.grid(True, alpha=0.3)
    _legend_outside(ax)
    fig.tight_layout()
    _save_fig(fig, fig_path, out_cfg['dpi'])
    print(f"  envelope over {n_sensors} sensor(s)")

    if out_cfg['save_selection_csv']:
        csv_path = fig_path.with_suffix('.csv')
        # Long format: the two series may sample different frequencies, so a
        # shared frequency column would not be well defined.
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(ENVELOPE_CSV_HEADER)
            for name, axis, st in computed:
                for j, freq in enumerate(axis):
                    for stat, values in (('min', st.lo), ('mean', st.mean),
                                         ('max', st.hi)):
                        writer.writerow(
                            [name, float(freq), stat, float(values[j])]
                        )
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
    val_freq_axis = None
    independent_grid = inp['validation_frequencies'] is not None
    if independent_grid and xlabel == FREQ_INDEX_LABEL:
        raise ValueError(
            "input.validation_frequencies places the validation data on a "
            "physical frequency axis, but the sidecar carries no 'frequencies' "
            "so the solution is plotted against its frequency index. Add "
            "'frequencies' to the sidecar, or drop "
            "input.validation_frequencies."
        )

    if has_validation:
        if independent_grid:
            val_freq_axis = load_frequency_spec(
                inp['validation_frequencies'], 'input.validation_frequencies'
            )
        val_diag = load_validation_diagonal(
            inp['validation_path'], inp['validation_var'],
            None if independent_grid else freq_indices,
        )
        if val_diag.shape[0] != len(selection):
            raise ValueError(
                f"Validation has {val_diag.shape[0]} locations but "
                f"{len(selection)} coordinates were selected; row k of the "
                "validation data must align with coordinate k."
            )
        if independent_grid:
            if val_diag.shape[1] != val_freq_axis.size:
                raise ValueError(
                    f"Validation holds {val_diag.shape[1]} frequencies but "
                    f"input.validation_frequencies gives "
                    f"{val_freq_axis.size}; the two must agree."
                )
        elif val_diag.shape[1] != sol_sel.shape[1]:
            raise ValueError(
                f"Validation frequency count {val_diag.shape[1]} does not "
                f"match solution {sol_sel.shape[1]} after slicing."
            )
        val_sel = val_diag
        print(f"Loaded validation diagonal: shape={val_diag.shape}")
        if independent_grid:
            print(
                f"  validation on its own grid: "
                f"[{val_freq_axis.min():.4g}, {val_freq_axis.max():.4g}]"
            )

    n_kinds = len(kinds)
    for kind in kinds:
        fig_path = _kind_path(config['output'], kind, n_kinds)
        if kind == 'lines':
            plot_lines(sol_sel, val_sel, freq_axis, selection, xlabel,
                       config, fig_path, val_freq_axis=val_freq_axis)
        elif kind == 'box':
            plot_box(sol_sel, val_sel, freq_axis, xlabel, config, fig_path)
        elif kind == 'error':
            plot_error(sol_sel, val_sel, selection, config, fig_path)
        elif kind == 'validation_db':
            plot_validation_db(sol_sel, val_sel, freq_axis, selection, xlabel,
                               config, fig_path, val_freq_axis=val_freq_axis)
        elif kind == 'envelope':
            plot_envelope(sol_sel, val_sel, freq_axis, xlabel, config,
                          fig_path, val_freq_axis=val_freq_axis)


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
