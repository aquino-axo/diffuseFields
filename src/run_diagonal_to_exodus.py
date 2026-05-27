"""
Driver script for writing the CPSD diagonal to an ExodusII sideset as a
time-varying sideset variable.

Consumes the diagonal `.npy` file (shape `(N, n_freq)`, real) produced by
`run_reconstruct_full_cpsd.py` and writes one sideset variable with one
time step per frequency. The sideset index dimension (`N`) is assumed to
match the POD basis exported from the same sideset by
`run_sideset_pod_export.py`; the row ordering is preserved end-to-end, so
no interpolation is performed.

Writes either in-place to `input.exodus_file` or to a separate
`output.exodus_file`. Writing to a separate file is required when the
in-place target already has its ExodusII num_sset_var dimension fully
populated (netCDF-3 dimensions are fixed at creation): in that case,
provide `output.copy_from_exodus_file` pointing at a clean mesh
(typically the original, sideset-variable-free mesh) that will be copied
to `output.exodus_file` before writing.

Usage:
    python run_diagonal_to_exodus.py config_diagonal_to_exodus.json
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict

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
        'output': {
            'variable_name': 'cpsd_diag',
            'use_frequency_as_time': True,
            'start_step': 1,
            'exodus_file': None,
            'copy_from_exodus_file': None,
            'overwrite': False,
        },
    }
    for section, section_defaults in defaults.items():
        if section not in config:
            config[section] = dict(section_defaults)
        else:
            for key, value in section_defaults.items():
                config[section].setdefault(key, value)

    inp = config['input']
    for key in ('diagonal_npy_path', 'exodus_file', 'sideset_id'):
        if inp[key] is None:
            raise ValueError(f"input.{key} is required")
    if not os.path.exists(inp['diagonal_npy_path']):
        raise FileNotFoundError(
            f"input.diagonal_npy_path not found: {inp['diagonal_npy_path']}"
        )
    if not os.path.exists(inp['exodus_file']):
        raise FileNotFoundError(
            f"input.exodus_file not found: {inp['exodus_file']}"
        )
    if not isinstance(inp['sideset_id'], int):
        raise ValueError("input.sideset_id must be an integer")

    if inp['sidecar_json_path'] is None:
        guess = Path(inp['diagonal_npy_path']).with_suffix('.json')
        if guess.exists():
            inp['sidecar_json_path'] = str(guess)
    if inp['sidecar_json_path'] is not None and not os.path.exists(
        inp['sidecar_json_path']
    ):
        raise FileNotFoundError(
            f"input.sidecar_json_path not found: {inp['sidecar_json_path']}"
        )

    out = config['output']
    if not isinstance(out['variable_name'], str) or not out['variable_name']:
        raise ValueError("output.variable_name must be a non-empty string")
    if not isinstance(out['start_step'], int) or out['start_step'] < 1:
        raise ValueError("output.start_step must be an int >= 1")
    if out['copy_from_exodus_file'] is not None:
        if out['exodus_file'] is None:
            raise ValueError(
                "output.copy_from_exodus_file requires output.exodus_file"
            )
        if not os.path.exists(out['copy_from_exodus_file']):
            raise FileNotFoundError(
                f"output.copy_from_exodus_file not found: "
                f"{out['copy_from_exodus_file']}"
            )

    return config


def _resolve_target_exodus(config: Dict[str, Any]) -> str:
    """
    Decide which exodus file we will open and (if needed) copy the seed
    file into place. Returns the absolute target file path.
    """
    inp = config['input']
    out = config['output']

    if out['exodus_file'] is None:
        return inp['exodus_file']

    target = out['exodus_file']
    if os.path.exists(target):
        if not out['overwrite']:
            return target
        # Overwrite: re-copy from the configured source.
        os.remove(target)

    copy_src = out['copy_from_exodus_file'] or inp['exodus_file']
    target_dir = os.path.dirname(os.path.abspath(target))
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    shutil.copyfile(copy_src, target)
    print(f"Copied seed exodus: {copy_src} -> {target}")
    return target


def load_diagonal(
    diagonal_path: str, sidecar_path: str | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Load the diagonal array and optional physical frequencies.

    Returns
    -------
    diag : ndarray, shape (N, n_freq), real (float64)
    frequencies : ndarray of float or None
        Physical frequencies if available in the sidecar; otherwise None.
    """
    diag = np.load(diagonal_path)
    if diag.ndim != 2:
        raise ValueError(
            f"Diagonal array must be 2D (N, n_freq); got shape {diag.shape}"
        )
    diag = np.asarray(diag, dtype=np.float64)

    frequencies = None
    if sidecar_path is not None:
        with open(sidecar_path, 'r') as f:
            sidecar = json.load(f)
        if sidecar.get('mode', 'diagonal') != 'diagonal':
            raise ValueError(
                f"Sidecar reports mode={sidecar.get('mode')!r}; expected "
                f"'diagonal'."
            )
        freqs = sidecar.get('frequencies')
        if freqs is not None and len(freqs) == diag.shape[1]:
            frequencies = np.asarray(freqs, dtype=float)
    return diag, frequencies


def write_diagonal_to_exodus(config: Dict[str, Any]) -> None:
    inp = config['input']
    out = config['output']

    diag, frequencies = load_diagonal(
        inp['diagonal_npy_path'], inp['sidecar_json_path']
    )
    n_entries, n_freq = diag.shape
    print(
        f"Loaded diagonal: {n_entries} entries x {n_freq} frequencies "
        f"(physical frequencies {'present' if frequencies is not None else 'unavailable'})"
    )

    exodus_file = _resolve_target_exodus(config)
    sideset_id = inp['sideset_id']
    var_name = out['variable_name']
    start_step = out['start_step']
    use_freq_time = out['use_frequency_as_time'] and frequencies is not None

    with ExodusSideInterpolator(exodus_file, mode='a') as db:
        params = db._exo.get_side_set_params(sideset_id)
        n_faces = params.num_sides
        if n_faces != n_entries:
            raise ValueError(
                f"Sideset {sideset_id} has {n_faces} faces, but diagonal "
                f"array has {n_entries} rows. The POD basis used for "
                f"reconstruction must come from this same sideset."
            )

        db.prepare_sideset_variables([var_name])

        for k in range(n_freq):
            step = start_step + k
            if use_freq_time:
                db._exo.put_time(step, float(frequencies[k]))
            else:
                db._exo.put_time(step, float(step))

            db.write_sideset_variable(
                sideset_id, var_name, diag[:, k], step=step
            )

        if use_freq_time:
            print(
                f"Wrote '{var_name}' at steps {start_step}..{start_step + n_freq - 1} "
                f"with time = frequency [Hz]."
            )
        else:
            print(
                f"Wrote '{var_name}' at steps {start_step}..{start_step + n_freq - 1} "
                f"with time = step index."
            )
    print(f"Updated exodus file: {exodus_file}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Write the CPSD diagonal to an ExodusII sideset as a '
            'time-varying sideset variable (one step per frequency).'
        )
    )
    parser.add_argument(
        'config_file',
        nargs='?',
        default='config_diagonal_to_exodus.json',
        help='Path to configuration JSON file '
             '(default: config_diagonal_to_exodus.json)',
    )
    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        print(f"Error: Configuration file '{args.config_file}' not found.")
        return 1

    print(f"Loading configuration from: {args.config_file}")
    config = validate_config(load_config(args.config_file))
    write_diagonal_to_exodus(config)
    return 0


if __name__ == '__main__':
    exit(main())
