"""
Driver script for exporting POD modes from a sideset to a .npy file.

Reads sideset variables of the form `{prefix}_ev{i}_real` and
`{prefix}_ev{i}_imag` written by `run_sideset_interpolation.py`, pairs
them into complex values, and stacks them into an array of shape
(n_faces, n_modes) saved as a .npy file.

Usage:
    python run_sideset_pod_export.py config_sideset_pod_export.json
"""

import argparse
import json
import os
import re
from typing import Dict, Any, List, Tuple

import numpy as np

from exodus_side_interpolator import ExodusSideInterpolator


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and fill in default values for configuration.

    Required fields:
    - input.exodus_file
    - input.sideset_id
    - output.npy_path

    Optional fields with defaults:
    - input.variable_prefix: "pressure"
    - input.time_step: 1
    """
    if 'input' not in config:
        raise ValueError("Configuration must have 'input' section")
    if 'output' not in config:
        raise ValueError("Configuration must have 'output' section")

    input_cfg = config['input']
    for key in ['exodus_file', 'sideset_id']:
        if key not in input_cfg:
            raise ValueError(f"input.{key} is required")

    if not os.path.exists(input_cfg['exodus_file']):
        raise ValueError(f"File not found: {input_cfg['exodus_file']}")

    if not isinstance(input_cfg['sideset_id'], int):
        raise ValueError("input.sideset_id must be an integer")

    input_cfg.setdefault('variable_prefix', 'pressure')
    input_cfg.setdefault('time_step', 1)

    output_cfg = config['output']
    if 'npy_path' not in output_cfg:
        raise ValueError("output.npy_path is required")

    return config


def find_mode_pairs(
    variable_names: List[str], prefix: str
) -> List[Tuple[int, str, str]]:
    """
    Find (real, imag) variable name pairs matching the POD mode pattern.

    Parameters
    ----------
    variable_names : list of str
        All sideset variable names in the file.
    prefix : str
        Variable prefix used during interpolation (e.g., 'pressure').

    Returns
    -------
    pairs : list of (index, real_name, imag_name)
        Sorted by mode index. Modes missing either component are skipped.
    """
    pattern = re.compile(
        rf"^{re.escape(prefix)}_ev(\d+)_(real|imag)$"
    )
    components: Dict[int, Dict[str, str]] = {}
    for name in variable_names:
        match = pattern.match(name)
        if match is None:
            continue
        idx = int(match.group(1))
        components.setdefault(idx, {})[match.group(2)] = name

    pairs: List[Tuple[int, str, str]] = []
    skipped: List[int] = []
    for idx in sorted(components):
        parts = components[idx]
        if 'real' in parts and 'imag' in parts:
            pairs.append((idx, parts['real'], parts['imag']))
        else:
            skipped.append(idx)

    if skipped:
        print(
            f"Warning: skipping mode indices with incomplete real/imag "
            f"pairs: {skipped}"
        )

    return pairs


def export_pod_modes(config: Dict[str, Any]) -> None:
    """Read POD modes from a sideset and save them to a .npy file."""
    input_cfg = config['input']
    output_cfg = config['output']

    exodus_file = input_cfg['exodus_file']
    sideset_id = input_cfg['sideset_id']
    prefix = input_cfg['variable_prefix']
    step = input_cfg['time_step']
    npy_path = output_cfg['npy_path']

    print(f"Opening ExodusII file: {exodus_file}")
    with ExodusSideInterpolator(exodus_file, mode='r') as db:
        names = db.get_sideset_variable_names()
        if not names:
            raise RuntimeError(
                f"No sideset variables found in '{exodus_file}'."
            )

        pairs = find_mode_pairs(names, prefix)
        if not pairs:
            raise RuntimeError(
                f"No POD mode variables matching '{prefix}_ev*_real/imag' "
                f"found. Available sideset variables: {names}"
            )

        n_modes = len(pairs)
        params = db._exo.get_side_set_params(sideset_id)
        n_faces = params.num_sides
        print(
            f"Sideset {sideset_id}: {n_faces} faces, "
            f"{n_modes} POD modes detected (prefix='{prefix}')"
        )

        modes = np.empty((n_faces, n_modes), dtype=np.complex128)
        for col, (idx, real_name, imag_name) in enumerate(pairs):
            real_vals = db.read_sideset_variable(
                sideset_id, real_name, step=step
            )
            imag_vals = db.read_sideset_variable(
                sideset_id, imag_name, step=step
            )
            modes[:, col] = real_vals + 1j * imag_vals
            print(f"  Read mode {idx}: {real_name} + i*{imag_name}")

    out_dir = os.path.dirname(os.path.abspath(npy_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    np.save(npy_path, modes)
    print(f"\nSaved POD modes: shape={modes.shape}, dtype={modes.dtype}")
    print(f"Output file: {npy_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Export POD modes from a sideset in an ExodusII file '
            'to a .npy file.'
        )
    )
    parser.add_argument(
        'config',
        nargs='?',
        default='config_sideset_pod_export.json',
        help=(
            'Path to configuration JSON file '
            '(default: config_sideset_pod_export.json)'
        ),
    )
    args = parser.parse_args()

    print(f"Loading configuration from: {args.config}")
    config = validate_config(load_config(args.config))

    print("\n" + "=" * 60)
    print("Sideset POD Mode Export")
    print("=" * 60)

    export_pod_modes(config)

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    exit(main())
