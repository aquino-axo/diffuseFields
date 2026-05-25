"""
Driver for reconstructing the full-space CPSD S* = Phi @ S_r @ Phi^h.

Reads results saved by run_cpsd_inverse.py together with the POD basis and
materializes either the full N x N CPSD (default) or only its diagonal for a
selected subset of frequencies. Kept separate from the inverse-solve driver
because the full-space CPSD can be very large.

Usage:
    python run_reconstruct_full_cpsd.py config_reconstruct_full_cpsd.json
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from cpsd_inverse import lift_to_full_space


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return json.load(f)


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    defaults: Dict[str, Dict[str, Any]] = {
        'input': {
            'inverse_results_dir': None,
            'pod_basis_path': None,
        },
        'reconstruction': {
            'freq_indices': None,    # default: all
            'alpha_index': 0,        # which alpha from a sweep (ignored if scalar)
            'mode': 'full',          # 'full' or 'diagonal'
            'dtype': 'complex128',   # 'complex64' or 'complex128'
        },
        'output': {
            'output_path': 'results_cpsd_inverse/full_cpsd.npy',
        },
    }
    for section, section_defaults in defaults.items():
        if section not in config:
            config[section] = dict(section_defaults)
        else:
            for key, value in section_defaults.items():
                config[section].setdefault(key, value)

    inp = config['input']
    for key in ('inverse_results_dir', 'pod_basis_path'):
        if inp[key] is None:
            raise ValueError(f"input.{key} is required")
        if not os.path.exists(inp[key]):
            raise FileNotFoundError(f"{key}: {inp[key]} not found")

    rec = config['reconstruction']
    if rec['mode'] not in ('full', 'diagonal'):
        raise ValueError(
            f"reconstruction.mode must be 'full' or 'diagonal', "
            f"got {rec['mode']}"
        )
    if rec['dtype'] not in ('complex64', 'complex128'):
        raise ValueError(
            f"reconstruction.dtype must be 'complex64' or 'complex128', "
            f"got {rec['dtype']}"
        )
    if not isinstance(rec['alpha_index'], int) or rec['alpha_index'] < 0:
        raise ValueError(
            f"reconstruction.alpha_index must be a non-negative int, "
            f"got {rec['alpha_index']}"
        )
    if rec['freq_indices'] is not None:
        if not isinstance(rec['freq_indices'], list):
            raise ValueError(
                "reconstruction.freq_indices must be null or a list of ints"
            )
        for v in rec['freq_indices']:
            if not isinstance(v, int) or v < 0:
                raise ValueError(
                    f"reconstruction.freq_indices entries must be "
                    f"non-negative ints, got {v}"
                )

    return config


_FILE_PATTERN = re.compile(r'^cpsd_inverse_freq(\d+)\.npz$')


def _discover_freq_files(inv_dir: Path) -> List[int]:
    indices: List[int] = []
    for p in inv_dir.iterdir():
        match = _FILE_PATTERN.match(p.name)
        if match is not None:
            indices.append(int(match.group(1)))
    return sorted(indices)


def reconstruct(config: Dict[str, Any]) -> None:
    inp = config['input']
    rec = config['reconstruction']
    out_cfg = config['output']

    inv_dir = Path(inp['inverse_results_dir'])
    phi = np.load(inp['pod_basis_path'])
    if phi.ndim != 2:
        raise ValueError(
            f"POD basis must be 2D (N, n_pod), got shape {phi.shape}"
        )

    available = _discover_freq_files(inv_dir)
    if not available:
        raise FileNotFoundError(
            f"No cpsd_inverse_freq*.npz files in {inv_dir}"
        )

    requested = rec['freq_indices']
    if requested is None:
        requested = available
    else:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(
                f"requested freq_indices not found in {inv_dir}: {missing}"
            )

    n_sel = len(requested)
    N, n_pod = phi.shape
    dtype = np.dtype(rec['dtype'])
    real_dtype = np.float32 if dtype == np.complex64 else np.float64

    print(
        f"Reconstructing {n_sel} frequency snapshots "
        f"(mode={rec['mode']}, dtype={dtype})"
    )

    if rec['mode'] == 'full':
        out = np.empty((N, N, n_sel), dtype=dtype)
    else:
        out = np.empty((N, n_sel), dtype=real_dtype)

    saved_frequencies = []
    for k, f_idx in enumerate(requested):
        data = np.load(inv_dir / f'cpsd_inverse_freq{f_idx}.npz')

        S_r = data['S_r']
        if S_r.ndim == 3:
            a_idx = rec['alpha_index']
            if not 0 <= a_idx < S_r.shape[2]:
                raise IndexError(
                    f"reconstruction.alpha_index {a_idx} out of range for "
                    f"sweep length {S_r.shape[2]} at freq idx {f_idx}"
                )
            S_r = S_r[:, :, a_idx]

        if S_r.shape != (n_pod, n_pod):
            raise ValueError(
                f"S_r at freq idx {f_idx} has shape {S_r.shape}; expected "
                f"({n_pod}, {n_pod}) to match POD basis"
            )

        if rec['mode'] == 'full':
            out[:, :, k] = lift_to_full_space(S_r, phi).astype(dtype)
        else:
            out[:, k] = lift_to_full_space(
                S_r, phi, diagonal_only=True
            ).astype(real_dtype)

        if 'frequency' in data.files:
            saved_frequencies.append(float(data['frequency']))
        print(f"  freq idx {f_idx}: done")

    out_path = Path(out_cfg['output_path'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out)
    print(
        f"Saved reconstruction to {out_path} "
        f"(shape={out.shape}, dtype={out.dtype})"
    )

    sidecar = {
        'freq_indices': requested,
        'mode': rec['mode'],
        'dtype': rec['dtype'],
        'alpha_index': rec['alpha_index'],
        'pod_basis_path': inp['pod_basis_path'],
        'inverse_results_dir': inp['inverse_results_dir'],
    }
    if saved_frequencies and len(saved_frequencies) == len(requested):
        sidecar['frequencies'] = saved_frequencies
    sidecar_path = out_path.with_suffix('.json')
    with open(sidecar_path, 'w') as f:
        json.dump(sidecar, f, indent=2)
    print(f"Saved sidecar to {sidecar_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Reconstruct full-space CPSD S* = Phi S_r Phi^h'
    )
    parser.add_argument(
        'config_file',
        nargs='?',
        default='config_reconstruct_full_cpsd.json',
        help='Path to configuration JSON file '
             '(default: config_reconstruct_full_cpsd.json)'
    )
    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        print(f"Error: Configuration file '{args.config_file}' not found.")
        return 1

    print(f"Loading configuration from: {args.config_file}")
    config = validate_config(load_config(args.config_file))

    reconstruct(config)
    return 0


if __name__ == '__main__':
    exit(main())
