"""
Shared parsing of frequency-vector specifications used across the drivers.

A *frequency spec* is one of

  * a list of positive numbers, ``[10, 15, 20]``;
  * a dict ``{"min": 10, "step": 5, "max": 30}``, expanded inclusively of
    ``max`` when ``max`` lands on a step and otherwise stopping below it;
  * a path to a 1-D array in ``.npy`` or ``.mat`` (``load_frequency_spec``
    only).

``field_name`` is the dotted config path of the field being parsed; it is
interpolated into every error message so the caller's user sees which of their
settings is at fault.
"""

import os
from typing import Any

import numpy as np
from scipy.io import loadmat


def parse_frequency_spec(spec: Any, field_name: str) -> np.ndarray:
    """Parse a list or a {min, step, max} dict into a 1-D float array."""
    if isinstance(spec, list):
        if len(spec) == 0:
            raise ValueError(f"{field_name} list cannot be empty")
        for f in spec:
            if not isinstance(f, (int, float)) or f <= 0:
                raise ValueError(
                    f"all {field_name} values must be positive numbers, got {f}"
                )
        return np.asarray(spec, dtype=float)

    if isinstance(spec, dict):
        for key in ('min', 'step', 'max'):
            if key not in spec:
                raise ValueError(f"{field_name} dict missing key '{key}'")
        f_min = spec['min']
        f_step = spec['step']
        f_max = spec['max']
        if f_min <= 0 or f_step <= 0 or f_max <= 0:
            raise ValueError(f"{field_name} values must be positive")
        if f_max < f_min:
            raise ValueError(f"{field_name}.max must be >= min")
        freqs = np.arange(f_min, f_max + f_step * 0.001, f_step)
        return freqs[freqs <= f_max + 1e-10].astype(float)

    raise ValueError(
        f"{field_name} must be a list or a dict with min/step/max, "
        f"got {type(spec).__name__}"
    )


def _load_1d_array(path: str, field_name: str) -> np.ndarray:
    """Load a 1-D float array from a .npy or single-variable .mat file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        arr = np.load(path)
    elif ext == '.mat':
        mat = loadmat(path)
        keys = sorted(k for k in mat.keys() if not k.startswith('__'))
        if len(keys) != 1:
            raise ValueError(
                f"{field_name} points at {path}, which holds {len(keys)} "
                f"variables ({keys}); the single-variable form is required. "
                f"Use a .npy file or an inline list instead."
            )
        arr = np.asarray(mat[keys[0]])
    else:
        raise ValueError(
            f"{field_name} path must be a .npy or .mat file, got {path}"
        )

    arr = np.atleast_1d(np.squeeze(np.asarray(arr)))
    if arr.ndim != 1:
        raise ValueError(
            f"{field_name} must be a 1-D array of frequencies; {path} holds "
            f"shape {np.asarray(arr).shape}"
        )
    return arr.astype(float)


def load_frequency_spec(spec: Any, field_name: str) -> np.ndarray:
    """Resolve a frequency spec, accepting a file path as well."""
    if isinstance(spec, str):
        if not os.path.exists(spec):
            raise FileNotFoundError(f"{field_name} not found: {spec}")
        return _load_1d_array(spec, field_name)
    return parse_frequency_spec(spec, field_name)
