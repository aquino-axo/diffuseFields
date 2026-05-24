"""
Round-trip test for sideset POD mode export.

Writes known complex eigenvectors as sideset variables using the same
naming convention as run_sideset_interpolation.py, then exports them
via run_sideset_pod_export.export_pod_modes and verifies the recovered
complex array matches the original to machine precision.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

SRC = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(SRC))

from exodus_side_interpolator import ExodusSideInterpolator
from run_sideset_pod_export import export_pod_modes

DATA_DIR = Path(__file__).parent / "data"
CUBE_FILE = DATA_DIR / "cube.e"
SIDESET_ID = 6
N_FACES = 16  # sideset 6 of the unit cube has 16 quad faces


def _skip_if_missing():
    if not CUBE_FILE.exists():
        print("  SKIPPED (cube.e not found)")
        return True
    try:
        import exodusii  # noqa: F401
    except ImportError:
        print("  SKIPPED (exodusii not available)")
        return True
    return False


def test_pod_round_trip():
    """
    Test: write known POD modes to a sideset, export to .npy,
    and assert exact recovery of the complex array.

    Uses three deterministic modes with distinct real and imaginary
    parts so any mis-pairing (e.g., swapped indices) would be detected.
    """
    print("Test: POD mode round-trip (write -> export -> load)...")
    if _skip_if_missing():
        return

    rng = np.random.default_rng(42)
    n_modes = 3
    expected = (
        rng.standard_normal((N_FACES, n_modes))
        + 1j * rng.standard_normal((N_FACES, n_modes))
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_cube = Path(tmpdir) / "cube.e"
        shutil.copy2(CUBE_FILE, tmp_cube)

        var_names = []
        for i in range(n_modes):
            var_names.append(f"pressure_ev{i+1}_real")
            var_names.append(f"pressure_ev{i+1}_imag")

        with ExodusSideInterpolator(str(tmp_cube), mode='a') as db:
            db.prepare_sideset_variables(var_names)
            for i in range(n_modes):
                db.write_sideset_variable(
                    SIDESET_ID,
                    f"pressure_ev{i+1}_real",
                    np.real(expected[:, i]),
                    step=1,
                )
                db.write_sideset_variable(
                    SIDESET_ID,
                    f"pressure_ev{i+1}_imag",
                    np.imag(expected[:, i]),
                    step=1,
                )

        npy_path = Path(tmpdir) / "pod_modes.npy"
        config = {
            "input": {
                "exodus_file": str(tmp_cube),
                "sideset_id": SIDESET_ID,
                "variable_prefix": "pressure",
                "time_step": 1,
            },
            "output": {"npy_path": str(npy_path)},
        }
        export_pod_modes(config)

        recovered = np.load(str(npy_path))

    assert recovered.shape == expected.shape, (
        f"Expected shape {expected.shape}, got {recovered.shape}"
    )
    assert recovered.dtype == np.complex128, (
        f"Expected complex128, got {recovered.dtype}"
    )
    np.testing.assert_array_equal(
        recovered, expected,
        err_msg="Round-tripped POD modes differ from originals",
    )

    print(f"  PASSED ({n_modes} modes x {N_FACES} faces recovered exactly)")


def run_all_tests():
    print("=" * 60)
    print("Sideset POD Export Tests")
    print("=" * 60)
    test_pod_round_trip()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
