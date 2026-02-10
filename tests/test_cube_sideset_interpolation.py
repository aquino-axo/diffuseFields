"""
Integration tests for sideset interpolation using a unit cube mesh.

Tests verify the end-to-end workflow: extract sideset centroids,
interpolate a constant pressure field, write sideset variables,
and confirm correct values (including zeros on unwritten sidesets).

Test data (in tests/data/):
- cube.e: unit cube mesh, 64 HEX8 elements, 6 sidesets (16 faces each)
  Sideset 6 is the Y=0.5 face.
- cube_coords.npy: 25 source points on the Y=0.5 plane, shape (25, 3)
- cube_p.npy: constant pressure = 3.1416 at each source point, shape (25,)
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from exodus_side_interpolator import ExodusSideInterpolator
from pressure_interpolator import PressureFieldInterpolator

DATA_DIR = Path(__file__).parent / "data"
CUBE_FILE = DATA_DIR / "cube.e"
CUBE_COORDS = DATA_DIR / "cube_coords.npy"
CUBE_PRESSURE = DATA_DIR / "cube_p.npy"

SIDESET_ID = 6
EXPECTED_PRESSURE = 3.1416


def _skip_if_missing():
    """Return True (and print skip message) if test data or exodusii unavailable."""
    if not CUBE_FILE.exists():
        print("  SKIPPED (cube.e not found)")
        return True
    try:
        import exodusii  # noqa: F401
    except ImportError:
        print("  SKIPPED (exodusii not available)")
        return True
    return False


def test_sideset_centroids_on_correct_face():
    """
    Test 1: Sideset 6 centroids lie on the Y=0.5 plane.

    The cube spans [-0.5, 0.5]^3. Sideset 6 is the Y+ face,
    so all centroid Y-coordinates must equal 0.5.
    """
    print("Test 1: Sideset 6 centroids on Y=0.5 plane...")
    if _skip_if_missing():
        return

    with ExodusSideInterpolator(str(CUBE_FILE)) as db:
        centroids = db.get_sideset_face_centroids(SIDESET_ID)

    assert centroids.shape == (16, 3), (
        f"Expected (16, 3), got {centroids.shape}"
    )
    np.testing.assert_allclose(
        centroids[:, 1], 0.5, atol=1e-12,
        err_msg="Sideset 6 centroids should all have Y=0.5"
    )

    print(f"  PASSED ({centroids.shape[0]} faces, all Y=0.5)")


def test_constant_pressure_interpolation():
    """
    Test 2: Interpolating a constant field reproduces the constant.

    Source: 25 points on Y=0.5 plane with constant pressure 3.1416.
    Target: 16 sideset 6 face centroids (also on Y=0.5 plane).
    Expected: all interpolated values equal 3.1416.
    """
    print("Test 2: Constant pressure interpolation...")
    if _skip_if_missing():
        return

    source_coords = np.load(str(CUBE_COORDS))
    source_pressure = np.load(str(CUBE_PRESSURE))

    with ExodusSideInterpolator(str(CUBE_FILE)) as db:
        centroids = db.get_sideset_face_centroids(SIDESET_ID)

    interpolator = PressureFieldInterpolator(
        source_coords, centroids, kernel='linear', smoothing=0.0
    )
    interpolated = interpolator.interpolate(source_pressure)

    np.testing.assert_allclose(
        np.real(interpolated), EXPECTED_PRESSURE, atol=1e-6,
        err_msg="Interpolated real part should equal source constant"
    )
    np.testing.assert_allclose(
        np.imag(interpolated), 0.0, atol=1e-12,
        err_msg="Interpolated imaginary part should be zero"
    )

    print(f"  PASSED (all 16 values = {EXPECTED_PRESSURE})")


def test_written_sideset_variables():
    """
    Test 3: Write interpolated pressure to exodus and read back.

    Writes pressure_real and pressure_imag to sideset 6 of a
    temporary copy of cube.e, then reads back via netCDF4 to verify.
    """
    print("Test 3: Written sideset variables correct...")
    if _skip_if_missing():
        return

    source_coords = np.load(str(CUBE_COORDS))
    source_pressure = np.load(str(CUBE_PRESSURE))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_cube = Path(tmpdir) / "cube.e"
        shutil.copy2(CUBE_FILE, tmp_cube)

        with ExodusSideInterpolator(str(tmp_cube), mode='a') as db:
            centroids = db.get_sideset_face_centroids(SIDESET_ID)

            interpolator = PressureFieldInterpolator(
                source_coords, centroids, kernel='linear', smoothing=0.0
            )
            interpolated = interpolator.interpolate(source_pressure)

            db.prepare_sideset_variables(
                ['pressure_real', 'pressure_imag']
            )
            db.write_sideset_variable(
                SIDESET_ID, 'pressure_real',
                np.real(interpolated), step=1
            )
            db.write_sideset_variable(
                SIDESET_ID, 'pressure_imag',
                np.imag(interpolated), step=1
            )

        # Read back via netCDF4
        import netCDF4 as nc
        with nc.Dataset(str(tmp_cube), 'r') as f:
            # Sideset 6 has internal id 6
            real_vals = f.variables['vals_sset_var1ss6'][0, :].data
            imag_vals = f.variables['vals_sset_var2ss6'][0, :].data

        np.testing.assert_allclose(
            real_vals, EXPECTED_PRESSURE, atol=1e-6,
            err_msg="pressure_real should equal source constant"
        )
        np.testing.assert_allclose(
            imag_vals, 0.0, atol=1e-12,
            err_msg="pressure_imag should be zero"
        )

    print("  PASSED")


def test_unwritten_sidesets_are_zero():
    """
    Test 4: Sidesets not written to contain zeros, not fill values.

    After writing only to sideset 6, all other sidesets (1-5) should
    have raw values of 0.0, not the netCDF default fill (~9.97e+36).
    """
    print("Test 4: Unwritten sidesets are zero...")
    if _skip_if_missing():
        return

    source_coords = np.load(str(CUBE_COORDS))
    source_pressure = np.load(str(CUBE_PRESSURE))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_cube = Path(tmpdir) / "cube.e"
        shutil.copy2(CUBE_FILE, tmp_cube)

        with ExodusSideInterpolator(str(tmp_cube), mode='a') as db:
            centroids = db.get_sideset_face_centroids(SIDESET_ID)

            interpolator = PressureFieldInterpolator(
                source_coords, centroids, kernel='linear', smoothing=0.0
            )
            interpolated = interpolator.interpolate(source_pressure)

            db.prepare_sideset_variables(
                ['pressure_real', 'pressure_imag']
            )
            db.write_sideset_variable(
                SIDESET_ID, 'pressure_real',
                np.real(interpolated), step=1
            )
            db.write_sideset_variable(
                SIDESET_ID, 'pressure_imag',
                np.imag(interpolated), step=1
            )

        # Read back and check non-target sidesets
        import netCDF4 as nc
        with nc.Dataset(str(tmp_cube), 'r') as f:
            for ss_iid in range(1, 7):
                if ss_iid == 6:
                    continue  # skip the target sideset
                for vi in [1, 2]:
                    key = f'vals_sset_var{vi}ss{ss_iid}'
                    raw = f.variables[key][0, :].data
                    assert np.all(raw == 0.0), (
                        f"{key} should be all zeros, "
                        f"got min={raw.min()}, max={raw.max()}"
                    )

    print("  PASSED (sidesets 1-5 all zero)")


def run_all_tests():
    """Run all cube sideset interpolation tests."""
    print("=" * 60)
    print("Cube Sideset Interpolation Tests")
    print("=" * 60)

    test_sideset_centroids_on_correct_face()
    test_constant_pressure_interpolation()
    test_written_sideset_variables()
    test_unwritten_sidesets_are_zero()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
