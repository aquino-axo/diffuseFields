"""
Unit tests for ExodusSideInterpolator.

Tests verify:
1. Triangle centroid and area computation
2. Quad area-weighted centroid computation (non-planar case)
3. Sideset extraction from an exodus file
4. Invalid sideset ID validation
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from exodus_side_interpolator import ExodusSideInterpolator


def test_triangle_centroid_and_area():
    """
    Test 1: Right triangle centroid and area.

    Triangle with vertices (0,0,0), (1,0,0), (0,1,0).
    Expected centroid: (1/3, 1/3, 0).
    Expected area: 0.5.
    """
    print("Test 1: Triangle centroid and area...")

    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    centroid, area = ExodusSideInterpolator._compute_face_centroid_and_area(
        coords
    )

    expected_centroid = np.array([1.0 / 3.0, 1.0 / 3.0, 0.0])
    np.testing.assert_allclose(centroid, expected_centroid, atol=1e-14)
    np.testing.assert_allclose(area, 0.5, atol=1e-14)

    print("  PASSED")


def test_quad_area_weighted_centroid():
    """
    Test 2: Non-planar quad area-weighted centroid.

    Quad with vertices:
      v0 = (0, 0, 0)
      v1 = (2, 0, 0)
      v2 = (2, 1, 0)
      v3 = (0, 1, 1)

    Fan triangulation from v0:
      Triangle 1: (v0, v1, v2) -> centroid = (4/3, 1/3, 0), area = 1.0
      Triangle 2: (v0, v2, v3) -> computed manually below.
    """
    print("Test 2: Quad area-weighted centroid (non-planar)...")

    v0 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([2.0, 0.0, 0.0])
    v2 = np.array([2.0, 1.0, 0.0])
    v3 = np.array([0.0, 1.0, 1.0])
    coords = np.array([v0, v1, v2, v3])

    # Triangle 1: v0, v1, v2
    c1 = (v0 + v1 + v2) / 3.0
    a1 = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

    # Triangle 2: v0, v2, v3
    c2 = (v0 + v2 + v3) / 3.0
    a2 = 0.5 * np.linalg.norm(np.cross(v2 - v0, v3 - v0))

    expected_centroid = (c1 * a1 + c2 * a2) / (a1 + a2)
    expected_area = a1 + a2

    centroid, area = ExodusSideInterpolator._compute_face_centroid_and_area(
        coords
    )

    np.testing.assert_allclose(centroid, expected_centroid, atol=1e-14)
    np.testing.assert_allclose(area, expected_area, atol=1e-14)

    # Verify this differs from simple geometric center (proving area-weighting matters)
    simple_centroid = np.mean(coords, axis=0)
    assert not np.allclose(centroid, simple_centroid, atol=1e-10), (
        "Area-weighted centroid should differ from simple average for "
        "non-planar quad"
    )

    print("  PASSED")


def test_sideset_extraction_from_exodus():
    """
    Test 3: Extract sideset face centroids from data/mug.e.

    Verifies shape, bounding box containment, and positive areas.
    """
    print("Test 3: Sideset extraction from mug.e...")

    data_dir = Path(__file__).parent.parent / "data"
    mug_file = data_dir / "mug.e"

    if not mug_file.exists():
        print("  SKIPPED (mug.e not found)")
        return

    try:
        import exodusii
    except ImportError:
        print("  SKIPPED (exodusii not available)")
        return

    with ExodusSideInterpolator(str(mug_file)) as db:
        sideset_ids = db.get_sideset_ids()
        assert len(sideset_ids) > 0, "Expected at least one sideset"

        sideset_id = sideset_ids[0]
        centroids = db.get_sideset_face_centroids(sideset_id)
        areas = db.get_sideset_face_areas(sideset_id)

        # Check shape
        params = db._exo.get_side_set_params(sideset_id)
        n_sides = params.num_sides
        assert centroids.shape == (n_sides, 3), (
            f"Expected shape ({n_sides}, 3), got {centroids.shape}"
        )
        assert areas.shape == (n_sides,), (
            f"Expected shape ({n_sides},), got {areas.shape}"
        )

        # Check centroids within mesh bounding box
        coords = db.get_coords()
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        assert np.all(centroids >= bbox_min - 1e-10), (
            "Centroids below bounding box"
        )
        assert np.all(centroids <= bbox_max + 1e-10), (
            "Centroids above bounding box"
        )

        # Check positive areas
        assert np.all(areas > 0), "All face areas should be positive"

    print(f"  PASSED (sideset {sideset_id}: {n_sides} faces)")


def test_invalid_sideset_id():
    """
    Test 4: Invalid sideset ID raises ValueError.
    """
    print("Test 4: Invalid sideset ID validation...")

    data_dir = Path(__file__).parent.parent / "data"
    mug_file = data_dir / "mug.e"

    if not mug_file.exists():
        print("  SKIPPED (mug.e not found)")
        return

    try:
        import exodusii
    except ImportError:
        print("  SKIPPED (exodusii not available)")
        return

    with ExodusSideInterpolator(str(mug_file)) as db:
        try:
            db.get_sideset_face_centroids(999999)
            assert False, "Expected ValueError for invalid sideset ID"
        except ValueError as e:
            assert "999999" in str(e), (
                f"Error message should mention the invalid ID: {e}"
            )

    print("  PASSED")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("ExodusSideInterpolator Tests")
    print("=" * 60)

    test_triangle_centroid_and_area()
    test_quad_area_weighted_centroid()
    test_sideset_extraction_from_exodus()
    test_invalid_sideset_id()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
