"""
Write a constant pressure field on sideset face centroids to a copy of mug.e.

Copies data/mug.e to data/mug-field.e, computes sideset face centroids,
and writes a constant sideset variable (pressure = 1.25) for verification.
"""

import shutil
import numpy as np
from pathlib import Path

from exodus_side_interpolator import ExodusSideInterpolator


def main():
    data_dir = Path("data")
    src_file = data_dir / "mug.e"
    dst_file = data_dir / "mug-field.e"

    # Copy the original file
    shutil.copy2(src_file, dst_file)
    print(f"Copied {src_file} -> {dst_file}")

    with ExodusSideInterpolator(str(dst_file), mode='a') as db:
        sideset_ids = db.get_sideset_ids()
        print(f"Sideset IDs: {sideset_ids}")

        for sid in sideset_ids:
            centroids = db.get_sideset_face_centroids(sid)
            areas = db.get_sideset_face_areas(sid)
            n_faces = centroids.shape[0]
            print(f"\nSideset {sid}: {n_faces} faces")
            print(f"  Area range: [{areas.min():.6e}, {areas.max():.6e}]")
            print(f"  Centroid x range: [{centroids[:,0].min():.4f}, {centroids[:,0].max():.4f}]")
            print(f"  Centroid y range: [{centroids[:,1].min():.4f}, {centroids[:,1].max():.4f}]")
            print(f"  Centroid z range: [{centroids[:,2].min():.4f}, {centroids[:,2].max():.4f}]")

            # Write a constant pressure field = 1.25
            pressure = 1.25 * np.ones(n_faces)
            db.write_sideset_variable(sid, "pressure", pressure, step=1)
            print(f"  Wrote 'pressure' variable (constant = 1.25)")

    print(f"\nDone. Output file: {dst_file}")


if __name__ == "__main__":
    main()
