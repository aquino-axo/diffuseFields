# ExodusSideInterpolator — Implementation Summary

## What was built

**`ExodusSideInterpolator`** — a class that wraps the `exodusii` Python package to read/write ExodusII databases, focused on sideset operations (face centroid extraction and sideset variable writing).

## Files created

| File | Purpose |
|------|---------|
| `exodus_side_interpolator.py` | Main class |
| `tests/test_exodus_side_interpolator.py` | 4 unit tests (all passing) |
| `write_sideset_field.py` | Demo script that writes a constant pressure (1.25) to `data/mug-field.e` |
| `run_sideset_interpolation.py` | Driver script for JSON-configured sideset interpolation |
| `config_sideset_interpolation.json` | Example configuration file |

## Class API

### Read operations

- `get_sideset_ids()` — list all sideset IDs
- `get_coords()` — node coordinates as `(n_nodes, 3)` array
- `get_sideset_face_centroids(sideset_id)` — area-weighted face centroids `(n_faces, 3)`
- `get_sideset_face_areas(sideset_id)` — face areas `(n_faces,)`

### Write operations

- `prepare_sideset_variables(variable_names)` — pre-register all sideset variable names and create netCDF storage (required before batch writes; NetCDF3 dimensions are fixed at creation)
- `write_sideset_variable(sideset_id, name, values, step)` — write a float array as a sideset variable (requires `mode='a'`). For a single variable, auto-prepares; for multiple variables, call `prepare_sideset_variables()` first.

### Infrastructure

- Context manager (`with` statement) for safe file handling
- Input validation on file existence, sideset IDs, and value shapes
- Caching of coordinates, connectivity, and element block map

## Key implementation details

### Area-weighted centroids

Triangle fan decomposition from first vertex; each sub-triangle's centroid is weighted by its area:

1. Fan-triangulate face from vertex 0: triangles (v0, v1, v2), (v0, v2, v3), ...
2. Each triangle: centroid = mean of 3 vertices, area = 0.5 * ||cross product||
3. Face centroid = sum(tri_centroid * tri_area) / sum(tri_area)
4. Degenerate case (total_area ≈ 0): return simple mean, area = 0

### exodusii library workarounds

The `exodusii` package has several bugs that required direct netCDF4 access:

- **`get_side_set_variable_names()`** raises `NotImplementedError` — we read variable names directly from `fh.variables['name_sset_var']`.
- **`put_side_set_variable_values()`** calls `get_edge_block_iid` instead of `get_side_set_iid` — we write values directly via `fh.variables[key][step - 1, :n_sides] = values`.
- **`put_side_set_variable_params()`** uses a nonexistent generic `num_side` dimension — we create per-sideset dimensions (`num_side_ss1`, `num_side_ss2`, etc.) directly.
- **`get_coords()`** fails on `mug.e` due to a coordinate name parsing bug — we read from `fh.variables['coord']` (shape `(3, n_nodes)`, transposed).

### FACE_NODE_MAP

Supports element types with 1-based side numbering mapped to 0-based local node indices:

- `HEX8` / `HEX`: 6 quad sides (4 nodes each)
- `TET4` / `TET`: 4 tri sides (3 nodes each)
- `TRI3`: 3 edge sides (2 nodes each)
- `QUAD4`: 4 edge sides (2 nodes each)

## Integration with PressureFieldInterpolator

The two classes are decoupled. The intended workflow:

```python
with ExodusSideInterpolator("mesh.e", mode='a') as db:
    centroids = db.get_sideset_face_centroids(sideset_id)

    interpolator = PressureFieldInterpolator(source_coords, centroids)
    pressure = interpolator.interpolate(source_pressure)

    db.write_sideset_variable(sideset_id, "pressure_real",
                              np.real(pressure))
    db.write_sideset_variable(sideset_id, "pressure_imag",
                              np.imag(pressure))
```

## Driver script: `run_sideset_interpolation.py`

A JSON-configured driver that automates the full workflow: load source data, interpolate onto sideset face centroids, and write results as sideset variables to the ExodusII file.

### Usage

```bash
python run_sideset_interpolation.py config_sideset_interpolation.json
python run_sideset_interpolation.py  # uses default config
```

### Configuration (`config_sideset_interpolation.json`)

```json
{
    "input": {
        "source_coordinates_path": "data/coordinates_cone_only.npy",
        "pressure_field_path": "results_cone/eigendata_freq0.npz",
        "input_type": "eigendata",
        "exodus_file": "data/mug.e",
        "sideset_id": 1
    },
    "interpolation": {
        "kernel": "thin_plate_spline",
        "smoothing": 0.0
    },
    "output": {
        "variable_prefix": "pressure",
        "time_step": 1
    }
}
```

**input (required):**

- `source_coordinates_path` — `.npy` file with source node coordinates `(n_source, 3)`
- `pressure_field_path` — `.npy` (pressure_field) or `.npz` (eigendata)
- `input_type` — `"pressure_field"` or `"eigendata"`
- `exodus_file` — path to ExodusII file (written to in-place)
- `sideset_id` — integer sideset ID

**interpolation (optional):**

- `kernel` — RBF kernel, default `"thin_plate_spline"`
- `smoothing` — RBF smoothing parameter, default `0.0`

**output (optional):**

- `variable_prefix` — prefix for sideset variable names, default `"pressure"`
- `time_step` — 1-based time step index, default `1`

### Variable naming convention

| Mode | Variables written |
| --- | --- |
| `pressure_field` | `{prefix}_real`, `{prefix}_imag` |
| `eigendata` (n eigenvectors) | `{prefix}_ev{i}_real`, `{prefix}_ev{i}_imag` (i = 1..n) |

### NetCDF3 batch write constraint

ExodusII uses NetCDF3, which only allows one unlimited dimension (`time_step`). The `num_sset_var` dimension is fixed at creation time, so all sideset variable names must be registered upfront via `prepare_sideset_variables()` before any writes. The driver handles this automatically.

## Test results

All 4 tests pass:

1. **Triangle centroid & area** — right triangle with known analytical result
2. **Non-planar quad centroid** — area-weighted result verified against manual two-triangle fan computation; confirmed to differ from simple geometric average
3. **Sideset extraction from `data/mug.e`** — 956 faces on sideset 1, centroids within mesh bounding box, all areas positive
4. **Invalid sideset ID** — raises `ValueError` with informative message

The write path was verified by writing constant 1.25 to `data/mug-field.e` and confirming via netCDF4 readback that both sidesets (956 and 152 faces) contain the correct values.
