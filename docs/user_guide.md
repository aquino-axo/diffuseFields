# Diffuse Acoustic Field Analysis - User Guide

This guide describes how to use the analysis tools for modeling diffuse acoustic fields.

## Overview

The codebase provides two categories of analysis:

### Free-Field Diffuse Analysis

Simulates diffuse fields in empty space using plane wave superposition. Use this to study convergence properties (e.g., how many plane waves are needed to accurately represent a diffuse field).

### Structural Scattering Analysis

Analyzes scattered pressure fields on structures (e.g., a cone) excited by diffuse acoustic loading. Uses transfer matrices from FE simulations where each column represents the structural response to a single incident plane wave. The eigenanalysis of the resulting CPSD matrix reveals the dominant spatial modes under diffuse excitation.

**Workflow**: First verify that the number of plane waves is sufficient using free-field analysis, then apply the same plane wave set to structural scattering problems.

## Tools

1. **Mesh Filtering** - Extract cone surface points, excluding the base disk
2. **Cone CPSD Analysis** - Eigenvalue decomposition of scattered field CPSD matrices
3. **Pressure Field Interpolation** - RBF-based interpolation between meshes
4. **Diffuse Field Simulation** - Free-field plane wave convergence studies

## 1. Mesh Filtering

The cone mesh may include a filled base disk that should be excluded for surface-only analysis.

### Usage

Using a config file:
```bash
python filter_cone_mesh.py config_filter.json
```

Using command-line arguments:
```bash
python filter_cone_mesh.py --base-z -0.1778 --coordinates data/coordinates.npy
```

Using defaults (base_z = -0.1778):
```bash
python filter_cone_mesh.py
```

### Configuration File (config_filter.json)

```json
{
    "coordinates_path": "data/coordinates.npy",
    "transfer_matrix_path": "data/Tmatrix.npy",
    "output_dir": "data",
    "base_z": -0.1778,
    "tolerance": 1e-6
}
```

### Command-Line Options

| Option | Description |
|--------|-------------|
| `config` | Path to JSON config file (optional, positional) |
| `--coordinates`, `-c` | Path to coordinates file (default: data/coordinates.npy) |
| `--transfer-matrix`, `-t` | Path to transfer matrix file (default: data/Tmatrix.npy) |
| `--output-dir`, `-o` | Output directory (default: data) |
| `--base-z`, `-z` | Z-coordinate of base disk to exclude (default: -0.1778) |
| `--tolerance` | Tolerance for z-coordinate comparison (default: 1e-6) |

### What It Does

- Loads coordinates and transfer matrix
- Filters out points at the specified `base_z` coordinate
- Saves filtered data:
  - `data/coordinates_cone_only.npy`
  - `data/Tmatrix_cone_only.npy`

### Example

```bash
# Filter with custom base location
python filter_cone_mesh.py --base-z -0.2 --output-dir filtered_data

# Or use a config file
python filter_cone_mesh.py config_filter.json
```

## 2. Cone CPSD Analysis

Performs eigenvalue decomposition of the cross-spectral power density (CPSD) matrix for diffuse field excitation on a cone surface.

### Usage

```bash
python run_cone_analysis.py config_cone_range.json
```

### Configuration File

```json
{
    "input": {
        "transfer_matrix_path": "data/Tmatrix_cone_only.npy",
        "coordinates_path": "data/coordinates_cone_only.npy",
        "directions_path": "data/directions.npy"
    },
    "physics": {
        "frequencies": {
            "min": 300.0,
            "step": 100.0,
            "max": 4000.0
        },
        "speed_of_sound": 343.0,
        "amplitude": 1.0
    },
    "eigenvalues": {
        "var_ratio": 0.999,
        "n_components": null,
        "solver": "direct",
        "freq_indices": null
    },
    "output": {
        "output_dir": "results_cone",
        "save_figures": true,
        "figure_format": "png",
        "save_eigenvectors": true,
        "plot_eigenvectors": true,
        "n_vectors_to_plot": 8,
        "plot_component": "magnitude"
    }
}
```

### Configuration Options

#### Input Section
| Parameter | Description |
|-----------|-------------|
| `transfer_matrix_path` | Path to transfer matrix (ndof × npws × nfreqs) |
| `coordinates_path` | Path to node coordinates (ndof × 3) |
| `directions_path` | Path to plane wave directions (npws × 3) |

#### Physics Section
| Parameter | Description |
|-----------|-------------|
| `frequencies` | List `[100, 200, 500]` or range `{"min": 300, "step": 100, "max": 4000}` |
| `speed_of_sound` | Speed of sound in m/s (default: 343.0) |
| `amplitude` | Plane wave amplitude (default: 1.0) |

#### Eigenvalues Section
| Parameter | Description |
|-----------|-------------|
| `var_ratio` | Variance ratio threshold (e.g., 0.999 keeps eigenvectors capturing 99.9% variance) |
| `n_components` | Fixed number of eigenvectors (overrides var_ratio if set) |
| `solver` | `"direct"` for full eigendecomposition, `"randomized"` for large problems |
| `freq_indices` | List of frequency indices to process (null = all) |

#### Output Section
| Parameter | Description |
|-----------|-------------|
| `output_dir` | Directory for results |
| `save_figures` | Save plots as image files |
| `figure_format` | `"png"` or `"pdf"` |
| `save_eigenvectors` | Save eigenvector data to .npz files |
| `plot_eigenvectors` | Generate eigenvector visualization plots |
| `n_vectors_to_plot` | Number of eigenvectors to visualize per frequency |
| `plot_component` | `"magnitude"`, `"real"`, `"imag"`, or `"phase"` |

### Output Files

```
results_cone/
├── eigendata_freq0.npz      # Eigendata for frequency 0 (300 Hz)
├── eigendata_freq1.npz      # Eigendata for frequency 1 (400 Hz)
├── ...
├── eigenvalues_summary.json # Summary statistics
├── n_components_vs_freq.png # Components needed vs frequency
├── variance_explained_*.png # Variance plots per frequency
└── eigenvectors_*.png       # Eigenvector visualizations
```

Each `.npz` file contains:
- `frequency`: Frequency in Hz
- `eigenvalues`: All computed eigenvalues
- `eigenvectors`: Retained eigenvectors (ndof × n_kept)
- `variance_explained`: Cumulative variance ratio

## 3. Pressure Field Interpolation

Interpolates complex-valued pressure fields from one mesh to another using RBF (Radial Basis Function) interpolation.

### Usage

```bash
python run_interpolation.py config_interpolation.json
```

### Configuration File

```json
{
    "input": {
        "source_coordinates_path": "data/coordinates_cone_only.npy",
        "target_coordinates_path": "data/target_coordinates.npy",
        "pressure_fields_path": "results_cone/eigendata_freq0.npz",
        "input_type": "eigendata"
    },
    "output": {
        "output_dir": "results_interpolation",
        "save_fields": true
    },
    "visualization": {
        "enabled": true,
        "n_fields": 3,
        "component": "magnitude",
        "save_figures": true,
        "figure_format": "png"
    }
}
```

### Configuration Options

#### Input Section
| Parameter | Description |
|-----------|-------------|
| `source_coordinates_path` | Path to source mesh coordinates |
| `target_coordinates_path` | Path to target mesh coordinates |
| `pressure_fields_path` | Path to pressure data (.npy or .npz) |
| `input_type` | `"pressure"` for raw .npy, `"eigendata"` for .npz from cone analysis |

#### Output Section
| Parameter | Description |
|-----------|-------------|
| `output_dir` | Directory for interpolated results |
| `save_fields` | Save interpolated pressure fields |

#### Visualization Section
| Parameter | Description |
|-----------|-------------|
| `enabled` | Generate comparison plots |
| `n_fields` | Number of fields to visualize (default: 3) |
| `component` | `"magnitude"`, `"real"`, `"imag"`, or `"phase"` |
| `save_figures` | Save plots to files |
| `figure_format` | `"png"` or `"pdf"` |

### Output Files

For `input_type: "eigendata"`:
```
results_interpolation/
├── interpolated_eigendata.npz  # Interpolated eigenvectors
├── interpolation_metadata.json # Processing metadata
└── comparison_*.png            # Source vs interpolated plots
```

For `input_type: "pressure"`:
```
results_interpolation/
├── interpolated_fields.npy     # Interpolated pressure fields
├── interpolation_metadata.json
└── comparison_*.png
```

### Programmatic Usage

```python
from pressure_interpolator import PressureFieldInterpolator
import numpy as np

# Load coordinates
source_coords = np.load("data/coordinates_cone_only.npy")
target_coords = np.load("data/target_coordinates.npy")

# Create interpolator
interpolator = PressureFieldInterpolator(source_coords, target_coords)

# Load pressure fields (complex-valued)
fields = np.load("pressure_fields.npy")  # Shape: (n_source, n_fields)

# Interpolate
interpolated = interpolator.interpolate(fields)  # Shape: (n_target, n_fields)
```

## 4. Diffuse Field Simulation

Generates synthetic diffuse acoustic fields and validates against the analytical sinc(kr) spatial correlation.

### Usage

```bash
python run_diffuse_field.py config.json
```

### Configuration File

```json
{
    "physics": {
        "frequency_min": 100.0,
        "frequency_max": 1000.0,
        "n_frequencies": 10,
        "speed_of_sound": 343.0
    },
    "simulation": {
        "n_plane_waves": 500,
        "n_realizations": 100,
        "amplitude": 1.0
    },
    "grid": {
        "grid_size": 21,
        "x_range": [-0.5, 0.5],
        "y_range": [-0.5, 0.5],
        "z_range": [-0.5, 0.5]
    },
    "eigenvalues": {
        "enabled": true,
        "n_components": 50,
        "freq_indices": [0, 4, 9]
    },
    "output": {
        "output_dir": "results_diffuse",
        "save_figures": true,
        "figure_format": "png"
    }
}
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `n_plane_waves` | Number of plane waves in superposition (more = better approximation) |
| `n_realizations` | Number of ensemble realizations for averaging |
| `grid_size` | Points per dimension for 3D grid |

### Output

- Correlation comparison plots (numerical vs analytical sinc(kr))
- Radial correlation profiles
- Normalized Mean Square Error (NMSE) vs frequency
- Eigenvalue decay plots

## Typical Workflow

### Cone Surface Analysis

```bash
# 1. Filter mesh to exclude base disk
python filter_cone_mesh.py

# 2. Run eigenanalysis on cone surface
python run_cone_analysis.py config_cone_range.json

# 3. (Optional) Interpolate eigenvectors to different mesh
python run_interpolation.py config_interpolation.json
```

### Diffuse Field Validation

```bash
# Generate synthetic diffuse field and validate correlation
python run_diffuse_field.py config.json
```

## Data Formats

### Transfer Matrix (Tmatrix.npy)
- Shape: `(ndof, npws, nfreqs)`
- Complex-valued pressure response at each DOF for each plane wave direction and frequency

### Coordinates (coordinates.npy)
- Shape: `(ndof, 3)`
- X, Y, Z coordinates of mesh nodes

### Directions (directions.npy)
- Shape: `(npws, 3)`
- Unit vectors for plane wave incident directions

### Eigendata (.npz)
- `frequency`: Scalar, frequency in Hz
- `eigenvalues`: Shape `(n_computed,)`
- `eigenvectors`: Shape `(ndof, n_kept)`, complex-valued
- `variance_explained`: Cumulative variance ratio

## Dependencies

- NumPy
- SciPy
- Matplotlib

## References

The diffuse field is modeled as a superposition of plane waves:

```
P(x) = (1/√N) Σ Aₙ exp(i(κDₙ·x + Φₙ))
```

The spatial correlation follows the sinc function:

```
G(r) = sinc(kr) = sin(kr)/(kr)
```

where k = 2πf/c is the wavenumber.
