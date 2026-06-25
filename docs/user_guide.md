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
5. **Eigenvector Basis Validation** - Reconstruction-error and basis-dimension checks
6. **POD Mode Export from Sideset** - Pair `_real`/`_imag` sideset variables into a complex `(n_faces, n_modes)` `.npy`
7. **CPSD Inverse Problem** - Recover a POD-reduced CPSD from sparse experimental sensor CPSDs (and lift back to full space)
8. **Basis-Projection Residual** - Project a per-frequency data transfer matrix onto a fixed (frequency-independent) basis column space and report the relative approximation error per frequency

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
        "freq_indices": null,
        "all_freqs_svd": true
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
| `all_freqs_svd` | When `true`, performs SVD using all frequency snapshots stacked together instead of per-frequency (default: `false`) |

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
├── eigendata_freq0.npz              # Eigendata for frequency 0 (300 Hz)
├── eigendata_freq1.npz              # Eigendata for frequency 1 (400 Hz)
├── ...
├── eigendata_all_freqs.npz          # Eigendata from all-frequencies SVD (if enabled)
├── summary.json                     # Summary statistics
├── variance_explained_freq*.png     # Variance plots per frequency
├── variance_explained_all_freqs.png # Variance plot for all-frequencies SVD
├── eigenvectors_freq*_real.png      # Eigenvector visualizations per frequency
├── eigenvectors_freq*_imag.png
├── eigenvectors_freq*_mag.png
├── eigenvectors_all_freqs_real.png  # Eigenvector visualizations for all-frequencies SVD
├── eigenvectors_all_freqs_imag.png
└── eigenvectors_all_freqs_mag.png
```

Each per-frequency `.npz` file contains:
- `frequency`: Frequency in Hz
- `eigenvalues`: All computed eigenvalues
- `eigenvectors`: Retained eigenvectors (ndof × n_kept)
- `variance_explained`: Cumulative variance ratio

The `eigendata_all_freqs.npz` file (when `all_freqs_svd` is enabled) contains:
- `frequencies`: Array of all frequencies used in the analysis
- `eigenvalues`: All eigenvalues from the stacked SVD
- `eigenvectors`: Retained eigenvectors capturing dominant modes across all frequencies
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

## 5. Eigenvector Basis Validation

Validates the accuracy and efficiency of the CPSD eigenvector basis through two complementary analyses.

### Usage

```bash
# Run both analyses
python validation/run_validation.py \
    --eigendata-dir results_cone \
    --validation-set validation_data.npy \
    --output-dir validation_results \
    --analysis both

# Run only basis dimension analysis
python validation/run_validation.py \
    --eigendata-dir results_cone \
    --output-dir validation_results \
    --analysis dimension

# Run only reconstruction validation
python validation/run_validation.py \
    --eigendata-dir results_cone \
    --validation-set validation_data.npy \
    --output-dir validation_results \
    --analysis reconstruction
```

### Basis Dimension Analysis

Analyzes eigenvalue decay to determine how many modes are needed at each frequency.

**Output**:

- Minimum modes needed for 90%, 95%, 99%, 99.9% variance capture
- Scaling law fit: N_modes ≈ A × f^β (theoretical β = 2)
- Eigenvalue decay and cumulative variance plots

### Reconstruction Accuracy Validation

Tests whether the eigenvector basis can accurately reconstruct pressure fields from a validation set.

**Metrics**:

- Relative L2 reconstruction error per frequency
- Error histograms by octave band

**Validation Set Requirements**:

- Shape: `(ndof, n_fields, nfreqs)` - multiple pressure field realizations
- Should contain fields not used in eigenvalue computation

### Preparing Validation Data

Use `compute_total_field.py` to generate total pressure fields (incident + scattered) for validation:

```bash
python validation/compute_total_field.py \
    --transfer-matrix data/Tmatrix_cone_only.npy \
    --coordinates data/coordinates_cone_only.npy \
    --directions data/directions.npy \
    --frequencies 300 400 500 \
    --output validation_total_field.npy
```

This computes H = D + T where:

- D is the incident field from plane waves
- T is the scattered field from the transfer matrix

### Validation Output Files

```
validation_results/
├── eigenvalue_decay.png           # Eigenvalue decay curves
├── basis_dimension_report.txt     # Mode requirements and scaling laws
├── reconstruction_error.png       # Mean/max error vs frequency
├── error_histogram.png            # Overall error distribution
└── error_histogram_octave_bands.png  # Error by octave band
```

## 6. POD Mode Export from Sideset

Reads paired `_real`/`_imag` sideset variables from an ExodusII file (written by `run_sideset_interpolation.py`, see `docs/exodus_side_interpolator_summary.md`) and stacks them into a complex `(n_faces, n_modes)` `.npy` array suitable for use as the POD basis `Φ` in the CPSD inverse problem.

### Usage

```bash
python run_sideset_pod_export.py config_sideset_pod_export.json
python run_sideset_pod_export.py  # uses default config_sideset_pod_export.json
```

### Configuration File

```json
{
    "input": {
        "exodus_file": "data/cube.e",
        "sideset_id": 6,
        "variable_prefix": "pressure",
        "time_step": 1
    },
    "output": {
        "npy_path": "results/sideset_pod_modes.npy"
    }
}
```

### Configuration Options

#### Input Section

| Parameter | Description |
|-----------|-------------|
| `exodus_file` | Path to ExodusII database containing the sideset variables |
| `sideset_id` | Integer sideset ID to read from |
| `variable_prefix` | Variable name prefix (default `"pressure"`); the script reads `{prefix}_ev{i}_real` and `{prefix}_ev{i}_imag` for each mode `i` |
| `time_step` | 1-based time-step index to read (default `1`) |

#### Output Section

| Parameter | Description |
|-----------|-------------|
| `npy_path` | Path to the output `.npy` file; parent directories are created if needed |

### Output

A single `.npy` file containing a complex `(n_faces, n_modes)` array whose column `i` is mode `i+1` paired from `{prefix}_ev{i}_real + 1j·{prefix}_ev{i}_imag`. Modes missing either component are skipped with a warning.

### Behavior Notes

- Modes are sorted by their numeric index (`ev1`, `ev2`, …, `ev10`, …).
- The script fails if no sideset variables matching the pattern are found in the file, and prints the list of variables that were available.
- This file is the typical source of the POD basis `Φ` consumed by `run_cpsd_inverse.py` and `run_reconstruct_full_cpsd.py` (Section 7).

## 7. CPSD Inverse Problem

Recovers a reduced (POD-coordinate) CPSD `S_r` of shape `(n_pod, n_pod)` per frequency from a measured sensor CPSD `Ĝ` of shape `(n_sensors, n_sensors)`, using a reduced transfer matrix `T_r = T Φ` of shape `(n_sensors, n_pod, n_freq)`. Solves a Tikhonov-regularized inverse problem per frequency via a closed-form SVD expression. See `docs/cpsd_inverse_summary.md` for the math and `DiffuseFields_Inversion.pdf` for the derivation.

### Usage

```bash
python run_cpsd_inverse.py config_cpsd_inverse.json
```

### Configuration File

```json
{
    "input": {
        "transfer_matrix_path": "results/Tr.npy",
        "transfer_matrix_var": null,
        "pod_basis_path": "results/sideset_pod_modes.npy",
        "experimental_cpsd_path": "data/exp_cpsd.mat",
        "experimental_cpsd_var": "Sxx"
    },
    "physics": {
        "frequencies": null
    },
    "regularization": {
        "alpha": 1e-6,
        "psd_tol_rel": 0.0
    },
    "output": {
        "output_dir": "results_cpsd_inverse",
        "save_figures": true,
        "figure_format": "png"
    }
}
```

### Configuration Options

#### Input Section
| Parameter | Description |
|-----------|-------------|
| `transfer_matrix_path` | `.npy` or `.mat` with reduced transfer matrix `T_r`, shape `(n_sensors, n_pod, n_freq)` |
| `transfer_matrix_var` | MATLAB variable name; required when path ends in `.mat`, ignored for `.npy` |
| `transfer_matrix_scale` | Real constant γ multiplied into `T_r` before solving (default `1.0`). Use this to reconcile a units mismatch between `T_r` and the experimental CPSD `Ĝ` |
| `pod_basis_path` | `.npy` with POD basis `Φ`, shape `(N, n_pod)` |
| `experimental_cpsd_path` | `.mat` with experimental CPSD `Ĝ`, shape `(n_sensors, n_sensors, n_freq)` |
| `experimental_cpsd_var` | MATLAB variable name for the CPSD inside the `.mat` |

#### Physics Section
| Parameter | Description |
|-----------|-------------|
| `frequencies` | Optional list or `{min, step, max}`. Used only for plot/summary labels; alignment of files is always by frequency index. Length must equal `T_r.shape[2]` when supplied. |

#### Regularization Section
| Parameter | Description |
|-----------|-------------|
| `alpha` | Scalar α applied to every frequency (mutually exclusive with `alpha_sweep`) |
| `alpha_sweep` | List of α values applied to every frequency (e.g. `[1e-8, 1e-6, 1e-4]`); SVD is reused across α |
| `psd_tol_rel` | Relative threshold for clipping `Ĝ`'s eigenvalues before PSD square root (default 0.0 = clip only strictly negative) |

#### Output Section
| Parameter | Description |
|-----------|-------------|
| `output_dir` | Directory for per-frequency `.npz` files and summary |
| `save_figures` | Save the residual-vs-frequency plot |
| `figure_format` | `"png"`, `"pdf"`, `"svg"`, or `"eps"` |

### Output Files

```
results_cpsd_inverse/
├── cpsd_inverse_freq0.npz       # Recovered S_r at frequency index 0
├── cpsd_inverse_freq1.npz
├── ...
├── summary.json                  # Inputs, α values, per-frequency residuals
└── residual_vs_frequency.png     # Diagnostic plot (one curve per α)
```

Each per-frequency `.npz` contains, in **scalar-α mode**:

- `S_r` — `(n_pod, n_pod)` complex
- `alpha` — scalar
- `residual_rel` — `||T_r S_r T_r^h − Ĝ||_F / ||Ĝ||_F`
- `frequency` — scalar in Hz (only when `physics.frequencies` is set)

…or in **sweep mode** (`alpha_sweep` set):

- `S_r` — `(n_pod, n_pod, n_alpha)` complex
- `alphas` — `(n_alpha,)` real
- `residuals_rel` — `(n_alpha,)` real
- `frequency` — scalar in Hz (when supplied)

### Reconstructing the Full-Space CPSD

The lifted CPSD `S* = Φ S_r Φ^h` of shape `(N, N, n_freq)` is potentially huge (`N` may be tens of thousands of sideset faces), so it is **not** materialized by the inverse driver. Use `run_reconstruct_full_cpsd.py` to lift selected frequencies, optionally only the diagonal.

```bash
python run_reconstruct_full_cpsd.py config_reconstruct_full_cpsd.json
```

#### Configuration File

```json
{
    "input": {
        "inverse_results_dir": "results_cpsd_inverse",
        "pod_basis_path": "results/sideset_pod_modes.npy"
    },
    "reconstruction": {
        "freq_indices": null,
        "alpha_index": 0,
        "mode": "diagonal",
        "dtype": "complex128"
    },
    "output": {
        "output_path": "results_cpsd_inverse/full_cpsd_diag.npy"
    }
}
```

#### Reconstruction Options

| Parameter | Description |
|-----------|-------------|
| `freq_indices` | List of frequency indices to reconstruct (null = all available) |
| `alpha_index` | Which α from the sweep to use (ignored for scalar-α inversion output) |
| `mode` | `"full"` writes `(N, N, n_freq_selected)` complex; `"diagonal"` writes real `(N, n_freq_selected)` |
| `dtype` | `"complex64"` (halves storage) or `"complex128"` |

The output `.npy` is accompanied by a sidecar `.json` with the frequencies (if known), reconstruction mode, dtype, and input paths.

### Plotting the Diagonal CPSD vs Frequency

`run_plot_cpsd_diagonal.py` plots the uplifted diagonal `(N, n_freq)` from the reconstruction step, optionally compared against a **validation** data set. Select the plot kind(s) with `plot.kind` — a string or list drawn from:

- `"lines"` — per-location autopower `S_ii(f)` vs frequency. With a validation set, the inverse solution is solid and the validation data dashed, sharing one colour per location (solution-only without one).
- `"box"` — at each frequency, the distribution of `S_ii(f)` across the selected locations as side-by-side solution/validation boxes (IQR box, 5th/95th-percentile whiskers); switches to median + percentile bands above 40 frequencies. *Requires validation.*
- `"error"` — per-location relative-L2 error of the solution vs validation autopower spectrum, `‖S_ii^sol − S_ii^val‖₂ / ‖S_ii^val‖₂`, as a bar chart sorted worst → best (optional `output.top_n` cap). Ranks which sensors the inversion reproduces best/worst. *Requires validation.*

```bash
python run_plot_cpsd_diagonal.py config_plot_cpsd_diagonal.json
```

The validation data is a full CPSD `(n_loc, n_loc, n_freq_full)` (complex) supplied via `input.validation_path` (+ `input.validation_var` for `.mat`). Its real diagonal is aligned to the solution **by `selection.coordinates` order** — validation row `k` is the `k`-th coordinate — so a validation set requires coordinate selection. Its frequency axis must span the full inversion frequency set; it is sliced to the reconstructed subset via the sidecar's `freq_indices`. See [`docs/cpsd_inversion_guide.md`](cpsd_inversion_guide.md) §8 for the full config-key reference, alignment rules, and CSV-export details.

### Programmatic Usage

```python
from cpsd_inverse import CPSDInverseSolver
import numpy as np
from scipy.io import loadmat

T_r = np.load("results/Tr.npy")                 # (n_sensors, n_pod, n_freq)
phi = np.load("results/sideset_pod_modes.npy")  # (N, n_pod)
G   = loadmat("data/exp_cpsd.mat")["Sxx"]       # (n_sensors, n_sensors, n_freq)

solver = CPSDInverseSolver(T_r, pod_basis=phi)

# Single frequency, single α
S_r, residuals = solver.solve_single_freq(
    freq_idx=0, G=G[:, :, 0], alphas=np.array([1e-6])
)
S_r_f0 = S_r[:, :, 0]   # (n_pod, n_pod)

# Lift back to full space (diagonal only)
diag_S_full = solver.reconstruct_full_cpsd(S_r_f0, diagonal_only=True)
```

## 8. Basis-Projection Residual Analysis

The **basis** is a single frequency-independent matrix `B` of shape `(ndof, npws_basis)`. The **data** is a per-frequency transfer matrix of the same form as `data/Tmatrix_cone_only.npy`, shape `(ndof, npws_data, nfreq)` — the frequency dimension is the third axis of the data. A 2D data array `(ndof, npws_data)` is accepted as a single frequency (`nfreq = 1`). At each frequency, the data columns are orthogonally projected onto the **column space** of the basis, and the relative residual of that best (least-squares) approximation is reported and plotted versus frequency. This answers: *how well can the basis's achievable pressure fields represent the data fields, frequency by frequency?*

The basis column space does not change with frequency, so its orthonormalization is computed once. With `D = data[:, :, i]` at frequency `i`:

- Orthonormalize the basis columns via thin SVD, keeping columns with singular value `s > rtol·s[0]` (numerical rank), giving an orthonormal `Q` (computed once).
- Best approximation: `D_hat = Q (Q^h D)` (orthogonal projection onto `col(B)`).
- Relative residual: `||D − D_hat||_F / ||D||_F`.

The basis column space lives in `C^ndof`, so the basis and data must share the same `ndof` (rows); the number of plane waves (columns) may differ. A basis stored as `(ndof, npws, 1)` is squeezed to 2D automatically.

### Usage

```bash
# Minimal: two .npy files
python run_basis_projection.py basis.npy data.npy

# With output directory and Hz-labeled x-axis
python run_basis_projection.py basis.npy data.npy \
    --output-dir results_projection --frequencies data/freqs.npy

# MATLAB inputs (variable auto-detected if the .mat has a single variable)
python run_basis_projection.py basis.mat data.mat --basis-var H --data-var H
```

### Command-Line Options

| Option | Description |
|--------|-------------|
| `basis` | Path to the frequency-independent basis matrix `(ndof, npws)` (`.npy` or `.mat`), positional |
| `data` | Path to the per-frequency data matrix `(ndof, npws, nfreq)` (`.npy` or `.mat`), positional |
| `--output-dir` | Output directory (default: `results_projection`) |
| `--basis-var` | Variable name inside a `.mat` basis file (auto-detected if single variable) |
| `--data-var` | Variable name inside a `.mat` data file (auto-detected if single variable) |
| `--rtol` | Relative singular-value threshold for basis numerical rank (default: `1e-12`) |
| `--frequencies` | Path to a 1-D `.npy` of frequencies or a comma-separated list, used to label the x-axis in Hz (default: frequency index) |
| `--figure-format` | `"png"`, `"pdf"`, `"svg"`, or `"eps"` (default: `png`) |
| `--no-plots` | Skip plot generation |

### Output Files

```
results_projection/
├── projection_report.json              # Metadata, per-frequency residuals, summary stats
├── relative_residual.csv               # frequency, relative_residual
└── relative_residual_vs_frequency.png  # Relative residual vs frequency plot
```

`projection_report.json` contains:

- `metadata`: basis/data paths, `ndof`, `npws_basis`, `npws_data`, `nfreq`, `rtol`, and `basis_rank` (a single integer, since the basis is frequency-independent)
- `per_frequency`: `frequencies`, `relative_residual` (null where `||D||_F = 0`)
- `summary`: mean / min / max relative residual, and the worst frequency

### Programmatic Usage

```python
from basis_projector import BasisProjection
import numpy as np

basis = np.load("basis.npy")   # (ndof, npws_basis)        frequency-independent
data  = np.load("data.npy")    # (ndof, npws_data, nfreq)  frequency on third axis
                               # (ndof, npws_data) is accepted as a single frequency

result = BasisProjection(basis, data, rtol=1e-12).project()
result["relative_residual"]    # (nfreq,) ||D - D_hat||_F / ||D||_F
result["basis_rank"]           # int, numerical rank of the (fixed) basis
```

**Note**: column-space projection requires matching `ndof`. A basis and data with different numbers of rows cannot be projected; the tool raises a clear `ValueError` (likewise if the basis is not 2D or the data is not 3D).

## Typical Workflow

### Cone Surface Analysis

```bash
# 1. Filter mesh to exclude base disk
python filter_cone_mesh.py config_filter.json

# 2. Run eigenanalysis on cone surface
python run_cone_analysis.py config_cone_range.json

# 3. (Optional) Validate the eigenvector basis
python validation/compute_total_field.py --transfer-matrix data/Tmatrix_cone_only.npy ...
python validation/run_validation.py --eigendata-dir results_cone --validation-set total_field.npy

# 4. (Optional) Interpolate eigenvectors to different mesh
python run_interpolation.py config_interpolation.json
```

### Free-Field Convergence Study

```bash
# Study plane wave convergence in empty space
python run_diffuse_field.py config.json
```

### CPSD Inversion from Experimental Sensor Data

End-to-end pipeline that turns 3D-bulk POD eigenvectors into a full-space CPSD estimate `S*` on the structure surface, conditioned on measured sensor cross-spectra:

```bash
# 1. Cone (or other surface) eigenanalysis -> 3D POD modes in results_cone/
python run_cone_analysis.py config_cone_range.json

# 2. Interpolate those POD modes onto an ExodusII sideset and write them
#    as paired _real/_imag sideset variables (one call per mode)
#    See docs/exodus_side_interpolator_summary.md for config details
python run_sideset_interpolation.py config_sideset_interpolation.json

# 3. Read the sideset variables back into a (n_faces, n_modes) complex .npy
#    that serves as the POD basis Phi for the inverse problem
python run_sideset_pod_export.py config_sideset_pod_export.json

# 4. Solve the per-frequency Tikhonov inverse for S_r given T_r, Phi, and
#    the experimental sensor CPSD G_hat (.mat)
python run_cpsd_inverse.py config_cpsd_inverse.json

# 5. (Optional) Lift the reduced CPSD back to the full sideset space:
#    S* = Phi @ S_r @ Phi^h, full matrix or diagonal-only
python run_reconstruct_full_cpsd.py config_reconstruct_full_cpsd.json
```

The reduced transfer matrix `T_r = T Φ` (shape `(n_sensors, n_pod, n_freq)`) must be produced separately and saved as `.npy` or `.mat`; it is the projection of your full structure-to-sensor transfer matrix onto the same POD basis written by step 3.

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

### Sideset POD Basis (sideset_pod_modes.npy)
- Shape: `(n_faces, n_modes)`, complex-valued
- Column `i` is the POD mode `i+1` paired from sideset variables `{prefix}_ev{i}_real + 1j*{prefix}_ev{i}_imag`
- Produced by `run_sideset_pod_export.py`; consumed as `Φ` by `run_cpsd_inverse.py` and `run_reconstruct_full_cpsd.py`

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
