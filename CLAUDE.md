# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a scientific research project for modeling 3D diffuse acoustic fields using plane wave superposition. The codebase simulates diffuse fields and validates them against the analytical sinc(kr) spatial correlation function. Applications include room acoustics, aerospace structural testing, and reverberation chamber design.

## Running the Simulation

```bash
python run_diffuse_field.py config.json
python run_diffuse_field.py  # Uses default config.json
```

No build step or package manager required. Dependencies: NumPy, SciPy, Matplotlib.

## Architecture

**diffuse_field.py** - Core `DiffuseField3D` class implementing:
- Plane wave superposition model with random phases and uniformly distributed directions
- Ensemble-averaged spatial correlation computation
- Analytical sinc(kr) reference correlation
- Eigenvalue decomposition of covariance matrices
- Visualization methods for correlation comparison, radial profiles, NMSE, and eigenvectors

**run_diffuse_field.py** - Driver script that:
- Loads and validates JSON configuration
- Orchestrates simulation workflow
- Computes correlations and optional eigenanalysis
- Generates plots and saves results

**config.json** - Simulation parameters including frequency range, physics constants (speed of sound), simulation settings (number of waves/realizations), and output options.

**cone_diffuse_field.py** - `ConeDiffuseField` class for cone surface CPSD analysis:
- Total field covariance: `C = Po² * H @ H^H` where `H = D + T` (incident + scattered)
- Per-frequency eigenanalysis via SVD of `H` (ndof × npws)
- All-frequencies eigenanalysis via SVD of stacked `H_all = [H_0 | ... | H_{nf-1}]` (ndof × npws·nf), capturing dominant spatial modes across all frequencies simultaneously

**run_cone_analysis.py** - Driver for cone analysis:
- Loads transfer matrix, coordinates, and directions from `.npy` files
- Computes per-frequency and optionally all-frequencies eigenanalysis
- Saves eigendata as `.npz` files and summary as JSON
- Generates variance explained and eigenvector plots

**config_cone.json / config_cone_range.json** - Cone analysis configuration. Key eigenvalue settings:
- `var_ratio`: variance ratio for truncation (default 0.99)
- `n_components`: fixed number of eigenvectors (overrides `var_ratio` if set)
- `solver`: `"direct"` (SVD of H) or `"randomized"` (matrix-free)
- `all_freqs_svd`: when `true`, performs SVD using all frequency snapshots stacked together instead of per-frequency, producing `eigendata_all_freqs.npz` and corresponding plots

**cone_visualizer.py** - `ConeVisualizer` class for 3D plots of pressure fields and eigenvalue decay on cone surfaces.

## Key Physics

The diffuse field is modeled as superposition of N plane waves:
```
P(x) = (1/√N) Σ Aₙ exp(i(κDₙ·x + Φₙ))
```

Spatial correlation follows the sinc function: `G(r) = sinc(kr) = sin(kr)/(kr)`

The Schroeder frequency marks the boundary above which natural diffuse fields occur. Below this frequency, active synthesis techniques using transfer matrix formulations and SVD-based inverse problems are needed.

## Specialized Agent

A custom agent (`diffuse-acoustic-field-expert`) is available for theoretical questions about:
- Spatial-temporal correlations and sinc/spherical Bessel formulations
- Schroeder frequency calculations
- Inverse problem formulation for diffuse field synthesis below Schroeder limit
- Low-rank approximation via spherical quadrature for large correlation matrices
- Tikhonov regularization strategies

## Other guidelines
-	Create classes respecting the single responsibility principle.
-	Make the code clean and extensible.
-	Create unit tests that are relevant to check correctness of the calculations.  Create a list of the tests first that I need to approve before proceeding.  
-	Keep the tests to the absolute necessary ones. Avoid tests that are trivial.
- When using an exodus database, you can refer to https://sandialabs.github.io/seacas-docs/sphinx/html/exodus.html
- Use exodusii when using the SEACAS package
