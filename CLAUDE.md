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
