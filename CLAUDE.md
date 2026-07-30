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

**basis_projector.py** - `BasisProjection` class for per-frequency basis-projection residual:
- Takes a frequency-independent basis `(ndof, npws_basis)` and a per-frequency data matrix `(ndof, npws_data, nfreq)` (frequency is the data's third axis); 2D data `(ndof, npws_data)` is treated as a single frequency
- Orthonormalizes the basis columns once via thin SVD (truncated at numerical rank `s > rtol*s[0]`); at each data frequency orthogonally projects the data columns onto the basis column space: `D_hat = Q @ (Q^H @ D)`
- Reports the relative residual `||D - D_hat||_F / ||D||_F` per frequency, plus the (scalar) basis rank and per-frequency data norm
- Basis and data must share `ndof` (rows); `npws` (columns) may differ

**run_basis_projection.py** - CLI driver for basis-projection analysis:
- Positional args `BASIS DATA` (`.npy` or `.mat`; for `.mat`, variable via `--basis-var`/`--data-var`, auto-detected when a single variable is present). A basis stored as `(ndof, npws, 1)` is squeezed to 2D.
- Options: `--output-dir`, `--rtol`, `--frequencies` (1-D `.npy` path or comma list to label the x-axis in Hz; defaults to frequency index), `--figure-format`, `--no-plots`
- Outputs `projection_report.json` (metadata incl. basis rank + per-frequency residuals + summary stats), `relative_residual.csv`, and a `relative_residual_vs_frequency.{fmt}` plot
- Usage: `python run_basis_projection.py basis.npy data.npy --output-dir results_projection`

**cpsd_inverse.py** - `CPSDInverseSolver` for the per-frequency regularized CPSD inverse problem of Aquino & Bonnet, JASA 2023 (`jasa23b.pdf`, Sec. III E). Recovers reduced CPSD `S_r` from a sparse sensor CPSD `Ĝ` given `T_r = T Φ`:
- `S_r = K K^h` with `K = Y diag(g(σ, α)) X^h Ψ`, where `T_r = X Σ Y^h` and `Ψ Ψ^h` is the PSD projection of `Ĝ`. PSD by construction for any α ≥ 0
- `spectral_filter(sigma, alpha, filter_form)` selects `g`: `'lavrentiev'` (default, `1/(σ+α)`) reproduces the paper's eq. 21 as printed; `'tikhonov'` (`σ/(σ²+α)`) is the minimizer of the least-squares problem eq. 19 states. **Eq. 20 of the paper misstates these as equal — they are not**; see `docs/cpsd_inverse_summary.md`
- `resolve_alphas(alphas, sigma_max, alpha_scaling, filter_form)` maps configured α to the applied α. Under `'relative'`, `α = value · σ_max(f)^p` with `p = 1` (lavrentiev) or `2` (tikhonov), keeping the configured value dimensionless. **Prefer `'relative'` for any band spanning resonances** — `σ_max` varies by orders of magnitude, so one absolute α damps very unevenly
- The paper's second method (eq. 24, entrywise `σ_iσ_j/(σ_i²σ_j²+α)`) is deliberately **not** implemented: it is not PSD-guaranteed for α > 0

**cpsd_inverse_cv.py** - `KFoldCVSelector` chooses α by k-fold cross-validation over the **sensor axis**. Score per (frequency, fold, α) is the held-out relative prediction residual plus `norm_weight · ‖S_r‖_F σ_max(f)²/‖Ĝ(f)‖_F`:
- The norm term (default weight `1e-2`) is essential near resonances: the CPSD forward map `S ↦ T_r S T_r^h` has condition number `cond(T_r)²`, and a pure prediction score under-regularizes by ~1 decade where `cond(T_r)` is worst. `norm_weight=0` restores the old prediction-only score
- It must be added to the **held-out** residual; adding it to the training residual is circular (returns α = the weight you supplied)
- The λ→α map uses the *full-matrix* `σ_max(f)`, not a fold's, so one candidate means one absolute α in every fold and in the refit

**run_cpsd_inverse.py** - Config-driven driver (`config_cpsd_inverse.json`). Key `regularization` settings: `alpha`/`alpha_sweep`, `psd_tol_rel`, `alpha_scaling`, `filter_form`; `cv` settings: `enabled`, `k_folds`, `alpha_mode`, `seed`, `norm_weight`. Writes per-frequency `.npz`, `summary.json` (incl. `alpha_scaling`, `filter_form`, `sigma_max_per_freq`, `alphas_effective`), `cv_results.npz`, and diagnostic plots. Full pipeline walkthrough in `docs/cpsd_inversion_guide.md`.

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
