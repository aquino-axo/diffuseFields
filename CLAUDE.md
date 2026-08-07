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

**frequency_spec.py** - Shared parsing of frequency-vector specs, used by `run_cpsd_inverse.py` (`physics.frequencies`) and `run_plot_cpsd_diagonal.py` (`input.validation_frequencies`):
- `parse_frequency_spec(spec, field_name)` accepts a list or `{min, step, max}` (expanded inclusively of `max` when it lands on a step, otherwise stopping below it); `load_frequency_spec` additionally accepts a path to a 1-D `.npy` or single-variable `.mat`
- `field_name` is the caller's dotted config path and is interpolated into every error message. The other drivers (`run_cone_analysis.py`, `run_total_field.py`) still carry their own `parse_frequencies` copies

**run_plot_cpsd_diagonal.py** - Plots the uplifted CPSD diagonal vs frequency, optionally overlaid with validation data (`config_plot_cpsd_diagonal.json`). Kinds: `lines`, `box`, `error`, `validation_db`, `envelope`. The validation frequency axis has two modes:
- **Shared grid** (default): the validation array spans the inversion's full frequency set and is sliced by the sidecar's `freq_indices`; all four kinds available
- **Independent grid** (`input.validation_frequencies` set): validation carries its own frequency vector, nothing is sliced, and each series is drawn against its own frequencies. Also requires a sidecar carrying `frequencies` (an index x-axis cannot be overlaid with Hz)
- Which kinds survive an independent grid follows from whether they *difference* the two spectra. `lines` works. `validation_db` works but renders its **dB overlay panel only** — the `ΔL` panel, its max/median box, and `*_error_stats.csv` are all omitted, and `db_error` is never called. `box` (categorical per-frequency positions) and `error` (pointwise relative-L2) are **rejected** — deliberately not resolved by interpolating the measurement onto the solution's grid
- On an independent grid `output.top_n` degrades from "the N worst sensors" to "the first N selected" (nothing to rank by); the driver prints this
- CSVs go long format where a shared frequency column would be ill-defined: `lines` always (`series,frequency,index,label,value`), `validation_db` on independent grids (`series,frequency,index,label,level_db`)
- `envelope` shows the min-max spread across all selected sensors in dB, one band per series, centred on the **energetic mean** `10*log10(mean_i(S_ii)/ref)` (average powers *then* convert — the field's energy average; averaging dB values gives the geometric mean, always lower). Taken within each series, so it needs no paired frequencies and works on independent grids; needs no validation file either. Aggregate, so `per_sensor`/`top_n` do not apply
- `REQUIRES_VALIDATION_KINDS` = kinds that cannot be drawn without validation data; `DIFFERENCE_KINDS` = kinds additionally needing one shared grid. The two are distinct: `validation_db` is in the first only
- Legends are anchored outside the axes via `_legend_outside`; `_save_fig` uses `bbox_inches='tight'` so they are not clipped

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
