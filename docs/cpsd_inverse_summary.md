# CPSD Inverse Problem — Implementation Summary

## What was built

A reduced-basis Tikhonov-regularized solver that recovers the cross-power
spectral density (CPSD) of a structural pressure field in a POD-reduced
coordinate system from sparse experimental sensor CPSDs.

Per the formulation in `DiffuseFields_Inversion.pdf`:

- `Φ ∈ C^(N×n)` — POD basis (full structural space → reduced)
- `T ∈ C^(m×N)` — observation transfer matrix (full structure → m sensors)
- `T_r = T Φ ∈ C^(m×n)` — reduced transfer matrix
- `Ĝ ∈ C^(m×m)` — measured sensor CPSD (one per frequency)
- `S_r ∈ C^(n×n)` — recovered reduced CPSD (the unknown)
- `S* = Φ S_r Φ^h` — lifted full-space CPSD

At each frequency we solve

```
S_r*(α) = argmin (1/2)·||T_r S_r T_r^h − Ĝ||_F² + (α/2)·||S_r||_F²
```

via the closed-form derived on slide 3 of the reference, using the reduced
SVD `T_r = X Σ Y^h`:

```
H_hat   = X^h Ĝ X
H_ij    = σ_i σ_j (H_hat)_ij / (σ_i² σ_j² + α)
S_r     = Y H Y^h
```

### Slide-3 correction

Slide 3 prints `Ĥ := Y Ĝ Y^h`. That is a typo — dimensions only work as
`Ĥ = X^h Ĝ X`. The implementation uses the dimensionally correct form,
computed via the equivalent identity `Ĥ = Z Z^h` where `Z = X^h Ψ` and
`Ψ Ψ^h` is the PSD projection of `Ĝ`.

## Files

| File | Purpose |
|------|---------|
| `src/cpsd_inverse.py` | `CPSDInverseSolver` class + `lift_to_full_space` helper |
| `src/run_cpsd_inverse.py` | Config-driven driver (load, validate, solve, save, plot) |
| `src/run_reconstruct_full_cpsd.py` | Config-driven driver to lift `S_r → S*` |
| `config_cpsd_inverse.json` | Example config for the inversion driver |
| `config_reconstruct_full_cpsd.json` | Example config for the reconstruction driver |
| `tests/test_cpsd_inverse.py` | 2 unit tests (recovery on synthetic data, scalar reduction) |

## Class API — `CPSDInverseSolver`

```python
solver = CPSDInverseSolver(reduced_transfer_matrix, pod_basis=None)
```

- `reduced_transfer_matrix` — shape `(n_sensors, n_pod, n_freq)`, complex
- `pod_basis` — optional `(N, n_pod)`, complex; required only if you later
  call `reconstruct_full_cpsd`

### Methods

- **`solve_single_freq(freq_idx, G, alphas, psd_tol_rel=0.0)`** — solves the
  inverse problem at one frequency for one or more regularization values.
  Returns `(S_r, residuals_rel)` where `S_r` has shape `(n_pod, n_pod,
  n_alpha)` and `residuals_rel` has shape `(n_alpha,)`. The SVD of `T_r` and
  the PSD projection of `Ĝ` are computed once and reused across all `alphas`.
- **`reconstruct_full_cpsd(S_r, diagonal_only=False)`** — lifts a reduced
  CPSD back to the full space: `Φ S_r Φ^h`. With `diagonal_only=True`,
  returns only the real-valued `(N,)` diagonal — useful when `N` is large.

### Module helper

- **`lift_to_full_space(S_r, pod_basis, diagonal_only=False)`** — free
  function used by the reconstruction driver and the class method.

## Numerical implementation details

### PSD projection of Ĝ

Experimental CPSDs from finite-time averaging or instrument noise can be
indefinite or non-Hermitian. We pre-process at each frequency:

1. `Ĝ ← (Ĝ + Ĝ^h)/2` — Hermitize.
2. Eigendecompose `Ĝ = U Λ U^h`.
3. Clip `λ_i < 0` (or `< tol_rel · max(|λ|)`) to zero.
4. Form `Ψ = U · diag(√λ)`. Then `Ψ Ψ^h` is the PSD projection of `Ĝ`.

This is then used as `Ĥ = (X^h Ψ)(X^h Ψ)^h = X^h (Ψ Ψ^h) X`.

### Sweeping α

When the config provides `regularization.alpha_sweep`, the SVD and `Ĥ` are
computed once per frequency; only the element-wise denominator
`σ_i² σ_j² + α` changes per α. Per-frequency cost is therefore roughly
constant in `n_alpha` after the SVD.

### Residual diagnostic

For each frequency and each α we save the relative Frobenius residual

```
||T_r S_r T_r^h − Ĝ||_F / ||Ĝ||_F
```

It is also plotted as a single `residual_vs_frequency.png` (one curve per α).

### Hermitization of S_r

`S_r = Y H Y^h` is theoretically Hermitian by construction, but in finite
precision it picks up tiny imaginary drift on the diagonal. We enforce
`S_r ← (S_r + S_r^h)/2` before saving.

## Driver: `run_cpsd_inverse.py`

```bash
python run_cpsd_inverse.py config_cpsd_inverse.json
python run_cpsd_inverse.py   # default config_cpsd_inverse.json
```

### Configuration

```json
{
  "input": {
    "transfer_matrix_path": "results/Tr.npy",
    "transfer_matrix_var": null,
    "transfer_matrix_scale": 1.0,
    "pod_basis_path":     "results/sideset_pod_modes.npy",
    "experimental_cpsd_path": "data/exp_cpsd.mat",
    "experimental_cpsd_var":  "Sxx"
  },
  "physics": { "frequencies": null },
  "regularization": { "alpha": 1e-6, "psd_tol_rel": 0.0 },
  "output": {
    "output_dir": "results_cpsd_inverse",
    "save_figures": true,
    "figure_format": "png"
  }
}
```

**`input` (required):**

| Field | Notes |
|---|---|
| `transfer_matrix_path` | `.npy` or `.mat` containing `T_r` of shape `(n_sensors, n_pod, n_freq)` |
| `transfer_matrix_var` | MATLAB variable name; required only when the path ends in `.mat`; ignored for `.npy` |
| `transfer_matrix_scale` | Real constant γ multiplied into `T_r` before solving (default `1.0`); used to reconcile a units mismatch between `T_r` and `Ĝ` |
| `pod_basis_path` | `.npy` containing `Φ` of shape `(N, n_pod)` |
| `experimental_cpsd_path` | `.mat` file containing `Ĝ` of shape `(n_sensors, n_sensors, n_freq)` |
| `experimental_cpsd_var` | MATLAB variable name to load from the `.mat` |

**`physics` (optional):**

| Field | Notes |
|---|---|
| `frequencies` | List `[100, 200, ...]` or `{min, step, max}`. Only used as metadata for plots and the summary; alignment of `T_r`, `Φ`, and `Ĝ` is always by frequency index. If supplied, length must equal `T_r.shape[2]`. |

**`regularization` (required: provide exactly one of `alpha` / `alpha_sweep`):**

| Field | Notes |
|---|---|
| `alpha` | Scalar α applied to every frequency |
| `alpha_sweep` | List of α applied to every frequency (e.g. `[1e-8, 1e-6, 1e-4]`); enables sweep mode |
| `psd_tol_rel` | Relative threshold for clipping `Ĝ`'s eigenvalues; `0.0` clips only strictly-negative ones |

**`output` (optional):**

| Field | Default | Notes |
|---|---|---|
| `output_dir` | `"results_cpsd_inverse"` | |
| `save_figures` | `true` | residual-vs-frequency PNG |
| `figure_format` | `"png"` | `png`, `pdf`, `svg`, `eps` |

### Output files

```
results_cpsd_inverse/
├── cpsd_inverse_freq0.npz
├── cpsd_inverse_freq1.npz
├── ...
├── summary.json
└── residual_vs_frequency.png
```

Each `cpsd_inverse_freqK.npz` contains, in scalar mode:

- `S_r` — `(n_pod, n_pod)` complex CPSD
- `alpha` — scalar
- `residual_rel` — scalar
- `frequency` — scalar (only when `physics.frequencies` was supplied)

…or in sweep mode:

- `S_r` — `(n_pod, n_pod, n_alpha)`
- `alphas` — `(n_alpha,)`
- `residuals_rel` — `(n_alpha,)`
- `frequency` — scalar (when supplied)

`summary.json` records the per-frequency residuals (lists per α), the
α values used, input file paths, and the frequency list when supplied.

The POD basis `Φ` is **not** copied into the output directory — the
reconstruction driver reads it directly from `pod_basis_path`.

## Driver: `run_reconstruct_full_cpsd.py`

Separated from the inverse driver so the potentially large full-space CPSD
`S* = Φ S_r Φ^h` (shape `(N, N, n_freq_selected)`) is only materialized
when explicitly requested.

```bash
python run_reconstruct_full_cpsd.py config_reconstruct_full_cpsd.json
```

### Configuration

```json
{
  "input": {
    "inverse_results_dir": "results_cpsd_inverse",
    "pod_basis_path":      "results/sideset_pod_modes.npy"
  },
  "reconstruction": {
    "freq_indices": null,
    "alpha_index":  0,
    "mode":         "diagonal",
    "dtype":        "complex128"
  },
  "output": {
    "output_path": "results_cpsd_inverse/full_cpsd_diag.npy"
  }
}
```

| Field | Notes |
|---|---|
| `freq_indices` | Subset of frequency indices to reconstruct (`null` = all) |
| `alpha_index` | Which α from the sweep to use (ignored for scalar-α inversion output) |
| `mode` | `"full"` writes `(N, N, n_freq_selected)` complex; `"diagonal"` writes real `(N, n_freq_selected)` |
| `dtype` | `"complex64"` (halves storage) or `"complex128"` |

### Output

- `output_path` — `.npy` with the requested array
- `output_path.json` — sidecar metadata (frequencies if known, mode, paths)

## Tests

Run with:

```bash
python tests/test_cpsd_inverse.py
```

Two correctness checks (the minimum set approved during planning, per
[CLAUDE.md](../CLAUDE.md)'s "keep tests to the absolute necessary ones"
guideline):

1. **Synthetic recovery** — for several `(m, n, n_freq)` shapes with
   `m ≥ n`, pick a random PSD `S_r_true`, form `Ĝ = T_r S_r_true T_r^h`,
   invert with α ≈ `1e-14 · σ_max⁴`, and assert
   `||S_r − S_r_true||_F / ||S_r_true||_F < 1e-5` and residual `< 1e-6`.
2. **n=1 scalar reduction** — verifies the closed-form reduces to
   `s = |t|² g / (|t|⁴ + α)` for several complex `t`, real `g`, and α
   triples (rtol `1e-12`).

Both tests pass.

## Convention notes (for cross-validation against measured data)

- **One-sided vs two-sided spectra.** The solver doesn't apply any spectral
  scaling — `Ĝ` is used as supplied. If your transfer matrix `T_r` was
  derived assuming a two-sided FFT convention but your measured `Ĝ` is
  one-sided (×2 on positive bins), the recovered `S_r` will be off by
  exactly that factor across all frequencies. Confirm both sides match
  before trusting absolute values.
- **Hermitization of Ĝ.** The driver enforces Hermiticity internally, so
  your `.mat` file can store any near-Hermitian estimate without manual
  symmetrization beforehand.
- **Conjugation conventions.** All complex math uses the standard
  `H^h := H.conj().T`. Inputs are read verbatim; no transposition or
  conjugation is applied to your `T_r`, `Φ`, or `Ĝ`.
