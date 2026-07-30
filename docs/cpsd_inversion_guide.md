# CPSD Inversion and Post-Processing — Handover Guide

This guide is a self-contained walkthrough of the five-step CPSD inversion
and post-processing pipeline, from POD-mode export off a sideset, through
solving the per-frequency inverse problem, to plotting the recovered CPSD
diagonal at chosen surface points.

It is intended for an engineer picking up the pipeline cold. Every step
is exercised by a config-driven driver in `src/`; this guide covers what
each driver consumes, what it produces, the relevant configuration keys,
and the gotchas we have already hit.

The math is summarized at a working level; see
[`jasa23b.pdf`](../jasa23b.pdf) (Aquino & Bonnet, JASA 2023, Sec. III E) for
the full derivation and
[`docs/cpsd_inverse_summary.md`](cpsd_inverse_summary.md) for
implementation notes on the solver class. The slide deck
[`DiffuseFields_Inversion.pdf`](../DiffuseFields_Inversion.pdf) presents the
paper's *second* regularization method (eq. 24), which is **not** what the code
implements — see [Math recap](#math-recap).

## Contents

1. [Pipeline at a glance](#pipeline-at-a-glance)
2. [Math recap (just enough to interpret)](#math-recap)
3. [Inputs you must have](#inputs-you-must-have)
4. [Step 1 — POD modes from sideset → `.npy`](#step-1)
5. [Step 2 — Per-frequency regularized inversion](#step-2)
   - [Row-index subset of the data](#step-2-row-subset)
   - [K-fold cross-validation for selecting α](#step-2-cv)
6. [Step 3 — Lift reduced CPSD to full space](#step-3)
7. [Step 4 — Write diagonal to an Exodus sideset](#step-4)
8. [Step 5 — Plot diagonal CPSD vs frequency at sensor points](#step-5)
9. [Worked example on `tests/data/cube.e`](#worked-example)
10. [File-format reference](#file-formats)
11. [Common pitfalls](#pitfalls)

---

<a name="pipeline-at-a-glance"></a>
## 1. Pipeline at a glance

```
                 (upstream: cone eigenanalysis, interpolation to sideset)
                                       │
                                       ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ Step 1  run_sideset_pod_export.py                            │
   │   exodus sideset vars  →  results/sideset_pod_modes.npy (Φ)  │
   └──────────────────────────────────────────────────────────────┘
                                       │
   ┌──────────────────────────────────────────────────────────────┐
   │ Step 2  run_cpsd_inverse.py                                  │
   │   T_r (.npy/.mat), Φ, Ĝ (.mat)  →  cpsd_inverse_freq*.npz    │
   │                                    summary.json              │
   │                                    residual_vs_frequency.png │
   └──────────────────────────────────────────────────────────────┘
                                       │
   ┌──────────────────────────────────────────────────────────────┐
   │ Step 3  run_reconstruct_full_cpsd.py                         │
   │   per-freq S_r + Φ  →  full_cpsd_diag.npy + sidecar .json    │
   │                        (or full N×N×n_freq cube)             │
   └──────────────────────────────────────────────────────────────┘
                                       │
              ┌────────────────────────┴────────────────────────┐
              ▼                                                 ▼
   ┌─────────────────────────┐                  ┌──────────────────────────┐
   │ Step 4  run_diagonal_   │                  │ Step 5  run_plot_cpsd_   │
   │   to_exodus.py          │                  │   diagonal.py            │
   │   write cpsd_diag var   │                  │   plot S_ii(f) at chosen │
   │   on the sideset        │                  │   indices / (x, y, z)    │
   └─────────────────────────┘                  └──────────────────────────┘
```

| Step | Driver | Config (default) | Output |
|---|---|---|---|
| 1 | `run_sideset_pod_export.py` | `config_sideset_pod_export.json` | `sideset_pod_modes.npy` |
| 2 | `run_cpsd_inverse.py` | `config_cpsd_inverse.json` | `cpsd_inverse_freq*.npz`, `summary.json` |
| 3 | `run_reconstruct_full_cpsd.py` | `config_reconstruct_full_cpsd.json` | `full_cpsd_diag.npy` (+ `.json` sidecar) |
| 4 | `run_diagonal_to_exodus.py` | `config_diagonal_to_exodus.json` | exodus file with `cpsd_diag` sideset variable |
| 5 | `run_plot_cpsd_diagonal.py` | `config_plot_cpsd_diagonal.json` | `diagonal_vs_frequency[_lines/_box/_error/_validation_db].png` (+ optional `.csv`, + `per_sensor/` for `validation_db`); solution-vs-validation overlay, per-location error, and measured-vs-computed dB comparison |

Steps 4 and 5 are post-processing siblings; both consume the diagonal
`.npy` and its sidecar from Step 3 and they can be run independently.

---

<a name="math-recap"></a>
## 2. Math recap

Symbols (all complex unless noted):

| Symbol | Shape | Meaning |
|---|---|---|
| `Φ` | `(N, n_pod)` | POD basis (full structural space → reduced) |
| `T` | `(m, N)` | Full transfer matrix (structure → `m` sensors) |
| `T_r = T Φ` | `(m, n_pod, n_freq)` | Reduced transfer matrix, per frequency |
| `Ĝ` | `(m, m, n_freq)` | Measured sensor CPSD, per frequency |
| `S_r` | `(n_pod, n_pod, n_freq)` | Recovered reduced CPSD (the unknown) |
| `S* = Φ S_r Φᴴ` | `(N, N, n_freq)` | Lifted full-space CPSD |

At each frequency we solve one regularized problem per column `φ_q` of `Ψ`
(where `Ψ Ψᴴ` is the PSD projection of `Ĝ`) and sum the outer products —
the structure of eqs. 19–21 in
[`jasa23b.pdf`](../jasa23b.pdf). With the reduced SVD `T_r = X Σ Yᴴ` and
`Z = Xᴴ Ψ`:

```
K   = Y diag(g(σ_i, α)) Z
S_r = K Kᴴ                    (PSD by construction for any α ≥ 0 and any g)
```

The diagonal filter `g` is selected by `regularization.filter_form`:

| `filter_form` | `g(σ, α)` | α units | notes |
|---|---|---|---|
| `"lavrentiev"` (default) | `1/(σ + α)` | σ | eq. 21 / Remark 2 as printed; reproduces the paper's published results |
| `"tikhonov"` | `σ/(σ² + α)` | σ² | minimizer of the least-squares problem eq. 19 states |

Both are monotone in σ, both keep `S_r` PSD, and both converge to the
minimum-norm solution as α → 0.

> **Important — eq. 20 of the reference contains an error.** It states that
> `Y (Σ + α I)⁻¹ Xᴴ φ_q` is the minimizer of the least-squares functional in
> eq. 19, `½‖T_r u − φ_q‖² + α‖u‖²`. It is not; that minimizer is
> `Y (Σ² + 2αI)⁻¹ Σ Xᴴ φ_q`. Remark 4's thin-QR route (QR of `[T_r; √α I]`)
> confirms eq. 19 is the intended problem and eq. 20 is the typo.
> `"lavrentiev"` is still a valid regularization — it solves the
> Lavrentiev-type equation
> `((T_rᴴT_r)^½ + αI) u = Qᴴ φ_q` for the polar factor — and it remains the
> default so existing results reproduce. Use `"tikhonov"` if you want eq. 19's
> minimizer. Full analysis in
> [`cpsd_inverse_summary.md`](cpsd_inverse_summary.md#the-error-in-eq-20).

**α scaling.** `σ_max(T_r(f))` varies by orders of magnitude across a band
containing resonances, so one absolute α applies very different damping at
different frequencies. `regularization.alpha_scaling` controls this:

| value | meaning |
|---|---|
| `"absolute"` (default) | α used as given, in the filter's own units |
| `"relative"` | α = value × `σ_max(T_r(f))^p`, so the value is dimensionless |

with `p = 1` for `"lavrentiev"` and `p = 2` for `"tikhonov"`. Because the power
follows the filter, a `"relative"` grid carries over unchanged if you switch
filters. **`"relative"` is recommended for any multi-frequency run**; the
default stays `"absolute"` only so existing configs keep their meaning.

The diagnostic shipped with each solve is the relative Frobenius residual

```
‖T_r S_r T_rᴴ − Ĝ‖_F / ‖Ĝ‖_F
```

computed against the raw (unclipped) `Ĝ`.

If you set `regularization.alpha_sweep`, the SVD and `Ψ` are computed
once per frequency and only the diagonal filter `g(σ_i, α)` changes — the
sweep is cheap.

---

<a name="inputs-you-must-have"></a>
## 3. Inputs you must have

| What | Format | Where it comes from |
|---|---|---|
| Reduced transfer matrix `T_r`, shape `(m, n_pod, n_freq)` | `.npy` or `.mat` | Built upstream by projecting `T` onto the same POD basis Φ written by Step 1. **Not produced by this pipeline.** |
| Exodus file with paired sideset variables `prefix_ev{i}_real`, `prefix_ev{i}_imag` | ExodusII (`.e`) | Written upstream by `run_sideset_interpolation.py`. |
| Measured sensor CPSD `Ĝ`, shape `(m, m, n_freq)` | `.mat` (single named variable) | Measurement campaign. |
| (Optional) Row-index subset of sensors to actually use | `.mat` (single named variable) | Built ad-hoc; see [Step 2 row-subset](#step-2-row-subset). |

**Frequency alignment is by index.** `T_r.shape[2]`, the third axis of
`Ĝ`, and (if supplied) `physics.frequencies` must all share the same
length. Physical frequencies in the config are *labels only*.

**Conjugation convention.** All complex math is standard
`Aᴴ = A.conj().T`. The drivers do not transpose or conjugate inputs —
hand them in the shape they advertise.

---

<a name="step-1"></a>
## 4. Step 1 — POD modes from sideset → `.npy`

Reads an Exodus file that already has paired POD-mode sideset variables
on a specified sideset, pairs them into complex columns, and writes a
single `(n_faces, n_modes)` complex `.npy` file. This `.npy` is the POD
basis `Φ` consumed by Steps 2 and 3.

**Driver:** [`src/run_sideset_pod_export.py`](../src/run_sideset_pod_export.py)
**Config:** [`config_sideset_pod_export.json`](../config_sideset_pod_export.json)

```bash
python src/run_sideset_pod_export.py config_sideset_pod_export.json
```

### Config keys

```json
{
  "input": {
    "exodus_file":     "data/cube.e",
    "sideset_id":      6,
    "variable_prefix": "pressure",
    "time_step":       1
  },
  "output": {
    "npy_path": "results/sideset_pod_modes.npy"
  }
}
```

| Key | Required? | Notes |
|---|---|---|
| `input.exodus_file` | yes | Path to the ExodusII file. |
| `input.sideset_id` | yes | Integer sideset ID. |
| `input.variable_prefix` | default `"pressure"` | Script reads `{prefix}_ev{i}_real` and `{prefix}_ev{i}_imag` for each `i`. |
| `input.time_step` | default `1` | 1-based time-step index to read. |
| `output.npy_path` | yes | Parent directories are created automatically. |

### What it produces

A single complex `(n_faces, n_modes)` array. Column `i` is the POD mode
`i+1`, paired as `{prefix}_ev{i}_real + 1j*{prefix}_ev{i}_imag`. Modes
missing either component are skipped with a warning. Modes are sorted by
numeric index, so `ev10` follows `ev9`.

### Gotchas

- The script fails if no variables match the pattern; it prints the
  available sideset variable names so you can fix `variable_prefix`.
- The number of faces is taken from the sideset itself, not the `.npy`,
  so this same `Φ` will line up with the same sideset later in Steps 4
  and 5.

---

<a name="step-2"></a>
## 5. Step 2 — Per-frequency regularized inversion

Solves `S_r(f)` per frequency given `T_r`, `Φ`, and `Ĝ`.

**Driver:** [`src/run_cpsd_inverse.py`](../src/run_cpsd_inverse.py)
**Config:** [`config_cpsd_inverse.json`](../config_cpsd_inverse.json)

```bash
python src/run_cpsd_inverse.py config_cpsd_inverse.json
```

### Config keys

```json
{
  "input": {
    "transfer_matrix_path":    "results/Tr.npy",
    "transfer_matrix_var":     null,
    "transfer_matrix_scale":   1.0,
    "pod_basis_path":          "results/sideset_pod_modes.npy",
    "experimental_cpsd_path":  "data/exp_cpsd.mat",
    "experimental_cpsd_var":   "Sxx",
    "row_indices_path":        null,
    "row_indices_var":         null,
    "row_indices_one_based":   true
  },
  "physics": { "frequencies": null },
  "regularization": {
    "alpha":         1e-6,
    "alpha_sweep":   null,
    "psd_tol_rel":   0.0,
    "alpha_scaling": "absolute",
    "filter_form":   "lavrentiev"
  },
  "output": {
    "output_dir":    "results_cpsd_inverse",
    "save_figures":  true,
    "figure_format": "png"
  }
}
```

#### `input`

| Key | Notes |
|---|---|
| `transfer_matrix_path` | `.npy` or `.mat` holding `T_r`, shape `(m, n_pod, n_freq)`. |
| `transfer_matrix_var` | MATLAB variable name; **required** when the path ends in `.mat`. |
| `transfer_matrix_scale` | Real, finite, non-zero γ multiplied into `T_r` before solving. Use this to reconcile a units mismatch between `T_r` and `Ĝ`. Default `1.0`. |
| `pod_basis_path` | `.npy` with Φ from Step 1, shape `(N, n_pod)`. |
| `experimental_cpsd_path` | `.mat` with Ĝ, shape `(m, m, n_freq)`. |
| `experimental_cpsd_var` | MATLAB variable name inside the `.mat`. |
| `row_indices_path` / `_var` / `_one_based` | See [Row-index subset](#step-2-row-subset). |

#### `physics`

| Key | Notes |
|---|---|
| `frequencies` | List `[f0, f1, …]` or `{min, step, max}`. Used *only* as metadata for plots and `summary.json`; if supplied, length must equal `T_r.shape[2]`. |

#### `regularization` — provide exactly one of `alpha` or `alpha_sweep`

| Key | Notes |
|---|---|
| `alpha` | Scalar α applied at every frequency. |
| `alpha_sweep` | List of α values, e.g. `[1e-8, 1e-6, 1e-4]`. The SVD of `T_r` and the PSD projection of Ĝ are computed once per frequency and reused across all α. |
| `psd_tol_rel` | Relative threshold for clipping Ĝ's eigenvalues before the PSD square root. `0.0` clips only strictly-negative eigenvalues. |
| `alpha_scaling` | `"absolute"` (default) or `"relative"` (α = value × `σ_max(T_r(f))^p`). **Use `"relative"` for any band spanning resonances.** See [Math recap](#math-recap). |
| `filter_form` | `"lavrentiev"` (default, `1/(σ+α)`) or `"tikhonov"` (`σ/(σ²+α)`). See [Math recap](#math-recap). |

Switching `filter_form` changes what α *means* (σ versus σ²), so an
`"absolute"` grid must be re-chosen. A `"relative"` grid carries over
unchanged, because the σ power follows the filter.

#### `output`

| Key | Notes |
|---|---|
| `output_dir` | Default `"results_cpsd_inverse"`. |
| `save_figures` | If true, writes `residual_vs_frequency.png` (one curve per α in sweep mode). |
| `figure_format` | `png`, `pdf`, `svg`, or `eps`. |

### What it produces

```
results_cpsd_inverse/
├── cpsd_inverse_freq0.npz
├── cpsd_inverse_freq1.npz
├── ...
├── summary.json
└── residual_vs_frequency.png
```

Each `cpsd_inverse_freqK.npz`, scalar-α mode:
- `S_r` — `(n_pod, n_pod)` complex
- `alpha` — scalar
- `residual_rel` — scalar
- `frequency` — scalar in Hz (when `physics.frequencies` supplied)

Sweep mode adds the α axis:
- `S_r` — `(n_pod, n_pod, n_alpha)`
- `alphas` — `(n_alpha,)`
- `residuals_rel` — `(n_alpha,)`

`summary.json` records the per-frequency residuals (lists per α), the α
values used, input file paths, the `transfer_matrix_scale` actually
applied, and the row-subset metadata described next.

<a name="step-2-row-subset"></a>
### 5.1 Row-index subset of the data

Use this when the inverse problem should use only a *subset* of the
sensor rows — for example, when the available data covers only a subset
of the sensors that the transfer matrix was built for.

The subset is an integer index set stored in a `.mat` file. When loaded,
it is applied symmetrically: `T_r' = T_r[I, :, :]` and
`Ĝ' = Ĝ[I, I, :]` per frequency. The rest of the pipeline runs against
the reduced `(T_r', Ĝ')` exactly as if the smaller problem had been
provided directly.

#### Keys (under `input`)

| Key | Notes |
|---|---|
| `row_indices_path` | Path to a `.mat` file. **Default `null`** → use all rows (no subset). |
| `row_indices_var` | MATLAB variable name. **Required** when `row_indices_path` is set. |
| `row_indices_one_based` | `true` (default) treats the indices as MATLAB 1-based and subtracts 1 internally. Set to `false` if you saved 0-based indices from Python. |

#### Index-set semantics

- Must be 1-D (any shape that squeezes to 1-D is accepted).
- Must be integer-valued (e.g. `int32`, `int64`, or floats that exactly
  represent integers).
- Must be **unique** after conversion.
- Must lie in the valid range:
  - `[1, m]` when `row_indices_one_based = true`
  - `[0, m−1]` when `row_indices_one_based = false`
- Order is preserved as given — useful if you care about a particular
  presentation order downstream (it does *not* affect the inversion
  result).

If `row_indices_path` is null, everything reverts to the full-row
behavior.

#### What gets recorded in `summary.json`

| Field | Meaning |
|---|---|
| `n_sensors_full` | Original `m` before any subsetting. |
| `row_indices_path` | Echoed config value (or `null`). |
| `row_indices_var` | Echoed config value (or `null`). |
| `row_indices_one_based` | Echoed config flag. |
| `row_indices` | Resolved 0-based index list actually used, or `null` if no subset. |

#### Error cases

The driver raises with a clear message in any of these:

- `row_indices_path` set but `row_indices_var` not set.
- `row_indices_path` set but the file isn't `.mat`.
- The variable is non-numeric, has more than one non-singleton axis,
  is empty, contains duplicates, or has indices out of range.

<a name="step-2-cv"></a>
### 5.2 K-fold cross-validation for selecting α

When you don't know a priori what α to pass, enable K-fold
cross-validation. The driver searches `regularization.alpha_sweep`,
scores each candidate on held-out sensors, picks α*, and **refits** on
the full (downselected) sensor set with that α*. The saved S_r is the
refit one — exactly the same scalar-α `.npz` schema as a non-CV scalar
run.

CV is performed on the **already-downselected** sensor set: if
`input.row_indices_path` is set, folds split that subset; otherwise
folds split all `m` rows. The math operates the same way the rest of
the pipeline does — symmetric row/column slicing of `T_r` and `G`.

#### Config keys (top-level `cv` block)

```json
"cv": {
  "enabled": false,
  "k_folds": 5,
  "alpha_mode": "global",
  "seed": 0,
  "save_fold_scores": false,
  "norm_weight": 1e-2
}
```

| Key | Notes |
|---|---|
| `cv.enabled` | Master switch. Default `false` → existing scalar/sweep behavior. |
| `cv.k_folds` | Integer ≥ 2; default `5`. Validation error if `\|I\| < k_folds`. |
| `cv.alpha_mode` | `"global"` (default) or `"per_freq"`. See below. |
| `cv.seed` | Seed for the `numpy.default_rng` shuffle of the index set. Same seed → same partition. Default `0`. |
| `cv.save_fold_scores` | When `true`, `cv_results.npz` includes the per-fold score array of shape `(n_freq, n_alpha, k_folds)`. Default `false` (saves memory). |
| `cv.norm_weight` | Dimensionless weight μ on the solution-norm term of the CV score; default `1e-2`. `0.0` gives the pure held-out prediction score. Useful range ≈ `1e-3`–`1e-1`. See [The score](#step-2-cv-score). |

When `cv.enabled` is `true`:

- `regularization.alpha_sweep` becomes the CV candidate grid and is
  **required**.
- `regularization.alpha` (scalar) is **forbidden**.

#### Algorithm

For each frequency `f`, each fold `k`, and each candidate α:

1. `I_train = I \ I_fold_k`, `I_val = I_fold_k`.
2. Hermitize and PSD-clip both `G[I_train, I_train, f]` and
   `G[I_val, I_val, f]` using `regularization.psd_tol_rel`.
3. SVD of `T_r[I_train, :, f]` once per (f, k); reused across α.
4. `S_r = K Kᴴ` via the same closed form as the non-CV solver.
5. Predict `Ĝ_pred = T_r[I_val, :, f] S_r T_r[I_val, :, f]ᴴ`.
6. ```
   score(f, α, k) = ‖Ĝ_pred − G_val_clipped‖_F / ‖G_val_clipped‖_F
                  + norm_weight · ‖S_r‖_F · σ_max(f)² / ‖Ĝ(f)‖_F
   ```

CV score per `(f, α)` is the mean over folds.

`σ_max(f)` here is that of the **full** `T_r[:, :, f]`, not the fold's
training subset, so one candidate maps to one and the same absolute α in
every fold and in the refit.

<a name="step-2-cv-score"></a>
#### The score, and why it has two terms

The first term alone is a pure data-fit criterion. At ill-conditioned
frequencies it is nearly blind to the directions the ill-conditioning
amplifies — many very different `S_r` reproduce almost the same sensor CPSD —
because the CPSD forward map `S ↦ T_r S T_rᴴ` has condition number
`cond(T_r)²`. Empirically a prediction-only score **under-regularizes by
about a decade** exactly where `cond(T_r)` is worst, i.e. in the bands
flanking resonances. The second term supplies the information the prediction
score lacks. Its `σ_max(f)²/‖Ĝ(f)‖_F` factor makes it dimensionless (since
`‖S_r‖ ~ ‖Ĝ‖/σ_max²`), so a single `norm_weight` works across a whole band.

The norm term must be added to the **held-out** residual, never the training
residual. `S_r(α)` already minimizes (training fit) + α·(penalty), so
minimizing (training fit) + μ·(penalty) over the family `{S_r(α)}` returns
α = μ identically — a circular criterion that just echoes the constant you
supplied. Scoring the fit on sensors the fold did not see breaks that fixed
point. (It is also why an L-curve takes the *corner* of
`(log‖S_r‖, log‖residual‖)` rather than a weighted sum: curvature is
invariant to the weighting.)

Selection:

- `alpha_mode = "global"` — aggregate `score(f, α)` by mean over
  frequencies → pick one scalar α* (default; recommended for stability
  across frequencies).
- `alpha_mode = "per_freq"` — independently `argmin_α score(f, α)` per
  frequency → vector α*(f).

Tie-breaking uses `numpy.argmin` (first occurrence), which on a sorted
log-spaced grid picks the smallest α among ties.

#### Refit

After selection, the driver calls `solver.solve_single_freq(f, G[:,:,f],
[α*(f)], psd_tol_rel)` for every frequency to produce the final S_r.
Per-frequency `.npz` files have the scalar-α schema (`S_r (n_pod,
n_pod)`, `alpha`, `residual_rel`, optional `frequency`) — downstream
post-processing (Steps 3–5) does not need to know CV happened.

#### Outputs added in CV mode

`results_cpsd_inverse/cv_results.npz` containing:

| Key | Shape & dtype |
|---|---|
| `alphas` | `(n_alpha,)` float64 — the searched grid |
| `scores` | `(n_freq, n_alpha)` float64 — mean over folds |
| `fold_scores` | `(n_freq, n_alpha, k_folds)` float64 — only if `save_fold_scores=true` |
| `alpha_star` | `(n_freq,)` or `(1,)` float64 |
| `alpha_mode` | scalar string |
| `k_folds` | scalar int |
| `seed` | scalar int |

`summary.json` gains a `cv` subsection (`{enabled, k_folds, alpha_mode,
seed, norm_weight, alpha_grid, alpha_star}`) plus `alphas_per_freq` (the α
actually used at each frequency in the refit — useful for audit when α
varies).

Every run — CV or not — also records `alpha_scaling`, `filter_form`,
`sigma_max_per_freq`, and `alphas_effective` (the absolute α actually applied
per frequency). Under `"relative"` scaling the grid values alone do not tell
you what was applied, so check `alphas_effective` when auditing a run.

Diagnostic plots written when `output.save_figures` is `true`:

- `cv_score_vs_alpha.png` — log-log CV score curves, one per frequency,
  with the global-mean curve overlaid; legend auto-disabled when
  `n_freq > 20`. Marker at α*.
- `cv_score_heatmap.png` — `(α, frequency)` pcolormesh, log α and log
  color, α*(f) overlaid as red stars (per-frequency mode) or a vertical
  dashed line (global mode).
- `residual_vs_frequency.png` — unchanged: post-refit relative residual
  on the full downselect with α*.

#### When to prefer per_freq vs global

- **Global** is the recommended default. It is more stable, easier to
  interpret, and fully avoids overfitting α to noisy CV scores at
  individual frequencies.
- **Per-frequency** is appropriate when the SNR and conditioning of
  `T_r(f)` are known to vary by orders of magnitude across the band and
  you have enough sensors per fold to make the per-band CV score
  trustworthy. Inspect `cv_score_heatmap.png` first to confirm α*(f) is
  smooth rather than jumpy.

#### CV error cases

The driver raises a clear message in any of these:

- `cv.enabled=true` and `regularization.alpha` is set (scalar mode).
- `cv.enabled=true` and `regularization.alpha_sweep` is null.
- `cv.k_folds < 2` or larger than `|I|`.
- `cv.alpha_mode` is not `"per_freq"` or `"global"`.

#### Cost note

Inner cost per `(frequency, fold)` is dominated by an SVD of
`T_r[I_train, :, f]` of size roughly `(|I|·(K-1)/K) × n_pod`; the α loop
inside reuses the SVD. Total cost ≈ `n_freq × K × O(m·n_pod²)` which is
trivial for the matrix sizes typical of this pipeline (a few seconds
even for hundreds of frequencies).

---

<a name="step-3"></a>
## 6. Step 3 — Lift reduced CPSD to full space

Materializes `S* = Φ S_r Φᴴ`. The full `(N, N, n_freq)` cube can be huge
(`N` is the number of sideset faces), so the default and recommended
mode is `diagonal` — keep only `diag(S*)`, shape `(N, n_freq)`, real.

**Driver:** [`src/run_reconstruct_full_cpsd.py`](../src/run_reconstruct_full_cpsd.py)
**Config:** [`config_reconstruct_full_cpsd.json`](../config_reconstruct_full_cpsd.json)

```bash
python src/run_reconstruct_full_cpsd.py config_reconstruct_full_cpsd.json
```

### Config keys

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

| Key | Notes |
|---|---|
| `input.inverse_results_dir` | Directory written by Step 2; the driver discovers `cpsd_inverse_freq*.npz` files inside it. |
| `input.pod_basis_path` | Same Φ used in Step 2. |
| `reconstruction.freq_indices` | List of indices to reconstruct (`null` = all available). |
| `reconstruction.alpha_index` | Which α from the sweep to use. Ignored when Step 2 ran in scalar-α mode. |
| `reconstruction.mode` | `"full"` writes complex `(N, N, n_freq_selected)`. `"diagonal"` writes real `(N, n_freq_selected)`. |
| `reconstruction.dtype` | `"complex64"` (halves storage and is plenty for plotting) or `"complex128"`. The diagonal output is real-typed accordingly (`float32`/`float64`). |

### What it produces

- The chosen `.npy` at `output.output_path`.
- A sidecar `.json` at the same path with `.json` suffix, containing the
  resolved `freq_indices`, `mode`, `dtype`, `alpha_index`, the resolved
  `frequencies` (when known), and the input paths.

The sidecar is what Steps 4 and 5 use to recover physical frequencies —
keep it next to the `.npy`.

---

<a name="step-4"></a>
## 7. Step 4 — Write diagonal to an Exodus sideset

Writes `diag(S*)` from Step 3 as a time-varying sideset variable. Each
frequency becomes one time step; the sideset row ordering matches Φ's
row ordering, so no interpolation happens here.

**Driver:** [`src/run_diagonal_to_exodus.py`](../src/run_diagonal_to_exodus.py)
**Config:** [`config_diagonal_to_exodus.json`](../config_diagonal_to_exodus.json)

```bash
python src/run_diagonal_to_exodus.py config_diagonal_to_exodus.json
```

### Config keys

```json
{
  "input": {
    "diagonal_npy_path": "results_cpsd_inverse/full_cpsd_diag.npy",
    "sidecar_json_path": "results_cpsd_inverse/full_cpsd_diag.json",
    "exodus_file":       "data/cube.e",
    "sideset_id":        6
  },
  "output": {
    "variable_name":          "cpsd_diag",
    "use_frequency_as_time":  true,
    "start_step":             1,
    "exodus_file":            "data/cube_diag.e",
    "copy_from_exodus_file":  "data/cube_backup.e",
    "overwrite":              true,
    "strip_sideset_vars":     true
  }
}
```

| Key | Notes |
|---|---|
| `input.diagonal_npy_path` | Real `(N, n_freq)` `.npy` from Step 3. |
| `input.sidecar_json_path` | Optional; defaults to the `.json` sibling of the `.npy`. Used to resolve physical frequencies. |
| `input.exodus_file` | The Exodus file whose *sideset geometry* the diagonal is keyed to. The driver verifies `N == n_faces` on the chosen sideset. |
| `input.sideset_id` | Integer sideset ID. |
| `output.variable_name` | Default `"cpsd_diag"`. Must be a non-empty string. |
| `output.use_frequency_as_time` | If true and the sidecar carries frequencies, the Exodus *time* axis is set to frequency in Hz (so post-processors will say "time = 1500 Hz"). Otherwise time = integer step index. |
| `output.start_step` | 1-based step index for the first frequency. Default 1. |
| `output.exodus_file` | If set, write to a separate file instead of modifying `input.exodus_file` in place. |
| `output.copy_from_exodus_file` | When writing to a new file, seed it from this path. Defaults to `input.exodus_file`. |
| `output.overwrite` | If true, an existing `output.exodus_file` is removed before seeding. |
| `output.strip_sideset_vars` | Default true. Strips pre-existing sideset variable metadata from the seed copy so the new `cpsd_diag` variable can be registered. **See pitfalls.** |

### What it produces

The named Exodus file with one new sideset variable (`cpsd_diag` by
default), one time step per frequency, on the chosen sideset only.
Other sidesets are left untouched.

### `strip_sideset_vars` — why this exists

Exodus stores the sideset-variable count in the fixed netCDF-3
dimension `num_sset_var`. Once a file has any sideset variables, that
dimension is locked. If you copy `data/cube.e` (which already has the
`pressure_ev{i}_*` POD modes written into it from upstream
interpolation) and try to register a *new* variable, the write will
fail. `strip_sideset_vars: true` solves this by seeding a clean copy
that drops `num_sset_var` and the existing per-sideset variable data,
keeping the mesh + sidesets intact. Set it to `false` only if the seed
file has no sideset variables to begin with.

---

<a name="step-5"></a>
## 8. Step 5 — Plot diagonal CPSD vs frequency at sensor points

Plots entries of the uplifted diagonal `(N, n_freq)` array as functions
of frequency. Entries are picked either by direct index into the
sideset-face dimension or by physical `(x, y, z)` coordinate; the latter
resolves to the nearest sideset face centroid.

Four **plot kinds** are available, selected by `plot.kind` (a string or a
list of strings):

| `plot.kind` | What it shows | Needs validation? |
| --- | --- | --- |
| `"lines"` | Per-location autopower `S_ii(f)` vs frequency. With a validation set, the inverse solution is drawn **solid** and the validation data **dashed**, sharing one colour per location. | No (overlay added if present) |
| `"box"` | At each frequency, the distribution of `S_ii(f)` **across the selected locations** as side-by-side boxes (solution vs validation). Box = 25–75th percentile, whiskers = 5th/95th. Above `BAND_FREQ_THRESHOLD` (40) frequencies it auto-switches to median + shaded percentile bands. | Yes |
| `"error"` | Per-location relative-L2 error of the solution autopower spectrum against the validation spectrum, `‖Sᵢᵢˢᵒˡ − Sᵢᵢᵛᵃˡ‖₂ / ‖Sᵢᵢᵛᵃˡ‖₂`, as a bar chart sorted worst → best. Use it to rank which sensors the inversion reproduces best/worst. | Yes |
| `"validation_db"` | Stacked two-panel **decibel** comparison. *Top:* level `L = 10·log10(Sᵢᵢ / db_ref)` in dB, **computed** (solid) vs **measured** (dashed); colour = sensor, with the per-sensor legend shown only for ≤ `COMBINED_LEGEND_MAX` (10) sensors. *Bottom:* signed level error `ΔL = 10·log10(S_meas / S_comp)` per location, with a highlight box giving `max\|ΔL\|` and `median\|ΔL\|` (the dB reference cancels in `ΔL`, so it needs none). Emits a combined "all sensors" figure **and** per-sensor figures. | Yes |

Passing a list (e.g. `["lines", "box", "error", "validation_db"]`) produces
all of them in one run; see the output-naming note below.

The `validation_db` kind uses the **measured/computed** vocabulary
throughout (measured = validation, computed = inverse solution), matching
the sign of `ΔL = S_meas − S_comp` in dB: `ΔL > 0` means the inversion
under-predicts that bin.

### Comparing against validation data

The **inverse solution** diagonal is `diag(Φ Sᵣ Φᴴ)` (Step 3). The
**validation data** is supplied separately as a full CPSD at the
locations you select — shape `(n_loc, n_loc, n_freq_full)`, complex — via
`input.validation_path` (and `input.validation_var` for `.mat`). The
driver extracts its real diagonal, then aligns it to the solution:

- **Locations.** `n_loc` must equal the number of `selection.coordinates`,
  and **validation row `k` is taken to align with the `k`-th coordinate**
  (which resolves to a sideset face). For this reason a validation set
  *requires* `selection.coordinates` (not `indices`/`"all"`), and the
  coordinate matching is **not** deduplicated: two coordinates resolving
  to the same face raise an error, as does a match farther than
  `selection.match_tolerance` (when set).
- **Frequencies.** The validation third axis must span the **full**
  frequency set used in the inversion, in order; it is sliced with the
  sidecar's `freq_indices` to match the reconstructed subset. So
  `n_freq_full` must be ≥ `max(freq_indices) + 1`.

The `box`, `error`, and `validation_db` kinds require a validation set;
`lines` works with or without one (without it, the original solution-only
line plot).

**Driver:** [`src/run_plot_cpsd_diagonal.py`](../src/run_plot_cpsd_diagonal.py)
**Config:** [`config_plot_cpsd_diagonal.json`](../config_plot_cpsd_diagonal.json)

```bash
python src/run_plot_cpsd_diagonal.py config_plot_cpsd_diagonal.json
```

### Config keys

```json
{
  "input": {
    "diagonal_npy_path": "results_cpsd_inverse/full_cpsd_diag.npy",
    "sidecar_json_path": "results_cpsd_inverse/full_cpsd_diag.json",
    "exodus_file":       "data/cube.e",
    "sideset_id":        6,
    "validation_path":   "data/validation_cpsd.mat",
    "validation_var":    "G_val"
  },
  "selection": {
    "indices":         [0, 100, 500],
    "coordinates":     [[0.3, 0.5, 0.2], [-0.3, 0.5, 0.0]],
    "match_tolerance": null
  },
  "plot": {
    "kind":      ["lines", "box", "error", "validation_db"],
    "log_scale": true,
    "title":     "CPSD diagonal vs frequency",
    "ylabel":    "S_ii",
    "xlabel":    null,
    "figsize":   [9, 5],
    "ylim":      null,
    "xlim":      null,
    "db_ref":    1.0,
    "db_floor":  1e-12
  },
  "output": {
    "figure_path":        "results_cpsd_inverse/diagonal_vs_frequency.png",
    "figure_format":      "png",
    "dpi":                150,
    "save_selection_csv": false,
    "top_n":              null,
    "per_sensor":         true
  }
}
```

| Key | Notes |
|---|---|
| `input.diagonal_npy_path` | Real `(N, n_freq)` `.npy` from Step 3 (the inverse solution). |
| `input.sidecar_json_path` | Optional; defaults to the `.json` sibling. Frequencies in the sidecar become the x-axis (else the index is used). Its `freq_indices` also slice the validation set. |
| `input.exodus_file` / `input.sideset_id` | **Required if `selection.coordinates` is provided** (i.e. always, when a validation set is used) — used to compute centroids. |
| `input.validation_path` | Optional `.npy`/`.mat` holding the validation full CPSD `(n_loc, n_loc, n_freq_full)`, complex. Enables the solution-vs-data overlay and is **required** for `box`/`error`/`validation_db`. |
| `input.validation_var` | Variable name inside the `.mat`; **required** for `.mat`, ignored for `.npy`. |
| `selection.indices` | List of non-negative ints, or `"all"` to plot every entry. Allowed only when no validation set is given. |
| `selection.coordinates` | List of `[x, y, z]` triples; each maps to the nearest sideset face centroid. **Required when a validation set is given** (row-by-row alignment). |
| `selection.match_tolerance` | Optional positive number; if set, a coordinate whose nearest centroid is farther than this raises an error (guards against misregistered validation coordinates). `null` ⇒ informational distance print only. |
| `plot.kind` | One of `"lines"`, `"box"`, `"error"`, `"validation_db"`, or a list of them. Default `"lines"`. |
| `plot.log_scale` | `true` ⇒ log y-axis for `lines` and `box`. The `error` bar chart is always linear; `validation_db` is always in dB and ignores this flag. |
| `plot.title` / `ylabel` / `xlabel` | Standard labels; `xlabel: null` auto-fills from the sidecar. For `validation_db` the top-panel y-label becomes `"{ylabel} level [dB re {db_ref}]"`. |
| `plot.figsize` | `[width, height]` in inches. For `validation_db` (two stacked panels), a taller figure such as `[10, 7]` reads better. |
| `plot.ylim` / `xlim` | Optional `[min, max]` pair (must have `min < max`); use this to zoom into a frequency band. For `validation_db`, `ylim` applies to the top (dB level) panel. |
| `plot.db_ref` | `validation_db` only. Positive reference in `L = 10·log10(Sᵢᵢ / db_ref)`. Default `1.0` (dB re 1 unit²/Hz); set to `(20e-6)**2` if the data is genuinely Pa²/Hz. The reference **cancels** in the `ΔL` error panel. |
| `plot.db_floor` | `validation_db` only. Positive relative floor: each location is clamped to `db_floor × (that location's peak)` before the log, so a vanishing/negative `Sᵢᵢ` yields a bounded level instead of `−inf`. Default `1e-12` (≈ −120 dB below peak). Clamped-sample counts are logged. |
| `output.figure_path` | Parent directories created automatically; `figure_format` is added if no suffix is present. When multiple kinds run, a `_lines`/`_box`/`_error`/`_validation_db` suffix is inserted into the stem (single-kind runs keep the bare path). For `validation_db` this path is the combined "all sensors" figure. |
| `output.save_selection_csv` | When `true`, writes a sibling `.csv` per kind: `lines` ⇒ one column per selected index (solution, and validation if present); `box` ⇒ per-frequency percentiles (5/25/50/75/95) for both series; `error` ⇒ ranked per-location errors; `validation_db` ⇒ per-frequency `Lcomp`/`Lmeas`/`ΔL` columns plus a `*_error_stats.csv` with per-sensor and pooled `max\|ΔL\|` / `median\|ΔL\|`. |
| `output.top_n` | Optional positive int. For `error`, show only the worst `N` locations. For `validation_db`, cap the number of **per-sensor** figures written (worst-error first); the skipped count is logged. `null` ⇒ all. |
| `output.per_sensor` | `validation_db` only. When `true` (default), also write one two-panel figure per selected sensor to a `per_sensor/` subdirectory beside `figure_path`, named `sensor_<faceidx>.<fmt>`. Set `false` to emit only the combined figure. |

### Coordinate → row resolution

For each `(x, y, z)` target:

1. Load all sideset face centroids: see
   [`exodus_side_interpolator.py:151-181`](../src/exodus_side_interpolator.py#L151-L181).
   Exodus does **not** store centroids — they are computed on the fly
   from node coordinates plus the (element, local_side) → node table.
2. Compute Euclidean distance to every centroid.
3. Pick `argmin`; print the matched centroid and distance to stdout.

The legend label for that trace is
`"node K (target=[x,y,z], d=...)"` so you can see whether the snap was
clean.

A `n_faces == N` consistency check is enforced — if your diagonal `.npy`
and the named sideset disagree in size, the driver bails out before
plotting.

### When you only want indices, you don't need Exodus

Skip `input.exodus_file` and `input.sideset_id` entirely; pass a list
under `selection.indices` (or the string `"all"`). The x-axis still uses
sidecar frequencies if present.

---

<a name="worked-example"></a>
## 9. Worked example on `tests/data/cube.e`

The unit-cube fixture in [`tests/data/cube.e`](../tests/data/cube.e) is
a `[−0.5, 0.5]³` mesh of 64 HEX8 elements with six sidesets, one per
face, 16 faces each. Sideset 6 is the `Y = +0.5` face. The shipping
configs (`config_*.json` at the repo root) are already wired to this
file and to sideset 6.

This recipe runs the full **post-processing** half of the pipeline
(Steps 3 → 5) on tiny synthetic artifacts. We skip Steps 1–2 because
exercising them realistically requires upstream POD interpolation + an
experimental CPSD; the bootstrap below produces matching `Φ`, `S_r`, and
sidecar files directly, so Steps 3–5 are pure plumbing.

> **Note.** The bootstrap data is for verifying file plumbing only; the
> numbers are not physically meaningful. Replace with real artifacts
> from Steps 1–2 once those are wired up.

### 9.1 Bootstrap: fabricate Φ, per-frequency `S_r`, and a sidecar

Run this once from the repo root (uses the `base` conda env which has
`exodusii`):

```bash
/Users/wilkinsaquino/miniforge/bin/python - <<'PY'
import json, os
import numpy as np

# Geometry on cube sideset 6 (Y=+0.5) has 16 face centroids.
N, n_pod, n_freq = 16, 4, 5
rng = np.random.default_rng(0)

os.makedirs("results", exist_ok=True)
os.makedirs("results_cpsd_inverse", exist_ok=True)

# A trivial complex POD basis with n_pod columns.
phi = (rng.standard_normal((N, n_pod))
       + 1j * rng.standard_normal((N, n_pod)))
np.save("results/sideset_pod_modes.npy", phi)

# Fake per-frequency reduced CPSDs (PSD by construction).
frequencies = np.linspace(500.0, 2500.0, n_freq).tolist()
for k in range(n_freq):
    A = (rng.standard_normal((n_pod, n_pod))
         + 1j * rng.standard_normal((n_pod, n_pod)))
    S_r = A @ A.conj().T
    np.savez(
        f"results_cpsd_inverse/cpsd_inverse_freq{k}.npz",
        S_r=S_r,
        alpha=1e-6,
        residual_rel=0.0,
        frequency=float(frequencies[k]),
    )
print("bootstrap: Φ and 5 per-frequency S_r written")
PY
```

### 9.2 Step 3 — reconstruct the diagonal

```bash
/Users/wilkinsaquino/miniforge/bin/python src/run_reconstruct_full_cpsd.py \
    config_reconstruct_full_cpsd.json
```

Expected: `results_cpsd_inverse/full_cpsd_diag.npy` of shape `(16, 5)`
and a `full_cpsd_diag.json` sidecar that includes the five physical
frequencies.

### 9.3 Step 4 — write diagonal to a fresh Exodus file

The shipping config writes to `data/cube_diag.e`, seeding from
`data/cube_backup.e` and stripping pre-existing sideset variables:

```bash
/Users/wilkinsaquino/miniforge/bin/python src/run_diagonal_to_exodus.py \
    config_diagonal_to_exodus.json
```

Expected: `data/cube_diag.e` exists, has a new `cpsd_diag` sideset
variable on sideset 6 with 5 time steps (one per frequency), and the
Exodus *time* values equal the physical frequencies. Open in
ParaView/Cubit; colour by `cpsd_diag` and scrub the time slider to step
through frequency.

### 9.4 Step 5 — plot CPSD diagonal at chosen points

The shipping config plots indices `[0, 100, 500]`. Those are out of
range for the 16-face cube example — edit them to `[0, 7, 15]` (or
switch to `"all"`), or use coordinates. For coordinates on sideset 6
(face `Y=+0.5`), valid centroid positions are at
`(x, 0.5, z)` with `x, z ∈ {±0.375, ±0.125}`. Example:

```json
"selection": {
  "indices": [0, 7, 15],
  "coordinates": [[0.3, 0.5, 0.2], [-0.3, 0.5, 0.0]]
}
```

Then:

```bash
/Users/wilkinsaquino/miniforge/bin/python src/run_plot_cpsd_diagonal.py \
    config_plot_cpsd_diagonal.json
```

Expected: `results_cpsd_inverse/diagonal_vs_frequency.png` with five
traces (three indices + two coordinates), each labelled with the
resolved sideset-face index. The stdout will show, for each coordinate,
the snapped centroid and Euclidean distance — `d ≈ 0.075` for the first
target, `d ≈ 0.225` for the second.

To exercise the validation overlay and the `box`/`error`/`validation_db`
kinds, drop the `indices`, keep `coordinates` only (so `n_loc` matches the
coordinate count), point `input.validation_path` at a
`(n_loc, n_loc, n_freq_full)` array, and set
`"kind": ["lines", "box", "error", "validation_db"]`. The run then writes
`diagonal_vs_frequency_lines.png`, `_box.png`, `_error.png`, and
`_validation_db.png`, the latter accompanied by per-sensor figures under
`results_cpsd_inverse/per_sensor/sensor_<faceidx>.png`.

### 9.5 Tear-down

```bash
rm -rf results/sideset_pod_modes.npy results_cpsd_inverse/ data/cube_diag.e
```

---

<a name="file-formats"></a>
## 10. File-format reference

| Artifact | Shape & dtype | Producer | Consumer |
|---|---|---|---|
| `sideset_pod_modes.npy` | `(n_faces, n_modes)` complex | Step 1 | Steps 2, 3 |
| `T_r` (`.npy`/`.mat`) | `(m, n_pod, n_freq)` complex | upstream | Step 2 |
| `Ĝ` (`.mat`) | `(m, m, n_freq)` complex | upstream | Step 2 |
| `row_indices.mat` | 1-D integer | upstream | Step 2 (optional) |
| `cpsd_inverse_freqK.npz` (scalar α) | `S_r (n_pod, n_pod)`, `alpha`, `residual_rel`, `frequency` | Step 2 | Step 3 |
| `cpsd_inverse_freqK.npz` (sweep α) | `S_r (n_pod, n_pod, n_alpha)`, `alphas`, `residuals_rel`, `frequency` | Step 2 | Step 3 |
| `summary.json` | metadata incl. row-subset and (when CV ran) `cv` block + `alphas_per_freq` | Step 2 | engineer (audit trail) |
| `cv_results.npz` (CV only) | `alphas`, `scores (n_freq, n_alpha)`, `alpha_star`, optional `fold_scores (n_freq, n_alpha, K)` | Step 2 (CV) | engineer (audit) |
| `cv_score_vs_alpha.png` (CV only) | log-log curves of CV score vs α | Step 2 (CV) | engineer |
| `cv_score_heatmap.png` (CV only) | `(α, frequency)` pcolormesh of CV score | Step 2 (CV) | engineer |
| `full_cpsd.npy` (mode=full) | `(N, N, n_freq_selected)` complex | Step 3 | downstream analysis |
| `full_cpsd_diag.npy` (mode=diagonal) | `(N, n_freq_selected)` real | Step 3 | Steps 4, 5 |
| `*_diag.json` sidecar | frequencies, mode, dtype, paths | Step 3 | Steps 4, 5 |
| Exodus with `cpsd_diag` var | sideset variable, one step per freq | Step 4 | ParaView/Cubit |
| validation CPSD (`.npy`/`.mat`) | `(n_loc, n_loc, n_freq_full)` complex | upstream | Step 5 (optional) |
| `diagonal_vs_frequency[_lines/_box/_error/_validation_db].png` (+ optional `.csv`, + `per_sensor/sensor_<faceidx>.png` for `validation_db`) | plot | Step 5 | engineer |

---

<a name="pitfalls"></a>
## 11. Common pitfalls

- **Frequency alignment is by index.** `T_r`, `Ĝ`, and the optional
  `physics.frequencies` list must agree on `n_freq`. Mismatch is caught
  at startup; mis-ordered frequencies are *not* — Step 2 trusts the
  index order it is given.
- **Units of `T_r` vs `Ĝ`.** If the inversion residual is uniformly off
  by orders of magnitude, suspect a unit mismatch. Reconcile with
  `input.transfer_matrix_scale` instead of pre-scaling the `.npy`/`.mat`
  on disk — the scale gets recorded in `summary.json` for traceability.
- **Sideset row ordering.** The whole post-processing half assumes:
  rows of `sideset_pod_modes.npy` ↔ rows of `S*` ↔ faces of the sideset
  in Exodus order. That ordering is established by Step 1 and must not
  drift between Steps 3, 4, and 5. The `n_faces == N` consistency check
  in Steps 4 and 5 catches obvious mistakes (e.g. wrong sideset ID); it
  cannot catch a *re-ordered* sideset of the same size.
- **One-sided vs two-sided spectra.** The solver does no spectral
  rescaling. If `T_r` assumes a two-sided FFT convention but `Ĝ` is
  one-sided, the recovered `S_r` will be off by a factor of two across
  all frequencies. Confirm both inputs use the same convention.
- **`num_sset_var` lock-up on Step 4.** If you point Step 4 at an Exodus
  file that already has sideset variables (e.g. `cube.e` after Step 1
  has filled it with POD modes), writing a new variable in place will
  fail. Set `output.exodus_file` + `output.copy_from_exodus_file` and
  leave `strip_sideset_vars: true` (the default) — the driver seeds a
  clean copy with `num_sset_var` dropped, preserving the mesh.
- **Coordinate plotting on the wrong sideset.** `run_plot_cpsd_diagonal`
  will happily snap your `(x, y, z)` to the nearest face on whatever
  sideset you point it at. If the sideset is on the `X = +0.5` face and
  your target is at `(0, 0, 0)`, you will get a centroid match — at
  large Euclidean distance. Always check the stdout `(distance=…)`
  printed for each target. With a validation set, set
  `selection.match_tolerance` to turn a far snap into a hard error
  instead of a silent mismatch.
- **Validation alignment is by coordinate order.** Row `k` of the
  validation CPSD is assumed to be the `k`-th `selection.coordinates`
  entry — there is no coordinate lookup inside the validation file. Keep
  the two lists in the same order, and keep the coordinates distinct
  (two coordinates snapping to the same face is a hard error in
  validation mode, since the alignment would be ambiguous). The
  validation frequency axis must span the **full** inversion frequency
  set (it is sliced by the sidecar's `freq_indices`), not just the
  reconstructed subset.
- **Row-index subset with 0-based indices from Python.** Don't forget
  to flip `input.row_indices_one_based` to `false`, or the loader will
  silently shift everything by 1 and (usually) trigger an
  out-of-range error.
- **CV folds split the downselect, not the full sensor set.** If you
  enable both `input.row_indices_path` and `cv.enabled`, the K folds are
  partitions of the downselected sensors — not the original `m`. Check
  that `k_folds` is reasonable relative to `|I|` (`|I|/K` sensors per
  held-out fold) before trusting the CV score.
- **CV α grid coverage.** If CV picks the smallest or largest α in the
  grid, your grid was likely too narrow. Widen it (e.g., extend by two
  orders of magnitude on the picked side) and re-run. The test suite
  guards against α* landing on a boundary on synthetic noise; on real
  data a boundary win is a config bug, not a noise floor.
- **An absolute α across a band spanning resonances.** `σ_max(T_r(f))` can
  vary by orders of magnitude over a modal band, so a single
  `alpha_scaling: "absolute"` value applies wildly different damping at
  different frequencies — heavy damping where the response is small, almost
  none at the peaks. Set `alpha_scaling: "relative"` for any multi-frequency
  run. This is the most common cause of an inversion that looks fine in the
  midband and blows up near resonances.
- **Error bands are wider than the resonances themselves.** The CPSD forward
  map has condition number `cond(T_r)²`, and `cond(T_r)` stays elevated for
  several half-power bandwidths on each flank of a mode. Excising only the
  resonant frequencies is not enough; expect a band roughly `±4` half-power
  bandwidths (`±8ζf`) wide to be affected.
- **`cv.norm_weight: 0` reproduces the old pure-prediction score, which
  under-regularizes.** It is kept only for reproducing earlier runs. Leave it
  at the `1e-2` default unless you are deliberately reproducing a pre-change
  result. See [The score](#step-2-cv-score).
- **Switching `filter_form` silently reinterprets an absolute α grid.** α has
  units of σ for `"lavrentiev"` and σ² for `"tikhonov"`. If you switch filters
  while using `alpha_scaling: "absolute"`, re-choose the grid. Under
  `"relative"` the grid carries over unchanged.
- **Do not follow Remark 4's thin-QR route for `"lavrentiev"`.** QR of
  `[T_r; √α I]` computes the *Tikhonov* solution, which is not eq. 21. It
  corresponds to `filter_form: "tikhonov"`. See [Math recap](#math-recap).
