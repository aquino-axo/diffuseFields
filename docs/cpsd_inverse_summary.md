# CPSD Inverse Problem — Implementation Summary

## What was built

A reduced-basis regularized solver that recovers the cross-power
spectral density (CPSD) of a structural pressure field in a POD-reduced
coordinate system from sparse experimental sensor CPSDs.

Per the formulation in Aquino & Bonnet, *Active design of diffuse acoustic
fields in enclosures*, JASA 2023 (`jasa23b.pdf`). The code uses the paper's
**first** regularization method (eqs. 19–21); the slide deck
`DiffuseFields_Inversion.pdf` presents the **second** (eq. 24), which is not
implemented. The spectral filter is selectable — see
[Which filter](#which-filter) — and there is a genuine error in eq. (20) worth
reading before comparing against the paper: see
[The error in eq. (20)](#the-error-in-eq-20).

- `Φ ∈ C^(N×n)` — POD basis (full structural space → reduced)
- `T ∈ C^(m×N)` — observation transfer matrix (full structure → m sensors)
- `T_r = T Φ ∈ C^(m×n)` — reduced transfer matrix
- `Ĝ ∈ C^(m×m)` — measured sensor CPSD (one per frequency)
- `S_r ∈ C^(n×n)` — recovered reduced CPSD (the unknown)
- `S* = Φ S_r Φ^h` — lifted full-space CPSD

At each frequency we use the structure of the paper's **first** regularization
method (eqs. 19–21): solve one regularized problem per column `φ_q` of `Ψ` and
sum the outer products. With the reduced SVD `T_r = X Σ Y^h`, `Ψ Ψ^h` the PSD
projection of `Ĝ`, and `Z = X^h Ψ`:

```text
K   = Y diag(g(σ_i, α)) Z
S_r = K K^h
```

`S_r` is PSD by construction for any α ≥ 0 and any `g`, being a sum of outer
products `Σ_q s_q s_q^h`. Two filters `g` are available via
`regularization.filter_form` — see [Which filter](#which-filter).

### Which filter

`regularization.filter_form` selects `g`. Both keep `S_r` PSD, both are
monotone in σ, and both → the minimum-norm solution as α → 0.

| property | `"lavrentiev"` (default) | `"tikhonov"` |
|---|---|---|
| `g(σ, α)` | `1/(σ + α)` | `σ/(σ² + α)` |
| solves | Lavrentiev eq. on `(T_r^hT_r)^{1/2}` | least-squares of eq. (19) |
| matches | eq. (21) / Remark 2 as printed | Remark 4's QR of `[T_r; √α I]` |
| α units | σ | σ² |
| reproduces published results | yes | no |

Use `"lavrentiev"` to reproduce the paper's numerical results (Sec. IV: "obtained as
per (21)"). Use `"tikhonov"` if you want the minimizer of the least-squares
problem eq. (19) actually states.

Measured trade-off: both reach the same *best-case* accuracy (oracle relative
error max 0.796 vs 0.798 on a synthetic modal problem), but `"lavrentiev"` is far
less forgiving of a mis-chosen α. With α one decade too small, the error
inflates by a max factor of 25.8 for `"lavrentiev"` versus 2.94 for `"tikhonov"`,
and the usable α window *narrows* as conditioning degrades for `"lavrentiev"`
(corr(log cond, window) = −0.57) while it widens for `"tikhonov"` (+0.39).
That difference matters in the bands flanking resonances.

**Switching changes what α means** (σ vs σ²), so the grid must be re-chosen.
With `alpha_scaling: "relative"` this is handled for you — the supplied value
is multiplied by `σ_max(f)` raised to the filter's own power, so it stays
dimensionless either way.

### Why eq. (24) (the paper's second method) is not implemented

`H_ij = σ_iσ_j (ZZ^h)_ij / (σ_i²σ_j² + α)`, `S_r = Y H Y^h` — the form shown in
the slide deck `DiffuseFields_Inversion.pdf` — is the exact Tikhonov minimizer
of `½‖T_rS_rT_r^h − Ĝ‖_F² + (α/2)‖S_r‖_F²`, but it is **not** PSD-guaranteed
for α > 0. Over 400 random instances with indefinite noisy `Ĝ` it came out
indefinite in 62 (worst min/max eigenvalue −5.7e-01), which would give negative
PSDs after lifting through `Φ S_r Φ^h`; the `K K^h` forms above failed 0 of 400.
Remark 3 asserts both are positive definite; that does not hold for eq. (24) at
α > 0.

### The error in eq. (20)

Eq. (20) states that `Y (Σ + α I)^{-1} X^h φ_q` is the minimizer of the
least-squares functional in eq. (19), `½‖T_r u − φ_q‖² + α‖u‖²`. It is not —
that minimizer is `Y (Σ² + 2αI)^{-1} Σ X^h φ_q`, filter `σ/(σ² + 2α)`. They
agree only if `σ_i(σ_i+α) = σ_i² + 2α` for every `i`, i.e. `σ_i = 2` for all
`i`. Verified numerically: the Tikhonov filter matches a brute-force solve of
eq. (19) to ~1e-15 across α, while eq. (20) deviates by 1.1e-07 (α=1e-6) up to
9.3e-02 (α=1).

**Remark 4 shows eq. (20) is the typo, not eq. (19).** It offers a thin QR of
`[T_r; √α I]` as an equally valid route for method 1; that factorization solves
`min ‖T_r u − φ‖² + α‖u‖²`, giving `σ/(σ² + α)`. Confirmed: the QR route agrees
with `filter_form="tikhonov"` to ~1e-15 and disagrees with `"lavrentiev"` by 1.8e-05
(α=1e-4) to 1.8e-02 (α=1e-1). So eq. (20) is almost certainly
`(Σ² + αI)^{-1}Σ` mis-cancelled to `(Σ + αI)^{-1}`.

Two reasons it went unnoticed: the convergence argument at lines 220–221 is
algebraically self-consistent with `(Σ+αI)^{-1}` as printed and correctly
proves `‖s_q(α) − s_q‖ → 0`, but *both* filters converge to the pseudo-inverse
solution as α → 0, so it cannot detect the error; and the typo then propagated
into Remark 2.

Eq. (20) is nonetheless a legitimate regularization, which is why `"lavrentiev"`
remains the default. Writing the polar decomposition `T_r = Q P` with
`Q = X Y^h` and `P = (T_r^hT_r)^{1/2}`, it is exactly the solution of

```text
(P + α I) u = Q^h φ_q
```

(matched to ~1e-15 at every α tested), i.e. Lavrentiev regularization of the
square-root operator rather than Tikhonov on `T_r^hT_r`. It also has a
variational characterization — it minimizes
`½ u^h P u − Re(u^h Q^h φ_q) + (α/2)‖u‖²` — just not a least-squares one.

**Factor-of-2 caveat.** Eq. (19) as printed uses `½‖·‖² + α‖u‖²`, whose
minimizer is `σ/(σ² + 2α)`, while Remark 4's QR gives `σ/(σ² + α)` — so
eq. (19) and Remark 4 differ from each other by a factor of 2 in α.
`filter_form="tikhonov"` uses the standard convention
`min ‖T_r u − φ‖² + α‖u‖²`, matching Remark 4. Mapping:
`α_here = 2 · α_eq19`.

## Files

| File | Purpose |
|------|---------|
| `src/cpsd_inverse.py` | `CPSDInverseSolver`, `spectral_filter`, `resolve_alphas`, `lift_to_full_space` |
| `src/cpsd_inverse_cv.py` | `KFoldCVSelector` (k-fold CV over the sensor axis) |
| `src/run_cpsd_inverse.py` | Config-driven driver (load, validate, solve, save, plot) |
| `src/run_reconstruct_full_cpsd.py` | Config-driven driver to lift `S_r → S*` |
| `config_cpsd_inverse.json` | Example config for the inversion driver |
| `config_reconstruct_full_cpsd.json` | Example config for the reconstruction driver |
| `tests/test_cpsd_inverse.py` | Unit tests — see [Tests](#tests) |

## Class API — `CPSDInverseSolver`

```python
solver = CPSDInverseSolver(reduced_transfer_matrix, pod_basis=None)
```

- `reduced_transfer_matrix` — shape `(n_sensors, n_pod, n_freq)`, complex
- `pod_basis` — optional `(N, n_pod)`, complex; required only if you later
  call `reconstruct_full_cpsd`

### Methods

- **`solve_single_freq(freq_idx, G, alphas, psd_tol_rel=0.0,
  alpha_scaling='absolute', filter_form='lavrentiev')`** — solves the inverse
  problem at one frequency for one or more regularization values. Returns `(S_r, residuals_rel)` where
  `S_r` has shape `(n_pod, n_pod, n_alpha)` and `residuals_rel` has shape
  `(n_alpha,)`. The SVD of `T_r` and the PSD projection of `Ĝ` are computed
  once and reused across all `alphas`. `alpha_scaling` selects how `alphas` is
  interpreted, and `filter_form` which spectral filter is applied — see
  [Scaling of α](#scaling-of-α) and [Which filter](#which-filter).
- **`sigma_max(freq_idx)`** — largest singular value of `T_r` at that
  frequency, so callers can report or reproduce the effective α.
- **`reconstruct_full_cpsd(S_r, diagonal_only=False)`** — lifts a reduced
  CPSD back to the full space: `Φ S_r Φ^h`. With `diagonal_only=True`,
  returns only the real-valued `(N,)` diagonal — useful when `N` is large.

### Module helpers

- **`lift_to_full_space(S_r, pod_basis, diagonal_only=False)`** — free
  function used by the reconstruction driver and the class method.
- **`resolve_alphas(alphas, sigma_max, alpha_scaling='absolute',
  filter_form='lavrentiev')`** — maps supplied parameters to the α actually applied.
  Shared by the solver and the CV selector so both resolve a given candidate
  identically.
- **`spectral_filter(sigma, alpha, filter_form='lavrentiev')`** — the diagonal
  filter `g` applied as `K = Y diag(g) X^h Ψ`.
- **`ALPHA_SCALINGS`, `FILTER_FORMS`, `ALPHA_SIGMA_POWER`** — valid values and
  the σ_max power each filter's α scales with.

## Class API — `KFoldCVSelector` (`src/cpsd_inverse_cv.py`)

```python
selector = KFoldCVSelector(solver, G, k_folds=5, seed=0,
                           save_fold_scores=False,
                           norm_weight=1e-2, alpha_scaling='absolute',
                           filter_form='lavrentiev')
alpha_star, scores, fold_scores = selector.select(
    alphas, psd_tol_rel=0.0, alpha_mode='global')
```

Selects α by k-fold cross-validation over the **sensor axis**. `alpha_scaling`
and `filter_form` must match what is later passed to `solve_single_freq` for
the refit; the
lam → α map uses the *full-matrix* `σ_max(f)`, not a fold's, so one candidate
means one absolute α in every fold and in the refit. See
[The CV score and `norm_weight`](#the-cv-score-and-norm_weight).

## Numerical implementation details

### PSD projection of Ĝ

Experimental CPSDs from finite-time averaging or instrument noise can be
indefinite or non-Hermitian. We pre-process at each frequency:

1. `Ĝ ← (Ĝ + Ĝ^h)/2` — Hermitize.
2. Eigendecompose `Ĝ = U Λ U^h`.
3. Clip `λ_i < 0` (or `< tol_rel · max(|λ|)`) to zero.
4. Form `Ψ = U · diag(√λ)`. Then `Ψ Ψ^h` is the PSD projection of `Ĝ`.

`Z = X^h Ψ` is then formed once per frequency.

### Scaling of α

α carries the filter's own units — σ for `"lavrentiev"`, σ² for `"tikhonov"`. Across
a band containing resonances `σ_max(f)` varies by orders of magnitude, so one
absolute α applies very different damping at different frequencies.
`regularization.alpha_scaling` controls the interpretation:

| value | meaning |
|---|---|
| `"absolute"` (default) | α used as given |
| `"relative"` | α = value × `σ_max(T_r(f))^p`, so the value is dimensionless |

with `p = 1` for `"lavrentiev"` and `p = 2` for `"tikhonov"` (`ALPHA_SIGMA_POWER`),
so a `"relative"` grid carries over unchanged if you switch filters.

`"relative"` is recommended for multi-frequency runs. The default stays
`"absolute"` so existing configs keep their meaning — the two differ by orders
of magnitude, so switching requires re-choosing the α grid.

`summary.json` records `alpha_scaling`, `filter_form`, `sigma_max_per_freq`,
and `alphas_effective` (the α actually applied) so a run is reproducible
either way.

### Sweeping α

When the config provides `regularization.alpha_sweep`, the SVD and `Z` are
computed once per frequency; only the diagonal filter `g(σ_i, α)` changes
per α. Per-frequency cost is therefore roughly constant in `n_alpha` after
the SVD.

### Residual diagnostic

For each frequency and each α we save the relative Frobenius residual

```
||T_r S_r T_r^h − Ĝ||_F / ||Ĝ||_F
```

It is also plotted as a single `residual_vs_frequency.png` (one curve per α).

### Hermiticity and definiteness of S_r

`S_r = K K^h` is Hermitian and positive semidefinite by construction to
within floating-point roundoff, so no explicit symmetrization or PSD
projection is applied to it. (`Ĝ` *is* Hermitized and PSD-clipped on input —
see above — but that is a separate step.)

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
  "regularization": {
    "alpha": 1e-6,
    "psd_tol_rel": 0.0,
    "alpha_scaling": "absolute",
    "filter_form": "lavrentiev"
  },
  "cv": {
    "enabled": false,
    "k_folds": 5,
    "alpha_mode": "global",
    "seed": 0,
    "save_fold_scores": false,
    "norm_weight": 1e-2
  },
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
| `alpha_scaling` | `"absolute"` (default) or `"relative"` (α = value × `σ_max(T_r(f))^p`). See [Scaling of α](#scaling-of-α) |
| `filter_form` | `"lavrentiev"` (default) or `"tikhonov"`. See [Which filter](#which-filter) |

**`cv` (optional; requires `regularization.alpha_sweep` as the candidate grid):**

| Field | Default | Notes |
|---|---|---|
| `enabled` | `false` | Turn on k-fold CV selection of α; incompatible with a scalar `alpha` |
| `k_folds` | `5` | Sensor-axis folds; must be ≥2 and ≤ number of sensor rows |
| `alpha_mode` | `"global"` | `"global"` picks one α for the band; `"per_freq"` picks one per frequency |
| `seed` | `0` | Fold-shuffle seed |
| `save_fold_scores` | `false` | Also store the `(n_freq, n_alpha, k_folds)` score array |
| `norm_weight` | `1e-2` | Weight μ on the solution-norm term of the CV score. `0.0` gives the pure held-out prediction score |

### The CV score and `norm_weight`

For each frequency, fold and candidate α the score is

```text
score = ||T_val S_r T_val^h − PSD_clip(Ĝ_val)||_F / ||PSD_clip(Ĝ_val)||_F
      + norm_weight · ||S_r||_F · σ_max(f)² / ||Ĝ(f)||_F
```

averaged over folds. The first term is held-out prediction; the second is the
solution norm, made dimensionless by `σ_max²/‖Ĝ‖_F` (since `‖S_r‖ ~ ‖Ĝ‖/σ_max²`)
so that one `norm_weight` applies across a whole band.

**Why the second term exists.** A pure prediction score is a data-fit
criterion, and at ill-conditioned frequencies it is nearly blind to the
directions the ill-conditioning amplifies — many very different `S_r`
reproduce almost the same sensor CPSD. The CPSD forward map
`S ↦ T_r S T_r^h` has condition number `cond(T_r)²`, and `cond(T_r)` stays
elevated for several half-power bandwidths around each resonance, so the
affected band is much wider than the resonance peaks. Empirically the
prediction-only score under-regularizes by about a decade there. Useful
`norm_weight` values run roughly `1e-3` to `1e-1`; above that the selection
over-regularizes.

**Why it must go on the held-out residual.** `S_r(α)` already minimizes
(training fit) + α·(penalty), so minimizing (training fit) + μ·(penalty) over
the family `{S_r(α)}` returns α = μ identically — a circular criterion that
just echoes the constant supplied. Scoring the fit on sensors the fold did not
see breaks that fixed point. This is also why an L-curve takes the *corner* of
(log‖S_r‖, log‖residual‖) rather than a weighted sum: curvature is invariant
to the weighting.

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
α values used, input file paths, and the frequency list when supplied, plus
`alpha_scaling`, `filter_form`, `sigma_max_per_freq`, and `alphas_effective`
(the α actually applied at each frequency). When CV ran it also records the `cv` block
including `norm_weight`, and writes `cv_results.npz`.

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

Sixteen correctness checks (the set approved during planning, per
[CLAUDE.md](../CLAUDE.md)'s "keep tests to the absolute necessary ones"
guideline):

1. **`test_recovery_synthetic`** — for several `(m, n, n_freq)` shapes with
   `m ≥ n`, pick a random PSD `S_r_true`, form `Ĝ = T_r S_r_true T_r^h`,
   invert with a tiny α, and assert
   `||S_r − S_r_true||_F / ||S_r_true||_F < 1e-5` and residual `< 1e-6`.
2. **`test_scalar_reduction`** — for `n=1`, verifies the method-1 closed form
   reduces to `s = g / (|t| + α)²` for several complex `t`, real `g`, and α
   (rtol `1e-12`).
3. **`test_apply_row_subset`** — `T_r[I,:,:]` and `G[I,I,:]` sliced
   symmetrically for a non-contiguous, unordered index set.
4. **`test_kfold_partition`** — the folds concatenate to a permutation of the
   input indices, each appearing exactly once.
5. **`test_cv_picks_best_alpha_synthetic`** — CV selects a sensible α on
   synthetic data.
6. **`test_refit_matches_direct_solve`** — the driver's refit at α\* equals a
   direct `solve_single_freq` call.
7. **`test_global_mode_single_alpha`** — `alpha_mode='global'` yields one
   scalar α for the band.
8. **`test_alpha_scaling_equivalence`** — `relative(λ)` ≡ `absolute(λ·σ_max)`;
   `resolve_alphas` rejects unknown scalings and non-positive/NaN/inf `σ_max`.
9. **`test_alpha_scaling_backward_compat`** — the `'absolute'`/`'lavrentiev'`
   defaults are bit-identical to passing them explicitly and match an
   independent implementation of eq. (21); `S_r` is PSD; CV defaults hold.
10. **`test_cv_norm_term_formula`** — `score(μ) − score(0)` equals
    `μ·mean_folds(‖S_r‖_F σ_max²/‖Ĝ‖_F)`, and `norm_weight=0` reproduces a
    hand-built pure prediction score.
11. **`test_cv_refit_alpha_consistency`** — CV uses the full-matrix `σ_max`,
    not fold-local, and the refit applies the same α.
12. **`test_relative_scaling_is_scale_invariant`** — under
    `T_r→γT_r`, `Ĝ→γ²Ĝ`, both `S_r` and the CV score are unchanged in
    `'relative'` mode, for both filters; `'absolute'` is *not* invariant.
13. **`test_norm_term_reduces_error_when_ill_conditioned`** — behavioural:
    `norm_weight=1e-2` never lowers α and lowers the true error.
14. **`test_filter_form_alpha_power`** — `'relative'` scales α by `σ_max¹` for
    `"lavrentiev"` and `σ_max²` for `"tikhonov"`, and the wrong power is
    detectably different.
15. **`test_both_filters_keep_psd`** — with indefinite `Ĝ`, both filters keep
    `S_r` Hermitian PSD across a 41-point α sweep. Includes a positive
    control: the eq. (24) form, computed inline, *is* indefinite on the same
    instance (min/max eig −3.0e-02), so the claim is non-trivial.
16. **`test_cv_and_refit_agree_per_filter`** — the selector and the refit
    resolve a candidate to the same α for each filter, where the σ power now
    also enters.

All sixteen pass. Note that the three `exodusii`-dependent test modules in
`tests/` fail to import without that package installed; that is unrelated to
this solver.

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
