# xi_gamma_combined handoff

## Goal

Produce a combined figure (`xi_gamma_combined.pdf`) with:

- **Top row (3 panels)**: Radial profiles of |ξ|(r) and |Γ|(r) for 6 epochs (colored by Δt from
  first sink formation), plus a phase-space scatter of |ξ| vs |Γ| with the stability boundary
  ξ^2.5/(850Γ) = 1. X-axis labels on top to reduce whitespace.
- **Bottom panel (full-width heatmap)**: log₁₀(ξ²·⁵/(850Γ)) as a function of time Δt [kyr] (x)
  and radius r [AU] (y), using all 619 snapshots from `mass_evolution.npz`.

The x-axis should start at **negative Δt** (pre-sink snapshots are present in the data).

### Definitions (Kratter & Lodato 2016, *Gravitational Instabilities in Circumstellar Disks*,
### Annu. Rev. Astron. Astrophys. 54, §3.6.2, after Kratter et al. 2010a)

- ξ = Ṁ_in / (c_s³/G)  — infall rate relative to the isothermal-sphere collapse rate
- Γ = Ṁ_in / (M_tot × Ω_K)  — ratio of orbital timescale to disk mass-doubling timescale;
  M_tot = M_enc_gas(r) + M_star_inside(r)
- Ṁ_in(r) = total mass infall rate through aperture at radius r (dM_enc/dt)
- Ω_K(r) = √(G M_tot(r) / r³)
- **Stability boundary: Γ = ξ^2.5/850  ⇔  ξ^2.5/(850Γ) = 1; above = unstable**

> **WARNING (fixed)**: The original handoff/code had ξ and Γ **swapped** relative to the paper
> (it defined "ξ = Ṁ/(M_tot Ω_K)" and "Γ = Ṁ/(c_s³/G)"). Because the ratio ξ^2.5/(850Γ) raises the
> *wrong* quantity to the 2.5 power, the heatmap did not show the paper's criterion. Fixed: ξ now
> takes c_s³/G and Γ takes M_tot Ω_K in every xi-gamma function.

---

## Data source

`frames18/mass_evolution.npz` — keys:

| Key | Shape | Contents |
|-----|-------|----------|
| `times_Myr` | (619,) | Absolute time of each snapshot |
| `t1_Myr` | scalar | Time of first sink formation |
| `r_AU` | (50,) | Radial bin centres [AU], log-spaced ~0.1–9000 AU |
| `r_edges_AU` | (51,) | Bin edges |
| `M_shell` | (619, 50) | Mass in each shell [Msun] — **includes sink mass in innermost bin** |
| `M_enc` | (619, 50) | Enclosed mass (gas + sinks) [Msun]. **Stored as the smoothed, monotonic**
|          |          | version so `cumsum(M_shell) == M_enc` exactly (see fix below) |
| `M_star` | (619,) | Total stellar (sink) mass [Msun] |
| `n_sinks` | (619,) | Number of sinks |

**Finding (verified on data)**: `M_enc[:, 0] == M_star` (sinks stored in the innermost bin), so
`M_enc` already contains the sink mass. Downstream code must **not** add `M_star` on top again.

`sink_data.npz` (in `paper_plots/`) — has per-sink formation/merger times and radii.

---

## Plotting code location

`notebooks/paper_figures.py`, functions:
- `plot_xi_gamma_combined()` (main combined figure)
- `plot_xi_gamma_ratio_heatmap()`, `plot_xi_gamma_ratio_timeseries()`
- `plot_xi_gamma_phase()`, `plot_xi_gamma_aperture()`, `_compute_xi_gamma_aperture()`
- `plot_disk_stability_criteria()` (the ξ/Γ radial-profile pair)

Called from `generate_paper_plots.py` via `make_all_figures`. Data producer:
`disk_analysis/compute_mass_evolution.py` → `frames18/mass_evolution.npz`.

---

## What was actually wrong (root cause, diagnosed from data)

### 1. Pre-formation center bug (the Δt = 0 discontinuity)

**The npz is NOT missing the coarse FIRE gas** — that earlier hypothesis was wrong. The
jeans-refinement gas near the proto-disk exists in every pre-formation snapshot (8–25 M☉ within
1000 AU from snap 28 onward). The real bug: `compute_mass_evolution.py` used the **median gas
coordinate** as the center before any sink exists, and that median sits ~5,000–28,000 AU from the
density peak (the proto-disk). With such an offset center, all disk gas falls outside the small
radial bins, so `M_enc`(≲1000 AU) ≈ 0 pre-formation and jumps to ~12 M☉ the moment a sink
pins the center. That jump fed Ω_K ∝ √(M_tot) and ξ ∝ 1/M_tot, producing the sharp Δt = 0
color boundary.

**Fix**: center = most massive sink when present, else the densest gas particle (same logic as
`notebooks.make_disk_movie_frames.find_center`). Verified: `M_enc`(562 AU) now runs smoothly
8.2 → 12.0 → 12.3 → 12.6 → 17.7 through formation (no discontinuity).

### 2. ξ/Γ definitions swapped relative to Kratter & Lodato §3.6.2

Code computed `"Γ" = Ṁ/(c_s³/G)` and `"ξ" = Ṁ/(M_tot Ω_K)` — the **opposite** of the paper.
Consequently `ξ^2.5/(850Γ)` raised the *wrong* quantity to the 2.5 power. The paper's ξ^2.5/(850Γ)
scales as (M_tot Ω_K)^1 · c_s^−7.5; the swapped version scaled as (M_tot Ω_K)^−2.5, which diverges
at small radii before formation and created an artificial "gradient at low Δt".

**Fix**: swap the definitions in all six functions (ξ = Ṁ/(c_s³/G), Γ = Ṁ/(M_tot Ω_K)) and correct
all axis/colorbar/legend labels. Verified: log₁₀(ξ^2.5/(850Γ)) is now smooth and ~+1…+5 dex at all
radii through formation, with no small-radius blow-up.

### 3. Sink-mass double counting in downstream M_tot

Several consumers computed `M_tot = Σ M_shell + M_star_t`. Since `M_shell[:,0] ≈ M_star` (sinks in
the innermost bin), this **added the stars twice**, inflating M_tot (and deflating ξ) at late times
when M_star ≳ M_gas. Fixed by using `M_enc` directly for M_tot (no `+ M_star`).

### 4. Forward-fill hack (masking the symptom)

`plot_xi_gamma_combined` forward-filled `M_tot` pre-formation from the first sink snapshot. This
was a presentation hack (rejected by the user) and is now removed — the corrected npz needs no
forward-fill.

### 5. Inconsistent M_enc / M_shell

The npz stored raw `M_enc` but a *smoothed* `M_shell` (= diff of smoothed M_enc), so
`cumsum(M_shell) != M_enc`. Now `M_enc` is stored as the smoothed monotonic array, so
`cumsum(M_shell) == M_enc` exactly, and M_tot (from M_enc) is consistent with Mdot (from M_shell).

---

## What M_tot and Mdot actually are (verified)

- `M_tot(r,t) = M_enc(r,t)` = enclosed gas + sinks at radius r (correct center). Verified the cutout
  is NOT missing gas: M_enc(<10⁴ AU) matches the full simulation (162 M☉ in both).
- `Mdot(r,t) = d M_enc(r,t)/dt` = net mass infall rate through radius r. At 562 AU the steady rate
  is ≈ 0.0085 M☉/yr, but the **instantaneous (adjacent-snapshot) rate spikes to 0.2–0.4 M☉/yr
  during the fragmentation bursts** (7 sinks by 1.2 kyr, 21 by 3.8 kyr). The heatmap previously used
  a ~1-kyr smoothing window that suppressed the bursts to ≤0.044 M☉/yr — changed to the
  instantaneous rate (consistent with the time-series/phase plots).

---

## Regime of validity — the key resolution

The paradox (plot says "stable" yet the disk forms 32 sinks) is resolved by KL16 §3.6.2:

> *"For low mass disks, where GI-induced transport is local, Rice et al. (2005) showed... a critical
> cooling rate for fragmentation translates... into an upper limit of α≈0.1. The boundary for
> fragmentation in ξ−Γ space is consistent with this value of α for equivalently low mass disks. If
> the disk mass grows above 0.1−0.2 M*, however, transport becomes inherently global..."*

So **Γ < ξ^2.5/850 is calibrated for M_d/M_* ≲ 0.1–0.2** (local transport). This disk is
disk-dominated during the fragmentation phase:

| Δt [kyr] | M_star | M_disk(<2000 AU) | M_d/M_* | sinks |
|----------|--------|------------------|---------|-------|
| +1.0     | 0.9    | 62.6             | **52**  | 1     |
| +3.5     | 17.7   | 80.0             | **3.8** | 17    |
| +8.0     | 101    | 45.7             | 0.4     | 32    |
| +15      | 175    | ~17              | 0.1     | 32    |

The disk is far outside the criterion's regime exactly when it fragments (global m=1 transport);
only at t≳10 kyr does it enter the regime, and there the criterion correctly gives unstable
(log₁₀ ratio ≈ +0.9). **The early "stable" verdict is an artifact of applying the criterion outside
its regime.**

**Plots now mark the region where M_d/M_* > 0.2** (hatched overlay + legend entry) where the ξ-Γ
criterion is not applicable.

### Stability master plot (for the massive-disk regime)

`plot_stability_master()` in `paper_figures.py` → `stability_criteria_master.pdf`.
Top row: radial profiles per epoch (plasma by Δt). Bottom rows: two **full-width,
horizontally-stretched** (Δt, r) heatmaps (Toomre-Q style).

- **A. Infall ratio Ṁ_in/Ṁ_max,GI** — Ṁ_max,GI = 3 c_s³ α_sat/(GQ) (Kratter et al. 2010a Eq. 28);
  fragmentation when > 1. α_sat = `ALPHA_SAT` (default 0.3). This is the mass-ratio-independent
  form of the ξ-Γ boundary (ξ-Γ's 850 constant assumes α≈0.1 for low-mass disks).
- **B. Disk mass ratio M_d/M_*** — regime; ξ-Γ valid only for ≲ `_MD_MS_REGIME` (0.2).

**Dropped: Gammie β = Ω_K t_cool.** The low-density H2 cooling limit (Galli & Palla 1998) is
invalid at the disk densities (n~1e9 cm⁻³, H2 in LTE): it gave t_cool ≈ minutes, β ≈ 1e-9
(spurious "always fragments"). Getting the LTE rate right (optical depth, ortho/para) is not worth
the effort for this diagnostic; Toomre Q + the infall ratio are the operative criteria here.

Data: `mass_evolution.npz` (M_enc, Mdot) + `frames18/qprofiles/` (Q, c_s per snapshot).
Style: `fig_stability` entry in `plot_style.py` — restyle by editing that one table.

---

## Current state of the code (after fix)

- `compute_mass_evolution.py`: density-peak center pre-formation; stores smoothed monotonic M_enc.
- `paper_figures.py`: all xi-gamma functions use ξ = Ṁ/(c_s³/G), Γ = Ṁ/(M_tot Ω_K); ratio
  ξ^2.5/(850Γ); M_tot from M_enc with no `+M_star`; no forward-fill; heatmap uses instantaneous Mdot.
- `paper_figures.py`: `_disk_mass_ratio()` + `_overlay_regime()` shade the heatmaps where
  M_d/M_* > 0.2 (criterion not applicable); phase plots annotated. Phase-plot legends at bottom.
- The npz was recomputed (619 snapshots) and the full paper pipeline regenerated on the cluster.

## How to run

```bash
# Recompute npz (only if the center/consistency logic changes again):
echo "bash run_compute_mass_evolution_nosub.sh" >> scripts.txt

# Regenerate all paper figures:
echo "bash run_paper_plots_nosub.sh" >> scripts.txt
```

Output: `paper_plots/light/xi_gamma_combined.pdf` (+ `light_png/` PNG and `dark/` PNG).
