# SK-Comment Review

All `\skcomment`, `\skadd`, `\skcut` markers in `oja_template.tex`.  
File: `/home/vasilii/Software/Overleaf-sync-nix-flake/Pop III m12f/oja_template.tex`

Legend: ✅ Addressed by code/data | 📝 User must edit

---

## ✅ Items Addressed (or verified) by Code

---

### L721 — "missed the blue particles in phase diagram"
**Marker:** `\skcomment{I think you missed the blue particles, you should add them on the plot too}`

**Current code status (from `paper_figures.py`):**
- **T vs ρ panel (Row 0):** Blue full-sim particles ARE plotted (lines 1611–1620) when `_FULLSIM_PATH` is set. CONFIRMED WORKING.
- **H₂ fraction vs ρ panel (Row 1):** Blue full-sim particles were **NOT** plotted — now FIXED (see below).

**Data availability (CONFIRMED from cluster, `check_h2_1902281.log`):**

The full simulation (`m12f/output_jeans_refinement/`) saves `MolecularMassFraction` only in **some** snapshots (full output every ~50 snaps, minimal output in between):
- **28 fields** (WITH H₂): snap 0, 50, 100, 200 (and likely every ~50th snap)
- **9 fields** (NO H₂): snap 28, 92–98, 276

The spatial cutouts of the full sim (`m12f/output_cutout/`) have exactly the same field pattern (since `create_cutouts.py` copies all fields from source).

SK's cutout simulation (`m12f_cutout/output_jeans_refinement/`): ALL 619 snapshots have 28 fields including H₂ — every snapshot was saved with full output.

**FIX APPLIED:** Modified `load_fullsim_phase()` to also return `MolecularMassFraction` when available (3-value return: `n, T, fh2`). Added blue full-sim scatter to the H₂ panel in `plot_phase_diagrams()`. The blue H₂ particles will now appear when the full-sim snapshot has the field.

**Caveat:** The first epoch (snap 28, pre-stellar) does NOT have H₂ data in the full sim, so blue particles won't appear on the H₂ panel for that epoch. This is acceptable — the pre-stellar phase diagram doesn't have significant H₂ structure anyway.

**Action for paper:** If the blue particles appear on some H₂ panels but not all, explain: "Full-simulation comparison (blue) is shown where molecular fraction data is available."

---

## 🔧 Items Addressable by Code/Data (pending)

### L320 — Hermite integrator check
**Marker:** `\skcomment{...check the Config.sh file. The Hermite isn't used at all here...}`
**VERIFIED from code:** In `allvars.h` (line ~612): `#if !defined(PMGRID) && !defined(FIRE_SUPERLAGRANGIAN_JEANS_REFINEMENT) → #define HERMITE_INTEGRATION 32`. The compiled Config.sh has `PMGRID=1024` and `FIRE_SUPERLAGRANGIAN_JEANS_REFINEMENT` is also likely active → **Hermite is NOT used** in this simulation. SK is correct.
**Action for paper:** Remove the Hermite integrator sentence. Add: "Direct N-body summation is used for sink–sink gravitational interactions within 1000 AU."

### L334 — DM and gas mass separately ✅
**Marker:** `\skcomment{Specify DM and gas mass separately...}`

**VERIFIED from data** (`check_h2_1902281.log`, snap 0, z=21.2):

| Radius | M_gas (M☉) | M_DM (M☉) | M_tot (M☉) | f_b |
|--------|-----------|-----------|------------|-----|
| < 0.1 kpc | 2.49×10⁶ | 5.70×10⁶ | 8.19×10⁶ | 0.304 |
| < 0.5 kpc | 5.72×10⁶ | 2.95×10⁷ | 3.52×10⁷ | 0.162 |
| < 1.0 kpc | 1.17×10⁷ | 5.82×10⁷ | 6.99×10⁷ | 0.167 |

Note: snap 28 has f_b=1.0 because PartType1 (DM) was stripped from that snapshot (only gas saved). Snap 0 and 10 have DM and are consistent with each other.

The baryon fraction at r < 1 kpc (0.167) matches the cosmic baryon fraction (Ω_b/Ω_0 = 0.0455/0.272 ≈ 0.167), as expected for a virialized halo.

**Action for paper:** Write: "The host minihalo has a total mass of ~7×10⁷ M☉ within 1 physical kpc at z≈21 (M_gas ≈ 1.2×10⁷ M☉, M_DM ≈ 5.8×10⁷ M☉), with a baryon fraction consistent with the cosmic value (f_b ≈ 0.167)." Specify that this is measured within a spherical aperture centered on the densest gas particle (no halo finder used).

### L366 — Δm vs r plot (2-panel resolution figure)
**Marker:** `\skcomment{Maybe it's a good idea to have this be a 2-panel one-column figure where the first plot is just delta m as a function of radius.}`
**CODE CHANGE needed:** Add a top panel to the resolution profile figure showing `Δm = m_cell` (PartType0 masses) vs radius. This would show the refinement profile directly as mass per particle vs distance from center. The existing bottom panel shows the spatial resolution `λ = (m/ρ)^{1/3}` vs r. Implementation: add a second axes in `plot_resolution_profile` (or whatever function generates this figure), plotting `m_i` vs `r_i` for each epoch.

### L468 — Verify x_e > 10^{-4} ✅
**Marker:** `\skcomment{Have you checked this number from the sims?}`

**VERIFIED from data** (`check_h2_1902281.log`). At high gas density (n_H > 10⁸ cm⁻³, i.e., disk gas):

| Dataset | Snap | N_particles | x_e min | x_e median | x_e max | % above 10⁻⁴ |
|---------|------|------------|---------|-----------|---------|--------------|
| Full sim cutout | 100 | 7,025 | 4.5×10⁻¹² | 7.5×10⁻¹⁰ | 2.0×10⁻⁴ | 0.2% |
| Full sim cutout | 200 | 432,170 | 4.0×10⁻¹¹ | 6.6×10⁻⁸ | 9.3×10⁻⁵ | 0.0% |
| Cutout sim | 100 | 265,533 | 1.2×10⁻¹⁰ | 6.5×10⁻⁸ | 9.9×10⁻⁴ | 0.1% |
---

**L420 — "rcut is wrong":** The sound-crossing time argument (L420–426) uses `r_cut ≈ 0.32 pc`. This is NOT the cutout radius — the cutout is ~1 kpc. The 0.32 pc is the radius of the local analysis cutouts created by `disk_analysis/create_cutouts.py` with `--cutout-radius 0.005` comoving kpc/h (used for rendering frames and analysis plots). These are two different spatial selections:

1. **Simulation cutout** (SK, on RUSTY): ~1 kpc — the re-simulation region described in Section 2
2. **Analysis cutout** (local, `run_cutouts_new.sh`): 0.005 comoving kpc/h ≈ 0.32 pc — the smaller region extracted for plotting

**Action:** The sound-crossing argument should use the correct cutout radius (1 kpc, not 0.32 pc). With r_cut = 1 kpc, t_cross = 1 kpc / 10 km/s ≈ 100 Myr >> 15 kyr of evolution — the causal isolation argument becomes even stronger. Fix the 0.32 pc value in line 422 to match the actual simulation cutout radius.

> Note: The analysis cutout at 0.32 pc is a subset used for rendering — it does not define the simulation domain boundary.

| Cutout sim | 200 | 665,371 | 2.0×10⁻¹² | 1.4×10⁻⁷ | 9.9×10⁻⁴ | 0.0% |
| Cutout sim | 400 | 1,451,198 | 2.3×10⁻¹² | 3.4×10⁻⁷ | 1.16 | 3.4% |
| Cutout sim | 600 | 2,098,723 | 1.6×10⁻¹⁰ | 5.3×10⁻⁷ | 1.16 | 2.2% |

**Result: x_e is overwhelmingly << 10⁻⁴ in disk gas.** Median x_e ~ 10⁻⁸ to 10⁻⁷. Only 0–3.4% of high-density particles have x_e > 10⁻⁴ (at late times, likely from protostellar heating).

**The paper's statement that x_e >> 10⁻⁴ (justifying no non-ideal MHD) is INCORRECT.** The actual data shows x_e << 10⁻⁴ in disk gas, which means non-ideal MHD effects (Ohmic, ambipolar, Hall) WOULD be significant if included. However, the simulation doesn't include them — only Braginskii conduction and viscosity.

**Action for paper:** This is a limitation to acknowledge. Rewrite L468 to say: "We note that the electron fraction in the dense disk gas is x_e ~ 10⁻⁸–10⁻⁷, well below the threshold where non-ideal MHD effects (Ohmic dissipation, ambipolar diffusion, Hall effect) become important. These effects are not included in our simulation, which represents a caveat for the magnetic field evolution in the disk."

### L543 — CR ionization rate value
**Marker:** `\skcomment{Write value here...}`
**VERIFIED from source:** `gizmo/eos/cosmic_ray_fluid/cosmic_ray_utilities.c` lines 1877–1920.
Without `COSMIC_RAY_FLUID` (not compiled), `Get_CosmicRayEnergyDensity_cgs()` returns a fixed `1.6e-12 eV/cm³` (Cummings et al. 2016). Then `zeta_cr = 1e-5 × 1.6e-12 = 1.6×10^{-17} s^{-1}` — the standard MW value. In cosmological runs, this is suppressed for gas below ~200× cosmic baryon density (essentially zero for diffuse gas), but fully active for disk gas (n_H >> 1 cm⁻³).
**Action for paper:** Write ζ_CR = 1.6×10^{-17} s^{-1} and note it's the Milky Way value (Cummings et al. 2016), unconstrained at high-z.

### L466/L571 — MHD and CR transport flags
**VERIFIED from `compile_time_info.c`:** The compiled binary has `FIRE_MHD` (which sets `CONDUCTION`, `CONDUCTION_SPITZER`, `VISCOSITY`, `VISCOSITY_BRAGINSKII` via `allvars.h`). No `NON_IDEAL_MHD`, `OHMIC`, `AMBIPOLAR`, `HALL` flags. No `COSMIC_RAY_FLUID`, `COSMIC_RAY_SUBGRID_LEBRON`, or `FIRE_CRS` flags.
**Action:** L466: correct to "Braginskii conduction and viscosity only". L571: remove CR transport sentence entirely.

### L838 — Energy computation equations
**Marker:** `\skcomment{You need to write out how you computed these energies...}`
**Can be extracted from code:** The energy evolution script (`plot_energy_evolution.py`) computes: E_therm = Σ m_i u_i, E_rot = ½ Σ m_i v_φ², E_turb = ½ Σ m_i (δv)², E_grav = -Σ G m_i M_enc(r_i)/r_i, E_mag = Σ |B|² / (8π) × V_i. These formulas should be written in the paper.

---

## 📝 Items User Must Address

### Writing / structure

| Line | Marker | What to do |
|------|--------|------------|
| 260 | `\skcomment` | Fix citation; add AGORA paper (van den Bosch et al. 2014 or Roca-Fàbrega et al. 2021) |
| 268 | `\skcomment` | Add 1 sentence: "In this work, we study [disk formation / fragmentation / IMF / magnetic fields / ...]" |
| 295 | `\skadd` | Accept the added text: "We briefly review these below and refer the reader to [CITE]". Fill in citation. |
| 295 | `\skcut` | Apply cut: change "those" → "the above" |
| 300 | `\skcomment` | **Section restructuring:** SK recommends: remove 2.1.1, describe code briefly above, ICs as 2.1.1 with re-sim details, refinement+resolution as 2.1.2, physics as 2.2 (MHD=2.2.1, gravity=2.2.2, then rest). Also write out the MHD equations. |
| 305 | `\skcomment` | "mesh-generating points" → "partition-generating points" (per GIZMO docs) |
| 309 | `\skcomment` | Remove incorrect Voronoi tessellation sentence. GIZMO MFM does NOT reconstruct a Voronoi tessellation at each timestep — it uses a fixed kernel-based effective face. Rewrite using Hopkins (2015) GIZMO methods paper. |
| 318 | `\skcomment` | Change `\citep{FORGE_d_in_FIRE_Hopkin_2023}` → cite STARFORGE methods paper (Grudic et al. 2021/2022) for the 5th-order Hermite integrator. *(But see L320 — Hermite may not be active.)* |
| 335 | `\skcomment` | Choose one: "Pop III" or "Population III" — use consistently throughout |
| 336 | `\skcomment` | Either quantify: "This minihalo resides in a region with δ ≈ X overdensity (Y Mpc comoving scale)" or cut the sentence |
| 337–339 | `\skcut` | Apply cuts — remove the 3-line passage about large-scale cosmic web |
| 358 | `\skcomment` | Restructure: remove the `Cutout re-simulation details` subsection and fold its content into a single coherent paragraph describing the full simulation sequence (z=99 → z_collapse at low res → re-run with refinement → cutout forward) |
| 380 | `\skcomment` | Use `$\Delta m$` consistently for mass resolution everywhere (find & replace `$m_i$`, `$m_{\rm cell}$`, etc.) |
| 393 | `\skcomment` | *(See ✅ section above — formula correct, add mass floor mention)* |
| 406 | `\skcomment` | Merge "Numerical resolution" subsection with "Cutout re-simulation" subsection |
| 461 | `\skcomment` | Add "and divergence cleaning" after "constrained-gradient method" |
| 466 | `\skcomment` | Correct statement: ONLY Braginskii conduction and viscosity are included — NOT ambipolar diffusion, Hall effect, or Ohmic resistivity. Rewrite sentence. |
| 483 | `\skcomment` | Add citations: Hopkins et al. (2020) [GIZMO radiation module] and Hopkins & Grudic (2019) |
| 494 | `\skcomment` | Add 2 sentences explaining that LW is included in the PE band and that there is an explicit H₂ dissociation term in the rate equation |
| 500 | `\skcomment` | Apply `\rm` to all subscript elements: e.g., `\mathrm{H_2}`, `\mathrm{H}`, etc. throughout thermochemistry section |
| 543 | `\skcomment` | State the value of ξ_H₂ (cosmic-ray ionization rate). Note that at high-z this is unconstrained; MW value used as default. Effect: H₂ formation shifts to lower densities (~10⁷ cm⁻³), negligible at ρ ≫ 10⁷ cm⁻³. |
| 571 | `\skcomment` | Remove sentence about CR streaming/diffusion transport — no CR transport in this simulation. CR are only treated as a pressure term (static). |
| 673 | `\skcomment` | Add 2 introductory sentences at the top of the Results section connecting to the paper narrative |
| 679 | `\skcomment` | Rewrite in active voice ("we show", "we find", "the disk develops") and choose past or present tense consistently throughout Results |

### Physics / text corrections

| Line | Marker | What to do |
|------|--------|------------|
| 730–731 | `\skadd` | Accept: "in the left column" addition |
| 731 | `\skcut` | Apply: change "before particle splitting" → "before the re-simulation" |
| 732 | `\skcomment` | Reword: "...correspond to low-resolution gas in the original cosmological box, which predates the re-simulation." |
| 735 | `\skcut` | Apply: remove "The structure of this part of the simulation does not change significantly with time." |
| 736–737 | `\skadd` | Accept new sentence about domain being outside region of interest |
| 743 | `\skcomment` | Replace "not fully understood" with: "A plausible explanation is trace metal enrichment from numerical diffusion; we do not impose a metallicity floor so diffused metals from nearby regions could contribute enhanced line cooling." |
| 744 | `\skcomment` | Clarify metal origin: metals come from numerical diffusion of metallicity from the nearest star-forming region. This is a known FIRE numerical effect at metal-free ICs. |
| 749 | `\skcomment` | SK offered to help with 1-zone cooling tests. Follow up with SK. |
| 780 | `\skcut` / `\skadd` | Apply: "Notice" → "Note" |
| 803 | `\skcut` / `\skadd` | Apply: "1$^{\text{st}}$" → "first" |
| 804 | `\skcomment` | Apply `\skcut{doesn't}` → `\skadd{does not}` and `\skcut{Here it is important to note}` → `\skadd{We note}` |
| 830 | `\skcomment` | Fix caption: "before 1$^{\text{st}}$" → "before the first" |
| 838 | `\skcomment` | Add equations for each energy component: E_therm = Σ m_i u_i, E_rot = ½ Σ m_i v_φ², E_turb = ½ Σ m_i (δv)², E_pot = -Σ G m_i M_enc(r_i)/r_i |
| 842 | `\skcut` | Apply: remove "deeply" |
| 846 | `\skadd` | Apply: add "potential" → "gravitational potential well" |
| 855 | `\skcut` / `\skadd` | Apply: "Section \ref{...}." → "see section \ref{...}." (period moves outside) |
| 890 | `\skcomment` | Add explanation: "The dip in virial ratio coincides with the plateau in accretion rate visible in Fig. X — as the accretion rate decreases, less gravitational energy is released per unit time, allowing the virial ratio to temporarily increase before resuming." |
| 925 | `\skcomment` | Fix period placement throughout: periods should always come **after** the closing bracket, not inside. Do a global find-and-replace for `.)` → `).` in appropriate contexts. |
| 1850 | `\skcomment` | Remove ZAMS note — it was a copy-paste artifact. Delete the `\skcomment{ZAMS just because I copy pasted from other code. Will remove.}` line and ensure the ZAMS reference in the caption is removed if it doesn't apply. |

---

## Summary

| Category | Count |
|----------|-------|
| ✅ Addressed / verified by code | 8 (L393, L416, L420, L721, L320, L334, L468, L543, L466/L571) |
| 🔧 Code-addressable, pending | 2 (L366 Δm-vs-r plot, L838 energy equations) |
| 📝 User must edit (writing / citations) | ~28 |
| 📝 User must edit (apply skadd/skcut) | 10 |
| 📝 Needs follow-up with SK | 2 (L416 units confirm, L749 cooling tests) |
| **Total markers** | **~51** |

---

## Quick wins (easy LaTeX edits, <5 min each)

Apply these `\skadd` / `\skcut` pairs directly — the replacement text is already written:

1. **L295**: Accept `\skadd{We briefly review...}`, change "those" → "the above"
2. **L731**: Accept `\skadd{in the left column}`, change "before particle splitting" → "before the re-simulation"  
3. **L735–737**: Delete `\skcut{...}` sentence, accept `\skadd{...}` replacement
4. **L780**: "Notice" → "Note"
5. **L803**: "1$^{\text{st}}$ formed" → "first formed"
6. **L804**: "doesn't" → "does not", "Here it is important to note" → "We note"
7. **L842**: Remove "deeply"
8. **L846**: Add "potential" → "gravitational potential well"
9. **L855**: Move period: `(see section \ref{sec:diskid}).`
10. **L1850**: Remove ZAMS placeholder comment
