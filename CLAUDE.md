# CLAUDE.md — SHIVAN/analysis

Analysis pipeline for FIRE+STARFORGE hybrid cosmological zoom-in simulations.

---

## Plot conventions (MUST follow)

- **NEVER split dm/dt into "inflow" vs "outflow" curves** unless the user explicitly asks for it. For any mass accretion rate plot (shell profile dm/dt, disk dM/dt, infall timescale), show a single dm/dt curve. Use a symlog or signed scale if needed — but one line per epoch/quantity, not two.
- Aspect ratio 4:3 (`figsize=(12,9)`) for all non-geometry plots. Face-on maps stay 1:1; grids stay whatever geometry requires.
- **NEVER put titles on plots** (`fig.suptitle`, `ax.set_title`, etc.). No figure or panel titles of any kind. Panels are identified by time labels (text annotations) or axis labels only.
Primary goal: produce disk movie frames and identify/characterize protostellar disks
forming at z~21 in the m12f galaxy simulation.

---

## Repository layout

```
/home/vasilii/research/trillium/scratch/SHIVAN/analysis/
├── notebooks/
│   └── make_disk_movie_frames.py   # Main movie-frame rendering pipeline
├── disk_analysis/
│   ├── disk_utils.py               # Standalone disk identification toolkit (657 lines)
│   ├── create_cutouts.py           # Spatial cutout HDF5 extractor
│   ├── analyze_L_stability.py      # Angular momentum stability over time
│   ├── plot_centers_xt.py          # Center position x-t plots
│   └── make_movie_from_frames.sh   # ffmpeg movie assembly script
├── meshoid_plotting/
│   └── starforge_plot.py           # Meshoid-based rendering + H2 chemistry (1607 lines)
├── jupytertest3.py                 # Main entry point → calls ntbk.main(Defaults())
├── jupytertest2.py                 # Angular momentum stability analysis
├── jupytertest.py                  # Zoom-in visualization
├── frames/                         # Generated PNG frames
└── plots/                          # Other plot outputs
```

---

## Simulation details

- **Simulation**: m12f FIRE-3 cosmological zoom-in with STARFORGE jeans refinement
- **Epoch**: z ≈ 21 (scale factor a ≈ 0.045)
- **Data location**: `/scratch/vasissua/COPY/2026-03/m12f/output_cutout/` (spatial cutouts)
- **Full sim**: `/scratch/vasissua/COPY/2026-03/m12f/output_new_jeans_refinement/`
- **Snapshots**: `snapshot_XXXX.hdf5` (not in a `snapshots/` subdir → `snapdir=False`)
- **Cutout generation**: `disk_analysis/create_cutouts.py` with `--cutout-radius 0.005` (comoving kpc/h, ≈ 67,000 AU / 0.32 pc physical at z~21)

**Why cutouts?** The full simulation has FIRE-resolution gas with smoothing lengths >> the 20 AU image box. Cutouts contain only the high-resolution jeans-refinement gas near the disk. Without them you get a single-pixel blob with no structure.

---

## GIZMO code unit conventions (after `convert_units_to_physical`)

| Quantity      | Unit                     | Notes                              |
|---------------|--------------------------|------------------------------------|
| Coordinates   | kpc (physical)           | × a/h from comoving kpc/h          |
| Velocities    | km/s                     | × √a                               |
| Masses        | 10^10 M_sun              | × 1/h; multiply by 1e10 for M_sun  |
| Density       | 10^10 M_sun / kpc^3      | × h/(a/h)^3                        |
| SmoothingLength | kpc (physical)         | same as Coordinates                |

**CGS constants** (from `generic_utils.constants`):
- `kpc = 3.086e21 cm`, `AU = 1.496e13 cm`, `Msun = 1.989e33 g`, `G = 6.67e-8 cm³/g/s²`

---

## Data loading

```python
from hybrid_sims_utils.read_snap import get_snap_data_hybrid, convert_units_to_physical

hdr, pdata, stardata, fire_stardata, refine_data, snapname = get_snap_data_hybrid(
    sim, path, snap_num, snapshot_suffix='', snapdir=False,
    verbose=False, custom_gas_fields=['Masses','Coordinates','SmoothingLength','Velocities','Density'])

hdr, pdata, stardata, fire_stardata = convert_units_to_physical(hdr, pdata, stardata, fire_stardata)
```

**Key particle types**:
- `PartType0` → gas (`pdata`)
- `PartType3` → refinement/tracer particles
- `PartType4` → FIRE stars (`fire_stardata`)
- `PartType5` → STARFORGE sink particles (`stardata`)

**6-value return** (post GUAC update): `hdr` is separate from `pdata`. The old 5-value API `(pdata, stardata, fire_stardata, refine_data, snapname)` is obsolete.

**Module reload issue**: If old GUAC is cached in `sys.modules`, import `hybrid_sims_utils.read_snap as _rsnap; importlib.reload(_rsnap)` before importing `notebooks.make_disk_movie_frames` to force the new API.

---

## Pipeline: `make_disk_movie_frames.py`

Entry point: `jupytertest3.py` → `ntbk.main(Defaults())`

**`Defaults()` parameters** (jupytertest3.py):
```python
path              = '/scratch/vasissua/COPY/2026-03/m12f/'
sim               = 'output_cutout'
outdir            = '/scratch/vasissua/SHIVAN/analysis/'
res               = 400            # pixels per axis
image_box         = 2e-5           # kpc full width (≈ 600 AU diameter)
r_search          = 1e-5           # kpc — L_hat computation radius
r_max             = 1e-5           # kpc — outer disk boundary
rho_thresh        = 1e-15          # g/cm³ — density threshold
aspect            = 0.3            # |z|/r_cyl cutoff
f_kep             = 0.3            # min v_phi/v_K
vmin, vmax        = 1e5, 1e8       # Msun/pc² colorbar range
ncores            = 1
cmap              = 'inferno'
reference_center  = None           # kpc; set to approximate refinement region center
reference_search_radius = 0.1      # kpc
```

**Key functions**:
- `find_center` — most massive sink (stable vs mass-weighted COM when secondaries form)
- `get_disk_axis` — angular momentum L_hat from gas within `r_search`
- `cylindrical_coords` — projects to (r_cyl, z, v_phi, v_r, v_z) in disk frame
- `compute_M_enc` — enclosed mass (vectorized argsort+cumsum, exclusive)
- `identify_disk` — geometric (r_cyl < r_max, |z|/r_cyl < aspect, ρ > ρ_thresh) + kinematic (v_phi > 0, v_phi/v_K > f_kep) stages; returns `(is_disk, com, L_hat, r_cyl, z, v_phi, v_K, com_vel)`
- `render_frame` — 3×4 panel figure (see below)

**`render_frame` output grid** (3×4, figsize 28×18):
```
Row 0: [Face-on SD small | Face-on SD 10× | Edge-on SD clean | Edge-on SD + disk overlay]
Row 1: [Face-on |δv|     | Face-on σ_|δv| | Edge-on |δv|     | Edge-on σ_|δv|           ]
Row 2: [v_r vs r phase   | v_phi vs r phase | (hidden)        | (hidden)                 ]
```
- Row 0: surface density in Msun/pc² (inferno, log)
- Row 1: rest-frame turbulent velocity |δv| [km/s] and dispersion σ_|δv| [km/s] (viridis, linear)
  - Streaming subtracted via 20-bin mass-weighted radial profiles of v_r(r) and v_phi(r)
  - |δv| = sqrt(δv_r² + δv_phi² + δv_z²) per particle, mass-weighted projected
- Row 2: 1D scatter of every gas particle (v_r vs r, v_phi vs r) with profile fit (red) overlaid

---

## Submitting cluster jobs via the command queue

A background daemon (`queue_runner.sh`) watches `scripts.txt` and runs commands one at a time every ~10 s.
Start it once per login session (or keep it alive with nohup):

```bash
cd /scratch/vasissua/SHIVAN/analysis
nohup bash queue_runner.sh &   # logs to nohup.out
```

**CRITICAL: Only one `debug` partition job at a time.** Trillium rejects a second `sbatch` to `debug` while one is already running. Never queue two `sbatch` commands back-to-back without waiting for the first to finish.

**Chaining debug jobs — REQUIRED pattern:** When submitting multiple dependent debug jobs (e.g. recompute npz then replot), you MUST use SLURM job dependencies. Capture the first job ID and pass it as a prerequisite to the next:

```bash
# Single line in scripts.txt — submits job2 only after job1 completes:
JOB1=$(sbatch run_mass_evolution.sh | awk '{print $NF}') && sbatch --dependency=afterany:$JOB1 run_paper_plots.sh
```

**Before submitting any debug job**, check whether one is already running:
```bash
squeue -u $USER -p debug
```
If a job is running, append `--dependency=afterany:RUNNING_JOBID` to the new `sbatch` call. Never submit a bare `sbatch` to debug while another debug job is active — it will be silently rejected.

**To submit the standard plotting job** (`jupytertest3.py` via Slurm):
```bash
echo "cd /scratch/vasissua/SHIVAN/analysis && sbatch run_plotter.sh" >> /scratch/vasissua/SHIVAN/analysis/scripts.txt
```

The daemon picks this up within 10 s, runs `sbatch run_plotter.sh` from the analysis directory, and the job enters the Slurm queue. Output goes to `output.log` (stdout from `jupytertest3.py`) and `slurm-JOBID.out` (Slurm wrapper stdout/stderr).

Other useful one-liners:
```bash
# Re-generate cutouts only
echo "python disk_analysis/create_cutouts.py --overwrite" >> scripts.txt

# Run energy evolution script standalone
echo "python disk_analysis/plot_energy_evolution.py" >> scripts.txt
```

Stop the daemon cleanly:
```bash
touch /scratch/vasissua/SHIVAN/analysis/queue_runner.stop
```

---

## Master frame layout rule

**Max 3 rows per figure.** When adding new panels that would push a figure beyond 3 rows, create a new Frame figure (C, D, …) instead of expanding existing ones. Current output per snapshot:
- `frame_maps_XXXX.png` — Frame A (3×4): SD maps | velocity maps | slice + B_z maps
- `frame_analysis_XXXX.png` — Frame B (3×4): scatter+Q | Q map+profiles | resolution+virial+μ
- `frame_phase_XXXX.png` — Frame C (1×4, optional): phase diagrams (`include_phase=True`)

---

## Figure captions rule

**Every new plot type must have a draft caption in `tex/captions.tex`.**

When adding a new figure output to the pipeline (new per-snapshot frame, new thematic individual frame, new evolution plot, etc.):

1. Add a `\subsection{}` entry to `tex/captions.tex` following the existing style:
   - `\includegraphics` path pointing to the actual output path under `\framesdir` (use `\samplesnap` for per-snapshot files)
   - Caption body explaining **how each quantity is computed** (include equations where non-obvious)
   - At least one `\fillme{...}` placeholder for the writer to describe patterns/results
2. Update the repository layout table in this file if a new output subdirectory is created.

The file `tex/captions.tex` is a self-contained LaTeX document (compiles standalone with `pdflatex`). Keep build artifacts (`.aux`, `.log`, `.pdf`, etc.) inside `tex/` — they are not part of the analysis source.

---

## Disk identification logic

1. **Pre-filter**: particles within `r_local = max(5×r_max, 2×r_search)` to avoid processing millions of low-res FIRE particles
2. **COM velocity subtraction**: mass-weighted within `r_search` → `com_vel`
3. **L_hat**: from angular momentum `L = Σ m_i (r_i × v_i)` within `r_search`
4. **Cylindrical projection**: `(r_cyl, z, v_phi, v_r, v_z)` aligned with L_hat
5. **Geometric filter**: `r_cyl < r_max`, `|z|/r_cyl < aspect`, `ρ > ρ_thresh`
6. **Kinematic filter**: `v_phi > 0` (co-rotating), `v_phi/v_K > f_kep`
7. **Bound extension**: extend to R_bound, H_bound cylinder (100th percentile)

---

## Cutout creation

```bash
cd /home/vasilii/research/trillium/scratch/SHIVAN/analysis/disk_analysis/
python create_cutouts.py --overwrite --cutout-radius 0.005
```

- Processes snapshots in **reverse** order (latest first) to propagate sink COM backward
- Keeps: PartType0, PartType3, PartType4, PartType5
- Skips: PartType1 (dark matter — 88M particles), PartType2 (disk stars)
- Output: `/scratch/vasissua/COPY/2026-03/m12f/output_cutout/snapshot_XXXX.hdf5`
- `--cutout-radius 0.005` comoving kpc/h ≈ 67,000 AU / 0.32 pc physical at z~21 (needed for pre-sink gas; smaller values give empty early frames)

---

## Movie assembly

```bash
# Skip non-sequential frame numbers with concat demuxer:
cd /scratch/vasissua/SHIVAN/analysis/
printf "file '%s'\n" $(ls frame_*.png | sort) > filelist.txt
ffmpeg -f concat -safe 0 -i filelist.txt -c:v libx264 -crf 18 -pix_fmt yuv420p disk_movie.mp4
```

---

## Known issues / design decisions

- **COM stability**: Use most massive sink, not mass-weighted COM of all sinks. When secondary sinks form far from the primary disk, COM jumps to empty space.
- **Edge-on swap**: `pos_edge = pos_small[:, [0, 2, 1]]` swaps y↔z axes to display edge-on view (x-horizontal, z-vertical).
- **Empty early frames**: if pre-sink cutout radius is too small (< ~0.003), gas outside the cutout sphere is missing and surface density is ~0.
- **GUAC path**: `/home/vasilii/research/trillium/home/PYTHON/GUAC/src/` on local machine; `/home/vasissua/PYTHON/GUAC/src/` on cluster.
- **`convert_scale_factor_to_time`**: Called with `pdata` but cosmological params are in `hdr`. The try/except fallback uses `hdr['Time']` (scale factor) directly when the call fails.
- **Velocity maps**: 5 extra Meshoid `SurfaceDensity` calls (mass norm + |v_z| + v_z + v_z² projections). Adds ~20–40% rendering time per frame.

---

## External dependencies

| Package      | Path / install                               |
|--------------|----------------------------------------------|
| GUAC         | `~/research/trillium/home/PYTHON/GUAC/src/`  |
| pfh_python   | `~/research/trillium/home/PYTHON/pfh_python/gizmopy/` |
| meshoid      | `pip install meshoid`                        |
| yt           | conda/pip (used in GUAC cosmology utils)     |
| cmasher      | optional — registers `cmr.*` colormaps       |

---

## Available skills (slash commands)

| Command | Purpose |
|---------|---------|
| `/submit-job run_paper_plots.sh` | Submit a SLURM job via queue_runner and monitor it |
| `/check-job [job-id]` | Check SLURM job status from slurm output files |
| `/view-plot energy_evolution` | Display a paper plot PNG (partial name match) |
| `/plot-update` | Edit paper figure code and submit the plotting job |
| `/tex-check disk identification` | Verify LaTeX description matches code implementation |
| `/run-local-sci script.py` | Run a Python script locally with scientific packages |
| `/sim-snapshot-info path.hdf5` | Inspect a GIZMO snapshot (header, fields, units) |

### Simulation data paths

| Data | Cluster path | Local path |
|------|-------------|------------|
| Cutout sim | `/scratch/vasissua/COPY/2026-03/m12f_cutout/output_jeans_refinement/` | `~/research/trillium/scratch/COPY/2026-03/m12f_cutout/output_jeans_refinement/` |
| Full sim | `/scratch/vasissua/COPY/2026-03/m12f/output_jeans_refinement/` | `~/research/trillium/scratch/COPY/2026-03/m12f/output_jeans_refinement/` |
| Spatial cutouts | `/scratch/vasissua/COPY/2026-03/m12f_cutout/output_cutout/` | `~/research/trillium/scratch/COPY/2026-03/m12f_cutout/output_cutout/` |
