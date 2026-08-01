"""
generate_paper_plots.py
-----------------------
Entry point for paper-quality multi-epoch figures.
Replaces the jupytertest naming convention.

Loads 6 key snapshots from the cutout simulation, extracts all quantities,
then generates multi-epoch master figures + time-evolution plots.
"""

import sys
import os
import importlib
import shutil
import time
import glob

import numpy as np

# ── Path setup ──
_CLUSTER = '/scratch/vasissua'
_LOCAL   = '/home/vasilii/research/trillium/scratch'
_BASE    = _CLUSTER if os.path.isdir(_CLUSTER) else _LOCAL

scratch_analysis_path = os.path.join(_BASE, 'SHIVAN/analysis/')
sys.path.insert(0, scratch_analysis_path)

import meshoid_plotting.starforge_plot as sfp
import meshoid_plotting.utility_funcs as utilf

if os.path.isdir(_CLUSTER):
    guac_src_path = "/home/vasissua/PYTHON/GUAC/src/"
    pfp_src_path = "/home/vasissua/PYTHON/pfh_python/gizmopy/"
else:
    guac_src_path = os.path.join(_LOCAL, "../home/PYTHON/GUAC/src/")
    pfp_src_path = os.path.join(_LOCAL, "../home/PYTHON/pfh_python/gizmopy/")
sys.path.insert(0, guac_src_path)
sys.path.insert(0, pfp_src_path)
import hybrid_sims_utils.read_snap as _rsnap
importlib.reload(_rsnap)

import notebooks.paper_figures as pf
import notebooks.make_disk_movie_frames as ntbk
import disk_analysis.plot_mass_evolution as pme
import disk_analysis.plot_energy_evolution as pee
import disk_analysis.plot_sink_history as psh

importlib.reload(pf)
importlib.reload(ntbk)
importlib.reload(pme)
importlib.reload(pee)
importlib.reload(psh)


# ── Configuration ──
OUTDIR = os.path.join(scratch_analysis_path, "paper_plots")

# Cutout simulation only
PATH = os.path.join(_BASE, 'COPY/2026-03/m12f_cutout/')
SIM = 'output_jeans_refinement'

# Full simulation (for phase diagram background; snap 27 has H₂ data)
FULLSIM_DIR = os.path.join(_BASE, 'COPY/2026-03/m12f/output_jeans_refinement/')

# Cutout simulation with FIRE+refinement gas (small files, ~11 MB each)
# Used for wide-radius profiles (500 pc) instead of full sim (39 GB per snap)
WIDESIM_DIR = os.path.join(_BASE, 'COPY/2026-03/m12f_cutout/output_jeans_refinement/')

# Existing frames18 dir for time-evolution data
FRAMES18_DIR = os.path.join(scratch_analysis_path, "frames18")


class Defaults:
    def __init__(self):
        self.path = PATH
        self.sim = SIM
        self.outdir = OUTDIR + '/'
        self.snap_start = None
        self.snap_end = None
        self.res = 400
        self.image_box = 2e-5
        self.r_search = 1e-5
        self.r_max = 1e-5
        self.rho_thresh = 1e-15
        self.aspect = 0.3
        self.f_kep = 0.3
        self.vmin = 1e5
        self.vmax = 1e8
        self.ncores = 1
        self.cmap = 'inferno'
        self.reference_center = None
        self.reference_search_radius = 0.1
        self.corotate = True
        self.vmax_vel = None
        self.min_gas_particles = 0
        self.min_gas_snap = 150
        self.include_phase_in_master = False


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(os.path.join(OUTDIR, 'light'), exist_ok=True)
    os.makedirs(os.path.join(OUTDIR, 'dark'), exist_ok=True)

    args = Defaults()

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: Find sink formation time (scan for earliest sink)
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("Phase 1: Determining sink formation time...")
    print("=" * 60)

    sink_form_time_Myr = None
    # Try to get from existing qprofile data
    qprofiles = sorted(glob.glob(os.path.join(FRAMES18_DIR, 'qprofiles', 'qprofile_*.npz')))
    for qp in qprofiles:
        d = dict(np.load(qp, allow_pickle=True))
        if 'sink_form_Myr' in d and len(d['sink_form_Myr']) > 0:
            t1 = float(np.min(d['sink_form_Myr']))
            if sink_form_time_Myr is None or t1 < sink_form_time_Myr:
                sink_form_time_Myr = t1
            break  # first file with sinks is enough since they're sorted

    if sink_form_time_Myr is not None:
        print(f"  Sink formation time: {sink_form_time_Myr:.6f} Myr")
    else:
        print("  WARNING: Could not determine sink formation time. Using absolute times.")

    # Full-sim path for phase diagram H₂ background (snap 27 only)
    pf._FULLSIM_PATH = FULLSIM_DIR if os.path.isdir(FULLSIM_DIR) else None
    if pf._FULLSIM_PATH:
        print(f"  Full-sim path set: {pf._FULLSIM_PATH}")
    else:
        print(f"  WARNING: Full-sim dir not found ({FULLSIM_DIR}); phase H₂ background unavailable")

    # Wide-sim path for 500 pc profiles (cutout sim: small files, ~11 MB each)
    pf._WIDESIM_PATH = WIDESIM_DIR if os.path.isdir(WIDESIM_DIR) else pf._FULLSIM_PATH
    if pf._WIDESIM_PATH:
        print(f"  Wide-sim path set: {pf._WIDESIM_PATH}")
    else:
        print(f"  WARNING: No wide-sim dir; wide profiles will use cutout only")


    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: Extract data for 6 epochs
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Phase 2: Loading 6 epoch snapshots...")
    print("=" * 60)

    epoch_data_list = []
    for epoch in pf.EPOCHS:
        snap = epoch['snap']
        print(f"\n  Loading snapshot {snap:04d} ({epoch['label']})...")
        t0 = time.perf_counter()
        try:
            ed = pf.extract_epoch_data(snap, args, sink_form_time_Myr)
            epoch_data_list.append(ed)
            dt = time.perf_counter() - t0
            print(f"    Done in {dt:.1f}s — {ed['n_stars']} sinks, label: {ed['time_label']}")
        except Exception as e:
            print(f"    ERROR loading snap {snap:04d}: {e}")
            import traceback; traceback.print_exc()

    if len(epoch_data_list) == 0:
        print("FATAL: No epochs loaded successfully.")
        sys.exit(1)

    print(f"\n  Successfully loaded {len(epoch_data_list)}/{len(pf.EPOCHS)} epochs.")

    # ── Alt-snap override: load alternative snaps where alt_snap != snap ──
    # Used for both phase diagrams and B-field phase plots.
    alt_epoch_data_list = list(epoch_data_list)  # copy; entries replaced below
    for i, (epoch, ed) in enumerate(zip(pf.EPOCHS, epoch_data_list)):
        alt_snap = epoch.get('alt_snap')
        if alt_snap is not None and alt_snap != epoch['snap']:
            print(f"\n  Loading alt snapshot {alt_snap:04d} "
                  f"(replacing snap {epoch['snap']:04d} for phase/bfield plots)...")
            try:
                ed_alt = pf.extract_epoch_data(alt_snap, args, sink_form_time_Myr)
                alt_epoch_data_list[i] = ed_alt
                print(f"    Done — {ed_alt['n_stars']} sinks, label: {ed_alt['time_label']}")
            except Exception as e:
                print(f"    WARNING: Could not load alt snap {alt_snap}: {e}")
                # fall back to original epoch data for this slot

    # ══════════════════════════════════════════════════════════════════════
    # Phase 3: Generate multi-epoch figures
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Phase 3: Generating multi-epoch figures...")
    print("=" * 60)

    # Pre-load sink_data if it already exists (from a prior run) so xi-gamma
    # heatmap gets formation/merger overlays even on the first full run.
    _sink_data_path = os.path.join(OUTDIR, 'sink_data.npz')
    _pre_merge_data = None
    if os.path.exists(_sink_data_path):
        _pre_merge_data = dict(np.load(_sink_data_path))

    # pf._FULLSIM_PATH already set above (Phase 1) for use in extract_epoch_data too
    pf.make_all_figures(epoch_data_list, OUTDIR, frames_dir=FRAMES18_DIR,
                        alt_epoch_data_list=alt_epoch_data_list,
                        merge_data=_pre_merge_data)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 4: Time-evolution plots (ALL snapshots via existing data)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Phase 4: Time-evolution plots (all snapshots)...")
    print("=" * 60)

    # ── Sink history (individual tracking) ──────────────────────────────────
    print("  Generating individual sink history plots...")
    cutout_dir = os.path.join(args.path, args.sim)
    sink_data_path = os.path.join(OUTDIR, 'sink_data.npz')
    try:
        psh.run(cutout_dir, OUTDIR, r_max_AU=2500)
    except Exception as e:
        print(f"  WARNING: Sink history failed: {e}")
        import traceback; traceback.print_exc()

    # Load sink_data for heatmap overlays
    merge_data   = None
    pos_history  = None
    if os.path.exists(sink_data_path):
        _sd = np.load(sink_data_path)
        merge_data  = dict(_sd)
        pos_history = dict(_sd)

    # Heatmaps from existing qprofile data
    evol_light = os.path.join(OUTDIR, 'light')
    if os.path.isdir(os.path.join(FRAMES18_DIR, 'qprofiles')):
        print("  Generating Q heatmap...")
        ntbk.make_Q_heatmap(FRAMES18_DIR, heatmap_path=os.path.join(evol_light, 'Q_heatmap.png'),
                            merge_data=merge_data, pos_history=pos_history)
        print("  Generating sigma heatmap...")
        ntbk.make_sigma_heatmap(FRAMES18_DIR, heatmap_path=os.path.join(evol_light, 'sigma_heatmap.png'),
                                merge_data=merge_data, pos_history=pos_history)
        print("  Generating sigma_r heatmap...")
        ntbk.make_sigma_r_heatmap(FRAMES18_DIR, heatmap_path=os.path.join(evol_light, 'sigma_r_heatmap.png'),
                                  merge_data=merge_data, pos_history=pos_history)

        # Merged Toomre Q figure: face-on Q grid (2×3) + Q heatmap below
        print("  Generating merged Toomre Q figure...")
        try:
            pf.plot_toomre_Q_merged(epoch_data_list, FRAMES18_DIR, OUTDIR,
                                     merge_data=merge_data, pos_history=pos_history)
        except Exception as e:
            print(f"  WARNING: Merged Q figure failed: {e}")
            import traceback; traceback.print_exc()
    else:
        print("  WARNING: No qprofiles in frames18/, skipping heatmaps.")

    # Mass + accretion rate heatmaps (from massprofiles/ subdir)
    mp_dir = os.path.join(FRAMES18_DIR, 'massprofiles')
    if os.path.isdir(mp_dir):
        print("  Generating mass + accretion heatmaps...")
        _t1_Myr = None
        if merge_data is not None and 'form_times_kyr' in merge_data:
            _ft = np.asarray(merge_data['form_times_kyr'])
            if len(_ft) > 0:
                _t1_Myr = float(_ft.min()) / 1e3   # kyr → Myr
        try:
            pf.plot_mass_accretion_heatmaps(FRAMES18_DIR, _t1_Myr, OUTDIR,
                                             merge_data=merge_data)
        except Exception as e:
            print(f"  WARNING: Mass accretion heatmaps failed: {e}")
            import traceback; traceback.print_exc()
    else:
        print("  No massprofiles/ dir — run run_mass_profiles.sh first.")

    # Mass and energy evolution (these scan all snapshots internally)
    # Skip full recomputation if cached .npz exists — just replot.
    args_evol = Defaults()
    args_evol.outdir = OUTDIR + '/'
    _mass_npz   = os.path.join(OUTDIR, 'mass_evolution.npz')
    _energy_npz = os.path.join(OUTDIR, 'energy_evolution.npz')

    if os.path.exists(_mass_npz):
        print("  Mass evolution npz found — replotting only (delete mass_evolution.npz to recompute)...")
        try:
            pme.plot_from_npz(_mass_npz, OUTDIR + '/')
        except Exception as e:
            print(f"  WARNING: Mass evolution replot failed: {e}; falling back to full run")
            try:
                pme.run(args_evol)
            except Exception as e2:
                print(f"  WARNING: Mass evolution failed: {e2}")
    else:
        print("  Generating mass evolution (scanning snapshots)...")
        try:
            pme.run(args_evol)
        except Exception as e:
            print(f"  WARNING: Mass evolution failed: {e}")

    if os.path.exists(_energy_npz):
        print("  Energy evolution npz found — replotting only (delete energy_evolution.npz to recompute)...")
        try:
            pee.plot_from_npz(_energy_npz, OUTDIR + '/')
        except Exception as e:
            print(f"  WARNING: Energy evolution replot failed: {e}; falling back to full run")
            try:
                pee.run(args_evol)
            except Exception as e2:
                print(f"  WARNING: Energy evolution failed: {e2}")
    else:
        print("  Generating energy evolution (scanning snapshots)...")
        try:
            pee.run(args_evol)
        except Exception as e:
            print(f"  WARNING: Energy evolution failed: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # Done
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("All paper figures generated.")
    print(f"Output directory: {OUTDIR}")
    print("=" * 60)

    # List outputs
    for subdir in ['light', 'dark']:
        d = os.path.join(OUTDIR, subdir)
        if os.path.isdir(d):
            files = sorted(os.listdir(d))
            print(f"\n  {subdir}/ ({len(files)} files):")
            for f in files:
                sz = os.path.getsize(os.path.join(d, f))
                print(f"    {f}  ({sz/1024:.0f} KB)")


if __name__ == '__main__':
    main()
