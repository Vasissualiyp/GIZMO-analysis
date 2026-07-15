"""
run_phase_diagrams.py  (repurposed from run_bfield_phase.py)
------------------------------------------------------------
Regenerate only the T/H2 phase diagrams:
  - phase_combined.png
  - phase_instant_median.png
Loads the 6 epoch snapshots, then calls plot_phase_diagrams.
"""

import sys
import os
import importlib
import glob
import numpy as np

scratch_analysis_path = "/scratch/vasissua/SHIVAN/analysis/"
sys.path.insert(0, scratch_analysis_path)

guac_src_path = "/home/vasissua/PYTHON/GUAC/src/"
pfp_src_path  = "/home/vasissua/PYTHON/pfh_python/gizmopy/"
sys.path.insert(0, guac_src_path)
sys.path.insert(0, pfp_src_path)

import hybrid_sims_utils.read_snap as _rsnap
importlib.reload(_rsnap)

import notebooks.paper_figures as pf
importlib.reload(pf)

OUTDIR      = os.path.join(scratch_analysis_path, "paper_plots")
FRAMES18    = os.path.join(scratch_analysis_path, "frames18")
PATH        = '/scratch/vasissua/COPY/2026-03/m12f_cutout/'
SIM         = 'output_jeans_refinement'

class Defaults:
    path = PATH; sim = SIM; outdir = OUTDIR + '/'
    res = 400; image_box = 2e-5; r_search = 1e-5; r_max = 1e-5
    rho_thresh = 1e-15; aspect = 0.3; f_kep = 0.3
    vmin = 1e5; vmax = 1e8; ncores = 1; cmap = 'inferno'
    reference_center = None; reference_search_radius = 0.1
    corotate = True; vmax_vel = None
    min_gas_particles = 0; min_gas_snap = 150
    include_phase_in_master = False

os.makedirs(os.path.join(OUTDIR, 'light'), exist_ok=True)
os.makedirs(os.path.join(OUTDIR, 'dark'),  exist_ok=True)

args = Defaults()

# Sink formation time
sink_form_time_Myr = None
for qp in sorted(glob.glob(os.path.join(FRAMES18, 'qprofiles', 'qprofile_*.npz'))):
    d = dict(np.load(qp, allow_pickle=True))
    if 'sink_form_Myr' in d and len(d['sink_form_Myr']) > 0:
        sink_form_time_Myr = float(np.min(d['sink_form_Myr']))
        break
print(f"Sink formation time: {sink_form_time_Myr} Myr")

# Load main epochs
epoch_data_list = []
for epoch in pf.EPOCHS:
    snap = epoch['snap']
    print(f"\nLoading snap {snap:04d} ({epoch['label']})...")
    try:
        ed = pf.extract_epoch_data(snap, args, sink_form_time_Myr)
        epoch_data_list.append(ed)
        print(f"  Done — {ed['n_stars']} sinks, {ed['time_label']}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

# Build alt-snap list (alt_snap replaces snap for phase/bfield plots)
alt_epoch_data_list = list(epoch_data_list)
for i, (epoch, ed) in enumerate(zip(pf.EPOCHS, epoch_data_list)):
    alt_snap = epoch.get('alt_snap')
    if alt_snap is not None and alt_snap != epoch['snap']:
        print(f"\nLoading alt snap {alt_snap:04d} (replacing {epoch['snap']:04d})...")
        try:
            ed_alt = pf.extract_epoch_data(alt_snap, args, sink_form_time_Myr)
            alt_epoch_data_list[i] = ed_alt
            print(f"  Done — {ed_alt['n_stars']} sinks, {ed_alt['time_label']}")
        except Exception as e:
            print(f"  WARNING: {e}")

# Load KG2023 reference data
kg_data = None
kg_path = os.path.join(OUTDIR, 'kg2023_phase_data.npz')
if not os.path.exists(kg_path):
    kg_path = os.path.join(OUTDIR, '..', 'paper_plots', 'kg2023_phase_data.npz')
if os.path.exists(kg_path):
    try:
        kgd = np.load(kg_path)
        kg_data = {k: kgd[k] for k in kgd.files}
        print(f"\nLoaded KG2023 data from {kg_path}")
    except Exception as e:
        print(f"WARNING: could not load KG2023 data: {e}")

print("\nGenerating T/H2 phase diagrams...")
pf.plot_phase_diagrams(alt_epoch_data_list, OUTDIR,
                       frames_dir=FRAMES18, kg_data=kg_data)
print("Done.")
