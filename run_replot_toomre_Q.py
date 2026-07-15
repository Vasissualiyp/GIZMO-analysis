"""Replot only the merged Toomre Q figure (6 face-on maps + heatmap).

Loads the 6 epoch snapshots (to get Q_fo_combined face-on maps), then
calls plot_toomre_Q_merged which reads the already-patched qprofile npz
files for the heatmap component.
"""
import sys, os, importlib, glob
import numpy as np

_BASE = '/scratch/vasissua' if os.path.isdir('/scratch/vasissua') else '/home/vasilii/research/trillium/scratch'
sys.path.insert(0, os.path.join(_BASE, 'SHIVAN/analysis/'))
sys.path.insert(0, '/home/vasissua/PYTHON/GUAC/src/' if os.path.isdir('/scratch/vasissua') else
                   '/home/vasilii/research/trillium/home/PYTHON/GUAC/src/')
sys.path.insert(0, '/home/vasissua/PYTHON/pfh_python/gizmopy/' if os.path.isdir('/scratch/vasissua') else
                   '/home/vasilii/research/trillium/home/PYTHON/pfh_python/gizmopy/')

import hybrid_sims_utils.read_snap as _rsnap
importlib.reload(_rsnap)
import notebooks.paper_figures as pf
importlib.reload(pf)

OUTDIR      = os.path.join(_BASE, 'SHIVAN/analysis/paper_plots')
FRAMES18    = os.path.join(_BASE, 'SHIVAN/analysis/frames18')
PATH        = os.path.join(_BASE, 'COPY/2026-03/m12f_cutout/')
SIM         = 'output_jeans_refinement'
FULLSIM_DIR = os.path.join(_BASE, 'COPY/2026-03/m12f/output_jeans_refinement/')

class Args:
    path = PATH; sim = SIM; outdir = OUTDIR + '/'
    res = 400; image_box = 2e-5; r_search = 1e-5; r_max = 1e-5
    rho_thresh = 1e-15; aspect = 0.3; f_kep = 0.3
    vmin = 1e5; vmax = 1e8; ncores = 1; cmap = 'inferno'
    reference_center = None; reference_search_radius = 0.1
    corotate = True; vmax_vel = None; min_gas_particles = 0; min_gas_snap = 150
    include_phase_in_master = False

args = Args()
pf._FULLSIM_PATH = FULLSIM_DIR if os.path.isdir(FULLSIM_DIR) else None

# Find first-sink formation time from qprofiles
sink_form_time_Myr = None
for qp in sorted(glob.glob(os.path.join(FRAMES18, 'qprofiles', 'qprofile_*.npz'))):
    d = np.load(qp, allow_pickle=True)
    if 'sink_form_Myr' in d and len(d['sink_form_Myr']) > 0:
        sink_form_time_Myr = float(np.min(d['sink_form_Myr']))
        break

# Load epoch data
epoch_data_list = []
os.makedirs(os.path.join(OUTDIR, 'light'), exist_ok=True)
os.makedirs(os.path.join(OUTDIR, 'dark'), exist_ok=True)

for epoch in pf.EPOCHS:
    snap = epoch['snap']
    print(f"  Loading snap {snap:04d}...", flush=True)
    try:
        ed = pf.extract_epoch_data(snap, args, sink_form_time_Myr)
        epoch_data_list.append(ed)
    except Exception as e:
        print(f"    ERROR: {e}")

# Load sink position history for r(t) overlay
merge_data = None
pos_history = None
sink_data_path = os.path.join(OUTDIR, 'sink_data.npz')
if os.path.exists(sink_data_path):
    _sd = np.load(sink_data_path, allow_pickle=True)
    merge_data  = dict(_sd)
    pos_history = dict(_sd)
    print(f"  Loaded sink_data.npz (keys: {list(_sd.keys())[:6]}...)")
else:
    print(f"  No sink_data.npz found at {sink_data_path}, skipping pos_history overlay.")

print("Plotting merged Toomre Q figure...", flush=True)
pf.plot_toomre_Q_merged(epoch_data_list, FRAMES18, OUTDIR,
                        merge_data=merge_data, pos_history=pos_history)
print("Done.")
