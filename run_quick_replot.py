"""Quick replot: only regenerate mass evolution + phase diagrams."""
import sys, os, importlib, time
sys.path.insert(0, "/scratch/vasissua/SHIVAN/analysis/")
sys.path.insert(0, "/home/vasissua/PYTHON/GUAC/src/")
sys.path.insert(0, "/home/vasissua/PYTHON/pfh_python/gizmopy/")

import hybrid_sims_utils.read_snap as _rsnap
importlib.reload(_rsnap)

import notebooks.paper_figures as pf
import disk_analysis.plot_mass_evolution as pme
import disk_analysis.plot_energy_evolution as pee
importlib.reload(pf)
importlib.reload(pme)
importlib.reload(pee)

OUTDIR = "/scratch/vasissua/SHIVAN/analysis/paper_plots"
PATH = '/scratch/vasissua/COPY/2026-03/m12f_cutout/'
SIM = 'output_jeans_refinement'
FULLSIM_DIR = '/scratch/vasissua/COPY/2026-03/m12f/output_jeans_refinement/'

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

args = Defaults()

# 1. Mass evolution (scans all snapshots)
print("=== Mass evolution ===")
pme.run(args)

# 1b. Energy evolution
print("\n=== Energy evolution ===")
pee.run(args)

# 2. Phase diagrams (need epoch data)
print("\n=== Phase diagrams ===")
import glob, numpy as np

sink_form_time_Myr = None
FRAMES18_DIR = os.path.join("/scratch/vasissua/SHIVAN/analysis/", "frames18")
qprofiles = sorted(glob.glob(os.path.join(FRAMES18_DIR, 'qprofiles', 'qprofile_*.npz')))
for qp in qprofiles:
    d = dict(np.load(qp, allow_pickle=True))
    if 'sink_form_Myr' in d and len(d['sink_form_Myr']) > 0:
        sink_form_time_Myr = float(np.min(d['sink_form_Myr']))
        break

epoch_data_list = []
for epoch in pf.EPOCHS:
    snap = epoch['snap']
    print(f"  Loading snap {snap:04d}...")
    try:
        ed = pf.extract_epoch_data(snap, args, sink_form_time_Myr)
        epoch_data_list.append(ed)
    except Exception as e:
        print(f"    ERROR: {e}")

pf._FULLSIM_PATH = FULLSIM_DIR if os.path.isdir(FULLSIM_DIR) else None
os.makedirs(os.path.join(OUTDIR, 'light'), exist_ok=True)
os.makedirs(os.path.join(OUTDIR, 'dark'), exist_ok=True)

print("  Generating phase diagrams...")
pf.plot_phase_diagrams(epoch_data_list, OUTDIR)
print("  Done.")
