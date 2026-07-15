"""
run_bfield_maps.py
------------------
Regenerate only the B-field 2D map grids (combined_Bz, combined_Bpol,
combined_Btor, combined_Br, combined_Bmag, combined_mu).
Uses the new midplane-slice B-field rendering in extract_epoch_data.
Loads all 6 epochs, then calls the relevant plot_grid_combined calls.
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

from matplotlib import colors as mcolors

OUTDIR   = os.path.join(scratch_analysis_path, "paper_plots")
FRAMES18 = os.path.join(scratch_analysis_path, "frames18")
PATH     = '/scratch/vasissua/COPY/2026-03/m12f_cutout/'
SIM      = 'output_jeans_refinement'

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

# Load all 6 epochs
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

if not epoch_data_list:
    print("FATAL: no epochs loaded"); sys.exit(1)

# B-field signed maps (Bz, Br, Btor, Bpol)
def _bfield_range(key):
    vals = []
    for ed in epoch_data_list:
        m = ed.get(key)
        if m is not None:
            vals.append(m[np.isfinite(m)])
    if not vals:
        return None
    all_v = np.concatenate(vals)
    if len(all_v) == 0:
        return None
    vmax = np.percentile(np.abs(all_v[all_v != 0]), 99) if np.any(all_v != 0) else 1e-10
    return mcolors.SymLogNorm(linthresh=max(vmax * 1e-3, 1e-12), vmin=-vmax, vmax=vmax)

for bcomp, blabel in [('Bz',   r'$B_z$'),
                       ('Br',   r'$B_r$'),
                       ('Btor', r'$B_\phi$'),
                       ('Bpol', r'$B_{\rm pol}$')]:
    fo_key = f'{bcomp}_fo'
    eo_key = f'{bcomp}_eo'
    b_norm = _bfield_range(fo_key) or _bfield_range(eo_key)
    if b_norm is None:
        print(f"  Skipping {bcomp} (no data)")
        continue
    print(f"\nGenerating combined {bcomp} grid...")
    pf.plot_grid_combined(epoch_data_list, fo_key, eo_key, OUTDIR, 'RdBu_r', b_norm,
                          f'{blabel} [G]', f'combined_{bcomp}.png')

# |B| magnitude
all_bmag = np.concatenate([
    ed['Bmag_fo'][ed['Bmag_fo'] > 0] for ed in epoch_data_list
    if ed.get('Bmag_fo') is not None and np.any(ed['Bmag_fo'] > 0)])
if len(all_bmag) > 0:
    bmag_norm = mcolors.LogNorm(vmin=max(np.percentile(all_bmag, 5), 1e-10),
                                vmax=np.percentile(all_bmag, 99.5))
    print("\nGenerating combined |B| magnitude grid...")
    pf.plot_grid_combined(epoch_data_list, 'Bmag_fo', 'Bmag_eo', OUTDIR,
                          'plasma', bmag_norm, r'$|B|$ [G]', 'combined_Bmag.png')

# Mass-to-flux ratio μ
all_mu_list = [ed['mu_fo'][np.isfinite(ed['mu_fo']) & (ed['mu_fo'] > 0)]
               for ed in epoch_data_list
               if ed.get('mu_fo') is not None
               and np.any(np.isfinite(ed['mu_fo']) & (ed['mu_fo'] > 0))]
if all_mu_list:
    all_mu = np.concatenate(all_mu_list)
    mu_norm = mcolors.LogNorm(vmin=max(np.percentile(all_mu, 1), 0.01),
                              vmax=min(np.percentile(all_mu, 99), 1e4))
    print("\nGenerating combined μ (mass-to-flux) grid...")
    pf.plot_grid_combined(epoch_data_list, 'mu_fo', 'mu_eo', OUTDIR,
                          'RdYlBu_r', mu_norm, r'$\mu_\Phi$', 'combined_mu.png',
                          contour_level=1.0)

print("\nDone.")
