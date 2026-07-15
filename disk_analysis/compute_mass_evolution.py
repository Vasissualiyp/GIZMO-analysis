"""
compute_mass_evolution.py
=========================
Scan all cutout snapshots and save a single mass_evolution.npz with
cumulative enclosed mass M(<r) and disk mass vs time.

Modelled on plot_energy_evolution.py — uses GUAC for consistent unit
conversion and disk identification.

Output: {frames_dir}/mass_evolution.npz
  times_Myr   (N,)       absolute time [Myr]
  t1_Myr      scalar     first sink formation time [Myr]
  r_AU        (N_r,)     bin centres [AU]
  r_edges_AU  (N_r+1,)   bin edges [AU]
  M_enc       (N, N_r)   cumulative enclosed gas mass [Msun]
  M_shell     (N, N_r)   differential shell mass = diff(M_enc) [Msun]
  M_disk      (N,)       disk-identified gas mass [Msun]
  M_star      (N,)       total sink mass [Msun]
  n_sinks     (N,)

Usage (cluster):
    python disk_analysis/compute_mass_evolution.py [--overwrite]
"""

import os
import sys
import glob
import argparse

import numpy as np
import h5py
from scipy.integrate import quad

# ── Path setup ──────────────────────────────────────────────────────────────
_CLUSTER = '/scratch/vasissua'
_LOCAL   = '/home/vasilii/research/trillium/scratch'
BASE     = _CLUSTER if os.path.isdir(_CLUSTER) else _LOCAL

CUTOUT_DIR = os.path.join(BASE, 'COPY/2026-03/m12f_cutout/output_jeans_refinement')
FRAMES_DIR = os.path.join(BASE, 'SHIVAN/analysis/frames18')

guac_src_path = os.path.join(BASE.replace('/scratch/vasissua', '/home/vasissua')
                             .replace('/home/vasilii/research/trillium/scratch',
                                      '/home/vasilii/research/trillium/home'),
                             'PYTHON/GUAC/src/')
pfp_src_path  = os.path.join(BASE.replace('/scratch/vasissua', '/home/vasissua')
                             .replace('/home/vasilii/research/trillium/scratch',
                                      '/home/vasilii/research/trillium/home'),
                             'PYTHON/pfh_python/gizmopy/')

# Cluster-style paths
if os.path.isdir(_CLUSTER):
    guac_src_path = '/home/vasissua/PYTHON/GUAC/src/'
    pfp_src_path  = '/home/vasissua/PYTHON/pfh_python/gizmopy/'
else:
    guac_src_path = '/home/vasilii/research/trillium/home/PYTHON/GUAC/src/'
    pfp_src_path  = '/home/vasilii/research/trillium/home/PYTHON/pfh_python/gizmopy/'

sys.path.insert(0, guac_src_path)
sys.path.insert(0, pfp_src_path)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from notebooks.make_disk_movie_frames import identify_disk
from generic_utils.constants import kpc, AU, Msun
from hybrid_sims_utils.read_snap import get_snap_data_hybrid, convert_units_to_physical


# ── Constants / bin setup ────────────────────────────────────────────────────
N_R       = 50
R_MIN_AU  = 0.1
R_MAX_AU  = 1e4

_r_min_kpc = R_MIN_AU * AU / kpc
_r_max_kpc = R_MAX_AU * AU / kpc
SPH_EDGES  = np.logspace(np.log10(_r_min_kpc), np.log10(_r_max_kpc), N_R + 1)
SPH_CTR_AU = np.sqrt(SPH_EDGES[:-1] * SPH_EDGES[1:]) * kpc / AU
SPH_EDGES_AU = SPH_EDGES * kpc / AU


# ── Cosmological time ────────────────────────────────────────────────────────
try:
    from generic_utils.cosmology import convert_scale_factor_to_time as _csftt
    def scale_to_Myr(a):
        try:
            return float(_csftt(a)) * 1e3   # Gyr → Myr
        except Exception:
            return _fallback_time(a)
except ImportError:
    def scale_to_Myr(a):
        return _fallback_time(a)

def _fallback_time(a, H0=70.4, Om=0.2726, OL=0.728):
    integrand = lambda ap: 1.0 / (ap * np.sqrt(Om / ap**3 + OL))
    t_Gyr, _ = quad(integrand, 1e-6, float(a))
    return t_Gyr / (H0 * 1.022e-3) * 1e3   # Myr


# ── Disk identification args ──────────────────────────────────────────────────
import types as _types
_DISK_ARGS = _types.SimpleNamespace(
    r_search  = 1e-5,
    r_max     = 1e-5,
    rho_thresh= 1e-15,
    aspect    = 0.3,
    f_kep     = 0.3,
    corotate  = True,
    vmax_vel  = None,
    min_gas_particles = 0,
)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutout-dir', default=CUTOUT_DIR)
    parser.add_argument('--frames-dir', default=FRAMES_DIR)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    out_path = os.path.join(args.frames_dir, 'mass_evolution.npz')
    if os.path.exists(out_path) and not args.overwrite:
        print(f"Output already exists: {out_path}")
        print("Use --overwrite to recompute.")
        return

    os.makedirs(args.frames_dir, exist_ok=True)

    snap_paths = sorted(glob.glob(os.path.join(args.cutout_dir, 'snapshot_*.hdf5')))
    if not snap_paths:
        sys.exit(f'No snapshots found in {args.cutout_dir}')
    print(f'Found {len(snap_paths)} snapshots in {args.cutout_dir}')

    def _snap_num(p):
        return int(os.path.basename(p).replace('snapshot_', '').replace('.hdf5', ''))

    snap_items = [(p, _snap_num(p)) for p in snap_paths]

    gas_fields = ['Masses', 'Coordinates', 'SmoothingLength',
                  'Velocities', 'Density']

    sim_name   = os.path.basename(args.cutout_dir)
    sim_path   = os.path.dirname(args.cutout_dir) + '/'  # GUAC does path+sim (no sep)

    times_Myr    = []
    snap_num_list= []
    M_enc_list   = []
    M_disk_list  = []
    M_star_list  = []
    n_sinks_list = []
    t1_Myr = None

    for idx, (snap_path, snap_num) in enumerate(snap_items):
        try:
            hdr, pdata, stardata, fsd, _, _ = get_snap_data_hybrid(
                sim_name, sim_path, snap_num,
                snapshot_suffix='', snapdir=False,
                refinement_tag=False, verbose=False,
                custom_gas_fields=gas_fields)
            hdr, pdata, stardata, fsd = convert_units_to_physical(
                hdr, pdata, stardata, fsd)
        except Exception as e:
            print(f'  snap {snap_num:04d}: load error — {e}')
            continue

        if 'Masses' not in pdata or 'Coordinates' not in pdata:
            continue

        t = scale_to_Myr(float(hdr['Time']))

        # First sink formation time
        try:
            with h5py.File(snap_path, 'r') as f:
                if ('PartType5' in f
                        and 'StellarFormationTime' in f['PartType5']
                        and f['PartType5/StellarFormationTime'].shape[0] > 0):
                    a_form = float(f['PartType5/StellarFormationTime'][:].min())
                    t_form = scale_to_Myr(a_form)
                    if t1_Myr is None or t_form < t1_Myr:
                        t1_Myr = t_form
        except Exception:
            pass

        # ── Center: most massive sink (fallback gas median) ──
        if (stardata is not None and 'Masses' in stardata
                and len(stardata['Masses']) > 0):
            center = stardata['Coordinates'][np.argmax(stardata['Masses'])]
        else:
            center = np.median(pdata['Coordinates'], axis=0)

        # ── Cumulative enclosed mass M(<r) = gas + sinks ──
        coords_rel     = pdata['Coordinates'] - center
        r3d_gas        = np.linalg.norm(coords_rel, axis=1)   # physical kpc
        gas_masses_Msun = pdata['Masses'] * 1e10               # Msun (gas only)

        # Combine gas + sink radii/masses for M_enc
        r3d_all        = r3d_gas.copy()
        m_all_Msun     = gas_masses_Msun.copy()
        if (stardata is not None and 'Masses' in stardata
                and 'Coordinates' in stardata
                and len(stardata['Masses']) > 0):
            r_sink   = np.linalg.norm(stardata['Coordinates'] - center, axis=1)
            m_sink   = stardata['Masses'] * 1e10
            r3d_all  = np.concatenate([r3d_all, r_sink])
            m_all_Msun = np.concatenate([m_all_Msun, m_sink])

        # Bin particles and compute cumulative M_enc at each bin edge
        M_enc = np.zeros(N_R)
        for b in range(N_R):
            # All particles (gas+sinks) within the outer edge of bin b
            M_enc[b] = m_all_Msun[r3d_all < SPH_EDGES[b + 1]].sum()

        # ── Disk mass via identify_disk (gas only — is_disk has len = n_gas) ──
        M_disk = 0.0
        try:
            is_disk, _, _, _, _, _, _, _ = identify_disk(
                pdata, stardata,
                r_search_kpc      = _DISK_ARGS.r_search,
                r_max_kpc         = _DISK_ARGS.r_max,
                rho_threshold_cgs = _DISK_ARGS.rho_thresh,
                aspect_ratio      = _DISK_ARGS.aspect,
                f_kep             = _DISK_ARGS.f_kep,
            )
            M_disk = gas_masses_Msun[is_disk].sum()  # gas only, matching is_disk length
        except Exception:
            pass   # pre-sink snapshots may have no disk

        # ── Sink totals ──
        M_star  = 0.0
        n_sinks = 0
        if (stardata is not None and 'Masses' in stardata
                and len(stardata['Masses']) > 0):
            M_star  = stardata['Masses'].sum() * 1e10
            n_sinks = len(stardata['Masses'])

        times_Myr.append(t)
        snap_num_list.append(snap_num)
        M_enc_list.append(M_enc)
        M_disk_list.append(M_disk)
        M_star_list.append(M_star)
        n_sinks_list.append(n_sinks)

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f'  snap {snap_num:04d}  t={t*1e3:.2f} kyr'
                  f'  M_enc_max={M_enc[-1]:.2f} Msun'
                  f'  M_disk={M_disk:.3f} Msun'
                  f'  n_sinks={n_sinks}'
                  f'  [{idx+1}/{len(snap_items)}]', flush=True)

    if not times_Myr:
        sys.exit('No snapshots processed.')

    from scipy.ndimage import gaussian_filter1d as _gf1d

    times_arr = np.array(times_Myr)
    M_enc_arr = np.array(M_enc_list)   # (N, N_r)

    # Smooth M_enc along the radial axis (sigma=1 bin) before differencing.
    # This prevents point-mass sinks crossing a single bin from creating
    # large M_shell spikes without affecting the broad shape.
    M_enc_sm = _gf1d(M_enc_arr, sigma=1.0, axis=1)
    # Enforce monotonicity (M_enc must be non-decreasing in r)
    M_enc_sm = np.maximum.accumulate(M_enc_sm, axis=1)

    # Differential shell mass = M_enc[b] - M_enc[b-1]
    M_shell_arr = np.diff(M_enc_sm, prepend=0.0, axis=1)   # (N, N_r)
    M_shell_arr = np.maximum(M_shell_arr, 0.0)              # shells are non-negative

    np.savez(out_path,
             times_Myr  = times_arr,
             snap_nums  = np.array(snap_num_list),
             t1_Myr     = np.array([t1_Myr if t1_Myr is not None else np.nan]),
             r_AU       = SPH_CTR_AU,
             r_edges_AU = SPH_EDGES_AU,
             M_enc      = M_enc_arr,
             M_shell    = M_shell_arr,
             M_disk     = np.array(M_disk_list),
             M_star     = np.array(M_star_list),
             n_sinks    = np.array(n_sinks_list))

    print(f'\nSaved {len(times_arr)} snapshots → {out_path}')
    print(f'  M_enc shape: {M_enc_arr.shape}')
    if t1_Myr is not None:
        print(f'  t1 = {t1_Myr*1e3:.3f} kyr')


if __name__ == '__main__':
    main()
