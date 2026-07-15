"""
compute_phase_stats.py
======================
Scan all cutout snapshots and save per-bin T and f_H2 statistics vs time.

For each snapshot, bins gas particles within the inner disk region
(r < 1.5e-5 kpc ≈ 450 AU of the most massive sink) into log-spaced
density bins and computes median + 16th/84th percentiles of T and f_H2.

Output: {frames_dir}/phase_stats.npz
  times_Myr    (N,)          absolute time [Myr]
  snap_nums    (N,)          snapshot numbers
  t1_Myr       scalar        first sink formation time [Myr]
  n_edges      (N_bins+1,)   log10(n [cm^-3]) bin edges
  n_ctr        (N_bins,)     log10(n) bin centres
  T_med        (N, N_bins)   median log10(T [K]) per bin
  T_p16        (N, N_bins)   16th percentile log10(T)
  T_p84        (N, N_bins)   84th percentile log10(T)
  fH2_med      (N, N_bins)   median log10(f_H2) per bin
  fH2_p16      (N, N_bins)   16th percentile log10(f_H2)
  fH2_p84      (N, N_bins)   84th percentile log10(f_H2)

Usage (cluster):
    python disk_analysis/compute_phase_stats.py [--overwrite]
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

from generic_utils.constants import kpc, AU, Msun
from hybrid_sims_utils.read_snap import get_snap_data_hybrid, convert_units_to_physical


# ── Physical constants ───────────────────────────────────────────────────────
m_p   = 1.6726e-24   # g (proton mass)
k_B   = 1.3806e-16   # erg/K
gamma = 5.0 / 3.0
mu    = 1.22          # mean molecular weight (neutral primordial gas)
km_s  = 1e5           # cm/s


# ── Bin setup ────────────────────────────────────────────────────────────────
N_BINS   = 60
LOG_N_MIN = -2.0    # log10(n [cm^-3])
LOG_N_MAX =  22.0
N_EDGES  = np.linspace(LOG_N_MIN, LOG_N_MAX, N_BINS + 1)
N_CTR    = 0.5 * (N_EDGES[:-1] + N_EDGES[1:])

# Inner disk region radius [kpc] — same as cut_small = 0.75 * image_box
R_CUT_KPC = 1.5e-5  # ≈ 450 AU


# ── Cosmological time ────────────────────────────────────────────────────────
try:
    from generic_utils.cosmology import convert_scale_factor_to_time as _csftt
    def scale_to_Myr(a):
        try:
            return float(_csftt(a)) * 1e3
        except Exception:
            return _fallback_time(a)
except ImportError:
    def scale_to_Myr(a):
        return _fallback_time(a)

def _fallback_time(a, H0=70.4, Om=0.2726, OL=0.728):
    integrand = lambda ap: 1.0 / (ap * np.sqrt(Om / ap**3 + OL))
    t_Gyr, _ = quad(integrand, 1e-6, float(a))
    return t_Gyr / (H0 * 1.022e-3) * 1e3


# ── H2 field name detection ──────────────────────────────────────────────────
_FH2_FIELDS = ['MolecularMassFraction', 'Molecular_Fraction',
                'MolecularHydrogenFraction', 'H2Fraction']

def _load_fh2(snap_file, mask):
    """Try to load f_H2 from a snapshot; return None if not found."""
    with h5py.File(snap_file, 'r') as f:
        if 'PartType0' not in f:
            return None
        for key in _FH2_FIELDS:
            if key in f['PartType0']:
                return f['PartType0'][key][:][mask]
    return None


# ── Per-snapshot statistics ──────────────────────────────────────────────────
def _bin_stats(log_n, log_vals):
    """For each density bin, return (median, p16, p84) of log_vals."""
    med = np.full(N_BINS, np.nan)
    p16 = np.full(N_BINS, np.nan)
    p84 = np.full(N_BINS, np.nan)
    idx = np.digitize(log_n, N_EDGES) - 1
    for b in range(N_BINS):
        m = (idx == b)
        if m.sum() >= 5:
            v = log_vals[m]
            med[b] = np.percentile(v, 50)
            p16[b] = np.percentile(v, 16)
            p84[b] = np.percentile(v, 84)
    return med, p16, p84


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cutout-dir', default=CUTOUT_DIR)
    parser.add_argument('--frames-dir', default=FRAMES_DIR)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    out_path = os.path.join(args.frames_dir, 'phase_stats.npz')
    if os.path.exists(out_path) and not args.overwrite:
        print(f'Output already exists: {out_path}')
        print('Use --overwrite to recompute.')
        return

    os.makedirs(args.frames_dir, exist_ok=True)

    snap_paths = sorted(glob.glob(os.path.join(args.cutout_dir, 'snapshot_*.hdf5')))
    if not snap_paths:
        sys.exit(f'No snapshots found in {args.cutout_dir}')
    print(f'Found {len(snap_paths)} snapshots in {args.cutout_dir}')

    def _snap_num(p):
        return int(os.path.basename(p).replace('snapshot_', '').replace('.hdf5', ''))

    snap_items = [(p, _snap_num(p)) for p in snap_paths]

    sim_name = os.path.basename(args.cutout_dir)
    sim_path = os.path.dirname(args.cutout_dir) + '/'

    gas_fields = ['Masses', 'Coordinates', 'Density', 'InternalEnergy']

    times_Myr  = []
    snap_nums  = []
    T_med_list  = []; T_p16_list  = []; T_p84_list  = []
    fH2_med_list= []; fH2_p16_list= []; fH2_p84_list= []
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

        if 'Density' not in pdata or 'InternalEnergy' not in pdata:
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

        # ── Inner region cut (same as cut_small in extract_epoch_data) ──
        r3d = np.linalg.norm(pdata['Coordinates'] - center, axis=1)
        cut = r3d < R_CUT_KPC

        if cut.sum() < 20:
            # Store NaN row — snapshot exists but no inner gas
            times_Myr.append(t)
            snap_nums.append(snap_num)
            T_med_list.append(np.full(N_BINS, np.nan))
            T_p16_list.append(np.full(N_BINS, np.nan))
            T_p84_list.append(np.full(N_BINS, np.nan))
            fH2_med_list.append(np.full(N_BINS, np.nan))
            fH2_p16_list.append(np.full(N_BINS, np.nan))
            fH2_p84_list.append(np.full(N_BINS, np.nan))
            continue

        # ── Physical quantities ──
        # Density: after convert_units_to_physical, in 10^10 Msun/kpc^3
        rho_cgs = pdata['Density'][cut] * (Msun * 1e10) / kpc**3   # g/cm^3
        n_cgs   = rho_cgs / (mu * m_p)                              # cm^-3

        # Temperature: InternalEnergy in (km/s)^2 after unit conversion
        u_cgs = pdata['InternalEnergy'][cut] * km_s**2              # erg/g
        T_K   = mu * m_p / k_B * (gamma - 1.0) * u_cgs             # K

        # Valid particles
        valid = (n_cgs > 0) & (T_K > 0) & np.isfinite(T_K)
        if valid.sum() < 10:
            times_Myr.append(t); snap_nums.append(snap_num)
            for lst in (T_med_list, T_p16_list, T_p84_list,
                        fH2_med_list, fH2_p16_list, fH2_p84_list):
                lst.append(np.full(N_BINS, np.nan))
            continue

        log_n = np.log10(n_cgs[valid])
        log_T = np.log10(T_K[valid])

        T_med, T_p16, T_p84 = _bin_stats(log_n, log_T)

        # ── f_H2 ──
        fh2 = _load_fh2(snap_path, cut)
        if fh2 is not None:
            vf = valid & (fh2 > 0) & np.isfinite(fh2)
            if vf.sum() >= 10:
                fH2_med, fH2_p16, fH2_p84 = _bin_stats(
                    log_n[vf[valid]], np.log10(fh2[vf]))
            else:
                fH2_med = fH2_p16 = fH2_p84 = np.full(N_BINS, np.nan)
        else:
            fH2_med = fH2_p16 = fH2_p84 = np.full(N_BINS, np.nan)

        times_Myr.append(t); snap_nums.append(snap_num)
        T_med_list.append(T_med);   T_p16_list.append(T_p16);   T_p84_list.append(T_p84)
        fH2_med_list.append(fH2_med); fH2_p16_list.append(fH2_p16); fH2_p84_list.append(fH2_p84)

        if (idx + 1) % 50 == 0 or idx == 0:
            n_valid = np.isfinite(T_med).sum()
            print(f'  snap {snap_num:04d}  t={t*1e3:.2f} kyr'
                  f'  cut={cut.sum()}  valid_bins={n_valid}'
                  f'  [{idx+1}/{len(snap_items)}]', flush=True)

    if not times_Myr:
        sys.exit('No snapshots processed.')

    np.savez(out_path,
             times_Myr = np.array(times_Myr),
             snap_nums  = np.array(snap_nums),
             t1_Myr     = np.array([t1_Myr if t1_Myr is not None else np.nan]),
             n_edges    = N_EDGES,
             n_ctr      = N_CTR,
             T_med      = np.array(T_med_list),
             T_p16      = np.array(T_p16_list),
             T_p84      = np.array(T_p84_list),
             fH2_med    = np.array(fH2_med_list),
             fH2_p16    = np.array(fH2_p16_list),
             fH2_p84    = np.array(fH2_p84_list))

    print(f'\nSaved {len(times_Myr)} snapshots → {out_path}')
    print(f'  t1_Myr = {t1_Myr}')
    print(f'  n_bins = {N_BINS}, log10(n) in [{LOG_N_MIN}, {LOG_N_MAX}]')


if __name__ == '__main__':
    main()
