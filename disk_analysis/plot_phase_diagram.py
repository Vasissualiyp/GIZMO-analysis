"""
plot_phase_diagram.py
---------------------
Two-panel phase diagram for each FIRE+STARFORGE snapshot:
  Left:  T vs ρ — 2D mass-weighted histogram with disk-particle contours
  Right: H2 fraction vs ρ — 2D mass-weighted histogram with disk contours

Output directory: {outdir}/T_H2_rho_phase_plots/

Usage (single snapshot):
    python disk_analysis/plot_phase_diagram.py --snap-num 271

Usage (all snapshots):
    python disk_analysis/plot_phase_diagram.py --all-snaps

Callable from other scripts via plot_all_phase_diagrams(args).
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors

guac_src_path = "/home/vasissua/PYTHON/GUAC/src/"
pfp_src_path  = "/home/vasissua/PYTHON/pfh_python/gizmopy/"
sys.path.insert(0, guac_src_path)
sys.path.insert(0, pfp_src_path)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from notebooks.make_disk_movie_frames import identify_disk
from generic_utils.constants import kpc, AU, Msun, G
from hybrid_sims_utils.read_snap import get_snap_data_hybrid, convert_units_to_physical

# ── Physical constants ────────────────────────────────────────────────────────
_GAMMA = 5.0 / 3.0
_kB    = 1.381e-16   # erg/K
_mp    = 1.673e-24   # g
_mu    = 1.0         # mean molecular weight (primordial neutral H approximation)

# ── Cosmological time conversion (module-level so all functions can use it) ──
try:
    from astropy.cosmology import Planck18 as _cosmo
    import astropy.units as _u_astropy
    def scale_to_Myr(a):
        return float(_cosmo.age(1.0 / float(a) - 1.0).to(_u_astropy.Myr).value)
except ImportError:
    print('WARNING: astropy not found; time axis will be scale factor, not Myr.')
    def scale_to_Myr(a):
        return float(a)

# H2 field names to try, in order of preference
_H2_FIELD_CANDIDATES = [
    'MolecularMassFraction',
    'Molecular_Fraction',
    'MolecularHydrogenFraction',
    'H2Fraction',
]


def internal_energy_to_T(u_kms2):
    """u [km/s]²  →  temperature [K], ideal gas γ=5/3."""
    return (_GAMMA - 1.0) * u_kms2 * 1e10 * (_mp * _mu / _kB)


def detect_h2_field(snap_path):
    """Return the name of the H2 field present in the snapshot, or None."""
    try:
        with h5py.File(snap_path, 'r') as f:
            if 'PartType0' not in f:
                return None
            for name in _H2_FIELD_CANDIDATES:
                if name in f['PartType0']:
                    return name
    except Exception:
        pass
    return None


def load_snap(snap_num, args, h2_field=None):
    """Load gas+sink data for one snapshot. Returns None on failure."""
    gas_fields = ['Masses', 'Coordinates', 'SmoothingLength',
                  'Velocities', 'Density', 'InternalEnergy']
    if h2_field is not None:
        gas_fields.append(h2_field)

    try:
        hdr, pdata, stardata, fsd, _, _ = get_snap_data_hybrid(
            args.sim, args.path, snap_num,
            snapshot_suffix='', snapdir=False,
            refinement_tag=False, verbose=False,
            custom_gas_fields=gas_fields)
        hdr, pdata, stardata, fsd = convert_units_to_physical(hdr, pdata, stardata, fsd)
    except Exception as e:
        print(f'  snap {snap_num:04d}: load error — {e}')
        return None

    return hdr, pdata, stardata


def process_snap(snap_num, args, h2_field=None):
    """Return arrays needed for both panels, or None on failure."""
    result = load_snap(snap_num, args, h2_field)
    if result is None:
        return None
    hdr, pdata, stardata = result

    time_Myr = scale_to_Myr(float(hdr['Time']))

    # Disk identification
    com = None
    try:
        is_disk, com, *_ = identify_disk(
            pdata, stardata,
            r_search_kpc      = args.r_search,
            r_max_kpc         = args.r_max,
            rho_threshold_cgs = args.rho_thresh,
            aspect_ratio      = args.aspect,
            f_kep             = args.f_kep,
        )
    except Exception as e:
        print(f'  snap {snap_num:04d}: identify_disk error — {e}')
        is_disk = np.zeros(len(pdata['Masses']), dtype=bool)

    # Restrict to local region (same pre-filter as identify_disk)
    r_local = max(args.r_max * 5, args.r_search * 2)
    origin  = com if com is not None else np.zeros(3)
    dists   = np.linalg.norm(pdata['Coordinates'] - origin, axis=1)
    local   = dists < r_local

    rho   = pdata['Density'][local].astype(np.float64) * 1e10 * Msun / kpc**3
    u     = pdata['InternalEnergy'][local].astype(np.float64)
    T     = internal_energy_to_T(u)
    mass  = pdata['Masses'][local].astype(np.float64) * 1e10
    disk  = is_disk[local]

    if h2_field is not None and h2_field in pdata:
        fh2 = pdata[h2_field][local].astype(np.float64)
    else:
        fh2 = None

    return rho, T, mass, disk, fh2, time_Myr


def plot_snap(snap_num, rho, T, mass, disk, fh2,
              time_Myr, outpath, rho_thresh=None, t1_Myr=None):
    """Two-panel figure: T vs ρ  |  log(f_H2) vs ρ."""
    n_bins = 200

    rho_lim     = (1e-25, 1e-10)
    T_lim       = (1e1,   1e6)
    log_fh2_lim = (-6.0,  0.0)   # log10(f_H2) range

    log_rho_edges = np.linspace(np.log10(rho_lim[0]), np.log10(rho_lim[1]), n_bins + 1)
    log_T_edges   = np.linspace(np.log10(T_lim[0]),   np.log10(T_lim[1]),   n_bins + 1)
    log_fh2_edges = np.linspace(log_fh2_lim[0], log_fh2_lim[1], n_bins + 1)

    if t1_Myr is not None:
        time_label = rf'$t - t_1 = {(time_Myr - t1_Myr)*1e3:.2f}$ kyr'
    else:
        time_label = rf'$t = {time_Myr*1e3:.2f}$ kyr'

    valid = (rho > 0) & (T > 0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('k')

    _tick_kw = dict(colors='w', which='both', direction='in', right=True, top=True)

    # ── Left panel: T vs ρ ────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor('k')

    H_all, _, _ = np.histogram2d(
        np.log10(rho[valid]), np.log10(T[valid]),
        bins=[log_rho_edges, log_T_edges],
        weights=mass[valid])
    H_all = np.where(H_all > 0, H_all, np.nan)

    vmin_h = np.nanpercentile(H_all[H_all > 0], 5) if np.any(H_all > 0) else 1e-3
    im = ax.pcolormesh(
        log_rho_edges, log_T_edges, H_all.T,
        norm=colors.LogNorm(vmin=vmin_h, vmax=np.nanmax(H_all)),
        cmap='inferno', rasterized=True)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label(r'Mass per bin ($M_\odot$)', color='w', fontsize=10)
    cb.ax.yaxis.set_tick_params(color='w')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='w')

    # Disk contours
    if disk.sum() > 0 and (valid & disk).sum() > 0:
        H_disk, _, _ = np.histogram2d(
            np.log10(rho[valid & disk]), np.log10(T[valid & disk]),
            bins=[log_rho_edges, log_T_edges],
            weights=mass[valid & disk])
        log_rho_ctr = 0.5 * (log_rho_edges[:-1] + log_rho_edges[1:])
        log_T_ctr   = 0.5 * (log_T_edges[:-1]   + log_T_edges[1:])
        try:
            ax.contour(log_rho_ctr, log_T_ctr, H_disk.T,
                       levels=4, colors='cyan', linewidths=0.8, alpha=0.85)
        except Exception:
            pass
        ax.scatter([], [], c='cyan', s=10, label='disk particles (contour)')

    if rho_thresh is not None:
        ax.axvline(np.log10(rho_thresh), color='r', ls='--', lw=1.5,
                   label=rf'$\rho_\mathrm{{thresh}}$')

    ax.set_xlabel(r'$\log_{10}\ \rho\ \mathrm{(g/cm^3)}$', color='w', fontsize=12)
    ax.set_ylabel(r'$\log_{10}\ T\ \mathrm{(K)}$',          color='w', fontsize=12)
    ax.set_title(r'$T$ vs $\rho$', color='w', fontsize=13)
    ax.tick_params(**_tick_kw)
    for sp in ax.spines.values(): sp.set_edgecolor('w')
    leg = ax.legend(fontsize=9, framealpha=0.3)
    for t in leg.get_texts(): t.set_color('w')

    # ── Right panel: log(f_H2) vs ρ ──────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('k')

    if fh2 is not None:
        valid2 = valid & (fh2 > 0)
        log_fh2 = np.log10(np.maximum(fh2, 1e-30))
        H_h2, _, _ = np.histogram2d(
            np.log10(rho[valid2]), log_fh2[valid2],
            bins=[log_rho_edges, log_fh2_edges],
            weights=mass[valid2])
        H_h2 = np.where(H_h2 > 0, H_h2, np.nan)

        vmin_h2 = np.nanpercentile(H_h2[H_h2 > 0], 5) if np.any(H_h2 > 0) else 1e-3
        im2 = ax2.pcolormesh(
            log_rho_edges, log_fh2_edges, H_h2.T,
            norm=colors.LogNorm(vmin=vmin_h2, vmax=np.nanmax(H_h2)),
            cmap='inferno', rasterized=True)
        cb2 = plt.colorbar(im2, ax=ax2)
        cb2.set_label(r'Mass per bin ($M_\odot$)', color='w', fontsize=10)
        cb2.ax.yaxis.set_tick_params(color='w')
        plt.setp(cb2.ax.yaxis.get_ticklabels(), color='w')

        # Disk contours
        if disk.sum() > 0 and (valid2 & disk).sum() > 0:
            H_disk2, _, _ = np.histogram2d(
                np.log10(rho[valid2 & disk]), log_fh2[valid2 & disk],
                bins=[log_rho_edges, log_fh2_edges],
                weights=mass[valid2 & disk])
            log_rho_ctr  = 0.5 * (log_rho_edges[:-1]  + log_rho_edges[1:])
            log_fh2_ctr  = 0.5 * (log_fh2_edges[:-1]  + log_fh2_edges[1:])
            try:
                ax2.contour(log_rho_ctr, log_fh2_ctr, H_disk2.T,
                            levels=4, colors='cyan', linewidths=0.8, alpha=0.85)
            except Exception:
                pass

        if rho_thresh is not None:
            ax2.axvline(np.log10(rho_thresh), color='r', ls='--', lw=1.5,
                        label=rf'$\rho_\mathrm{{thresh}}$')
            leg2 = ax2.legend(fontsize=9, framealpha=0.3)
            for t in leg2.get_texts(): t.set_color('w')

        ax2.set_ylabel(r'$\log_{10}\ f_{\rm H_2}$', color='w', fontsize=12)
    else:
        ax2.text(0.5, 0.5, 'H₂ field not found\nin snapshot',
                 ha='center', va='center', color='w', fontsize=14,
                 transform=ax2.transAxes)

    ax2.set_xlabel(r'$\log_{10}\ \rho\ \mathrm{(g/cm^3)}$', color='w', fontsize=12)
    ax2.set_title(r'$\log_{10}\ f_{\rm H_2}$ vs $\rho$', color='w', fontsize=13)
    ax2.tick_params(**_tick_kw)
    for sp in ax2.spines.values(): sp.set_edgecolor('w')

    fig.suptitle(f'Snap {snap_num:04d}   {time_label}', color='w', fontsize=12)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, facecolor='k')
    plt.close(fig)


def find_t1_Myr(snap_items):
    """Scan all snapshots (header-only) to find time of first sink formation."""
    t1 = None
    for snap_path, _ in snap_items:
        try:
            with h5py.File(snap_path, 'r') as f:
                if ('PartType5' in f
                        and 'StellarFormationTime' in f['PartType5']
                        and f['PartType5/StellarFormationTime'].shape[0] > 0):
                    a_form = float(f['PartType5/StellarFormationTime'][:].min())
                    t_form = scale_to_Myr(a_form)
                    if t1 is None or t_form < t1:
                        t1 = t_form
        except Exception:
            pass
    return t1


def plot_all_phase_diagrams(args):
    """
    Generate one phase-diagram PNG per snapshot.
    Called from jupytertest3.py after the main movie loop.
    `args` must have .path, .sim, .outdir and disk-id parameters.
    """
    snap_pattern = os.path.join(args.path, args.sim, 'snapshot_*.hdf5')
    snap_paths   = sorted(glob.glob(snap_pattern))
    if not snap_paths:
        print(f'plot_all_phase_diagrams: no snapshots found in {snap_pattern}')
        return

    def _sn(p): return int(os.path.basename(p).replace('snapshot_', '').replace('.hdf5', ''))
    snap_items = [(sp, _sn(sp)) for sp in snap_paths]
    if getattr(args, 'snap_start', None) is not None:
        snap_items = [(sp, n) for sp, n in snap_items if n >= args.snap_start]
    if getattr(args, 'snap_end',   None) is not None:
        snap_items = [(sp, n) for sp, n in snap_items if n <= args.snap_end]

    phase_dir = os.path.join(args.outdir, 'T_H2_rho_phase_plots')
    os.makedirs(phase_dir, exist_ok=True)

    # Detect H2 field from the last snapshot (most likely to have sinks + full fields)
    h2_field = detect_h2_field(snap_items[-1][0])
    if h2_field:
        print(f'  H2 field detected: {h2_field}')
    else:
        print('  No H2 field found — right panel will show placeholder.')

    print(f'  Finding t1 (first sink formation)...')
    t1_Myr = find_t1_Myr(snap_items)
    if t1_Myr is not None:
        print(f'  t1 = {t1_Myr*1e3:.2f} kyr')
    else:
        print('  No sinks found yet — using absolute time.')

    print(f'  Generating {len(snap_items)} phase diagrams → {phase_dir}/')
    for i, (snap_path, snap_num) in enumerate(snap_items):
        outpath = os.path.join(phase_dir, f'phase_{snap_num:04d}.png')
        if os.path.exists(outpath):
            continue
        result = process_snap(snap_num, args, h2_field)
        if result is None:
            continue
        rho, T, mass, disk, fh2, time_Myr = result
        plot_snap(snap_num, rho, T, mass, disk, fh2,
                  time_Myr=time_Myr, outpath=outpath,
                  rho_thresh=getattr(args, 'rho_thresh', 1e-15),
                  t1_Myr=t1_Myr)
        print(f'  snap {snap_num:04d}  [{i+1}/{len(snap_items)}]', flush=True)

    print(f'  Phase diagrams done → {phase_dir}/')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--path',       default='/scratch/vasissua/COPY/2026-03/m12f_cutout/')
    p.add_argument('--sim',        default='output_jeans_refinement')
    p.add_argument('--outdir',     default='/scratch/vasissua/SHIVAN/analysis/frames/')
    p.add_argument('--snap-num',   type=int, default=271,
                   help='Snapshot to plot (ignored with --all-snaps)')
    p.add_argument('--all-snaps',  action='store_true',
                   help='Generate one diagram per snapshot')
    p.add_argument('--snap-start', type=int, default=None)
    p.add_argument('--snap-end',   type=int, default=None)
    p.add_argument('--r-search',   type=float, default=1e-5)
    p.add_argument('--r-max',      type=float, default=1e-5)
    p.add_argument('--rho-thresh', type=float, default=1e-15)
    p.add_argument('--aspect',     type=float, default=0.3)
    p.add_argument('--f-kep',      type=float, default=0.3)
    args = p.parse_args()

    phase_dir = os.path.join(args.outdir, 'T_H2_rho_phase_plots')
    os.makedirs(phase_dir, exist_ok=True)

    snap_pattern = os.path.join(args.path, args.sim, 'snapshot_*.hdf5')
    all_snaps    = sorted(glob.glob(snap_pattern))
    def _sn(p_): return int(os.path.basename(p_).replace('snapshot_', '').replace('.hdf5', ''))

    if args.all_snaps:
        snap_items = [(sp, _sn(sp)) for sp in all_snaps]
        if args.snap_start is not None: snap_items = [(sp, n) for sp, n in snap_items if n >= args.snap_start]
        if args.snap_end   is not None: snap_items = [(sp, n) for sp, n in snap_items if n <= args.snap_end]
    else:
        # Find the path for the requested snapshot
        snap_map = {_sn(sp): sp for sp in all_snaps}
        if args.snap_num not in snap_map:
            sys.exit(f'Snapshot {args.snap_num} not found in {snap_pattern}')
        snap_items = [(snap_map[args.snap_num], args.snap_num)]

    # Detect H2 field
    h2_field = detect_h2_field(snap_items[-1][0])
    if h2_field:
        print(f'H2 field detected: {h2_field}')
    else:
        print('No H2 field found — right panel will show placeholder.')

    print('Finding t1 (first sink formation)...')
    t1_Myr = find_t1_Myr(snap_items)
    print(f't1 = {t1_Myr*1e3:.2f} kyr' if t1_Myr is not None else 'No sinks found.')

    for i, (snap_path, snap_num) in enumerate(snap_items):
        outpath = os.path.join(phase_dir, f'phase_{snap_num:04d}.png')
        if os.path.exists(outpath):
            print(f'  snap {snap_num:04d}: skipped (exists)')
            continue
        result = process_snap(snap_num, args, h2_field)
        if result is None:
            continue
        rho, T, mass, disk, fh2, time_Myr = result
        plot_snap(snap_num, rho, T, mass, disk, fh2,
                  time_Myr=time_Myr, outpath=outpath,
                  rho_thresh=args.rho_thresh, t1_Myr=t1_Myr)
        print(f'  snap {snap_num:04d}  [{i+1}/{len(snap_items)}]', flush=True)


if __name__ == '__main__':
    main()
