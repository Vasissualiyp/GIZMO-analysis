"""
plot_mass_evolution.py
----------------------
Plot stellar mass, disk gas mass, star formation efficiency, and total
accretion rate vs time from the full FIRE+STARFORGE simulation.

Disk gas is identified with the same identify_disk() logic used by the
movie pipeline — no cutout files are used.

Usage:
    python disk_analysis/plot_mass_evolution.py [--path PATH] [--sim SIM] ...
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

guac_src_path = "/home/vasissua/PYTHON/GUAC/src/"
pfp_src_path = "/home/vasissua/PYTHON/pfh_python/gizmopy/"
sys.path.insert(0, guac_src_path)
sys.path.insert(0, pfp_src_path)

# ── Path setup: add analysis root so we can import from notebooks/ ────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from notebooks.make_disk_movie_frames import identify_disk
from generic_utils.constants import kpc, AU, Msun, G
from hybrid_sims_utils.read_snap import get_snap_data_hybrid, convert_units_to_physical

try:
    from astropy.cosmology import Planck18 as cosmo
    import astropy.units as u_astropy
    def scale_to_Myr(a):
        return float(cosmo.age(1.0 / float(a) - 1.0).to(u_astropy.Myr).value)
except ImportError:
    print('WARNING: astropy not found; time axis will be scale factor, not Myr.')
    def scale_to_Myr(a):
        return float(a)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--path',       default='/scratch/vasissua/COPY/2026-03/m12f_cutout/')
    p.add_argument('--sim',        default='output_jeans_refinement')
    p.add_argument('--outdir',     default='/scratch/vasissua/SHIVAN/analysis/plots/')
    p.add_argument('--r-search',   type=float, default=1e-5)
    p.add_argument('--r-max',      type=float, default=1e-5)
    p.add_argument('--rho-thresh', type=float, default=1e-15)
    p.add_argument('--aspect',     type=float, default=0.3)
    p.add_argument('--f-kep',      type=float, default=0.3)
    p.add_argument('--aperture',   type=float, default=None,
                   help='Fixed spherical aperture radius [kpc] for gas mass '
                        '(default: 5 × r-max)')
    p.add_argument('--snap-start', type=int,   default=None)
    p.add_argument('--snap-end',   type=int,   default=None)
    args = p.parse_args()

    snap_pattern = os.path.join(args.path, args.sim, 'snapshot_*.hdf5')
    snap_paths   = sorted(glob.glob(snap_pattern))
    if not snap_paths:
        sys.exit(f'No snapshots found: {snap_pattern}')

    def _snap_num(path):
        return int(os.path.basename(path).replace('snapshot_', '').replace('.hdf5', ''))

    snap_items = [(sp, _snap_num(sp)) for sp in snap_paths]
    if args.snap_start is not None:
        snap_items = [(sp, n) for sp, n in snap_items if n >= args.snap_start]
    if args.snap_end is not None:
        snap_items = [(sp, n) for sp, n in snap_items if n <= args.snap_end]

    aperture_kpc = args.aperture if args.aperture is not None else 5.0 * args.r_max
    print(f'Processing {len(snap_items)} snapshots from {args.path}{args.sim}/')
    print(f'Fixed aperture radius: {aperture_kpc*1e3:.2f} pc  '
          f'({aperture_kpc/args.r_max:.0f}× r_max)')

    gas_fields = ['Masses', 'Coordinates', 'SmoothingLength', 'Velocities', 'Density']

    times_Myr   = []
    M_disk_arr  = []   # disk gas mass [Msun]
    M_apert_arr = []   # gas in fixed aperture [Msun]
    M_star_arr  = []   # all sink masses [Msun]
    t1_Myr      = None

    for i, (snap_path, snap_num) in enumerate(snap_items):
        try:
            hdr, pdata, stardata, fsd, _, _ = get_snap_data_hybrid(
                args.sim, args.path, snap_num,
                snapshot_suffix='', snapdir=False,
                refinement_tag=False, verbose=False,
                custom_gas_fields=gas_fields)
            hdr, pdata, stardata, fsd = convert_units_to_physical(hdr, pdata, stardata, fsd)
        except Exception as e:
            print(f'  snap {snap_num:04d}: load error — {e}')
            continue

        t = scale_to_Myr(float(hdr['Time']))

        # ── Sink mass (all sinks in full sim) ─────────────────────────────────
        if stardata and len(stardata.get('Masses', [])) > 0:
            m_star = float(np.sum(stardata['Masses'])) * 1e10
        else:
            m_star = 0.0

        # StellarFormationTime is not always exposed by GUAC — read via h5py
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

        # ── Disk gas mass via identify_disk + fixed-aperture gas mass ─────────
        try:
            is_disk, com, *_ = identify_disk(
                pdata, stardata,
                r_search_kpc      = args.r_search,
                r_max_kpc         = args.r_max,
                rho_threshold_cgs = args.rho_thresh,
                aspect_ratio      = args.aspect,
                f_kep             = args.f_kep,
            )
            m_disk = float(np.sum(pdata['Masses'][is_disk])) * 1e10

            dists_from_com = np.linalg.norm(pdata['Coordinates'] - com, axis=1)
            m_apert = float(np.sum(pdata['Masses'][dists_from_com < aperture_kpc])) * 1e10
        except Exception as e:
            print(f'  snap {snap_num:04d}: identify_disk error — {e}')
            m_disk  = 0.0
            m_apert = 0.0

        times_Myr.append(t)
        M_disk_arr.append(m_disk)
        M_apert_arr.append(m_apert)
        M_star_arr.append(m_star)
        print(f'  snap {snap_num:04d}  t={t:.4f} Myr  '
              f'M_disk={m_disk:.3f}  M_apert={m_apert:.3f}  M_star={m_star:.3f} Msun  '
              f'[{i+1}/{len(snap_items)}]', flush=True)

    if not times_Myr:
        sys.exit('No snapshots processed successfully.')

    times_Myr   = np.array(times_Myr)
    M_disk_arr  = np.array(M_disk_arr)
    M_apert_arr = np.array(M_apert_arr)
    M_star_arr  = np.array(M_star_arr)

    M_tot_disk  = M_disk_arr  + M_star_arr
    M_tot_apert = M_apert_arr + M_star_arr
    f_star_disk  = np.where(M_tot_disk  > 0, M_star_arr / M_tot_disk,  0.0)
    f_star_apert = np.where(M_tot_apert > 0, M_star_arr / M_tot_apert, 0.0)

    if t1_Myr is not None:
        t_plot = times_Myr - t1_Myr
        xlabel = r'$t - t_1$ (Myr)   [$t_1$ = first sink formation]'
        print(f'\nFirst sink at t_1 = {t1_Myr:.4f} Myr')
    else:
        t_plot = times_Myr
        xlabel = 'Time (Myr)'

    os.makedirs(args.outdir, exist_ok=True)

    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.patch.set_facecolor('k')
    ax1, ax2, ax3 = axes

    apert_label = rf'$r < {aperture_kpc*1e3:.1f}\ \rm pc$ aperture'

    _floor = 1e-3
    # Panel 1: gas masses + stellar mass
    ax1.semilogy(t_plot, np.maximum(M_star_arr,  _floor), 'y-',  lw=2,
                 label=r'$M_*$ (all sinks)')
    ax1.semilogy(t_plot, np.maximum(M_disk_arr,  _floor), 'c-',  lw=2,
                 label=r'$M_{\rm gas}$ (identify\_disk)')
    ax1.semilogy(t_plot, np.maximum(M_apert_arr, _floor), 'c--', lw=1.5,
                 label=r'$M_{\rm gas}$ (' + apert_label + ')')
    ax1.semilogy(t_plot, np.maximum(M_tot_disk,  _floor), 'w-',  lw=1, alpha=0.5,
                 label=r'$M_{\rm disk}+M_*$')
    ax1.semilogy(t_plot, np.maximum(M_tot_apert, _floor), 'w--', lw=1, alpha=0.5,
                 label=r'$M_{\rm apert}+M_*$')
    ax1.set_ylabel(r'Mass ($M_\odot$)', color='w')
    ax1.set_title('Mass evolution', color='w')
    leg = ax1.legend(fontsize=8, framealpha=0.3, ncol=2)
    for txt in leg.get_texts():
        txt.set_color('w')
    ax1.tick_params(colors='w', which='both', direction='in', right=True, top=True)
    for sp in ax1.spines.values():
        sp.set_edgecolor('w')

    # Panel 2: SFE — disk vs aperture denominator
    ax2.plot(t_plot, f_star_disk  * 100, 'g-',  lw=2,
             label=r'$M_* / (M_{\rm disk}+M_*)$  [disk only]')
    ax2.plot(t_plot, f_star_apert * 100, 'g--', lw=1.5,
             label=r'$M_* / (M_{\rm apert}+M_*)$  [' + apert_label + ']')
    ax2.set_ylabel(r'$f_*$  (%)', color='w')
    ax2.set_title('Star formation efficiency', color='w')
    leg2 = ax2.legend(fontsize=8, framealpha=0.3)
    for txt in leg2.get_texts():
        txt.set_color('w')
    ax2.tick_params(colors='w', which='both', direction='in', right=True, top=True)
    for sp in ax2.spines.values():
        sp.set_edgecolor('w')

    # Panel 3: stellar accretion rate + gas depletion rate from aperture
    if len(t_plot) > 1:
        dt_yr  = np.diff(times_Myr) * 1e6
        t_mid  = 0.5 * (t_plot[:-1] + t_plot[1:])

        dMstar_dt = np.diff(M_star_arr)  / np.maximum(dt_yr, 1.0)
        dMapert_dt = np.diff(M_apert_arr) / np.maximum(dt_yr, 1.0)
        # Gas depletion rate = -dM_gas/dt  (positive when gas is being consumed)
        gas_depletion = -dMapert_dt

        pos_star  = dMstar_dt   > 0
        pos_depl  = gas_depletion > 0
        if pos_star.any():
            ax3.semilogy(t_mid[pos_star], dMstar_dt[pos_star],      'm-',  lw=1.5,
                         label=r'$\dot{M}_*$ (stellar accretion)')
        if pos_depl.any():
            ax3.semilogy(t_mid[pos_depl], gas_depletion[pos_depl],  'c--', lw=1.5,
                         label=r'$-\dot{M}_{\rm gas,apert}$ (gas depletion rate)')
        ax3.set_ylabel(r'Rate  ($M_\odot$/yr)', color='w')
        ax3.set_title('Accretion & gas depletion rates', color='w')
        leg3 = ax3.legend(fontsize=8, framealpha=0.3)
        for txt in leg3.get_texts():
            txt.set_color('w')
        ax3.tick_params(colors='w', which='both', direction='in', right=True, top=True)
        for sp in ax3.spines.values():
            sp.set_edgecolor('w')

    axes[-1].set_xlabel(xlabel, color='w')
    plt.tight_layout()

    outpath = os.path.join(args.outdir, 'mass_evolution.png')
    fig.savefig(outpath, dpi=150, facecolor='k')
    plt.close(fig)
    print(f'\nSaved → {outpath}')


if __name__ == '__main__':
    main()
