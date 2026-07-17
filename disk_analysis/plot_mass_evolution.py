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

plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 22,
    'axes.titlesize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'xtick.major.size': 8,
    'xtick.minor.size': 4,
    'ytick.major.size': 8,
    'ytick.minor.size': 4,
    'xtick.major.width': 2.4,
    'xtick.minor.width': 1.6,
    'ytick.major.width': 2.4,
    'ytick.minor.width': 1.6,
    'axes.linewidth': 2.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
})

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

min_fit_stellar_mass = 1e0 # Do not use data points for linear fit with total stellar mass below this


def _darken_fig(fig):
    """Convert a white-bg figure to dark in-place."""
    BG = '#181818'; FG = 'white'
    fig.patch.set_facecolor(BG)
    for ax in fig.axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, which='both')
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        ax.title.set_color(FG)
        for sp in ax.spines.values(): sp.set_edgecolor(FG)
        for txt in ax.get_xticklabels() + ax.get_yticklabels():
            txt.set_color(FG)
        leg = ax.get_legend()
        if leg:
            leg.get_frame().set_facecolor('#2a2a2a')
            leg.get_frame().set_edgecolor('#555')
            for t in leg.get_texts(): t.set_color(FG)
        for line in ax.get_lines():
            c = line.get_color()
            if c in ('k', 'black', '#000000', '#222222'):
                line.set_color(FG)

try:
    from astropy.cosmology import Planck18 as cosmo
    import astropy.units as u_astropy
    def scale_to_Myr(a):
        return float(cosmo.age(1.0 / float(a) - 1.0).to(u_astropy.Myr).value)
except ImportError:
    print('WARNING: astropy not found; time axis will be scale factor, not Myr.')
    def scale_to_Myr(a):
        return float(a)


def run(args):
    """
    Run mass evolution analysis with a Defaults-like object or argparse namespace.
    Attributes not present on *args* fall back to sensible defaults via getattr().
    """
    # Normalise attribute names that argparse hyphenates
    r_search  = getattr(args, 'r_search',   getattr(args, 'r-search',   1e-5))
    r_max     = getattr(args, 'r_max',      getattr(args, 'r-max',      1e-5))
    rho_thresh= getattr(args, 'rho_thresh', getattr(args, 'rho-thresh', 1e-15))
    aspect    = getattr(args, 'aspect',     0.3)
    f_kep     = getattr(args, 'f_kep',      getattr(args, 'f-kep',      0.3))
    aperture  = getattr(args, 'aperture',   None)
    snap_start= getattr(args, 'snap_start', getattr(args, 'snap-start', None))
    snap_end  = getattr(args, 'snap_end',   getattr(args, 'snap-end',   None))

    # Build a simple namespace so the rest of the function can use args.* uniformly
    import types
    _a = types.SimpleNamespace(
        path       = args.path,
        sim        = args.sim,
        outdir     = args.outdir,
        r_search   = r_search,
        r_max      = r_max,
        rho_thresh = rho_thresh,
        aspect     = aspect,
        f_kep      = f_kep,
        aperture   = aperture,
        snap_start = snap_start,
        snap_end   = snap_end,
    )
    args = _a

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
    r_SOI_arr   = []   # sphere-of-influence radius [AU] (NaN if undefined)
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

        if 'Masses' not in pdata or 'Coordinates' not in pdata:
            print(f'  snap {snap_num:04d}: no gas fields, skipping')
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
        r_SOI = np.nan
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

            # ── Sphere of influence: 3D radius where M_gas_enc(r) = M_* ──────
            if m_star > 0:
                sort_idx   = np.argsort(dists_from_com)
                r_sorted   = dists_from_com[sort_idx]
                m_sorted   = pdata['Masses'][sort_idx] * 1e10   # Msun
                m_gas_cum  = np.cumsum(m_sorted)
                # Interpolate to find r where cumulative gas mass = M_*
                if m_gas_cum[-1] >= m_star:
                    r_SOI_kpc = float(np.interp(m_star, m_gas_cum, r_sorted))
                    r_SOI = r_SOI_kpc * kpc / AU   # AU
        except Exception as e:
            print(f'  snap {snap_num:04d}: identify_disk error — {e}')
            m_disk  = 0.0
            m_apert = 0.0

        times_Myr.append(t)
        M_disk_arr.append(m_disk)
        M_apert_arr.append(m_apert)
        M_star_arr.append(m_star)
        r_SOI_arr.append(r_SOI)
        print(f'  snap {snap_num:04d}  t={t*1e3:.2f} kyr  '
              f'M_disk={m_disk:.3f}  M_apert={m_apert:.3f}  M_star={m_star:.3f} Msun  '
              f'[{i+1}/{len(snap_items)}]', flush=True)

    if not times_Myr:
        sys.exit('No snapshots processed successfully.')

    times_Myr   = np.array(times_Myr)
    M_disk_arr  = np.array(M_disk_arr)
    M_apert_arr = np.array(M_apert_arr)
    M_star_arr  = np.array(M_star_arr)
    r_SOI_arr   = np.array(r_SOI_arr, dtype=float)

    M_tot_disk  = M_disk_arr  + M_star_arr
    M_tot_apert = M_apert_arr + M_star_arr
    f_star_disk  = np.where(M_tot_disk  > 0, M_star_arr / M_tot_disk,  0.0)
    f_star_apert = np.where(M_tot_apert > 0, M_star_arr / M_tot_apert, 0.0)

    if t1_Myr is not None:
        t_plot = (times_Myr - t1_Myr) * 1e3   # kyr
        xlabel = r'$t - t_1$ (kyr)   [$t_1$ = first sink formation]'
        print(f'\nFirst sink at t_1 = {t1_Myr*1e3:.2f} kyr')
    else:
        t_plot = times_Myr * 1e3   # kyr
        xlabel = 'Time (kyr)'

    os.makedirs(args.outdir, exist_ok=True)

    # --- Figure A: mass + SFE (2 panels, compact) ---
    fig_a, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 18), sharex=True)
    fig_a.patch.set_facecolor('w')
    fig_a.subplots_adjust(hspace=0)
    # --- Figure B: rates + r_SOI (2 panels, compact) ---
    fig_b, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 18), sharex=True)
    fig_b.patch.set_facecolor('w')
    fig_b.subplots_adjust(hspace=0)
    for _ax in [ax1, ax2, ax3, ax4]:
        _ax.set_facecolor('w')
        _ax.tick_params(colors='k', which='both')
        for spine in _ax.spines.values(): spine.set_edgecolor('k')

    apert_label = rf'$r < {aperture_kpc*1e3:.1f}\ \rm pc$ aperture'

    # Panel 1: gas masses + stellar mass (linear y)
    ax1.plot(t_plot, M_star_arr,  color='darkorange', lw=4.0, label=r'$M_*$')
    ax1.plot(t_plot, M_disk_arr,  'c-',  lw=4.0,   label=r'$M_{\rm gas,disk}$')
    ax1.plot(t_plot, M_apert_arr, 'c--', lw=3.0, label=r'$M_{\rm gas,apert}$')
    ax1.plot(t_plot, M_tot_disk,  '#222222',  lw=2.0, alpha=0.5, label=r'$M_{\rm disk}+M_*$')
    ax1.plot(t_plot, M_tot_apert, '#555555', lw=2.0, alpha=0.5, label=r'$M_{\rm apert}+M_*$')
    # Power-law fit M_* ~ A*(t-t1)^alpha for t > t1
    fit_mask = (t_plot > 0) & (t_plot <= 10.0) & (M_star_arr > min_fit_stellar_mass)
    if fit_mask.sum() >= 3:
        log_t = np.log(t_plot[fit_mask])
        log_m = np.log(M_star_arr[fit_mask])
        alpha, log_A = np.polyfit(log_t, log_m, 1)
        A = np.exp(log_A)
        t_fit = np.linspace(t_plot[fit_mask][0], t_plot[fit_mask][-1], 200)
        ax1.plot(t_fit, A * t_fit**alpha, 'r--', lw=3.0,
                 label=rf'fit: $M_* \propto t^{{{alpha:.2f}}}$')
        print(f'Power-law fit: M_* ∝ t^{alpha:.3f}  (A = {A:.4g} Msun/kyr^alpha)')

    ax1.set_ylabel(r'Mass ($M_\odot$)', color='k')
    ax1.tick_params(colors='k', which='both', direction='in', right=True, top=True,
                    labelbottom=False)
    _t_pos = t_plot[t_plot > 0]
    _xlim = [5e0, _t_pos.max() * 1.1] if len(_t_pos) > 1 else [5e0, 1e1]
    ax1.set_xlim(_xlim)
    ax1.set_ylim([10, 400])
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    for sp in ax1.spines.values():
        sp.set_edgecolor('k')

    # Panel 2: SFE — disk vs aperture denominator
    ax2.plot(t_plot, f_star_disk  * 100, 'g-',  lw=4.0,
             label=r'$M_* / (M_{\rm disk}+M_*)$  [disk only]')
    ax2.plot(t_plot, f_star_apert * 100, 'g--', lw=3.0,
             label=r'$M_* / (M_{\rm apert}+M_*)$  [' + apert_label + ']')
    ax2.set_ylabel(r'$f_*$  (%)', color='k')
    ax2.tick_params(colors='k', which='both', direction='in', right=True, top=True)
    for sp in ax2.spines.values():
        sp.set_edgecolor('k')

    # Panel 3: stellar accretion rate + gas depletion rate from aperture
    ax3.set_xscale('log')
    ax3.set_xlim(_xlim)
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
            ax3.semilogy(t_mid[pos_star], dMstar_dt[pos_star],      'm-',  lw=3.0,
                         label=r'$\dot{M}_*$ (stellar accretion)')
        if pos_depl.any():
            ax3.semilogy(t_mid[pos_depl], gas_depletion[pos_depl],  'c--', lw=3.0,
                         label=r'$-\dot{M}_{\rm gas,apert}$ (gas depletion rate)')
        ax3.set_ylabel(r'Rate  ($M_\odot$/yr)', color='k')
        ax3.tick_params(colors='k', which='both', direction='in', right=True, top=True,
                        labelbottom=False)
        for sp in ax3.spines.values():
            sp.set_edgecolor('k')
        ax3.set_ylim([1e-4, 6e-2])

    # Panel 4: sphere of influence radius r_SOI(t)
    valid_soi = np.isfinite(r_SOI_arr) & (r_SOI_arr > 0)
    if valid_soi.any():
        ax4.semilogy(t_plot[valid_soi], r_SOI_arr[valid_soi], 'orange', lw=4.0,
                     label=r'$r_{\rm SOI}$  ($M_{\rm gas,enc} = M_*$)')
    ax4.set_ylabel(r'$r_{\rm SOI}$ (AU)', color='k')
    ax4.tick_params(colors='k', which='both', direction='in', right=True, top=True)
    _soi_max = r_SOI_arr[np.isfinite(r_SOI_arr)].max() * 1.5 if np.any(np.isfinite(r_SOI_arr)) else 1e5
    ax4.set_ylim([1e3, _soi_max])
    for sp in ax4.spines.values():
        sp.set_edgecolor('k')

    # x-axis labels on bottom panels only
    ax2.set_xlabel(xlabel, color='k')
    ax4.set_xlabel(xlabel, color='k')

    # Save figure A (mass + SFE)
    outpath_a = os.path.join(args.outdir, 'light', 'mass_evolution.png')
    os.makedirs(os.path.dirname(outpath_a), exist_ok=True)
    fig_a.savefig(outpath_a, dpi=150, facecolor='w', bbox_inches='tight')
    fig_a.savefig(outpath_a.replace('.png', '.pdf'), facecolor='w', bbox_inches='tight')
    dark_a = outpath_a.replace('/light/', '/dark/')
    os.makedirs(os.path.dirname(dark_a), exist_ok=True)
    _darken_fig(fig_a)
    fig_a.savefig(dark_a, dpi=150, facecolor='#181818', bbox_inches='tight')
    plt.close(fig_a)
    print(f'\nSaved → {outpath_a}')

    # Save figure B (rates + r_SOI)
    outpath_b = os.path.join(args.outdir, 'light', 'mass_evolution_rates.png')
    fig_b.savefig(outpath_b, dpi=150, facecolor='w', bbox_inches='tight')
    fig_b.savefig(outpath_b.replace('.png', '.pdf'), facecolor='w', bbox_inches='tight')
    dark_b = outpath_b.replace('/light/', '/dark/')
    os.makedirs(os.path.dirname(dark_b), exist_ok=True)
    _darken_fig(fig_b)
    fig_b.savefig(dark_b, dpi=150, facecolor='#181818', bbox_inches='tight')
    plt.close(fig_b)
    print(f'\nSaved → {outpath_b}')


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
    run(p.parse_args())


if __name__ == '__main__':
    main()
