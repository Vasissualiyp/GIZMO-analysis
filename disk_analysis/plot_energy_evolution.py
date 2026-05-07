"""
plot_energy_evolution.py
------------------------
Plot disk energy components vs time from FIRE+STARFORGE snapshots.

Energies are computed for disk-classified gas particles only (using the same
identify_disk() criterion as the movie pipeline).

Components:
  E_rot   = (1/2) * sum_i m_i * v_phi_i²      — rotational kinetic energy
  E_turb  = (1/2) * sum_i m_i * |δv_i|²       — turbulent kinetic energy
  E_pot   ≈ -G * sum_i M_enc(r_i) * m_i / r_i  — gravitational potential energy
                                                   (cylindrical approx; includes star)
  E_tot   = E_rot + E_turb + E_pot             — total mechanical energy

All energies are stored in erg; plotted in units of 10^44 erg.

Usage:
    python disk_analysis/plot_energy_evolution.py [--path PATH] [--sim SIM] ...
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
pfp_src_path  = "/home/vasissua/PYTHON/pfh_python/gizmopy/"
sys.path.insert(0, guac_src_path)
sys.path.insert(0, pfp_src_path)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from notebooks.make_disk_movie_frames import identify_disk, rotation_matrix_to_z
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

# Convenient energy unit: 10^44 erg
_E_UNIT = 1e44   # erg


def _compute_disk_energies(pdata, stardata, args):
    """
    Identify disk particles and compute E_rot, E_turb, E_pot, E_tot.
    Returns (E_rot, E_turb, E_pot, E_tot) in erg, or (nan, nan, nan, nan) on failure.
    """
    try:
        is_disk, com, L_hat, _, _, _, _, com_vel = identify_disk(
            pdata, stardata,
            r_search_kpc      = args.r_search,
            r_max_kpc         = args.r_max,
            rho_threshold_cgs = args.rho_thresh,
            aspect_ratio      = args.aspect,
            f_kep             = args.f_kep,
        )
    except Exception as e:
        print(f'    identify_disk error: {e}')
        return np.nan, np.nan, np.nan, np.nan

    n_disk = int(is_disk.sum())
    if n_disk < 5:
        return np.nan, np.nan, np.nan, np.nan

    rot = rotation_matrix_to_z(L_hat)

    # ── Disk particle kinematics in disk frame ────────────────────────────────
    pos_disk  = (pdata['Coordinates'][is_disk] - com) @ rot.T  # [kpc], z ‖ L_hat
    vel_disk  = (pdata['Velocities'][is_disk]  - (com_vel if com_vel is not None
                                                   else np.zeros(3))) @ rot.T  # km/s
    mass_disk = pdata['Masses'][is_disk] * 1e10   # Msun
    hsml_disk = pdata['SmoothingLength'][is_disk]   # kpc (gravitational softening proxy)

    r_cyl  = np.linalg.norm(pos_disk[:, :2], axis=1)   # kpc, cylindrical radius
    # Gravitational softening: use mean disk smoothing length as minimum radius.
    # Without this, particles sitting at r_cyl ≈ 0 (inside the sink accretion radius)
    # produce M_enc/r → ∞ and E_pot → -∞.
    r_soft_kpc = max(float(np.mean(hsml_disk)), AU / kpc)   # kpc; at least 1 AU
    safe_r = np.maximum(r_cyl, r_soft_kpc)

    e_r_x = pos_disk[:, 0] / np.maximum(r_cyl, 1e-30)
    e_r_y = pos_disk[:, 1] / np.maximum(r_cyl, 1e-30)
    v_r   =  vel_disk[:, 0] * e_r_x + vel_disk[:, 1] * e_r_y   # km/s
    v_phi = -vel_disk[:, 0] * e_r_y + vel_disk[:, 1] * e_r_x   # km/s
    v_z   =  vel_disk[:, 2]                                      # km/s

    # Streaming subtraction: 20-bin mass-weighted radial profiles
    N_BINS  = 20
    r_outer = max(np.percentile(r_cyl, 95), 1e-20)
    bins    = np.linspace(0.0, r_outer, N_BINS + 1)
    bidx    = np.clip(np.digitize(r_cyl, bins) - 1, 0, N_BINS - 1)

    vr_prof   = np.zeros(N_BINS)
    vphi_prof = np.zeros(N_BINS)
    for b in range(N_BINS):
        mb = bidx == b
        if mb.sum() == 0:
            continue
        w = mass_disk[mb]; wsum = w.sum()
        vr_prof[b]   = np.dot(v_r[mb],   w) / wsum
        vphi_prof[b] = np.dot(v_phi[mb], w) / wsum

    dv_r   = v_r   - vr_prof[bidx]
    dv_phi = v_phi - vphi_prof[bidx]
    dv_z   = v_z
    v_rest = np.sqrt(dv_r**2 + dv_phi**2 + dv_z**2)   # |δv| per particle [km/s]

    # ── Kinetic energies ──────────────────────────────────────────────────────
    m_g        = mass_disk * Msun     # g
    v_phi_cms  = v_phi  * 1e5         # cm/s
    v_rest_cms = v_rest * 1e5         # cm/s

    E_rot  = 0.5 * float(np.sum(m_g * v_phi_cms**2))   # erg
    E_turb = 0.5 * float(np.sum(m_g * v_rest_cms**2))  # erg

    # ── Gravitational potential energy (annular-binned, robust) ───────────────
    # Per-particle summation (M_enc * m / r) overflows to -inf when any particle
    # sits at r_cyl ≈ 0 inside the sink accretion sphere.  Instead, compute the
    # potential using the same N_BINS annular bins: for each bin use the bin-centre
    # radius and the mass-weighted mean enclosed mass.  The bin-centre is always
    # >= half a bin-width above r=0, so no singularity.
    M_star_Msun = (float(np.sum(stardata['Masses'])) * 1e10
                   if stardata and len(stardata.get('Masses', [])) > 0 else 0.0)

    # Build cumulative gas-mass profile vs radius (all disk particles sorted by r)
    sort_r      = np.argsort(r_cyl)
    m_sorted    = mass_disk[sort_r]                          # Msun, sorted by r
    M_gas_cum   = np.concatenate([[0.0], np.cumsum(m_sorted)])   # Msun, length N+1

    E_pot = 0.0
    for b in range(N_BINS):
        mb = bidx == b
        if mb.sum() == 0:
            continue
        r_lo_kpc, r_hi_kpc = bins[b], bins[b + 1]
        r_ctr_kpc = 0.5 * (r_lo_kpc + r_hi_kpc)
        # Apply softening: bin centre must be at least r_soft away from origin
        r_eff_cm  = max(r_ctr_kpc, r_soft_kpc) * kpc          # cm

        # Gas mass enclosed within r_lo (exclusive of this bin)
        n_enc = int(np.searchsorted(r_cyl[sort_r], r_lo_kpc))
        M_gas_enc_Msun = float(M_gas_cum[n_enc])

        M_enc_Msun = M_star_Msun + M_gas_enc_Msun             # Msun
        M_bin_Msun = float(np.sum(mass_disk[mb]))              # Msun in this bin

        E_pot -= G * (M_enc_Msun * Msun) * (M_bin_Msun * Msun) / r_eff_cm  # erg

    E_tot  = E_rot + E_turb + E_pot

    return float(E_rot), float(E_turb), float(E_pot), float(E_tot)


def run(args):
    """
    Run energy evolution analysis with a Defaults-like object or argparse namespace.
    Attributes not present on *args* fall back to sensible defaults via getattr().
    """
    import types
    r_search   = getattr(args, 'r_search',   getattr(args, 'r-search',   1e-5))
    r_max      = getattr(args, 'r_max',      getattr(args, 'r-max',      1e-5))
    rho_thresh = getattr(args, 'rho_thresh', getattr(args, 'rho-thresh', 1e-15))
    aspect     = getattr(args, 'aspect',     0.3)
    f_kep      = getattr(args, 'f_kep',      getattr(args, 'f-kep',      0.3))
    snap_start = getattr(args, 'snap_start', getattr(args, 'snap-start', None))
    snap_end   = getattr(args, 'snap_end',   getattr(args, 'snap-end',   None))
    save_npz   = getattr(args, 'save_npz',   getattr(args, 'save-npz',   True))

    _a = types.SimpleNamespace(
        path       = args.path,
        sim        = args.sim,
        outdir     = args.outdir,
        r_search   = r_search,
        r_max      = r_max,
        rho_thresh = rho_thresh,
        aspect     = aspect,
        f_kep      = f_kep,
        snap_start = snap_start,
        snap_end   = snap_end,
        save_npz   = save_npz,
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

    print(f'Processing {len(snap_items)} snapshots from {args.path}{args.sim}/')

    gas_fields = ['Masses', 'Coordinates', 'SmoothingLength',
                  'Velocities', 'Density', 'InternalEnergy']

    times_Myr  = []
    E_rot_arr  = []
    E_turb_arr = []
    E_pot_arr  = []
    E_tot_arr  = []
    t1_Myr     = None

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

        # Track first sink formation time
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

        E_rot, E_turb, E_pot, E_tot = _compute_disk_energies(pdata, stardata, args)

        times_Myr.append(t)
        E_rot_arr.append(E_rot)
        E_turb_arr.append(E_turb)
        E_pot_arr.append(E_pot)
        E_tot_arr.append(E_tot)
        print(f'  snap {snap_num:04d}  t={t*1e3:.2f} kyr  '
              f'E_rot={E_rot / _E_UNIT:.3f}  E_turb={E_turb / _E_UNIT:.3f}  '
              f'E_pot={E_pot / _E_UNIT:.3f}  E_tot={E_tot / _E_UNIT:.3f}  '
              f'[×10^44 erg]  [{i+1}/{len(snap_items)}]', flush=True)

    if not times_Myr:
        sys.exit('No snapshots processed successfully.')

    times_Myr  = np.array(times_Myr)
    E_rot_arr  = np.array(E_rot_arr)
    E_turb_arr = np.array(E_turb_arr)
    E_pot_arr  = np.array(E_pot_arr)
    E_tot_arr  = np.array(E_tot_arr)

    if t1_Myr is not None:
        t_plot = (times_Myr - t1_Myr) * 1e3   # kyr
        xlabel = r'$t - t_1$ (kyr)   [$t_1$ = first sink formation]'
        print(f'\nFirst sink at t_1 = {t1_Myr*1e3:.2f} kyr')
    else:
        t_plot = times_Myr * 1e3   # kyr
        xlabel = 'Time (kyr)'

    os.makedirs(args.outdir, exist_ok=True)

    # ── Save NPZ ──────────────────────────────────────────────────────────────
    if args.save_npz:
        npz_path = os.path.join(args.outdir, 'energy_evolution.npz')
        np.savez(npz_path,
                 times_Myr = times_Myr,
                 t1_Myr    = np.array([t1_Myr if t1_Myr is not None else np.nan]),
                 E_rot     = E_rot_arr,
                 E_turb    = E_turb_arr,
                 E_pot     = E_pot_arr,
                 E_tot     = E_tot_arr)
        print(f'  NPZ saved → {npz_path}')

    # ── Derived quantities ────────────────────────────────────────────────────
    E_kin_arr   = E_rot_arr + E_turb_arr          # total kinetic
    E_abspot_arr = np.abs(E_pot_arr)              # |E_pot|, always positive
    # Virial ratio: 2 E_kin / |E_pot|  (= 1 at virial equilibrium; < 1 contracting)
    with np.errstate(divide='ignore', invalid='ignore'):
        virial_arr = np.where(E_abspot_arr > 0,
                              2.0 * E_kin_arr / E_abspot_arr, np.nan)
    # Total mechanical energy: E_tot = E_kin + E_pot  (negative = bound)
    # E_tot is already computed; here we re-derive for clarity.
    E_tot_arr = E_kin_arr + E_pot_arr

    # ── Plot ──────────────────────────────────────────────────────────────────
    # Layout (3 panels, sharex):
    #   Panel 1: E_rot, E_turb, |E_pot| on same axes (log-scale) — shows transfer
    #   Panel 2: E_tot (= E_kin + E_pot, negative for bound system)
    #   Panel 3: Virial ratio 2 E_kin / |E_pot|
    plt.style.use('dark_background')
    _style = dict(colors='w', which='both', direction='in', right=True, top=True)

    fig, (ax_comp, ax_tot, ax_vir) = plt.subplots(3, 1, figsize=(12, 14),
                                                    sharex=True)
    fig.patch.set_facecolor('k')

    def _leg(ax):
        leg = ax.legend(fontsize=9, framealpha=0.3)
        for txt in leg.get_texts(): txt.set_color('w')

    def _style_ax(ax, ylabel, title):
        ax.set_facecolor('k')
        ax.set_xlabel(xlabel, color='w', fontsize=10)
        ax.set_ylabel(ylabel, color='w', fontsize=10)
        ax.set_title(title, color='w', fontsize=11)
        ax.tick_params(**_style)
        for sp in ax.spines.values(): sp.set_edgecolor('w')

    # ── Panel 1: kinetic components + |E_pot| ────────────────────────────────
    # All three are positive — use log-y to compare magnitudes.
    # When |E_pot| > E_kin the system is sub-virial (still contracting).
    for E_arr, color, label in [
        (E_rot_arr,   'gold',   r'$E_{\rm rot}$'),
        (E_turb_arr,  'cyan',   r'$E_{\rm turb}$'),
        (E_kin_arr,   'lime',   r'$E_{\rm kin} = E_{\rm rot}+E_{\rm turb}$'),
        (E_abspot_arr,'tomato', r'$|E_{\rm pot}|$  (gravitational well depth)'),
    ]:
        valid = np.isfinite(E_arr) & (E_arr > 0)
        if valid.any():
            ax_comp.semilogy(t_plot[valid], E_arr[valid] / _E_UNIT,
                             lw=2, color=color, label=label)
    _style_ax(ax_comp,
              r'Energy ($10^{44}$ erg)',
              r'Energy components  [solid: $E_{\rm kin}$;  tomato: $|E_{\rm pot}|$]')
    ax_comp.annotate(r'$|E_{\rm pot}| > E_{\rm kin}$ → sub-virial (contracting)',
                     xy=(0.02, 0.05), xycoords='axes fraction',
                     color='w', fontsize=9, alpha=0.7)
    _leg(ax_comp)

    # ── Panel 2: total mechanical energy (negative = bound) ───────────────────
    # Note: E_tot = E_kin + E_pot.  Since E_pot < 0 and |E_pot| >> E_kin for a
    # bound disk, E_tot is negative.  This is correct — it is NOT a sign error.
    # Plotting |E_tot| with a note that the sign is negative.
    valid_tot = np.isfinite(E_tot_arr)
    pos_tot   = valid_tot & (E_tot_arr > 0)
    neg_tot   = valid_tot & (E_tot_arr < 0)
    if neg_tot.any():
        ax_tot.semilogy(t_plot[neg_tot], np.abs(E_tot_arr[neg_tot]) / _E_UNIT,
                        'w-', lw=2,
                        label=r'$|E_{\rm tot}|$  (bound: $E_{\rm tot}<0$)')
    if pos_tot.any():
        ax_tot.semilogy(t_plot[pos_tot], E_tot_arr[pos_tot] / _E_UNIT,
                        'w--', lw=2, label=r'$E_{\rm tot}$ (unbound)')
    ax_tot.annotate(
        r'$E_{\rm tot} = E_{\rm kin} + E_{\rm pot}$; negative because $|E_{\rm pot}| \gg E_{\rm kin}$',
        xy=(0.02, 0.05), xycoords='axes fraction', color='w', fontsize=9, alpha=0.7)
    _style_ax(ax_tot, r'$|E_{\rm tot}|$  ($10^{44}$ erg)',
              r'Total mechanical energy  (dashed = unbound, solid = $|E_{\rm tot}|$ bound)')
    _leg(ax_tot)

    # ── Panel 3: virial ratio ─────────────────────────────────────────────────
    valid_vir = np.isfinite(virial_arr)
    if valid_vir.any():
        ax_vir.semilogy(t_plot[valid_vir], virial_arr[valid_vir], 'w-', lw=2,
                        label=r'$2E_{\rm kin}/|E_{\rm pot}|$')
    ax_vir.axhline(1.0, color='r',    lw=1.5, ls='--', label='virial equilibrium = 1')
    ax_vir.axhline(2.0, color='orange', lw=1,  ls=':',  label='unbound = 2')
    _style_ax(ax_vir,
              r'$2E_{\rm kin} / |E_{\rm pot}|$',
              'Virial ratio  (< 1 contracting,  ≈ 1 virialized,  > 2 unbound)')
    _leg(ax_vir)

    fig.suptitle(
        'Disk energy evolution\n'
        r'(disk gas only; $E_{\rm pot}$ annular-binned cylindrical approx.)',
        color='w', fontsize=12)
    fig.tight_layout()

    plot_path = os.path.join(args.outdir, 'energy_evolution.png')
    fig.savefig(plot_path, dpi=150, facecolor='k')
    plt.close(fig)
    print(f'  Plot saved → {plot_path}')


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
    p.add_argument('--snap-start', type=int,   default=None)
    p.add_argument('--snap-end',   type=int,   default=None)
    p.add_argument('--save-npz',   action='store_true', default=True,
                   help='Save computed energies to NPZ for later use')
    run(p.parse_args())


if __name__ == '__main__':
    main()
