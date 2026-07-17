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

plt.rcParams.update({
    'font.size': 30,
    'axes.labelsize': 33,
    'axes.titlesize': 33,
    'xtick.labelsize': 27,
    'ytick.labelsize': 27,
    'legend.fontsize': 28,
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

# Convenient energy unit: 10^44 erg
_E_UNIT = 1e44   # erg


def _compute_disk_energies(pdata, stardata, args):
    """
    Identify disk particles and compute E_rot, E_turb, E_pot, E_tot, E_therm, E_mag, E_therm_cs.
    Returns 7-tuple in erg, or (nan,)*7 on failure.
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
        return (np.nan,) * 7

    n_disk = int(is_disk.sum())
    if n_disk < 5:
        return (np.nan,) * 7

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

    # ── Gravitational potential energy (pairwise sum with softening) ────────────
    eps_cm = r_soft_kpc * kpc  # softening length in cm
    pos_cm = pos_disk * kpc    # 3D positions in cm (rotation-invariant distances)

    # Gas–gas: E_pot_gg = -G Σ_{i<j} m_i m_j / sqrt(r_ij² + ε²)
    E_pot = 0.0
    for i in range(n_disk):
        dr = pos_cm[i + 1:] - pos_cm[i]
        r_soft = np.sqrt(np.sum(dr**2, axis=1) + eps_cm**2)
        E_pot -= G * m_g[i] * np.sum(m_g[i + 1:] / r_soft)

    # Sink–gas: -G Σ_s Σ_i m_s m_i / sqrt(r_{si}² + ε²)
    if stardata and len(stardata.get('Masses', [])) > 0:
        sk_pos_cm = (stardata['Coordinates'] - com) * kpc  # cm
        sk_m_g = stardata['Masses'] * 1e10 * Msun          # grams
        for s in range(len(sk_m_g)):
            dr = pos_cm - sk_pos_cm[s]
            r_soft = np.sqrt(np.sum(dr**2, axis=1) + eps_cm**2)
            E_pot -= G * sk_m_g[s] * np.sum(m_g / r_soft)

    E_tot  = E_rot + E_turb + E_pot

    # ── Thermal energy ───────────────────────────────────────────────────────
    if 'InternalEnergy' in pdata:
        u_disk = pdata['InternalEnergy'][is_disk].astype(np.float64)  # (km/s)²
        u_disk_cgs = u_disk * 1e10                    # cm²/s²
        E_therm = float(np.sum(m_g * u_disk_cgs))     # erg
    else:
        E_therm = np.nan

    # ── Thermal energy from SoundSpeed: u = c_s² / (γ(γ-1)), γ = 5/3 ────
    if 'SoundSpeed' in pdata:
        cs_disk = pdata['SoundSpeed'][is_disk].astype(np.float64)  # km/s
        cs_cgs = cs_disk * 1e5                                     # cm/s
        gamma = 5.0 / 3.0
        u_from_cs = cs_cgs**2 / (gamma * (gamma - 1.0))           # cm²/s²
        E_therm_cs = float(np.sum(m_g * u_from_cs))               # erg
    else:
        E_therm_cs = np.nan

    # ── Magnetic energy: (1/8π) ∫ B² dV ≈ Σ B²/(8π) × (m/ρ) ────────────
    if 'MagneticField' in pdata:
        B_disk = pdata['MagneticField'][is_disk]     # Gauss (after unit conversion)
        B2 = np.sum(B_disk**2, axis=1)               # |B|² in Gauss²
        rho_disk_cgs = pdata['Density'][is_disk].astype(np.float64) * 1e10 * Msun / kpc**3
        vol_disk = m_g / rho_disk_cgs                 # cm³ per particle
        E_mag = float(np.sum(B2 / (8.0 * np.pi) * vol_disk))  # erg
    else:
        E_mag = np.nan

    return float(E_rot), float(E_turb), float(E_pot), float(E_tot), float(E_therm), float(E_mag), float(E_therm_cs)


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
                  'Velocities', 'Density', 'InternalEnergy', 'MagneticField',
                  'SoundSpeed']

    times_Myr   = []
    E_rot_arr   = []
    E_turb_arr  = []
    E_pot_arr   = []
    E_tot_arr   = []
    E_therm_arr = []
    E_mag_arr   = []
    E_therm_cs_arr = []
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

        E_rot, E_turb, E_pot, E_tot, E_therm, E_mag, E_therm_cs = _compute_disk_energies(pdata, stardata, args)

        times_Myr.append(t)
        E_rot_arr.append(E_rot)
        E_turb_arr.append(E_turb)
        E_pot_arr.append(E_pot)
        E_tot_arr.append(E_tot)
        E_therm_arr.append(E_therm)
        E_mag_arr.append(E_mag)
        E_therm_cs_arr.append(E_therm_cs)
        print(f'  snap {snap_num:04d}  t={t*1e3:.2f} kyr  '
              f'E_rot={E_rot / _E_UNIT:.3f}  E_turb={E_turb / _E_UNIT:.3f}  '
              f'E_pot={E_pot / _E_UNIT:.3f}  E_therm={E_therm / _E_UNIT:.3f}  '
              f'E_therm_cs={E_therm_cs / _E_UNIT:.3f}  E_mag={E_mag / _E_UNIT:.3f}  '
              f'[×10^44 erg]  [{i+1}/{len(snap_items)}]', flush=True)

    if not times_Myr:
        sys.exit('No snapshots processed successfully.')

    times_Myr   = np.array(times_Myr)
    E_rot_arr   = np.array(E_rot_arr)
    E_turb_arr  = np.array(E_turb_arr)
    E_pot_arr   = np.array(E_pot_arr)
    E_tot_arr   = np.array(E_tot_arr)
    E_therm_arr = np.array(E_therm_arr)
    E_mag_arr   = np.array(E_mag_arr)
    E_therm_cs_arr = np.array(E_therm_cs_arr)

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
                 E_tot     = E_tot_arr,
                 E_therm   = E_therm_arr,
                 E_mag     = E_mag_arr,
                 E_therm_cs = E_therm_cs_arr)
        print(f'  NPZ saved → {npz_path}')

    # ── Derived quantities ────────────────────────────────────────────────────
    E_kin_arr   = E_rot_arr + E_turb_arr          # total kinetic
    E_abspot_arr = np.abs(E_pot_arr)              # |E_pot|, always positive
    # Virial ratio: 2 E_kin / |E_pot|  (= 1 at virial equilibrium; < 1 contracting)
    with np.errstate(divide='ignore', invalid='ignore'):
        virial_arr = np.where(E_abspot_arr > 0,
                              2.0 * E_kin_arr / E_abspot_arr, np.nan)
        virial_therm_arr = np.where(E_abspot_arr > 0,
                                    2.0 * (E_kin_arr + E_therm_arr) / E_abspot_arr, np.nan)
        virial_full_arr = np.where(E_abspot_arr > 0,
                                   2.0 * (E_kin_arr + E_therm_arr + E_mag_arr) / E_abspot_arr, np.nan)
    # Total mechanical energy: E_tot = E_kin + E_pot  (negative = bound)
    # E_tot is already computed; here we re-derive for clarity.
    E_tot_arr = E_kin_arr + E_pot_arr

    # ── Plot ──────────────────────────────────────────────────────────────────
    # Layout (2 panels, sharex, 16:9 aspect):
    #   Panel 1: E_rot, E_turb, |E_pot| on same axes (log-scale) — shows transfer
    #   Panel 2: Virial ratio 2 E_kin / |E_pot|
    _style = dict(colors='k', which='both', direction='in', right=True, top=True)

    fig, (ax_comp, ax_vir) = plt.subplots(2, 1, figsize=(12, 18),
                                           sharex=True)
    fig.patch.set_facecolor('w')
    fig.subplots_adjust(hspace=0)
    # Explicit x-axis limits so the plot always extends to the final snapshot.
    # Use the full t_plot range (linear scale); pad the right edge slightly.
    _t_finite = t_plot[np.isfinite(t_plot)]
    if len(_t_finite) > 1:
        _t_lo = _t_finite.min()
        _t_hi = _t_finite.max()
        _span = max(_t_hi - _t_lo, 1.0)
        ax_comp.set_xlim([_t_lo - _span * 0.02, _t_hi + _span * 0.02])

    def _leg(ax):
        leg = ax.legend(framealpha=0.8, facecolor='w', ncol=2)
        for txt in leg.get_texts(): txt.set_color('k')

    def _style_ax(ax, ylabel, title=None, is_bottom=False):
        ax.set_facecolor('w')
        if is_bottom:
            ax.set_xlabel(xlabel, color='k')
        else:
            ax.tick_params(labelbottom=False)
        ax.set_ylabel(ylabel, color='k')
        ax.tick_params(**_style)
        for sp in ax.spines.values(): sp.set_edgecolor('k')

    # ── Panel 1: kinetic components + |E_pot| ────────────────────────────────
    # All three are positive — use log-y to compare magnitudes.
    # When |E_pot| > E_kin the system is sub-virial (still contracting).
    for E_arr, color, label in [
        (E_rot_arr,   '#1f77b4',  r'$E_{\rm rot}$'),
        (E_turb_arr,  '#2ca02c',  r'$E_{\rm turb}$'),
        (E_kin_arr,   '#000000',  r'$E_{\rm kin} = E_{\rm rot}+E_{\rm turb}$'),
        (E_abspot_arr,'#d62728',  r'$|E_{\rm pot}|$'),
        (E_therm_arr, '#ff7f0e',  r'$E_{\rm therm}$'),
        (E_mag_arr,   '#9467bd',  r'$E_{\rm mag}$'),
    ]:
        valid = np.isfinite(E_arr) & (E_arr > 0)
        if valid.any():
            ax_comp.semilogy(t_plot[valid], E_arr[valid] / _E_UNIT,
                             lw=4.0, color=color, label=label)
    _style_ax(ax_comp,
              r'Energy ($10^{44}$ erg)',
              r'Energy components  [solid: $E_{\rm kin}$;  tomato: $|E_{\rm pot}|$]')
    _leg(ax_comp)

    # ── Panel 2: virial ratio ─────────────────────────────────────────────────
    valid_vir = np.isfinite(virial_arr)
    if valid_vir.any():
        ax_vir.semilogy(t_plot[valid_vir], virial_arr[valid_vir], '#222222', lw=4.0,
                        label=r'$2E_{\rm kin}/|E_{\rm pot}|$')
    valid_vir_th = np.isfinite(virial_therm_arr)
    if valid_vir_th.any():
        ax_vir.semilogy(t_plot[valid_vir_th], virial_therm_arr[valid_vir_th], '#ff7f0e', lw=4.0,
                        label=r'$2(E_{\rm kin}+E_{\rm therm})/|E_{\rm pot}|$')
    valid_vir_full = np.isfinite(virial_full_arr)
    if valid_vir_full.any():
        ax_vir.semilogy(t_plot[valid_vir_full], virial_full_arr[valid_vir_full], '#9467bd', lw=4.0,
                        label=r'$2(E_{\rm kin}+E_{\rm therm}+E_{\rm mag})/|E_{\rm pot}|$')
    ax_vir.axhline(1.0, color='r',    lw=3.0, ls='--', label='virial equilibrium = 1')
    _style_ax(ax_vir,
              r'$2E_{\rm kin} / |E_{\rm pot}|$',
              'Virial ratio  (< 1 contracting,  ≈ 1 virialized,  > 2 unbound)',
              is_bottom=True)
    _leg(ax_vir)
    ax_vir.set_ylim(5e-1, 3e0)
    # Show only integer powers of 10 on y-axis (remove minor tick labels like 6×10^-1)
    import matplotlib.ticker as _ticker
    ax_vir.yaxis.set_major_locator(_ticker.LogLocator(base=10, numticks=10))
    ax_vir.yaxis.set_major_formatter(_ticker.LogFormatterSciNotation(labelOnlyBase=True))
    ax_vir.yaxis.set_minor_locator(_ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax_vir.yaxis.set_minor_formatter(_ticker.NullFormatter())

    plot_path = os.path.join(args.outdir, 'light', 'energy_evolution.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=150, facecolor='w', bbox_inches='tight')
    fig.savefig(plot_path.replace('.png', '.pdf'), facecolor='w', bbox_inches='tight')
    # Dark version
    dark_path = plot_path.replace('/light/', '/dark/')
    os.makedirs(os.path.dirname(dark_path), exist_ok=True)
    _darken_fig(fig)
    fig.savefig(dark_path, dpi=150, facecolor='#181818', bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot saved → {plot_path}')

    # ── Plot 2: same layout but E_therm replaced by E_therm_cs (from SoundSpeed) ─
    fig2, (ax2_comp, ax2_vir) = plt.subplots(2, 1, figsize=(12, 18), sharex=True)
    fig2.patch.set_facecolor('w')
    fig2.subplots_adjust(hspace=0)
    if len(_t_finite) > 1:
        ax2_comp.set_xlim([_t_lo - _span * 0.02, _t_hi + _span * 0.02])

    for E_arr, color, label in [
        (E_rot_arr,      '#1f77b4',  r'$E_{\rm rot}$'),
        (E_turb_arr,     '#2ca02c',  r'$E_{\rm turb}$'),
        (E_kin_arr,      '#000000',  r'$E_{\rm kin} = E_{\rm rot}+E_{\rm turb}$'),
        (E_abspot_arr,   '#d62728',  r'$|E_{\rm pot}|$'),
        (E_therm_cs_arr, '#ff7f0e',  r'$E_{\rm therm}(c_s)$'),
        (E_mag_arr,      '#9467bd',  r'$E_{\rm mag}$'),
    ]:
        valid = np.isfinite(E_arr) & (E_arr > 0)
        if valid.any():
            ax2_comp.semilogy(t_plot[valid], E_arr[valid] / _E_UNIT,
                              lw=4.0, color=color, label=label)
    _style_ax(ax2_comp, r'Energy ($10^{44}$ erg)')
    _leg(ax2_comp)

    valid_vir = np.isfinite(virial_arr)
    if valid_vir.any():
        ax2_vir.semilogy(t_plot[valid_vir], virial_arr[valid_vir], '#222222', lw=4.0,
                         label=r'$2E_{\rm kin}/|E_{\rm pot}|$')
    valid_vir_th = np.isfinite(virial_therm_arr)
    if valid_vir_th.any():
        ax2_vir.semilogy(t_plot[valid_vir_th], virial_therm_arr[valid_vir_th], '#ff7f0e', lw=4.0,
                         label=r'$2(E_{\rm kin}+E_{\rm therm})/|E_{\rm pot}|$')
    valid_vir_full = np.isfinite(virial_full_arr)
    if valid_vir_full.any():
        ax2_vir.semilogy(t_plot[valid_vir_full], virial_full_arr[valid_vir_full], '#9467bd', lw=4.0,
                         label=r'$2(E_{\rm kin}+E_{\rm therm}+E_{\rm mag})/|E_{\rm pot}|$')
    ax2_vir.axhline(1.0, color='r', lw=3.0, ls='--', label='virial equilibrium = 1')
    _style_ax(ax2_vir, r'$2E_{\rm kin} / |E_{\rm pot}|$', is_bottom=True)
    _leg(ax2_vir)
    ax2_vir.set_ylim(5e-1, 3e0)
    ax2_vir.yaxis.set_major_locator(_ticker.LogLocator(base=10, numticks=10))
    ax2_vir.yaxis.set_major_formatter(_ticker.LogFormatterSciNotation(labelOnlyBase=True))
    ax2_vir.yaxis.set_minor_locator(_ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax2_vir.yaxis.set_minor_formatter(_ticker.NullFormatter())

    cs_plot_path = os.path.join(args.outdir, 'light', 'energy_evolution_cs.png')
    fig2.savefig(cs_plot_path, dpi=150, facecolor='w', bbox_inches='tight')
    fig2.savefig(cs_plot_path.replace('.png', '.pdf'), facecolor='w', bbox_inches='tight')
    cs_dark_path = cs_plot_path.replace('/light/', '/dark/')
    os.makedirs(os.path.dirname(cs_dark_path), exist_ok=True)
    _darken_fig(fig2)
    fig2.savefig(cs_dark_path, dpi=150, facecolor='#181818', bbox_inches='tight')
    plt.close(fig2)
    print(f'  Plot saved → {cs_plot_path}')


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
