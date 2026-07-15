"""
plot_velocity_power_spectrum.py
-------------------------------
Compute and plot the 3D velocity power spectrum of disk gas.

For each snapshot:
  1. Identify disk and rotate to face-on frame
  2. Build two velocity fields in the disk frame:
       - Full velocity:     v = (v_x, v_y, v_z)  [COM-subtracted only]
       - Turbulent velocity: δv = v minus streaming profiles <v_r>(r), <v_phi>(r)
  3. Grid each field onto a 3D cube via Meshoid.InterpToGrid
  4. Compute 3D FFT and spherical-shell average |v(k)|² → E(k) using
     GetPowerSpectrum from starforge_tools/2point_statistics/ComputePowerSpectrum.py
  5. Plot both E(k) curves on the same axes with Kolmogorov k^{-5/3}, Burgers k^{-2},
     Kraichnan k^{-3} reference slopes and injection/dissipation scale markers

Output layout:
  {outdir}/velocity_power_spectra/
      plots/   vps_XXXX.png
      data/    vps_XXXX.npz
"""

import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'xtick.major.width': 2.4,
    'xtick.minor.width': 1.6,
    'ytick.major.width': 2.4,
    'ytick.minor.width': 1.6,
    'axes.linewidth': 2.0,
})
from scipy.fft import fftn, fftfreq
from scipy.stats import binned_statistic
from meshoid import Meshoid

guac_src_path = "/home/vasissua/PYTHON/GUAC/src/"
pfp_src_path  = "/home/vasissua/PYTHON/pfh_python/gizmopy/"
sys.path.insert(0, guac_src_path)
sys.path.insert(0, pfp_src_path)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from notebooks.make_disk_movie_frames import (
    identify_disk, rotation_matrix_to_z, _scale_to_Myr,
)
from generic_utils.constants import kpc, AU, Msun
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


# ── GetPowerSpectrum — sourced from starforge_tools/2point_statistics/ComputePowerSpectrum.py ──
# (copied here because that file has module-level docopt/Parallel calls that prevent import)

def GetPowerSpectrum(grid, res):
    """
    Compute 3D power spectrum from a volumetric grid.
    grid : (res, res, res) for scalar field, or (3, res, res, res) for vector field.
    Returns (k_int, power_spectrum) where k_int is the integer shell wavenumber and
    power_spectrum is the power density per unit shell width (unnormalized DFT units).
    """
    if grid.shape[0] == 3:
        vk = np.array([fftn(V) for V in grid])
        vkSqr = np.sum(np.abs(vk) ** 2, axis=0)
    else:
        vk = fftn(grid)
        vkSqr = np.abs(vk) ** 2
    freqs   = fftfreq(res)
    freq3d  = np.array(np.meshgrid(freqs, freqs, freqs, indexing="ij"))
    intfreq = np.int_(np.around(freq3d * res))
    intkSqr = np.sum(np.abs(intfreq) ** 2, axis=0)
    intk    = intkSqr ** 0.5
    kbins   = np.arange(intk.max()) * (1 + 1e-15)
    power_in_bin = binned_statistic(
        intk.flatten(), vkSqr.flatten(), bins=kbins, statistic="sum")[0]
    power_spectrum = power_in_bin / np.diff(kbins)
    power_spectrum[power_spectrum == 0] = np.nan
    return kbins[1:], power_spectrum


def _grid_and_ps(v_field, pos_cut, mass_cut, hsml_cut, M, box_kpc, res_vps):
    """
    Interpolate v_field (N×3) onto a res_vps³ grid via M (pre-built Meshoid),
    then compute GetPowerSpectrum.  Returns (k_AU, E_k).
    """
    gridsize_AU = box_kpc * kpc / AU
    center0     = np.zeros(3)
    vgrid = M.InterpToGrid(v_field, size=box_kpc, res=res_vps, center=center0)
    vgrid = np.rollaxis(vgrid, -1, 0)   # (3, R, R, R)
    k_int, power = GetPowerSpectrum(vgrid, res_vps)
    dx_AU = gridsize_AU / res_vps
    k_AU  = k_int * 2.0 * np.pi / gridsize_AU
    E_k   = power * dx_AU ** 3
    return k_AU, E_k


# Aperture multipliers relative to image_box used for multi-aperture VPS comparison.
# Each entry is a scale factor; 1.0 = same as main image_box (disk scale).
VPS_APERTURE_SCALES = [0.5, 1.0, 2.0, 4.0]


def process_snap(args, snap_path, snap_num):
    """
    Returns (k_AU, E_k_turb, E_k_full, E_k_apertures, time_kyr,
             r_disk_AU, sml_mean_AU, snap_num)
    or None on failure.
      E_k_turb     : streaming-subtracted turbulent spectrum (disk box)
      E_k_full     : full COM-subtracted spectrum (disk box)
      E_k_apertures: dict {scale: E_k} for each aperture in VPS_APERTURE_SCALES
    """
    gas_fields = ['Masses', 'Coordinates', 'SmoothingLength', 'Velocities', 'Density']
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

    if 'Masses' not in pdata or 'Coordinates' not in pdata:
        return None

    time_kyr = _scale_to_Myr(float(hdr['Time'])) * 1e3

    try:
        is_disk, com, L_hat, r_cyl, z, v_phi, v_K, com_vel = identify_disk(
            pdata, stardata,
            r_search_kpc      = args.r_search,
            r_max_kpc         = args.r_max,
            rho_threshold_cgs = args.rho_thresh,
            aspect_ratio      = args.aspect,
            f_kep             = args.f_kep,
        )
    except Exception as e:
        print(f'  snap {snap_num:04d}: identify_disk error — {e}')
        return None

    if is_disk.sum() < 10:
        print(f'  snap {snap_num:04d}: too few disk particles ({is_disk.sum()}), skipping')
        return None

    # ── Rotate to face-on frame ───────────────────────────────────────────────
    rot = rotation_matrix_to_z(L_hat)

    pos_all  = pdata['Coordinates'] - com
    vel_all  = pdata['Velocities'] - (com_vel if com_vel is not None else 0.0)
    mass_all = pdata['Masses']
    hsml_all = pdata['SmoothingLength']

    half_box = args.image_box / 2.0
    dists    = np.linalg.norm(pos_all, axis=1)
    cut      = dists < half_box * 1.5
    if cut.sum() < 10:
        print(f'  snap {snap_num:04d}: too few particles in box ({cut.sum()}), skipping')
        return None

    pos_cut  = (pos_all[cut])  @ rot.T   # face-on: [x_fo, y_fo, z_fo]
    vel_cut  = (vel_all[cut])  @ rot.T   # COM-subtracted, face-on frame
    mass_cut = mass_all[cut]
    hsml_cut = hsml_all[cut]

    # ── Streaming subtraction ─────────────────────────────────────────────────
    r_xy     = np.linalg.norm(pos_cut[:, :2], axis=1)
    safe_rxy = np.maximum(r_xy, 1e-30)
    e_r_x    = pos_cut[:, 0] / safe_rxy
    e_r_y    = pos_cut[:, 1] / safe_rxy
    v_r_cut  =  vel_cut[:, 0] * e_r_x + vel_cut[:, 1] * e_r_y
    v_phi_cut= -vel_cut[:, 0] * e_r_y + vel_cut[:, 1] * e_r_x

    N_BINS  = 20
    r_outer = max(np.percentile(r_xy, 95), 1e-20)
    bins    = np.linspace(0.0, r_outer, N_BINS + 1)
    bidx    = np.clip(np.digitize(r_xy, bins) - 1, 0, N_BINS - 1)

    vr_prof   = np.zeros(N_BINS)
    vphi_prof = np.zeros(N_BINS)
    for b in range(N_BINS):
        mb = bidx == b
        if mb.sum() > 0:
            w = mass_cut[mb]; wsum = w.sum()
            vr_prof[b]   = np.dot(v_r_cut[mb],   w) / wsum
            vphi_prof[b] = np.dot(v_phi_cut[mb], w) / wsum

    # Turbulent residuals in face-on Cartesian frame
    dv_r   = v_r_cut   - vr_prof[bidx]
    dv_phi = v_phi_cut - vphi_prof[bidx]
    dv_x   = dv_r * e_r_x - dv_phi * e_r_y
    dv_y   = dv_r * e_r_y + dv_phi * e_r_x
    dv_z   = vel_cut[:, 2]   # no mean vertical streaming

    # ── 3D grid + power spectra ───────────────────────────────────────────────
    try:
        res_vps = min(getattr(args, 'vps_res', 128), 256)
        box_kpc = args.image_box

        M = Meshoid(pos_cut, mass_cut, hsml_cut)

        # Turbulent PS: streaming-subtracted (δv_r, δv_phi, v_z)
        dv_field       = np.column_stack([dv_x, dv_y, dv_z])
        k_AU, E_k_turb = _grid_and_ps(dv_field, pos_cut, mass_cut, hsml_cut,
                                       M, box_kpc, res_vps)

        # Full PS: COM-subtracted only (v_x, v_y, v_z)
        v_field = vel_cut
        _,   E_k_full  = _grid_and_ps(v_field, pos_cut, mass_cut, hsml_cut,
                                       M, box_kpc, res_vps)

        # Multi-aperture turbulent PS: compute for each scale in VPS_APERTURE_SCALES.
        # Particles are re-cut to each aperture sphere; k-grids differ between apertures.
        E_k_apertures = {}
        for scale in VPS_APERTURE_SCALES:
            ap_box  = box_kpc * scale
            ap_half = ap_box / 2.0
            ap_cut  = np.linalg.norm(pos_cut, axis=1) < ap_half * 1.5
            if ap_cut.sum() < 10:
                continue
            p_ap = pos_cut[ap_cut];  m_ap = mass_cut[ap_cut];  h_ap = hsml_cut[ap_cut]
            dv_ap = dv_field[ap_cut]
            try:
                M_ap = Meshoid(p_ap, m_ap, h_ap)
                k_ap, E_ap = _grid_and_ps(dv_ap, p_ap, m_ap, h_ap, M_ap, ap_box, res_vps)
                E_k_apertures[scale] = (k_ap, E_ap)
            except Exception:
                pass

    except Exception as e:
        print(f'  snap {snap_num:04d}: gridding/FFT error — {e}')
        return None

    r_disk_AU   = (np.percentile(r_xy[is_disk[cut]], 90) * kpc / AU
                   if is_disk[cut].sum() > 0 else np.nan)
    sml_mean_AU = (float(np.mean(hsml_cut[is_disk[cut]])) * kpc / AU
                   if is_disk[cut].sum() > 0 else np.nan)

    return k_AU, E_k_turb, E_k_full, E_k_apertures, time_kyr, r_disk_AU, sml_mean_AU, snap_num


def plot_vps(k_AU, E_k_turb, E_k_full,
             snap_num, time_kyr, r_disk_AU, sml_mean_AU,
             outpath, t1_kyr=None, E_k_apertures=None):
    """
    Log-log power spectrum plot.  Shows turbulent (streaming-subtracted) and full
    (COM-subtracted) spectra on the main disk aperture, plus optional aperture curves.
    """
    valid_t = (k_AU > 0) & (E_k_turb > 0)
    valid_f = (k_AU > 0) & (E_k_full  > 0)
    # Drop lowest-k bin (DC artefact) from both
    if valid_t.any(): valid_t[np.argmax(valid_t)] = False
    if valid_f.any(): valid_f[np.argmax(valid_f)] = False

    if not valid_t.any():
        print(f'  snap {snap_num:04d}: no valid turbulent E(k) bins, skipping plot')
        return

    k_inj  = 2.0 * np.pi / r_disk_AU  if np.isfinite(r_disk_AU)   and r_disk_AU   > 0 else None
    k_diss = 2.0 * np.pi / sml_mean_AU if np.isfinite(sml_mean_AU) and sml_mean_AU > 0 else None

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('w')
    ax.set_facecolor('w')

    # Aperture comparison curves — plotted first so main curves sit on top.
    # Colors from a perceptually uniform palette; linestyle dashed to distinguish from main.
    _ap_colors  = ['#e69f00', '#56b4e9', '#009e73', '#cc79a7']  # colorblind-safe
    _ap_ls      = ['--', '-.', (0,(3,1,1,1)), ':']
    if E_k_apertures:
        sorted_scales = sorted(E_k_apertures.keys())
        for _ci, sc in enumerate(sorted_scales):
            k_ap, E_ap = E_k_apertures[sc]
            v_ap = (k_ap > 0) & (E_ap > 0)
            if v_ap.any():
                v_ap[np.argmax(v_ap)] = False   # drop DC bin
            if not v_ap.any():
                continue
            _col = _ap_colors[_ci % len(_ap_colors)]
            _ls  = _ap_ls[_ci % len(_ap_ls)]
            ax.loglog(k_ap[v_ap], E_ap[v_ap],
                      color=_col, lw=2.8, ls=_ls, alpha=0.85,
                      label=rf'$E_\mathrm{{turb}}$ aperture ×{sc:.1g}')

    # Full velocity PS (dimmer, behind main turbulent)
    if valid_f.any():
        ax.loglog(k_AU[valid_f], E_k_full[valid_f], color='steelblue', lw=3.0,
                  alpha=0.7, label=r'$E(k)$ full (no streaming sub.)')

    # Turbulent PS (primary, disk aperture)
    ax.loglog(k_AU[valid_t], E_k_turb[valid_t], 'k-', lw=4.0,
              label=r'$E(k)$ turbulent disk (streaming sub.)')

    # Power-law fit in the inertial range (turbulent disk spectrum only)
    k_lo = k_inj  if k_inj  is not None else k_AU[valid_t][0]
    k_hi = k_diss if k_diss is not None else k_AU[valid_t][-1]
    fit_mask = valid_t & (k_AU >= k_lo) & (k_AU <= k_hi)
    if fit_mask.sum() >= 3:
        alpha_fit, log_A = np.polyfit(np.log(k_AU[fit_mask]), np.log(E_k_turb[fit_mask]), 1)
        A_fit = np.exp(log_A)
        k_fit_arr = k_AU[fit_mask]
        ax.loglog(k_fit_arr, A_fit * k_fit_arr**alpha_fit, 'r-', lw=5.0,
                  label=rf'fit (inertial): $E \propto k^{{{alpha_fit:.2f}}}$')

    # Reference slopes anchored at injection scale
    if k_inj is not None:
        E_at_inj = float(np.interp(k_inj, k_AU[valid_t], E_k_turb[valid_t]))
        k_ref = k_AU[valid_t]
        ax.loglog(k_ref, E_at_inj * (k_ref / k_inj)**(-5/3), 'c--', lw=2.4,
                  alpha=0.8, label=r'Kolmogorov $k^{-5/3}$')
        ax.loglog(k_ref, E_at_inj * (k_ref / k_inj)**(-2),   'm--', lw=2.4,
                  alpha=0.8, label=r'Burgers $k^{-2}$')
        ax.loglog(k_ref, E_at_inj * (k_ref / k_inj)**(-3),   color='olive', lw=2.4,
                  alpha=0.8, label=r'Kraichnan 2D $k^{-3}$')

    # Vertical lines: injection (goldenrod) and dissipation (darkorange)
    if k_inj is not None:
        ax.axvline(k_inj,  color='goldenrod', ls=':', lw=2.4,
                   label=f'injection ($r_{{disk}}={r_disk_AU:.0f}$ AU)')
    if k_diss is not None:
        ax.axvline(k_diss, color='darkorange', ls=':', lw=2.4,
                   label=f'dissipation (SML$={sml_mean_AU:.0f}$ AU)')

    if t1_kyr is not None:
        title = rf'Snap {snap_num:04d}   $t - t_1 = {time_kyr - t1_kyr:.2f}$ kyr'
    else:
        title = rf'Snap {snap_num:04d}   $t = {time_kyr:.2f}$ kyr'
    ax.set_xlabel(r'$k$ (AU$^{-1}$)', color='k', fontsize=12)
    ax.set_ylabel(r'$E(k)$  (km/s)$^2$ AU$^3$  [3D spherical avg]', color='k', fontsize=12)
    ax.tick_params(colors='k', which='both', direction='in', right=True, top=True)
    for sp in ax.spines.values():
        sp.set_edgecolor('k')

    leg = ax.legend(fontsize=8, framealpha=0.8, facecolor='w')
    for txt in leg.get_texts():
        txt.set_color('k')

    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=120, facecolor='w')
    # Dark version
    dark_path = outpath.replace('/light/', '/dark/')
    if dark_path != outpath:
        os.makedirs(os.path.dirname(dark_path), exist_ok=True)
        _darken_fig(fig)
        fig.savefig(dark_path, dpi=120, facecolor='#181818')
    plt.close(fig)


def _find_t1_kyr(snap_items, path, sim):
    """Scan snapshots for earliest StellarFormationTime → t1 in kyr."""
    import h5py
    t1 = None
    for snap_path, _ in snap_items:
        try:
            with h5py.File(snap_path, 'r') as f:
                if ('PartType5' in f
                        and 'StellarFormationTime' in f['PartType5']
                        and f['PartType5/StellarFormationTime'].shape[0] > 0):
                    a = float(f['PartType5/StellarFormationTime'][:].min())
                    t = _scale_to_Myr(a) * 1e3
                    if t1 is None or t < t1:
                        t1 = t
        except Exception:
            pass
    return t1


def plot_all_vps(args):
    """Callable from jupytertest3.py. Processes all snapshots."""
    snap_pattern = os.path.join(args.path, args.sim, 'snapshot_*.hdf5')
    snap_paths   = sorted(glob.glob(snap_pattern))
    if not snap_paths:
        print(f'No snapshots found: {snap_pattern}')
        return

    def _snap_num(p):
        return int(os.path.basename(p).replace('snapshot_', '').replace('.hdf5', ''))

    snap_items = [(p, _snap_num(p)) for p in snap_paths]
    if getattr(args, 'snap_start', None) is not None:
        snap_items = [(p, n) for p, n in snap_items if n >= args.snap_start]
    if getattr(args, 'snap_end', None) is not None:
        snap_items = [(p, n) for p, n in snap_items if n <= args.snap_end]

    # Separate subdirectories for plots and raw data
    outdir_plots = os.path.join(args.outdir, 'light', 'velocity_power_spectra', 'plots')
    outdir_data  = os.path.join(args.outdir, 'velocity_power_spectra', 'data')
    os.makedirs(outdir_plots, exist_ok=True)
    os.makedirs(outdir_data,  exist_ok=True)

    print('Finding t1...')
    t1_kyr = _find_t1_kyr(snap_items, args.path, args.sim)
    print(f't1 = {t1_kyr:.2f} kyr' if t1_kyr is not None else 'No sinks found.')

    for i, (snap_path, snap_num) in enumerate(snap_items):
        outpath_png = os.path.join(outdir_plots, f'vps_{snap_num:04d}.png')
        outpath_npz = os.path.join(outdir_data,  f'vps_{snap_num:04d}.npz')

        if os.path.exists(outpath_png):
            print(f'  snap {snap_num:04d}: exists, skipping')
            continue

        print(f'  snap {snap_num:04d}: processing [{i+1}/{len(snap_items)}]...', flush=True)
        result = process_snap(args, snap_path, snap_num)
        if result is None:
            continue

        k_AU, E_k_turb, E_k_full, E_k_apertures, time_kyr, r_disk_AU, sml_mean_AU, _sn = result

        # Build savez kwargs: add per-aperture arrays with key 'ap_<scale>'
        npz_kw = dict(
            k_AU        = k_AU,
            E_k_turb    = E_k_turb,
            E_k_full    = E_k_full,
            time_kyr    = np.array([time_kyr]),
            r_disk_AU   = np.array([r_disk_AU]),
            sml_mean_AU = np.array([sml_mean_AU]),
        )
        for sc, (k_ap, E_ap) in E_k_apertures.items():
            sc_str = f'{sc:.1g}'.replace('.', 'p')
            npz_kw[f'ap_{sc_str}_k']   = k_ap
            npz_kw[f'ap_{sc_str}_E']   = E_ap
        np.savez(outpath_npz, **npz_kw)

        try:
            plot_vps(k_AU, E_k_turb, E_k_full,
                     snap_num, time_kyr, r_disk_AU, sml_mean_AU,
                     outpath_png, t1_kyr=t1_kyr, E_k_apertures=E_k_apertures)
            if os.path.exists(outpath_png):
                print(f'  snap {snap_num:04d}  t={time_kyr:.2f} kyr  saved → {outpath_png}',
                      flush=True)
        except Exception as e:
            print(f'  snap {snap_num:04d}: plot_vps error — {e}', flush=True)

    print(f'\nDone → {outdir_plots}')
    plot_resolution_check(outdir_data, t1_kyr, outdir_plots)


def plot_resolution_check(data_dir, t1_kyr, plots_dir):
    """Plot r_disk / SML_mean vs time. Reads npz from data_dir, saves png to plots_dir."""
    npz_files = sorted(glob.glob(os.path.join(data_dir, 'vps_*.npz')))
    if not npz_files:
        print('  resolution check: no npz files found')
        return

    times, ratios, r_disks, smls = [], [], [], []
    for f in npz_files:
        d = np.load(f)
        t   = float(d['time_kyr'])
        r   = float(d['r_disk_AU'])
        sml = float(d['sml_mean_AU'])
        if np.isfinite(r) and np.isfinite(sml) and sml > 0:
            times.append(t - t1_kyr if t1_kyr is not None else t)
            ratios.append(r / sml)
            r_disks.append(r)
            smls.append(sml)

    if not times:
        print('  resolution check: no valid data')
        return

    times   = np.array(times)
    ratios  = np.array(ratios)
    r_disks = np.array(r_disks)
    smls    = np.array(smls)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    fig.patch.set_facecolor('w')
    xlabel = r'$t - t_1$ (kyr)' if t1_kyr is not None else 'Time (kyr)'

    ax1 = axes[0]
    ax1.set_facecolor('w')
    ax1.semilogy(times, ratios, '#222222', lw=4.0, label=r'$r_{\rm disk}\ /\ \langle h \rangle$')
    ax1.axhline(10, color='r', ls='--', lw=2.0, label='ratio = 10  (resolved)')
    ax1.set_ylabel(r'$r_{\rm disk}\ /\ \langle h \rangle$', color='k', fontsize=12)
    ax1.tick_params(colors='k', which='both', direction='in', right=True, top=True)
    for sp in ax1.spines.values(): sp.set_edgecolor('k')
    leg = ax1.legend(fontsize=9, framealpha=0.8, facecolor='w')
    for t in leg.get_texts(): t.set_color('k')

    ax2 = axes[1]
    ax2.set_facecolor('w')
    ax2.semilogy(times, r_disks, 'c-',  lw=4.0, label=r'$r_{\rm disk}$ (AU)')
    ax2.semilogy(times, smls,    'm--', lw=4.0, label=r'$\langle h \rangle$ (AU)')
    ax2.set_xlabel(xlabel, color='k', fontsize=12)
    ax2.set_ylabel('Scale (AU)', color='k', fontsize=12)
    ax2.tick_params(colors='k', which='both', direction='in', right=True, top=True)
    for sp in ax2.spines.values(): sp.set_edgecolor('k')
    leg2 = ax2.legend(fontsize=9, framealpha=0.8, facecolor='w')
    for t in leg2.get_texts(): t.set_color('k')

    plt.tight_layout()
    outpath = os.path.join(plots_dir, 'resolution_check.png')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=150, facecolor='w')
    # Dark version
    dark_path = outpath.replace('/light/', '/dark/')
    if dark_path != outpath:
        os.makedirs(os.path.dirname(dark_path), exist_ok=True)
        _darken_fig(fig)
        fig.savefig(dark_path, dpi=150, facecolor='#181818')
    plt.close(fig)
    print(f'  Resolution check saved → {outpath}')


if __name__ == '__main__':
    import argparse

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
    p.add_argument('--image-box',  type=float, default=2e-5,
                   help='Face-on box full width [kpc]')
    p.add_argument('--res',        type=int,   default=400,
                   help='Grid resolution for surface density maps')
    p.add_argument('--vps-res',    type=int,   default=128,
                   help='Grid resolution for 3D velocity FFT (capped at 256)')
    p.add_argument('--snap-start', type=int,   default=None)
    p.add_argument('--snap-end',   type=int,   default=None)
    args = p.parse_args()
    plot_all_vps(args)
