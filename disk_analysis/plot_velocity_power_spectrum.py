"""
plot_velocity_power_spectrum.py
-------------------------------
Compute and plot the 3D velocity power spectrum of disk gas.

For each snapshot:
  1. Identify disk and rotate to face-on frame
  2. Subtract streaming velocity (mass-weighted radial profiles of v_r, v_phi)
  3. Grid all three turbulent velocity components (δv_x, δv_y, δv_z) onto a
     3D cube via Meshoid.InterpToGrid (volumetric interpolation)
  4. Compute 3D FFT and spherical-shell average |v(k)|² → E(k) using
     GetPowerSpectrum from starforge_tools/2point_statistics/ComputePowerSpectrum.py
  5. Plot E(k) vs k with Kolmogorov k^{-5/3}, Burgers k^{-2}, Kraichnan k^{-3}
     reference slopes, injection and dissipation scales marked

Output: {outdir}/velocity_power_spectra/vps_XXXX.png
"""

import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def process_snap(args, snap_path, snap_num):
    """Return (k_AU, E_k, time_kyr, r_disk_AU, sml_mean_AU) or None on failure."""
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

    pos_all = pdata['Coordinates'] - com
    vel_all = pdata['Velocities']  - (com_vel if com_vel is not None else 0.0)
    mass_all = pdata['Masses']
    hsml_all = pdata['SmoothingLength']

    # Work within image_box_kpc (small box)
    half_box = args.image_box / 2.0
    dists    = np.linalg.norm(pos_all, axis=1)
    cut      = dists < half_box * 1.5
    if cut.sum() < 10:
        print(f'  snap {snap_num:04d}: too few particles in box ({cut.sum()}), skipping')
        return None

    pos_cut  = (pos_all[cut])  @ rot.T   # face-on: [x_fo, y_fo, z_fo]
    vel_cut  = (vel_all[cut])  @ rot.T
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
    r_outer = np.percentile(r_xy, 95)
    r_outer = max(r_outer, 1e-20)
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

    # Turbulent velocity components in face-on Cartesian frame
    dv_r   = v_r_cut   - vr_prof[bidx]
    dv_phi = v_phi_cut - vphi_prof[bidx]
    # Convert back to Cartesian face-on (x, y)
    dv_x = dv_r * e_r_x - dv_phi * e_r_y
    dv_y = dv_r * e_r_y + dv_phi * e_r_x
    # Vertical component: no mean rotation to subtract, just remove COM offset
    dv_z = vel_cut[:, 2]

    # ── 3D grid via Meshoid.InterpToGrid + GetPowerSpectrum ──────────────────
    # Uses starforge_tools approach: volumetric 3D interpolation → 3D FFT →
    # spherical-shell average.  Resolution capped at 128 to limit memory (~2 GB).
    try:
        res_vps     = min(getattr(args, 'vps_res', 128), 256)
        box_kpc     = args.image_box
        center0     = np.zeros(3)
        gridsize_AU = box_kpc * kpc / AU

        # δv field: shape (N, 3)
        dv = np.column_stack([dv_x, dv_y, dv_z])

        M = Meshoid(pos_cut, mass_cut, hsml_cut)
        # InterpToGrid returns (res, res, res, 3) for a vector field
        vgrid = M.InterpToGrid(dv, size=box_kpc, res=res_vps, center=center0)
        vgrid = np.rollaxis(vgrid, -1, 0)   # → (3, res, res, res)

        # 3D FFT, spherical k-shell average
        k_int, power = GetPowerSpectrum(vgrid, res_vps)

        # Physical k [1/AU]; normalize by voxel volume so E(k) ~ (km/s)² AU³
        dx_AU = gridsize_AU / res_vps
        k_AU  = k_int * 2.0 * np.pi / gridsize_AU
        E_k   = power * dx_AU ** 3

    except Exception as e:
        print(f'  snap {snap_num:04d}: gridding/FFT error — {e}')
        return None

    r_disk_AU   = (np.percentile(r_xy[is_disk[cut]], 90) * kpc / AU
                   if is_disk[cut].sum() > 0 else np.nan)
    sml_mean_AU = (float(np.mean(hsml_cut[is_disk[cut]])) * kpc / AU
                   if is_disk[cut].sum() > 0 else np.nan)

    return k_AU, E_k, time_kyr, r_disk_AU, sml_mean_AU, snap_num


def plot_vps(k_AU, E_k, snap_num, time_kyr, r_disk_AU, sml_mean_AU, outpath, t1_kyr=None):
    """Log-log power spectrum plot with reference slopes and scale markers."""
    valid = (k_AU > 0) & (E_k > 0)
    valid[np.argmax(valid)] = False   # drop lowest-k point (DC artefact)
    if not valid.any():
        print(f'  snap {snap_num:04d}: no valid E(k) bins, skipping plot')
        return

    # Angular wavenumber k = 2π/λ, consistent with k_AU = k_int * 2π / gridsize_AU
    k_inj  = 2.0 * np.pi / r_disk_AU  if np.isfinite(r_disk_AU)   and r_disk_AU   > 0 else None
    k_diss = 2.0 * np.pi / sml_mean_AU if np.isfinite(sml_mean_AU) and sml_mean_AU > 0 else None

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('k')
    ax.set_facecolor('k')

    ax.loglog(k_AU[valid], E_k[valid], 'w-', lw=2, label='E(k)')

    # Power-law fit in the inertial range: k_inj ≤ k ≤ k_diss
    k_lo = k_inj  if k_inj  is not None else k_AU[valid][0]
    k_hi = k_diss if k_diss is not None else k_AU[valid][-1]
    fit_mask = valid & (k_AU >= k_lo) & (k_AU <= k_hi)
    if fit_mask.sum() >= 3:
        alpha_fit, log_A = np.polyfit(np.log(k_AU[fit_mask]), np.log(E_k[fit_mask]), 1)
        A_fit = np.exp(log_A)
        k_fit_arr = k_AU[fit_mask]
        ax.loglog(k_fit_arr, A_fit * k_fit_arr**alpha_fit, 'r-', lw=2.5,
                  label=rf'fit (inertial): $E \propto k^{{{alpha_fit:.2f}}}$')

    # Reference slopes anchored at injection scale
    if k_inj is not None:
        E_at_inj = float(np.interp(k_inj, k_AU[valid], E_k[valid]))
        k_ref = k_AU[valid]
        ax.loglog(k_ref, E_at_inj * (k_ref / k_inj)**(-5/3), 'c--', lw=1.2,
                  alpha=0.8, label=r'Kolmogorov $k^{-5/3}$')
        ax.loglog(k_ref, E_at_inj * (k_ref / k_inj)**(-2),   'm--', lw=1.2,
                  alpha=0.8, label=r'Burgers $k^{-2}$')
        ax.loglog(k_ref, E_at_inj * (k_ref / k_inj)**(-3),   'y--', lw=1.2,
                  alpha=0.8, label=r'Kraichnan 2D $k^{-3}$')

    # Vertical lines: injection and dissipation scales
    if k_inj is not None:
        ax.axvline(k_inj,  color='yellow', ls=':', lw=1,
                   label=f'injection ($r_{{disk}}$={r_disk_AU:.0f} AU)')
    if k_diss is not None:
        ax.axvline(k_diss, color='orange', ls=':', lw=1,
                   label=f'dissipation (SML={sml_mean_AU:.0f} AU)')

    if t1_kyr is not None:
        dt = time_kyr - t1_kyr
        title = rf'Snap {snap_num:04d}   $t - t_1 = {dt:.2f}$ kyr'
    else:
        title = rf'Snap {snap_num:04d}   $t = {time_kyr:.2f}$ kyr'
    ax.set_title(title, color='w', fontsize=12)
    ax.set_xlabel(r'$k$ (AU$^{-1}$)', color='w', fontsize=12)
    ax.set_ylabel(r'$E(k)$  (km/s)$^2$ AU$^3$  [3D spherical avg]', color='w', fontsize=12)
    ax.tick_params(colors='w', which='both', direction='in', right=True, top=True)
    for sp in ax.spines.values():
        sp.set_edgecolor('w')

    leg = ax.legend(fontsize=9, framealpha=0.3)
    for txt in leg.get_texts():
        txt.set_color('w')

    plt.tight_layout()
    fig.savefig(outpath, dpi=120, facecolor='k')
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

    outdir_vps = os.path.join(args.outdir, 'velocity_power_spectra')
    os.makedirs(outdir_vps, exist_ok=True)

    print(f'Finding t1...')
    t1_kyr = _find_t1_kyr(snap_items, args.path, args.sim)
    print(f't1 = {t1_kyr:.2f} kyr' if t1_kyr is not None else 'No sinks found.')

    for i, (snap_path, snap_num) in enumerate(snap_items):
        outpath = os.path.join(outdir_vps, f'vps_{snap_num:04d}.png')
        if os.path.exists(outpath):
            print(f'  snap {snap_num:04d}: exists, skipping')
            continue

        print(f'  snap {snap_num:04d}: processing [{i+1}/{len(snap_items)}]...', flush=True)
        result = process_snap(args, snap_path, snap_num)
        if result is None:
            continue

        k_AU, E_k, time_kyr, r_disk_AU, sml_mean_AU, _sn = result
        # Save per-snap data for resolution check plot
        np.savez(os.path.join(outdir_vps, f'vps_{snap_num:04d}.npz'),
                 k_AU=k_AU, E_k=E_k,
                 time_kyr=np.array([time_kyr]),
                 r_disk_AU=np.array([r_disk_AU]),
                 sml_mean_AU=np.array([sml_mean_AU]))
        try:
            plot_vps(k_AU, E_k, snap_num, time_kyr, r_disk_AU, sml_mean_AU,
                     outpath, t1_kyr=t1_kyr)
            if os.path.exists(outpath):
                print(f'  snap {snap_num:04d}  t={time_kyr:.2f} kyr  saved → {outpath}',
                      flush=True)
        except Exception as e:
            print(f'  snap {snap_num:04d}: plot_vps error — {e}', flush=True)

    print(f'\nDone → {outdir_vps}')
    plot_resolution_check(outdir_vps, t1_kyr, args.outdir)


def plot_resolution_check(vps_dir, t1_kyr, outdir):
    """Plot r_disk / SML_mean vs time from saved npz files."""
    npz_files = sorted(glob.glob(os.path.join(vps_dir, 'vps_*.npz')))
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
    fig.patch.set_facecolor('k')

    xlabel = r'$t - t_1$ (kyr)' if t1_kyr is not None else 'Time (kyr)'

    ax1 = axes[0]
    ax1.set_facecolor('k')
    ax1.semilogy(times, ratios, 'w-', lw=2, label=r'$r_{\rm disk}\ /\ \langle h \rangle$')
    ax1.axhline(10, color='r', ls='--', lw=1, label='ratio = 10  (resolved)')
    ax1.set_ylabel(r'$r_{\rm disk}\ /\ \langle h \rangle$', color='w', fontsize=12)
    ax1.set_title('Numerical resolution check', color='w', fontsize=13)
    ax1.tick_params(colors='w', which='both', direction='in', right=True, top=True)
    for sp in ax1.spines.values(): sp.set_edgecolor('w')
    leg = ax1.legend(fontsize=9, framealpha=0.3)
    for t in leg.get_texts(): t.set_color('w')

    ax2 = axes[1]
    ax2.set_facecolor('k')
    ax2.semilogy(times, r_disks, 'c-',  lw=2, label=r'$r_{\rm disk}$ (AU)')
    ax2.semilogy(times, smls,    'm--', lw=2, label=r'$\langle h \rangle$ (AU)')
    ax2.set_xlabel(xlabel, color='w', fontsize=12)
    ax2.set_ylabel('Scale (AU)', color='w', fontsize=12)
    ax2.tick_params(colors='w', which='both', direction='in', right=True, top=True)
    for sp in ax2.spines.values(): sp.set_edgecolor('w')
    leg2 = ax2.legend(fontsize=9, framealpha=0.3)
    for t in leg2.get_texts(): t.set_color('w')

    plt.tight_layout()
    outpath = os.path.join(outdir, 'resolution_check.png')
    fig.savefig(outpath, dpi=150, facecolor='k')
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
                   help='Grid resolution for velocity projection')
    p.add_argument('--snap-start', type=int,   default=None)
    p.add_argument('--snap-end',   type=int,   default=None)
    args = p.parse_args()
    plot_all_vps(args)
