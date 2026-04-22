"""
make_disk_movie_frames.py
-------------------------
Produces per-snapshot PNG frames showing face-on and edge-on surface density maps
with the identified disk highlighted. Run from the popIII_analysis directory or
anywhere with GUAC on the Python path.

Output: one PNG per snapshot in --outdir (default: ../plots/disk_movie_frames/)
Naming: frame_XXXXXX.png  (zero-padded snapshot number for easy ffmpeg ordering)

Example ffmpeg command to assemble the PNGs into a movie afterwards:
    ffmpeg -framerate 10 -pattern_type glob -i 'frame_*.png' \
           -c:v libx264 -crf 18 -pix_fmt yuv420p disk_movie.mp4
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')   # non-interactive backend for batch rendering
import matplotlib.pyplot as plt
from matplotlib import colors

import numpy as np
from meshoid import Meshoid

# ── GUAC imports ──────────────────────────────────────────────────────────────
from generic_utils.fire_utils import *
from generic_utils.constants import *        # kpc, AU, Msun, G (CGS floats)
from hybrid_sims_utils.read_snap import *

try:
    from astropy.cosmology import Planck18 as _cosmo
    import astropy.units as _u_astropy
    def _scale_to_Myr(a):
        """Convert GIZMO scale factor → cosmic time in Myr (Planck18 cosmology)."""
        return float(_cosmo.age(1.0 / float(a) - 1.0).to(_u_astropy.Myr).value)
except ImportError:
    print('WARNING: astropy not available; times will be in scale-factor units.')
    def _scale_to_Myr(a):
        return float(a)


# ═══════════════════════════════════════════════════════════════════════════════
# Disk identification helpers (copied from DiskIdentification.ipynb)
# ═══════════════════════════════════════════════════════════════════════════════

def find_center(pdata, stardata, reference_center=None, reference_search_radius=None):
    if stardata and len(stardata.get('Masses', [])) > 0:
        # Use the most massive sink as center — stable when secondary sinks form
        # far from the primary disk (COM would jump to empty space between them)
        idx = np.argmax(stardata['Masses'])
        com = stardata['Coordinates'][idx]
    else:
        pos = pdata['Coordinates']
        if reference_center is not None and reference_search_radius is not None:
            dists = np.linalg.norm(pos - reference_center, axis=1)
            mask = dists < reference_search_radius
            if mask.sum() > 0:
                idx = np.argmax(pdata['Density'][mask])
                com = pos[mask][idx]
            else:
                idx = np.argmax(pdata['Density'])
                com = pos[idx]
        else:
            idx = np.argmax(pdata['Density'])
            com = pos[idx]
    return com


def get_disk_axis(gas_pos_kpc, gas_vel_kms, gas_masses_Msun, r_search_kpc):
    dists = np.linalg.norm(gas_pos_kpc, axis=1)
    mask  = dists < r_search_kpc
    if mask.sum() < 4:
        return np.array([0., 0., 1.])
    pos_cm  = gas_pos_kpc[mask] * kpc
    vel_cms = gas_vel_kms[mask] * 1e5
    m_g     = gas_masses_Msun[mask] * Msun
    L       = np.sum(m_g[:, None] * np.cross(pos_cm, vel_cms), axis=0)
    L_mag   = np.linalg.norm(L)
    if L_mag == 0:
        return np.array([0., 0., 1.])
    L_hat = L / L_mag
    # Ensure consistent hemisphere — prevents frame-to-frame L_hat flipping
    # (e.g. snap 209 flicker caused by L vector transiently crossing z=0 plane)
    if L_hat[2] < 0:
        L_hat = -L_hat
    return L_hat


def cylindrical_coords(pos_kpc, vel_kms, L_hat):
    z_kpc     = pos_kpc @ L_hat
    v_z_kms   = vel_kms @ L_hat
    r_perp    = pos_kpc - z_kpc[:, None] * L_hat
    r_cyl_kpc = np.linalg.norm(r_perp, axis=1)
    safe_r    = np.maximum(r_cyl_kpc, 1e-30)
    e_r       = r_perp / safe_r[:, None]
    e_phi     = np.cross(L_hat, e_r)
    v_r_kms   = np.einsum('ij,ij->i', vel_kms, e_r)
    v_phi_kms = np.einsum('ij,ij->i', vel_kms, e_phi)
    return r_cyl_kpc, z_kpc, v_phi_kms, v_r_kms, v_z_kms


def compute_M_enc(r_cyl_kpc, gas_masses_Msun, M_stars_Msun):
    """
    Enclosed mass per gas particle (gas cumsum + stellar contribution).

    M_stars_Msun : scalar  – all stellar mass placed at r=0 (only correct for
                             a single central sink; was the original behaviour)
                   ndarray – per-particle enclosed stellar mass [Msun], so each
                             sink only contributes gravity to particles outside
                             its own orbital radius.
    """
    sort_idx      = np.argsort(r_cyl_kpc)
    sorted_masses = gas_masses_Msun[sort_idx]
    gas_cumsum    = np.concatenate([[0.0], np.cumsum(sorted_masses[:-1])])
    star_contrib  = M_stars_Msun[sort_idx] if np.ndim(M_stars_Msun) > 0 else M_stars_Msun
    M_enc_sorted  = (star_contrib + gas_cumsum) * Msun
    M_enc         = np.empty(len(r_cyl_kpc))
    M_enc[sort_idx] = M_enc_sorted
    return M_enc


def extend_disk_to_bounds(is_disk_kinematic, r_cyl_kpc, z_kpc, percentile=100):
    if is_disk_kinematic.sum() == 0:
        return is_disk_kinematic.copy(), 0.0, 0.0
    R_bound = np.percentile(r_cyl_kpc[is_disk_kinematic], percentile)
    H_bound = np.percentile(np.abs(z_kpc[is_disk_kinematic]), percentile)
    is_disk_bounded = (r_cyl_kpc < R_bound) & (np.abs(z_kpc) < H_bound)
    return is_disk_bounded, R_bound, H_bound


def identify_disk(pdata, stardata,
                  r_search_kpc      = 1e-5,
                  r_max_kpc         = 1e-5,
                  rho_threshold_cgs = 1e-15,
                  aspect_ratio      = 0.3,
                  f_kep             = 0.3,
                  use_bounds        = True,
                  bounds_percentile = 100,
                  reference_center  = None,
                  reference_search_radius = None):
    com       = find_center(pdata, stardata, reference_center, reference_search_radius)
    gas_pos_kpc_all = pdata['Coordinates'] - com
    gas_dists_all   = np.linalg.norm(gas_pos_kpc_all, axis=1)

    # ── Pre-filter to local region before expensive per-particle ops ──────────
    # The full FIRE sim can have millions of particles; we only need those near
    # the disk. r_max * 5 safely encloses enough mass for M_enc to be accurate.
    r_local   = max(r_max_kpc * 5, r_search_kpc * 2)
    local     = gas_dists_all < r_local

    gas_pos_kpc     = gas_pos_kpc_all[local]
    gas_masses_Msun = pdata['Masses'][local] * 1e10
    gas_vel         = pdata['Velocities'][local]
    gas_dens        = pdata['Density'][local]

    search_mask = np.linalg.norm(gas_pos_kpc, axis=1) < r_search_kpc
    if search_mask.sum() > 0:
        com_vel = (np.sum(gas_vel[search_mask] * gas_masses_Msun[search_mask, None], axis=0)
                   / np.sum(gas_masses_Msun[search_mask]))
    else:
        com_vel = np.zeros(3)

    gas_vel_com = gas_vel - com_vel
    L_hat       = get_disk_axis(gas_pos_kpc, gas_vel_com, gas_masses_Msun, r_search_kpc)

    r_cyl_kpc, z_kpc, v_phi_kms, v_r_kms, v_z_kms = cylindrical_coords(
        gas_pos_kpc, gas_vel_com, L_hat)

    M_stars_Msun = (np.sum(stardata['Masses']) * 1e10
                    if stardata and len(stardata.get('Masses', [])) > 0 else 0.0)
    M_enc_g  = compute_M_enc(r_cyl_kpc, gas_masses_Msun, M_stars_Msun)
    r_cyl_cm = np.maximum(r_cyl_kpc * kpc, 1e-10)
    v_K_kms  = np.sqrt(G * M_enc_g / r_cyl_cm) / 1e5

    rho_gcm3   = gas_dens.astype(np.float64) * 1e10 * Msun / kpc**3
    safe_r_cyl = np.maximum(r_cyl_kpc, 1e-30)

    is_disk_local = (
        (r_cyl_kpc < r_max_kpc) &
        (np.abs(z_kpc) / safe_r_cyl < aspect_ratio) &
        (rho_gcm3 > rho_threshold_cgs) &
        (v_phi_kms > 0) &
        (v_phi_kms / np.maximum(v_K_kms, 1e-10) > f_kep)
    )

    if use_bounds:
        is_disk_local, _, _ = extend_disk_to_bounds(
            is_disk_local, r_cyl_kpc, z_kpc, percentile=bounds_percentile)

    # ── Map back to full particle array ───────────────────────────────────────
    N_all         = len(pdata['Masses'])
    is_disk       = np.zeros(N_all, dtype=bool)
    r_cyl_out     = np.zeros(N_all)
    z_out         = np.zeros(N_all)
    v_phi_out     = np.zeros(N_all)
    v_K_out       = np.zeros(N_all)

    is_disk[local]   = is_disk_local
    r_cyl_out[local] = r_cyl_kpc
    z_out[local]     = z_kpc
    v_phi_out[local] = v_phi_kms
    v_K_out[local]   = v_K_kms

    return is_disk, com, L_hat, r_cyl_out, z_out, v_phi_out, v_K_out, com_vel


# ═══════════════════════════════════════════════════════════════════════════════
# Rotation matrix and frame rendering
# ═══════════════════════════════════════════════════════════════════════════════

def rotation_matrix_to_z(L_hat):
    z_hat = np.array([0., 0., 1.])
    v     = np.cross(L_hat, z_hat)
    s     = np.linalg.norm(v)
    c     = np.dot(L_hat, z_hat)
    if s > 1e-10:
        vx  = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot = np.eye(3) + vx + vx @ vx * (1 - c) / s**2
    else:
        rot = np.eye(3) if c > 0 else -np.eye(3)
    return rot


def render_frame(pdata, stardata, snap_num, time_Myr,
                 is_disk, com, L_hat,
                 image_box_kpc, res,
                 vmin, vmax, cmap,
                 outpath,
                 com_vel=None,
                 corotate=True,
                 vmax_vel=None,
                 v_K=None,
                 data_outdir=None,
                 include_phase=False,
                 h2_field=None,
                 sink_form_Myr=None,
                 sink_r_AU=None):
    """
    Render a 3×4 panel and save to outpath.
      Row 0: face-on SD (small) | face-on SD (10×) | edge-on SD (clean) | edge-on SD (disk overlay)
      Row 1: face-on |δv|       | face-on σ_|δv|   | edge-on |δv|       | edge-on σ_|δv|
      Row 2: v_r vs r phase plot | v_phi vs r phase plot | Q vs r | face-on Toomre Q map

    corotate : if True (default), the face-on view rotates with the disk —
               the most massive sink is pinned to the +x axis each frame, so
               a rigid rotating disk appears frozen.
    vmax_vel : fixed colorbar ceiling [km/s] for all velocity panels.
               Set to a finite value to prevent per-frame auto-scaling from
               causing flicker in the assembled movie.  None → auto (99th pct).
    """
    rot = rotation_matrix_to_z(L_hat)

    # ── Co-rotating frame for face-on panels ─────────────────────────────────
    # Pin the most massive sink to the +x axis so the disk appears stationary.
    phi_ref = 0.0
    if corotate and stardata and len(stardata.get('Masses', [])) > 0:
        idx_ms   = np.argmax(stardata['Masses'])
        ref_disk = (stardata['Coordinates'][idx_ms] - com) @ rot.T
        phi_ref  = np.arctan2(ref_disk[1], ref_disk[0])
    c_phi, s_phi = np.cos(-phi_ref), np.sin(-phi_ref)
    R_ip  = np.array([[c_phi, -s_phi, 0.], [s_phi, c_phi, 0.], [0., 0., 1.]])
    rot_fo = R_ip @ rot    # face-on rotation (includes co-rotation)
    # Edge-on uses plain rot — in-plane rotation doesn't change the edge-on view

    gas_dists = np.linalg.norm(pdata['Coordinates'] - com, axis=1)

    # Small-box particles
    cut_small  = gas_dists < image_box_kpc * 0.75
    coords_s   = pdata['Coordinates'][cut_small] - com
    pos_fo     = coords_s @ rot_fo.T   # face-on, co-rotated
    pos_small  = coords_s @ rot.T      # edge-on basis
    mass_small = pdata['Masses'][cut_small]
    hsml_small = pdata['SmoothingLength'][cut_small]
    disk_small = is_disk[cut_small]
    pos_edge   = pos_small[:, [0, 2, 1]]   # swap y↔z for edge-on projection

    # Large-box particles (10× zoomed-out face-on panel)
    cut_large    = gas_dists < image_box_kpc * 10 * 0.75
    pos_fo_large = (pdata['Coordinates'][cut_large] - com) @ rot_fo.T
    mass_large   = pdata['Masses'][cut_large]
    hsml_large   = pdata['SmoothingLength'][cut_large]

    center0         = np.zeros(3)
    extent_AU       = image_box_kpc      * kpc / AU
    extent_AU_large = image_box_kpc * 10 * kpc / AU
    half_AU         = extent_AU       / 2
    half_AU_large   = extent_AU_large / 2

    ax_AU       = np.linspace(-half_AU,       half_AU,       res)
    ax_AU_large = np.linspace(-half_AU_large, half_AU_large, res)
    X,  Y  = np.meshgrid(ax_AU,       ax_AU,       indexing='ij')
    XL, YL = np.meshgrid(ax_AU_large, ax_AU_large, indexing='ij')

    norm_small = colors.LogNorm(vmin=vmin,        vmax=vmax)
    norm_large = colors.LogNorm(vmin=vmin / 100,  vmax=vmax / 10)

    # ── Surface density projections (Meshoid objects reused for velocity maps) ─
    def _surf(pos, mass, hsml, size):
        if len(pos) == 0:
            return np.zeros((res, res)), None
        M = Meshoid(pos, mass, hsml)
        return M.SurfaceDensity(M.m * 1e10, center=center0, size=size, res=res) / 1e6, M

    sig_fo,       M_fo  = _surf(pos_fo,       mass_small, hsml_small, image_box_kpc)
    sig_fo_large, _     = _surf(pos_fo_large, mass_large, hsml_large, image_box_kpc * 10)
    sig_eo,       M_eo  = _surf(pos_edge,     mass_small, hsml_small, image_box_kpc)

    # ── Rest-frame velocity maps (edge-on view) ───────────────────────────────
    # Subtract bulk COM velocity; rotate to disk frame (L_hat → z_hat = [0,0,1]).
    # Decompose into cylindrical (v_r, v_phi, v_z) in the disk plane.
    # Fit radial profiles v_r(r) and v_phi(r) via mass-weighted binning,
    # then subtract them to get residual (turbulent) velocities:
    #   δv_r_i   = v_r_i   - <v_r>(r_cyl_i)
    #   δv_phi_i = v_phi_i - <v_phi>(r_cyl_i)
    #   δv_z_i   = v_z_i                        (no bulk vertical profile)
    #   |δv|_i   = sqrt(δv_r² + δv_phi² + δv_z²)
    vel_raw = pdata['Velocities'][cut_small]
    vel_com = vel_raw - (com_vel if com_vel is not None else np.zeros(3))
    vel_rot = vel_com @ rot.T           # disk frame: L_hat → z_hat

    # Cylindrical decomposition in the rotated frame
    r_xy     = np.linalg.norm(pos_small[:, :2], axis=1)   # r_cyl [kpc]
    safe_rxy = np.maximum(r_xy, 1e-30)
    e_r_x    = pos_small[:, 0] / safe_rxy
    e_r_y    = pos_small[:, 1] / safe_rxy
    v_r      =  vel_rot[:, 0] * e_r_x  + vel_rot[:, 1] * e_r_y    # radial
    v_phi    = -vel_rot[:, 0] * e_r_y  + vel_rot[:, 1] * e_r_x    # azimuthal
    v_z      =  vel_rot[:, 2]                                       # vertical

    # Mass-weighted radial profiles in N_BINS annular bins (up to 95th-pct radius)
    N_BINS  = 20
    r_outer = np.percentile(r_xy, 95) if len(r_xy) > 0 else 1.0
    r_outer = max(r_outer, 1e-20)
    bins    = np.linspace(0.0, r_outer, N_BINS + 1)
    bidx    = np.clip(np.digitize(r_xy, bins) - 1, 0, N_BINS - 1)

    vr_prof   = np.zeros(N_BINS)
    vphi_prof = np.zeros(N_BINS)
    for b in range(N_BINS):
        mb = bidx == b
        if mb.sum() > 0:
            w = mass_small[mb]
            wsum = w.sum()
            vr_prof[b]   = np.dot(v_r[mb],   w) / wsum
            vphi_prof[b] = np.dot(v_phi[mb], w) / wsum

    # Residual (turbulent) velocities after streaming subtraction
    dv_r   = v_r   - vr_prof[bidx]
    dv_phi = v_phi - vphi_prof[bidx]
    dv_z   = v_z                            # no vertical bulk profile
    v_rest = np.sqrt(dv_r**2 + dv_phi**2 + dv_z**2)   # |δv| per particle [km/s]

    def _vel_maps(Mobj, size):
        """Mass-weighted mean |δv| and σ_|δv| projected through Mobj."""
        if Mobj is None or len(v_rest) == 0:
            return np.zeros((res, res)), np.zeros((res, res))
        sm    = np.maximum(Mobj.SurfaceDensity(Mobj.m,          center=center0, size=size, res=res), 1e-40)
        vm    = Mobj.SurfaceDensity(Mobj.m * v_rest,    center=center0, size=size, res=res) / sm
        v2m   = Mobj.SurfaceDensity(Mobj.m * v_rest**2, center=center0, size=size, res=res) / sm
        return vm, np.sqrt(np.maximum(v2m - vm**2, 0.0))

    vrest_eo,  sigv_eo  = _vel_maps(M_eo, image_box_kpc)   # edge-on
    vrest_fo,  sigv_fo  = _vel_maps(M_fo, image_box_kpc)   # face-on

    # Linear color norms — shared scale across face-on and edge-on panels.
    # vmax_vel (if set) fixes the ceiling across all frames to prevent flicker.
    if vmax_vel is not None:
        _vmax_v = float(vmax_vel)
        _vmax_s = float(vmax_vel)
    else:
        _ref_sig = sig_eo > 0
        _fo_sig  = sig_fo > 0
        _vmax_v  = float(np.percentile(
                       np.concatenate([vrest_eo[_ref_sig], vrest_fo[_fo_sig] if _fo_sig.any() else [0.]]), 99
                   )) if _ref_sig.any() else 1.0
        _vmax_s  = float(np.percentile(
                       np.concatenate([sigv_eo[_ref_sig],  sigv_fo[_fo_sig]  if _fo_sig.any() else [0.]]), 99
                   )) if _ref_sig.any() else 1.0
    norm_vrest = colors.Normalize(vmin=0, vmax=max(_vmax_v, 0.1))
    norm_sigv  = colors.Normalize(vmin=0, vmax=max(_vmax_s, 0.1))

    # ── Toomre Q = σ_r · κ / (π · G · Σ),  κ ≈ Ω (Keplerian) ───────────────
    v_K_small      = v_K[cut_small] if v_K is not None else np.zeros(len(mass_small))
    bin_centers_kpc = (bins[:-1] + bins[1:]) / 2   # kpc, used by phase + Q panels

    Sigma_prof   = np.zeros(N_BINS)   # g/cm²  (per-annulus surface density)
    sigma_r_prof = np.zeros(N_BINS)   # km/s   (mass-weighted radial dispersion)
    Omega_prof   = np.zeros(N_BINS)   # km/s/kpc ≡ v_K/r_cyl per bin

    for b in range(N_BINS):
        mb = bidx == b
        if mb.sum() == 0:
            continue
        r_lo, r_hi = bins[b], bins[b + 1]
        area_kpc2  = np.pi * max(r_hi**2 - r_lo**2, 1e-40)
        w    = mass_small[mb]
        wsum = w.sum()
        Sigma_prof[b]   = wsum * 1e10 * Msun / (area_kpc2 * kpc**2)   # g/cm²
        vr2_mw          = np.dot(v_r[mb]**2, w) / wsum
        sigma_r_prof[b] = np.sqrt(max(vr2_mw - vr_prof[b]**2, 0.0))   # km/s
        # Keplerian Ω = v_K / r_cyl, mass-weighted per annulus
        Omega_prof[b]   = np.dot(
            v_K_small[mb] / np.maximum(r_xy[mb], 1e-30), w
        ) / wsum   # km/s/kpc

    with np.errstate(divide='ignore', invalid='ignore'):
        Q_prof = np.where(
            Sigma_prof > 0,
            (sigma_r_prof * 1e5) * (Omega_prof * 1e5 / kpc) / (np.pi * G * Sigma_prof),
            np.nan,
        )

    # Face-on σ_r map — mass-weighted radial velocity dispersion projected along z
    if M_fo is not None and len(v_r) > 0:
        _sm  = np.maximum(M_fo.SurfaceDensity(M_fo.m,          center=center0, size=image_box_kpc, res=res), 1e-40)
        _vrm = M_fo.SurfaceDensity(M_fo.m * v_r,    center=center0, size=image_box_kpc, res=res) / _sm
        _v2m = M_fo.SurfaceDensity(M_fo.m * v_r**2, center=center0, size=image_box_kpc, res=res) / _sm
        sigma_r_fo = np.sqrt(np.maximum(_v2m - _vrm**2, 0.0))   # km/s, shape (res, res)
    else:
        sigma_r_fo = np.zeros((res, res))

    # Ω interpolated onto pixel grid (X, Y are in AU)
    r_px_kpc     = np.sqrt(X**2 + Y**2) * AU / kpc
    Omega_px_cgs = np.interp(r_px_kpc, bin_centers_kpc, Omega_prof,
                              left=0.0, right=0.0) * 1e5 / kpc   # 1/s

    # Face-on Q map
    pc_cm         = kpc / 1e3                                     # cm per pc
    Sigma_fo_gcm2 = np.maximum(sig_fo, 0.0) * Msun / pc_cm**2    # Msun/pc² → g/cm²
    with np.errstate(divide='ignore', invalid='ignore'):
        Q_fo = np.where(
            Sigma_fo_gcm2 > 0,
            (sigma_r_fo * 1e5) * Omega_px_cgs / (np.pi * G * Sigma_fo_gcm2),
            np.nan,
        )
    Q_fo = np.where(np.isfinite(Q_fo), Q_fo, 0.0)

    # ── Row-3 profile quantities ─────────────────────────────────────────────
    _GAMMA = 5.0 / 3.0

    # Volumetric density profile — use finer binning for better x-resolution
    N_RHO = 100
    bins_rho   = np.linspace(0.0, r_outer, N_RHO + 1)
    bidx_rho   = np.clip(np.digitize(r_xy, bins_rho) - 1, 0, N_RHO - 1)
    bin_ctr_rho_AU = (bins_rho[:-1] + bins_rho[1:]) / 2 * kpc / AU

    rho_cgs_small = pdata['Density'][cut_small].astype(np.float64) * 1e10 * Msun / kpc**3
    rho_prof = np.zeros(N_RHO)
    for b in range(N_RHO):
        mb = bidx_rho == b
        if mb.sum() == 0:
            continue
        w = mass_small[mb]; wsum = w.sum()
        rho_prof[b] = np.dot(rho_cgs_small[mb], w) / wsum

    # Sound speed from InternalEnergy [km/s] — c_s = sqrt(γ(γ-1)u)
    if 'InternalEnergy' in pdata:
        u_small  = pdata['InternalEnergy'][cut_small]                          # (km/s)²
        cs_small = np.sqrt(_GAMMA * (_GAMMA - 1.0) * np.maximum(u_small, 0.0))  # km/s
    else:
        u_small  = np.zeros(len(mass_small))
        cs_small = np.zeros(len(mass_small))
    cs_prof = np.zeros(N_BINS)
    for b in range(N_BINS):
        mb = bidx == b
        if mb.sum() == 0:
            continue
        w = mass_small[mb]; wsum = w.sum()
        cs_prof[b] = np.dot(cs_small[mb], w) / wsum

    # Mass-weighted mean turbulent speed |δv| per annulus
    vturb_prof = np.zeros(N_BINS)
    for b in range(N_BINS):
        mb = bidx == b
        if mb.sum() == 0:
            continue
        w = mass_small[mb]; wsum = w.sum()
        vturb_prof[b] = np.dot(v_rest[mb], w) / wsum

    # Mach number per annulus
    with np.errstate(divide='ignore', invalid='ignore'):
        mach_prof = np.where(cs_prof > 0, vturb_prof / cs_prof, np.nan)

    # Save Q profile in data_outdir (not the frames subdir)
    _npz_dir = data_outdir if data_outdir is not None else os.path.dirname(outpath)
    np.savez(
        os.path.join(_npz_dir, f'qprofile_{snap_num:04d}.npz'),
        r_kpc    = bin_centers_kpc,
        r_AU     = bin_centers_kpc * kpc / AU,
        Q        = Q_prof,
        Sigma    = Sigma_prof,
        sigma_r  = sigma_r_prof,
        Omega    = Omega_prof,
        time_Myr      = np.array([time_Myr]),
        snap_num      = np.array([snap_num]),
        n_sinks       = np.array([len(stardata['Masses']) if stardata and len(stardata.get('Masses', [])) > 0 else 0]),
        sink_form_Myr = np.array(sink_form_Myr) if sink_form_Myr is not None else np.array([]),
        sink_r_AU     = np.array(sink_r_AU)     if sink_r_AU     is not None else np.array([]),
    )

    n_stars   = len(stardata['Masses']) if stardata and len(stardata.get('Masses', [])) > 0 else 0
    M_stars   = np.sum(stardata['Masses']) * 1e10 if n_stars > 0 else 0.0
    M_disk    = np.sum(mass_small[disk_small]) * 1e10
    R_disk_AU = (np.percentile(np.linalg.norm(pos_small[disk_small, :2], axis=1), 90) * kpc / AU
                 if disk_small.sum() > 0 else 0.0)
    M_gas_total = np.sum(pdata['Masses']) * 1e10   # all gas in cutout [Msun]
    f_star = M_stars / (M_stars + M_gas_total) if (M_stars + M_gas_total) > 0 else 0.0

    # Central density: mass-weighted mean ρ within 10 AU of primary sink
    _r_central_kpc = 10.0 * AU / kpc
    _r3d_small     = gas_dists[cut_small]
    _central_mask  = _r3d_small < _r_central_kpc
    if _central_mask.sum() > 0:
        _w = mass_small[_central_mask]
        rho_central = float(np.dot(rho_cgs_small[_central_mask], _w) / _w.sum())
    else:
        rho_central = 0.0

    disk_fo_AU = pos_fo[disk_small]   * kpc / AU
    disk_eo_AU = pos_edge[disk_small] * kpc / AU

    # Layout: 4 rows × 4 cols (wide); optional 5th row for phase diagrams.
    #   Row 0: surface density — face-on small | face-on 10× | edge-on clean | edge-on overlay
    #   Row 1: velocity maps  — face-on |δv|  | face-on σ    | edge-on |δv|  | edge-on σ
    #   Row 2: phase plots    — v_r vs r | v_phi vs r | Q vs r | face-on Q map
    #   Row 3: profiles       — ρ(r) | c_s & σ_r & σ_turb vs r | Mach number | SML histogram
    #   Row 4 (optional):     — T vs ρ (2 cols wide) | log f_H2 vs ρ (2 cols wide)
    import matplotlib.gridspec as _gs
    if include_phase:
        fig = plt.figure(figsize=(28, 30))
        _grid = _gs.GridSpec(5, 4, figure=fig, hspace=0.42, wspace=0.32)
        axes = np.array([[fig.add_subplot(_grid[r, c]) for c in range(4)] for r in range(4)])
        _ax_ph_T   = fig.add_subplot(_grid[4, :2])
        _ax_ph_fh2 = fig.add_subplot(_grid[4, 2:])
    else:
        fig, axes = plt.subplots(4, 4, figsize=(28, 24))
        _ax_ph_T   = None
        _ax_ph_fh2 = None
    fig.patch.set_facecolor('k')

    # (ax, sig, Xg, Yg, norm, disk_scatter_AU, show_overlay, half_extent,
    #  title, xlabel, ylabel, colorbar_label, colormap)
    panels = [
        # ── Row 0: surface density ────────────────────────────────────────────
        (axes[0, 0], sig_fo,       X,  Y,  norm_small, disk_fo_AU, False,
         half_AU,       'Face-on (all gas)',      'x (AU)', 'y (AU)',
         r'$\Sigma$ (M$_\odot$/pc$^2$)', cmap),
        (axes[0, 1], sig_fo_large, XL, YL, norm_large, None,       False,
         half_AU_large, 'Face-on (10× zoom out)', 'x (AU)', 'y (AU)',
         r'$\Sigma$ (M$_\odot$/pc$^2$)', cmap),
        (axes[0, 2], sig_eo,       X,  Y,  norm_small, disk_eo_AU, False,
         half_AU,       'Edge-on (all gas)',       'x (AU)', 'z (AU)',
         r'$\Sigma$ (M$_\odot$/pc$^2$)', cmap),
        (axes[0, 3], sig_eo,       X,  Y,  norm_small, disk_eo_AU, True,
         half_AU,       'Edge-on (disk overlay)',  'x (AU)', 'z (AU)',
         r'$\Sigma$ (M$_\odot$/pc$^2$)', cmap),
        # ── Row 1: rest-frame velocity maps ──────────────────────────────────
        (axes[1, 0], vrest_fo, X, Y, norm_vrest, None, False,
         half_AU, r'Face-on $|\delta v|$ (rest-frame)',  'x (AU)', 'y (AU)',
         r'$|\delta v|$ (km/s)', 'viridis'),
        (axes[1, 1], sigv_fo,  X, Y, norm_sigv,  None, False,
         half_AU, r'Face-on $\sigma_{|\delta v|}$',      'x (AU)', 'y (AU)',
         r'$\sigma_{|\delta v|}$ (km/s)', 'viridis'),
        (axes[1, 2], vrest_eo, X, Y, norm_vrest, None, False,
         half_AU, r'Edge-on $|\delta v|$ (rest-frame)',  'x (AU)', 'z (AU)',
         r'$|\delta v|$ (km/s)', 'viridis'),
        (axes[1, 3], sigv_eo,  X, Y, norm_sigv,  None, False,
         half_AU, r'Edge-on $\sigma_{|\delta v|}$',      'x (AU)', 'z (AU)',
         r'$\sigma_{|\delta v|}$ (km/s)', 'viridis'),
    ]

    for ax, sig, Xg, Yg, norm, dpos, show_overlay, half, title, xlabel, ylabel, clabel, panel_cmap in panels:
        ax.set_facecolor('k')
        if isinstance(norm, colors.LogNorm):
            sig_plot = np.where(sig > 0, sig, norm.vmin)
        else:
            sig_plot = sig
        im = ax.pcolormesh(Xg, Yg, sig_plot, norm=norm, cmap=panel_cmap)
        cb = plt.colorbar(im, ax=ax)
        cb.set_label(clabel, color='w', fontsize=9)
        cb.ax.yaxis.set_tick_params(color='w')
        plt.setp(cb.ax.yaxis.get_ticklabels(), color='w')
        if show_overlay and disk_small.sum() > 0:
            ax.scatter(dpos[:, 0], dpos[:, 1], s=0.5, alpha=0.3, c='cyan', rasterized=True)
        ax.set_xlabel(xlabel, color='w', fontsize=10)
        ax.set_ylabel(ylabel, color='w', fontsize=10)
        ax.set_title(title, color='w', fontsize=11)
        ax.tick_params(colors='w', which='both', direction='in', right=True, top=True)
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        for spine in ax.spines.values():
            spine.set_edgecolor('w')

    # ── Row 2: 1D phase plots + Toomre Q ─────────────────────────────────────
    r_xy_AU  = r_xy * kpc / AU
    bin_AU   = bin_centers_kpc * kpc / AU   # AU (same bins used by Q)
    r_max_AU = image_box_kpc / 2 * kpc / AU

    # Panels [2,0] and [2,1]: velocity phase-space scatter + profile fit
    for ax, ydata, yfit, ylabel, title, ptcolor in [
        (axes[2, 0], v_r,   vr_prof,   r'$v_r$ (km/s)',    r'$v_r$ vs $r$',    'cyan'),
        (axes[2, 1], v_phi, vphi_prof, r'$v_\phi$ (km/s)', r'$v_\phi$ vs $r$', 'orange'),
    ]:
        ax.set_facecolor('k')
        ax.scatter(r_xy_AU, ydata, s=0.3, alpha=0.15, c=ptcolor,
                   rasterized=True, label='gas particles')
        ax.plot(bin_AU, yfit, 'r-', lw=2, label='profile fit')
        ax.axhline(0, color='w', lw=0.5, ls='--', alpha=0.4)
        ax.axvline(r_max_AU, color='w', lw=0.5, ls=':', alpha=0.4, label=r'$r_{\rm box}/2$')
        ax.set_xlim(0, r_max_AU * 1.05)
        ax.set_ylim(-20, 20)
        ax.set_xlabel('r (AU)', color='w', fontsize=10)
        ax.set_ylabel(ylabel, color='w', fontsize=10)
        ax.set_title(title, color='w', fontsize=11)
        ax.tick_params(colors='w', which='both', direction='in', right=True, top=True)
        for spine in ax.spines.values():
            spine.set_edgecolor('w')
        leg = ax.legend(fontsize=8, framealpha=0.3)
        for text in leg.get_texts():
            text.set_color('w')

    # Panel [2,2]: Q vs r (1D, azimuthally averaged)
    ax_q1d = axes[2, 2]
    ax_q1d.set_facecolor('k')
    valid_q = np.isfinite(Q_prof) & (Q_prof > 0)
    if valid_q.any():
        ax_q1d.semilogy(bin_AU[valid_q], Q_prof[valid_q], 'w-o', ms=4, lw=1.5)
    ax_q1d.axhline(1.0, color='r', lw=1.5, ls='--', label='Q = 1')
    ax_q1d.set_xlim(0, r_max_AU * 1.05)
    ax_q1d.set_ylim(0.1, 100)
    ax_q1d.set_xlabel('r (AU)', color='w', fontsize=10)
    ax_q1d.set_ylabel('Toomre Q', color='w', fontsize=10)
    ax_q1d.set_title('Q vs r (azimuthal avg)', color='w', fontsize=11)
    ax_q1d.tick_params(colors='w', which='both', direction='in', right=True, top=True)
    for spine in ax_q1d.spines.values():
        spine.set_edgecolor('w')
    leg_q = ax_q1d.legend(fontsize=8, framealpha=0.3)
    for t in leg_q.get_texts():
        t.set_color('w')

    # Panel [2,3]: face-on Toomre Q map
    ax_qmap = axes[2, 3]
    ax_qmap.set_facecolor('k')
    Q_plot  = np.where(Q_fo > 0, Q_fo, np.nan)
    norm_Q  = colors.LogNorm(vmin=0.1, vmax=10)
    im_q    = ax_qmap.pcolormesh(X, Y, Q_plot, norm=norm_Q, cmap='RdYlGn')
    cb_q    = plt.colorbar(im_q, ax=ax_qmap)
    cb_q.set_label('Toomre Q', color='w', fontsize=9)
    cb_q.ax.yaxis.set_tick_params(color='w')
    plt.setp(cb_q.ax.yaxis.get_ticklabels(), color='w')
    # Q=1 contour
    try:
        ax_qmap.contour(X, Y, np.where(np.isfinite(Q_fo), Q_fo, 1.0),
                        levels=[1.0], colors='k', linewidths=1.5)
    except Exception:
        pass
    ax_qmap.set_xlabel('x (AU)', color='w', fontsize=10)
    ax_qmap.set_ylabel('y (AU)', color='w', fontsize=10)
    ax_qmap.set_title('Face-on Toomre Q', color='w', fontsize=11)
    ax_qmap.tick_params(colors='w', which='both', direction='in', right=True, top=True)
    ax_qmap.set_xlim(-half_AU, half_AU)
    ax_qmap.set_ylim(-half_AU, half_AU)
    for spine in ax_qmap.spines.values():
        spine.set_edgecolor('w')

    # ── Row 3: ρ(r) | c_s/σ/σ_turb vs r | Mach | SML histogram ──────────────
    _style = dict(colors='w', which='both', direction='in', right=True, top=True)

    # [3,0] Volumetric density profile
    ax_rho = axes[3, 0]
    ax_rho.set_facecolor('k')
    valid_rho = rho_prof > 0
    if valid_rho.any():
        ax_rho.semilogy(bin_ctr_rho_AU[valid_rho], rho_prof[valid_rho], 'w-', lw=1.5)
    ax_rho.set_xlabel('r (AU)', color='w', fontsize=10)
    ax_rho.set_ylabel(r'$\rho$ (g/cm³)', color='w', fontsize=10)
    ax_rho.set_title(r'Density profile $\rho(r)$', color='w', fontsize=11)
    ax_rho.set_xlim(0, r_max_AU * 1.05)
    ax_rho.tick_params(**_style)
    for sp in ax_rho.spines.values(): sp.set_edgecolor('w')

    # [3,1] c_s, σ_r, σ_turb vs r on same axes
    ax_vel = axes[3, 1]
    ax_vel.set_facecolor('k')
    valid_b = bin_AU > 0
    ax_vel.plot(bin_AU[valid_b], cs_prof[valid_b],    'r-',  lw=2,   label=r'$c_s$')
    ax_vel.plot(bin_AU[valid_b], sigma_r_prof[valid_b], 'c-', lw=2,  label=r'$\sigma_r$')
    ax_vel.plot(bin_AU[valid_b], vturb_prof[valid_b], 'y--', lw=1.5, label=r'$\langle|\delta v|\rangle$')
    ax_vel.set_xlabel('r (AU)', color='w', fontsize=10)
    ax_vel.set_ylabel('Velocity (km/s)', color='w', fontsize=10)
    ax_vel.set_title(r'$c_s$, $\sigma_r$, $\langle|\delta v|\rangle$ vs $r$', color='w', fontsize=11)
    ax_vel.set_xlim(0, r_max_AU * 1.05)
    ax_vel.set_ylim(bottom=0)
    ax_vel.tick_params(**_style)
    for sp in ax_vel.spines.values(): sp.set_edgecolor('w')
    leg_vel = ax_vel.legend(fontsize=8, framealpha=0.3)
    for t in leg_vel.get_texts(): t.set_color('w')

    # [3,2] Mach number profile
    ax_mach = axes[3, 2]
    ax_mach.set_facecolor('k')
    valid_m = np.isfinite(mach_prof) & (mach_prof > 0)
    if valid_m.any():
        ax_mach.semilogy(bin_AU[valid_m], mach_prof[valid_m], 'w-o', ms=4, lw=1.5)
    ax_mach.axhline(1.0, color='r', lw=1.5, ls='--', label='Ma = 1')
    ax_mach.set_xlabel('r (AU)', color='w', fontsize=10)
    ax_mach.set_ylabel(r'$\mathcal{M} = \langle|\delta v|\rangle / c_s$', color='w', fontsize=10)
    ax_mach.set_title('Mach number profile', color='w', fontsize=11)
    ax_mach.set_xlim(0, r_max_AU * 1.05)
    ax_mach.tick_params(**_style)
    for sp in ax_mach.spines.values(): sp.set_edgecolor('w')
    leg_m = ax_mach.legend(fontsize=8, framealpha=0.3)
    for t in leg_m.get_texts(): t.set_color('w')

    # [3,3] Smoothing-length histogram (disk particles only)
    ax_sml = axes[3, 3]
    ax_sml.set_facecolor('k')
    sml_all_AU  = hsml_small * kpc / AU
    sml_disk_AU = hsml_small[disk_small] * kpc / AU if disk_small.sum() > 0 else np.array([])
    ax_sml.hist(sml_all_AU,  bins=50, color='c',      alpha=0.5, label='all (small box)')
    if len(sml_disk_AU) > 0:
        ax_sml.hist(sml_disk_AU, bins=50, color='yellow', alpha=0.7, label='disk particles')
    ax_sml.set_xlabel('SmoothingLength (AU)', color='w', fontsize=10)
    ax_sml.set_ylabel('N particles', color='w', fontsize=10)
    ax_sml.set_title('Resolution: SML histogram', color='w', fontsize=11)
    ax_sml.tick_params(**_style)
    for sp in ax_sml.spines.values(): sp.set_edgecolor('w')
    leg_sml = ax_sml.legend(fontsize=8, framealpha=0.3)
    for t in leg_sml.get_texts(): t.set_color('w')

    # Sink positions — face-on panels use rot_fo (co-rotating), edge-on use rot
    if n_stars > 0:
        sc         = stardata['Coordinates'] - com
        star_fo_AU = sc @ rot_fo.T * kpc / AU
        star_eo_AU = sc @ rot.T    * kpc / AU
        for ax, sp, half in [
            (axes[0, 0], star_fo_AU[:, :2],     half_AU),
            (axes[0, 1], star_fo_AU[:, :2],     half_AU_large),
            (axes[0, 2], star_eo_AU[:, [0, 2]], half_AU),
            (axes[0, 3], star_eo_AU[:, [0, 2]], half_AU),
            (axes[1, 0], star_fo_AU[:, :2],     half_AU),
            (axes[1, 1], star_fo_AU[:, :2],     half_AU),
            (axes[1, 2], star_eo_AU[:, [0, 2]], half_AU),
            (axes[1, 3], star_eo_AU[:, [0, 2]], half_AU),
        ]:
            in_view = (np.abs(sp[:, 0]) < half) & (np.abs(sp[:, 1]) < half)
            if in_view.any():
                ax.scatter(sp[in_view, 0], sp[in_view, 1], s=20, c='white', marker='*',
                           zorder=5, edgecolors='yellow', linewidths=0.5)

    # ── Row 4 (optional): inline phase diagrams ──────────────────────────────
    if include_phase and _ax_ph_T is not None:
        _GAMMA_P = 5.0 / 3.0; _kB_P = 1.381e-16; _mp_P = 1.673e-24
        _rho_lim = (1e-25, 1e-10); _T_lim = (1e1, 1e6)
        _nb = 150
        _rho_e = np.linspace(np.log10(_rho_lim[0]), np.log10(_rho_lim[1]), _nb + 1)
        _T_e   = np.linspace(np.log10(_T_lim[0]),   np.log10(_T_lim[1]),   _nb + 1)

        if 'InternalEnergy' in pdata:
            _u  = pdata['InternalEnergy'][cut_small].astype(np.float64)
            _TK = (_GAMMA_P - 1.0) * _u * 1e10 * (_mp_P / _kB_P)
        else:
            _TK = np.ones(len(mass_small))
        _rho_ph = rho_cgs_small
        _m_ph   = mass_small * 1e10
        _vld    = (_rho_ph > 0) & (_TK > 0)

        def _phase_hist2d(ax, y_vals, y_edges, ylabel, ytitle):
            ax.set_facecolor('k')
            _vld2 = _vld & (y_vals > 0) if ylabel.startswith('log') else _vld
            if _vld2.any():
                _H, _, _ = np.histogram2d(
                    np.log10(_rho_ph[_vld2]), y_vals[_vld2],
                    bins=[_rho_e, y_edges], weights=_m_ph[_vld2])
                _H = np.where(_H > 0, _H, np.nan)
                if np.any(_H > 0):
                    _im = ax.pcolormesh(_rho_e, y_edges, _H.T,
                        norm=colors.LogNorm(
                            vmin=np.nanpercentile(_H[_H > 0], 5),
                            vmax=np.nanmax(_H)),
                        cmap='inferno', rasterized=True)
                    _cb = plt.colorbar(_im, ax=ax)
                    _cb.set_label(r'Mass ($M_\odot$/bin)', color='w', fontsize=8)
                    _cb.ax.yaxis.set_tick_params(color='w')
                    plt.setp(_cb.ax.yaxis.get_ticklabels(), color='w')
                    # Disk contours
                    if disk_small.sum() > 0 and (_vld2 & disk_small).any():
                        _Hd, _, _ = np.histogram2d(
                            np.log10(_rho_ph[_vld2 & disk_small]),
                            y_vals[_vld2 & disk_small],
                            bins=[_rho_e, y_edges],
                            weights=_m_ph[_vld2 & disk_small])
                        _rc = 0.5 * (_rho_e[:-1] + _rho_e[1:])
                        _yc = 0.5 * (y_edges[:-1] + y_edges[1:])
                        try:
                            ax.contour(_rc, _yc, _Hd.T, levels=4,
                                       colors='cyan', linewidths=0.8, alpha=0.85)
                        except Exception:
                            pass
            ax.axvline(np.log10(rho_threshold_cgs), color='r', ls='--', lw=1.2,
                       label=r'$\rho_\mathrm{thresh}$')
            ax.set_xlabel(r'$\log_{10}\ \rho\ \mathrm{(g/cm^3)}$', color='w', fontsize=10)
            ax.set_ylabel(ylabel, color='w', fontsize=10)
            ax.set_title(ytitle, color='w', fontsize=11)
            ax.tick_params(**_style)
            for sp in ax.spines.values(): sp.set_edgecolor('w')
            _l = ax.legend(fontsize=8, framealpha=0.3)
            for _t in _l.get_texts(): _t.set_color('w')

        # T vs ρ
        _phase_hist2d(_ax_ph_T, np.log10(np.maximum(_TK, 1e-30)),
                      _T_e, r'$\log_{10}\ T\ \mathrm{(K)}$', r'$T$ vs $\rho$')

        # log(f_H2) vs ρ
        _ax_ph_fh2.set_facecolor('k')
        if h2_field is not None and h2_field in pdata:
            _fh2 = pdata[h2_field][cut_small].astype(np.float64)
            _fh2_e = np.linspace(-6, 0, _nb + 1)
            _phase_hist2d(_ax_ph_fh2, np.log10(np.maximum(_fh2, 1e-8)),
                          _fh2_e,
                          r'$\log_{10}\ f_{\rm H_2}$',
                          r'$\log_{10}\ f_{\rm H_2}$ vs $\rho$')
        else:
            _ax_ph_fh2.text(0.5, 0.5, 'H₂ field not loaded\n(set h2_field in Defaults)',
                            ha='center', va='center', color='w', fontsize=11,
                            transform=_ax_ph_fh2.transAxes)
            _ax_ph_fh2.set_xlabel(r'$\log_{10}\ \rho\ \mathrm{(g/cm^3)}$', color='w', fontsize=10)
            _ax_ph_fh2.set_title(r'$\log_{10}\ f_{\rm H_2}$ vs $\rho$', color='w', fontsize=11)
            for sp in _ax_ph_fh2.spines.values(): sp.set_edgecolor('w')

    fig.suptitle(
        f'Snap {snap_num:04d}   t = {time_Myr*1e3:.2f} kyr   '
        f'N_stars = {n_stars}   M_stars = {M_stars:.3f} Msun   '
        f'M_disk = {M_disk:.2f} Msun   R_disk = {R_disk_AU:.0f} AU   '
        f'f_star = {f_star*100:.2f}%   '
        f'ρ_central = {rho_central:.2e} g/cm³',
        color='w', fontsize=11
    )
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, facecolor='k')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Q heatmap (time × radius)
# ═══════════════════════════════════════════════════════════════════════════════

def make_Q_heatmap(outdir, heatmap_path=None):
    """
    Load all qprofile_*.npz files from outdir and produce a heatmap:
      x-axis: time [Myr],  y-axis: r [AU],  colorbar: Toomre Q (log).
    A Q=1 contour is drawn in black.
    Saves to outdir/Q_heatmap.png (or heatmap_path if provided).
    """
    profile_files = sorted(glob.glob(os.path.join(outdir, 'qprofile_*.npz')))
    if not profile_files:
        print('  make_Q_heatmap: no qprofile_*.npz files found, skipping.')
        return

    times, Q_rows, r_AU_ref, n_sinks_list = [], [], None, []
    # Accumulate per-sink birth events: keyed by formation time (Myr) → r (AU)
    _seen_form_Myr = {}   # form_Myr → r_AU (keeps first appearance)
    for f in sorted(profile_files):   # sorted → chronological
        d = np.load(f)
        t = float(d['time_Myr'])
        Q = d['Q'].copy()
        r = d['r_AU'].copy()
        times.append(t)
        Q_rows.append((r, Q))
        if r_AU_ref is None:
            r_AU_ref = r
        # n_sinks may not exist in older npz files; default to 0
        n_sinks_list.append(int(d['n_sinks'][0]) if 'n_sinks' in d else 0)
        # Collect sink birth events (first snapshot each sink_form_Myr is seen)
        if 'sink_form_Myr' in d and 'sink_r_AU' in d:
            for tf, rf in zip(d['sink_form_Myr'], d['sink_r_AU']):
                tf_key = round(float(tf), 6)   # avoid float key collisions
                if tf_key not in _seen_form_Myr:
                    _seen_form_Myr[tf_key] = float(rf)

    # Determine t₁ = earliest StellarFormationTime across all sinks.
    # Using the actual formation time (not the snapshot time) ensures the
    # first sink marker lands exactly at t - t₁ = 0 on the heatmap.
    sort_idx    = np.argsort(times)
    times_arr   = np.array(times)[sort_idx]
    n_sinks_arr = np.array(n_sinks_list)[sort_idx]
    t1_Myr = float(min(_seen_form_Myr.keys())) if _seen_form_Myr else None
    # Fall back to first snapshot with sinks if no formation times were recorded
    if t1_Myr is None:
        sink_snaps = np.where(n_sinks_arr > 0)[0]
        t1_Myr = float(times_arr[sink_snaps[0]]) if len(sink_snaps) > 0 else None
    Q_mat     = np.zeros((len(times_arr), len(r_AU_ref)))
    for i, idx in enumerate(sort_idx):
        r_i, Q_i = Q_rows[idx]
        # replace NaN with 0 before interpolating, mask after
        Q_clean = np.where(np.isfinite(Q_i), Q_i, 0.0)
        Q_mat[i] = np.interp(r_AU_ref, r_i, Q_clean, left=0.0, right=0.0)

    # Replace zeros with NaN for plotting
    Q_mat = np.where(Q_mat > 0, Q_mat, np.nan)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('k')
    ax.set_facecolor('k')

    # Shift time axis to t - t₁, convert Myr → kyr for display
    t_shifted = (times_arr - t1_Myr if t1_Myr is not None else times_arr) * 1e3

    # pcolormesh needs 2D coordinate grids; use cell edges
    dt   = np.diff(t_shifted)
    t_lo = np.concatenate([[t_shifted[0] - dt[0]/2],  t_shifted[:-1] + dt/2])
    t_hi = np.concatenate([t_shifted[:-1] + dt/2,    [t_shifted[-1] + dt[-1]/2]])
    T    = np.concatenate([t_lo, [t_hi[-1]]])

    dr   = np.diff(r_AU_ref)
    r_lo = np.concatenate([[r_AU_ref[0] - dr[0]/2], r_AU_ref[:-1] + dr/2])
    r_hi = np.concatenate([r_AU_ref[:-1] + dr/2,   [r_AU_ref[-1] + dr[-1]/2]])
    R    = np.concatenate([r_lo, [r_hi[-1]]])

    Tg, Rg = np.meshgrid(T, R, indexing='ij')
    im = ax.pcolormesh(Tg, Rg, Q_mat,
                       norm=colors.LogNorm(vmin=0.1, vmax=10),
                       cmap='RdYlGn', rasterized=True)
    cb = plt.colorbar(im, ax=ax, label='Toomre Q')
    cb.ax.yaxis.set_tick_params(color='w')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='w')
    cb.set_label('Toomre Q', color='w')

    # Q=1 contour on cell-centre grids
    Tc, Rc = np.meshgrid(t_shifted, r_AU_ref, indexing='ij')
    Q_filled = np.where(np.isfinite(Q_mat), Q_mat, 1.0)
    try:
        ax.contour(Tc, Rc, Q_filled, levels=[1.0], colors='k', linewidths=1.5)
    except Exception:
        pass

    # Star-formation events: gold ★ at (t_form - t₁, r_at_birth)
    # Only plot sinks within the heatmap's r range to avoid out-of-bounds markers
    if _seen_form_Myr and t1_Myr is not None:
        birth_t = (np.array(list(_seen_form_Myr.keys())) - t1_Myr) * 1e3   # kyr
        birth_r = np.array(list(_seen_form_Myr.values()))
        r_min_hm, r_max_hm = r_AU_ref[0], r_AU_ref[-1]
        in_range = (birth_r >= r_min_hm) & (birth_r <= r_max_hm)
        if in_range.any():
            ax.scatter(birth_t[in_range], birth_r[in_range], marker='*', s=120,
                       color='gold', edgecolors='w', linewidths=0.5,
                       zorder=5, label='sink formation')
            ax.legend(facecolor='#222', edgecolor='w', labelcolor='w', fontsize=9)

    xlabel = (r'$t - t_1$ (kyr)   [$t_1$ = first sink formation]'
              if t1_Myr is not None else 'Time (kyr)')
    ax.set_xlabel(xlabel,    color='w', fontsize=12)
    ax.set_ylabel('r (AU)',  color='w', fontsize=12)
    ax.set_title('Toomre Q evolution', color='w', fontsize=13)
    ax.tick_params(colors='w', which='both', direction='in', right=True, top=True)
    for spine in ax.spines.values():
        spine.set_edgecolor('w')

    plt.tight_layout()
    if heatmap_path is None:
        heatmap_path = os.path.join(outdir, 'Q_heatmap.png')
    fig.savefig(heatmap_path, dpi=150, facecolor='k')
    plt.close(fig)
    print(f'  Q heatmap saved → {heatmap_path}')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--path',    default='/mnt/home/skhullar/ceph/projects/SFIRE/m12f/')
    p.add_argument('--sim',     default='output_new_jeans_refinement')
    p.add_argument('--outdir',  default='/mnt/home/skhullar/analysis/popIII_analysis/plots/disk_movie_frames/')
    p.add_argument('--snap-start',  type=int, default=None, help='First snapshot number (inclusive)')
    p.add_argument('--snap-end',    type=int, default=None, help='Last snapshot number (inclusive)')
    p.add_argument('--res',         type=int, default=400,  help='Image resolution (pixels per axis)')
    p.add_argument('--image-box',   type=float, default=2e-5, help='Image box full width [kpc]')
    p.add_argument('--r-search',    type=float, default=1e-5, help='L_hat search radius [kpc]')
    p.add_argument('--r-max',       type=float, default=1e-5, help='Outer disk boundary [kpc]')
    p.add_argument('--rho-thresh',  type=float, default=1e-15, help='Density threshold [g/cm^3]')
    p.add_argument('--aspect',      type=float, default=0.3,  help='Aspect ratio |z|/r_cyl cutoff')
    p.add_argument('--f-kep',       type=float, default=0.3,  help='Keplerian fraction threshold')
    p.add_argument('--vmin',        type=float, default=1e5,  help='Colorbar min [Msun/pc^2]')
    p.add_argument('--vmax',        type=float, default=1e8,  help='Colorbar max [Msun/pc^2]')
    p.add_argument('--ncores',      type=int,   default=1,    help='Number of parallel cores')
    p.add_argument('--cmap',        default='inferno',
                   help='Colormap name (matplotlib or cmasher, e.g. cmr.ember)')
    p.add_argument('--no-corotate', dest='corotate', action='store_false', default=True,
                   help='Disable co-rotating face-on frame (default: enabled)')
    p.add_argument('--vmax-vel',    type=float, default=None,
                   help='Fixed velocity colorbar ceiling [km/s] to prevent flicker; '
                        'None = auto-scale per frame')
    p.add_argument('--min-gas-particles', type=int, default=80000,
                   help='Skip frame if PartType0 count is below this threshold '
                        '(only enforced for snap >= --min-gas-snap)')
    p.add_argument('--min-gas-snap', type=int, default=150,
                   help='Snapshot number at which the --min-gas-particles check activates')
    return p.parse_args()


def process_snapshot(args_tuple):
    """Process a single snapshot — designed to be called in parallel."""
    import time as _time
    try:
        import cmasher  # registers cmr.* colormaps with matplotlib
    except ImportError:
        pass
    (snap_path, snap_num, outdir, image_box_kpc, res, vmin, vmax,
     path, sim,
     r_search_kpc, r_max_kpc, rho_threshold_cgs, aspect_ratio, f_kep,
     cmap, reference_center, reference_search_radius,
     corotate, vmax_vel,
     min_gas_particles, min_gas_snap,
     include_phase_in_master, h2_field) = args_tuple

    frames_dir = os.path.join(outdir, 'master_frames')
    os.makedirs(frames_dir, exist_ok=True)
    outpath = os.path.join(frames_dir, f'frame_{snap_num:04d}.png')
    if os.path.exists(outpath):
        return snap_num, 'skipped (exists)', 0.0

    t0 = _time.perf_counter()

    gas_fields = ['Masses', 'Coordinates', 'SmoothingLength',
                  'Velocities', 'Density', 'ParticleIDs', 'InternalEnergy']
    if include_phase_in_master and h2_field is not None:
        gas_fields.append(h2_field)
    try:
        hdr, pdata, stardata, fsd, _, _ = get_snap_data_hybrid(
            sim, path, snap_num, snapshot_suffix='', snapdir=False,
            refinement_tag=False, verbose=False, custom_gas_fields=gas_fields)
        hdr, pdata, stardata, fsd = convert_units_to_physical(hdr, pdata, stardata, fsd)
    except Exception as e:
        return snap_num, f'load error: {e}', 0.0

    # Skip frames with too few gas particles (flicker guard).
    # Only enforced for snap >= min_gas_snap to allow early (pre-refinement) snaps through.
    n_gas = len(pdata['Masses'])
    if snap_num >= min_gas_snap and n_gas < min_gas_particles:
        return snap_num, f'skipped (n_gas={n_gas} < {min_gas_particles})', 0.0

    time_Myr = _scale_to_Myr(hdr['Time'])

    try:
        is_disk, com, L_hat, r_cyl, z, v_phi, v_K, com_vel = identify_disk(
            pdata, stardata,
            r_search_kpc            = r_search_kpc,
            r_max_kpc               = r_max_kpc,
            rho_threshold_cgs       = rho_threshold_cgs,
            aspect_ratio            = aspect_ratio,
            f_kep                   = f_kep,
            use_bounds              = True,
            reference_center        = reference_center,
            reference_search_radius = reference_search_radius,
        )
        # Read sink formation times directly from HDF5 (GUAC may not expose this field)
        import h5py as _h5py
        _sink_form_Myr = np.array([])
        _sink_r_AU     = np.array([])
        try:
            with _h5py.File(snap_path, 'r') as _hf:
                if ('PartType5' in _hf
                        and 'StellarFormationTime' in _hf['PartType5']
                        and _hf['PartType5/StellarFormationTime'].shape[0] > 0):
                    _sft = _hf['PartType5/StellarFormationTime'][:]   # scale factors
                    _sink_form_Myr = np.array([_scale_to_Myr(float(a)) for a in _sft])
                    if stardata and len(stardata.get('Coordinates', [])) > 0:
                        _dr = stardata['Coordinates'] - com
                        _sink_r_AU = np.linalg.norm(_dr, axis=1) * kpc / AU
        except Exception:
            pass

        render_frame(pdata, stardata, snap_num, time_Myr,
                     is_disk, com, L_hat,
                     image_box_kpc   = image_box_kpc,
                     res             = res,
                     vmin            = vmin,
                     vmax            = vmax,
                     cmap            = cmap,
                     outpath         = outpath,
                     com_vel         = com_vel,
                     corotate        = corotate,
                     vmax_vel        = vmax_vel,
                     v_K             = v_K,
                     data_outdir     = outdir,
                     include_phase   = include_phase_in_master,
                     h2_field        = h2_field,
                     sink_form_Myr   = _sink_form_Myr,
                     sink_r_AU       = _sink_r_AU)
    except Exception as e:
        return snap_num, f'render error: {e}', 0.0

    return snap_num, 'ok', _time.perf_counter() - t0


def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    snap_pattern = os.path.join(args.path, args.sim, 'snapshot_*.hdf5')
    snap_paths   = sorted(glob.glob(snap_pattern))[::-1]
    if not snap_paths:
        sys.exit(f'No snapshots found matching: {snap_pattern}')

    # Parse snapshot numbers and apply range filter
    def snap_num_from_path(p):
        return int(os.path.basename(p).replace('snapshot_', '').replace('.hdf5', ''))

    snap_items = [(p, snap_num_from_path(p)) for p in snap_paths]
    if args.snap_start is not None:
        snap_items = [(p, n) for p, n in snap_items if n >= args.snap_start]
    if args.snap_end is not None:
        snap_items = [(p, n) for p, n in snap_items if n <= args.snap_end]

    print(f'Processing {len(snap_items)} snapshots → {args.outdir}')
    print(f'Parameters: r_search={args.r_search*1e3:.1f} pc  r_max={args.r_max*1e3:.1f} pc  '
          f'rho_thresh={args.rho_thresh:.0e} g/cm3  aspect={args.aspect}  f_kep={args.f_kep}')
    print(f'Image: {args.res}x{args.res} px  box={args.image_box*1e3:.1f} pc  '
          f'vmin={args.vmin:.0e}  vmax={args.vmax:.0e}  ncores={args.ncores}')
    print()

    reference_center        = getattr(args, 'reference_center', None)
    reference_search_radius = getattr(args, 'reference_search_radius', None)
    corotate                = getattr(args, 'corotate', True)
    vmax_vel                = getattr(args, 'vmax_vel', None)
    min_gas_particles       = getattr(args, 'min_gas_particles', 80000)
    min_gas_snap            = getattr(args, 'min_gas_snap', 150)
    include_phase_in_master = getattr(args, 'include_phase_in_master', False)

    # Detect H2 field once from the last snapshot (avoids per-snapshot probing)
    h2_field = None
    if include_phase_in_master and snap_items:
        import h5py as _h5py
        _H2_CANDIDATES = ['MolecularMassFraction', 'Molecular_Fraction',
                          'MolecularHydrogenFraction', 'H2Fraction']
        try:
            with _h5py.File(snap_items[0][0], 'r') as _f:
                for _name in _H2_CANDIDATES:
                    if 'PartType0' in _f and _name in _f['PartType0']:
                        h2_field = _name
                        break
        except Exception:
            pass
        print(f'  Phase panels: h2_field = {h2_field!r}')

    os.makedirs(os.path.join(args.outdir, 'master_frames'), exist_ok=True)

    task_args = [
        (p, n, args.outdir, args.image_box, args.res, args.vmin, args.vmax,
         args.path, args.sim,
         args.r_search, args.r_max, args.rho_thresh, args.aspect, args.f_kep,
         args.cmap, reference_center, reference_search_radius,
         corotate, vmax_vel,
         min_gas_particles, min_gas_snap,
         include_phase_in_master, h2_field)
        for p, n in snap_items
    ]

    import time as _time

    def _fmt_eta(seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f'{h}h{m:02d}m{s:02d}s' if h else f'{m}m{s:02d}s'

    n_total   = len(task_args)
    t_start   = _time.perf_counter()
    completed = 0

    def _report(snap_num, status, elapsed):
        nonlocal completed
        completed += 1
        wall = _time.perf_counter() - t_start
        avg  = wall / completed
        eta  = avg * (n_total - completed)
        if status == 'ok':
            print(f'  snapshot_{snap_num:04d} ok ({elapsed:.1f}s) '
                  f'[{completed}/{n_total}]  ETA {_fmt_eta(eta)}', flush=True)
        elif status.startswith('skipped'):
            print(f'  snapshot_{snap_num:04d} {status} '
                  f'[{completed}/{n_total}]  ETA {_fmt_eta(eta)}', flush=True)
        else:
            print(f'  snapshot_{snap_num:04d} FAILED: {status} '
                  f'[{completed}/{n_total}]  ETA {_fmt_eta(eta)}', flush=True)

    if args.ncores > 1:
        import multiprocessing as _mp
        # 'spawn' avoids fork-deadlocks when yt (threaded) is already imported
        ctx = _mp.get_context('spawn')
        with ctx.Pool(processes=args.ncores) as pool:
            for snap_num, status, elapsed in pool.imap_unordered(process_snapshot, task_args):
                _report(snap_num, status, elapsed)
    else:
        for task in task_args:
            snap_num, status, elapsed = process_snapshot(task)
            _report(snap_num, status, elapsed)

    _frames_dir = os.path.join(args.outdir, 'master_frames')
    print(f'\nDone. Frames saved to: {_frames_dir}')
    print('To assemble with ffmpeg:')
    print(f'  cd {_frames_dir} && '
          f'printf "file \'%s\'\\n" $(ls frame_*.png | sort) > filelist.txt && '
          f'ffmpeg -y -f concat -safe 0 -r 10 -i filelist.txt '
          f'-c:v libx264 -crf 18 -pix_fmt yuv420p ../disk_movie.mp4')

    print('\nBuilding Toomre Q heatmap...')
    make_Q_heatmap(args.outdir)


if __name__ == '__main__':
    args = parse_args()
    main(args)
