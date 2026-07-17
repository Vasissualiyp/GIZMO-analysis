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
# Module-level dark-theme helpers (used by heatmaps, analysis scripts, etc.)
# ═══════════════════════════════════════════════════════════════════════════════

def _darken_fig(fig):
    """Convert a white-bg figure to dark in-place."""
    BG = '#000000'; FG = 'white'
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


def _save_fig_dual(fig, light_path, dark_dir=None):
    """Save figure in both white and dark backgrounds, PNG + PDF."""
    os.makedirs(os.path.dirname(light_path), exist_ok=True)
    fig.savefig(light_path, dpi=150, facecolor='w', bbox_inches='tight')
    pdf_path = os.path.splitext(light_path)[0] + '.pdf'
    fig.savefig(pdf_path, facecolor='w', bbox_inches='tight')
    if dark_dir is None:
        dark_path = light_path.replace('/light/', '/dark/')
    else:
        dark_path = os.path.join(dark_dir, os.path.basename(light_path))
    os.makedirs(os.path.dirname(dark_path), exist_ok=True)
    _darken_fig(fig)
    fig.savefig(dark_path, dpi=150, facecolor='#000000', bbox_inches='tight')
    plt.close(fig)


# Module-level accumulators for de-rotated projection
_derot_prev_time_Myr = None        # previous snapshot time for dt computation
_derot_cumul_theta = None           # cumulative rotation angle per radial bin (rad)
_derot_bin_centers_kpc = None       # radial bin centers used for theta accumulation

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
                 outpath_analysis=None,
                 com_vel=None,
                 corotate=True,
                 vmax_vel=None,
                 v_K=None,
                 data_outdir=None,
                 include_phase=False,
                 h2_field=None,
                 sink_form_Myr=None,
                 sink_r_AU=None,
                 global_ranges=None):
    """
    Renders three figures and saves to outpath / outpath_analysis / frame_phase_*.png:
      Frame A (3×4): SD maps | velocity maps | slice+Bz maps
      Frame B (3×4): scatter+Q | Q map+profiles | resolution+virial+μ
      Frame C (1×4, optional): phase diagrams (when include_phase=True)

    DESIGN RULE: keep each figure to ≤ 3 rows. When adding panels that would
    push a figure beyond 3 rows, create a new Frame figure instead.

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

    # ── Toomre Q = σ_r · κ / (π · G · Σ) ──────────────────────────────────────
    # κ computed numerically: κ² = (2Ω/r) · d(r²Ω)/dr
    # where Ω = v_phi(r)/r from the measured rotation profile.
    v_K_small      = v_K[cut_small] if v_K is not None else np.zeros(len(mass_small))
    bin_centers_kpc = (bins[:-1] + bins[1:]) / 2   # kpc, used by phase + Q panels

    Sigma_prof   = np.zeros(N_BINS)   # g/cm²  (per-annulus surface density)
    sigma_r_prof = np.zeros(N_BINS)   # km/s   (mass-weighted radial dispersion)
    Omega_prof   = np.zeros(N_BINS)   # km/s/kpc — measured Ω = vphi/r per bin

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
        # Measured Ω = vphi / r from mass-weighted rotation profile
        r_mid = bin_centers_kpc[b]
        if r_mid > 0 and vphi_prof[b] != 0:
            Omega_prof[b] = abs(vphi_prof[b]) / r_mid   # km/s/kpc
        else:
            Omega_prof[b] = 0.0

    # Epicyclic frequency: κ² = (2Ω/r) · d(r²Ω)/dr
    kappa_prof = np.zeros(N_BINS)
    r2Omega = bin_centers_kpc**2 * Omega_prof  # r²Ω  [km/s·kpc]
    for b in range(N_BINS):
        r_b = bin_centers_kpc[b]
        if r_b <= 0 or Omega_prof[b] <= 0:
            continue
        # Central differences where possible, one-sided at edges
        if b > 0 and b < N_BINS - 1 and Omega_prof[b-1] > 0 and Omega_prof[b+1] > 0:
            dr = bin_centers_kpc[b+1] - bin_centers_kpc[b-1]
            d_r2Omega = r2Omega[b+1] - r2Omega[b-1]
        elif b < N_BINS - 1 and Omega_prof[b+1] > 0:
            dr = bin_centers_kpc[b+1] - bin_centers_kpc[b]
            d_r2Omega = r2Omega[b+1] - r2Omega[b]
        elif b > 0 and Omega_prof[b-1] > 0:
            dr = bin_centers_kpc[b] - bin_centers_kpc[b-1]
            d_r2Omega = r2Omega[b] - r2Omega[b-1]
        else:
            # Fallback: Keplerian κ = Ω
            kappa_prof[b] = Omega_prof[b]
            continue
        kappa_sq = (2.0 * Omega_prof[b] / r_b) * (d_r2Omega / max(dr, 1e-30))
        kappa_prof[b] = np.sqrt(max(kappa_sq, 0.0))

    with np.errstate(divide='ignore', invalid='ignore'):
        Q_prof = np.where(
            (Sigma_prof > 0) & (kappa_prof > 0),
            (sigma_r_prof * 1e5) * (kappa_prof * 1e5 / kpc) / (np.pi * G * Sigma_prof),
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

    # κ interpolated onto pixel grid (X, Y are in AU)
    r_px_kpc     = np.sqrt(X**2 + Y**2) * AU / kpc
    kappa_px_cgs = np.interp(r_px_kpc, bin_centers_kpc, kappa_prof,
                              left=0.0, right=0.0) * 1e5 / kpc   # 1/s

    # Face-on Q map
    pc_cm         = kpc / 1e3                                     # cm per pc
    Sigma_fo_gcm2 = np.maximum(sig_fo, 0.0) * Msun / pc_cm**2    # Msun/pc² → g/cm²
    with np.errstate(divide='ignore', invalid='ignore'):
        Q_fo = np.where(
            Sigma_fo_gcm2 > 0,
            (sigma_r_fo * 1e5) * kappa_px_cgs / (np.pi * G * Sigma_fo_gcm2),
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

    # ── Resolution profile: dx = (m/ρ)^(1/3) ────────────────────────────────
    _mass_g = mass_small * 1e10 * Msun      # g
    _dx_AU  = (_mass_g / rho_cgs_small)**(1.0 / 3.0) / AU   # AU
    dx_prof = np.zeros(N_BINS)
    for b in range(N_BINS):
        mb = bidx == b
        if mb.sum() == 0:
            continue
        w = mass_small[mb]; wsum = w.sum()
        dx_prof[b] = np.dot(_dx_AU[mb], w) / wsum

    # ── Virial parameter: α = v_rms_turb² · r / (G · M_enc) ─────────────────
    _n_stars_loc  = (len(stardata['Masses'])
                     if stardata and len(stardata.get('Masses', [])) > 0 else 0)
    _M_star_Msun  = np.sum(stardata['Masses']) * 1e10 if _n_stars_loc > 0 else 0.0
    _sort_r       = np.argsort(r_xy)
    _m_sorted_Msun = mass_small[_sort_r] * 1e10       # Msun
    _r_sorted_AU   = r_xy[_sort_r] * kpc / AU         # AU
    _M_cum_Msun    = np.cumsum(_m_sorted_Msun)         # cumulative gas mass [Msun]
    vrms_sq_prof   = np.zeros(N_BINS)                  # km²/s² (turbulent)
    M_enc_bin      = np.zeros(N_BINS)                  # total enclosed mass [Msun]
    for b in range(N_BINS):
        mb = bidx == b
        if mb.sum() == 0:
            M_enc_bin[b] = np.nan; continue
        w = mass_small[mb]; wsum = w.sum()
        vrms_sq_prof[b] = np.dot(v_rest[mb]**2, w) / wsum          # (km/s)²
        r_out_AU = bins[b + 1] * kpc / AU
        idx_enc  = int(np.searchsorted(_r_sorted_AU, r_out_AU))
        M_enc_bin[b] = float(_M_cum_Msun[min(idx_enc, len(_M_cum_Msun) - 1)]) if idx_enc > 0 else 0.0
    M_enc_bin = M_enc_bin + _M_star_Msun   # add stellar contribution
    _r_bin_AU = bin_centers_kpc * kpc / AU
    with np.errstate(divide='ignore', invalid='ignore'):
        virial_prof = np.where(
            M_enc_bin > 0,
            (vrms_sq_prof * (1e5)**2) * (_r_bin_AU * AU) / (G * M_enc_bin * Msun),
            np.nan,
        )

    # ── Mass-to-flux ratio: μ = 2π√G · Σ / |B_z| ────────────────────────────
    if 'MagneticField' in pdata:
        _B_raw = pdata['MagneticField'][cut_small]   # (N, 3) — Gauss after GUAC conversion
        _B_rot = _B_raw @ rot.T                       # rotate to disk frame (L_hat → z)
        _Bz    = _B_rot[:, 2]                         # z-component (normal to disk plane)
        _sqrt_G = np.sqrt(G)                          # sqrt(cm³/g/s²)
        mf_prof = np.zeros(N_BINS)
        for b in range(N_BINS):
            mb = bidx == b
            if mb.sum() == 0:
                mf_prof[b] = np.nan; continue
            w = mass_small[mb]; wsum = w.sum()
            _Bz_mw = np.dot(np.abs(_Bz[mb]), w) / wsum   # mass-weighted |B_z| [Gauss]
            if _Bz_mw > 0 and Sigma_prof[b] > 0:
                mf_prof[b] = 2.0 * np.pi * _sqrt_G * Sigma_prof[b] / _Bz_mw
            else:
                mf_prof[b] = np.nan
        # Face-on |B_z| map via Meshoid projection
        if M_fo is not None:
            _sm_norm = np.maximum(
                M_fo.SurfaceDensity(M_fo.m, center=center0, size=image_box_kpc, res=res), 1e-40)
            Bz_fo_map = M_fo.SurfaceDensity(
                M_fo.m * np.abs(_Bz), center=center0, size=image_box_kpc, res=res) / _sm_norm
        else:
            Bz_fo_map = None
    else:
        _Bz = None; mf_prof = np.full(N_BINS, np.nan); Bz_fo_map = None

    # ── Face-on midplane density slice ────────────────────────────────────────
    # Select particles within 1 median smoothing length of the disk midplane (z=0)
    _z_cut_kpc  = max(np.median(hsml_small), np.percentile(np.abs(pos_fo[:, 2]), 5))
    _slice_mask = np.abs(pos_fo[:, 2]) < _z_cut_kpc
    if _slice_mask.sum() >= 5:
        sig_slice, _ = _surf(pos_fo[_slice_mask], mass_small[_slice_mask],
                             hsml_small[_slice_mask], image_box_kpc)
    else:
        sig_slice = np.zeros((res, res))

    # Save Q profile in its own subdirectory
    _npz_dir = os.path.join(data_outdir if data_outdir is not None else os.path.dirname(outpath),
                            'qprofiles')
    os.makedirs(_npz_dir, exist_ok=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        _Q_combined = np.where(
            (Sigma_prof > 0) & (kappa_prof > 0),
            (np.sqrt(sigma_r_prof**2 + cs_prof**2) * 1e5) * (kappa_prof * 1e5 / kpc) / (np.pi * G * Sigma_prof),
            np.nan)
    np.savez(
        os.path.join(_npz_dir, f'qprofile_{snap_num:04d}.npz'),
        r_kpc    = bin_centers_kpc,
        r_AU     = bin_centers_kpc * kpc / AU,
        Q        = Q_prof,
        Q_combined = _Q_combined,
        cs_prof  = cs_prof,
        Sigma    = Sigma_prof,
        sigma_r  = sigma_r_prof,
        Omega    = Omega_prof,
        kappa    = kappa_prof,
        vphi     = vphi_prof,
        time_Myr      = np.array([time_Myr]),
        snap_num      = np.array([snap_num]),
        n_sinks       = np.array([len(stardata['Masses']) if stardata and len(stardata.get('Masses', [])) > 0 else 0]),
        sink_form_Myr = np.array(sink_form_Myr) if sink_form_Myr is not None else np.array([]),
        sink_r_AU     = np.array(sink_r_AU)     if sink_r_AU     is not None else np.array([]),
        Q_fo_2d  = Q_fo,
        X_grid   = X,
        Y_grid   = Y,
    )

    # Save spherical shell mass profile alongside qprofile
    _mp_dir = os.path.join(data_outdir if data_outdir is not None else os.path.dirname(outpath),
                           'massprofiles')
    os.makedirs(_mp_dir, exist_ok=True)
    _N_SPH_MP = 50
    _r_sph_max_kpc_mp = 1e4 * AU / kpc   # 10,000 AU
    _r_sph_min_kpc_mp = 10.0 * AU / kpc  # 10 AU
    _sph_edges_mp = np.logspace(np.log10(_r_sph_min_kpc_mp),
                                np.log10(_r_sph_max_kpc_mp), _N_SPH_MP + 1)
    _sph_ctr_AU_mp = np.sqrt(_sph_edges_mp[:-1] * _sph_edges_mp[1:]) * kpc / AU
    _cut_mp = gas_dists < _r_sph_max_kpc_mp
    _m_shell_mp = np.zeros(_N_SPH_MP)
    if _cut_mp.sum() > 3:
        _r_mp    = gas_dists[_cut_mp]
        _mass_mp = pdata['Masses'][_cut_mp]
        _bidx_mp = np.clip(np.digitize(_r_mp, _sph_edges_mp) - 1, 0, _N_SPH_MP - 1)
        for _b in range(_N_SPH_MP):
            _mb = _bidx_mp == _b
            if _mb.sum() > 0:
                _m_shell_mp[_b] = _mass_mp[_mb].sum() * 1e10   # Msun
    np.savez(
        os.path.join(_mp_dir, f'massprofile_{snap_num:04d}.npz'),
        r_AU     = _sph_ctr_AU_mp,
        m_shell  = _m_shell_mp,
        time_Myr = np.array([time_Myr]),
        n_sinks  = np.array([len(stardata['Masses']) if stardata and len(stardata.get('Masses', [])) > 0 else 0]),
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

    # ── Master figure setup ───────────────────────────────────────────────────
    fig_c = None; _ax_ph_T = None; _ax_ph_fh2 = None

    r_xy_AU  = r_xy * kpc / AU
    bin_AU   = bin_centers_kpc * kpc / AU
    r_max_AU = image_box_kpc / 2 * kpc / AU

    # Sink coordinates in face-on and edge-on frames
    if n_stars > 0:
        _sc        = stardata['Coordinates'] - com
        star_fo_AU = _sc @ rot_fo.T * kpc / AU
        star_eo_AU = _sc @ rot.T    * kpc / AU
    else:
        star_fo_AU = star_eo_AU = np.zeros((0, 3))

    _title = (
        f'Snap {snap_num:04d}   t = {time_Myr*1e3:.2f} kyr   '
        f'N_stars = {n_stars}   M_stars = {M_stars:.3f} Msun   '
        f'M_disk = {M_disk:.2f} Msun   R_disk = {R_disk_AU:.0f} AU   '
        f'f_star = {f_star*100:.2f}%   '
        f'ρ_central = {rho_central:.2e} g/cm³'
    )

    # Output path helpers
    outdir_pub  = os.path.dirname(outpath)                        # .../framesXX/light/master_frames/
    _base       = os.path.dirname(os.path.dirname(outdir_pub))    # .../framesXX/
    outdir_dark = os.path.join(_base, 'dark', 'master_frames')

    def _pub_path(name):
        return os.path.join(outdir_pub,  f'{name}_{snap_num:04d}.png')

    def _dark_path(name):
        return os.path.join(outdir_dark, f'{name}_{snap_num:04d}.png')

    # ── Dark-background helpers ───────────────────────────────────────────────
    def _darken(fig):
        BG = '#000000'; FG = 'white'
        fig.patch.set_facecolor(BG)
        for ax in fig.axes:
            ax.set_facecolor(BG)
            ax.tick_params(colors=FG, which='both')
            ax.xaxis.label.set_color(FG)
            ax.yaxis.label.set_color(FG)
            ax.title.set_color(FG)
            for sp in ax.spines.values():
                sp.set_edgecolor(FG)
            for txt in ax.get_xticklabels() + ax.get_yticklabels():
                txt.set_color(FG)
            leg = ax.get_legend()
            if leg:
                leg.get_frame().set_facecolor('#2a2a2a')
                leg.get_frame().set_edgecolor('#555')
                for t in leg.get_texts():
                    t.set_color(FG)
            for line in ax.get_lines():
                if line.get_color() in ('k', 'black', '#000000', '#222222'):
                    line.set_color(FG)
        # Also darken colorbars (axes that are not data axes)
        for ax in fig.axes:
            for coll in ax.collections:
                pass  # colorbars handled via fig.axes loop above

    def _save_dual(fig, pub_path, dark_path):
        os.makedirs(os.path.dirname(pub_path),  exist_ok=True)
        os.makedirs(os.path.dirname(dark_path), exist_ok=True)
        fig.savefig(pub_path,  dpi=150, facecolor='w',       bbox_inches='tight')
        _darken(fig)
        fig.savefig(dark_path, dpi=150, facecolor='#000000', bbox_inches='tight')
        plt.close(fig)

    # Shared tick style
    _tkw = dict(which='both', direction='in', right=True, top=True)

    def _style_ax(ax, xlabel, ylabel, half=None):
        """Apply axis labels, tick params, spines; no title."""
        ax.set_facecolor('w')
        ax.set_xlabel(xlabel, color='k', fontsize=12)
        ax.set_ylabel(ylabel, color='k', fontsize=12)
        ax.tick_params(colors='k', labelsize=10, **_tkw)
        for sp in ax.spines.values():
            sp.set_edgecolor('k')
        if half is not None:
            ax.set_xlim(-half, half)
            ax.set_ylim(-half, half)

    def _add_stars(ax, spos2, half):
        """Scatter white/gold star markers for sinks within the view."""
        if n_stars > 0 and len(spos2):
            inv = (np.abs(spos2[:, 0]) < half) & (np.abs(spos2[:, 1]) < half)
            if inv.any():
                ax.scatter(spos2[inv, 0], spos2[inv, 1], s=30, c='white',
                           marker='*', zorder=5, edgecolors='gold', linewidths=0.5)

    _sfo2 = star_fo_AU[:, :2]     if n_stars > 0 else np.zeros((0, 2))
    _seo2 = star_eo_AU[:, [0, 2]] if n_stars > 0 else np.zeros((0, 2))

    # ── fig1: Surface density (face-on | edge-on+overlay), 1×2 compact ───────
    fig1, (ax_fo, ax_eo) = plt.subplots(1, 2, figsize=(9, 5),
                                         sharey=True,
                                         gridspec_kw={'wspace': 0})
    fig1.patch.set_facecolor('w')
    _d_fo = np.where(sig_fo > 0, sig_fo, norm_small.vmin)
    _d_eo = np.where(sig_eo > 0, sig_eo, norm_small.vmin)
    im_fo = ax_fo.pcolormesh(X, Y, _d_fo, norm=norm_small, cmap=cmap)
    im_eo = ax_eo.pcolormesh(X, Y, _d_eo, norm=norm_small, cmap=cmap)
    if disk_small.sum() > 0:
        ax_eo.scatter(disk_eo_AU[:, 0], disk_eo_AU[:, 1],
                      s=0.5, alpha=0.3, c='cyan', rasterized=True)
    _add_stars(ax_fo, _sfo2, half_AU)
    _add_stars(ax_eo, _seo2, half_AU)
    _style_ax(ax_fo, 'x (AU)', 'y (AU)', half_AU)
    _style_ax(ax_eo, 'x (AU)', '',       half_AU)
    ax_eo.tick_params(labelleft=False)
    _cb1 = fig1.colorbar(im_eo, ax=ax_eo, pad=0.02)
    _cb1.set_label(r'$\Sigma\;(\mathrm{M}_\odot\,\mathrm{pc}^{-2})$', color='k', fontsize=11)
    _cb1.ax.yaxis.set_tick_params(color='k', labelsize=10)
    plt.setp(_cb1.ax.yaxis.get_ticklabels(), color='k')
    _save_dual(fig1, _pub_path('frame_sd'), _dark_path('frame_sd'))

    # ── fig2: Rest-frame velocity |δv| (face-on | edge-on), 1×2 compact ──────
    fig2, (ax_vfo, ax_veo) = plt.subplots(1, 2, figsize=(9, 5),
                                           sharey=True,
                                           gridspec_kw={'wspace': 0})
    fig2.patch.set_facecolor('w')
    im_vfo = ax_vfo.pcolormesh(X, Y, vrest_fo, norm=norm_vrest, cmap='viridis')
    im_veo = ax_veo.pcolormesh(X, Y, vrest_eo, norm=norm_vrest, cmap='viridis')
    _add_stars(ax_vfo, _sfo2, half_AU)
    _add_stars(ax_veo, _seo2, half_AU)
    _style_ax(ax_vfo, 'x (AU)', r'$|\delta v|$ (km s$^{-1}$)', half_AU)
    _style_ax(ax_veo, 'x (AU)', '',                             half_AU)
    ax_veo.tick_params(labelleft=False)
    _cb2 = fig2.colorbar(im_veo, ax=ax_veo, pad=0.02)
    _cb2.set_label(r'$|\delta v|\;(\mathrm{km\,s}^{-1})$', color='k', fontsize=11)
    _cb2.ax.yaxis.set_tick_params(color='k', labelsize=10)
    plt.setp(_cb2.ax.yaxis.get_ticklabels(), color='k')
    _save_dual(fig2, _pub_path('frame_vel'), _dark_path('frame_vel'))

    # ── fig3: Kinematics scatter, 2×1 stacked (hspace=0, sharex) ─────────────
    fig3, (ax_vp, ax_vr) = plt.subplots(2, 1, figsize=(5.5, 8),
                                         sharex=True,
                                         gridspec_kw={'hspace': 0})
    fig3.patch.set_facecolor('w')
    ax_vp.scatter(r_xy_AU, v_phi, s=0.3, alpha=0.15, c='darkorange', rasterized=True)
    ax_vp.plot(bin_AU, vphi_prof, 'r-', lw=2, label='profile')
    ax_vp.axhline(0, color='k', lw=0.8, ls='--')
    ax_vp.axvline(r_max_AU, color='k', lw=0.8, ls=':')
    ax_vp.set_ylim(-20, 20)
    ax_vp.set_ylabel(r'$v_\phi\;(\mathrm{km\,s}^{-1})$', color='k', fontsize=12)
    ax_vp.set_facecolor('w')
    ax_vp.tick_params(colors='k', labelsize=10, labelbottom=False, **_tkw)
    for sp in ax_vp.spines.values(): sp.set_edgecolor('k')
    _lp = ax_vp.legend(fontsize=9, framealpha=0.8, facecolor='w')
    for _t in _lp.get_texts(): _t.set_color('k')
    ax_vr.scatter(r_xy_AU, v_r, s=0.3, alpha=0.15, c='cyan', rasterized=True)
    ax_vr.plot(bin_AU, vr_prof, 'r-', lw=2, label='profile')
    ax_vr.axhline(0, color='k', lw=0.8, ls='--')
    ax_vr.axvline(r_max_AU, color='k', lw=0.8, ls=':')
    ax_vr.set_ylim(-10, 10)
    ax_vr.set_xlim(0, r_max_AU * 1.05)
    ax_vr.set_xlabel('r (AU)', color='k', fontsize=12)
    ax_vr.set_ylabel(r'$v_r\;(\mathrm{km\,s}^{-1})$', color='k', fontsize=12)
    ax_vr.set_facecolor('w')
    ax_vr.tick_params(colors='k', labelsize=10, **_tkw)
    for sp in ax_vr.spines.values(): sp.set_edgecolor('k')
    _lr = ax_vr.legend(fontsize=9, framealpha=0.8, facecolor='w')
    for _t in _lr.get_texts(): _t.set_color('k')
    _save_dual(fig3, _pub_path('frame_kin'), _dark_path('frame_kin'))

    # ── fig4: Toomre Q map + 1D profile, 1×2 (separate axes types) ───────────
    fig4, (ax_qmap, ax_q1d) = plt.subplots(1, 2, figsize=(9, 5))
    fig4.patch.set_facecolor('w')
    Q_plot = np.where(Q_fo > 0, Q_fo, np.nan)
    norm_Q = colors.LogNorm(vmin=0.1, vmax=10)
    im_q   = ax_qmap.pcolormesh(X, Y, Q_plot, norm=norm_Q, cmap='RdYlGn')
    try:
        ax_qmap.contour(X, Y, np.where(np.isfinite(Q_fo), Q_fo, 1.0),
                        levels=[1.0], colors='k', linewidths=1.5)
    except Exception:
        pass
    _add_stars(ax_qmap, _sfo2, half_AU)
    _cb4 = fig4.colorbar(im_q, ax=ax_qmap, pad=0.02)
    _cb4.set_label('Toomre Q', color='k', fontsize=11)
    _cb4.ax.yaxis.set_tick_params(color='k', labelsize=10)
    plt.setp(_cb4.ax.yaxis.get_ticklabels(), color='k')
    _style_ax(ax_qmap, 'x (AU)', 'y (AU)', half_AU)
    valid_q = np.isfinite(Q_prof) & (Q_prof > 0)
    if valid_q.any():
        ax_q1d.semilogy(bin_AU[valid_q], Q_prof[valid_q], 'k-o', ms=4, lw=1.5)
    ax_q1d.axhline(1.0, color='r', lw=1.5, ls='--', label='Q = 1')
    ax_q1d.set_xlim(0, r_max_AU * 1.05)
    ax_q1d.set_ylim(0.1, 100)
    _style_ax(ax_q1d, 'r (AU)', 'Toomre Q')
    _lq = ax_q1d.legend(fontsize=9, framealpha=0.8, facecolor='w')
    for _t in _lq.get_texts(): _t.set_color('k')
    fig4.tight_layout()
    _save_dual(fig4, _pub_path('frame_toomre'), _dark_path('frame_toomre'))

    # ── fig5: Radial profiles, 2×1 stacked ──────────────────────────────────
    fig5, (ax_rho, ax_vel5) = plt.subplots(2, 1, figsize=(5.5, 8),
                                            gridspec_kw={'hspace': 0.15})
    fig5.patch.set_facecolor('w')
    valid_rho = (rho_prof > 0) & (bin_ctr_rho_AU > 0)
    if valid_rho.any():
        ax_rho.loglog(bin_ctr_rho_AU[valid_rho], rho_prof[valid_rho], '#222222', lw=1.5)
        # Power-law fit
        if valid_rho.sum() >= 4:
            _fit_mask = valid_rho & (bin_ctr_rho_AU > bin_ctr_rho_AU[valid_rho][1])
            if _fit_mask.sum() >= 3:
                _lr = np.log10(bin_ctr_rho_AU[_fit_mask])
                _lrho = np.log10(rho_prof[_fit_mask])
                _slope, _intercept = np.polyfit(_lr, _lrho, 1)
                _fit_r = bin_ctr_rho_AU[_fit_mask]
                ax_rho.loglog(_fit_r, 10**(_intercept) * _fit_r**_slope,
                              'r--', lw=1, alpha=0.7)
                ax_rho.text(0.95, 0.95, rf'$\rho \propto r^{{{_slope:.1f}}}$',
                            transform=ax_rho.transAxes, ha='right', va='top',
                            fontsize=10, color='r')
    if global_ranges is not None:
        ax_rho.set_ylim(global_ranges['rho_ylim'])
    ax_rho.set_ylabel(r'$\rho\;(\mathrm{g\,cm}^{-3})$', color='k', fontsize=12)
    ax_rho.set_facecolor('w')
    ax_rho.tick_params(colors='k', labelsize=10, labelbottom=False, **_tkw)
    for sp in ax_rho.spines.values(): sp.set_edgecolor('k')
    valid_b = bin_AU > 0
    ax_vel5.plot(bin_AU[valid_b], cs_prof[valid_b],      'r-',    lw=2,   label=r'$c_s$')
    ax_vel5.plot(bin_AU[valid_b], sigma_r_prof[valid_b], 'c-',    lw=2,   label=r'$\sigma_r$')
    ax_vel5.plot(bin_AU[valid_b], vturb_prof[valid_b],   'olive', lw=1.5, label=r'$\langle|\delta v|\rangle$')
    ax_vel5.set_xlim(0, r_max_AU * 1.05)
    if global_ranges is not None:
        ax_vel5.set_ylim(0, global_ranges['vel_ymax'] * 1.05)
    else:
        ax_vel5.set_ylim(bottom=0)
    ax_vel5.set_xlabel('r (AU)', color='k', fontsize=12)
    ax_vel5.set_ylabel(r'Velocity $(\mathrm{km\,s}^{-1})$', color='k', fontsize=12)
    ax_vel5.set_facecolor('w')
    ax_vel5.tick_params(colors='k', labelsize=10, **_tkw)
    for sp in ax_vel5.spines.values(): sp.set_edgecolor('k')
    _lv5 = ax_vel5.legend(fontsize=9, framealpha=0.8, facecolor='w')
    for _t in _lv5.get_texts(): _t.set_color('k')
    _save_dual(fig5, _pub_path('frame_prof'), _dark_path('frame_prof'))

    # ── Individual themed figures ────────────────────────────────────────────
    # Each is 1–2 rows so it can be "sprinkled" individually into a paper.
    # Saved to {data_outdir}/individual_frames/frame_{theme}_{snap_num:04d}.png
    if data_outdir is not None:
        _idir = os.path.join(data_outdir, 'light', 'individual_frames')
        _idir_dark = os.path.join(data_outdir, 'dark', 'individual_frames')
        os.makedirs(_idir, exist_ok=True)

        def _ifpath(theme):
            return os.path.join(_idir, f'frame_{theme}_{snap_num:04d}.png')

        def _if_dark_path(theme):
            return os.path.join(_idir_dark, f'frame_{theme}_{snap_num:04d}.png')

        # ── helpers ──────────────────────────────────────────────────────────
        def _ax_base(ax, xl, yl, half=None):
            ax.set_facecolor('w')
            ax.set_xlabel(xl, color='k', fontsize=11)
            ax.set_ylabel(yl, color='k', fontsize=11)
            ax.tick_params(colors='k', which='both', direction='in', right=True, top=True)
            for sp in ax.spines.values(): sp.set_edgecolor('k')
            if half is not None:
                ax.set_xlim(-half, half); ax.set_ylim(-half, half)

        def _cb(fig, im, ax, label):
            c = fig.colorbar(im, ax=ax)
            c.set_label(label, color='k', fontsize=10)
            c.ax.yaxis.set_tick_params(color='k')
            plt.setp(c.ax.yaxis.get_ticklabels(), color='k')

        def _star_scatter(ax, spos, half):
            if n_stars > 0:
                inv = (np.abs(spos[:, 0]) < half) & (np.abs(spos[:, 1]) < half)
                if inv.any():
                    ax.scatter(spos[inv, 0], spos[inv, 1], s=30, c='white',
                               marker='*', zorder=5, edgecolors='yellow', linewidths=0.5)

        def _draw_sd(ax, fig, Xg, Yg, sig, norm_p, half, xl, yl, overlay=None):
            _d = np.where(sig > 0, sig, norm_p.vmin)
            im = ax.pcolormesh(Xg, Yg, _d, norm=norm_p, cmap=cmap)
            _cb(fig, im, ax, r'$\Sigma$ (M$_\odot$/pc$^2$)')
            if overlay is not None and len(overlay) > 0:
                ax.scatter(overlay[:, 0], overlay[:, 1], s=0.5, alpha=0.3,
                           c='cyan', rasterized=True)
            _ax_base(ax, xl, yl, half)

        def _draw_vel(ax, fig, Xg, Yg, vmap, norm_p, half, xl, yl, vl, vcmap='viridis'):
            im = ax.pcolormesh(Xg, Yg, vmap, norm=norm_p, cmap=vcmap)
            _cb(fig, im, ax, vl)
            _ax_base(ax, xl, yl, half)

        def _add_stars_sd(ax, spos2, half):
            if spos2 is not None and len(spos2):
                _star_scatter(ax, spos2, half)

        # ── 1. Density maps (face-on + edge-on, shared colorbar) ────────────
        fig_i = plt.figure(figsize=(11, 5))
        gs = fig_i.add_gridspec(1, 3, wspace=0,
                                width_ratios=[1, 1, 0.05])
        ax_fo = fig_i.add_subplot(gs[0, 0])
        ax_eo = fig_i.add_subplot(gs[0, 1], sharey=ax_fo)
        cax = fig_i.add_subplot(gs[0, 2])
        fig_i.patch.set_facecolor('w')

        _d_fo_i = np.where(sig_fo > 0, sig_fo, norm_small.vmin)
        _d_eo_i = np.where(sig_eo > 0, sig_eo, norm_small.vmin)
        im_fo_i = ax_fo.pcolormesh(X, Y, _d_fo_i, norm=norm_small, cmap=cmap)
        ax_eo.pcolormesh(X, Y, _d_eo_i, norm=norm_small, cmap=cmap)
        cb_i = fig_i.colorbar(im_fo_i, cax=cax)
        cb_i.set_label(r'$\Sigma$ (M$_\odot$/pc$^2$)', color='k', fontsize=12)
        cb_i.ax.yaxis.set_tick_params(color='k', labelsize=10)
        plt.setp(cb_i.ax.yaxis.get_ticklabels(), color='k')

        for _a in [ax_fo, ax_eo]:
            _a.set_xlim(-half_AU, half_AU); _a.set_ylim(-half_AU, half_AU)
            _a.set_facecolor('w')
            _a.tick_params(colors='k', which='both', direction='in',
                           right=True, top=True, labelsize=10)
            for sp in _a.spines.values(): sp.set_edgecolor('k')
        ax_eo.tick_params(labelleft=False)
        ax_fo.set_ylabel('y (AU)', color='k', fontsize=12)
        ax_fo.set_xlabel('x (AU)', color='k', fontsize=12)
        ax_eo.set_xlabel('x (AU)', color='k', fontsize=12)
        # Remove clashing max tick on left, min tick on right
        ax_fo.xaxis.get_major_locator().set_params(prune='upper')
        ax_eo.xaxis.get_major_locator().set_params(prune='lower')
        # Stars
        if n_stars > 0:
            for _a, sp2 in [(ax_fo, star_fo_AU[:, :2]),
                            (ax_eo, star_eo_AU[:, [0,2]])]:
                iv = (np.abs(sp2[:, 0]) < half_AU) & (np.abs(sp2[:, 1]) < half_AU)
                if iv.any():
                    _a.scatter(sp2[iv, 0], sp2[iv, 1], s=20, c='w',
                               marker='*', zorder=5, edgecolors='gold', lw=0.5)
        _save_dual(fig_i, _ifpath('density'), _if_dark_path('density'))

        # ── 2. Velocity dispersion maps ──────────────────────────────────────
        fig_i = plt.figure(figsize=(12, 10))
        gs = fig_i.add_gridspec(2, 3, wspace=0, hspace=0,
                                width_ratios=[1, 1, 0.05])
        ax_vfo2 = fig_i.add_subplot(gs[0, 0])
        ax_veo2 = fig_i.add_subplot(gs[0, 1], sharey=ax_vfo2)
        ax_sfo2 = fig_i.add_subplot(gs[1, 0])
        ax_seo2 = fig_i.add_subplot(gs[1, 1], sharey=ax_sfo2)
        # Two colorbars: one for |dv| (top), one for sigma (bottom)
        cax_top = fig_i.add_subplot(gs[0, 2])
        cax_bot = fig_i.add_subplot(gs[1, 2])
        fig_i.patch.set_facecolor('w')

        # Row 0: |dv| face-on + edge-on (shared colorbar)
        im_vfo2 = ax_vfo2.pcolormesh(X, Y, vrest_fo, norm=norm_vrest, cmap='viridis')
        im_veo2 = ax_veo2.pcolormesh(X, Y, vrest_eo, norm=norm_vrest, cmap='viridis')
        cb_v2 = fig_i.colorbar(im_vfo2, cax=cax_top)
        cb_v2.set_label(r'$|\delta v|$ (km/s)', color='k', fontsize=12)
        cb_v2.ax.yaxis.set_tick_params(color='k', labelsize=10)
        plt.setp(cb_v2.ax.yaxis.get_ticklabels(), color='k')

        # Row 1: sigma_|dv| face-on + edge-on (shared colorbar)
        im_sfo2 = ax_sfo2.pcolormesh(X, Y, sigv_fo, norm=norm_sigv, cmap='viridis')
        im_seo2 = ax_seo2.pcolormesh(X, Y, sigv_eo, norm=norm_sigv, cmap='viridis')
        cb_s2 = fig_i.colorbar(im_sfo2, cax=cax_bot)
        cb_s2.set_label(r'$\sigma_{|\delta v|}$ (km/s)', color='k', fontsize=12)
        cb_s2.ax.yaxis.set_tick_params(color='k', labelsize=10)
        plt.setp(cb_s2.ax.yaxis.get_ticklabels(), color='k')

        # Styling
        for _a in [ax_vfo2, ax_veo2, ax_sfo2, ax_seo2]:
            _a.set_xlim(-half_AU, half_AU); _a.set_ylim(-half_AU, half_AU)
            _a.set_facecolor('w')
            _a.tick_params(colors='k', which='both', direction='in',
                           right=True, top=True, labelsize=10)
            for sp in _a.spines.values(): sp.set_edgecolor('k')
        ax_veo2.tick_params(labelleft=False)
        ax_seo2.tick_params(labelleft=False)
        ax_vfo2.tick_params(labelbottom=False)
        ax_veo2.tick_params(labelbottom=False, labelleft=False)
        ax_vfo2.set_ylabel('y (AU)', color='k', fontsize=12)
        ax_sfo2.set_ylabel('y (AU)', color='k', fontsize=12)
        ax_sfo2.set_xlabel('x (AU)', color='k', fontsize=12)
        ax_seo2.set_xlabel('x (AU)', color='k', fontsize=12)
        # Remove clashing max tick on left, min tick on right
        ax_sfo2.xaxis.get_major_locator().set_params(prune='upper')
        ax_seo2.xaxis.get_major_locator().set_params(prune='lower')
        # Stars
        if n_stars > 0:
            for _a, sp2 in [(ax_vfo2, star_fo_AU[:, :2]), (ax_veo2, star_eo_AU[:, [0,2]]),
                             (ax_sfo2, star_fo_AU[:, :2]), (ax_seo2, star_eo_AU[:, [0,2]])]:
                iv = (np.abs(sp2[:, 0]) < half_AU) & (np.abs(sp2[:, 1]) < half_AU)
                if iv.any():
                    _a.scatter(sp2[iv, 0], sp2[iv, 1], s=20, c='w',
                               marker='*', zorder=5, edgecolors='gold', lw=0.5)
        _save_dual(fig_i, _ifpath('velocity'), _if_dark_path('velocity'))

        # ── 3. Kinematics: velocity vs r scatter ─────────────────────────────
        fig_i, _ax = plt.subplots(1, 3, figsize=(21, 7))
        fig_i.patch.set_facecolor('w')
        for ax_k, ydata, yfit, ylabel_k, ptc, ylim_k in [
            (_ax[0], v_r,    vr_prof,    r'$v_r$ (km/s)',         'cyan',   (-10, 10)),
            (_ax[1], v_phi,  vphi_prof,  r'$v_\phi$ (km/s)',      'orange', (-20, 20)),
            (_ax[2], v_rest, vturb_prof, r'$|\delta v|$ (km/s)',  'lime',   (0,   20)),
        ]:
            ax_k.scatter(r_xy_AU, ydata, s=0.3, alpha=0.15, c=ptc, rasterized=True)
            ax_k.plot(bin_AU, yfit, 'r-', lw=2, label='profile')
            ax_k.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
            ax_k.axvline(r_max_AU, color='k', lw=0.5, ls=':', alpha=0.4)
            ax_k.set_xlim(0, r_max_AU * 1.05); ax_k.set_ylim(*ylim_k)
            _ax_base(ax_k, 'r (AU)', ylabel_k)
            _leg = ax_k.legend(fontsize=9, framealpha=0.8, facecolor='w')
            for _t in _leg.get_texts(): _t.set_color('k')
        fig_i.tight_layout()
        _save_dual(fig_i, _ifpath('kinematics'), _if_dark_path('kinematics'))

        # ── 4. Toomre Q ───────────────────────────────────────────────────────
        fig_i, _ax = plt.subplots(1, 2, figsize=(14, 7))
        fig_i.patch.set_facecolor('w')
        _Q_plot = np.where(Q_fo > 0, Q_fo, np.nan)
        im_q = _ax[0].pcolormesh(X, Y, _Q_plot, norm=colors.LogNorm(0.1, 10), cmap='RdYlGn')
        _cb(fig_i, im_q, _ax[0], 'Toomre Q')
        try:
            _ax[0].contour(X, Y, np.where(np.isfinite(Q_fo), Q_fo, 1.0),
                           levels=[1.0], colors='k', linewidths=1.5)
        except Exception:
            pass
        _add_stars_sd(_ax[0], _sfo2, half_AU)
        _ax_base(_ax[0], 'x (AU)', 'y (AU)', half_AU)
        _vq = np.isfinite(Q_prof) & (Q_prof > 0)
        if _vq.any():
            _ax[1].semilogy(bin_AU[_vq], Q_prof[_vq], 'k-o', ms=5, lw=2)
        _ax[1].axhline(1.0, color='r', lw=1.5, ls='--', label='Q = 1')
        _ax[1].set_xlim(0, r_max_AU * 1.05); _ax[1].set_ylim(0.1, 100)
        _ax_base(_ax[1], 'r (AU)', 'Toomre Q')
        _leg2 = _ax[1].legend(fontsize=9, framealpha=0.8, facecolor='w')
        for _t in _leg2.get_texts(): _t.set_color('k')
        fig_i.tight_layout()
        _save_dual(fig_i, _ifpath('toomre'), _if_dark_path('toomre'))

        # ── 5a. Log-axis profiles: rho + dx (shared log x) ────────────────────
        fig_i, (_ax_rho2, _ax_dx) = plt.subplots(2, 1, figsize=(6, 8),
                                                   sharex=True,
                                                   gridspec_kw={'hspace': 0})
        fig_i.patch.set_facecolor('w')
        _vr2 = (rho_prof > 0) & (bin_ctr_rho_AU > 0)
        if _vr2.any():
            _ax_rho2.loglog(bin_ctr_rho_AU[_vr2], rho_prof[_vr2], '#222222', lw=2)
            # Power-law fit
            _fm = _vr2 & (bin_ctr_rho_AU > bin_ctr_rho_AU[_vr2][1]) if _vr2.sum() >= 4 else np.zeros_like(_vr2)
            if _fm.sum() >= 3:
                _lr = np.log10(bin_ctr_rho_AU[_fm])
                _lrho = np.log10(rho_prof[_fm])
                _sl, _ic = np.polyfit(_lr, _lrho, 1)
                _fr = bin_ctr_rho_AU[_fm]
                _ax_rho2.loglog(_fr, 10**_ic * _fr**_sl, 'r--', lw=1, alpha=0.7)
                _ax_rho2.text(0.95, 0.95, rf'$\rho \propto r^{{{_sl:.1f}}}$',
                              transform=_ax_rho2.transAxes, ha='right', va='top',
                              fontsize=10, color='r')
        if global_ranges is not None:
            _ax_rho2.set_ylim(global_ranges['rho_ylim'])
        _ax_base(_ax_rho2, '', r'$\rho$ (g/cm³)')
        _ax_rho2.tick_params(labelbottom=False)
        # dx resolution
        _vdx = (dx_prof > 0) & (bin_AU > 0)
        if _vdx.any():
            _ax_dx.loglog(bin_AU[_vdx], dx_prof[_vdx], 'k-o', ms=5, lw=2)
        _ax_dx.axhline(np.median(_dx_AU), color='c', lw=1, ls='--', label='median dx')
        _ax_base(_ax_dx, 'r (AU)', r'$\Delta x$ (AU)')
        _leg_dx = _ax_dx.legend(fontsize=9, framealpha=0.8, facecolor='w')
        for _t in _leg_dx.get_texts(): _t.set_color('k')
        _save_dual(fig_i, _ifpath('profiles_log'), _if_dark_path('profiles_log'))

        # ── 5b. Linear-axis profiles: velocities + Mach (shared linear x) ────
        fig_i, (_ax_vel2, _ax_mach) = plt.subplots(2, 1, figsize=(6, 8),
                                                     sharex=True,
                                                     gridspec_kw={'hspace': 0})
        fig_i.patch.set_facecolor('w')
        _vb = bin_AU > 0
        _ax_vel2.plot(bin_AU[_vb], cs_prof[_vb],      'r-',  lw=2, label=r'$c_s$')
        _ax_vel2.plot(bin_AU[_vb], sigma_r_prof[_vb],  'c-',  lw=2, label=r'$\sigma_r$')
        _ax_vel2.plot(bin_AU[_vb], vturb_prof[_vb],    'olive', lw=2, label=r'$\langle|\delta v|\rangle$')
        _ax_vel2.set_xlim(0, r_max_AU * 1.05)
        if global_ranges is not None:
            _ax_vel2.set_ylim(0, global_ranges['vel_ymax'] * 1.05)
        _ax_base(_ax_vel2, '', r'Velocity (km/s)')
        _ax_vel2.tick_params(labelbottom=False)
        _leg3 = _ax_vel2.legend(fontsize=9, framealpha=0.8, facecolor='w')
        for _t in _leg3.get_texts(): _t.set_color('k')
        # Mach
        _vm = np.isfinite(mach_prof) & (mach_prof > 0)
        if _vm.any():
            _ax_mach.semilogy(bin_AU[_vm], mach_prof[_vm], 'k-o', ms=5, lw=2)
        _ax_mach.axhline(1.0, color='r', lw=1.5, ls='--', label='Ma = 1')
        _ax_mach.set_xlim(0, r_max_AU * 1.05)
        if global_ranges is not None:
            _lo, _hi = global_ranges['mach_ylim']
            _ax_mach.set_ylim(max(_lo * 0.5, 1e-3), _hi * 2.0)
        _ax_base(_ax_mach, 'r (AU)', r'$\mathcal{M}$')
        _leg4 = _ax_mach.legend(fontsize=9, framealpha=0.8, facecolor='w')
        for _t in _leg4.get_texts(): _t.set_color('k')
        _save_dual(fig_i, _ifpath('profiles_lin'), _if_dark_path('profiles_lin'))

        # ── 5c. Velocity profiles: cs, v_phi, v_r, |dv| (disk-scale) ─────────
        fig_i, _ax_vp = plt.subplots(1, 1, figsize=(6, 4))
        fig_i.patch.set_facecolor('w')
        _ax_vp.plot(bin_AU[_vb], cs_prof[_vb],     'r-',    lw=2, label=r'$c_s$')
        _ax_vp.plot(bin_AU[_vb], vphi_prof[_vb],   'orange', lw=2, label=r'$v_\phi$')
        _ax_vp.plot(bin_AU[_vb], np.abs(vr_prof[_vb]), 'c-', lw=2, label=r'$|v_r|$')
        _ax_vp.plot(bin_AU[_vb], vturb_prof[_vb],  'olive', lw=1.5, label=r'$|\delta v|$')
        _ax_vp.set_xlim(0, r_max_AU * 1.05)
        _ax_vp.set_ylim(bottom=0)
        _ax_base(_ax_vp, 'r (AU)', r'Velocity (km/s)')
        _lvp = _ax_vp.legend(fontsize=9, framealpha=0.8, facecolor='w')
        for _t in _lvp.get_texts(): _t.set_color('k')
        fig_i.tight_layout()
        _save_dual(fig_i, _ifpath('velprof'), _if_dark_path('velprof'))

        # ── 5d. Wide velocity profiles (log x, full radial range) ─────────────
        # Compute wide profiles with 50 log-spaced bins
        _r_outer_wide = np.percentile(r_xy_AU, 95) if len(r_xy_AU) > 0 else r_max_AU * 5
        _r_outer_wide = max(_r_outer_wide, r_max_AU * 2)
        _nbw = 50
        _bins_w = np.logspace(np.log10(max(r_xy_AU.min(), 0.1) if len(r_xy_AU) > 0 else 0.1),
                              np.log10(_r_outer_wide), _nbw + 1)
        _bc_w = np.sqrt(_bins_w[:-1] * _bins_w[1:])  # geometric mean
        _cs_w = np.zeros(_nbw); _vp_w = np.zeros(_nbw)
        _vr_w = np.zeros(_nbw); _vt_w = np.zeros(_nbw)
        _bidx_w = np.clip(np.digitize(r_xy_AU, _bins_w) - 1, 0, _nbw - 1)
        for _b in range(_nbw):
            _mb = _bidx_w == _b
            if _mb.sum() < 2:
                continue
            _w = mass_small[_mb]; _ws = _w.sum()
            if _ws <= 0:
                continue
            _cs_w[_b] = np.dot(cs_small[_mb], _w) / _ws
            _vp_w[_b] = np.dot(v_phi[_mb], _w) / _ws
            _vr_w[_b] = np.dot(np.abs(v_r[_mb]), _w) / _ws
            _vt_w[_b] = np.dot(v_rest[_mb], _w) / _ws
        _vw = (_cs_w > 0) | (_vp_w > 0)
        fig_i, _ax_vpw = plt.subplots(1, 1, figsize=(7, 4))
        fig_i.patch.set_facecolor('w')
        if _vw.any():
            _ax_vpw.plot(_bc_w[_vw], _cs_w[_vw],         'r-',    lw=2, label=r'$c_s$')
            _ax_vpw.plot(_bc_w[_vw], np.abs(_vp_w[_vw]), 'orange', lw=2, label=r'$|v_\phi|$')
            _ax_vpw.plot(_bc_w[_vw], _vr_w[_vw],         'c-',    lw=2, label=r'$|v_r|$')
            _ax_vpw.plot(_bc_w[_vw], _vt_w[_vw],         'olive', lw=1.5, label=r'$|\delta v|$')
        _ax_vpw.set_xscale('log')
        _ax_vpw.axvline(r_max_AU, color='gray', lw=0.8, ls=':', alpha=0.6, label=r'$r_{\rm max}$')
        _ax_vpw.set_ylim(bottom=0)
        _ax_base(_ax_vpw, 'r (AU)', r'Velocity (km/s)')
        _lvpw = _ax_vpw.legend(fontsize=9, framealpha=0.8, facecolor='w', ncol=2)
        for _t in _lvpw.get_texts(): _t.set_color('k')
        fig_i.tight_layout()
        _save_dual(fig_i, _ifpath('velprof_wide'), _if_dark_path('velprof_wide'))

        # ── 6. Stability: virial + mass-to-flux ───────────────────────────────
        fig_i, _ax = plt.subplots(1, 2, figsize=(14, 7))
        fig_i.patch.set_facecolor('w')
        _vvir = np.isfinite(virial_prof) & (virial_prof > 0)
        if _vvir.any():
            _ax[0].semilogy(bin_AU[_vvir], virial_prof[_vvir], 'k-o', ms=5, lw=2)
        _ax[0].axhline(1.0, color='r',    lw=1.5, ls='--', label=r'$\alpha=1$')
        _ax[0].axhline(2.0, color='orange', lw=1, ls=':',  label=r'$\alpha=2$')
        _ax[0].set_xlim(0, r_max_AU * 1.05); _ax[0].set_ylim(1e-2, 1e2)
        _ax_base(_ax[0], 'r (AU)', r'$\alpha_{\rm vir}$')
        _leg6 = _ax[0].legend(fontsize=9, framealpha=0.8, facecolor='w')
        for _t in _leg6.get_texts(): _t.set_color('k')
        _vmf = np.isfinite(mf_prof) & (mf_prof > 0)
        if _vmf.any():
            _ax[1].semilogy(bin_AU[_vmf], mf_prof[_vmf], 'k-o', ms=5, lw=2)
            _ax[1].axhline(1.0, color='r', lw=1.5, ls='--', label=r'$\mu=1$')
            _leg7 = _ax[1].legend(fontsize=9, framealpha=0.8, facecolor='w')
            for _t in _leg7.get_texts(): _t.set_color('k')
        else:
            _ax[1].text(0.5, 0.5, 'B field not loaded', ha='center', va='center',
                        color='k', fontsize=12, transform=_ax[1].transAxes)
        _ax[1].set_xlim(0, r_max_AU * 1.05)
        _ax_base(_ax[1], 'r (AU)', r'$\mu = 2\pi\sqrt{G}\,\Sigma/|B_z|$')
        fig_i.tight_layout()
        _save_dual(fig_i, _ifpath('stability'), _if_dark_path('stability'))

        # ── 7. B field (only when available) ─────────────────────────────────
        if Bz_fo_map is not None and np.any(Bz_fo_map > 0):
            fig_i, _ax = plt.subplots(1, 2, figsize=(14, 7))
            fig_i.patch.set_facecolor('w')
            _Bz_v = Bz_fo_map > 0
            _Bz_vmin = float(np.percentile(Bz_fo_map[_Bz_v], 1))
            _Bz_vmax = float(np.percentile(Bz_fo_map[_Bz_v], 99))
            _Bz_n   = colors.LogNorm(vmin=max(_Bz_vmin, 1e-10), vmax=max(_Bz_vmax, 1e-9))
            im_bz = _ax[0].pcolormesh(X, Y, np.where(_Bz_v, Bz_fo_map, _Bz_n.vmin),
                                       norm=_Bz_n, cmap='plasma')
            _cb(fig_i, im_bz, _ax[0], r'$|B_z|$ (G)')
            _add_stars_sd(_ax[0], _sfo2, half_AU)
            _ax_base(_ax[0], 'x (AU)', 'y (AU)', half_AU)
            if _vmf.any():
                _ax[1].semilogy(bin_AU[_vmf], mf_prof[_vmf], 'k-o', ms=5, lw=2)
                _ax[1].axhline(1.0, color='r', lw=1.5, ls='--', label=r'$\mu=1$')
                _leg8 = _ax[1].legend(fontsize=9, framealpha=0.8, facecolor='w')
                for _t in _leg8.get_texts(): _t.set_color('k')
            _ax[1].set_xlim(0, r_max_AU * 1.05)
            _ax_base(_ax[1], 'r (AU)', r'$\mu(r)$')
            fig_i.tight_layout()
            _save_dual(fig_i, _ifpath('bfield'), _if_dark_path('bfield'))

        # ── 8. De-rotated density projection ─────────────────────────────────
        global _derot_prev_time_Myr, _derot_cumul_theta, _derot_bin_centers_kpc
        # Update cumulative rotation angle from v_phi profile
        if _derot_cumul_theta is None or _derot_bin_centers_kpc is None:
            _derot_cumul_theta = np.zeros(N_BINS)
            _derot_bin_centers_kpc = bin_centers_kpc.copy()
        if _derot_prev_time_Myr is not None and time_Myr > _derot_prev_time_Myr:
            dt_Myr = time_Myr - _derot_prev_time_Myr
            dt_s = dt_Myr * 1e6 * 3.156e7  # Myr → seconds
            # Omega = vphi / r  [km/s / kpc]
            _omega = np.where(bin_centers_kpc > 0,
                              vphi_prof / bin_centers_kpc,  # km/s/kpc
                              0.0)
            # Convert to rad/s: (km/s/kpc) * (1e5 cm/s per km/s) / (kpc cm) = rad/s
            _omega_rad_s = _omega * 1e5 / kpc
            _derot_cumul_theta += _omega_rad_s * dt_s  # rad
        _derot_prev_time_Myr = time_Myr

        # De-rotate face-on particle positions
        _r_part = np.sqrt(pos_fo[:, 0]**2 + pos_fo[:, 1]**2)
        _phi_part = np.arctan2(pos_fo[:, 1], pos_fo[:, 0])
        # Interpolate cumulative theta at each particle's radius
        _theta_shift = np.interp(_r_part, _derot_bin_centers_kpc, _derot_cumul_theta)
        _phi_derot = _phi_part - _theta_shift
        _pos_derot = np.column_stack([
            _r_part * np.cos(_phi_derot),
            _r_part * np.sin(_phi_derot),
            pos_fo[:, 2]
        ])
        # Render de-rotated face-on surface density
        sig_derot, _ = _surf(_pos_derot, mass_small, hsml_small, image_box_kpc)
        fig_i = plt.figure(figsize=(6, 5.5))
        fig_i.patch.set_facecolor('w')
        ax_dr = fig_i.add_subplot(111)
        _d_dr = np.where(sig_derot > 0, sig_derot, norm_small.vmin)
        im_dr = ax_dr.pcolormesh(X, Y, _d_dr, norm=norm_small, cmap=cmap)
        _cb(fig_i, im_dr, ax_dr, r'$\Sigma$ (M$_\odot$/pc$^2$)')
        _add_stars_sd(ax_dr, _sfo2, half_AU)
        _ax_base(ax_dr, 'x (AU)', 'y (AU)', half_AU)
        ax_dr.set_title('De-rotated face-on', color='k', fontsize=11)
        fig_i.tight_layout()
        _save_dual(fig_i, _ifpath('derotated'), _if_dark_path('derotated'))


# ═══════════════════════════════════════════════════════════════════════════════
# Q heatmap (time × radius)
# ═══════════════════════════════════════════════════════════════════════════════

def _add_formation_markers(ax, merge_data, r_AU_ref):
    """Add gold star markers for sink formation events using sink_data.npz.

    Uses form_times_kyr and form_r_AU from the cross-snapshot sink tracking
    (plot_sink_history.py), which uses most-massive-sink centering — consistent
    with the position history lines.
    """
    if merge_data is None:
        return
    ft = np.asarray(merge_data['form_times_kyr'])
    fr = np.asarray(merge_data['form_r_AU'])
    if len(ft) == 0:
        return
    r_max_hm = r_AU_ref[-1]
    r_min_hm = r_AU_ref[0]
    in_range = fr <= r_max_hm
    if in_range.any():
        plot_r = np.maximum(fr[in_range], r_min_hm)
        ax.scatter(ft[in_range], plot_r, marker='*', s=120,
                   color='gold', edgecolors='k', linewidths=0.5,
                   zorder=5, label='sink formation')


def _add_merge_markers(ax, merge_data, r_AU_ref, t1_Myr):
    """Add red × markers for sink merger events to a heatmap axis."""
    if merge_data is None:
        return
    mt  = np.asarray(merge_data['merge_times_kyr'])
    mr  = np.asarray(merge_data['merge_r_AU'])
    if len(mt) == 0:
        return
    r_max_hm = r_AU_ref[-1]
    r_min_hm = r_AU_ref[0]
    in_range = mr <= r_max_hm
    if in_range.any():
        plot_r = np.maximum(mr[in_range], r_min_hm)
        ax.scatter(mt[in_range], plot_r, marker='x', s=80,
                   color='red', linewidths=1.5, zorder=5, label='sink merger')


def _add_pos_lines(ax, pos_history):
    """Overlay per-sink r(t) trajectories as coloured lines on a heatmap axis.

    pos_history must contain keys:
      n_sink_series  : [n]  number of sinks saved
      sink_t_0, sink_r_0, sink_t_1, sink_r_1, …  : per-sink arrays
    """
    if pos_history is None:
        return
    n_key = pos_history.get('n_sink_series')
    if n_key is None:
        return
    n = int(np.asarray(n_key).flat[0])
    if n == 0:
        return
    cmap_sinks = plt.colormaps.get_cmap('tab20')
    for i in range(n):
        t_arr = pos_history.get(f'sink_t_{i}')
        r_arr = pos_history.get(f'sink_r_{i}')
        if t_arr is None or r_arr is None:
            continue
        t_arr = np.asarray(t_arr)
        r_arr = np.asarray(r_arr)
        if len(t_arr) == 0:
            continue
        ax.plot(t_arr, r_arr, color=cmap_sinks(i % 20),
                lw=0.9, alpha=0.75, zorder=4)


def _get_Q_cmap():
    """Return a RdYlGn colormap with set_under/set_over for out-of-range Q."""
    cmap = plt.colormaps.get_cmap('RdYlGn').copy()
    cmap.set_under(cmap(0.0))
    cmap.set_over(cmap(1.0))
    cmap.set_bad('white', alpha=0)
    return cmap


def make_Q_heatmap(outdir, heatmap_path=None, merge_data=None, pos_history=None):
    """
    Load all qprofile_*.npz files from outdir and produce a heatmap:
      x-axis: time [Myr],  y-axis: r [AU],  colorbar: Toomre Q (log).
    A Q=1 contour is drawn in black.
    Saves to outdir/Q_heatmap.png (or heatmap_path if provided).
    """
    profile_files = sorted(glob.glob(os.path.join(outdir, 'qprofiles', 'qprofile_*.npz')))
    if not profile_files:
        print('  make_Q_heatmap: no qprofile_*.npz files found in qprofiles/, skipping.')
        return

    times, Q_rows, n_sinks_list = [], [], []
    # Accumulate per-sink birth events: keyed by formation time (Myr) → r (AU)
    _seen_form_Myr = {}   # form_Myr → r_AU (keeps first appearance)
    for f in sorted(profile_files):   # sorted → chronological
        d = np.load(f)
        t = float(np.atleast_1d(d['time_Myr'])[0])
        Q = d['Q'].copy()
        r = d['r_AU'].copy()
        times.append(t)
        Q_rows.append((r, Q))
        # n_sinks may not exist in older npz files; default to 0
        n_sinks_list.append(int(d['n_sinks'][0]) if 'n_sinks' in d else 0)
        # Collect sink birth events (first snapshot each sink_form_Myr is seen)
        if 'sink_form_Myr' in d and 'sink_r_AU' in d:
            for tf, rf in zip(d['sink_form_Myr'], d['sink_r_AU']):
                tf_key = round(float(tf), 6)   # avoid float key collisions
                if tf_key not in _seen_form_Myr:
                    _seen_form_Myr[tf_key] = float(rf)

    # Use the r-grid from the snapshot with the widest radial range as the
    # interpolation reference.  Taking r_AU_ref from the first (earliest) file
    # fails when early snapshots have no disk gas and save near-zero radii —
    # all later Q values then extrapolate to 0 → NaN → blank heatmap.
    r_AU_ref = max(Q_rows, key=lambda x: float(x[0].max()) if len(x[0]) > 0 else 0.0)[0]

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
    Q_mat     = np.full((len(times_arr), len(r_AU_ref)), np.nan)
    for i, idx in enumerate(sort_idx):
        r_i, Q_i = Q_rows[idx]
        valid = np.isfinite(Q_i) & (Q_i > 0)
        if valid.sum() >= 2:
            Q_mat[i] = np.interp(r_AU_ref, r_i[valid], Q_i[valid])
        elif valid.sum() == 1:
            Q_mat[i] = Q_i[valid][0]

    # Forward-fill: if a time step has no data, copy previous step's values
    for i in range(1, Q_mat.shape[0]):
        if np.all(np.isnan(Q_mat[i])):
            Q_mat[i] = Q_mat[i - 1]

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('w')
    ax.set_facecolor('w')

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
    _Q_cmap = _get_Q_cmap()
    im = ax.pcolormesh(Tg, Rg, Q_mat,
                       norm=colors.LogNorm(vmin=0.1, vmax=10),
                       cmap=_Q_cmap, rasterized=True)
    cb = plt.colorbar(im, ax=ax, label='Toomre Q', extend='both')
    cb.ax.yaxis.set_tick_params(color='k')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='k')
    cb.set_label('Toomre Q', color='k')

    # Q=1 contour on cell-centre grids
    Tc, Rc = np.meshgrid(t_shifted, r_AU_ref, indexing='ij')
    Q_filled = np.where(np.isfinite(Q_mat), Q_mat, 1.0)
    try:
        ax.contour(Tc, Rc, Q_filled, levels=[1.0], colors='k', linewidths=1.5)
    except Exception:
        pass

    # Star-formation events from sink_data.npz (consistent centering with pos lines)
    _add_formation_markers(ax, merge_data, r_AU_ref)

    # Merger markers
    _add_merge_markers(ax, merge_data, r_AU_ref, t1_Myr)
    ax.legend(facecolor='white', edgecolor='k', labelcolor='k', fontsize=9)

    xlabel = (r'$t - t_1$ (kyr)   [$t_1$ = first sink formation]'
              if t1_Myr is not None else 'Time (kyr)')
    ax.set_xlabel(xlabel,    color='k', fontsize=12)
    ax.set_ylabel('r (AU)',  color='k', fontsize=12)
    ax.tick_params(colors='k', which='both', direction='in', right=True, top=True)
    for spine in ax.spines.values():
        spine.set_edgecolor('k')

    # Crop left edge only (exclude pre-disk stray snapshots); always show to end.
    n_finite = np.isfinite(Q_mat).sum(axis=1)
    peak_cov = n_finite.max()
    if peak_cov > 0:
        dense_rows = np.where(n_finite >= peak_cov * 0.25)[0]
        t_dense    = t_shifted[dense_rows]
        t_lo_lim   = np.percentile(t_dense, 5)
        t_hi_lim   = t_shifted.max()          # always extend to last snapshot
        span       = max(t_hi_lim - t_lo_lim, 1.0)
        ax.set_xlim([t_lo_lim - span * 0.05, t_hi_lim + span * 0.02])

    plt.tight_layout()
    if heatmap_path is None:
        heatmap_path = os.path.join(outdir, 'light', 'Q_heatmap.png')
    _save_fig_dual(fig, heatmap_path)
    print(f'  Q heatmap saved → {heatmap_path}')

    # Position-history overlay version
    if pos_history is not None:
        fig2, ax2 = plt.subplots(figsize=(16, 9))
        fig2.patch.set_facecolor('w')
        ax2.set_facecolor('w')
        ax2.pcolormesh(Tg, Rg, Q_mat, norm=colors.LogNorm(vmin=0.1, vmax=10),
                       cmap=_Q_cmap, rasterized=True)
        _add_pos_lines(ax2, pos_history)
        try:
            ax2.contour(Tc, Rc, Q_filled, levels=[1.0], colors='k', linewidths=1.5)
        except Exception:
            pass
        _add_formation_markers(ax2, merge_data, r_AU_ref)
        _add_merge_markers(ax2, merge_data, r_AU_ref, t1_Myr)
        ax2.legend(facecolor='white', edgecolor='k', labelcolor='k', fontsize=9)
        ax2.set_xlabel(xlabel, color='k', fontsize=12)
        ax2.set_ylabel('r (AU)', color='k', fontsize=12)
        ax2.tick_params(colors='k', which='both', direction='in', right=True, top=True)
        for spine in ax2.spines.values(): spine.set_edgecolor('k')
        if peak_cov > 0:
            ax2.set_xlim([t_lo_lim - span * 0.05, t_hi_lim + span * 0.02])
        pos_path = heatmap_path.replace('.png', '_pos.png')
        plt.tight_layout()
        _save_fig_dual(fig2, pos_path)
        print(f'  Q heatmap (pos overlay) saved → {pos_path}')


def make_sigma_heatmap(outdir, heatmap_path=None, merge_data=None, pos_history=None):
    """
    Load all qprofile_*.npz files and produce a surface-density heatmap:
      x-axis: time [kyr relative to first sink],  y-axis: r [AU],
      colorbar: Σ (g/cm²) — log scale.
    Saves to outdir/sigma_heatmap.png (or heatmap_path if provided).
    """
    profile_files = sorted(glob.glob(os.path.join(outdir, 'qprofiles', 'qprofile_*.npz')))
    if not profile_files:
        print('  make_sigma_heatmap: no qprofile_*.npz files found in qprofiles/, skipping.')
        return

    times, Sigma_rows, n_sinks_list = [], [], []
    _seen_form_Myr = {}
    for f in sorted(profile_files):
        d = np.load(f)
        t     = float(np.atleast_1d(d['time_Myr'])[0])
        Sigma = d['Sigma'].copy()  # g/cm² per annulus
        r     = d['r_AU'].copy()
        times.append(t)
        Sigma_rows.append((r, Sigma))
        n_sinks_list.append(int(d['n_sinks'][0]) if 'n_sinks' in d else 0)
        if 'sink_form_Myr' in d and 'sink_r_AU' in d:
            for tf, rf in zip(d['sink_form_Myr'], d['sink_r_AU']):
                tf_key = round(float(tf), 6)
                if tf_key not in _seen_form_Myr:
                    _seen_form_Myr[tf_key] = float(rf)

    # Reference r-grid: widest radial range
    r_AU_ref = max(Sigma_rows, key=lambda x: float(x[0].max()) if len(x[0]) > 0 else 0.0)[0]

    sort_idx    = np.argsort(times)
    times_arr   = np.array(times)[sort_idx]
    n_sinks_arr = np.array(n_sinks_list)[sort_idx]
    t1_Myr = float(min(_seen_form_Myr.keys())) if _seen_form_Myr else None
    if t1_Myr is None:
        sink_snaps = np.where(n_sinks_arr > 0)[0]
        t1_Myr = float(times_arr[sink_snaps[0]]) if len(sink_snaps) > 0 else None

    Sigma_mat = np.zeros((len(times_arr), len(r_AU_ref)))
    for i, idx in enumerate(sort_idx):
        r_i, S_i = Sigma_rows[idx]
        S_clean   = np.where(np.isfinite(S_i) & (S_i > 0), S_i, 0.0)
        Sigma_mat[i] = np.interp(r_AU_ref, r_i, S_clean, left=0.0, right=0.0)
    Sigma_mat = np.where(Sigma_mat > 0, Sigma_mat, np.nan)

    # Forward-fill: if a time step has no data, copy previous step's values
    for i in range(1, Sigma_mat.shape[0]):
        if np.all(np.isnan(Sigma_mat[i])):
            Sigma_mat[i] = Sigma_mat[i - 1]

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('w')
    ax.set_facecolor('w')

    t_shifted = (times_arr - t1_Myr if t1_Myr is not None else times_arr) * 1e3

    # Mask outlier columns: snapshots where the disk wasn't found show as
    # anomalously low Σ (dark blue streaks).  Flag any column whose median is
    # more than 2 decades below the running median of its ±10 neighbours.
    col_med = np.nanmedian(Sigma_mat, axis=1)          # (n_times,)
    win = 10
    running_med = np.full_like(col_med, np.nan)
    for i in range(len(col_med)):
        lo, hi = max(0, i - win), min(len(col_med), i + win + 1)
        neighbours = col_med[lo:hi]
        running_med[i] = np.nanmedian(neighbours[neighbours > 0]) if np.any(neighbours > 0) else np.nan
    bad_cols = np.where(
        np.isfinite(col_med) & np.isfinite(running_med) &
        (col_med < running_med / 1e2)
    )[0]
    if len(bad_cols):
        print(f'  make_sigma_heatmap: masking {len(bad_cols)} outlier column(s) at '
              f't = {t_shifted[bad_cols]} kyr')
        Sigma_mat[bad_cols, :] = np.nan

    dt   = np.diff(t_shifted)
    t_lo = np.concatenate([[t_shifted[0] - dt[0]/2],  t_shifted[:-1] + dt/2])
    t_hi = np.concatenate([t_shifted[:-1] + dt/2,    [t_shifted[-1] + dt[-1]/2]])
    T    = np.concatenate([t_lo, [t_hi[-1]]])

    dr   = np.diff(r_AU_ref)
    r_lo = np.concatenate([[r_AU_ref[0] - dr[0]/2], r_AU_ref[:-1] + dr/2])
    r_hi = np.concatenate([r_AU_ref[:-1] + dr/2,   [r_AU_ref[-1] + dr[-1]/2]])
    R    = np.concatenate([r_lo, [r_hi[-1]]])

    # Σ in g/cm² — bulk range ~5 to 1e3 g/cm² (near-sink outliers excluded)
    Tg, Rg = np.meshgrid(T, R, indexing='ij')
    im = ax.pcolormesh(Tg, Rg, Sigma_mat,
                       norm=colors.LogNorm(vmin=5, vmax=1e3),
                       cmap='plasma', rasterized=True)
    cb = plt.colorbar(im, ax=ax)
    cb.ax.yaxis.set_tick_params(color='k')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='k')
    cb.set_label(r'$\Sigma$ (g/cm$^2$)', color='k')

    # Sink formation markers (from sink_data.npz — consistent centering)
    _add_formation_markers(ax, merge_data, r_AU_ref)

    _add_merge_markers(ax, merge_data, r_AU_ref, t1_Myr)
    ax.legend(facecolor='white', edgecolor='k', labelcolor='k', fontsize=9)

    xlabel = (r'$t - t_1$ (kyr)   [$t_1$ = first sink formation]'
              if t1_Myr is not None else 'Time (kyr)')
    ax.set_xlabel(xlabel,       color='k', fontsize=12)
    ax.set_ylabel('r (AU)',     color='k', fontsize=12)
    ax.tick_params(colors='k', which='both', direction='in', right=True, top=True)
    for spine in ax.spines.values():
        spine.set_edgecolor('k')

    # Crop left edge only; always show to last snapshot.
    n_finite = np.isfinite(Sigma_mat).sum(axis=1)
    peak_cov = n_finite.max()
    if peak_cov > 0:
        dense_rows = np.where(n_finite >= peak_cov * 0.25)[0]
        t_dense    = t_shifted[dense_rows]
        t_lo_lim   = np.percentile(t_dense, 5)
        t_hi_lim   = t_shifted.max()
        span       = max(t_hi_lim - t_lo_lim, 1.0)
        ax.set_xlim([t_lo_lim - span * 0.05, t_hi_lim + span * 0.02])

    plt.tight_layout()
    if heatmap_path is None:
        heatmap_path = os.path.join(outdir, 'light', 'sigma_heatmap.png')
    _save_fig_dual(fig, heatmap_path)
    print(f'  Sigma heatmap saved → {heatmap_path}')

    # Position-history overlay version
    if pos_history is not None:
        fig2, ax2 = plt.subplots(figsize=(16, 9))
        fig2.patch.set_facecolor('w')
        ax2.set_facecolor('w')
        ax2.pcolormesh(Tg, Rg, Sigma_mat, norm=colors.LogNorm(vmin=5, vmax=1e3),
                       cmap='plasma', rasterized=True)
        _add_pos_lines(ax2, pos_history)
        _add_formation_markers(ax2, merge_data, r_AU_ref)
        _add_merge_markers(ax2, merge_data, r_AU_ref, t1_Myr)
        ax2.legend(facecolor='white', edgecolor='k', labelcolor='k', fontsize=9)
        ax2.set_xlabel(xlabel, color='k', fontsize=12)
        ax2.set_ylabel('r (AU)', color='k', fontsize=12)
        ax2.tick_params(colors='k', which='both', direction='in', right=True, top=True)
        for spine in ax2.spines.values(): spine.set_edgecolor('k')
        if peak_cov > 0:
            ax2.set_xlim([t_lo_lim - span * 0.05, t_hi_lim + span * 0.02])
        pos_path = heatmap_path.replace('.png', '_pos.png')
        plt.tight_layout()
        _save_fig_dual(fig2, pos_path)
        print(f'  Sigma heatmap (pos overlay) saved → {pos_path}')


# ═══════════════════════════════════════════════════════════════════════════════
# Sigma_r heatmap (time × radius)
# ═══════════════════════════════════════════════════════════════════════════════

def make_sigma_r_heatmap(outdir, heatmap_path=None, merge_data=None, pos_history=None):
    """
    Load all qprofile_*.npz files and produce a velocity-dispersion heatmap:
      x-axis: time [kyr relative to first sink],  y-axis: r [AU],
      colorbar: sigma_r (km/s) — log scale.
    Saves to outdir/light/sigma_r_heatmap.png (or heatmap_path if provided).
    """
    profile_files = sorted(glob.glob(os.path.join(outdir, 'qprofiles', 'qprofile_*.npz')))
    if not profile_files:
        print('  make_sigma_r_heatmap: no qprofile_*.npz files found in qprofiles/, skipping.')
        return

    times, sigma_r_rows, n_sinks_list = [], [], []
    _seen_form_Myr = {}
    for f in sorted(profile_files):
        d = np.load(f)
        t       = float(np.atleast_1d(d['time_Myr'])[0])
        sigma_r = d['sigma_r'].copy()  # km/s per annulus
        r       = d['r_AU'].copy()
        times.append(t)
        sigma_r_rows.append((r, sigma_r))
        n_sinks_list.append(int(d['n_sinks'][0]) if 'n_sinks' in d else 0)
        if 'sink_form_Myr' in d and 'sink_r_AU' in d:
            for tf, rf in zip(d['sink_form_Myr'], d['sink_r_AU']):
                tf_key = round(float(tf), 6)
                if tf_key not in _seen_form_Myr:
                    _seen_form_Myr[tf_key] = float(rf)

    # Reference r-grid: widest radial range
    r_AU_ref = max(sigma_r_rows, key=lambda x: float(x[0].max()) if len(x[0]) > 0 else 0.0)[0]

    sort_idx    = np.argsort(times)
    times_arr   = np.array(times)[sort_idx]
    n_sinks_arr = np.array(n_sinks_list)[sort_idx]
    t1_Myr = float(min(_seen_form_Myr.keys())) if _seen_form_Myr else None
    if t1_Myr is None:
        sink_snaps = np.where(n_sinks_arr > 0)[0]
        t1_Myr = float(times_arr[sink_snaps[0]]) if len(sink_snaps) > 0 else None

    sigma_r_mat = np.zeros((len(times_arr), len(r_AU_ref)))
    for i, idx in enumerate(sort_idx):
        r_i, S_i = sigma_r_rows[idx]
        S_clean   = np.where(np.isfinite(S_i) & (S_i > 0), S_i, 0.0)
        sigma_r_mat[i] = np.interp(r_AU_ref, r_i, S_clean, left=0.0, right=0.0)
    sigma_r_mat = np.where(sigma_r_mat > 0, sigma_r_mat, np.nan)

    # Forward-fill: if a time step has no data, copy previous step's values
    for i in range(1, sigma_r_mat.shape[0]):
        if np.all(np.isnan(sigma_r_mat[i])):
            sigma_r_mat[i] = sigma_r_mat[i - 1]

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('w')
    ax.set_facecolor('w')

    t_shifted = (times_arr - t1_Myr if t1_Myr is not None else times_arr) * 1e3

    dt   = np.diff(t_shifted)
    t_lo = np.concatenate([[t_shifted[0] - dt[0]/2],  t_shifted[:-1] + dt/2])
    t_hi = np.concatenate([t_shifted[:-1] + dt/2,    [t_shifted[-1] + dt[-1]/2]])
    T    = np.concatenate([t_lo, [t_hi[-1]]])

    dr   = np.diff(r_AU_ref)
    r_lo = np.concatenate([[r_AU_ref[0] - dr[0]/2], r_AU_ref[:-1] + dr/2])
    r_hi = np.concatenate([r_AU_ref[:-1] + dr/2,   [r_AU_ref[-1] + dr[-1]/2]])
    R    = np.concatenate([r_lo, [r_hi[-1]]])

    Tg, Rg = np.meshgrid(T, R, indexing='ij')
    im = ax.pcolormesh(Tg, Rg, sigma_r_mat,
                       norm=colors.LogNorm(vmin=0.1, vmax=30),
                       cmap='plasma', rasterized=True)
    cb = plt.colorbar(im, ax=ax)
    cb.ax.yaxis.set_tick_params(color='k')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='k')
    cb.set_label(r'$\sigma_r$ (km/s)', color='k')

    # Sink formation markers (from sink_data.npz — consistent centering)
    _add_formation_markers(ax, merge_data, r_AU_ref)

    _add_merge_markers(ax, merge_data, r_AU_ref, t1_Myr)
    ax.legend(facecolor='white', edgecolor='k', labelcolor='k', fontsize=9)

    xlabel = (r'$t - t_1$ (kyr)   [$t_1$ = first sink formation]'
              if t1_Myr is not None else 'Time (kyr)')
    ax.set_xlabel(xlabel,       color='k', fontsize=12)
    ax.set_ylabel('r (AU)',     color='k', fontsize=12)
    ax.tick_params(colors='k', which='both', direction='in', right=True, top=True)
    for spine in ax.spines.values():
        spine.set_edgecolor('k')

    # Crop left edge only; always show to last snapshot.
    n_finite = np.isfinite(sigma_r_mat).sum(axis=1)
    peak_cov = n_finite.max()
    if peak_cov > 0:
        dense_rows = np.where(n_finite >= peak_cov * 0.25)[0]
        t_dense    = t_shifted[dense_rows]
        t_lo_lim   = np.percentile(t_dense, 5)
        t_hi_lim   = t_shifted.max()
        span       = max(t_hi_lim - t_lo_lim, 1.0)
        ax.set_xlim([t_lo_lim - span * 0.05, t_hi_lim + span * 0.02])

    plt.tight_layout()
    if heatmap_path is None:
        heatmap_path = os.path.join(outdir, 'light', 'sigma_r_heatmap.png')
    _save_fig_dual(fig, heatmap_path)
    print(f'  Sigma_r heatmap saved → {heatmap_path}')

    # Position-history overlay version
    if pos_history is not None:
        fig2, ax2 = plt.subplots(figsize=(16, 9))
        fig2.patch.set_facecolor('w')
        ax2.set_facecolor('w')
        ax2.pcolormesh(Tg, Rg, sigma_r_mat, norm=colors.LogNorm(vmin=0.1, vmax=30),
                       cmap='plasma', rasterized=True)
        _add_pos_lines(ax2, pos_history)
        _add_formation_markers(ax2, merge_data, r_AU_ref)
        _add_merge_markers(ax2, merge_data, r_AU_ref, t1_Myr)
        ax2.legend(facecolor='white', edgecolor='k', labelcolor='k', fontsize=9)
        ax2.set_xlabel(xlabel, color='k', fontsize=12)
        ax2.set_ylabel('r (AU)', color='k', fontsize=12)
        ax2.tick_params(colors='k', which='both', direction='in', right=True, top=True)
        for spine in ax2.spines.values(): spine.set_edgecolor('k')
        if peak_cov > 0:
            ax2.set_xlim([t_lo_lim - span * 0.05, t_hi_lim + span * 0.02])
        pos_path = heatmap_path.replace('.png', '_pos.png')
        plt.tight_layout()
        _save_fig_dual(fig2, pos_path)
        print(f'  Sigma_r heatmap (pos overlay) saved → {pos_path}')


# ═══════════════════════════════════════════════════════════════════════════════
# Q combo frames (2-pass: Q projection + profile + heatmap with time marker)
# ═══════════════════════════════════════════════════════════════════════════════

def make_Q_combo_frames(outdir, merge_data=None):
    """
    For each snapshot, produce a 3-panel figure:
      [0,0]  Face-on Toomre Q projection (from saved Q_fo_2d in npz)
      [0,1]  Q(r) radial profile
      [1,:]  Q heatmap (time × radius) with red vertical line at current time.
    Saves to outdir/light/Q_combo/frame_Q_combo_XXXX.png + dark version.
    """
    profile_files = sorted(glob.glob(os.path.join(outdir, 'qprofiles', 'qprofile_*.npz')))
    if not profile_files:
        print('  make_Q_combo_frames: no qprofile files found, skipping.')
        return

    # ── Load all data ────────────────────────────────────────────────────────
    all_data = []
    _seen_form_Myr = {}
    for f in sorted(profile_files):
        d = np.load(f, allow_pickle=True)
        all_data.append(d)
        if 'sink_form_Myr' in d and 'sink_r_AU' in d:
            for tf, rf in zip(d['sink_form_Myr'], d['sink_r_AU']):
                tf_key = round(float(tf), 6)
                if tf_key not in _seen_form_Myr:
                    _seen_form_Myr[tf_key] = float(rf)

    times_arr = np.array([float(np.atleast_1d(d['time_Myr'])[0]) for d in all_data])
    sort_idx = np.argsort(times_arr)
    times_arr = times_arr[sort_idx]
    all_data = [all_data[i] for i in sort_idx]

    # Reference r grid (widest range)
    r_AU_ref = max(all_data, key=lambda d: float(d['r_AU'].max()) if len(d['r_AU']) > 0 else 0.0)['r_AU']

    # t1 = first sink formation
    t1_Myr = float(min(_seen_form_Myr.keys())) if _seen_form_Myr else None
    if t1_Myr is None:
        n_sinks_arr = np.array([int(d['n_sinks'][0]) if 'n_sinks' in d else 0 for d in all_data])
        sink_snaps = np.where(n_sinks_arr > 0)[0]
        t1_Myr = float(times_arr[sink_snaps[0]]) if len(sink_snaps) > 0 else None

    t_shifted = (times_arr - t1_Myr if t1_Myr is not None else times_arr) * 1e3  # kyr

    # Build Q heatmap matrix
    Q_mat = np.full((len(times_arr), len(r_AU_ref)), np.nan)
    for i, d in enumerate(all_data):
        Q_i = d['Q']
        r_i = d['r_AU']
        valid = np.isfinite(Q_i) & (Q_i > 0)
        if valid.sum() >= 2:
            Q_mat[i] = np.interp(r_AU_ref, r_i[valid], Q_i[valid])
        elif valid.sum() == 1:
            Q_mat[i] = Q_i[valid][0]
    for i in range(1, Q_mat.shape[0]):
        if np.all(np.isnan(Q_mat[i])):
            Q_mat[i] = Q_mat[i - 1]

    # Heatmap grid edges
    dt = np.diff(t_shifted)
    if len(dt) == 0:
        print('  make_Q_combo_frames: only 1 snapshot, skipping.')
        return
    t_lo = np.concatenate([[t_shifted[0] - dt[0]/2], t_shifted[:-1] + dt/2])
    t_hi = np.concatenate([t_shifted[:-1] + dt/2, [t_shifted[-1] + dt[-1]/2]])
    T_edges = np.concatenate([t_lo, [t_hi[-1]]])
    dr = np.diff(r_AU_ref)
    r_lo = np.concatenate([[r_AU_ref[0] - dr[0]/2], r_AU_ref[:-1] + dr/2])
    r_hi = np.concatenate([r_AU_ref[:-1] + dr/2, [r_AU_ref[-1] + dr[-1]/2]])
    R_edges = np.concatenate([r_lo, [r_hi[-1]]])
    Tg, Rg = np.meshgrid(T_edges, R_edges, indexing='ij')

    # Dense time window for xlim
    n_finite = np.isfinite(Q_mat).sum(axis=1)
    peak_cov = n_finite.max()
    t_xlim = None
    if peak_cov > 0:
        dense_rows = np.where(n_finite >= peak_cov * 0.25)[0]
        t_dense = t_shifted[dense_rows]
        t_lo_lim = np.percentile(t_dense, 5)
        t_hi_lim = np.percentile(t_dense, 95)
        span = max(t_hi_lim - t_lo_lim, 1.0)
        t_xlim = [t_lo_lim - span * 0.1, t_hi_lim + span * 0.1]

    # Star birth events for heatmap
    birth_t = birth_r = None
    if _seen_form_Myr and t1_Myr is not None:
        birth_t = (np.array(list(_seen_form_Myr.keys())) - t1_Myr) * 1e3
        birth_r = np.array(list(_seen_form_Myr.values()))

    out_light = os.path.join(outdir, 'light', 'Q_combo')
    out_dark = os.path.join(outdir, 'dark', 'Q_combo')
    os.makedirs(out_light, exist_ok=True)
    os.makedirs(out_dark, exist_ok=True)

    # ── Per-snapshot combo figure ────────────────────────────────────────────
    for i, d in enumerate(all_data):
        snap_num = int(d['snap_num'][0])
        t_now = t_shifted[i]

        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.3,
                              height_ratios=[1, 0.8])
        fig.patch.set_facecolor('w')

        # [0,0]: Face-on Q projection
        ax_qmap = fig.add_subplot(gs[0, 0])
        if 'Q_fo_2d' in d and 'X_grid' in d:
            Xg = d['X_grid']; Yg = d['Y_grid']
            Q_2d = d['Q_fo_2d']
            Q_plot = np.where(Q_2d > 0, Q_2d, np.nan)
            im_q = ax_qmap.pcolormesh(Xg, Yg, Q_plot,
                                       norm=colors.LogNorm(0.1, 10), cmap='RdYlGn')
            fig.colorbar(im_q, ax=ax_qmap, label='Toomre Q')
            try:
                ax_qmap.contour(Xg, Yg, np.where(np.isfinite(Q_2d), Q_2d, 1.0),
                                levels=[1.0], colors='k', linewidths=1.5)
            except Exception:
                pass
            half = float(Xg.max())
            ax_qmap.set_xlim(-half, half); ax_qmap.set_ylim(-half, half)
        ax_qmap.set_xlabel('x (AU)', color='k')
        ax_qmap.set_ylabel('y (AU)', color='k')
        ax_qmap.set_facecolor('w')
        ax_qmap.tick_params(colors='k', which='both', direction='in')
        for sp in ax_qmap.spines.values(): sp.set_edgecolor('k')

        # [0,1]: Q(r) profile
        ax_qr = fig.add_subplot(gs[0, 1])
        ax_qr.set_facecolor('w')
        vq = np.isfinite(d['Q']) & (d['Q'] > 0)
        if vq.any():
            ax_qr.semilogy(d['r_AU'][vq], d['Q'][vq], 'k-o', ms=4, lw=2)
        ax_qr.axhline(1.0, color='r', lw=1.5, ls='--', label='Q = 1')
        ax_qr.set_ylim(0.1, 100)
        ax_qr.set_xlabel('r (AU)', color='k')
        ax_qr.set_ylabel('Toomre Q', color='k')
        ax_qr.tick_params(colors='k', which='both', direction='in')
        for sp in ax_qr.spines.values(): sp.set_edgecolor('k')
        _lq = ax_qr.legend(fontsize=9, framealpha=0.8, facecolor='w')
        for _t in _lq.get_texts(): _t.set_color('k')

        # [1,:]: Q heatmap with red line
        ax_hm = fig.add_subplot(gs[1, :])
        ax_hm.set_facecolor('w')
        im_hm = ax_hm.pcolormesh(Tg, Rg, Q_mat,
                                   norm=colors.LogNorm(vmin=0.1, vmax=10),
                                   cmap=_get_Q_cmap(), rasterized=True)
        cb_hm = fig.colorbar(im_hm, ax=ax_hm, extend='both')
        cb_hm.set_label('Toomre Q', color='k')
        cb_hm.ax.yaxis.set_tick_params(color='k')
        plt.setp(cb_hm.ax.yaxis.get_ticklabels(), color='k')
        # Q=1 contour
        Tc, Rc = np.meshgrid(t_shifted, r_AU_ref, indexing='ij')
        Q_filled = np.where(np.isfinite(Q_mat), Q_mat, 1.0)
        try:
            ax_hm.contour(Tc, Rc, Q_filled, levels=[1.0], colors='k', linewidths=1.0)
        except Exception:
            pass
        # Sink markers (prefer sink_data.npz; fall back to qprofile data)
        if merge_data is not None:
            _add_formation_markers(ax_hm, merge_data, r_AU_ref)
        elif birth_t is not None:
            r_max_hm = r_AU_ref[-1]
            in_range = birth_r <= r_max_hm
            if in_range.any():
                ax_hm.scatter(birth_t[in_range],
                              np.maximum(birth_r[in_range], r_AU_ref[0]),
                              marker='*', s=80, color='gold', edgecolors='k',
                              linewidths=0.5, zorder=5)
        # Red line at current time
        ax_hm.axvline(t_now, color='red', lw=2, ls='-', zorder=6)
        xlabel = (r'$t - t_1$ (kyr)' if t1_Myr is not None else 'Time (kyr)')
        ax_hm.set_xlabel(xlabel, color='k', fontsize=12)
        ax_hm.set_ylabel('r (AU)', color='k', fontsize=12)
        ax_hm.tick_params(colors='k', which='both', direction='in')
        for sp in ax_hm.spines.values(): sp.set_edgecolor('k')
        if t_xlim is not None:
            ax_hm.set_xlim(t_xlim)

        lpath = os.path.join(out_light, f'frame_Q_combo_{snap_num:04d}.png')
        dpath = os.path.join(out_dark, f'frame_Q_combo_{snap_num:04d}.png')
        _save_fig_dual(fig, lpath)
        print(f'  Q combo frame saved → snap {snap_num:04d}')

    print(f'  Q combo: {len(all_data)} frames saved to {out_light}')


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
     include_phase_in_master, h2_field,
     global_ranges) = args_tuple

    frames_dir = os.path.join(outdir, 'light', 'master_frames')
    os.makedirs(frames_dir, exist_ok=True)
    outpath          = os.path.join(frames_dir, f'frame_sd_{snap_num:04d}.png')
    outpath_analysis = os.path.join(frames_dir, f'frame_vel_{snap_num:04d}.png')
    if os.path.exists(outpath) and os.path.exists(outpath_analysis):
        return snap_num, 'skipped (exists)', 0.0

    t0 = _time.perf_counter()

    gas_fields = ['Masses', 'Coordinates', 'SmoothingLength',
                  'Velocities', 'Density', 'ParticleIDs', 'InternalEnergy',
                  'MagneticField']
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
    if 'Masses' not in pdata:
        return snap_num, 'skipped (no gas Masses field)', 0.0
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
                     image_box_kpc     = image_box_kpc,
                     res               = res,
                     vmin              = vmin,
                     vmax              = vmax,
                     cmap              = cmap,
                     outpath           = outpath,
                     outpath_analysis  = outpath_analysis,
                     com_vel           = com_vel,
                     corotate          = corotate,
                     vmax_vel          = vmax_vel,
                     v_K               = v_K,
                     data_outdir       = outdir,
                     include_phase     = include_phase_in_master,
                     h2_field          = h2_field,
                     sink_form_Myr     = _sink_form_Myr,
                     sink_r_AU         = _sink_r_AU,
                     global_ranges     = global_ranges)
    except Exception as e:
        return snap_num, f'render error: {e}', 0.0

    return snap_num, 'ok', _time.perf_counter() - t0


def _compute_global_ranges(snap_items, args, reference_center, reference_search_radius):
    """
    Fast pre-scan: load gas fields for every snapshot, compute per-annulus profiles
    (streaming-subtracted, no Meshoid), and return global axis ranges for all row-3
    panels plus vmax_vel for the velocity colormap.
    Returns a dict with keys:
      vmax_vel  – float, km/s ceiling for |δv| colormap
      rho_ylim  – (ymin, ymax) g/cm³ for density profile [3,0]
      vel_ymax  – float, km/s upper limit for c_s/σ_r/|δv| panel [3,1]
      mach_ylim – (ymin, ymax) for Mach profile [3,2]
      sml_xlim  – (0, xmax) AU for SML histogram [3,3]
    """
    _gas_fields = ['Masses', 'Coordinates', 'SmoothingLength', 'Velocities',
                   'Density', 'InternalEnergy']
    N_BINS = 20
    _GAMMA = 5.0 / 3.0
    pcts        = []
    rho_vals    = []
    vel_max_list = []
    mach_vals   = []
    sml_max_list = []
    print('  Pre-scanning for global axis ranges...')
    for snap_path, snap_num in snap_items:
        try:
            hdr, pdata, stardata, fsd, _, _ = get_snap_data_hybrid(
                args.sim, args.path, snap_num,
                snapshot_suffix='', snapdir=False,
                refinement_tag=False, verbose=False,
                custom_gas_fields=_gas_fields)
            hdr, pdata, stardata, fsd = convert_units_to_physical(hdr, pdata, stardata, fsd)
        except Exception:
            continue
        if 'Masses' not in pdata or len(pdata['Masses']) < 10:
            continue
        try:
            _, com, L_hat, _, _, _, _, com_vel = identify_disk(
                pdata, stardata,
                r_search_kpc=args.r_search, r_max_kpc=args.r_max,
                rho_threshold_cgs=args.rho_thresh, aspect_ratio=args.aspect,
                f_kep=args.f_kep,
                reference_center=reference_center,
                reference_search_radius=reference_search_radius)
            rot = rotation_matrix_to_z(L_hat)
            half = args.image_box * 0.75
            dists = np.linalg.norm(pdata['Coordinates'] - com, axis=1)
            cut = dists < half
            if cut.sum() < 10:
                continue
            pos_s  = (pdata['Coordinates'][cut] - com) @ rot.T
            vel_s  = (pdata['Velocities'][cut] - com_vel) @ rot.T
            mass_s = pdata['Masses'][cut]
            rho_s  = pdata['Density'][cut].astype(np.float64) * 1e10 * Msun / kpc**3
            sml_s  = pdata['SmoothingLength'][cut] * kpc / AU
            r_xy   = np.linalg.norm(pos_s[:, :2], axis=1)
            safe_r = np.maximum(r_xy, 1e-30)
            e_r_x  = pos_s[:, 0] / safe_r;  e_r_y = pos_s[:, 1] / safe_r
            v_r    =  vel_s[:, 0] * e_r_x + vel_s[:, 1] * e_r_y
            v_phi  = -vel_s[:, 0] * e_r_y + vel_s[:, 1] * e_r_x
            v_z    =  vel_s[:, 2]
            r_out  = max(np.percentile(r_xy, 95), 1e-20)
            bins   = np.linspace(0.0, r_out, N_BINS + 1)
            bidx   = np.clip(np.digitize(r_xy, bins) - 1, 0, N_BINS - 1)
            vr_p   = np.zeros(N_BINS);  vph_p = np.zeros(N_BINS)
            rho_p  = np.zeros(N_BINS)
            sigma_r_p = np.zeros(N_BINS)
            vturb_p   = np.zeros(N_BINS)
            if 'InternalEnergy' in pdata:
                u_s  = pdata['InternalEnergy'][cut]
                cs_s = np.sqrt(_GAMMA * (_GAMMA - 1.0) * np.maximum(u_s, 0.0))
            else:
                cs_s = np.zeros(len(mass_s))
            cs_p = np.zeros(N_BINS)
            for b in range(N_BINS):
                mb = bidx == b
                if mb.sum() == 0:
                    continue
                w = mass_s[mb]; ws = w.sum()
                vr_p[b]  = np.dot(v_r[mb],  w) / ws
                vph_p[b] = np.dot(v_phi[mb], w) / ws
                rho_p[b] = np.dot(rho_s[mb], w) / ws
                cs_p[b]  = np.dot(cs_s[mb],  w) / ws
                vr2_mw   = np.dot(v_r[mb]**2, w) / ws
                sigma_r_p[b] = np.sqrt(max(vr2_mw - vr_p[b]**2, 0.0))
            v_rest = np.sqrt((v_r - vr_p[bidx])**2 + (v_phi - vph_p[bidx])**2 + v_z**2)
            for b in range(N_BINS):
                mb = bidx == b
                if mb.sum() == 0:
                    continue
                w = mass_s[mb]; ws = w.sum()
                vturb_p[b] = np.dot(v_rest[mb], w) / ws
            pcts.append(float(np.percentile(v_rest, 99)))
            valid_rho = rho_p > 0
            if valid_rho.any():
                rho_vals.extend(rho_p[valid_rho].tolist())
            all_vel = np.concatenate([cs_p, sigma_r_p, vturb_p])
            pos_vel = all_vel[np.isfinite(all_vel) & (all_vel > 0)]
            if pos_vel.size > 0:
                vel_max_list.append(float(np.max(pos_vel)))
            with np.errstate(divide='ignore', invalid='ignore'):
                mach_p = np.where(cs_p > 0, vturb_p / cs_p, np.nan)
            valid_m = np.isfinite(mach_p) & (mach_p > 0)
            if valid_m.any():
                mach_vals.extend(mach_p[valid_m].tolist())
            sml_max_list.append(float(np.percentile(sml_s, 99)))
        except Exception:
            continue
    vmax_vel = float(np.percentile(pcts, 99)) if pcts else 5.0
    rho_ylim = (
        float(np.percentile(rho_vals, 1)),
        float(np.percentile(rho_vals, 99)),
    ) if len(rho_vals) >= 2 else (1e-20, 1e-10)
    vel_ymax  = float(np.percentile(vel_max_list, 95)) if vel_max_list else 5.0
    mach_ylim = (
        float(np.percentile(mach_vals, 1)),
        float(np.percentile(mach_vals, 99)),
    ) if len(mach_vals) >= 2 else (0.1, 10.0)
    sml_xlim  = (0.0, float(np.percentile(sml_max_list, 95)) * 1.05) if sml_max_list else (0.0, 100.0)
    print(f'  vmax_vel={vmax_vel:.3f} km/s  vel_ymax={vel_ymax:.2f} km/s  '
          f'mach=[{mach_ylim[0]:.2f},{mach_ylim[1]:.2f}]  sml_xmax={sml_xlim[1]:.1f} AU')
    return dict(vmax_vel=vmax_vel, rho_ylim=rho_ylim, vel_ymax=vel_ymax,
                mach_ylim=mach_ylim, sml_xlim=sml_xlim)


def main(args):
    # Reset module-level de-rotation accumulators for this run
    global _derot_prev_time_Myr, _derot_cumul_theta, _derot_bin_centers_kpc
    _derot_prev_time_Myr = None
    _derot_cumul_theta = None
    _derot_bin_centers_kpc = None

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

    # Auto-compute global axis ranges via a fast pre-scan (no Meshoid gridding)
    global_ranges = _compute_global_ranges(
        snap_items, args, reference_center, reference_search_radius)
    if vmax_vel is None:
        vmax_vel = global_ranges['vmax_vel']
        print(f'  Auto vmax_vel = {vmax_vel:.3f} km/s  (global 99th pct of |δv|)')
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

    os.makedirs(os.path.join(args.outdir, 'light', 'master_frames'), exist_ok=True)

    task_args = [
        (p, n, args.outdir, args.image_box, args.res, args.vmin, args.vmax,
         args.path, args.sim,
         args.r_search, args.r_max, args.rho_thresh, args.aspect, args.f_kep,
         args.cmap, reference_center, reference_search_radius,
         corotate, vmax_vel,
         min_gas_particles, min_gas_snap,
         include_phase_in_master, h2_field,
         global_ranges)
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

    _frames_dir = os.path.join(args.outdir, 'light', 'master_frames')
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
