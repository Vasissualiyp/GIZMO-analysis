"""
disk_utils.py
-------------
Disk identification and analysis utilities for GIZMO/FIRE/STARFORGE simulations.

Methods adapted from DiskIdentification.ipynb (Shivan Khullar).

The disk is identified using a hybrid geometric + kinematic approach:
  1. Find disk center (sink CoM or densest gas particle)
  2. Compute rotation axis L_hat from angular momentum
  3. Project to cylindrical coordinates aligned with L_hat
  4. Apply geometric pre-selection (r_cyl, |z|/r_cyl, density)
  5. Apply kinematic filter (co-rotating, rotationally supported)
  6. Optionally extend to bounding cylinder

Unit conventions (GIZMO code units after standard conversion):
  - Coordinates: kpc
  - Velocities: km/s
  - Masses: 10^10 M_sun
  - Density: 10^10 M_sun / kpc^3
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Physical constants (CGS)
# ═══════════════════════════════════════════════════════════════════════════════

kpc  = 3.08567758e21   # cm
AU   = 1.495978707e13  # cm
Msun = 1.98841e33      # g
G    = 6.6743e-8       # cm^3 / g / s^2


# ═══════════════════════════════════════════════════════════════════════════════
# Centering functions
# ═══════════════════════════════════════════════════════════════════════════════

def find_center_from_stars(stardata):
    """
    Compute mass-weighted center of mass from sink/star particles.

    Parameters
    ----------
    stardata : dict
        Dictionary with 'Coordinates' and 'Masses' arrays.

    Returns
    -------
    com : ndarray, shape (3,)
        Center of mass position [kpc].
    """
    if not stardata or len(stardata.get('Masses', [])) == 0:
        return None
    com = (np.sum(stardata['Coordinates'] * stardata['Masses'][:, None], axis=0)
           / np.sum(stardata['Masses']))
    return com


def find_center_from_density(pdata):
    """
    Find center at location of densest gas particle.

    Parameters
    ----------
    pdata : dict
        Gas particle data with 'Coordinates' and 'Density'.

    Returns
    -------
    center : ndarray, shape (3,)
        Position of densest particle [kpc].
    """
    idx = np.argmax(pdata['Density'])
    return pdata['Coordinates'][idx].copy()


def find_center(pdata, stardata):
    """
    Find disk center: mass-weighted CoM of sinks if any exist,
    otherwise the densest gas particle.

    Parameters
    ----------
    pdata : dict
        Gas particle data.
    stardata : dict or None
        Sink/star particle data.

    Returns
    -------
    com : ndarray, shape (3,)
        Center position [kpc].
    """
    com = find_center_from_stars(stardata)
    if com is not None:
        return com
    return find_center_from_density(pdata)


# ═══════════════════════════════════════════════════════════════════════════════
# Rotation axis (angular momentum)
# ═══════════════════════════════════════════════════════════════════════════════

def get_disk_axis(gas_pos_kpc, gas_vel_kms, gas_masses_code, r_search_kpc):
    """
    Compute disk rotation axis L_hat from gas angular momentum within r_search.

    Parameters
    ----------
    gas_pos_kpc : ndarray, shape (N, 3)
        Gas positions relative to center [kpc].
    gas_vel_kms : ndarray, shape (N, 3)
        Gas velocities in COM frame [km/s].
    gas_masses_code : ndarray, shape (N,)
        Gas masses [10^10 M_sun].
    r_search_kpc : float
        Radius within which to compute L_hat [kpc].

    Returns
    -------
    L_hat : ndarray, shape (3,)
        Normalized angular momentum vector.
    """
    dists = np.linalg.norm(gas_pos_kpc, axis=1)
    mask = dists < r_search_kpc

    if mask.sum() < 4:
        print("Warning: fewer than 4 particles in r_search, defaulting L_hat = z")
        return np.array([0., 0., 1.])

    # Convert to CGS for angular momentum calculation
    pos_cm = gas_pos_kpc[mask] * kpc              # kpc -> cm
    vel_cms = gas_vel_kms[mask] * 1e5             # km/s -> cm/s
    m_g = gas_masses_code[mask] * 1e10 * Msun     # code -> g

    L = np.sum(m_g[:, None] * np.cross(pos_cm, vel_cms), axis=0)
    L_mag = np.linalg.norm(L)

    if L_mag > 0:
        return L / L_mag
    else:
        print("Warning: L = 0, defaulting L_hat = z")
        return np.array([0., 0., 1.])


# ═══════════════════════════════════════════════════════════════════════════════
# Coordinate transformations
# ═══════════════════════════════════════════════════════════════════════════════

def cylindrical_coords(pos_kpc, vel_kms, L_hat):
    """
    Project positions and velocities into cylindrical coordinates aligned with L_hat.

    Parameters
    ----------
    pos_kpc : ndarray, shape (N, 3)
        Positions relative to center [kpc].
    vel_kms : ndarray, shape (N, 3)
        Velocities [km/s].
    L_hat : ndarray, shape (3,)
        Disk rotation axis (unit vector).

    Returns
    -------
    r_cyl_kpc : ndarray, shape (N,)
        Cylindrical radius (distance from rotation axis) [kpc].
    z_kpc : ndarray, shape (N,)
        Height along L_hat (signed) [kpc].
    v_phi_kms : ndarray, shape (N,)
        Azimuthal velocity, positive = co-rotating with L_hat [km/s].
    v_r_kms : ndarray, shape (N,)
        Radial velocity in disk plane, positive = outward [km/s].
    v_z_kms : ndarray, shape (N,)
        Velocity along L_hat [km/s].
    """
    z_kpc = pos_kpc @ L_hat                          # scalar projection
    v_z_kms = vel_kms @ L_hat

    r_perp = pos_kpc - z_kpc[:, None] * L_hat        # vector in disk plane
    r_cyl_kpc = np.linalg.norm(r_perp, axis=1)

    safe_r = np.maximum(r_cyl_kpc, 1e-30)
    e_r = r_perp / safe_r[:, None]                   # radial unit vector
    e_phi = np.cross(L_hat, e_r)                     # azimuthal unit vector (RHR)

    v_r_kms = np.einsum('ij,ij->i', vel_kms, e_r)
    v_phi_kms = np.einsum('ij,ij->i', vel_kms, e_phi)

    return r_cyl_kpc, z_kpc, v_phi_kms, v_r_kms, v_z_kms


def rotation_matrix_to_z(L_hat):
    """
    Build rotation matrix R such that R @ L_hat = [0, 0, 1].

    This aligns the disk plane with the x-y plane for face-on viewing.

    Parameters
    ----------
    L_hat : ndarray, shape (3,)
        Disk rotation axis (unit vector).

    Returns
    -------
    R : ndarray, shape (3, 3)
        Rotation matrix.
    """
    z_hat = np.array([0., 0., 1.])
    v = np.cross(L_hat, z_hat)
    s = np.linalg.norm(v)
    c = np.dot(L_hat, z_hat)

    if s > 1e-10:
        # Rodrigues' rotation formula
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 - c) / s**2
    else:
        # L_hat is already parallel to z
        R = np.eye(3) if c > 0 else -np.eye(3)

    return R


# ═══════════════════════════════════════════════════════════════════════════════
# Enclosed mass and Keplerian velocity
# ═══════════════════════════════════════════════════════════════════════════════

def compute_M_enc(r_cyl_kpc, gas_masses_Msun, M_stars_Msun):
    """
    Compute enclosed mass interior to each particle's cylindrical radius.

    M_enc[i] = M_stars + sum(gas_mass[j] for j where r_cyl[j] < r_cyl[i]).
    Vectorized via argsort + cumsum; exclusive of particle i itself.

    Parameters
    ----------
    r_cyl_kpc : ndarray, shape (N,)
        Cylindrical radii [kpc].
    gas_masses_Msun : ndarray, shape (N,)
        Gas masses [M_sun].
    M_stars_Msun : float
        Total stellar/sink mass [M_sun].

    Returns
    -------
    M_enc : ndarray, shape (N,)
        Enclosed mass at each particle's radius [g].
    """
    sort_idx = np.argsort(r_cyl_kpc)
    sorted_masses = gas_masses_Msun[sort_idx]

    # Cumulative mass EXCLUDING current particle (shift by 1)
    cumsum = np.concatenate([[0.0], np.cumsum(sorted_masses[:-1])])
    M_enc_sorted = (M_stars_Msun + cumsum) * Msun   # -> grams

    M_enc = np.empty(len(r_cyl_kpc))
    M_enc[sort_idx] = M_enc_sorted
    return M_enc


def compute_keplerian_velocity(r_cyl_kpc, M_enc_g):
    """
    Compute Keplerian velocity at each cylindrical radius.

    Parameters
    ----------
    r_cyl_kpc : ndarray, shape (N,)
        Cylindrical radii [kpc].
    M_enc_g : ndarray, shape (N,)
        Enclosed mass [g].

    Returns
    -------
    v_K_kms : ndarray, shape (N,)
        Keplerian velocity [km/s].
    """
    r_cyl_cm = np.maximum(r_cyl_kpc * kpc, 1e-10)   # cm
    v_K_cms = np.sqrt(G * M_enc_g / r_cyl_cm)
    v_K_kms = v_K_cms / 1e5                          # cm/s -> km/s
    return v_K_kms


# ═══════════════════════════════════════════════════════════════════════════════
# Disk identification
# ═══════════════════════════════════════════════════════════════════════════════

def extend_disk_to_bounds(is_disk_kinematic, r_cyl_kpc, z_kpc, percentile=100):
    """
    Extend kinematically identified disk to fill its bounding cylinder.

    This eliminates holes due to sub-Keplerian or low-v_phi gas.

    Parameters
    ----------
    is_disk_kinematic : ndarray, shape (N,), dtype=bool
        Kinematically selected disk particles.
    r_cyl_kpc : ndarray, shape (N,)
        Cylindrical radii [kpc].
    z_kpc : ndarray, shape (N,)
        Heights along L_hat [kpc].
    percentile : float
        Percentile used to define R_bound and H_bound (default 100).

    Returns
    -------
    is_disk_bounded : ndarray, shape (N,), dtype=bool
        All gas within the bounding cylinder.
    R_bound_kpc : float
        Bounding cylindrical radius [kpc].
    H_bound_kpc : float
        Bounding half-thickness [kpc].
    """
    if is_disk_kinematic.sum() == 0:
        return is_disk_kinematic.copy(), 0.0, 0.0

    R_bound_kpc = np.percentile(r_cyl_kpc[is_disk_kinematic], percentile)
    H_bound_kpc = np.percentile(np.abs(z_kpc[is_disk_kinematic]), percentile)

    is_disk_bounded = (r_cyl_kpc < R_bound_kpc) & (np.abs(z_kpc) < H_bound_kpc)

    return is_disk_bounded, R_bound_kpc, H_bound_kpc


def identify_disk(pdata, stardata,
                  r_search_kpc=1e-5,
                  r_max_kpc=1e-5,
                  rho_threshold_cgs=1e-15,
                  aspect_ratio=0.3,
                  f_kep=0.3,
                  use_bounds=True,
                  bounds_percentile=100,
                  precomputed_center=None,
                  precomputed_L_hat=None):
    """
    Identify disk gas particles using a hybrid geometric + kinematic criterion.

    Method summary:
      1. Find center (sink CoM or densest gas)
      2. Compute disk axis L_hat from angular momentum
      3. Project to cylindrical coordinates
      4. Geometric pre-selection: r_cyl < r_max, |z|/r_cyl < aspect_ratio, rho > threshold
      5. Kinematic filter: v_phi > 0, v_phi/v_K > f_kep
      6. Optionally extend to bounding cylinder

    Parameters
    ----------
    pdata : dict
        Gas particle data with keys: 'Coordinates', 'Velocities', 'Masses', 'Density'.
        Units: kpc, km/s, 10^10 M_sun, 10^10 M_sun/kpc^3.
    stardata : dict or None
        Sink/star particle data with 'Coordinates' and 'Masses'.
    r_search_kpc : float
        Sphere radius for computing COM velocity and L_hat [kpc].
    r_max_kpc : float
        Maximum cylindrical radius for disk candidates [kpc].
    rho_threshold_cgs : float
        Minimum density to be considered disk gas [g/cm^3].
    aspect_ratio : float
        Maximum |z| / r_cyl (geometric thinness).
    f_kep : float
        Minimum v_phi / v_K (rotational support fraction).
    use_bounds : bool
        Whether to extend kinematic selection to bounding cylinder.
    bounds_percentile : float
        Percentile for bounding cylinder (100 = strict box).
    precomputed_center : ndarray or None
        If provided, use this center instead of computing it.
    precomputed_L_hat : ndarray or None
        If provided, use this L_hat instead of computing it.

    Returns
    -------
    is_disk : ndarray, shape (N,), dtype=bool
        Boolean mask of disk particles.
    com : ndarray, shape (3,)
        Disk center [kpc].
    L_hat : ndarray, shape (3,)
        Disk rotation axis unit vector.
    r_cyl_kpc : ndarray, shape (N,)
        Cylindrical radii of all gas particles [kpc].
    z_kpc : ndarray, shape (N,)
        Heights of all gas particles along L_hat [kpc].
    v_phi_kms : ndarray, shape (N,)
        Azimuthal velocities [km/s].
    v_K_kms : ndarray, shape (N,)
        Keplerian velocities at each particle's r_cyl [km/s].
    """
    # ── Center ────────────────────────────────────────────────────────────────
    if precomputed_center is not None:
        com = precomputed_center.copy()
    else:
        com = find_center(pdata, stardata)

    gas_pos_kpc = pdata['Coordinates'] - com
    gas_dists = np.linalg.norm(gas_pos_kpc, axis=1)
    gas_masses_Msun = pdata['Masses'] * 1e10   # code -> Msun

    # ── COM velocity (from gas within r_search) ───────────────────────────────
    search_mask = gas_dists < r_search_kpc
    if search_mask.sum() > 0:
        com_vel = (np.sum(pdata['Velocities'][search_mask]
                          * gas_masses_Msun[search_mask, None], axis=0)
                   / np.sum(gas_masses_Msun[search_mask]))
    else:
        com_vel = np.zeros(3)

    gas_vel_com = pdata['Velocities'] - com_vel   # km/s, COM frame

    # ── Disk axis ─────────────────────────────────────────────────────────────
    if precomputed_L_hat is not None:
        L_hat = precomputed_L_hat / np.linalg.norm(precomputed_L_hat)
    else:
        L_hat = get_disk_axis(gas_pos_kpc, gas_vel_com, pdata['Masses'], r_search_kpc)

    # ── Cylindrical coordinates ───────────────────────────────────────────────
    r_cyl_kpc, z_kpc, v_phi_kms, v_r_kms, v_z_kms = cylindrical_coords(
        gas_pos_kpc, gas_vel_com, L_hat)

    # ── Enclosed mass -> Keplerian velocity ───────────────────────────────────
    M_stars_Msun = (np.sum(stardata['Masses']) * 1e10
                    if stardata and len(stardata.get('Masses', [])) > 0 else 0.0)
    M_enc_g = compute_M_enc(r_cyl_kpc, gas_masses_Msun, M_stars_Msun)
    v_K_kms = compute_keplerian_velocity(r_cyl_kpc, M_enc_g)

    # ── Density in g/cm^3 ─────────────────────────────────────────────────────
    rho_gcm3 = pdata['Density'] * 1e10 * Msun / kpc**3

    # ── Stage 1: Geometric pre-selection ──────────────────────────────────────
    safe_r_cyl = np.maximum(r_cyl_kpc, 1e-30)
    is_within = r_cyl_kpc < r_max_kpc
    is_equatorial = np.abs(z_kpc) / safe_r_cyl < aspect_ratio
    is_dense = rho_gcm3 > rho_threshold_cgs

    # ── Stage 2: Kinematic filter ─────────────────────────────────────────────
    is_corotating = v_phi_kms > 0
    is_supported = v_phi_kms / np.maximum(v_K_kms, 1e-10) > f_kep

    is_disk = is_within & is_equatorial & is_dense & is_corotating & is_supported

    # ── Extend to bounding cylinder (fills holes) ─────────────────────────────
    if use_bounds:
        is_disk, R_bound, H_bound = extend_disk_to_bounds(
            is_disk, r_cyl_kpc, z_kpc, percentile=bounds_percentile)

    return is_disk, com, L_hat, r_cyl_kpc, z_kpc, v_phi_kms, v_K_kms


# ═══════════════════════════════════════════════════════════════════════════════
# Disk property computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_disk_properties(pdata, stardata, is_disk, r_cyl_kpc, z_kpc, L_hat):
    """
    Compute summary properties of the identified disk.

    Parameters
    ----------
    pdata : dict
        Gas particle data.
    stardata : dict or None
        Sink/star particle data.
    is_disk : ndarray, shape (N,), dtype=bool
        Boolean mask of disk particles.
    r_cyl_kpc : ndarray, shape (N,)
        Cylindrical radii [kpc].
    z_kpc : ndarray, shape (N,)
        Heights along L_hat [kpc].
    L_hat : ndarray, shape (3,)
        Disk rotation axis.

    Returns
    -------
    props : dict
        Dictionary with disk properties:
        - M_disk_Msun: disk gas mass [M_sun]
        - M_stars_Msun: total stellar mass [M_sun]
        - R_disk_kpc, R_disk_AU: disk radius (90th percentile) [kpc, AU]
        - H_disk_kpc, H_disk_AU: scale height (90th percentile) [kpc, AU]
        - H_over_R: aspect ratio
        - n_disk_particles: number of disk particles
        - n_stars: number of sink/star particles
        - L_hat: rotation axis vector
    """
    gas_masses_Msun = pdata['Masses'] * 1e10
    disk_masses = gas_masses_Msun[is_disk]
    M_disk = disk_masses.sum()

    n_stars = len(stardata['Masses']) if stardata and len(stardata.get('Masses', [])) > 0 else 0
    M_stars = np.sum(stardata['Masses']) * 1e10 if n_stars > 0 else 0.0

    if is_disk.sum() > 0:
        R_disk_kpc = np.percentile(r_cyl_kpc[is_disk], 90)
        H_disk_kpc = np.percentile(np.abs(z_kpc[is_disk]), 90)
    else:
        R_disk_kpc = H_disk_kpc = 0.0

    return {
        'M_disk_Msun': M_disk,
        'M_stars_Msun': M_stars,
        'R_disk_kpc': R_disk_kpc,
        'R_disk_AU': R_disk_kpc * kpc / AU,
        'H_disk_kpc': H_disk_kpc,
        'H_disk_AU': H_disk_kpc * kpc / AU,
        'H_over_R': H_disk_kpc / R_disk_kpc if R_disk_kpc > 0 else 0.0,
        'n_disk_particles': int(is_disk.sum()),
        'n_stars': n_stars,
        'L_hat': L_hat,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Rotation curve diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rotation_curve(pdata, stardata, com, L_hat, r_max_kpc,
                           n_bins=50):
    """
    Compute binned rotation curve for diagnostic plotting.

    Parameters
    ----------
    pdata : dict
        Gas particle data.
    stardata : dict or None
        Sink/star particle data.
    com : ndarray, shape (3,)
        Disk center [kpc].
    L_hat : ndarray, shape (3,)
        Disk rotation axis.
    r_max_kpc : float
        Maximum radius to include [kpc].
    n_bins : int
        Number of radial bins.

    Returns
    -------
    r_bins_kpc : ndarray, shape (n_bins,)
        Bin centers [kpc].
    v_phi_median : ndarray, shape (n_bins,)
        Median azimuthal velocity per bin [km/s].
    v_phi_std : ndarray, shape (n_bins,)
        Standard deviation of v_phi per bin [km/s].
    v_K_bins : ndarray, shape (n_bins,)
        Keplerian velocity at bin centers [km/s].
    """
    gas_pos_kpc = pdata['Coordinates'] - com
    gas_dists = np.linalg.norm(gas_pos_kpc, axis=1)
    gas_masses_Msun = pdata['Masses'] * 1e10

    # COM velocity
    search_mask = gas_dists < r_max_kpc
    if search_mask.sum() > 0:
        com_vel = (np.sum(pdata['Velocities'][search_mask]
                          * gas_masses_Msun[search_mask, None], axis=0)
                   / np.sum(gas_masses_Msun[search_mask]))
    else:
        com_vel = np.zeros(3)

    gas_vel_com = pdata['Velocities'] - com_vel

    # Cylindrical coordinates
    r_cyl_kpc, z_kpc, v_phi_kms, _, _ = cylindrical_coords(
        gas_pos_kpc, gas_vel_com, L_hat)

    # Enclosed mass
    M_stars_Msun = (np.sum(stardata['Masses']) * 1e10
                    if stardata and len(stardata.get('Masses', [])) > 0 else 0.0)

    # Radial bins
    r_edges = np.linspace(0, r_max_kpc * 2, n_bins + 1)
    r_bins_kpc = 0.5 * (r_edges[1:] + r_edges[:-1])

    v_phi_median = np.zeros(n_bins)
    v_phi_std = np.zeros(n_bins)
    v_K_bins = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (r_cyl_kpc >= r_edges[i]) & (r_cyl_kpc < r_edges[i + 1])
        if mask.sum() > 0:
            v_phi_median[i] = np.median(v_phi_kms[mask])
            v_phi_std[i] = np.std(v_phi_kms[mask])

        # Keplerian velocity at bin center
        M_enc_bin = M_stars_Msun + np.sum(gas_masses_Msun[r_cyl_kpc < r_bins_kpc[i]])
        r_cm = r_bins_kpc[i] * kpc
        if r_cm > 0:
            v_K_bins[i] = np.sqrt(G * M_enc_bin * Msun / r_cm) / 1e5

    return r_bins_kpc, v_phi_median, v_phi_std, v_K_bins


# ═══════════════════════════════════════════════════════════════════════════════
# Integration with existing starforge_plot.py
# ═══════════════════════════════════════════════════════════════════════════════

def identify_disk_from_data_dict(data_dict,
                                  r_search_kpc=1e-5,
                                  r_max_kpc=1e-5,
                                  rho_threshold_cgs=1e-15,
                                  aspect_ratio=0.3,
                                  f_kep=0.3,
                                  use_bounds=True):
    """
    Identify disk from a data_dict as returned by setup_meshoid.

    This is a convenience wrapper for integration with the existing plotting code.

    Parameters
    ----------
    data_dict : dict
        Dictionary from setup_meshoid with 'pdata', 'star_data', 'center', 'L'.
    r_search_kpc : float
        Sphere radius for computing L_hat [kpc].
    r_max_kpc : float
        Outer disk boundary [kpc].
    rho_threshold_cgs : float
        Minimum density [g/cm^3].
    aspect_ratio : float
        Maximum |z| / r_cyl.
    f_kep : float
        Minimum v_phi / v_K.
    use_bounds : bool
        Whether to extend to bounding cylinder.

    Returns
    -------
    is_disk : ndarray, dtype=bool
        Boolean mask of disk particles.
    disk_props : dict
        Dictionary of disk properties.
    """
    pdata = data_dict['pdata']
    stardata = data_dict.get('star_data', None)

    # Use pre-computed center and L from data_dict if available
    precomputed_center = data_dict.get('original_center', data_dict.get('center', None))
    precomputed_L = data_dict.get('L', None)

    is_disk, com, L_hat, r_cyl, z, v_phi, v_K = identify_disk(
        pdata, stardata,
        r_search_kpc=r_search_kpc,
        r_max_kpc=r_max_kpc,
        rho_threshold_cgs=rho_threshold_cgs,
        aspect_ratio=aspect_ratio,
        f_kep=f_kep,
        use_bounds=use_bounds,
        precomputed_center=precomputed_center,
        precomputed_L_hat=precomputed_L
    )

    disk_props = compute_disk_properties(pdata, stardata, is_disk, r_cyl, z, L_hat)

    return is_disk, disk_props
