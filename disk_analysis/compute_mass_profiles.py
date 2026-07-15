"""
compute_mass_profiles.py
========================
Standalone script: reads every cutout snapshot with h5py, computes
spherical shell mass and accretion rate profiles (10 AU → 10^4 AU),
and saves one massprofile_XXXX.npz per snapshot.

Output: {frames_dir}/massprofiles/massprofile_XXXX.npz
  Each file contains:
    r_AU     - bin centres [AU]
    m_shell  - shell mass [Msun] per bin
    mdot     - net mass inflow rate [Msun/yr] (positive = inward)
    time_Myr - snapshot time [Myr]
    n_sinks  - number of sink particles

Usage (cluster):
    python disk_analysis/compute_mass_profiles.py [--overwrite]
"""

import os, sys, glob, argparse
import numpy as np
import h5py
from scipy.integrate import quad

# ── Paths ──────────────────────────────────────────────────────────────────
_CLUSTER = '/scratch/vasissua'
_LOCAL   = '/home/vasilii/research/trillium/scratch'
BASE     = _CLUSTER if os.path.isdir(_CLUSTER) else _LOCAL

CUTOUT_DIR  = os.path.join(BASE, 'COPY/2026-03/m12f_cutout/output_cutout')
FRAMES_DIR  = os.path.join(BASE, 'SHIVAN/analysis/frames18')
OUT_DIR     = os.path.join(FRAMES_DIR, 'massprofiles')

# ── Constants ──────────────────────────────────────────────────────────────
AU    = 1.496e13   # cm
kpc   = 3.086e21   # cm
Msun  = 1.989e33   # g

N_SPH         = 50
R_MIN_AU      = 10.0
R_MAX_AU      = 1e4
R_SEARCH_KPC  = 1e-5    # COM velocity search radius (kpc)

_r_min_kpc  = R_MIN_AU  * AU / kpc
_r_max_kpc  = R_MAX_AU  * AU / kpc
SPH_EDGES   = np.logspace(np.log10(_r_min_kpc), np.log10(_r_max_kpc), N_SPH + 1)
SPH_CTR_AU  = np.sqrt(SPH_EDGES[:-1] * SPH_EDGES[1:]) * kpc / AU

# Conversion: 1e10 Msun * km/s / kpc  →  Msun/yr
_C_MDOT = 1e10 * 3.156e7 / 3.086e16   # ≈ 10.23


# ── Cosmological time ──────────────────────────────────────────────────────
def scale_factor_to_Myr(a, H0=70.4, Omega_m=0.2726, Omega_L=0.728):
    """Flat ΛCDM time integral from 0 to a, returned in Myr."""
    integrand = lambda ap: 1.0 / (ap * np.sqrt(Omega_m / ap**3 + Omega_L))
    t_Gyr, _ = quad(integrand, 1e-6, a)
    t_Gyr   /= (H0 * 1.022e-3)   # H0 km/s/Mpc → Gyr^-1;  result in Gyr
    return t_Gyr * 1e3            # Gyr → Myr


# ── Per-snapshot computation ───────────────────────────────────────────────
def process_snapshot(snap_path):
    """Return (r_AU, m_shell, mdot, time_Myr, n_sinks) or None on failure."""
    try:
        with h5py.File(snap_path, 'r') as f:
            hdr   = dict(f['Header'].attrs)
            a     = float(hdr['Time'])
            h_hub = float(hdr.get('HubbleParam', 0.702))

            n_gas  = int(hdr['NumPart_Total'][0])
            n_sink = int(hdr['NumPart_Total'][5])
            if n_gas == 0:
                return None

            # ── Centre: most massive sink (fallback: gas median) ──
            if n_sink > 0 and 'PartType5' in f:
                s_pos = f['PartType5/Coordinates'][:] * (a / h_hub)   # physical kpc
                s_mas = f['PartType5/Masses'][:]
                com   = s_pos[np.argmax(s_mas)]
            else:
                g_pos_raw = f['PartType0/Coordinates'][:] * (a / h_hub)
                com = np.median(g_pos_raw, axis=0)

            # ── Gas particles ──
            g_pos = f['PartType0/Coordinates'][:] * (a / h_hub)   # physical kpc
            g_vel = f['PartType0/Velocities'][:] * np.sqrt(a)      # km/s
            g_mas = f['PartType0/Masses'][:]                        # 1e10 Msun/h
            g_mas = g_mas / h_hub                                   # 1e10 Msun

            coords_rel = g_pos - com
            r3d        = np.linalg.norm(coords_rel, axis=1)        # physical kpc

            # ── COM velocity (within R_SEARCH_KPC) ──
            cut_com = r3d < R_SEARCH_KPC
            if cut_com.sum() > 3:
                w = g_mas[cut_com]
                com_vel = np.dot(w, g_vel[cut_com]) / w.sum()
            else:
                com_vel = np.zeros(3)

            # ── Restrict to 10^4 AU sphere ──
            cut = r3d < _r_max_kpc
            if cut.sum() < 3:
                return None

            r    = r3d[cut]
            mass = g_mas[cut]
            vel  = g_vel[cut] - com_vel
            r_hat = coords_rel[cut] / np.maximum(r[:, np.newaxis], 1e-30)
            vr    = np.sum(vel * r_hat, axis=1)   # km/s, positive=outward

            # ── Bin ──
            bidx = np.clip(np.digitize(r, SPH_EDGES) - 1, 0, N_SPH - 1)
            m_shell = np.zeros(N_SPH)
            mdot    = np.zeros(N_SPH)
            for b in range(N_SPH):
                mb = bidx == b
                if mb.sum() == 0:
                    continue
                m_b = mass[mb]
                dr  = SPH_EDGES[b + 1] - SPH_EDGES[b]
                m_shell[b] = m_b.sum() * 1e10                          # Msun
                mdot[b]    = -np.sum(m_b * vr[mb]) / dr * _C_MDOT     # Msun/yr

            time_Myr = scale_factor_to_Myr(a)
            return SPH_CTR_AU.copy(), m_shell, mdot, time_Myr, n_sink

    except Exception as e:
        print(f"    WARNING: failed on {os.path.basename(snap_path)}: {e}")
        return None


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true',
                        help='Recompute even if output file already exists')
    parser.add_argument('--cutout-dir', default=CUTOUT_DIR)
    parser.add_argument('--frames-dir', default=FRAMES_DIR)
    args = parser.parse_args()

    out_dir = os.path.join(args.frames_dir, 'massprofiles')
    os.makedirs(out_dir, exist_ok=True)

    snaps = sorted(glob.glob(os.path.join(args.cutout_dir, 'snapshot_*.hdf5')))
    print(f"Found {len(snaps)} snapshots in {args.cutout_dir}")
    print(f"Output → {out_dir}")

    n_done = 0
    for snap_path in snaps:
        snap_num = int(os.path.basename(snap_path).replace('snapshot_', '').replace('.hdf5', ''))
        out_path = os.path.join(out_dir, f'massprofile_{snap_num:04d}.npz')

        if os.path.exists(out_path) and not args.overwrite:
            continue

        result = process_snapshot(snap_path)
        if result is None:
            continue

        r_AU, m_shell, mdot, time_Myr, n_sinks = result
        np.savez(out_path,
                 r_AU=r_AU, m_shell=m_shell, mdot=mdot,
                 time_Myr=np.array([time_Myr]),
                 n_sinks=np.array([n_sinks]))
        n_done += 1
        if n_done % 50 == 0:
            print(f"  Processed {n_done} snapshots...")

    print(f"Done. Wrote {n_done} massprofile files to {out_dir}")


if __name__ == '__main__':
    main()
