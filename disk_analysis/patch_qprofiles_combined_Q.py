"""
patch_qprofiles_combined_Q.py
==============================
Patch existing qprofile_XXXX.npz files to add cs_prof and Q_combined
(thermal + turbulent Toomre Q) without re-rendering all movie frames.

For each qprofile file the script:
  1. Reads stored sigma_r, kappa, Sigma (turbulent Q components).
  2. Loads the corresponding snapshot to compute mass-weighted sound-speed
     profile cs_prof in the same radial bins (using 3D radius as approximation).
  3. Saves Q_combined = sqrt(sigma_r² + cs²)·κ / (π G Σ) back into the npz.

Usage:
    python disk_analysis/patch_qprofiles_combined_Q.py [--overwrite]
    python disk_analysis/patch_qprofiles_combined_Q.py --backup-only
"""

import os
import sys
import glob
import shutil
import argparse

import numpy as np
import h5py

# ── Path setup ───────────────────────────────────────────────────────────────
_CLUSTER = '/scratch/vasissua'
_LOCAL   = '/home/vasilii/research/trillium/scratch'
BASE     = _CLUSTER if os.path.isdir(_CLUSTER) else _LOCAL

CUTOUT_DIR  = os.path.join(BASE, 'COPY/2026-03/m12f_cutout/output_jeans_refinement')
FRAMES_DIR  = os.path.join(BASE, 'SHIVAN/analysis/frames18')

# ── Physical constants (CGS) ─────────────────────────────────────────────────
_GAMMA = 5.0 / 3.0
G      = 6.674e-8    # cm³/g/s²
kpc    = 3.086e21    # cm
AU     = 1.496e13    # cm
Msun   = 1.989e33    # g


def _snap_num_from_path(p):
    return int(os.path.basename(p).replace('snapshot_', '').replace('.hdf5', ''))


def backup_qprofiles(qprofiles_dir):
    backup_dir = qprofiles_dir.rstrip('/') + '_turb_only_backup'
    if os.path.isdir(backup_dir):
        print(f"  Backup already exists: {backup_dir} — skipping.")
        return
    shutil.copytree(qprofiles_dir, backup_dir)
    print(f"  Backed up {qprofiles_dir} → {backup_dir}")


def compute_cs_prof(snap_path, r_kpc_centers):
    """Load snapshot, compute mass-weighted cs(r) in bins matching r_kpc_centers.

    Uses 3D spherical radius (good approximation since c_s is isotropic).
    Bin edges are reconstructed from uniform-spacing assumption.
    """
    try:
        with h5py.File(snap_path, 'r') as f:
            if 'PartType0' not in f:
                return None
            pt0 = f['PartType0']
            if 'InternalEnergy' not in pt0 or 'Masses' not in pt0 or 'Coordinates' not in pt0:
                return None
            u      = pt0['InternalEnergy'][:]   # (km/s)²
            m      = pt0['Masses'][:]           # code units
            coords = pt0['Coordinates'][:]      # comoving kpc/h
            hdr    = dict(f['Header'].attrs)
            a      = float(hdr['Time'])
            h_hub  = float(hdr.get('HubbleParam', 0.7))

            # Most massive sink for center
            center = None
            if 'PartType5' in f and 'Masses' in f['PartType5'] and f['PartType5/Masses'].shape[0] > 0:
                sm = f['PartType5/Masses'][:]
                sc = f['PartType5/Coordinates'][:] * (a / h_hub)
                center = sc[np.argmax(sm)]
    except Exception as e:
        print(f"    h5py load error: {e}")
        return None

    coords_phys = coords * (a / h_hub)   # physical kpc
    if center is None:
        center = np.median(coords_phys, axis=0)

    r3d = np.linalg.norm(coords_phys - center, axis=1)   # physical kpc

    # Reconstruct bin edges from uniformly-spaced centers
    n = len(r_kpc_centers)
    if n < 2:
        return None
    dr = r_kpc_centers[1] - r_kpc_centers[0]
    edges = np.concatenate([[0.0], r_kpc_centers + dr * 0.5])

    cs = np.sqrt(_GAMMA * (_GAMMA - 1.0) * np.maximum(u, 0.0))   # km/s

    cs_prof = np.zeros(n)
    for b in range(n):
        mask = (r3d >= edges[b]) & (r3d < edges[b + 1])
        if mask.sum() > 0:
            w = m[mask]
            cs_prof[b] = np.dot(cs[mask], w) / w.sum()
    return cs_prof


def patch_qprofile(qp_path, snap_path, overwrite=False):
    d = np.load(qp_path, allow_pickle=False)

    # Skip if already patched and not overwriting
    if 'Q_combined' in d.files and not overwrite:
        return False

    r_kpc  = d['r_kpc']
    sigma_r = d['sigma_r']   # km/s
    kappa   = d['kappa']     # km/s/kpc
    Sigma   = d['Sigma']     # g/cm²

    cs_prof = compute_cs_prof(snap_path, r_kpc)
    if cs_prof is None:
        return False

    with np.errstate(divide='ignore', invalid='ignore'):
        Q_combined = np.where(
            (Sigma > 0) & (kappa > 0),
            (np.sqrt(sigma_r**2 + cs_prof**2) * 1e5) * (kappa * 1e5 / kpc) / (np.pi * G * Sigma),
            np.nan)

    # Rebuild dict with new keys added
    data = {k: d[k] for k in d.files}
    data['cs_prof']    = cs_prof
    data['Q_combined'] = Q_combined

    np.savez(qp_path, **data)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames-dir',  default=FRAMES_DIR)
    parser.add_argument('--cutout-dir',  default=CUTOUT_DIR)
    parser.add_argument('--overwrite',   action='store_true',
                        help='Re-patch files that already have Q_combined')
    parser.add_argument('--backup-only', action='store_true',
                        help='Only create backup, do not patch')
    args = parser.parse_args()

    qprofiles_dir = os.path.join(args.frames_dir, 'qprofiles')
    if not os.path.isdir(qprofiles_dir):
        sys.exit(f"No qprofiles directory found: {qprofiles_dir}")

    print(f"Backing up {qprofiles_dir} ...")
    backup_qprofiles(qprofiles_dir)

    if args.backup_only:
        return

    qp_files = sorted(glob.glob(os.path.join(qprofiles_dir, 'qprofile_*.npz')))
    print(f"Found {len(qp_files)} qprofile files to patch")

    # Build snap_num → snap_path map
    snap_map = {}
    for p in glob.glob(os.path.join(args.cutout_dir, 'snapshot_*.hdf5')):
        snap_map[_snap_num_from_path(p)] = p

    n_patched = 0
    n_skipped = 0
    n_failed  = 0
    for i, qp_path in enumerate(qp_files):
        try:
            d_peek = np.load(qp_path, allow_pickle=False)
            snap_num = int(d_peek['snap_num'][0])
        except Exception as e:
            print(f"  [{i+1}] could not read {os.path.basename(qp_path)}: {e}")
            n_failed += 1
            continue

        snap_path = snap_map.get(snap_num)
        if snap_path is None:
            n_failed += 1
            continue

        ok = patch_qprofile(qp_path, snap_path, overwrite=args.overwrite)
        if ok:
            n_patched += 1
        else:
            n_skipped += 1

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{len(qp_files)}]  patched={n_patched}  skipped={n_skipped}  failed={n_failed}",
                  flush=True)

    print(f"\nDone.  patched={n_patched}  skipped={n_skipped}  failed={n_failed}")


if __name__ == '__main__':
    main()
