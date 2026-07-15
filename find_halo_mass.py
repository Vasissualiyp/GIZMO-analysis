"""
Find the DM halo mass (M_200c) in the FIRE zoom-in simulation.

Strategy:
1. Find the earliest snapshot that has DM coordinate data (snap 0 has it)
2. Read PartType1 (high-res DM) and PartType2 (low-res DM) masses + coordinates
3. Identify high-res DM particles (smallest mass species)
4. Shrinking-sphere center refinement
5. Build spherical enclosed-mass profile
6. Find R_200c where mean enclosed density = 200 * rho_crit(z)
"""

import numpy as np
import h5py
import glob
import os
import sys

# Support both cluster and local mount paths
_cluster_path = '/scratch/vasissua/COPY/2026-03/m12f/output_jeans_refinement'
_local_path = '/home/vasilii/research/trillium/scratch/COPY/2026-03/m12f/output_jeans_refinement'
SIM_DIR = _cluster_path if os.path.isdir(_cluster_path) else _local_path

# Find a snapshot that has DM coordinate data
# Later snapshots may have dropped DM output to save space
snaps = sorted(glob.glob(os.path.join(SIM_DIR, 'snapshot_*.hdf5')))
snap_path = None
for sp in snaps:
    with h5py.File(sp, 'r') as f:
        if 'PartType1' in f and 'Coordinates' in f['PartType1']:
            snap_path = sp
            break
    # Also try from the end
if snap_path is None:
    for sp in reversed(snaps):
        with h5py.File(sp, 'r') as f:
            if 'PartType1' in f and 'Coordinates' in f['PartType1']:
                snap_path = sp
                break

if snap_path is None:
    print("ERROR: No snapshot found with DM coordinate data!")
    sys.exit(1)

print(f"Using snapshot: {snap_path}")

with h5py.File(snap_path, 'r') as f:
    hdr = dict(f['Header'].attrs)

    # Cosmological parameters
    a = hdr['Time']
    z = 1.0 / a - 1.0
    h = hdr['HubbleParam']
    Omega_m = hdr.get('Omega0', hdr.get('Omega_Matter', 0.272))
    Omega_L = hdr.get('OmegaLambda', hdr.get('Omega_Lambda', 0.728))
    Omega_b = hdr.get('Omega_Baryon', 0.0455)
    print(f"Scale factor a = {a:.6f}, z = {z:.2f}, h = {h}")
    print(f"Omega_m = {Omega_m}, Omega_L = {Omega_L}, Omega_b = {Omega_b}")

    # Collect DM particles (PartType1 = high-res, PartType2 = low-res boundary)
    dm_coords_list = []
    dm_masses_list = []
    dm_type_list = []
    for pt_idx in [1, 2]:
        pt_name = f'PartType{pt_idx}'
        if pt_name not in f or 'Coordinates' not in f[pt_name]:
            continue
        coords_i = f[pt_name + '/Coordinates'][:]
        if 'Masses' in f[pt_name]:
            masses_i = f[pt_name + '/Masses'][:]
        else:
            masses_i = np.full(coords_i.shape[0], hdr['MassTable'][pt_idx])
        dm_coords_list.append(coords_i)
        dm_masses_list.append(masses_i)
        dm_type_list.append(np.full(len(masses_i), pt_idx, dtype=int))
        print(f"  {pt_name}: N = {len(masses_i)}, "
              f"m_range = [{masses_i.min():.6e}, {masses_i.max():.6e}] (1e10 Msun/h)")

    # Also load gas (PartType0) for total mass profile
    gas_coords = None
    gas_masses = None
    if 'PartType0' in f and 'Coordinates' in f['PartType0']:
        gas_coords = f['PartType0/Coordinates'][:]
        gas_masses = f['PartType0/Masses'][:]
        print(f"  PartType0 (gas): N = {len(gas_masses)}")

dm_coords = np.concatenate(dm_coords_list, axis=0)
dm_masses = np.concatenate(dm_masses_list, axis=0)
dm_types = np.concatenate(dm_type_list, axis=0)

# Convert to physical units
dm_coords_phys = dm_coords * a / h   # physical kpc
dm_masses_msun = dm_masses * 1e10 / h  # Msun

if gas_coords is not None:
    gas_coords_phys = gas_coords * a / h
    gas_masses_msun = gas_masses * 1e10 / h

print(f"\nTotal DM particles: {len(dm_masses)}")
print(f"Total DM mass: {dm_masses_msun.sum():.4e} Msun")

# Identify high-res DM (PartType1 with smallest mass)
pt1_mask = dm_types == 1
if pt1_mask.any():
    m_pt1 = dm_masses[pt1_mask]
    unique_m1 = np.unique(np.round(m_pt1, 12))
    print(f"\nPartType1 unique masses: {len(unique_m1)}")
    for um in unique_m1[:5]:
        n = np.sum(np.isclose(m_pt1, um, rtol=1e-4))
        print(f"  m = {um:.8e} (1e10 Msun/h) = {um*1e10/h:.4e} Msun, N = {n}")
    m_hires = unique_m1[0]
    hires_mask = pt1_mask & np.isclose(dm_masses, m_hires, rtol=1e-3)
else:
    hires_mask = np.ones(len(dm_masses), dtype=bool)

n_hires = hires_mask.sum()
print(f"\nHigh-res DM particles: N = {n_hires}, m_p = {dm_masses_msun[hires_mask][0]:.4e} Msun")

# Shrinking-sphere center
coords_hr = dm_coords_phys[hires_mask]
masses_hr = dm_masses_msun[hires_mask]

com = np.average(coords_hr, weights=masses_hr, axis=0)
print(f"Initial COM (physical kpc): [{com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f}]")

for r_frac in [0.5, 0.3, 0.15, 0.08, 0.04, 0.02]:
    r = np.sqrt(np.sum((coords_hr - com)**2, axis=1))
    r_cut = r_frac * r.max()
    mask = r < r_cut
    if mask.sum() > 50:
        com = np.average(coords_hr[mask], weights=masses_hr[mask], axis=0)

print(f"Refined COM (physical kpc): [{com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f}]")

# Build enclosed mass profile using ALL matter (DM + gas)
all_coords = [dm_coords_phys]
all_masses = [dm_masses_msun]
if gas_coords is not None:
    all_coords.append(gas_coords_phys)
    all_masses.append(gas_masses_msun)

all_coords = np.concatenate(all_coords, axis=0)
all_masses = np.concatenate(all_masses, axis=0)

r_all = np.sqrt(np.sum((all_coords - com)**2, axis=1))
sort_idx = np.argsort(r_all)
r_sorted = r_all[sort_idx]
m_sorted = all_masses[sort_idx]
m_enc = np.cumsum(m_sorted)

# Also DM-only profile
r_dm = np.sqrt(np.sum((dm_coords_phys - com)**2, axis=1))
sort_dm = np.argsort(r_dm)
r_dm_sorted = r_dm[sort_dm]
m_dm_enc = np.cumsum(dm_masses_msun[sort_dm])

# Critical density
G_cgs = 6.674e-8
H0_cgs = h * 100 * 1e5 / (3.086e24)  # s^-1
H_z = H0_cgs * np.sqrt(Omega_m * (1+z)**3 + Omega_L)
rho_crit_cgs = 3 * H_z**2 / (8 * np.pi * G_cgs)

Msun_cgs = 1.989e33
kpc_cm = 3.086e21
rho_crit = rho_crit_cgs / Msun_cgs * kpc_cm**3  # Msun/kpc^3

print(f"\nz = {z:.2f}")
print(f"H(z) = {H_z:.4e} s^-1 = {H_z*3.086e24/1e5:.2f} km/s/Mpc")
print(f"rho_crit = {rho_crit:.4e} Msun/kpc^3")

# Find R_200c (total matter)
valid = r_sorted > 0
vol = (4.0/3.0) * np.pi * r_sorted[valid]**3
mean_rho = m_enc[np.where(valid)[0]] / vol

target_200 = 200 * rho_crit

above = mean_rho >= target_200
if above.any():
    last_above = np.where(above)[0][-1]
    idx_200 = np.where(valid)[0][last_above]
    R_200 = r_sorted[idx_200]
    M_200 = m_enc[idx_200]
    print(f"\n{'='*60}")
    print(f"  M_200c (total) = {M_200:.4e} Msun")
    print(f"  R_200c         = {R_200:.6f} kpc = {R_200*1e3:.2f} pc")
    print(f"{'='*60}")

    check_rho = M_200 / ((4.0/3.0)*np.pi*R_200**3)
    print(f"  Check: mean rho / (200*rho_crit) = {check_rho/target_200:.4f}")
else:
    print(f"\nWARNING: mean density never reaches 200*rho_crit = {target_200:.4e} Msun/kpc^3")
    print(f"  Max mean density: {mean_rho.max():.4e}")
    print(f"  Total mass: {m_enc[-1]:.4e} Msun")

# DM-only R_200
valid_dm = r_dm_sorted > 0
vol_dm = (4.0/3.0) * np.pi * r_dm_sorted[valid_dm]**3
mean_rho_dm = m_dm_enc[np.where(valid_dm)[0]] / vol_dm

above_dm = mean_rho_dm >= target_200
if above_dm.any():
    last_dm = np.where(above_dm)[0][-1]
    idx_dm = np.where(valid_dm)[0][last_dm]
    R_200_dm = r_dm_sorted[idx_dm]
    M_200_dm = m_dm_enc[idx_dm]
    print(f"\n  M_200c (DM only)  = {M_200_dm:.4e} Msun")
    print(f"  R_200c (DM only)  = {R_200_dm:.6f} kpc = {R_200_dm*1e3:.2f} pc")

# Bryan & Norman virial overdensity
x_bn = Omega_m * (1+z)**3 / (Omega_m * (1+z)**3 + Omega_L) - 1
Delta_vir = 18*np.pi**2 + 82*x_bn - 39*x_bn**2
target_vir = Delta_vir * rho_crit
above_vir = mean_rho >= target_vir
if above_vir.any():
    last_vir = np.where(above_vir)[0][-1]
    idx_vir = np.where(valid)[0][last_vir]
    R_vir = r_sorted[idx_vir]
    M_vir = m_enc[idx_vir]
    print(f"\n  Bryan & Norman Delta_vir = {Delta_vir:.1f}")
    print(f"  M_vir (total) = {M_vir:.4e} Msun")
    print(f"  R_vir         = {R_vir:.6f} kpc = {R_vir*1e3:.2f} pc")

# Print enclosed mass at a few radii for context
print(f"\n--- Enclosed mass profile ---")
for r_test_pc in [10, 50, 100, 500, 1000, 5000]:
    r_test = r_test_pc / 1e3  # kpc
    idx = np.searchsorted(r_sorted, r_test)
    if idx < len(m_enc):
        print(f"  r = {r_test_pc:>5d} pc: M_enc = {m_enc[idx]:.4e} Msun")

print("\nDone.")
