"""Last-snapshot sink classification: in disk vs outside disk vs outside aperture."""
import numpy as np
import h5py, glob, os

_CLUSTER = '/scratch/vasissua'
_LOCAL   = '/home/vasilii/research/trillium/scratch'
BASE     = _CLUSTER if os.path.isdir(_CLUSTER) else _LOCAL

CUTOUT_DIR = os.path.join(BASE, 'COPY/2026-03/m12f_cutout/output_jeans_refinement')
_AU = 1.496e13; _kpc = 3.086e21

snaps = sorted(glob.glob(os.path.join(CUTOUT_DIR, 'snapshot_*.hdf5')))
print(f"Total snapshots found: {len(snaps)}")
last_snap = snaps[-1]
print(f"Last snapshot: {os.path.basename(last_snap)}\n")

r_aperture = 2500.0   # AU — disk aperture (r_max parameter)

with h5py.File(last_snap, 'r') as f:
    hdr   = dict(f['Header'].attrs)
    a     = float(hdr['Time']); h = float(hdr.get('HubbleParam', 0.7))
    z     = float(hdr.get('Redshift', 0.0))
    npart = hdr.get('NumPart_ThisFile', hdr.get('NumPart_Total', [0]*6))
    print(f"Scale factor a={a:.5f}  z={z:.3f}")
    print(f"NumPart [Gas, DM, x, Ref, Stars, Sinks]: {list(npart)}\n")

    # Gas
    gas_coords = f['PartType0/Coordinates'][:].astype(np.float64) * (a/h)
    gas_mass   = f['PartType0/Masses'][:].astype(np.float64) * 1e10 / h

    # Sinks
    pt5        = f['PartType5']
    sk_masses  = pt5['Masses'][:].astype(np.float64) * 1e10 / h
    sk_coords  = pt5['Coordinates'][:].astype(np.float64) * (a/h)
    sk_pids    = pt5['ParticleIDs'][:]

n_sinks = len(sk_masses)

# Primary sink = most massive
primary_idx = np.argmax(sk_masses)
center       = sk_coords[primary_idx]
print(f"Primary sink mass: {sk_masses[primary_idx]:.3f} Msun  (idx={primary_idx})")

# 3D distances of each sink from primary
sk_r_AU = np.linalg.norm(sk_coords - center, axis=1) * _kpc / _AU

# Gas distances from primary
gas_r_AU     = np.linalg.norm(gas_coords - center, axis=1) * _kpc / _AU
gas_in_apt   = gas_r_AU[gas_r_AU <= r_aperture]
r_disk_est   = np.percentile(gas_in_apt, 90) if len(gas_in_apt) > 0 else r_aperture

print(f"Total gas particles          : {len(gas_r_AU)}")
print(f"Gas within aperture          : {len(gas_in_apt)}")
print(f"Gas max r in aperture        : {gas_in_apt.max():.1f} AU" if len(gas_in_apt) > 0 else "")
print(f"Total sinks in last snapshot : {n_sinks}")
print(f"Disk aperture  r_max         : {r_aperture:.0f} AU")
print(f"Gas disk size  (90th pct)    : {r_disk_est:.0f} AU\n")

in_disk         = sk_r_AU <= r_disk_est
in_apt_not_disk = (sk_r_AU > r_disk_est) & (sk_r_AU <= r_aperture)
outside_apt     = sk_r_AU > r_aperture

print(f"Inside disk (r ≤ {r_disk_est:.0f} AU)           : {in_disk.sum()}")
for i in np.where(in_disk)[0]:
    print(f"  PID={sk_pids[i]}  r={sk_r_AU[i]:7.1f} AU   M={sk_masses[i]:.3f} Msun")

print(f"\nInside aperture but outside disk ({r_disk_est:.0f}–{r_aperture:.0f} AU) : {in_apt_not_disk.sum()}")
for i in np.where(in_apt_not_disk)[0]:
    print(f"  PID={sk_pids[i]}  r={sk_r_AU[i]:7.1f} AU   M={sk_masses[i]:.3f} Msun")

print(f"\nOutside disk aperture (r > {r_aperture:.0f} AU)    : {outside_apt.sum()}")
for i in np.where(outside_apt)[0]:
    print(f"  PID={sk_pids[i]}  r={sk_r_AU[i]:7.1f} AU   M={sk_masses[i]:.3f} Msun")

print(f"\nSummary: {in_disk.sum()} in disk | {in_apt_not_disk.sum()} in aperture but outside disk | {outside_apt.sum()} outside aperture")
print(f"Total check: {in_disk.sum() + in_apt_not_disk.sum() + outside_apt.sum()} (should be {n_sinks})")

# Also show full gas extent (all gas, not just within aperture)
print(f"\nFull gas extent: max r = {gas_r_AU.max():.1f} AU = {gas_r_AU.max()/1000:.2f} kAU")
print(f"Gas r percentiles [50, 75, 90, 95, 99, 100]th:")
for p in [50, 75, 90, 95, 99, 100]:
    print(f"  {p:3d}th pct: {np.percentile(gas_r_AU, p):10.1f} AU")
