"""Cumulative DM mass vs radius from the full simulation.

Uses a 39 GB snapshot that has PartType1 data (smaller snapshots had DM stripped).
Centers on the most massive sink particle.
"""
import numpy as np
import h5py
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 22,
    'axes.titlesize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
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

_CLUSTER = '/scratch/vasissua'
_LOCAL   = '/home/vasilii/research/trillium/scratch'
BASE     = _CLUSTER if os.path.isdir(_CLUSTER) else _LOCAL

# Use snapshot_270 — latest 39 GB snapshot with DM data
SNAP_PATH = os.path.join(BASE, 'COPY/2026-03/m12f/output_jeans_refinement/snapshot_270.hdf5')
OUTDIR    = os.path.join(BASE, 'SHIVAN/analysis/paper_plots')
os.makedirs(os.path.join(OUTDIR, 'light'), exist_ok=True)

_AU  = 1.496e13   # cm
_kpc = 3.086e21   # cm
_pc  = _kpc / 1e3  # cm
_AU_per_kpc = _kpc / _AU

print(f"Loading: {os.path.basename(SNAP_PATH)}")

with h5py.File(SNAP_PATH, 'r') as f:
    hdr = dict(f['Header'].attrs)
    a   = float(hdr['Time'])
    h   = float(hdr.get('HubbleParam', 0.7))
    z   = float(hdr.get('Redshift', 0.0))

    npart = list(hdr.get('NumPart_ThisFile', hdr.get('NumPart_Total', [])))
    print(f"a={a:.5f}  z={z:.3f}  h={h:.3f}")
    print(f"NumPart: {npart}")

    # Center: most massive sink
    sk_m   = f['PartType5/Masses'][:] * 1e10 / h   # Msun
    sk_pos = f['PartType5/Coordinates'][:] * (a / h)  # physical kpc
    center = sk_pos[np.argmax(sk_m)]
    print(f"Center (most massive sink): {center} phys kpc, M={sk_m.max():.2f} Msun")

    # DM
    print("Loading DM coordinates (81M particles)...")
    dm_pos  = f['PartType1/Coordinates'][:].astype(np.float64) * (a / h)  # physical kpc
    dm_mass_each = float(f['PartType1/Masses'][0]) * 1e10 / h  # Msun (all equal)
    n_dm = len(dm_pos)
    print(f"  n_DM={n_dm}, m_DM={dm_mass_each:.2f} Msun each")

print("Computing distances...")
r_kpc = np.linalg.norm(dm_pos - center, axis=1)

# Sort by radius
sort_idx = np.argsort(r_kpc)
r_sorted = r_kpc[sort_idx]
M_cum = np.arange(1, n_dm + 1) * dm_mass_each  # Msun

# Print some reference points
for r_ref in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    n_inside = np.searchsorted(r_sorted, r_ref)
    M_inside = n_inside * dm_mass_each
    print(f"  r < {r_ref:8.3f} kpc:  N_DM={n_inside:>10,}  M_DM={M_inside:.2e} Msun")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))

# Downsample for plotting (81M points is too many)
# Log-spaced sampling
n_plot = 5000
idx_plot = np.unique(np.logspace(0, np.log10(n_dm - 1), n_plot).astype(int))

ax.loglog(r_sorted[idx_plot], M_cum[idx_plot], color='#1f77b4', lw=2)

ax.set_xlabel('r [physical kpc]')
ax.set_ylabel(r'$M_{\rm DM}(<r)$ [$M_\odot$]')
ax.tick_params(which='both', top=True, right=True)

# Secondary x-axis in pc
ax_pc = ax.secondary_xaxis('top', functions=(lambda x: x * 1e3, lambda x: x / 1e3))
ax_pc.set_xlabel('r [pc]')
ax_pc.tick_params(direction='in', which='both')

# Reference lines
ax.axhline(1e8, color='grey', ls=':', lw=1, alpha=0.5)
ax.axhline(1e9, color='grey', ls=':', lw=1, alpha=0.5)
ax.axhline(1e10, color='grey', ls=':', lw=1, alpha=0.5)
ax.axhline(1e11, color='grey', ls=':', lw=1, alpha=0.5)

# Info annotation
info = f'snap 270  z={z:.2f}  a={a:.4f}\nn_DM={n_dm:,}\nm_DM={dm_mass_each:.1f} Msun'
ax.text(0.03, 0.97, info, transform=ax.transAxes, va='top', fontsize=14,
        family='monospace', bbox=dict(fc='white', alpha=0.7, ec='none'))

plt.tight_layout()
out_path = os.path.join(OUTDIR, 'light', 'dm_cumulative_mass.png')
fig.savefig(out_path, dpi=150, facecolor='w')
print(f"Saved: {out_path}")

# ── Plot 2: DM density profile ρ(r) ─────────────────────────────────────────
# Bin into log-spaced radial bins and compute mean density in each shell
r_min_kpc = 1e-4   # 0.1 pc
r_max_kpc = 200.0
n_bins = 80
r_edges = np.logspace(np.log10(r_min_kpc), np.log10(r_max_kpc), n_bins + 1)
r_cen   = np.sqrt(r_edges[:-1] * r_edges[1:])

rho_dm  = np.zeros(n_bins)
for i, (r0, r1) in enumerate(zip(r_edges[:-1], r_edges[1:])):
    mask = (r_kpc >= r0) & (r_kpc < r1)
    n_in = mask.sum()
    vol_kpc3 = (4.0 / 3.0) * np.pi * (r1**3 - r0**3)  # kpc^3
    vol_cm3  = vol_kpc3 * (_kpc * 1e3)**3              # cm^3 (1 kpc = 3.086e21 cm; 1 pc = 3.086e18 cm)
    # Actually use kpc^3 directly and convert to Msun/pc^3 at the end
    rho_dm[i] = (n_in * dm_mass_each) / (vol_kpc3 * 1e9)  # Msun / pc^3

fig2, ax2 = plt.subplots(figsize=(12, 9))

valid_bins = rho_dm > 0
ax2.loglog(r_cen[valid_bins] * 1e3, rho_dm[valid_bins], color='#1f77b4', lw=2)

ax2.set_xlabel('r [pc]')
ax2.set_ylabel(r'$\rho_{\rm DM}$ [$M_\odot\,{\rm pc}^{-3}$]')
ax2.tick_params(which='both', top=True, right=True)

# Secondary x-axis in kpc
ax2_kpc = ax2.secondary_xaxis('top', functions=(lambda x: x / 1e3, lambda x: x * 1e3))
ax2_kpc.set_xlabel('r [kpc]')
ax2_kpc.tick_params(direction='in', which='both')

ax2.text(0.03, 0.97, info, transform=ax2.transAxes, va='top', fontsize=14,
         family='monospace', bbox=dict(fc='white', alpha=0.7, ec='none'))

plt.tight_layout()
out_path2 = os.path.join(OUTDIR, 'light', 'dm_density_profile.png')
fig2.savefig(out_path2, dpi=150, facecolor='w')
print(f"Saved: {out_path2}")
