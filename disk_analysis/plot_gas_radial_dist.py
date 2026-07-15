"""
Plot N_gas vs log10(r_AU) for several cutout snapshots.
Helps verify the actual physical extent of the cutout at different times.
"""
import numpy as np
import h5py, glob, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_CLUSTER = '/scratch/vasissua'
_LOCAL   = '/home/vasilii/research/trillium/scratch'
BASE     = _CLUSTER if os.path.isdir(_CLUSTER) else _LOCAL

CUTOUT_DIR = os.path.join(BASE, 'COPY/2026-03/m12f_cutout/output_jeans_refinement')
OUTDIR     = os.path.join(BASE, 'SHIVAN/analysis/paper_plots')
_AU  = 1.496e13
_kpc = 3.086e21

snaps = sorted(glob.glob(os.path.join(CUTOUT_DIR, 'snapshot_*.hdf5')))
print(f"Total snapshots: {len(snaps)}")

# Pick 6 evenly-spaced snapshots
n_pick  = 6
indices = np.linspace(0, len(snaps) - 1, n_pick, dtype=int)
chosen  = [snaps[i] for i in indices]

fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=False)
axes = axes.ravel()

# Log bins in kpc — will be set after seeing data range
# Use a wide range; x-axis autoscales
_r_edges_kpc = np.logspace(-7, 3, 120)   # 1e-7 to 1e3 kpc
_r_ctr_kpc   = np.sqrt(_r_edges_kpc[:-1] * _r_edges_kpc[1:])

for ax, snap_path in zip(axes, chosen):
    label = os.path.basename(snap_path).replace('snapshot_', '').replace('.hdf5', '')
    print(f"\nLoading snap {label}...")

    with h5py.File(snap_path, 'r') as f:
        hdr    = dict(f['Header'].attrs)
        a      = float(hdr['Time'])
        hh     = float(hdr.get('HubbleParam', 0.7))
        z      = float(hdr.get('Redshift', 0.0))
        coords = f['PartType0/Coordinates'][:].astype(np.float64) * (a / hh)
        masses = f['PartType0/Masses'][:].astype(np.float64) * 1e10 / hh
        n_sinks = f['PartType5/Masses'].shape[0] if 'PartType5' in f else 0
        if n_sinks > 0:
            sk_m   = f['PartType5/Masses'][:] * 1e10 / hh
            sk_pos = f['PartType5/Coordinates'][:] * (a / hh)

    # Center: most massive sink if available, else mass-weighted gas COM
    if n_sinks > 0:
        center    = sk_pos[np.argmax(sk_m)]
        ctr_label = 'sink'
    else:
        center    = np.average(coords, weights=masses, axis=0)
        ctr_label = 'gas COM'

    r_kpc = np.linalg.norm(coords - center, axis=1)

    counts, _ = np.histogram(r_kpc, bins=_r_edges_kpc)
    m_hist, _ = np.histogram(r_kpc, bins=_r_edges_kpc, weights=masses)

    r_max_kpc = r_kpc.max()
    r90_kpc   = np.percentile(r_kpc, 90)
    r_apt_kpc = 2500.0 * _AU / _kpc   # 2500 AU in kpc
    print(f"  snap {label}: a={a:.4f}, z={z:.2f}, n_gas={len(r_kpc)}, n_sinks={n_sinks}")
    print(f"  r90={r90_kpc:.4f} kpc,  r_max={r_max_kpc:.4f} kpc,  center={ctr_label}")

    ax.step(_r_ctr_kpc, counts + 0.5, where='mid', color='steelblue', lw=1.5)
    ax2 = ax.twinx()
    ax2.step(_r_ctr_kpc, m_hist + 1e-10, where='mid', color='firebrick',
             lw=1.2, ls='--', alpha=0.7)
    ax2.set_yscale('log')
    ax2.set_ylabel('M_gas / bin [Msun]', color='firebrick', fontsize=8)
    ax2.tick_params(axis='y', labelcolor='firebrick', labelsize=7)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('N_gas / bin')

    ax.axvline(r_apt_kpc, color='orange', lw=1, ls=':', alpha=0.8)

    info = (f'snap {label}  z={z:.2f}  a={a:.4f}\n'
            f'n_gas={len(r_kpc)}  n_sinks={n_sinks}\n'
            f'r_max={r_max_kpc:.3f} kpc  ctr={ctr_label}')
    ax.text(0.97, 0.97, info, transform=ax.transAxes,
            ha='right', va='top', fontsize=7.5, family='monospace',
            bbox=dict(fc='white', alpha=0.7, ec='none'))

    handles = [
        matplotlib.lines.Line2D([0], [0], color='steelblue', lw=1.5, label='N_gas / bin'),
        matplotlib.lines.Line2D([0], [0], color='firebrick', lw=1.2, ls='--', label='M_gas / bin'),
        matplotlib.lines.Line2D([0], [0], color='orange', lw=1, ls=':', label='r=2500 AU'),
    ]
    ax.legend(handles=handles, fontsize=7, loc='upper left')

plt.tight_layout()
out_path = os.path.join(OUTDIR, 'gas_radial_distribution.png')
plt.savefig(out_path, dpi=100)
print(f"\nSaved: {out_path}")
