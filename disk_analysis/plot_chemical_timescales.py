"""Chemical vs dynamical timescales using actual H2 and ionisation chemistry fields.

Loads full-sim snapshot 270 (last snapshot with MolecularMassFraction,
ElectronAbundance, Temperature) to compute per-particle:

  t_ff   = sqrt(3π / (32Gρ))                          [free-fall time]
  t_H2   = 1 / (k_H- × xe × n_H)                     [H2 formation via H- route]
  t_cool ≈ u / Λ_H2(n,T,fH2)                         [H2 line cooling time]

Key result: at n_H ≳ 10^7 cm⁻³ (disk densities) chemistry/cooling are fast
compared to dynamics (t_chem/t_ff ≪ 1), so the gas is in chemical + thermal
equilibrium and H2 formation is not a bottleneck for disk physics.

Usage (cluster):
    python disk_analysis/plot_chemical_timescales.py
"""

import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plot_style import apply_style

# ── Physical constants (CGS) ────────────────────────────────────────────────
_G    = 6.674e-8          # cm^3 g^-1 s^-2
_mH   = 1.673e-24         # g
_kB   = 1.381e-16         # erg K^-1
_yr   = 3.156e7           # s yr^-1
_XH   = 0.76              # hydrogen mass fraction (primordial)
_gm1  = 2.0 / 3.0        # γ-1 (FIRE uses 5/3 → γ-1=2/3)

# ── GIZMO code-unit conversion constants ────────────────────────────────────
_rho_unit_Msun_per_kpc3 = 6.77e-32   # g/cm^3 per (M_sun/kpc^3)

# ── Data paths ───────────────────────────────────────────────────────────────
_FULL_SIM_DIR = '/scratch/vasissua/COPY/2026-03/m12f/output_jeans_refinement'
_SNAP_NUM     = 270   # last snapshot with MolecularMassFraction + ElectronAbundance
_OUTDIR       = '/scratch/vasissua/SHIVAN/analysis/paper_plots'

# Max particles to scatter-plot (subsample beyond this to keep plot fast)
_MAX_SCATTER  = 500_000


def _load_snap(path):
    """Return dict of per-particle arrays + header from a full-sim snapshot."""
    import h5py
    with h5py.File(path, 'r') as f:
        hdr = dict(f['Header'].attrs)
        pt0 = f['PartType0']
        rho  = pt0['Density'][:]
        T    = pt0['Temperature'][:]        # stored in K directly
        xe   = pt0['ElectronAbundance'][:]  # n_e / n_H (mass-weighted)
        fH2  = pt0['MolecularMassFraction'][:]  # 2*n_H2 / n_H
    return rho, T, xe, fH2, hdr


def _rho_to_nH(rho_code, a, h):
    """Code-unit density → physical hydrogen number density (cm^-3)."""
    rho_phys = rho_code * 1e10 * _rho_unit_Msun_per_kpc3 * h**4 / a**3
    return rho_phys, rho_phys * _XH / _mH


def t_freefall(rho_phys):
    """Free-fall time in years."""
    return np.sqrt(3.0 * np.pi / (32.0 * _G * rho_phys)) / _yr


def t_H2_formation(n_H, T, xe):
    """H2 formation timescale via the H⁻ route (years).

    H + e⁻ → H⁻ + γ   k1 ≈ 1.4e-18 T  cm³/s  (T < 6000 K)
    t_H2 ≈ 1 / (k1 × n_e)   where n_e = xe × n_H
    """
    T_c = np.clip(T, 10.0, 6000.0)
    k1  = 1.4e-18 * T_c
    n_e = np.clip(xe, 1e-12, None) * n_H
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(n_e > 0, 1.0 / (k1 * n_e), np.inf)
    return t / _yr


def t_H2_cooling(n_H, T, fH2):
    """Approximate H2 line cooling timescale (years).

    Λ_H2 ≈ 1e-24 × n_H² × fH2   [erg/cm³/s] (Hollenbach & McKee 1979, low-n limit)
    t_cool = u_vol / Λ_H2   where u_vol = n_H kB T / (γ-1)
    """
    u_vol  = n_H * _kB * T / _gm1
    Lambda = 1.0e-24 * n_H**2 * np.clip(fH2, 1e-6, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(Lambda > 0, u_vol / Lambda, np.inf)
    return t / _yr


def _subsample(rng, *arrays, n_max):
    """Randomly subsample all arrays to at most n_max rows (consistent selection)."""
    n = len(arrays[0])
    if n <= n_max:
        return arrays
    idx = rng.choice(n, size=n_max, replace=False)
    return tuple(a[idx] for a in arrays)


def make_plots(rho_code, T, xe, fH2, a, h, snap_label, outdir):
    rho_phys, n_H = _rho_to_nH(rho_code, a, h)

    tff   = t_freefall(rho_phys)
    tH2   = t_H2_formation(n_H, T, xe)
    tcool = t_H2_cooling(n_H, T, fH2)

    ratio_H2   = tH2   / tff
    ratio_cool = tcool / tff

    valid = (
        (n_H > 0) & (tff > 0)
        & np.isfinite(ratio_H2) & np.isfinite(ratio_cool)
        & np.isfinite(T) & np.isfinite(fH2)
    )
    n_v   = n_H[valid]
    T_v   = T[valid]
    xe_v  = xe[valid]
    fH2_v = fH2[valid]
    rH2   = ratio_H2[valid]
    rcool = ratio_cool[valid]

    # ── Print summary before (possibly) subsampling ──────────────────────────
    print(f'\n  Snapshot {snap_label}  ({valid.sum()} valid particles)')
    print(f'  n_H range:   {n_v.min():.2e}  –  {n_v.max():.2e}  cm^-3')
    print(f'  T range:     {T_v.min():.1f}  –  {T_v.max():.1f}  K')
    print(f'  xe range:    {xe_v.min():.2e}  –  {xe_v.max():.2e}')
    print(f'  fH2 range:   {fH2_v.min():.2e}  –  {fH2_v.max():.2e}')
    for ratio, name in [(rH2, 't_H2/t_ff'), (rcool, 't_cool/t_ff')]:
        q = np.percentile(ratio, [5, 25, 50, 75, 95])
        print(f'  {name}:  p5={q[0]:.2e}  p25={q[1]:.2e}  median={q[2]:.2e}'
              f'  p75={q[3]:.2e}  p95={q[4]:.2e}')
    frac_fast_H2   = np.mean(rH2   < 1.0)
    frac_fast_cool = np.mean(rcool < 1.0)
    print(f'\n  Fraction with t_H2/t_ff < 1   (chemistry fast): {frac_fast_H2:.2%}')
    print(f'  Fraction with t_cool/t_ff < 1 (cooling fast):   {frac_fast_cool:.2%}')
    print(f'  Fraction with fH2 > 0.01:   {np.mean(fH2_v > 0.01):.2%}')
    print(f'  Fraction with fH2 > 0.1:    {np.mean(fH2_v > 0.1):.2%}')

    # ── Subsample for scatter plots ───────────────────────────────────────────
    rng = np.random.default_rng(42)
    n_v_s, T_v_s, fH2_v_s, rH2_s, rcool_s = _subsample(
        rng, n_v, T_v, fH2_v, rH2, rcool, n_max=_MAX_SCATTER)

    # ── Figure ───────────────────────────────────────────────────────────────
    apply_style('fig_3')
    _lw = plt.rcParams['lines.linewidth']
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.patch.set_facecolor('w')

    T_norm   = mcolors.LogNorm(vmin=max(T_v_s.min(), 10), vmax=max(T_v_s.max(), 11))
    log_n    = np.log10(n_v_s)

    # Panel 0: t_H2 / t_ff  vs  n_H
    ax = axes[0]
    sc0 = ax.scatter(log_n, np.log10(np.clip(rH2_s, 1e-6, 1e6)),
                     c=T_v_s, norm=T_norm, cmap='inferno',
                     s=0.5, alpha=0.3, rasterized=True)
    ax.axhline(0, color='w', lw=_lw * 0.6, ls='--')
    ax.set_xlabel(r'$\log_{10}\,n_{\rm H}\;({\rm cm}^{-3})$')
    ax.set_ylabel(r'$\log_{10}\,(t_{\rm H_2\,form} / t_{\rm ff})$')
    cb0 = fig.colorbar(sc0, ax=ax, pad=0.01)
    cb0.set_label('T (K)')

    # Panel 1: t_cool / t_ff  vs  n_H
    ax = axes[1]
    sc1 = ax.scatter(log_n, np.log10(np.clip(rcool_s, 1e-12, 1e6)),
                     c=T_v_s, norm=T_norm, cmap='inferno',
                     s=0.5, alpha=0.3, rasterized=True)
    ax.axhline(0, color='w', lw=_lw * 0.6, ls='--')
    ax.set_xlabel(r'$\log_{10}\,n_{\rm H}\;({\rm cm}^{-3})$')
    ax.set_ylabel(r'$\log_{10}\,(t_{\rm cool} / t_{\rm ff})$')
    cb1 = fig.colorbar(sc1, ax=ax, pad=0.01)
    cb1.set_label('T (K)')

    # Panel 2: fH2  vs  n_H, coloured by t_H2/t_ff
    ax = axes[2]
    ratio_norm = mcolors.LogNorm(vmin=1e-2, vmax=1e3)
    sc2 = ax.scatter(log_n, np.log10(np.clip(fH2_v_s, 1e-10, 1.0)),
                     c=np.clip(rH2_s, 1e-2, 1e3), norm=ratio_norm, cmap='RdYlBu_r',
                     s=0.5, alpha=0.3, rasterized=True)
    ax.axhline(0, color='k', lw=_lw * 0.4, ls=':')
    ax.set_xlabel(r'$\log_{10}\,n_{\rm H}\;({\rm cm}^{-3})$')
    ax.set_ylabel(r'$\log_{10}\,f_{\rm H_2}$')
    cb2 = fig.colorbar(sc2, ax=ax, pad=0.01)
    cb2.set_label(r'$t_{\rm H_2\,form} / t_{\rm ff}$')

    for ax in axes:
        ax.set_facecolor('w')
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', right=True, top=True)
        for sp in ax.spines.values(): sp.set_edgecolor('k')
        ax.text(0.03, 0.97, snap_label, transform=ax.transAxes, va='top',
                fontsize=plt.rcParams['legend.fontsize'] * 0.8,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

    os.makedirs(os.path.join(outdir, 'light'), exist_ok=True)
    out = os.path.join(outdir, 'light', 'chemical_timescales.png')
    fig.savefig(out, dpi=150, facecolor='w', bbox_inches='tight')
    fig.savefig(out.replace('.png', '.pdf'), facecolor='w', bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Saved → {out}')


def main():
    # Try 3- and 4-digit filename formats
    for fmt in ('snapshot_%03d.hdf5', 'snapshot_%04d.hdf5'):
        snap = os.path.join(_FULL_SIM_DIR, fmt % _SNAP_NUM)
        if os.path.exists(snap):
            break
    else:
        print(f'Snapshot {_SNAP_NUM} not found in {_FULL_SIM_DIR}')
        return

    print(f'Loading {snap} ...')
    rho_code, T, xe, fH2, hdr = _load_snap(snap)
    a = float(hdr['Time'])
    h = float(hdr['HubbleParam'])
    snap_label = f'snap {_SNAP_NUM}  (a={a:.4f},  z={1/a-1:.1f})'
    print(f'  a={a:.4f}  h={h:.4f}  N_gas={len(rho_code):,}')

    make_plots(rho_code, T, xe, fH2, a, h, snap_label, _OUTDIR)


if __name__ == '__main__':
    main()
