"""
plot_sink_history.py
---------------------
Track individual sink particles (PartType5) across all cutout snapshots
using (ParticleID, ParticleChildIDsNumber) as the unique key.

Produces:
  - mass_evolution_individual.png  : M(t) per sink
  - accretion_individual.png       : dM/dt(t) per sink (smoothed)
  - sink_position_history.png      : r(t) per sink as scatter + heatmap
  - sink_data.npz                  : merge/formation events for heatmap overlays

Usage (standalone or called from generate_paper_plots.py):
  python disk_analysis/plot_sink_history.py [--outdir /path/to/outdir]
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.ndimage import gaussian_filter1d

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from plot_style import apply_style

# ── Path setup ────────────────────────────────────────────────────────────────
_ANALYSIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

try:
    from notebooks.make_disk_movie_frames import _save_fig_dual, _scale_to_Myr
except ImportError:
    def _save_fig_dual(fig, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _scale_to_Myr(a):
        # Very rough fallback; prefer GUAC version
        return float(a) * 13800.0

# ── Cosmological scale-factor to time helper ─────────────────────────────────
_T1_MYR = None  # set once the first sink is found

def _a_to_Myr(a):
    """Convert GIZMO scale factor to physical time in Myr via GUAC if available."""
    try:
        return _scale_to_Myr(float(a))
    except Exception:
        return float(a) * 13800.0


# ── Physical constants ────────────────────────────────────────────────────────
_kpc_cm = 3.086e21   # cm per kpc
_AU_cm  = 1.496e13   # cm per AU
_Msun_g = 1.989e33   # g per Msun

# GIZMO code units after convert_units_to_physical:
#   Coordinates: physical kpc
#   Masses     : 10^10 Msun  (× 1e10 to get Msun)
# But for cutout snapshots read raw here (no GUAC), coordinates are in
# comoving kpc/h and masses in 10^10 Msun/h. We convert below.


def _load_snapshot_sinks(snap_path):
    """Return dict with sink data from a snapshot, or None if no sinks."""
    with h5py.File(snap_path, 'r') as f:
        if 'PartType5' not in f:
            return None
        pt5 = f['PartType5']
        if pt5['Masses'].shape[0] == 0:
            return None

        hdr  = dict(f['Header'].attrs)
        a    = float(hdr['Time'])        # scale factor
        h    = float(hdr.get('HubbleParam', 0.7))
        z    = float(hdr.get('Redshift', 0.0))

        masses = pt5['Masses'][:] * 1e10 / h   # Msun
        coords = pt5['Coordinates'][:]           # comoving kpc/h
        # Convert to physical kpc
        coords_phys = coords * a / h             # physical kpc

        pids = pt5['ParticleIDs'][:]             # uint64
        cids_raw = pt5['ParticleChildIDsNumber'][:] if 'ParticleChildIDsNumber' in pt5 else None
        cids = cids_raw if cids_raw is not None else np.zeros(len(pids), dtype=np.int64)

        # StellarFormationTime (scale factor of formation)
        if 'StellarFormationTime' in pt5:
            form_a = pt5['StellarFormationTime'][:]
        else:
            form_a = np.full(len(pids), a)

        return {
            'a': a,
            't_Myr': _a_to_Myr(a),
            'h': h,
            'masses': masses,           # Msun
            'coords': coords_phys,      # physical kpc
            'pids': pids,
            'cids': cids,
            'form_a': form_a,
        }


def _compute_r_AU(coords_kpc, center_kpc):
    """3D distance from center, in AU."""
    delta = coords_kpc - center_kpc
    return np.linalg.norm(delta, axis=1) * _kpc_cm / _AU_cm


def run(cutout_dir, outdir, r_max_AU=2500, fullsim_dir=None):
    """Main routine: scan all cutout snaps, track sinks, produce plots.

    Parameters
    ----------
    r_max_AU : float
        Maximum radius in AU to include in plots.  Sinks outside this radius
        are excluded so that far-flung ejected sinks don't dominate the axes.
        Default 2063 AU corresponds to r_max = 1e-5 kpc at z~21.
    """

    snap_paths = sorted(glob.glob(os.path.join(cutout_dir, 'snapshot_*.hdf5')))
    if not snap_paths:
        print(f'No snapshots found in {cutout_dir}')
        return

    print(f'Found {len(snap_paths)} snapshots in {cutout_dir}')

    # ── Pass 1: collect all sink appearances ─────────────────────────────────
    # sink_records[key] = list of (t_Myr, mass_Msun, r_AU)
    sink_records = {}
    sink_form_Myr = {}   # key → formation time Myr

    prev_keys = set()
    prev_t    = None

    merge_events = []  # list of (t_merge_Myr, r_last_AU, form_Myr)

    t1_Myr = None  # first sink formation time
    first_sink_center_kpc = None   # physical kpc position of primary sink at t1

    snap_times = []  # (snap_num, t_Myr) for merge time lookup
    snap_path_map = {}  # snap_num → actual file path
    snap_n_sinks_list = []  # (t_Myr, n_sinks) for number-of-sinks vs time plot

    for snap_path in snap_paths:
        snap_num = int(os.path.basename(snap_path)
                       .replace('snapshot_', '').replace('.hdf5', ''))
        snap_path_map[snap_num] = snap_path
        data = _load_snapshot_sinks(snap_path)

        if data is None:
            # No sinks in this snapshot — still record the time from the header
            try:
                with h5py.File(snap_path, 'r') as _f:
                    _a = float(_f['Header'].attrs['Time'])
                    _t = _a_to_Myr(_a)
            except Exception:
                _t = None
            if prev_keys:
                # All previous sinks have merged (shouldn't happen in practice)
                pass
            snap_times.append((snap_num, _t))
            if _t is not None:
                snap_n_sinks_list.append((_t, 0))
            prev_keys = set()
            prev_t = None
            continue

        t = data['t_Myr']
        snap_times.append((snap_num, t))

        keys = list(zip(data['pids'].tolist(), data['cids'].tolist()))
        current_keys = set(keys)

        # Find disk center: most massive sink (matches make_disk_movie_frames.py)
        idx_max = np.argmax(data['masses'])
        center_kpc = data['coords'][idx_max]
        r_AU = _compute_r_AU(data['coords'], center_kpc)

        for idx, key in enumerate(keys):
            m = data['masses'][idx]
            r = r_AU[idx]
            f_a = data['form_a'][idx]
            f_t = _a_to_Myr(f_a)

            if key not in sink_records:
                sink_records[key] = []
                sink_form_Myr[key] = f_t
                if t1_Myr is None or f_t < t1_Myr:
                    t1_Myr = f_t
                    first_sink_center_kpc = center_kpc.copy()

            sink_records[key].append((t, m, r))

        # Detect mergers: keys present previously but absent now
        if prev_keys:
            merged_keys = prev_keys - current_keys
            for mk in merged_keys:
                # last appearance was at prev_t
                if mk in sink_records and sink_records[mk]:
                    _, last_m, last_r = sink_records[mk][-1]
                    merge_events.append({
                        't_merge_Myr': t,
                        'r_AU': last_r,
                        'form_Myr': sink_form_Myr.get(mk, t1_Myr or t),
                    })

        prev_keys = current_keys
        prev_t = t
        snap_n_sinks_list.append((t, len(current_keys)))

    if t1_Myr is None:
        print('No sinks found in any snapshot.')
        return

    print(f'  Found {len(sink_records)} unique sink particles')
    print(f'  First sink formation: t₁ = {t1_Myr:.6f} Myr')
    print(f'  Merger events: {len(merge_events)}')

    # ── Build arrays per sink ─────────────────────────────────────────────────
    sink_keys = sorted(sink_records.keys(),
                       key=lambda k: sink_form_Myr.get(k, 0))

    # Also collect all (t, r) for position history heatmap — filtered to disk
    all_t_kyr = []
    all_r_AU  = []

    sink_series = {}  # key → (t_kyr_arr, m_arr, r_arr)  [r already clipped to r_max_AU]
    for key in sink_keys:
        recs = np.array(sink_records[key])  # (N, 3): t_Myr, mass, r_AU
        t_kyr = (recs[:, 0] - t1_Myr) * 1e3
        m_arr = recs[:, 1]
        r_arr = recs[:, 2]
        sink_series[key] = (t_kyr, m_arr, r_arr)
        # Only include (t, r) pairs within disk radius for the heatmap
        in_disk = r_arr <= r_max_AU
        all_t_kyr.extend(t_kyr[in_disk].tolist())
        all_r_AU.extend(r_arr[in_disk].tolist())

    all_t_kyr = np.array(all_t_kyr)
    all_r_AU  = np.array(all_r_AU)

    # Formation and merge arrays
    form_times_kyr = np.array([(sink_form_Myr[k] - t1_Myr) * 1e3 for k in sink_keys])

    if merge_events:
        merge_times_kyr = np.array([(e['t_merge_Myr'] - t1_Myr) * 1e3 for e in merge_events])
        merge_r_AU      = np.array([e['r_AU'] for e in merge_events])
    else:
        merge_times_kyr = np.array([])
        merge_r_AU      = np.array([])

    # Formation radii: r at first snapshot
    form_r_AU = np.array([sink_series[k][2][0] if len(sink_series[k][2]) > 0 else np.nan
                          for k in sink_keys])

    # ── Save sink_data.npz ────────────────────────────────────────────────────
    os.makedirs(os.path.join(outdir, 'light'), exist_ok=True)
    npz_path = os.path.join(outdir, 'sink_data.npz')

    # Per-sink series (filtered to disk radius) as numbered keys so no pickle needed
    per_sink_extra = {'n_sink_series': np.array([len(sink_keys)])}
    for i, key in enumerate(sink_keys):
        t_kyr_i, _, r_arr_i = sink_series[key]
        in_disk_i = r_arr_i <= r_max_AU
        per_sink_extra[f'sink_t_{i}'] = t_kyr_i[in_disk_i]
        per_sink_extra[f'sink_r_{i}'] = r_arr_i[in_disk_i]

    np.savez(npz_path,
             merge_times_kyr=merge_times_kyr,
             merge_r_AU=merge_r_AU,
             form_times_kyr=form_times_kyr,
             form_r_AU=form_r_AU,
             pos_t_kyr=all_t_kyr,
             pos_r_AU=all_r_AU,
             **per_sink_extra)
    print(f'  Saved {npz_path}')

    # ── Color cycle for sinks ─────────────────────────────────────────────────
    cmap_sinks = plt.colormaps.get_cmap('tab20')
    n_sinks = len(sink_keys)

    def _sink_color(i):
        return cmap_sinks(i % 20)

    # ── Total stellar mass M_total(t) ──────────────────────────────────────────
    _all_times_Myr = sorted(set(
        rec[0] for recs in sink_records.values() for rec in recs
    ))
    _total_t_kyr = np.array([(t - t1_Myr) * 1e3 for t in _all_times_Myr])
    _total_m = np.zeros(len(_all_times_Myr))
    for j_t, t_Myr in enumerate(_all_times_Myr):
        for key in sink_keys:
            recs = np.array(sink_records[key])
            idx = np.searchsorted(recs[:, 0], t_Myr)
            if idx > 0 and idx <= len(recs):
                _total_m[j_t] += recs[min(idx, len(recs) - 1), 1]

    # ═══════════════════════════════════════════════════════════════════════════
    # Plot 1: M(t) per sink
    # ═══════════════════════════════════════════════════════════════════════════
    fig1, ax1 = plt.subplots(figsize=(12, 9))
    fig1.patch.set_facecolor('w')
    ax1.set_facecolor('w')

    for i, key in enumerate(sink_keys):
        t_kyr, m_arr, _ = sink_series[key]
        ax1.plot(t_kyr, m_arr, color=_sink_color(i), lw=2.0, alpha=0.85,
                 label=f'sink {i+1}' if n_sinks <= 10 else None)

    ax1.plot(_total_t_kyr, _total_m, color='k', lw=3.0, ls='-', zorder=5,
             label=r'$M_{\rm total}$')

    # Merge markers: red "x" at last known (t, m) for each merged sink
    last_snap_t = max(recs[-1][0] for recs in sink_records.values())
    _first_merge = True
    for key in sink_keys:
        recs = sink_records[key]
        if recs[-1][0] < last_snap_t - 1e-10:
            t_last_kyr = (recs[-1][0] - t1_Myr) * 1e3
            m_last = recs[-1][1]
            ax1.scatter(t_last_kyr, m_last, marker='x', s=100, color='red',
                       linewidths=2, zorder=6,
                       label='merger' if _first_merge else None)
            _first_merge = False

    ax1.set_xlabel(r'$\Delta t$ (kyr)', color='k', fontsize=27)
    ax1.set_ylabel(r'$M_*$ ($M_\odot$)', color='k', fontsize=27)
    ax1.tick_params(colors='k', which='both', direction='in', right=True, top=True)
    for sp in ax1.spines.values(): sp.set_edgecolor('k')
    ax1.legend(fontsize=44, facecolor='w', edgecolor='k')
    ax1.margins(x=0)

    _save_fig_dual(fig1, os.path.join(outdir, 'light', 'mass_evolution_individual.png'))
    print('  Saved mass_evolution_individual.png')

    # ═══════════════════════════════════════════════════════════════════════════
    # Plot 1b: M(t) per sink — log y-axis
    # ═══════════════════════════════════════════════════════════════════════════
    fig1b, ax1b = plt.subplots(figsize=(12, 9))
    fig1b.patch.set_facecolor('w')
    ax1b.set_facecolor('w')

    for i, key in enumerate(sink_keys):
        t_kyr, m_arr, _ = sink_series[key]
        ax1b.plot(t_kyr, m_arr, color=_sink_color(i), lw=2.0, alpha=0.85,
                  label=f'sink {i+1}' if n_sinks <= 10 else None)

    ax1b.plot(_total_t_kyr, _total_m, color='k', lw=3.0, ls='-', zorder=5,
              label=r'$M_{\rm total}$')

    # Merge markers
    _first_merge_b = True
    for key in sink_keys:
        recs = sink_records[key]
        if recs[-1][0] < last_snap_t - 1e-10:
            t_last_kyr = (recs[-1][0] - t1_Myr) * 1e3
            m_last = recs[-1][1]
            ax1b.scatter(t_last_kyr, m_last, marker='x', s=100, color='red',
                        linewidths=2, zorder=6,
                        label='merger' if _first_merge_b else None)
            _first_merge_b = False

    ax1b.set_yscale('log')
    ax1b.set_xlabel(r'$\Delta t$ (kyr)', color='k', fontsize=27)
    ax1b.set_ylabel(r'$M_*$ ($M_\odot$)', color='k', fontsize=27)
    ax1b.tick_params(colors='k', which='both', direction='in', right=True, top=True)
    for sp in ax1b.spines.values(): sp.set_edgecolor('k')
    ax1b.legend(fontsize=44, facecolor='w', edgecolor='k')
    ax1b.margins(x=0)

    _save_fig_dual(fig1b, os.path.join(outdir, 'light', 'mass_evolution_individual_log.png'))
    print('  Saved mass_evolution_individual_log.png')

    # ═══════════════════════════════════════════════════════════════════════════
    # Plot 1c: M(t) per sink — log-log axes
    # ═══════════════════════════════════════════════════════════════════════════
    apply_style('fig_17')
    _lw18 = plt.rcParams['lines.linewidth']
    fig1c, ax1c = plt.subplots(figsize=(12, 9))
    fig1c.patch.set_facecolor('w')
    ax1c.set_facecolor('w')

    for i, key in enumerate(sink_keys):
        t_kyr, m_arr, _ = sink_series[key]
        # Log-log requires t > 0; skip the formation point (t=0) for each sink
        pos_t = t_kyr > 0
        if pos_t.sum() < 2:
            continue
        ax1c.plot(t_kyr[pos_t], m_arr[pos_t], color=_sink_color(i), lw=_lw18 * 0.7, alpha=0.85,
                  label=f'sink {i+1}' if n_sinks <= 10 else None)

    _pos_total = _total_t_kyr > 0
    if _pos_total.sum() > 1:
        ax1c.plot(_total_t_kyr[_pos_total], _total_m[_pos_total],
                  color='k', lw=_lw18, ls='-', zorder=5, label=r'$M_{\rm total}$')

    ax1c.set_xscale('log')
    ax1c.set_yscale('log')
    # x-axis: min 1 kyr; y-axis: min 0.1 Msun
    _all_t_pos = []
    for key in sink_keys:
        t_kyr, m_arr, _ = sink_series[key]
        _all_t_pos.extend(t_kyr[t_kyr > 0].tolist())
    if _all_t_pos:
        _t_min_loglog = max(1.0, min(_all_t_pos) * 0.8)
        _t_max_loglog = max(_all_t_pos) * 1.1
        ax1c.set_xlim(_t_min_loglog, _t_max_loglog)
        # Reference m ∝ t² line anchored at (1 kyr, 1 Msun)
        _t_ref = np.logspace(np.log10(_t_min_loglog), np.log10(_t_max_loglog), 100)
        ax1c.plot(_t_ref, 1.0 * (_t_ref / 1.0)**2, 'k--', lw=_lw18 * 0.85,
                  alpha=0.5, label=r'$m \propto t^2$')
    ax1c.set_ylim(bottom=0.1)
    ax1c.set_xlabel(r'$\Delta t$ (kyr)', color='k')
    ax1c.set_ylabel(r'$M_*$ ($M_\odot$)', color='k')
    ax1c.tick_params(colors='k', which='both', direction='in', right=True, top=True)
    for sp in ax1c.spines.values(): sp.set_edgecolor('k')
    ax1c.legend(fontsize=plt.rcParams['legend.fontsize'], facecolor='w', edgecolor='k')
    ax1c.margins(0)

    _save_fig_dual(fig1c, os.path.join(outdir, 'light', 'mass_evolution_individual_loglog.png'))
    print('  Saved mass_evolution_individual_loglog.png')

    # ═══════════════════════════════════════════════════════════════════════════
    # Plot 2: dM/dt(t) per sink
    # ═══════════════════════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(12, 9))
    fig2.patch.set_facecolor('w')
    ax2.set_facecolor('w')

    for i, key in enumerate(sink_keys):
        t_kyr, m_arr, _ = sink_series[key]
        if len(t_kyr) < 3:
            continue
        dt_yr = np.diff(t_kyr) * 1e3   # kyr → yr
        dm    = np.diff(m_arr)          # Msun
        dmdot = dm / np.maximum(dt_yr, 1e-10)  # Msun/yr
        t_mid = 0.5 * (t_kyr[:-1] + t_kyr[1:])

        # Smooth with Gaussian filter (σ=2 steps)
        if len(dmdot) > 4:
            dmdot = gaussian_filter1d(dmdot, sigma=2)

        ax2.plot(t_mid, np.maximum(dmdot, 0), color=_sink_color(i), lw=2.0, alpha=0.85)

    ax2.set_yscale('log')
    ax2.set_xlabel(r'$\Delta t$ (kyr)', color='k', fontsize=27)
    ax2.set_ylabel(r'$\dot{M}$ ($M_\odot$ yr$^{-1}$)', color='k', fontsize=27)
    ax2.tick_params(colors='k', which='both', direction='in', right=True, top=True)
    for sp in ax2.spines.values(): sp.set_edgecolor('k')

    _save_fig_dual(fig2, os.path.join(outdir, 'light', 'accretion_individual.png'))
    print('  Saved accretion_individual.png')

    # ═══════════════════════════════════════════════════════════════════════════
    # Plot 3: r(t) sink position history
    # ═══════════════════════════════════════════════════════════════════════════
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 5))
    for ax in (ax3a, ax3b):
        ax.set_facecolor('w')
    fig3.patch.set_facecolor('w')

    # Left: line plot per sink — clip r to r_max_AU
    for i, key in enumerate(sink_keys):
        t_kyr, _, r_arr = sink_series[key]
        in_disk = r_arr <= r_max_AU
        if in_disk.any():
            ax3a.plot(t_kyr[in_disk], r_arr[in_disk], color=_sink_color(i), lw=2.0, alpha=0.8)

    # Formation markers (only those within disk)
    fm_in = form_r_AU <= r_max_AU
    if fm_in.any():
        ax3a.scatter(form_times_kyr[fm_in], form_r_AU[fm_in],
                     marker='*', s=60, color='gold', edgecolors='k', linewidths=0.5,
                     zorder=5, label='formation')
    # Merge markers (only those within disk)
    mg_in = merge_r_AU <= r_max_AU if len(merge_r_AU) > 0 else np.array([], dtype=bool)
    if mg_in.any():
        ax3a.scatter(merge_times_kyr[mg_in], merge_r_AU[mg_in],
                     marker='x', s=60, color='red', linewidths=1.5,
                     zorder=5, label='merger')
    ax3a.set_xlim(left=0)
    ax3a.set_ylim(0, r_max_AU)
    ax3a.set_xlabel(r'$\Delta t$ (kyr)', color='k', fontsize=27)
    ax3a.set_ylabel('r (AU)', color='k', fontsize=27)
    ax3a.legend(fontsize=54, facecolor='w', edgecolor='k')
    ax3a.tick_params(colors='k', which='both', direction='in', right=True, top=True)
    for sp in ax3a.spines.values(): sp.set_edgecolor('k')

    # Right: 2D histogram (position density) — already filtered to r_max_AU
    if len(all_t_kyr) > 0:
        t_range = (all_t_kyr.min(), all_t_kyr.max())
        r_range = (0, r_max_AU)
        H_pos, t_edges, r_edges = np.histogram2d(
            all_t_kyr, all_r_AU,
            bins=[80, 60],
            range=[t_range, r_range])
        H_pos = np.where(H_pos > 0, H_pos, np.nan)
        Tg, Rg = np.meshgrid(t_edges, r_edges, indexing='ij')
        im = ax3b.pcolormesh(Tg, Rg, H_pos,
                             norm=colors.LogNorm(vmin=0.5, vmax=float(np.nanmax(H_pos))),
                             cmap='hot_r', rasterized=True)
        plt.colorbar(im, ax=ax3b, label='N snapshots per cell')

        # Overlay formation and merger markers (filtered to disk)
        if fm_in.any():
            ax3b.scatter(form_times_kyr[fm_in], form_r_AU[fm_in],
                         marker='*', s=60, color='gold', edgecolors='k', linewidths=0.5, zorder=5)
        if len(merge_times_kyr) > 0 and mg_in.any():
            ax3b.scatter(merge_times_kyr[mg_in], merge_r_AU[mg_in],
                         marker='x', s=60, color='cyan', linewidths=1.5, zorder=5)

    ax3b.set_ylim(0, r_max_AU)
    ax3b.set_xlabel(r'$\Delta t$ (kyr)', color='k', fontsize=27)
    ax3b.set_ylabel('r (AU)', color='k', fontsize=27)
    ax3b.tick_params(colors='k', which='both', direction='in', right=True, top=True)
    for sp in ax3b.spines.values(): sp.set_edgecolor('k')

    fig3.tight_layout()
    _save_fig_dual(fig3, os.path.join(outdir, 'light', 'sink_position_history.png'))
    print('  Saved sink_position_history.png')

    # ═══════════════════════════════════════════════════════════════════════════
    # Plot 4: Number of sinks vs time
    # ═══════════════════════════════════════════════════════════════════════════
    if snap_n_sinks_list:
        _ns_arr = np.array(sorted(snap_n_sinks_list, key=lambda x: x[0]))
        _ns_t   = (_ns_arr[:, 0] - t1_Myr) * 1e3   # kyr relative to t1
        _ns_n   = _ns_arr[:, 1]

        # ── Lin-lin plot ──
        fig4, ax4 = plt.subplots(figsize=(12, 9))
        fig4.patch.set_facecolor('w')
        ax4.set_facecolor('w')
        ax4.step(_ns_t, _ns_n, where='post', color='#1f77b4', lw=3.6)
        ax4.set_xlabel(r'$\Delta t$ (kyr)', color='k', fontsize=27)
        ax4.set_ylabel('Number of sink particles', color='k', fontsize=27)
        ax4.set_xlim(left=1.0)  # start at 1 kyr
        ax4.tick_params(colors='k', which='both', direction='in', right=True, top=True)
        ax4.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        for sp in ax4.spines.values(): sp.set_edgecolor('k')
        _save_fig_dual(fig4, os.path.join(outdir, 'light', 'sink_count_history.png'))
        print('  Saved sink_count_history.png')

        # ── Log-log plot (t >= 1 kyr, N > 0) + t^1.5 power law ──
        _pos = (_ns_t >= 1.0) & (_ns_n > 0)
        if _pos.sum() >= 2:
            apply_style('fig_11')
            _lw12 = plt.rcParams['lines.linewidth']
            fig4b, ax4b = plt.subplots(figsize=(12, 9))
            fig4b.patch.set_facecolor('w')
            ax4b.set_facecolor('w')
            ax4b.step(_ns_t[_pos], _ns_n[_pos], where='post', color='#1f77b4', lw=_lw12)
            ax4b.set_xscale('log'); ax4b.set_yscale('log')
            # Overlay t^1.5 power law: scale to match data at t=1 kyr
            _t_pl = np.logspace(0, np.log10(_ns_t[_pos].max()), 200)
            _N_at_1 = float(_ns_n[_pos][0]) if float(_ns_n[_pos][0]) > 0 else 1.0
            ax4b.plot(_t_pl, _N_at_1 * _t_pl**1.5, color='k', ls='--', lw=_lw12,
                      label=r'$N \propto t^{1.5}$')
            ax4b.legend(loc='lower right', fontsize=plt.rcParams['legend.fontsize'], framealpha=0.8)
            ax4b.set_xlabel(r'$\Delta t$ (kyr)', color='k')
            ax4b.set_ylabel('Number of sink particles', color='k')
            ax4b.tick_params(colors='k', which='both', direction='in', right=True, top=True)
            for sp in ax4b.spines.values(): sp.set_edgecolor('k')
            _save_fig_dual(fig4b, os.path.join(outdir, 'light', 'sink_count_history_loglog.png'))
            print('  Saved sink_count_history_loglog.png')

    # ═══════════════════════════════════════════════════════════════════════════
    # Pre-sink radial profile plots
    # ═══════════════════════════════════════════════════════════════════════════
    _plot_presink_profiles(snap_times, snap_path_map, t1_Myr,
                          first_sink_center_kpc, fullsim_dir or cutout_dir, outdir)

    print('plot_sink_history.py done.')
    return {
        'merge_times_kyr': merge_times_kyr,
        'merge_r_AU': merge_r_AU,
        'form_times_kyr': form_times_kyr,
        'form_r_AU': form_r_AU,
        'pos_t_kyr': all_t_kyr,
        'pos_r_AU': all_r_AU,
    }


# ── Pre-sink profile helpers ──────────────────────────────────────────────────

def _find_snap_path(snap_dir, snap_num, snap_path_map=None):
    """Locate snapshot file, trying common zero-padding formats."""
    # First try the stored actual path (different base dir, same snap_num)
    if snap_path_map and snap_num in snap_path_map:
        orig = snap_path_map[snap_num]
        # If loading from a different dir, reconstruct filename only
        orig_basename = os.path.basename(orig)
        candidate = os.path.join(snap_dir, orig_basename)
        if os.path.exists(candidate):
            return candidate
    # Try common formats
    for fmt in (f'snapshot_{snap_num:04d}.hdf5',
                f'snapshot_{snap_num:03d}.hdf5',
                f'snapshot_{snap_num}.hdf5'):
        p = os.path.join(snap_dir, fmt)
        if os.path.exists(p):
            return p
    return None


def _compute_presink_radial_profile(snap_path, r_kpc_edges, ref_center_kpc):
    """Load snapshot, return (t_Myr, rho_prof [g/cm³], mshell_prof [Msun]).

    Loads all PartType0 coordinates to build a spatial mask, then fancy-indexes
    the fields for only the nearby particles.  Works for both cutout (~4k particles)
    and full-sim (~81M) snapshots.
    """
    N_BINS   = len(r_kpc_edges) - 1
    r_max_kpc = r_kpc_edges[-1] * 2.0   # generous margin

    try:
        with h5py.File(snap_path, 'r') as f:
            if 'PartType0' not in f or f['PartType0/Masses'].shape[0] == 0:
                return None
            hdr = dict(f['Header'].attrs)
            a   = float(hdr['Time'])
            h   = float(hdr.get('HubbleParam', 0.7))

            # Load all coordinates (1–2 GB for full sim; fast on cluster FS)
            coords_all = f['PartType0/Coordinates'][:].astype(np.float64) * (a / h)
            r3d_all    = np.linalg.norm(coords_all - ref_center_kpc, axis=1)
            del coords_all

            idx = np.where(r3d_all < r_max_kpc)[0]
            if len(idx) == 0:
                return _a_to_Myr(a), np.full(N_BINS, np.nan), np.zeros(N_BINS)

            r3d      = r3d_all[idx];  del r3d_all
            mass     = f['PartType0/Masses'][idx].astype(np.float64) * 1e10 / h
            rho_code = f['PartType0/Density'][idx].astype(np.float64)
            rho      = rho_code * (1e10 * h**2 / a**3) * _Msun_g / _kpc_cm**3

    except Exception as e:
        print(f'    load error {os.path.basename(snap_path)}: {e}')
        return None

    bidx  = np.searchsorted(r_kpc_edges, r3d, side='right') - 1
    valid = (bidx >= 0) & (bidx < N_BINS)

    rho_prof    = np.full(N_BINS, np.nan)
    mshell_prof = np.zeros(N_BINS)

    for b in range(N_BINS):
        mb = valid & (bidx == b)
        if mb.sum() > 0:
            w = mass[mb]
            rho_prof[b]    = np.dot(rho[mb], w) / w.sum()
            mshell_prof[b] = w.sum()

    return _a_to_Myr(a), rho_prof, mshell_prof


def _plot_presink_profiles(snap_times, snap_path_map, t1_Myr,
                           ref_center_kpc, snap_dir, outdir):
    """Plot pre-sink radial profiles: ρ(r), M_shell(r), dm/dt(r) for 5 epochs.

    Parameters
    ----------
    snap_path_map : dict, snap_num → original file path (used for filename format)
    ref_center_kpc : array (3,) — physical kpc center (first-sink position at t1)
    snap_dir : directory containing snapshots to use for gas loading (full sim)
    """
    if ref_center_kpc is None:
        print('  No reference center available — skipping presink profiles')
        return

    # Collect pre-sink (snap_num, t_Myr) pairs
    presink = [(sn, t) for sn, t in snap_times if t is not None and t < t1_Myr - 1e-9]
    if len(presink) < 5:
        print(f'  Only {len(presink)} pre-sink snaps with known times — skipping presink profiles')
        return

    print(f'  {len(presink)} pre-sink snapshots; using {snap_dir} for gas profiles')

    # 5 evenly spaced indices
    sel_idx  = np.round(np.linspace(0, len(presink) - 1, 5)).astype(int)
    sel_info = [presink[i] for i in sel_idx]

    # Radial bins: log-spaced 10–500,000 AU
    N_BINS       = 40
    r_AU_edges   = np.logspace(np.log10(10.0), np.log10(500000.0), N_BINS + 1)
    r_AU_ctrs    = np.sqrt(r_AU_edges[:-1] * r_AU_edges[1:])   # geometric centres
    r_kpc_edges  = r_AU_edges * _AU_cm / _kpc_cm

    # Load selected snaps + immediate neighbours for finite-diff dm/dt
    presink_snums = [si[0] for si in presink]
    cache = {}   # snap_num → (t_Myr, rho_prof, mshell_prof)

    for i, (sn, _) in enumerate(sel_info):
        pos = presink_snums.index(sn)
        for offset in (-1, 0, 1):
            j = pos + offset
            if 0 <= j < len(presink_snums):
                sn_j = presink_snums[j]
                if sn_j not in cache:
                    path = _find_snap_path(snap_dir, sn_j, snap_path_map)
                    if path:
                        print(f'    Loading snap {sn_j} from {os.path.basename(path)}...', flush=True)
                        result = _compute_presink_radial_profile(path, r_kpc_edges, ref_center_kpc)
                        if result is not None:
                            cache[sn_j] = result
                            print(f'      t={result[0]:.3f} Myr  n_bins_with_mass={int((result[2]>0).sum())}')
                        else:
                            print(f'      FAILED to load profile')
                    else:
                        print(f'    snap {sn_j} not found in {snap_dir}')

    # Build arrays for plotting
    rho_list    = []
    mshell_list = []
    dmdt_list   = []
    t_list      = []

    for sn, t_nominal in sel_info:
        pos = presink_snums.index(sn)
        sn_lo = presink_snums[max(0, pos - 1)]
        sn_hi = presink_snums[min(len(presink_snums) - 1, pos + 1)]

        # density and shell mass from this snap
        if sn in cache:
            t_s, rho_s, m_s = cache[sn]
        else:
            t_s, rho_s, m_s = t_nominal, np.full(N_BINS, np.nan), np.zeros(N_BINS)
        rho_list.append(rho_s)
        mshell_list.append(m_s)
        t_list.append(t_s)

        # dm/dt from neighbours
        if sn_lo in cache and sn_hi in cache and sn_lo != sn_hi:
            t_lo, _, m_lo = cache[sn_lo]
            t_hi, _, m_hi = cache[sn_hi]
            dt_yr = (t_hi - t_lo) * 1e6   # Myr → yr
            dmdt_s = (m_hi - m_lo) / max(dt_yr, 1e-30)
        else:
            dmdt_s = np.full(N_BINS, np.nan)
        dmdt_list.append(dmdt_s)

    # ── Figure: 3 stacked panels sharing x-axis ──────────────────────────────
    cmap_pre = plt.colormaps.get_cmap('plasma')
    cols     = [cmap_pre(0.1 + 0.8 * i / 4) for i in range(5)]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10),
                             sharex=True,
                             gridspec_kw=dict(hspace=0))
    fig.patch.set_facecolor('w')

    ax_rho, ax_m, ax_dm = axes

    from scipy.ndimage import gaussian_filter1d

    for k in range(5):
        dt_t1_yr = (t_list[k] - t1_Myr) * 1e6   # negative yr (pre-sink)
        if abs(dt_t1_yr) >= 1000:
            lbl = f'{dt_t1_yr/1e3:.1f} kyr'
        else:
            lbl = f'{dt_t1_yr:.0f} yr'
        # rho
        valid_rho = np.isfinite(rho_list[k]) & (rho_list[k] > 0)
        if valid_rho.any():
            ax_rho.plot(r_AU_ctrs[valid_rho], rho_list[k][valid_rho],
                        color=cols[k], lw=3.0, label=lbl)
        # M_shell
        valid_m = mshell_list[k] > 0
        if valid_m.any():
            ax_m.plot(r_AU_ctrs[valid_m], mshell_list[k][valid_m],
                      color=cols[k], lw=3.0)
        # dm/dt: smooth along r to reduce shot noise
        dm = dmdt_list[k].copy()
        if np.isfinite(dm).sum() > 4:
            dm_smooth = gaussian_filter1d(np.where(np.isfinite(dm), dm, 0.0),
                                          sigma=1.5)
            dm = np.where(np.isfinite(dmdt_list[k]), dm_smooth, np.nan)
        valid_dm = np.isfinite(dm)
        if valid_dm.any():
            ax_dm.plot(r_AU_ctrs[valid_dm], dm[valid_dm],
                       color=cols[k], lw=3.0)

    ax_rho.set_yscale('log')
    ax_rho.set_xscale('log')
    ax_rho.set_ylabel(r'$\rho$ (g cm$^{-3}$)', fontsize=27)
    ax_rho.tick_params(colors='k', which='both', direction='in', right=True, top=True,
                       labelbottom=False)
    for sp in ax_rho.spines.values(): sp.set_edgecolor('k')
    ax_rho.legend(fontsize=48, facecolor='w', edgecolor='k', loc='upper right')

    ax_m.set_yscale('log')
    ax_m.set_ylabel(r'$M_{\rm shell}$ ($M_\odot$)', fontsize=27)
    ax_m.tick_params(colors='k', which='both', direction='in', right=True, top=True,
                     labelbottom=False)
    for sp in ax_m.spines.values(): sp.set_edgecolor('k')

    # dm/dt can be +/-; use symlog
    dmdt_all = np.concatenate([d[np.isfinite(d)] for d in dmdt_list])
    _dmdt_nonzero = dmdt_all[dmdt_all != 0] if len(dmdt_all) > 0 else np.array([])
    if len(_dmdt_nonzero) > 0:
        linthresh = max(1e-8, np.nanpercentile(np.abs(_dmdt_nonzero), 10))
        ax_dm.set_yscale('symlog', linthresh=linthresh)
    ax_dm.axhline(0, color='k', lw=1.2, ls='--', alpha=0.4)
    ax_dm.set_ylabel(r'$\dot{M}_{\rm shell}$ ($M_\odot$ yr$^{-1}$)', fontsize=27)
    ax_dm.set_xlabel(r'$r$ (AU)', fontsize=27)
    ax_dm.tick_params(colors='k', which='both', direction='in', right=True, top=True)
    for sp in ax_dm.spines.values(): sp.set_edgecolor('k')

    ax_dm.set_xlim(r_AU_ctrs[0] * 0.8, r_AU_ctrs[-1] * 1.2)

    out_path = os.path.join(outdir, 'light', 'presink_profiles.png')
    _save_fig_dual(fig, out_path)
    print(f'  Saved presink_profiles.png')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cutout-dir',
                   default='/scratch/vasissua/COPY/2026-03/m12f_cutout/output_cutout',
                   help='Directory containing cutout snapshot_*.hdf5 files (for sink tracking)')
    p.add_argument('--fullsim-dir',
                   default='/scratch/vasissua/COPY/2026-03/m12f/output_jeans_refinement',
                   help='Full-sim snapshot directory used for pre-sink gas profiles')
    p.add_argument('--outdir',
                   default='/scratch/vasissua/SHIVAN/analysis/paper_plots',
                   help='Output directory (light/ subdir will be created)')
    args = p.parse_args()
    run(args.cutout_dir, args.outdir, fullsim_dir=args.fullsim_dir)


if __name__ == '__main__':
    main()
