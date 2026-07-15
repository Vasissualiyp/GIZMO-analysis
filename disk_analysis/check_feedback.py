"""
check_feedback.py
-----------------
Check whether STARFORGE feedback (jets, radiation) is dynamically important
at each snapshot. Looks at:
  1. Sink particle fields (ProtoStellarStage, Luminosity, etc.)
  2. High-velocity gas (jet signatures): bipolar outflows along sink spin axis
  3. Mass outflow rate through spherical shells
  4. Sink ages relative to fragmentation burst

Usage:
    python check_feedback.py
Output:
    feedback_check.txt  — summary of findings
"""

import os, sys, glob
import numpy as np
import h5py

# ── Paths ────────────────────────────────────────────────────────────────────
CUTOUT_DIR = '/scratch/vasissua/COPY/2026-03/m12f_cutout/output_jeans_refinement'
FULL_DIR   = '/scratch/vasissua/COPY/2026-03/m12f/output_cutout'
OUTFILE    = '/scratch/vasissua/SHIVAN/analysis/feedback_check.txt'

# Snapshots to check: before burst, during burst, after burst, late
SNAPS = [50, 100, 150, 193, 200, 206, 210, 220, 250, 300, 350, 400, 429]

# Physical constants (CGS)
kpc_cm = 3.086e21
AU_cm  = 1.496e13
Msun_g = 1.989e33
G_cgs  = 6.674e-8
yr_s   = 3.156e7

def load_snap(snap_dir, snap_num):
    """Load a snapshot, return header dict + particle data."""
    fname = os.path.join(snap_dir, f'snapshot_{snap_num:03d}.hdf5')
    if not os.path.exists(fname):
        return None, None, None, None
    f = h5py.File(fname, 'r')
    hdr = dict(f['Header'].attrs)

    # Gas (PartType0)
    gas = {}
    if 'PartType0' in f:
        for key in ['Coordinates', 'Velocities', 'Masses', 'Density', 'SmoothingLength']:
            if key in f['PartType0']:
                gas[key] = f['PartType0'][key][:]

    # Sinks (PartType5)
    sinks = {}
    if 'PartType5' in f:
        sink_fields = list(f['PartType5'].keys())
        for key in sink_fields:
            try:
                sinks[key] = f['PartType5'][key][:]
            except Exception:
                pass
        sinks['_fields'] = sink_fields

    f.close()
    return hdr, gas, sinks, fname


def convert_units(hdr, gas, sinks):
    """Convert GIZMO code units to physical (kpc, km/s, 1e10 Msun)."""
    a = float(hdr.get('Time', 1.0))
    h = float(hdr.get('HubbleParam', 0.7))
    if a == 0:
        a = 1.0

    for data in [gas, sinks]:
        if 'Coordinates' in data:
            data['Coordinates'] = data['Coordinates'] * a / h  # physical kpc
        if 'Velocities' in data:
            data['Velocities'] = data['Velocities'] * np.sqrt(a)  # km/s
        if 'Masses' in data:
            data['Masses'] = data['Masses'] / h  # 1e10 Msun
        if 'Density' in data:
            data['Density'] = data['Density'] * h / (a/h)**3  # 1e10 Msun/kpc^3


def analyze_snapshot(snap_dir, snap_num, out):
    """Analyze one snapshot for feedback signatures."""
    hdr, gas, sinks, fname = load_snap(snap_dir, snap_num)
    if hdr is None:
        out.write(f'\n  Snap {snap_num:04d}: FILE NOT FOUND\n')
        return

    n_gas   = int(hdr.get('NumPart_ThisFile', [0]*6)[0])
    n_sinks = int(hdr.get('NumPart_ThisFile', [0]*6)[5]) if len(hdr.get('NumPart_ThisFile', [0]*6)) > 5 else 0

    out.write(f'\n{"="*70}\n')
    out.write(f'  Snap {snap_num:04d}  |  n_gas={n_gas}  n_sinks={n_sinks}\n')
    out.write(f'{"="*70}\n')

    if n_sinks == 0:
        out.write('  No sinks — skipping\n')
        return

    convert_units(hdr, gas, sinks)

    # ── 1. Sink particle fields ──────────────────────────────────────────
    out.write(f'\n  Sink fields available: {sinks.get("_fields", [])}\n')

    # Report feedback-related fields
    for field in ['ProtoStellarStage', 'ProtoStellarRadius', 'Luminosity',
                  'Luminosity_Sinks', 'BH_Specific_AngMom', 'Jet_Velocity',
                  'SinkRadius', 'BH_Mass', 'StellarFormationTime']:
        if field in sinks:
            vals = sinks[field]
            if vals.ndim == 1:
                out.write(f'    {field}: min={np.min(vals):.4g}  max={np.max(vals):.4g}  '
                          f'mean={np.mean(vals):.4g}\n')
            else:
                out.write(f'    {field}: shape={vals.shape}  '
                          f'norm_range=[{np.min(np.linalg.norm(vals,axis=1)):.4g}, '
                          f'{np.max(np.linalg.norm(vals,axis=1)):.4g}]\n')

    # Sink masses in Msun
    if 'Masses' in sinks:
        M_sinks_Msun = sinks['Masses'] * 1e10
        out.write(f'\n  Sink masses (Msun): min={np.min(M_sinks_Msun):.4g}  '
                  f'max={np.max(M_sinks_Msun):.4g}  total={np.sum(M_sinks_Msun):.4g}\n')

    # Sink formation times (if available)
    if 'StellarFormationTime' in sinks:
        form_a = sinks['StellarFormationTime']
        out.write(f'  Formation scale factors: min={np.min(form_a):.6g}  max={np.max(form_a):.6g}\n')
        current_a = float(hdr.get('Time', 1.0))
        out.write(f'  Current scale factor: {current_a:.6g}\n')
        # Relative ages (delta_a as proxy)
        delta_a = current_a - form_a
        out.write(f'  Sink ages (delta_a): min={np.min(delta_a):.4g}  max={np.max(delta_a):.4g}  '
                  f'median={np.median(delta_a):.4g}\n')
        # Sort by age
        order = np.argsort(-delta_a)
        n_show = min(5, len(order))
        out.write(f'  Oldest {n_show} sinks:\n')
        for i in range(n_show):
            idx = order[i]
            m = M_sinks_Msun[idx] if 'Masses' in sinks else 0
            out.write(f'    #{idx}: M={m:.3g} Msun  delta_a={delta_a[idx]:.4g}  '
                      f'form_a={form_a[idx]:.6g}\n')

    # ── 2. High-velocity gas (jet check) ─────────────────────────────────
    if 'Coordinates' in gas and 'Velocities' in gas and n_sinks > 0:
        # Find most massive sink as center
        if 'Masses' in sinks:
            i_main = np.argmax(sinks['Masses'])
        else:
            i_main = 0
        center = sinks['Coordinates'][i_main]
        v_center = sinks['Velocities'][i_main]

        pos_rel = gas['Coordinates'] - center  # kpc
        vel_rel = gas['Velocities'] - v_center  # km/s
        r_kpc = np.linalg.norm(pos_rel, axis=1)
        v_mag = np.linalg.norm(vel_rel, axis=1)

        # Within 500 AU of main sink
        r_AU = r_kpc * kpc_cm / AU_cm
        near = r_AU < 500
        n_near = near.sum()

        if n_near > 0:
            v_near = v_mag[near]
            out.write(f'\n  Gas within 500 AU of main sink: n={n_near}\n')
            out.write(f'    |v| (km/s): median={np.median(v_near):.2f}  '
                      f'p90={np.percentile(v_near,90):.2f}  '
                      f'p99={np.percentile(v_near,99):.2f}  '
                      f'max={np.max(v_near):.2f}\n')

            # Check for bipolar structure: decompose velocity into
            # component along z-axis (proxy for disk normal) vs in-plane
            # Use angular momentum of gas within 200 AU as disk normal
            inner = r_AU < 200
            if inner.sum() > 10 and 'Masses' in gas:
                m_inner = gas['Masses'][inner]
                p_inner = pos_rel[inner]
                v_inner = vel_rel[inner]
                L = np.sum(m_inner[:, None] * np.cross(p_inner, v_inner), axis=0)
                L_hat = L / (np.linalg.norm(L) + 1e-30)

                # For gas within 500 AU: decompose velocity
                v_par = np.sum(vel_rel[near] * L_hat, axis=1)  # along L_hat
                v_perp = np.sqrt(v_mag[near]**2 - v_par**2 + 1e-30)

                # Height above disk
                z_AU = np.sum(pos_rel[near] * L_hat, axis=1) * kpc_cm / AU_cm

                # Fast gas: |v| > 10 km/s
                fast = v_mag[near] > 10
                n_fast = fast.sum()
                out.write(f'    Fast gas (|v|>10 km/s): n={n_fast} ({100*n_fast/n_near:.1f}%)\n')
                if n_fast > 0:
                    out.write(f'      v_parallel (along L): median={np.median(np.abs(v_par[fast])):.2f}  '
                              f'max={np.max(np.abs(v_par[fast])):.2f} km/s\n')
                    out.write(f'      v_perp (in-plane):    median={np.median(v_perp[fast]):.2f}  '
                              f'max={np.max(v_perp[fast]):.2f} km/s\n')
                    out.write(f'      |z| (AU):             median={np.median(np.abs(z_AU[fast])):.1f}  '
                              f'max={np.max(np.abs(z_AU[fast])):.1f}\n')

                    # Bipolar check: is fast gas preferentially at high |z|?
                    fast_high_z = fast & (np.abs(z_AU) > 50)
                    fast_low_z  = fast & (np.abs(z_AU) < 50)
                    out.write(f'      Fast gas at |z|>50 AU: {fast_high_z.sum()}  '
                              f'at |z|<50 AU: {fast_low_z.sum()}\n')

                # Very fast gas: |v| > 30 km/s (strong jet signature)
                vfast = v_mag[near] > 30
                n_vfast = vfast.sum()
                out.write(f'    Very fast gas (|v|>30 km/s): n={n_vfast}\n')
                if n_vfast > 0:
                    out.write(f'      v_parallel: median={np.median(np.abs(v_par[vfast])):.2f}  '
                              f'max={np.max(np.abs(v_par[vfast])):.2f} km/s\n')
                    out.write(f'      |z| (AU): median={np.median(np.abs(z_AU[vfast])):.1f}  '
                              f'max={np.max(np.abs(z_AU[vfast])):.1f}\n')

    # ── 3. Mass outflow rate through shells ──────────────────────────────
    if 'Coordinates' in gas and 'Velocities' in gas and 'Masses' in gas and n_sinks > 0:
        v_r = np.sum(vel_rel * pos_rel, axis=1) / (r_kpc + 1e-30)  # radial velocity km/s
        for R_AU in [100, 200, 500]:
            R_kpc = R_AU * AU_cm / kpc_cm
            shell = (r_kpc > 0.8*R_kpc) & (r_kpc < 1.2*R_kpc)
            if shell.sum() > 0:
                m_shell = gas['Masses'][shell] * 1e10  # Msun
                vr_shell = v_r[shell]
                outflow = vr_shell > 0
                inflow  = vr_shell < 0
                # Mass flux ~ sum(m * v_r) / dr, rough estimate
                dr_kpc = 0.4 * R_kpc
                dr_cm  = dr_kpc * kpc_cm
                Mdot_out = np.sum(m_shell[outflow] * vr_shell[outflow] * 1e5) * Msun_g / dr_cm / yr_s  # Msun/yr (rough)
                Mdot_in  = np.sum(m_shell[inflow]  * np.abs(vr_shell[inflow]) * 1e5) * Msun_g / dr_cm / yr_s
                # Convert back to Msun/yr properly
                Mdot_out_msunyr = Mdot_out / Msun_g * yr_s
                Mdot_in_msunyr  = Mdot_in  / Msun_g * yr_s
                out.write(f'\n  Shell at r={R_AU} AU (n={shell.sum()}): '
                          f'Mdot_out~{Mdot_out_msunyr:.2g}  '
                          f'Mdot_in~{Mdot_in_msunyr:.2g} Msun/yr  '
                          f'out/in={Mdot_out_msunyr/(Mdot_in_msunyr+1e-30):.2f}\n')


def main():
    with open(OUTFILE, 'w') as out:
        out.write('FEEDBACK CHECK\n')
        out.write('=' * 70 + '\n')

        # Check cutout sim (has all snapshots through 429)
        out.write('\n\n>>> CUTOUT SIMULATION <<<\n')
        out.write(f'    {CUTOUT_DIR}\n')
        for s in SNAPS:
            try:
                analyze_snapshot(CUTOUT_DIR, s, out)
            except Exception as e:
                out.write(f'\n  Snap {s:04d}: ERROR — {e}\n')

        # Check full sim for a couple of snapshots
        out.write('\n\n>>> FULL SIMULATION <<<\n')
        out.write(f'    {FULL_DIR}\n')
        for s in [50, 206, 276]:
            try:
                analyze_snapshot(FULL_DIR, s, out)
            except Exception as e:
                out.write(f'\n  Snap {s:04d}: ERROR — {e}\n')

    print(f'Done. Results in {OUTFILE}')


if __name__ == '__main__':
    main()
