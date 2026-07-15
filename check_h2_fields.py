"""Check for H2/molecular fraction fields, ElectronAbundance, and halo mass in snapshots."""
import h5py, os, glob
import numpy as np

full_dir = '/scratch/vasissua/COPY/2026-03/m12f/output_jeans_refinement'
full_cutout_dir = '/scratch/vasissua/COPY/2026-03/m12f/output_cutout'
cutout_dir = '/scratch/vasissua/COPY/2026-03/m12f_cutout/output_jeans_refinement'
spatial_cutout_dir = '/scratch/vasissua/COPY/2026-03/m12f_cutout/output_cutout'


def find_snap(d, sn):
    """Try both 3-digit and 4-digit snapshot filenames."""
    for fmt in ['snapshot_%03d.hdf5', 'snapshot_%04d.hdf5']:
        p = os.path.join(d, fmt % sn)
        if os.path.exists(p):
            return p
    return None


def extract_snap_num(basename):
    """Extract snapshot number from filename like snapshot_028.hdf5."""
    return int(basename.replace('snapshot_', '').replace('.hdf5', ''))


# ── 1. Check all field names in each sim type ──
for label, d in [('FULL_JREF', full_dir), ('FULL_CUTOUT', full_cutout_dir),
                 ('CUTOUT_JREF', cutout_dir), ('SPATIAL_CUTOUT', spatial_cutout_dir)]:
    print(f'\n=== {label}: {d} ===')
    if not os.path.isdir(d):
        print('  DIRECTORY NOT FOUND')
        continue
    snaps = sorted(glob.glob(os.path.join(d, 'snapshot_*.hdf5')))
    print(f'  Total snapshots: {len(snaps)}')

    check_nums = set()
    if snaps:
        for s in [snaps[0], snaps[-1]]:
            check_nums.add(extract_snap_num(os.path.basename(s)))
        for n in [28, 50, 92, 94, 96, 98, 100, 200, 400, 600]:
            check_nums.add(n)

    for sn in sorted(check_nums):
        fp = find_snap(d, sn)
        if fp is None:
            continue
        try:
            with h5py.File(fp, 'r') as h:
                if 'PartType0' not in h:
                    print(f'  snap {sn}: no PartType0')
                    continue
                fields = sorted(h['PartType0'].keys())
                mol_fields = [k for k in fields if 'mol' in k.lower() or 'h2' in k.lower() or 'molecular' in k.lower()]
                elec_fields = [k for k in fields if 'electron' in k.lower() or 'xe' in k.lower()]
                print(f'  snap {sn}: {len(fields)} fields.')
                print(f'    Molecular: {mol_fields if mol_fields else "NONE"}')
                print(f'    Electron:  {elec_fields if elec_fields else "NONE"}')
                if sn in (28, 96, 200, 0) or sn == sorted(check_nums)[0]:
                    print(f'    All PartType0 fields: {fields}')
                    ptypes = [k for k in h.keys() if k.startswith('PartType')]
                    print(f'    Particle types: {ptypes}')
        except Exception as e:
            print(f'  snap {sn}: ERROR {e}')

# ── 2. Check ElectronAbundance values in disk gas (for L468 x_e check) ──
print('\n\n=== ElectronAbundance check (for L468: x_e >> 1e-4?) ===')
for label, d in [('FULL_CUTOUT', full_cutout_dir), ('CUTOUT_JREF', cutout_dir),
                 ('SPATIAL_CUTOUT', spatial_cutout_dir)]:
    for sn in [100, 200, 400, 600]:
        fp = find_snap(d, sn)
        if fp is None:
            continue
        try:
            with h5py.File(fp, 'r') as h:
                if 'PartType0' not in h:
                    continue
                if 'ElectronAbundance' not in h['PartType0']:
                    print(f'  {label} snap {sn}: ElectronAbundance NOT in PartType0')
                    continue
                xe = h['PartType0/ElectronAbundance'][:]
                rho = h['PartType0/Density'][:]
                print(f'  {label} snap {sn}: x_e min={xe.min():.2e}, median={np.median(xe):.2e}, max={xe.max():.2e}')
                rho_cgs = rho * 1e10 * 1.989e33 / (3.086e21)**3
                nH = rho_cgs / 1.673e-24
                high_dens = nH > 1e8
                if high_dens.any():
                    xe_hd = xe[high_dens]
                    print(f'    At nH > 1e8 cm^-3 ({high_dens.sum()} particles): '
                          f'x_e min={xe_hd.min():.2e}, median={np.median(xe_hd):.2e}, max={xe_hd.max():.2e}')
                    frac_above = (xe_hd > 1e-4).sum() / len(xe_hd) * 100
                    print(f'    Fraction with x_e > 1e-4: {frac_above:.1f}%')
        except Exception as e:
            print(f'  {label} snap {sn}: ERROR {e}')

# ── 3. Halo mass check (for L334: DM + gas mass) ──
print('\n\n=== Halo mass check (for L334) ===')
for sn in [0, 10, 28]:
    fp = find_snap(full_dir, sn)
    if fp is None:
        continue
    try:
        with h5py.File(fp, 'r') as h:
            hdr = dict(h['Header'].attrs)
            a = hdr['Time']
            hubble = hdr.get('HubbleParam', 0.702)
            print(f'\n  FULL snap {sn} (a={a:.5f}, z={1/a-1:.1f}, h={hubble}):')

            if 'PartType0' in h:
                gas_coords = h['PartType0/Coordinates'][:] * a / hubble
                gas_mass = h['PartType0/Masses'][:] * 1e10 / hubble

                if 'PartType5' in h and h['PartType5/Masses'].shape[0] > 0:
                    sink_coords = h['PartType5/Coordinates'][:] * a / hubble
                    sink_mass = h['PartType5/Masses'][:] * 1e10 / hubble
                    center = np.average(sink_coords, axis=0, weights=sink_mass)
                else:
                    rho = h['PartType0/Density'][:]
                    center = gas_coords[np.argmax(rho)]

                for r_kpc in [0.1, 0.5, 1.0]:
                    dist_gas = np.linalg.norm(gas_coords - center, axis=1)
                    m_gas = gas_mass[dist_gas < r_kpc].sum()

                    m_dm = 0
                    if 'PartType1' in h:
                        dm_coords = h['PartType1/Coordinates'][:] * a / hubble
                        dm_mass = h['PartType1/Masses'][:] * 1e10 / hubble
                        dist_dm = np.linalg.norm(dm_coords - center, axis=1)
                        m_dm = dm_mass[dist_dm < r_kpc].sum()

                    print(f'    r < {r_kpc} kpc: M_gas={m_gas:.2e} Msun, M_DM={m_dm:.2e} Msun, '
                          f'M_tot={m_gas+m_dm:.2e} Msun, f_b={m_gas/(m_gas+m_dm+1e-30):.3f}')
    except Exception as e:
        print(f'  FULL snap {sn}: ERROR {e}')

print('\nDone.')
