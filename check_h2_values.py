"""Quick check: what are the actual MolecularMassFraction values in snap 0 and snap 50/100/200?"""
import h5py, os
import numpy as np

base = '/scratch/vasissua/COPY/2026-03/m12f/output_jeans_refinement'

for sn in [0, 50, 100, 200]:
    for fmt in ['snapshot_%03d.hdf5', 'snapshot_%04d.hdf5']:
        fp = os.path.join(base, fmt % sn)
        if os.path.exists(fp):
            break
    else:
        print(f"snap {sn}: NOT FOUND")
        continue

    with h5py.File(fp, 'r') as f:
        if 'PartType0' not in f or 'MolecularMassFraction' not in f['PartType0']:
            print(f"snap {sn}: no MolecularMassFraction")
            continue
        fh2 = f['PartType0/MolecularMassFraction'][:]
        rho = f['PartType0/Density'][:]
        n_total = len(fh2)
        n_pos = (fh2 > 0).sum()
        n_sig = (fh2 > 1e-6).sum()
        print(f"snap {sn}: N={n_total}, fH2>0: {n_pos} ({100*n_pos/n_total:.1f}%), "
              f"fH2>1e-6: {n_sig} ({100*n_sig/n_total:.1f}%)")
        if n_pos > 0:
            fh2_pos = fh2[fh2 > 0]
            print(f"  fH2 (where >0): min={fh2_pos.min():.2e}, median={np.median(fh2_pos):.2e}, "
                  f"max={fh2_pos.max():.2e}")
        # Check density range
        rho_cgs = rho * 1e10 * 1.989e33 / (3.086e21)**3
        nH = rho_cgs / 1.673e-24
        print(f"  nH: min={nH.min():.2e}, median={np.median(nH):.2e}, max={nH.max():.2e} cm^-3")

print("\nDone.")
