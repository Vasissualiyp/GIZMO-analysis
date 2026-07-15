"""Quick diagnostic: print Sigma range from qprofiles for frames9 and frames10."""
import glob, numpy as np

for label, outdir in [('frames9',  '/scratch/vasissua/SHIVAN/analysis/frames9'),
                      ('frames10', '/scratch/vasissua/SHIVAN/analysis/frames10')]:
    files = sorted(glob.glob(f'{outdir}/qprofiles/qprofile_*.npz'))
    if not files:
        print(f'{label}: no npz files found')
        continue
    all_S = []
    for f in files:
        d = np.load(f)
        if 'Sigma' in d:
            S = d['Sigma']
            S = S[np.isfinite(S) & (S > 0)]
            all_S.append(S)
    if all_S:
        all_S = np.concatenate(all_S)
        pcts = np.percentile(all_S, [1, 5, 25, 50, 75, 95, 99])
        print(f'{label}: n={len(all_S)}, min={all_S.min():.3e}, max={all_S.max():.3e}')
        print(f'  percentiles 1/5/25/50/75/95/99: ' +
              ' / '.join(f'{p:.2e}' for p in pcts))
