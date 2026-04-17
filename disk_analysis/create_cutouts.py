"""
create_cutouts.py
-----------------
Create spatial cutouts from full FIRE+STARFORGE snapshots.

For each snapshot, finds the center (sink COM or densest gas near a
reference point) and extracts all particles within a cutout radius.
Processes snapshots in reverse order so that the sink position from
later snapshots can guide center-finding in earlier (pre-sink) snapshots.

Usage:
    python create_cutouts.py [--cutout-radius 5e-4] [--overwrite]

Output goes to output_cutout/ alongside the original output directory.
"""

import argparse
import os
import sys
import glob
import time

import h5py
import numpy as np


def find_center_raw(f, reference_center=None, search_radius=None):
    """
    Find the center of the region of interest in a snapshot (comoving kpc/h).

    Priority:
      1. Mass-weighted COM of PartType5 (STARFORGE sinks)
      2. Densest gas within search_radius of reference_center
      3. Globally densest gas particle (fallback)
    """
    # Try sinks first
    if 'PartType5' in f and f['PartType5/Masses'].shape[0] > 0:
        mass = f['PartType5/Masses'][:]
        coords = f['PartType5/Coordinates'][:]
        com = np.sum(coords * mass[:, None], axis=0) / np.sum(mass)
        return com, 'sinks'

    # Fall back to densest gas
    coords = f['PartType0/Coordinates'][:]
    density = f['PartType0/Density'][:]

    if reference_center is not None and search_radius is not None:
        dists = np.linalg.norm(coords - reference_center, axis=1)
        mask = dists < search_radius
        if mask.sum() > 0:
            idx = np.argmax(density[mask])
            return coords[mask][idx], 'density_ref'

    # Global densest
    idx = np.argmax(density)
    return coords[idx], 'density_global'


def create_cutout(input_path, output_path, center, cutout_radius, verbose=True):
    """
    Create a cutout HDF5 file containing only particles within cutout_radius
    of center (in comoving kpc/h).
    """
    with h5py.File(input_path, 'r') as fin, h5py.File(output_path, 'w') as fout:
        # Copy header
        fin.copy('Header', fout)

        # Only include particle types used by the analysis pipeline
        # PartType0=gas, PartType3=refinement, PartType4=FIRE stars, PartType5=sinks
        # Skip PartType1 (dark matter) and PartType2 (disk stars) — not used, can be huge
        KEEP_TYPES = {'PartType0', 'PartType3', 'PartType4', 'PartType5'}
        part_types = [k for k in fin.keys() if k in KEEP_TYPES]
        numpart_total = np.array(fout['Header'].attrs['NumPart_Total'])
        numpart_thisfile = np.array(fout['Header'].attrs['NumPart_ThisFile'])

        for pt in part_types:
            if 'Coordinates' not in fin[pt]:
                continue

            coords = fin[pt + '/Coordinates'][:]
            dists = np.linalg.norm(coords - center, axis=1)
            mask = dists < cutout_radius

            n_selected = mask.sum()
            if n_selected == 0:
                continue

            grp = fout.create_group(pt)
            for field in fin[pt].keys():
                data = fin[pt][field][:]
                if data.shape[0] == len(mask):
                    grp.create_dataset(field, data=data[mask])
                else:
                    # Non-per-particle data, copy as-is
                    grp.create_dataset(field, data=data)

            # Update particle counts in header
            pt_idx = int(pt.replace('PartType', ''))
            numpart_total[pt_idx] = n_selected
            numpart_thisfile[pt_idx] = n_selected

        fout['Header'].attrs['NumPart_Total'] = numpart_total
        fout['Header'].attrs['NumPart_ThisFile'] = numpart_thisfile

        if verbose:
            total = sum(numpart_thisfile)
            print(f'  Wrote {output_path}: {total} particles '
                  f'({numpart_thisfile[0]} gas, {numpart_thisfile[5] if len(numpart_thisfile) > 5 else 0} sinks)')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input-dir', default='/scratch/vasissua/COPY/2026-03/m12f_cutout/output_jeans_refinement',
                   help='Path to the full snapshot directory')
    #p.add_argument('--input-dir', default='/scratch/vasissua/COPY/2026-03/m12f/output_jeans_refinement',
    #               help='Path to the full snapshot directory')
    p.add_argument('--output-dir', default=None,
                   help='Path for cutout output (default: sibling output_cutout dir)')
    p.add_argument('--cutout-radius', type=float, default=5e-4,
                   help='Cutout radius in comoving kpc/h (default: 5e-4, ~100 AU physical at z~21)')
    p.add_argument('--reference-search-radius', type=float, default=0.01,
                   help='Search radius for densest gas when no sinks (comoving kpc/h)')
    p.add_argument('--overwrite', action='store_true',
                   help='Overwrite existing cutout files')
    args = p.parse_args()

    input_dir = args.input_dir
    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir.rstrip('/')), 'output_cutout')
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Find all snapshots
    snap_paths = sorted(glob.glob(os.path.join(input_dir, 'snapshot_*.hdf5')))
    if not snap_paths:
        sys.exit(f'No snapshots found in {input_dir}')

    print(f'Found {len(snap_paths)} snapshots in {input_dir}')
    print(f'Output directory: {output_dir}')
    print(f'Cutout radius: {args.cutout_radius} comoving kpc/h')
    print()

    # Process in REVERSE order so we can propagate center to earlier snapshots
    snap_paths_rev = snap_paths[::-1]
    prev_center = None
    t_start = time.perf_counter()

    for i, snap_path in enumerate(snap_paths_rev):
        basename = os.path.basename(snap_path)
        out_path = os.path.join(output_dir, basename)

        if os.path.exists(out_path) and not args.overwrite:
            # Still read center from existing cutout for propagation
            with h5py.File(snap_path, 'r') as f:
                prev_center, method = find_center_raw(
                    f, reference_center=prev_center,
                    search_radius=args.reference_search_radius)
            print(f'  {basename}: skipped (exists), center={method}')
            continue

        t0 = time.perf_counter()
        with h5py.File(snap_path, 'r') as f:
            center, method = find_center_raw(
                f, reference_center=prev_center,
                search_radius=args.reference_search_radius)
            prev_center = center

            a = f['Header'].attrs['Time']
            z = f['Header'].attrs['Redshift']
            n_gas_full = f['PartType0/Coordinates'].shape[0]

        create_cutout(snap_path, out_path, center, args.cutout_radius)

        elapsed = time.perf_counter() - t0
        wall = time.perf_counter() - t_start
        avg = wall / (i + 1)
        eta = avg * (len(snap_paths) - i - 1)
        eta_m, eta_s = divmod(int(eta), 60)
        eta_h, eta_m = divmod(eta_m, 60)

        print(f'  {basename}: center={method} z={z:.1f} N_full={n_gas_full} '
              f'({elapsed:.1f}s) [{i+1}/{len(snap_paths)}] '
              f'ETA {eta_h}h{eta_m:02d}m{eta_s:02d}s')

    print('\nDone.')


if __name__ == '__main__':
    main()
