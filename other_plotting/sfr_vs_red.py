import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor

def read_snapshot_data(snapshot_file):
    with h5py.File(snapshot_file, 'r') as f:
        # Extracting redshift and SFR
        redshift = f['Header'].attrs['Redshift']
        try:
            sfr = f['PartType0']['StarFormationRate'][:]  # Assuming SFR is stored for PartType0
        except KeyError:  # Handle snapshots without SFR data
            sfr = np.array([0])
    return redshift, np.mean(sfr) if sfr.size > 0 else 0

def plot_sfr_vs_redshift(snapshot_dir, output_filename):
    snapshot_files = sorted([os.path.join(snapshot_dir, f) for f in os.listdir(snapshot_dir) if f.startswith('snapshot_')],
                            key=lambda x: int(x.split('_')[-1].split('.')[0]))

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(read_snapshot_data, snapshot_files))

    # Sorting results by redshift (in decreasing order for plot)
    results.sort(key=lambda x: x[0], reverse=True)
    redshifts, avg_sfrs = zip(*results)  # Unzipping

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(redshifts, avg_sfrs, marker='o')
    plt.xlabel('Redshift')
    plt.ylabel('Average Star Formation Rate (Solar Masses/Year)')
    plt.title('Average Star Formation Rate vs. Redshift')
    plt.gca().invert_xaxis()  # Higher redshifts are earlier in time
    plt.grid(True)
    plt.savefig(output_filename)
    plt.close()


# Example usage
snapshot_dir = '/cita/d/www/home/vpustovoit/plots/'
output_filename = '../sfr_vs_redshift.png'
plot_sfr_vs_redshift(snapshot_dir, output_filename)

