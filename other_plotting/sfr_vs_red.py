import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor

# Example usage
snapshot_dir = '../output/2024.02.01:2/'
output_filename = '/cita/d/www/home/vpustovoit/plots/sfr_vs_redshift.png'
use_log = False  # Set to True for logarithmic redshift
use_scale_factor = False  # Set to True to use scale factor instead of redshift
z_range = (0, 15)  # Specify the range of redshifts to plot, None for all

# ----------- SOURCE CODE BEGIN ----------- {{{
def read_snapshot_data(snapshot_file):
    with h5py.File(snapshot_file, 'r') as f:
        redshift = f['Header'].attrs['Redshift']
        try:
            sfr = f['PartType0']['StarFormationRate'][:]  # Assuming SFR is stored for PartType0
        except KeyError:  # Handle snapshots without SFR data
            sfr = np.array([0])
    scale_factor = 1 / (1 + redshift)
    return redshift, scale_factor, np.mean(sfr) if sfr.size > 0 else 0

def plot_sfr_vs_redshift(snapshot_dir, output_filename, use_log=False, use_scale_factor=False, z_range=None, initiate_plot=True):
    snapshot_files = sorted([os.path.join(snapshot_dir, f) for f in os.listdir(snapshot_dir) if f.startswith('snapshot_')],
                            key=lambda x: int(x.split('_')[-1].split('.')[0]))

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(read_snapshot_data, snapshot_files))

    # Filter by redshift range if specified
    if z_range is not None:
        min_z, max_z = z_range
        results = [r for r in results if min_z <= r[0] <= max_z]

    # Ordering results by redshift or scale factor
    results.sort(key=lambda x: x[1] if use_scale_factor else x[0])
    print(results[0])

    if use_scale_factor:
        #x_values, _, avg_sfrs = zip(*results)
        _, x_values, avg_sfrs = zip(*results)
    else:
        x_values, _, avg_sfrs = zip(*results)
        #_, x_values, avg_sfrs = zip(*results)
        if use_log:
            x_values = np.log10(x_values)

    # Plotting
    if initiate_plot:
        plt.figure(figsize=(10, 6))
        plt.xlabel('Scale Factor (a)' if use_scale_factor else 'Log Redshift' if use_log else 'Redshift')
        plt.ylabel('Average Star Formation Rate (Solar Masses/Year)')
        title = 'Average Star Formation Rate vs. ' + ('Scale Factor' if use_scale_factor else 'Log Redshift' if use_log else 'Redshift')
        plt.title(title)
        if not use_scale_factor:
            plt.gca().invert_xaxis()  # Higher redshifts are earlier in time, except for scale factor
        plt.grid(True)

    plt.plot(x_values, avg_sfrs, linestyle='-', marker='')  # Continuous line without large points
    plt.savefig(output_filename)

    return plt

plt = plot_sfr_vs_redshift(snapshot_dir, output_filename, use_log, use_scale_factor, z_range)
plt = plot_sfr_vs_redshift(snapshot_dir, output_filename, use_log, use_scale_factor, z_range, initiate_plot=False)
plt.close()

#}}}
