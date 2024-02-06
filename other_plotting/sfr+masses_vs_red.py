import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor

# Utility functions
def read_particle_data(snapshot_file, data_type):
    with h5py.File(snapshot_file, 'r') as f:
        redshift = f['Header'].attrs['Redshift']
        scale_factor = 1 / (1 + redshift)
        if data_type == 'SFR':
            try:
                data = f['PartType0']['StarFormationRate'][:]  # Assuming SFR is stored for PartType0
            except KeyError:  # Handle snapshots without SFR data
                data = np.array([0])
        elif data_type in ['stellar_mass', 'gas_mass']:
            part_type = 'PartType0' if data_type == 'gas_mass' else 'PartType4'
            try:
                data = f[part_type]['Masses'][:]
            except KeyError:
                data = np.array([0])
        else:
            data = np.array([0])
        data_size = len(data)
    return redshift, scale_factor, np.mean(data) if data_type=='SFR' else np.sum(data)

# Top-level function for processing snapshot data
def process_snapshot_data(args):
    snapshot_file, property_name = args
    return read_particle_data(snapshot_file, property_name)

def calculate_property(snapshot_dir, property_name, z_range=None, use_scale_factor=False):
    snapshot_files = sorted([os.path.join(snapshot_dir, f) for f in os.listdir(snapshot_dir) if f.startswith('snapshot_')],
                            key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Preparing arguments as tuples for each snapshot file
    args = [(snapshot_file, property_name) for snapshot_file in snapshot_files]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_snapshot_data, args))

    if z_range is not None:
        min_z, max_z = z_range
        results = [r for r in results if min_z <= r[0] <= max_z]

    results.sort(key=lambda x: x[1])  # Sort by scale factor for consistency
    redshifts, scale_factors, properties = zip(*results)
    return scale_factors if use_scale_factor else redshifts, properties


def plot_property(x_values, y_values, output_filename, x_label, y_label, title, use_scale_factor):
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, linestyle='-', marker='')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if not use_scale_factor:
        plt.gca().invert_xaxis()  # Higher redshifts are earlier in time
    plt.grid(True)
    plt.savefig(output_filename)
    plt.close()

def run_analysis(snapshot_dir, output_dir, use_scale_factor=False, z_range=None):
    properties = ['SFR', 'stellar_mass', 'gas_mass']

    for property_name in properties:
        # Now passing use_scale_factor to calculate_property
        x_values, y_values = calculate_property(snapshot_dir, property_name, z_range, use_scale_factor)
        output_filename = os.path.join(output_dir, f'{property_name}_vs_redshift.png')
        plot_property(x_values, 
                      y_values, 
                      output_filename, 
                      'Scale Factor' if use_scale_factor else 'Redshift', 
                      f'Average {property_name}', 
                      f'Average {property_name} vs. {"Scale Factor" if use_scale_factor else "Redshift"}', 
                      use_scale_factor)

# Example usage
snapshot_dir = '../output/2024.02.01:2/'
output_dir = '/cita/d/www/home/vpustovoit/plots/'
run_analysis(snapshot_dir, output_dir, use_scale_factor=False, z_range=(0, 15))

