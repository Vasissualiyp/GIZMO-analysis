import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from concurrent.futures import ProcessPoolExecutor


# ------------------------ SOURCE CODE BEGIN -------------------------------

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
        elif data_type == 'galaxy_size':
            data = calculate_half_mass_radius(f)
        else:
            data = np.array([0])
    return redshift, scale_factor, np.mean(data) if data_type == 'SFR' else (data if data_type == 'galaxy_size' else np.sum(data))
import numpy as np

def calculate_center_of_mass(positions, masses):
    """Calculate the center of mass given positions and masses."""
    return np.sum(positions.T * masses, axis=1) / np.sum(masses)

def calculate_half_mass_radius(f):
    """
    Calculate the half mass radius for the stellar component.
    Arguments:
        f: opened hdf5 file
    """
    if 'PartType4' in f.keys():
        # Load stellar positions and masses
        positions = f['PartType4']['Coordinates'][:]  # Shape: (N, 3)
        masses = f['PartType4']['Masses'][:]  # Shape: (N,)

        # Step 1: Calculate the center of mass
        center_of_mass = calculate_center_of_mass(positions, masses)

        # Step 2: Calculate distances from the center of mass
        distances = np.sqrt(np.sum((positions - center_of_mass)**2, axis=1))

        # Step 3: Determine the half stellar mass radius
        sorted_indices = np.argsort(distances)
        sorted_masses = masses[sorted_indices]
        cumulative_mass = np.cumsum(sorted_masses)
        total_mass = np.sum(masses)
        half_mass_index = np.where(cumulative_mass >= total_mass / 2)[0][0]
        half_mass_radius = distances[sorted_indices[half_mass_index]]
        return half_mass_radius
    else:
        return 0


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


def plot_property(x_values, y_values, x_label, y_label, title, plt_label, use_scale_factor, initiate_plot, subplot=False):
    if subplot:
        plt.subplot(subplot)

    plt.plot(x_values, y_values, linestyle='-', marker='', label=plt_label)

    if initiate_plot:
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        if not use_scale_factor:
            plt.gca().invert_xaxis()  # Higher redshifts are earlier in time
    plt.grid(True)

def get_property_description(property_name):
    if property_name == 'SFR':
        return 'Average'
    elif property_name in ['stellar_mass', 'gas_mass']:
        return 'Total'
    elif property_name == 'galaxy_size':
        return 'Half-stellar mass'
    else:
        return ''

def create_plot_arrangement(number_of_plots, row_column_or_tile):
    max_plots = 9
    if number_of_plots < max_plots:
        if row_column_or_tile == 'column':
            number_of_rows = number_of_plots
            number_of_columns = 1
        elif row_column_or_tile == 'row':
            number_of_rows = 1
            number_of_columns = number_of_plots
        elif row_column_or_tile == 'tile':
            print("Create an arrangement yourself")
            sys.exit(1)
        else:
            print(f'create_plot_arrangement requires row, column or tile as a final argument. You have passed: {row_column_or_tile}')
            sys.exit(1)
    else:
        print(f"Cannot handle situation with more than {max_plots} plots")
        sys.exit(1)
    rows_columns = number_of_rows * 100 + number_of_columns * 10
    list_of_plots = []
    for i in range(0, number_of_plots):
        new_plot = rows_columns + i + 1
        list_of_plots.append(new_plot)
    return list_of_plots


def run_analysis(snapshot_dir, output_filename, figure_label, figsize=(10,8), use_scale_factor=False, z_range=None):
    """
    Main function that plots all the quantities for a certain folder with snapshots.

    Arguments:
    snapshot_dir (str): The directory with all the snapshots 
    output_filename (str/bool): Filename of the plot. 
                                Can pass 'False' if more plotting needs to be done later in the same instance(the plot won't be closed)
    figure_label: For multiple plots, what's the name on the legend?
    figsize: What should be the size of the plot that matplotlib sets up?
    use_scale_factor: True - use a=1/(1+z) for x-values, False - use redshfit
    z_range: Range of redshifts (scale factors) over the x-axis
    """
    properties =     ['SFR',           'stellar_mass',              'gas_mass',                  'galaxy_size']
    property_units = [r"$M_\odot$/yr", r"$M_\odot \times 10^{10}$", r"$M_\odot \times 10^{10}$", "kpc"        ]
    plot_indecies = create_plot_arrangement(len(properties),'column')
    #plot_indecies =  [411, 412, 413, 414]

    # If there is no plotting instance, start it
    initiate_plot = False
    if not len(plt.get_fignums()):
        plt.figure(figsize=figsize)
        initiate_plot = True

    for plot_index, property_name in enumerate(properties):
        subfig = plot_indecies[plot_index]
        y_axis_units = property_units[plot_index]

        # Now passing use_scale_factor to calculate_property
        x_values, y_values = calculate_property(snapshot_dir, property_name, z_range, use_scale_factor)
        property_description = get_property_description(property_name)

        x_label = 'Scale Factor' if use_scale_factor else 'Redshift' 
        y_label = f'{property_description} {property_name}, {y_axis_units}' 
        plt_title = f'{property_description} {property_name} vs. {"Scale Factor" if use_scale_factor else "Redshift"}'

        plot_property(x_values, 
                      y_values, 
                      x_label, y_label, plt_title,
                      figure_label,
                      use_scale_factor,
                      initiate_plot,
                      subfig)
        if output_filename:
            plt.legend()

    print(f'Finished plotting for {figure_label}')

    if output_filename:
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()

# ------------------------ SOURCE CODE END-------------------------------

# Example usage
snapshot_dir1 = '../output/2024.02.01:2/'
legend1 = 'hdf5 ICs'
snapshot_dir2 = '../output/2024.02.06:5/'
legend2 = 'binary ICs'
output_filename = '/cita/d/www/home/vpustovoit/plots/total_plot.png'
z_range = (0, 10)
figsize = (10,15)

run_analysis(snapshot_dir1, False,           legend1, figsize, use_scale_factor=False, z_range=z_range)
run_analysis(snapshot_dir2, output_filename, legend2, figsize, use_scale_factor=False, z_range=z_range)
