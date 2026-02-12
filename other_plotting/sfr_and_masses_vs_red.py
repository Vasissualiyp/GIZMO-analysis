import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re
from concurrent.futures import ProcessPoolExecutor

scratch = '/scratch/vasissua/'
shivan1_path = os.path.join(scratch, 'SHIVAN')
shivan2_path = os.path.join(scratch, 'SHIVAN2')
ranch_path = os.path.join(scratch, 'RANCH_DATA')

def main():
    # Example usage
    #fei_folder = '/fs/lustre/project/murray/FIRE/FIRE_2/Fei_analysis/md/m12i_res7100_md/output'
    ranch_out = os.path.join(ranch_path, "m12i_r7100", "output")
    myrun_out = os.path.join(shivan2_path, "output", "2025-09-25")
    snapshot_dirs = [ ranch_out           , myrun_out            ]
    legends =       [ "FIRE2 Frontera Run", "Trillium FIRE3 run" ]
    
    output_filename = 'sfr_plot.png'
    z_range = (0, 20)
    figsize = (10,15)
    
    plot_comparisons(snapshot_dirs, legends, output_filename, figsize, use_scale_factor=False, z_range=z_range)
    #run_analysis(snapshot_dir1, False,           legend1, figsize, use_scale_factor=False, z_range=z_range)
    #run_analysis(snapshot_dir2, output_filename, legend2, figsize, use_scale_factor=False, z_range=z_range)

# ------------------------ SOURCE CODE BEGIN -------------------------------

# Processing data in the snapshots, calculations
def read_particle_data(snapshot_files, data_type, debug):
    #print(f"Files passed to read_particle_data: {snapshot_files}")
    #print()

    if debug:
        print(f"Plotting for {data_type}...")
    # Ensure 'snapshot_files' is a list, even if it's a single entry
    snapshot_files = [snapshot_files] if isinstance(snapshot_files, str) else snapshot_files
    full_data = []
    for snapshot_file in snapshot_files:
        print(f"Reading snapshot file {snapshot_file}...")
        with h5py.File(snapshot_file, 'r') as f:
            redshift, scale_factor, data = read_particle_data_single_snapshot(f, data_type)
        full_data.append(data)
        #print(f"full data for {data_type}:")
        #print(full_data)

    return redshift, scale_factor, np.sum(full_data) if data_type == 'SFR' else (np.mean(full_data) if data_type == 'galaxy_size' else np.sum(full_data))

def read_particle_data_single_snapshot(f, data_type):
    """
    Arguments:
        f: opened hdf5 file
        data_type: What kind of data do you want to read
    """
    redshift = f['Header'].attrs['Redshift']
    scale_factor = 1 / (1 + redshift)
    if data_type == 'SFR':
        try:
            data = f['PartType0']['StarFormationRate'][:]  # Assuming SFR is stored for PartType0
            data = np.sum(data)
        except KeyError:  # Handle snapshots without SFR data
            data = np.array([0])
    elif data_type in ['stellar_mass', 'gas_mass']:
        part_type = 'PartType0' if data_type == 'gas_mass' else 'PartType4'
        try:
            data = f[part_type]['Masses'][:]
            data = np.sum(data)
        except KeyError:
            data = 0
    elif data_type == 'galaxy_size':
        data = calculate_half_mass_radius(f)
    else:
        data = np.array([0])
        date = 0

    return redshift, scale_factor, data

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

def process_snapshot_data(snapshot_file, property_name, debug=False):
    """ 
    Top-level function for processing snapshot data. Needed for parallelziation
    """
    #snapshot_file, property_name = args
    return read_particle_data(snapshot_file, property_name, debug)

# Working with directories and obtaining snapshot locations
def check_directory_structure(snapshot_dir, snapshot_naming_conv = 'snapshot_'):

    """
    Checks if the provided directory contains subdirectories named `snapdir_XXX`.
    Args:
        snapshot_dir (str): The path to the directory to check.
    Returns:
        bool: True if `snapdir_XXX` subdirectories are found, False otherwise.
    """

    check = any(os.path.isdir(os.path.join(snapshot_dir, d)) and d.startswith('snapdir_') for d in os.listdir(snapshot_dir)) 

    return check

def collect_snapshot_files(snapshot_dir, snapshot_naming_conv):

    """
    Collects HDF5 snapshot files from a directory. It handles both scenarios:
    when files are directly in the directory or spread across `snapdir_XXX` subdirectories.

    Args:
        snapshot_dir (str): The path to the directory from which to collect files.
        snapshot_naming_conv: The prefix for the name of the snapshot

    Returns:
        list: A list of snapshots, where each snapshot can be a list of a single file (direct case)
              or a list of files (snapdir_XXX case).
    """

    snapshot_entries = []
    has_snapdir = check_directory_structure(snapshot_dir)

    if has_snapdir: # When all snapshots are in separate directories
        for subdir in sorted(os.listdir(snapshot_dir)): # Loop over all subdirectories
            # Look only at directories with a certain naming convention that exist
            if os.path.isdir(os.path.join(snapshot_dir, subdir)) and subdir.startswith('snapdir_'): 
                snapshot_naming_conv = "snapshot_"
                files = obtain_snapshots_from_subdirectory(snapshot_dir, subdir, snapshot_naming_conv)
                # Assuming that each subdir contains files for a single snapshot, we group them together.
                if files:  # Check if we actually found files to ensure we don't add empty lists.
                    snapshot_entries.append(files)
    else: # Case when all snapshots are in the same folder
        # When snapshots are directly in the directory, treat each file as a separate snapshot.
        snapshot_files = obtain_snapshots_following_naming_convention(snapshot_dir, snapshot_naming_conv)
        snapshot_entries = [[file] for file in snapshot_files]  # Wrap each file in its own list for consistency.

    return snapshot_entries

def obtain_snapshots_from_subdirectory(snapshot_dir, subdir, snapshot_naming_conv):

    subdir_path = os.path.join(snapshot_dir, subdir)
    files = obtain_snapshots_following_naming_convention(subdir_path, snapshot_naming_conv)

    return files

def obtain_snapshots_following_naming_convention(snapshot_dir, naming_conv):

    snapshot_files = [os.path.join(snapshot_dir, f) for f in sorted(os.listdir(snapshot_dir))
                      if f.startswith(naming_conv) and (f.endswith('.hdf5') or f.endswith('.h5'))]

    return snapshot_files

def calculate_property(snapshot_dir, property_name, z_range=None, use_scale_factor=False,
                       snapshot_naming_conv = "snapshot_"):

    # Check if the snapshots are within subdirectories or directly in the snapshot_dir
    snapshot_files = collect_snapshot_files(snapshot_dir, snapshot_naming_conv)
    #print(f"Extracted snapshot files: {snapshot_files}")

    # Preparing arguments as tuples for each snapshot file
    # Ensure the first tuple includes a True for the debug flag
    args = [(snapshot_files[0], property_name, True)] + [(snapshot_file, property_name, False) for snapshot_file in snapshot_files[1:]]

    # Parallel processing of snapshot files
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_snapshot_data, *zip(*args)))  # Adjusted for correct argument unpacking

    # Filter and sort results based on z_range and scale_factor
    if z_range is not None:
        min_z, max_z = z_range
        results = [r for r in results if min_z <= r[0] <= max_z]

    results.sort(key=lambda x: x[1])  # Assuming x[1] is scale factor for sorting

    redshifts, scale_factors, properties = zip(*results)

    return (scale_factors if use_scale_factor else redshifts, properties)

# Plotting
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

def get_property_description(property_name):
    if property_name == 'SFR':
        return 'Total'
    elif property_name in ['stellar_mass', 'gas_mass']:
        return 'Total'
    elif property_name == 'galaxy_size':
        return 'Half-stellar mass'
    else:
        return ''

# Main looping functions
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
    #properties =     ['SFR'          ]
    #property_units = [r"$M_\odot$/yr"]
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

def plot_comparisons(snapshot_dirs, legends, output_filename, figsize, use_scale_factor=False, z_range=None):
    # Iterate over each snapshot directory and its corresponding legend
    for i, (snapshot_dir, legend) in enumerate(zip(snapshot_dirs, legends)):
        print(f"Working with {legend}")
        # Check if the current directory is the last in the list
        if i < len(snapshot_dirs) - 1:
            # If not the last, perhaps pass None or handle differently
            temp_output_filename = None  # or a method to generate a temporary filename if required
        else:
            # If it is the last directory, use the actual output_filename
            temp_output_filename = output_filename

        # Call the analysis function with the appropriate filename
        run_analysis(snapshot_dir, temp_output_filename, legend, figsize, use_scale_factor, z_range)

if __name__ == '__main__':
    main()

# ------------------------ SOURCE CODE END-------------------------------
