# Import libraries and other files 
from meshoid import Meshoid
import re
import threading
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import h5py
import sys
import os
sys.path.append('/cita/h/home-2/vpustovoit/.local/lib/python3.10/site-packages')
from meshoid import Meshoid
sys.path.append('../')
from yt_plotting.utils import *
from yt_plotting.funcdef_snap_to_plot import get_number_of_snapshots
from other_plotting.sfr_and_masses_vs_red import create_plot_arrangement
#Imports for parallelization
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures

group_name=''
# Read system arguments
if len(sys.argv) > 1:
    day_attempt = sys.argv[1]
else:
    day_attempt = '2024.02.01:2/'
    #day_attempt = '2024.02.06:5/'

snapno = '001'

ParticleType = 'PartType0'
plottype = 'density'

# For gas
clrmax = 1e-1
clrmin = 1e-10
## For gas
#clrmax = 1e-1
#clrmin = 1e-4

OriginalBoxSize = 60000 # In kpc
SizeOfShownBox = OriginalBoxSize / 100 * 3.5
# For 2D plots (plottype = temperature, density)
axis_of_projection='all'

# Legacy 
#Units
time_units='redshift'
boxsize_units='Mpc'
density_units='g/cm**3'
temperature_units='K'
velocity_units='km/s'
smoothing_length_units='Mpc'
first_snapshot=0
#zoom=1000 # set 128 for density and 20 for weighted_temperature
zoom=10 # set 128 for density and 20 for weighted_temperature
custom_center=[0,0,0]

#color map limits
colorbar_lims = (clrmin, clrmax)

# Getting the in/out directories
name_appendix = ParticleType + '/' + axis_of_projection + '_' + plottype + '/'
#input_file = 'snapshot_'+snapno+'.hdf5'
#output_file = '2Dplot'+snapno+'.png'
# In/Out Directories
input_dir='/fs/lustre/scratch/vpustovoit/FIRE_TEST2/output/' + day_attempt
#out_dir='./densityplots/'
output_dir='/cita/d/www/home/vpustovoit/plots/' + day_attempt + name_appendix

#input_file = input_dir + input_file
#output_file = output_dir + output_file

#Units Array 
units = Units(
    time=time_units,
    boxsize=boxsize_units,
    density=density_units,
    temperature=temperature_units,
    velocity=velocity_units,
    smoothing_length=smoothing_length_units,
    axis_of_projection=axis_of_projection,
    group_name=group_name,
    ParticleType=ParticleType,
    file_name='',  # For the filename. Used later in the code
    clr_range=colorbar_lims,
    start=first_snapshot,
    custom_center=custom_center,
    zoom=zoom
)


# Extract snapshot number from the filename 
def extract_snapshot_number(file_path):
    """
    Extracts the snapshot number as a string from a given file path.

    Parameters:
        file_path (str): The path to the HDF5 file.

    Returns:
        str: The snapshot number as a string if found, otherwise None.
    """
    # Use regular expression to find the snapshot number
    match = re.search(r'snapshot_([\d]+)\.hdf5', file_path)
    if match:
        return match.group(1)
    else:
        return None

def define_fields_to_access():

    # Define a dictionary mapping particle types to their fields
    fields_by_particle_type = {
        "PartType0": ["Density", "Coordinates", "Velocities", "SmoothingLength"],
        # Add other particle types and their fields here
        "default": ["Masses", "Coordinates", "Velocities"]  # Default fields for other types
    }

    return fields_by_particle_type

def extract_particle_data_from_file(input_file, ParticleType, fields_to_access_dict, density_cut_factor, density_cut_min = 0):
    
    # Use the dictionary to dynamically access fields based on ParticleType
    fields_to_accesss = fields_to_access_dict.get(ParticleType, fields_to_access_dict["default"])
    
    F = h5py.File(input_file,"r")
    redshift = F['Header'].attrs['Redshift']
    if ParticleType == "PartType0":
        rho = F[ParticleType]["Density"][:]
    else:
        rho = F[ParticleType]["Masses"][:]
    
    #density_cut = (rho*1e-2 > clrmin)
    density_cut = (rho*density_cut_factor > density_cut_min)
    pdata = {}
    
    # Now iterate over the fields to access for the current ParticleType
    for field in fields_to_access_dict:
        pdata[field] = F[ParticleType][field][:][density_cut]
    
    F.close()

    return pdata

def combine_particle_data_from_directory(snapshot_dir, ParticleType, fields_to_access_dict, density_cut_factor, density_cut_min=0):
    # Initialize an empty dictionary to hold the combined data
    combined_pdata = {}

    # List all hdf5 files in the snapshot directory
    snapshot_files = [f for f in os.listdir(snapshot_dir) if f.endswith('.hdf5')]

    # Sort files to ensure order, if necessary
    snapshot_files.sort()

    # Iterate over each file and extract data
    for snapshot_file in snapshot_files:
        file_path = os.path.join(snapshot_dir, snapshot_file)

        # Extract data using the existing function
        pdata = extract_particle_data_from_file(file_path, ParticleType, fields_to_access_dict, density_cut_factor, density_cut_min)

        # Combine pdata from each file
        for field, data in pdata.items():
            if field in combined_pdata:
                # Concatenate data for existing fields
                combined_pdata[field] = np.concatenate((combined_pdata[field], data))
            else:
                # Initialize field in combined_pdata
                combined_pdata[field] = data

    return combined_pdata

def extract_particle_data_from_file_or_dir(input_path, ParticleType, fields_to_access_dict, density_cut_factor, density_cut_min=0):
    # Check if the input path is a file or directory
    if os.path.isfile(input_path):
        # It's a file, use the existing function to extract data
        return extract_particle_data_from_file(input_path, ParticleType, fields_to_access_dict, density_cut_factor, density_cut_min)
    elif os.path.isdir(input_path):
        # It's a directory, use the combine function to handle multiple files
        return combine_particle_data_from_directory(input_path, ParticleType, fields_to_access_dict, density_cut_factor, density_cut_min)
    else:
        # The input path is neither a file nor a directory, raise an error
        raise ValueError("The input path is neither a file nor a directory")

# Plot single snapshot 
def plot_for_single_snapshot_mesh(input_file, output_dir, debug=False):
    """
    Creates a plot for a single snapshot.
    Arguments:
        input_file: the .hdf5 file of snapshot or directory with the snapshot files
        output_dir: the directory where the plot should be saved
        debug: flag for debugging
    """
    
    snapno = extract_snapshot_number(input_file)
    
    fields_to_access = define_fields_to_access()
    pdata = extract_particle_data_from_file_or_dir(input_file, ParticleType, fields_to_access, density_cut_factor = 1e-2, density_cut_min = 0)
    
    M = Create_Meshoid(pdata, ParticleType, debug)
    planes = ["z","x","y"]
    #planes = ['x']
    subfig_id = [131, 132, 133]
    fig, axs = plt.subplots(1, 3, figsize=(16, 6))  # Create a figure and a 1x3 grid of subplots
    
    for plane_index, plane in enumerate(planes):
        add_colorbar = plane_index == len(planes) - 1
        plot_single_projection(M, plane, redshift, snapno, axs[plane_index], fig, add_colorbar, debug)  # Pass the specific axis

    # Check if the directory exists
    if not os.path.exists(output_dir):
        # If not, create the directory
        os.makedirs(output_dir)

    #plt.show()
    plt.subplots_adjust(wspace=0.3)  # Adjust this value as needed
    output_file = '2Dplot'+snapno+'.png'
    output_file = output_dir + output_file
    print(f"Saved the plot {output_file}")
    M = None

    plt.savefig(output_file)
    plt.close()
    
def plot_single_projection(M, plane_of_proj, redshift, snapno, ax, fig, add_colorbar, debug=False):
    rmax = SizeOfShownBox 
    res = 800
    X = Y = np.linspace(-rmax, rmax, res)
    X, Y = np.meshgrid(X, Y)
    sigma_gas_msun_pc2 = M.SurfaceDensity(M.m, plane=plane_of_proj, center=np.array([0,0,0]), size=SizeOfShownBox, res=res)*1e4
    
    p = ax.pcolormesh(X, Y, sigma_gas_msun_pc2, norm=colors.LogNorm(vmin=clrmin, vmax=clrmax) if clrmin else None)

    # Create a sidebar scale
    if add_colorbar:
        # Make room for the colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(p, cax=cax, label=r"$\Sigma_{gas}$ $(\rm M_\odot\,pc^{-2})$")
    ax.set_title(f"Gas Density {plane_of_proj}-projection, z={redshift:.2f}")

    ax.set_aspect('equal')
    #fig.colorbar(p, ax=ax, label=r"$\Sigma_{gas}$ $(\rm M_\odot\,pc^{-2})$")

    set_axes_labels(ax, plane_of_proj)

    if debug:
        print(f'Plotted projection {plane_of_proj}')

def set_axes_labels(ax, plane):
    """
    Creates labels, for the other 2 axes, given the axis of the plane of projection
    """
    if plane == 'x':
        # For x-projection, we're looking at the YZ plane
        ax.set_xlabel("Y (kpc)")
        ax.set_ylabel("Z (kpc)")
    elif plane == 'y':
        # For y-projection, we're looking at the XZ plane
        ax.set_xlabel("X (kpc)")
        ax.set_ylabel("Z (kpc)")
    elif plane == 'z':
        # For z-projection, we're looking at the XY plane
        ax.set_xlabel("X (kpc)")
        ax.set_ylabel("Y (kpc)")
    else:
        raise ValueError(f"Invalid plane '{plane}'. Expected 'x', 'y', or 'z'.")

# Create meshoid based on the particle type 
def Create_Meshoid(pdata, ParticleType, debug=False):
    pos = pdata["Coordinates"]
    center = np.median(pos,axis=0)
    pos -= center
    radius_cut = np.sum(pos*pos,axis=1) < SizeOfShownBox*SizeOfShownBox
    # Below for PartType0
    if ParticleType == "PartType0":
        pos, mass, hsml, v = pos[radius_cut], pdata["Density"][radius_cut], pdata["SmoothingLength"][radius_cut], pdata["Velocities"][radius_cut]
        M = Meshoid(pos, mass, hsml)
    # Below for PartType1
    else:
        pos, mass, v = pos[radius_cut], pdata["Masses"][radius_cut], pdata["Velocities"][radius_cut]
        # Here meshoid will automatically find the adaptive smoothing lengths
        M = Meshoid(pos, mass)
        hsml = M.SmoothingLength()
        if debug:
            print('Smoothing lengths have been found adaptively')
        # Here we will create meshoid with larger smoothing lengths
        hsml = hsml * 10
        M = Meshoid(pos, mass, hsml)
        if debug:
            print('Meshoid was created successfully')
    if debug:
        print("Created meshoid")
    return M

def find_snapshot_or_directory(input_dir, snapno):
    """
    Attempts to find a snapshot file or directory based on the snapshot number.

    Parameters:
    - input_dir (str): The base directory where the snapshot files or directories are expected to be found.
    - snapno (str): The snapshot number as a string.

    Returns:
    - str: The path to the snapshot file or directory if found.

    Raises:
    - FileNotFoundError: If neither the snapshot file nor the directory exists.
    """
    # Construct the file and directory names
    snapshot_file = os.path.join(input_dir, 'snapshot_' + snapno + '.hdf5')
    snapshot_dir = os.path.join(input_dir, 'snapdir_' + snapno)

    # Check if the snapshot file exists
    if os.path.isfile(snapshot_file):
        print('hdf5 found')
        return snapshot_file

    # If not, check if the snapshot directory exists
    elif os.path.isdir(snapshot_dir):
        print('dir found')
        return snapshot_dir

    # If neither exists, raise an error
    else:
        raise FileNotFoundError(f"Neither snapshot file nor directory exists for snapshot number {snapno} in {input_dir}")

# The main function that calls everything else. Parallelized and non-parallelized versions.
def snap_to_plot_mesh_nonparallel(input_dir, output_dir):

    debug = True
    max_time = 6 * 60 * 60 # Define max time (in seconds) that
    num_snapshots=get_number_of_snapshots(input_dir)

    i=0
    while i<num_snapshots - units.start:
        # Eternal plotting mode 
        time_since_snap=0
        snapno=int_to_str(i+units.start,100)
        input_file = find_snapshot_or_directory(input_dir, snapno)
        print(f"Working with {input_file}")

        # Check if the directory exists
        if not os.path.exists(output_dir):
            # If not, create the directory
            os.makedirs(output_dir)

        if i < num_snapshots - units.start:
            time.sleep(5)
            plot_for_single_snapshot_mesh(input_file, output_dir, debug)
            i+=1
            time_since_snap=0
        else:
            print('Executed successfully. Exiting...')
            exit()
        #    print_time_since_last_snapshot(time_since_snap, max_time)
        #    time_since_snap+=5
        #    time.sleep(5)

def snap_to_plot_mesh_parallel(input_dir, output_dir):
    increase_plots_limit()

    max_time = 6 * 60 * 60  # Define max time (in seconds)
    num_snapshots = get_number_of_snapshots(input_dir)

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def plot_snapshot(i, debug):
        snapno = int_to_str(i + units.start, 100)
        input_file = os.path.join(input_dir, 'snapshot_' + snapno + '.hdf5')
        plot_for_single_snapshot_mesh(input_file, output_dir, debug)

    # Use ThreadPoolExecutor to parallelize snapshot processing
    with ThreadPoolExecutor() as executor:
        # Submit tasks to the executor for each snapshot
        futures = [executor.submit(plot_snapshot, i, i == 0) for i in range(num_snapshots - units.start)]

        # Optionally, wait for all futures to complete and handle exceptions
        for future in as_completed(futures):
            try:
                future.result()  # This will raise any exceptions caught during the execution of the task
            except Exception as e:
                print(f"An error occurred: {e}")

def replot_small_files(input_dir, output_dir, threshold=0.5):
    plot_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    file_sizes = {f: os.path.getsize(os.path.join(output_dir, f)) for f in plot_files}
    max_size = max(file_sizes.values())

    small_files = [f for f, size in file_sizes.items() if size < threshold * max_size]

    for f in small_files:
        snapno = f.replace('2Dplot', '').replace('.png', '')
        input_file = os.path.join(input_dir, 'snapshot_' + snapno + '.hdf5')
        print(f"Replotting {f} due to small size.")
        plot_for_single_snapshot_mesh(input_file, output_dir, debug=False)  # Assuming debug=False for replotting

def snap_to_plot_mesh(input_dir, output_dir, parallel=False):
    if parallel:
        snap_to_plot_mesh_parallel(input_dir, output_dir)
    else:
        snap_to_plot_mesh_nonparallel(input_dir, output_dir)

    # After initial plotting, check and replot small files
    replot_small_files(input_dir, output_dir)

# Function to handle the user's input
def ask_user():
    global user_response
    user_response = input("Increase plot limit? (y/[n]): ").strip().lower()
    if user_response == '':
        user_response = 'y'  # Default to 'yes' if no response

def increase_plots_limit():
    # Default user response
    user_response = None
    
    # Set a timer for 5 seconds to wait for user input
    timer = threading.Timer(5.0, lambda: None)  # Does nothing on timeout, just expires
    timer.start()
    
    ask_user()  # Prompt the user for input
    
    timer.cancel()  # Cancel the timer if user responds before timeout
    
    # Check the user's response or default action after timeout
    if user_response in ['y', None]:  # None represents no response (default to 'y' after timeout)
        print("Increasing plot limit.")
        mpl.rcParams['figure.max_open_warning'] = 50
    else:
        print("Not increasing plot limit.")

"""
if __name__ == '__main__':

    main()
"""
snap_to_plot_mesh(input_dir, output_dir, parallel=False)
