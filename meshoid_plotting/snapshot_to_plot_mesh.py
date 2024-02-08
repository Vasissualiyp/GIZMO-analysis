# Import libraries and other files 
from meshoid import Meshoid
import re
import matplotlib.colors as colors
import matplotlib.pyplot as plt
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
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
print('Import finished')

group_name=''
# Read system arguments
if len(sys.argv) > 1:
    day_attempt = sys.argv[1]
else:
    day_attempt = '2024.02.06:5/'

snapno = '001'

ParticleType = 'PartType0'
plottype = 'density'

# For gas
clrmax = 1e-2
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

# Plot single snapshot 
def plot_for_single_snapshot_mesh(input_file, output_dir):
    
    snapno = extract_snapshot_number(input_file)
    F = h5py.File(input_file,"r")
    redshift = F['Header'].attrs['Redshift']
    if ParticleType == "PartType0":
        rho = F[ParticleType]["Density"][:]
    else:
        rho = F[ParticleType]["Masses"][:]
    
    #density_cut = (rho*1e-2 > clrmin)
    density_cut = (rho*1e-2 > 0)
    pdata = {}
    
    #for field in "Masses", "Coordinates", "SmoothingLength", "Velocities":
    if ParticleType == "PartType0":
        for field in "Density", "Coordinates", "Velocities", "SmoothingLength":
            pdata[field] = F[ParticleType][field][:][density_cut]
    else:
        for field in "Masses", "Coordinates", "Velocities":
            pdata[field] = F[ParticleType][field][:][density_cut]
    
    F.close()
    
    #print(pdata["Coordinates"])
    
    M = Create_Meshoid(pdata, ParticleType)
    planes = ['x','y','z']
    #planes = ['x']
    subfig_id = [131, 132, 133]
    fig, axs = plt.subplots(1, 3, figsize=(16, 6))  # Create a figure and a 1x3 grid of subplots
    
    for plane_index, plane in enumerate(planes):
        add_colorbar = plane_index == len(planes) - 1
        plot_single_projection(M, plane, redshift, snapno, axs[plane_index], fig, add_colorbar)  # Pass the specific axis

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
    
def plot_single_projection(M, plane, redshift, snapno, ax, fig, add_colorbar):
    rmax = SizeOfShownBox 
    res = 800
    X = Y = np.linspace(-rmax, rmax, res)
    X, Y = np.meshgrid(X, Y)
    sigma_gas_msun_pc2 = M.SurfaceDensity(M.m, plane=plane, center=np.array([0,0,0]), size=SizeOfShownBox, res=res)*1e4
    
    p = ax.pcolormesh(X, Y, sigma_gas_msun_pc2, norm=colors.LogNorm(vmin=clrmin, vmax=clrmax) if clrmin else None)

    # Create a sidebar scale
    if add_colorbar:
        # Make room for the colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(p, cax=cax, label=r"$\Sigma_{gas}$ $(\rm M_\odot\,pc^{-2})$")
    ax.set_title(f"Gas Density {plane}-projection, z={redshift:.2f}")

    print(f"ax: {ax}")
    print(f"plt: {plt}")

    ax.set_title(f"Gas Density {plane}-projection, z={redshift:.2f}")
    ax.set_aspect('equal')
    #fig.colorbar(p, ax=ax, label=r"$\Sigma_{gas}$ $(\rm M_\odot\,pc^{-2})$")

    set_axes_labels(ax, plane)

    print(f'Plotted projection {plane}')

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
def Create_Meshoid(pdata, ParticleType):
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
        print('Smoothing lengths have been found adaptively')
        # Here we will create meshoid with larger smoothing lengths
        hsml = hsml * 10
        M = Meshoid(pos, mass, hsml)
        print('Meshoid was created successfully')
    print("Created meshoid")
    return M

def snap_to_plot_mesh(input_dir, output_dir):
    max_time = 6 * 60 * 60 # Define max time (in seconds) that
    num_snapshots=get_number_of_snapshots(input_dir)

    i=0
    while i<num_snapshots - units.start:
        # Eternal plotting mode 
        time_since_snap=0
        snapno=int_to_str(i+units.start,100)
        input_file = 'snapshot_' + snapno + '.hdf5'
        input_file = input_dir + input_file
        print(f"Working with {input_file}")

        # Check if the directory exists
        if not os.path.exists(output_dir):
            # If not, create the directory
            os.makedirs(output_dir)

        if i < num_snapshots - units.start:
            time.sleep(5)
            plot_for_single_snapshot_mesh(input_file, output_dir)
            i+=1
            time_since_snap=0
        else:
            print('Executed successfully. Exiting...')
            exit()
        #    print_time_since_last_snapshot(time_since_snap, max_time)
        #    time_since_snap+=5
        #    time.sleep(5)

def snap_to_plot_mesh_parallel(input_dir, output_dir):

    print('Entered the func')
    max_time = 6 * 60 * 60  # Define max time (in seconds)
    num_snapshots = get_number_of_snapshots(input_dir)
    

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    def plot_snapshot(i):
        snapno = int_to_str(i + units.start, 100)
        input_file = os.path.join(input_dir, 'snapshot_' + snapno + '.hdf5')
        plot_for_single_snapshot_mesh(input_file, output_dir)

    # Use ThreadPoolExecutor to parallelize snapshot processing
    with ThreadPoolExecutor() as executor:
        # Submit tasks to the executor for each snapshot
        futures = [executor.submit(plot_snapshot, i) for i in range(num_snapshots - units.start)]

        # Optionally, wait for all futures to complete and handle exceptions
        for future in concurrent.futures.as_completed(futures):

            try:
                future.result()  # This will raise any exceptions caught during the execution of the task
            except Exception as e:
                print(f"An error occurred: {e}")

    print('Executed successfully. Exiting...')


#snap_to_plot_mesh_parallel(input_dir, output_dir)
snap_to_plot_mesh(input_dir, output_dir)
