# This code contains utility functions that are used in other funcitons
# Libraries {{{
import os 
from scipy.interpolate import Rbf
import time
import pandas as pd
import h5py
import numpy as np 
import yt 
import unyt
from PIL import Image
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt #}}}

#Constants
HubbleParam = 0.7
gamma = 5.0/3.0
R = 8.31 * unyt.J / unyt.K / unyt.mol

# Custom density computation from the mass and smoothing length {{{
def compute_density_from_mass(field, data):
    # Assuming you have a field named "Masses" for particle masses,
    # and a field named "SmoothingLength" for the smoothing lengths.
    
    particle_masses = data["PartType2", "Masses"]
    smoothing_lengths = data["PartType2", "Softening_KernelRadius"]
    
    # Calculate density using your SPH algorithm
    # This is a placeholder; you will need to replace this with your actual density computation
    density = particle_masses / (smoothing_lengths ** 3)
    
    return density
#}}}

#This function converts the number of snapshot into a string for its name {{{
def int_to_str(i, n): 
    
    n_digits = len(str(n)) 
     
    # Convert the integer to a string 
    s = str(i) 
     
    # Add leading zeros to the string to make it n characters long 
    s = s.zfill(n_digits) 
     
    return s #}}}

# SECTION | Plotting utilities {{{
# Function for annotating the plots {{{
def annotate(snapshot, plot, plottype, units, flags):
    
    #Put the units in {{{
    time_units = units[0]
    boxsize_units = units[1]
    density_units = units[2]
    temperature_units = units[3]
    velocity_units = units[4]
    smoothing_length_units = units[5]
    axis_of_projection = units[6]
    file_path = units[9]
    #}}}

    if 'custom_loader' in flags:
        # Path to the HDF5 file
        #file_path = "output/2023.08.25:7/snapshot_022.hdf5"
        
        # Open the file in read mode
        with h5py.File(file_path, 'r') as file:
            # Navigate to the Header group
            header_group = file['Header']
        
            # Extract the Redshift and Time attributes
            redshift = header_group.attrs['Redshift']
            code_time = header_group.attrs['Time']
        
            #print(f"Redshift: {redshift}")
            #print(f"Time: {time}")

    else:
        redshift = float(snapshot.current_redshift) 
        code_time = float(snapshot.current_time) 
        redshift = 1 / code_time - 1 # Since code time is the scaling factor a
    print(plottype)
    if plottype in ['density']:
        # annotate the plot {{{
        if time_units=='redshift':
            plot.annotate_title("Density Plot, z={:.6g}".format(redshift)) 
        elif time_units=='code':
            plot.annotate_title("Density Plot, a={:.2g}".format(code_time)) 
        else:
            time_yrs=code_time * 0.978*10**9 / HubbleParam * unyt.yr
            time_yrs=time_yrs.to_value(time_units)
            plot.annotate_title("Density Plot, t={:.2g}".format(time_yrs) + " " + time_units)  
    #}}}
    elif plottype in ['density-2', 'density-3']:
        # annotate the plot {{{
        if time_units=='redshift':
            plot.title("Density Plot, z={:.6g}".format(redshift)) 
        elif time_units=='code':
            plot.title("Density Plot, a={:.2g}".format(code_time)) 
        else:
            time_yrs=code_time * 0.978*10**9 / HubbleParam * unyt.yr
            time_yrs=time_yrs.to_value(time_units)
            plot.title("Density Plot, t={:.2g}".format(time_yrs) + " " + time_units)  
        plot.xlabel('x, ' + boxsize_units)
        plot.ylabel('y, ' + boxsize_units) #}}}
    elif plottype in ['deposited_density']:
        # annotate the plot {{{
        if time_units=='redshift':
            plot.annotate_title("Density Plot, z={:.6g}".format(redshift)) 
        elif time_units=='code':
            plot.annotate_title("Density Plot, a={:.2g}".format(code_time)) 
        else:
            time_yrs=code_time * 0.978*10**9 / HubbleParam * unyt.yr
            time_yrs=time_yrs.to_value(time_units)
            plot.annotate_title("Density Plot, t={:.2g}".format(time_yrs) + " " + time_units)  
    #}}}
    elif plottype in ['mass-gridded']:
        # annotate the plot {{{
        if time_units=='redshift':
            plt.title("Density Plot, z={:.6g}".format(redshift)) 
        elif time_units=='code':
            plt.title("Density Plot, a={:.2g}".format(code_time)) 
        else:
            time_yrs=code_time * 0.978*10**9 / HubbleParam * unyt.yr
            time_yrs=time_yrs.to_value(time_units)
            plt.title("Density Plot, t={:.2g}".format(time_yrs) + " " + time_units)  
        #plt.xlabel('x, ' + boxsize_units)
        #plt.ylabel('y, ' + boxsize_units) #}}}

    elif plottype=='temperature':
        # annotate the plot {{{
        if time_units=='redshift':
            plot.annotate_title("Temperature Plot, z={:.6g}".format(redshift)) 
        elif time_units=='code':
            plot.annotate_title("Temperature Plot, t={:.2g}".format(code_time)) 
        else:
            time_yrs=code_time * 0.978*10**9 / HubbleParam * unyt.yr
            time_yrs=time_yrs.to_value(time_units)
            plot.annotate_title("Temperature Plot, t={:.2g}".format(time_yrs), " ", time_units) 
    #}}}
    elif plottype=='smooth_length':
        # annotate the plot {{{
        if time_units=='redshift':
            plot.annotate_title("Smoothing Lengths Plot, z={:.6g}".format(redshift)) 
        elif time_units=='code':
            plot.annotate_title("Smoothing Lengths Plot, t={:.2g}".format(code_time)) 
        else:
            time_yrs=code_time * 0.978*10**9 / HubbleParam * unyt.yr
            time_yrs=time_yrs.to_value(time_units)
            plot.annotate_title("Smoothing Lengths Plot, t={:.2g}".format(time_yrs) + " "+ time_units) 
    #}}}
    elif plottype=='density_profile':
        # annotate the plot {{{
        # Set the time units
        time_yrs=code_time * 0.978 / HubbleParam * unyt.Gyr
        time_yrs=time_yrs.to_value(time_units)
        # annotate
        plot.title('Density Profile, t={:.2g}'.format(time_yrs) + ' ' + time_units)
        plot.xlabel('x, ' + boxsize_units)
        plot.ylabel('density, ' + density_units ) #}}}
    elif plottype=='shock_velocity':
        # annotate the plot {{{
        # Set the time units
        time_yrs=code_time * 0.978 / HubbleParam * unyt.Gyr
        time_yrs=time_yrs.to_value(time_units)
        # annotate
        plot.title('Velocity Profile, t={:.2g}'.format(time_yrs) + ' ' + time_units)
        print('The time for annotation is: ' + str(time_yrs))
        plot.xlabel('x, ' + boxsize_units)
        plot.ylabel('velocity, ' + velocity_units ) #}}}
    elif plottype=='smoothing_length_hist':
        # annotate the plot {{{
        plot.xlabel('Smoothing Length ' + smoothing_length_units )
        plot.ylabel('Count')
        plot.title('Smoothing Length Histogram')
    #}}}
#}}}

# This function is needed to wrap the second part of the coords around {{{
def wrap_second_half(arr):
    half_len = int(len(arr)//2)
    # Taking an odd number of elemetns into consideration
    if len(arr) % 2==1:
        half_len += 1
    half_units = max(arr)
    for i in range(half_len, len(arr)):
        arr[i] = arr[i] - half_units
    return arr
#}}}

# Function that erases unnecessary (symmetric) data {{{
def cutoff_arrays(x, cutoff, *ys):
    cut_indices = [i for i in range(len(x)) if x[i] >= cutoff]
    cut_x = [x[i] for i in cut_indices]
    cut_ys = [[y[i] for i in cut_indices] for y in ys]
    return cut_x, cut_ys
#}}}

# The funciton that is needed to sort the arrays in 1D based on the values in the coordinate arrays {{{
def sort_arrays(x, *y):
    # sort indices based on x array
    indices = sorted(range(len(x)), key=lambda i: x[i])
    
    # sort all arrays based on sorted indices
    x_sorted = [x[i] for i in indices]
    y_sorted = [[y_arr[i] for i in indices] for y_arr in y]
    
    return x_sorted, y_sorted
#}}}

# LEGACY | This function combines two plots into a single one {{{
"""
def combine_snapshots(folder1, folder2, output_folder):
    #Combines snapshots from folder1 and folder2 into a single picture with one picture on the left and another one on the right.
    #The resulting images are saved in output_folder.
    # get the list of files in folder1 and sort them
    files1 = os.listdir(folder1)
    files1.sort()

    # get the list of files in folder2 and sort them
    files2 = os.listdir(folder2)
    files2.sort()

    # iterate over the files and combine them
    for i in range(len(files1)):
        # open the images
        img1 = Image.open(os.path.join(folder1, files1[i]))
        img2 = Image.open(os.path.join(folder2, files2[i]))

        # resize the images if their heights are different
        if img1.height != img2.height:
            if img1.height < img2.height:
                img1 = img1.resize((int(img1.width * img2.height / img1.height), img2.height), Image.LANCZOS)
            else:
                img2 = img2.resize((int(img2.width * img1.height / img2.height), img1.height), Image.LANCZOS)

        # create a new image with twice the width
        new_img = Image.new('RGB', (img1.width*2, img1.height))

        # paste the images side by side
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (img1.width, 0))

        # save the new image
        new_img.save(os.path.join(output_folder, f'snapshot_{i:03d}.png'))
        percent = int(i / len(files1) * 100)
        print('Combining images... ' + str(percent) + "%")
"""
#}}}

# This function combines n plots into a single one {{{
def combine_snapshots(output_folder, *folders):

    """
    #Combines snapshots from input folders into a single picture with one picture on the left and another one on the right.
    #The resulting images are saved in output_folder.
    """

    # get the list of files in all input folders and sort them
    files = [sorted(os.listdir(folder)) for folder in folders]

    # get the total number of files in the folders
    num_files = len(files[0])

    # iterate over the files and combine them
    for i in range(num_files):

        # open the images from each folder
        imgs = []
        for folder in folders:
            img = Image.open(os.path.join(folder, files[folders.index(folder)][i]))
            imgs.append(img)

        # resize the images if their heights are different
        heights = [img.height for img in imgs]
        max_height = max(heights)
        for j in range(len(imgs)):
            if imgs[j].height < max_height:
                imgs[j] = imgs[j].resize((int(imgs[j].width * max_height / imgs[j].height), max_height), Image.LANCZOS)

        # create a new image with twice the width
        new_img = Image.new('RGB', (sum([img.width for img in imgs]), max_height))

        # paste the images side by side
        x_offset = 0
        for img in imgs:
            new_img.paste(img, (x_offset, 0))
            x_offset += img.width

        # save the new image
        new_img.save(os.path.join(output_folder, f'snapshot_{i:03d}.png'))

        percent = int(i / num_files * 100)
        print(f"\rCombining images... {percent}%", end="")
    print("\rCombining images... Done!")
#}}}
#}}}

# Function that centers the distribution and gives the size of the box {{{
def center_and_find_box(df):
    # Convert coordinates to numpy array
    coordinates = np.array(df['Coordinates'])

    # Calculate geometric center in 3D
    center = np.mean(coordinates, axis=0)

    # Center the coordinates
    centered_coordinates = coordinates - center

    # Update the dictionary
    df['Coordinates'] = centered_coordinates

    # Find the dimensions of the bounding cube
    min_coord = np.min(centered_coordinates, axis=0)
    max_coord = np.max(centered_coordinates, axis=0)

    # Calculate the size of the box
    box_size = np.max(max_coord - min_coord)

    return df, box_size
#}}}

# SECTION |  Custom Loaders {{{
# Custom loader that loads the entirety of hdf5 file {{{
def custom_load_all_data(hdf5_file_path, group_name, ParticleType, flags):
    # Dictionary to store all the data
    #ParticleType = 'PartType0'
    data_dict = {}
    Restricted_groups = ['Coordinates', "SmoothingLength", "Density", "Masses", "Softening_KernelRadius"] # The only groups that are needed for plotting
    
    #print(f'Opening {hdf5_file_path}...')
    # Open the HDF5 file
    with h5py.File(hdf5_file_path, 'r') as f:
        if group_name !='':
            # Loop through all datasets in the specified group
            print('Opening the hdf5 file. Here are its particle types:')
            for subgroup in f[group_name]:
                print(subgroup)
            print()
            print("Here is all available data:")
            for subgroup in f[group_name][ParticleType]:
                # Store the data in the dictionary
                if 'RestrictedLoad' in flags:
                    if subgroup in Restricted_groups:
                        data_dict[subgroup] = np.array(f[group_name][ParticleType][subgroup])
                else:
                    data_dict[subgroup] = np.array(f[group_name][ParticleType][subgroup])
                print(subgroup)
            print()
        
        else:
            # Loop through all datasets in the specified group
            print('Particle Types:')
            for subgroup in f:
                print(subgroup)

            print("Here is all available data:")
            for subgroup in f:
                print(subgroup)
            for subgroup in f[ParticleType]:
                # Store the data in the dictionary
                if 'RestrictedLoad' in flags:
                    if subgroup in Restricted_groups:
                        data_dict[subgroup] = np.array(f[ParticleType][subgroup])
                else:
                    data_dict[subgroup] = np.array(f[ParticleType][subgroup])
                print(subgroup)
            print()
    
    # If 'Coordinates' dataset is present, calculate plot_params
    if 'Coordinates' in data_dict:
        coords = data_dict['Coordinates']
        plot_params = np.zeros((3, 3))
        for i in (0, 1, 2):
            plot_params[0, i] = min(coords[:, i])  # min border
            plot_params[1, i] = max(coords[:, i])  # max border
            plot_params[2, i] = (plot_params[1, i] - plot_params[0, i]) / 2  # origin
        return data_dict, plot_params
    else:
        return data_dict
#}}}

# Custom loader that loads only what is needed for plotting {{{
def custom_load_noyt(hdf5_file_path, group_name):
    # Open the HDF5 file
    with h5py.File(hdf5_file_path, 'r') as f:
        # Get the group's dataset
        dataset = f[group_name]['PartType1']['Coordinates']
        masses = np.array(f[group_name]['PartType1']['Masses'])
        dens = np.array(f[group_name]['PartType1']['Density'])
        smoothlen = np.array(f[group_name]['PartType1']['SmoothingLength'])

        # Extract coordinates
        data = np.array(dataset)

    # Separate the data into different arrays
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # Define a bounding box within which the particles are loaded
    # bbox = [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    bbox = [[min(x), max(x)], [min(y), max(y)], [min(z), max(z)]]

    # Creating parameters of the plots
    plot_params = np.zeros((3,3))
    for i in (0,1,2):
        plot_params[0,i] = min( data[:,i]) # min border 
        plot_params[1,i] = max( data[:,i]) # max border 
        plot_params[2,i] = (plot_params[1,i] - plot_params[0,i])/2   # origin

    return x,y,z,dens,smoothlen, plot_params
#}}}

# Custom loader that loads only what is needed for plotting, outputs as the yt dataframe {{{
def custom_load(hdf5_file_path, group_name):
    # Open the HDF5 file
    with h5py.File(hdf5_file_path, 'r') as f:
        # Get the group's dataset
        dataset = f[group_name]['PartType1']['Coordinates']
        masses = np.array(f[group_name]['PartType1']['Masses'])
        dens = np.array(f[group_name]['PartType1']['Density'])
        smoothlen = np.array(f[group_name]['PartType1']['SmoothingLength'])

        # Extract coordinates
        data = np.array(dataset)

    # Separate the data into different arrays
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # Create a dictionary with field names as keys and numpy arrays as values
    cloud_data = dict(
        particle_position_x=x,
        particle_position_y=y,
        particle_position_z=z,
        particle_mass=masses,
        density=dens,
        smoothing_length = smoothlen,
    )

    # Define a bounding box within which the particles are loaded
    # bbox = [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    bbox = [[min(x), max(x)], [min(y), max(y)], [min(z), max(z)]]

    # Pass the cloud_data dictionary to yt.load_particles
    ds = yt.load_particles(cloud_data, bbox=bbox)
    #ds.add_sph_fields()

    # Creating parameters of the plots
    plot_params = np.zeros((3,3))
    for i in (0,1,2):
        plot_params[0,i] = min( data[:,i]) # min border 
        plot_params[1,i] = max( data[:,i]) # max border 
        plot_params[2,i] = (plot_params[1,i] - plot_params[0,i])/2   # origin

    return ds, plot_params
#}}}
# }}}

# FFT Upscaler {{{
def increase_resolution_with_fft(data_dict, n):
    # Check if 'Density' and 'Coordinates' are present in the data dictionary
    if 'Density' in data_dict and 'Coordinates' in data_dict:
        # Extract the density data
        original_density = data_dict['Density']

        # Ensure density data is 3D
        if len(original_density.shape) == 3:
            # Perform 3D Fourier interpolation on the density data
            original_shape = original_density.shape
            spectrum = fftpack.fftn(original_density)
            new_shape = tuple((n-1) * np.array(original_shape))
            padded_spectrum = np.pad(spectrum, [(0, nz) for nz in new_shape], 'constant')
            interpolated_density = np.abs(fftpack.ifftn(padded_spectrum))

            # Add the interpolated density data to the dictionary
            data_dict['Density'] = interpolated_density

            # Extract coordinates and calculate new coordinates based on the increased resolution
            x, y, z = data_dict['Coordinates'].T
            new_x = np.linspace(min(x), max(x), len(x) * n)
            new_y = np.linspace(min(y), max(y), len(y) * n)
            new_z = np.linspace(min(z), max(z), len(z) * n)

            # Add the new coordinates to the dictionary
            data_dict['Coordinates'] = np.column_stack((new_x, new_y, new_z))
        else:
            print("Density data must be 3D for Fourier interpolation.")
    else:
        print("Density and/or Coordinates not found in the data dictionary.")

    return data_dict
#}}}

# RBF Upscaler {{{
def increase_resolution_with_rbf(data_dict, n, flags):
    """
    Increase the resolution of the data in the dictionary using Radial Basis Function interpolation.

    Parameters:
    - data_dict (dict): Dictionary containing the data with keys as variable names and values as arrays.
    - n (int): Factor by which to increase the resolution.
    - flags (dict): Dictionary containing various flags, including a debugging flag.

    Returns:
    - dict: A new dictionary containing the interpolated data with increased resolution.
    """
    # Check if 'Coordinates' are present in the data dictionary
    if 'Coordinates' not in data_dict:
        print("Coordinates not found in the data dictionary.")
        return None

    # Extract the original coordinates
    coords = data_dict['Coordinates']
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # Generate new coordinates based on the increased resolution
    new_x = np.linspace(min(x), max(x), len(x) * n)
    new_y = np.linspace(min(y), max(y), len(y) * n)
    new_z = np.linspace(min(z), max(z), len(z) * n)

    # Initialize a new dictionary to store the interpolated data
    new_data_dict = {}

    # Store the new coordinates in the new dictionary
    new_coords = np.column_stack((new_x, new_y, new_z))
    new_data_dict['Coordinates'] = new_coords

    # Initialize new Particle IDs
    if 'ParticleIDs' in data_dict:
        max_original_id = max(data_dict['ParticleIDs'])
        new_id_counter = max_original_id + 1
        new_particle_ids = []

    # Initialize new ParticleIDGenerationNumber and ParticleChildIDsNumber
    if 'ParticleIDGenerationNumber' in data_dict:
        new_particle_id_gen_nums = []
    if 'ParticleChildIDsNumber' in data_dict:
        new_particle_child_ids_nums = []

    multiDArrays = ['Coordinates', 'Metallicity','Velocities']
    # Loop through all the variables in the data dictionary
    for variable_name, variable_data in data_dict.items():
        # Skip the 'Coordinates' key as we have already handled it
        if variable_name not in multiDArrays:
            # Perform RBF interpolation on the variable data
            # debugging {{{
            if 'debugging' in flags:
                print(f"Interpolating {variable_name}...")
                print(f"Type of x: {type(x)}")
                print(f"Type of y: {type(y)}")
                print(f"Type of z: {type(z)}")
                print(f"Type of variable_data: {type(variable_data)}")
                print(f"Number of NaN values in variable_data: {np.isnan(variable_data).sum()}")
                print(f"Number of Inf values in variable_data: {np.isinf(variable_data).sum()}")
                print(f"Data type of elements in variable_data: {variable_data.dtype}")
                print(f"Length of x: {np.shape(x)}")
                print(f"Length of y: {np.shape(y)}")
                print(f"Length of z: {np.shape(z)}")
                print(f"Length of variable_data: {np.shape(variable_data)}")
                print()
            #}}}
            rbf = Rbf(x,y,z, variable_data, function='multiquadric', smooth=1)
            small_positive_value=1e-10
            interpolated_data = np.array([rbf(xi, yi, zi) for xi, yi, zi in zip(new_x, new_y, new_z)])
            interpolated_data = np.maximum(interpolated_data, small_positive_value)

            # Special handling for ParticleIDs, ParticleIDGenerationNumber, and ParticleChildIDsNumber {{{
            if variable_name in ['ParticleIDs', 'ParticleIDGenerationNumber', 'ParticleChildIDsNumber']:
                if variable_name == 'ParticleIDs':
                    new_particle_ids.extend([new_id_counter + i for i in range(len(interpolated_data))])
                    new_id_counter += len(interpolated_data)
                elif variable_name == 'ParticleIDGenerationNumber':
                    new_particle_id_gen_nums.extend(interpolated_data.astype(int))
                elif variable_name == 'ParticleChildIDsNumber':
                    new_particle_child_ids_nums.extend(interpolated_data.astype(int))
                #}}}
            else:
            # Store the interpolated data in the new dictionary
                new_data_dict[variable_name] = interpolated_data

    # Special handling for ParticleIDs, ParticleIDGenerationNumber, and ParticleChildIDsNumber {{{
    # If new Particle IDs were created, add them to the new data dictionary
    if 'ParticleIDs' in data_dict:
        new_data_dict['ParticleIDs'] = np.array(new_particle_ids)

    # If new ParticleIDGenerationNumber were created, add them to the new data dictionary
    if 'ParticleIDGenerationNumber' in data_dict:
        new_data_dict['ParticleIDGenerationNumber'] = np.array(new_particle_id_gen_nums)

    # If new ParticleChildIDsNumber were created, add them to the new data dictionary
    if 'ParticleChildIDsNumber' in data_dict:
        new_data_dict['ParticleChildIDsNumber'] = np.array(new_particle_child_ids_nums)
    #}}}


    if 'debugging' in flags:
        print("Interpolation complete.")

    return new_data_dict
#}}}
