# This code contains utility functions that are used in other funcitons
# Libraries {{{
import os 
import h5py
import numpy as np 
import yt 
import unyt
from PIL import Image
import matplotlib.pyplot as plt #}}}

#Constants
HubbleParam = 0.7
gamma = 5.0/3.0
R = 8.31 * unyt.J / unyt.K / unyt.mol



#This function converts the number of snapshot into a string for its name {{{
def int_to_str(i, n): 
    
    n_digits = len(str(n)) 
     
    # Convert the integer to a string 
    s = str(i) 
     
    # Add leading zeros to the string to make it n characters long 
    s = s.zfill(n_digits) 
     
    return s #}}}

# Function for annotating the plots {{{
def annotate(snapshot, plot, plottype, units):
    
    #Put the units in {{{
    time_units = units[0]
    boxsize_units = units[1]
    density_units = units[2]
    temperature_units = units[3]
    velocity_units = units[4]
    smoothing_length_units = units[5]
    axis_of_projection = units[6]
    #}}}

    if plottype=='density':
        print(plottype)
        # annotate the plot {{{
        if time_units=='redshift':
            redshift = float(snapshot.current_redshift) 
            plot.annotate_title("Density Plot, z={:.6g}".format(redshift)) 
        elif time_units=='code':
            code_time = float(snapshot.current_time) 
            plot.annotate_title("Density Plot, t={:.2g}".format(code_time)) 
        else:
            code_time = float(snapshot.current_time) 
            time_yrs=code_time * 0.978*10**9 / HubbleParam * unyt.yr
            time_yrs=time_yrs.to_value(time_units)
            plot.annotate_title("Density Plot, t={:.2g}".format(time_yrs) + " " + time_units)  
    #}}}
    elif plottype=='temperature':
        print(plottype)
        # annotate the plot {{{
        if time_units=='redshift':
            redshift = float(snapshot.current_redshift) 
            plot.annotate_title("Temperature Plot, z={:.6g}".format(redshift)) 
        elif time_units=='code':
            code_time = float(snapshot.current_time) 
            plot.annotate_title("Temperature Plot, t={:.2g}".format(code_time)) 
        else:
            code_time = float(snapshot.current_time) 
            time_yrs=code_time * 0.978*10**9 / HubbleParam * unyt.yr
            time_yrs=time_yrs.to_value(time_units)
            plot.annotate_title("Temperature Plot, t={:.2g}".format(time_yrs), " ", time_units) 
    #}}}
    elif plottype=='smooth_length':
        print(plottype)
        # annotate the plot {{{
        if time_units=='redshift':
            redshift = float(snapshot.current_redshift) 
            plot.annotate_title("Smoothing Lengths Plot, z={:.6g}".format(redshift)) 
        elif time_units=='code':
            code_time = float(snapshot.current_time) 
            plot.annotate_title("Smoothing Lengths Plot, t={:.2g}".format(code_time)) 
        else:
            code_time = float(snapshot.current_time) 
            time_yrs=code_time * 0.978*10**9 / HubbleParam * unyt.yr
            time_yrs=time_yrs.to_value(time_units)
            plot.annotate_title("Smoothing Lengths Plot, t={:.2g}".format(time_yrs) + " "+ time_units) 
    #}}}
    elif plottype=='density_profile':
        print(plottype)
        # annotate the plot {{{
        code_time = float(snapshot.current_time) 
        # Set the time units
        time_yrs=code_time * 0.978 / HubbleParam * unyt.Gyr
        time_yrs=time_yrs.to_value(time_units)
        # annotate
        plot.title('Density Profile, t={:.2g}'.format(time_yrs) + ' ' + time_units)
        plot.xlabel('x, ' + boxsize_units)
        plot.ylabel('density, ' + density_units ) #}}}
    elif plottype=='shock_velocity':
        print(plottype)
        # annotate the plot {{{
        # Set the time units
        code_time = float(snapshot.current_time) 
        time_yrs=code_time * 0.978 / HubbleParam * unyt.Gyr
        time_yrs=time_yrs.to_value(time_units)
        # annotate
        plot.title('Velocity Profile, t={:.2g}'.format(time_yrs) + ' ' + time_units)
        print('The time for annotation is: ' + str(time_yrs))
        plot.xlabel('x, ' + boxsize_units)
        plot.ylabel('velocity, ' + velocity_units ) #}}}
    elif plottype=='smoothing_length_hist':
        print(plottype)
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
    return cut_x, *cut_ys
#}}}

# The funciton that is needed to sort the arrays in 1D based on the values in the coordinate arrays {{{
def sort_arrays(x, *y):
    # sort indices based on x array
    indices = sorted(range(len(x)), key=lambda i: x[i])
    
    # sort all arrays based on sorted indices
    x_sorted = [x[i] for i in indices]
    y_sorted = [[y_arr[i] for i in indices] for y_arr in y]
    
    return x_sorted, *y_sorted
#}}}

# This function combines two plots into a single one {{{
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
def custom_load_noyt(hdf5_file_path, group_name):
    # Open the HDF5 file
    with h5py.File(hdf5_file_path, 'r') as f:
        # Get the group's dataset
        dataset = f[group_name]['PartType0']['Coordinates']
        masses = np.array(f[group_name]['PartType0']['Masses'])
        dens = np.array(f[group_name]['PartType0']['Density'])
        smoothlen = np.array(f[group_name]['PartType0']['SmoothingLength'])

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

def custom_load(hdf5_file_path, group_name):
    # Open the HDF5 file
    with h5py.File(hdf5_file_path, 'r') as f:
        # Get the group's dataset
        dataset = f[group_name]['PartType0']['Coordinates']
        masses = np.array(f[group_name]['PartType0']['Masses'])
        dens = np.array(f[group_name]['PartType0']['Density'])
        smoothlen = np.array(f[group_name]['PartType0']['SmoothingLength'])

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

