# This code contains utility functions that are used in other funcitons
# Libraries {{{
import os 
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
def annotate(snapshot, plt, plottype, units):
    
    #Put the units in {{{
    time_units = units[0]
    boxsize_units = units[1]
    density_units = units[2]
    temperature_units = units[3]
    velocity_units = units[4]
    #}}}

    if plottype=='density':
        # annotate the plot {{{
        if time_units=='redshift':
            redshift = float(snapshot.current_redshift) 
            plt.annotate_title("Density Plot, z={:.6g}".format(redshift)) 
        elif time_units=='code':
            code_time = float(snapshot.current_time) 
            plt.annotate_title("Density Plot, t={:.2g}".format(code_time)) 
        else:
            code_time = float(snapshot.current_time) 
            time_yrs=code_time * 0.978*10**9 / HubbleParam * unyt.yr
            time_yrs=time_yrs.to_value(time_units)
            plt.annotate_title("Density Plot, t={:.2g}".format(time_yrs) + " " + time_units)  
    #}}}
    elif plottype=='temperature':
        # annotate the plot {{{
        if time_units=='redshift':
            redshift = float(snapshot.current_redshift) 
            plt.annotate_title("Temperature Plot, z={:.6g}".format(redshift)) 
        elif time_units=='code':
            code_time = float(snapshot.current_time) 
            plt.annotate_title("Temperature Plot, t={:.2g}".format(code_time)) 
        else:
            code_time = float(snapshot.current_time) 
            time_yrs=code_time * 0.978*10**9 / HubbleParam * unyt.yr
            time_yrs=time_yrs.to_value(time_units)
            plt.annotate_title("Temperature Plot, t={:.2g}".format(time_yrs), " ", time_units) 
    #}}}
    elif plottype=='density_profile':
        code_time = float(snapshot.current_time) 
        # annotate the plot {{{
        # Set the time units
        time_yrs=code_time * 0.978 / HubbleParam * unyt.Gyr
        time_yrs=time_yrs.to_value(time_units)
        # annotate
        plt.title('Density Profile, t={:.2g}'.format(time_yrs) + ' ' + time_units)
        plt.xlabel('x, ' + boxsize_units)
        plt.ylabel('density, ' + density_units ) #}}}
    elif plottype=='shock_velocity':
        code_time = float(snapshot.current_time) 
        # annotate the plot {{{
        # Set the time units
        time_yrs=code_time * 0.978 / HubbleParam * unyt.Gyr
        time_yrs=time_yrs.to_value(time_units)
        # annotate
        plt.title('Velocity Profile, t={:.2g}'.format(time_yrs) + ' ' + time_units)
        plt.xlabel('x, ' + boxsize_units)
        plt.ylabel('velocity, ' + velocity_units ) #}}}
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
        print('Combining images... ' + str(percent) + "%")
    print('Combining images... Done!')
#}}}

