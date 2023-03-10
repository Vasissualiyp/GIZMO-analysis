#This code was written by Vasilii Pustovoit, CITA, 2023.02
#import libraries {{{
import os 
import numpy as np 
import yt 
import unyt
from PIL import Image
import matplotlib.pyplot as plt #}}}

#--------------------------------START OF EDITABLE PART-------------------------------
# Choose what kind of a plot you want to create:
# Possibilities: density_profile; density; temperature
plottype='temperature' 

# In/Out Directories
input_dir='./snapshot2/'
out_dir='./temperature_plots_merged/'

#Units
time_units='Myr'
boxsize_units='Mpc'
density_units='g/cm**3'
temperature_units='K'

# For 2D plots (plottype = temperature, density)
axis_of_projection='y'

#Enabling of different parts of the code
SinglePlotMode=True
plotting=True
#For 2D plots
colorbarlims=False
custom_center=False 
#For double plotting
double_plot=True
InitialPlotting=False

#color map limits
clrmin=2e-3
clrmax=1e-1

#Constants
HubbleParam = 0.7

#For 2-plot mode, names of intermediate folders: 
#{{{
input_dir1  ='./snapshot2/'
input_dir2  ='./snapshot2/'
plottype1='temperature' #possibilities: density_profile; density
plottype2='density' #possibilities: density_profile; density
out_dir1    ='./temperature_plots/'
out_dir2    ='./snapshot2_plots2/'
#}}}

#---------------------------------END OF EDITABLE PART--------------------------------

# Array of units {{{
units = []
units.append(time_units)
units.append(boxsize_units)
units.append(density_units)
units.append(temperature_units)
#}}}

# Functions definitions {{{
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
            plt.annotate_title("Density Plot, t={:.2g}".format(time_yrs), " ", time_units)  
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
        # annotate the plot {{{
        # Set the time units
        time_yrs=code_time * 0.978 / HubbleParam * unyt.Gyr
        time_yrs=time_yrs.to_value(time_units)
        # annotate
        plt.title('Density Profile, t={:.2g}'.format(time_yrs) + ' ' + time_units)
        plt.xlabel('x, ' + boxsize_units)
        plt.ylabel('density, ' + density_units ) #}}}
#}}}

#The function that creates plots for specified directories with the specified parameters {{{
def snap_to_plot(input_dir, out_dir, plottype, units): 
    
    #Put the units in {{{
    time_units = units[0]
    boxsize_units = units[1]
    density_units = units[2]
    temperature_units = units[3]
    #}}}

    # List the contents of the input folder  {{{
    contents = os.listdir(input_dir) 
    
    # Count the number of snapshots by counting the number of files with a .hdf5 extension 
    num_snapshots = sum(1 for item in contents if item.endswith('.hdf5')) 
    #}}}
    
    #Working with the output directory {{{
    #Option for only working on a single snapshot
    if SinglePlotMode==True:
        num_snapshots=1
        out_dir='./single'
    
    # Create the output folder if it does not exist 
    if not os.path.exists(out_dir): 
        os.makedirs(out_dir) 
    #}}}
    
    # Loop through every snapshot {{{
    for i in range(num_snapshots): 
         
        snapno=int_to_str(i,100) 
     
        # Load the snapshot {{{
        filename=input_dir+'snapshot_'+snapno+'.hdf5' 
        ds = yt.load(filename) 
    
            # Adjust the plot center {{{
        if custom_center==True:
            # Compute the center of the sphere
            plot_center = ds.arr([0.5, 0.5, 0.5], "code_length")
        else:
            plot_center = ds.arr([0, 0, 0], "code_length") #}}}
            
        # Make a plot {{{
        #2D Density plot {{{
        if plottype=='density':
            #Create Plot {{{
            p = yt.ProjectionPlot(ds, axis_of_projection,  ("gas", "density"), center=plot_center)
            
            #Set colorbar limits
            if colorbarlims==True:
                p.set_zlim(("gas", "density"), zmin=(clrmin, "g/cm**2"), zmax=(clrmax, "g/cm**2"))
            #}}}
        
            annotate(ds, p, plottype, units)
        #}}}
    
        #2D Temperature plot {{{
        if plottype=='temperature':
            #Create Plot {{{
            p = yt.ProjectionPlot(ds, axis_of_projection,  ("gas", "temperature"), center=plot_center)
            
            #Set colorbar limits
            if colorbarlims==True:
                p.set_zlim(("gas", "temperature"), zmin=(clrmin, "K"), zmax=(clrmax, "K"))
            #}}}
        
            annotate(ds, p, plottype, units)
        #}}}
    
        #1D density profile plot {{{
        elif plottype=='density_profile':
            ad=ds.r[:,1,1] #Only look at a slice in y=z=1
            #Load the data {{{
            x=ad[('gas','x')]
            #y=ad[('gas','y')]
            density=ad[('gas','density')]
            print(density) 
            #Put the data into the appropriate units
            x_plot=np.array(x.to(boxsize_units))
            #y_plot=np.array(y.to(boxsize_units))     
            density_plot=np.array(density.to(density_units)) 
            
            #Sort the data in increasing order
            sorted_indecies=np.argsort(x_plot)
            x_plot=x_plot[sorted_indecies]
            density_plot=density_plot[sorted_indecies]
            #}}}
            #Create plot {{{
            #plt.scatter(x_plot, density_plot)  
            plt.plot(x_plot,density_plot)
            plt.yscale('log')

            annotate(ds, plt, plottype, units)
            
        #}}}
       #}}}
    
       #Save the plot / Output {{{
        if plotting==True:
            if plottype=='density':
                p.save(out_dir+'plot'+snapno+'.png') 
            elif plottype=='density_profile':
                plt.savefig(out_dir+'plot'+snapno+'.png')
                plt.clf()
            elif plottype=='temperature':
                p.save(out_dir+'plot'+snapno+'.png')
        else:
            print("{:.2g}".format(time_yrs)," " + time_units) #}}}
    #}}}
#}}}

# This function combines two plots into a single one {{{
def combine_snapshots(folder1, folder2, output_folder):
    """
    Combines snapshots from folder1 and folder2 into a single picture with one picture on the left and another one on the right.
    The resulting images are saved in output_folder.
    """
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
#}}}
#}}}
#}}}
#}}}

if double_plot==False:
    snap_to_plot(input_dir,out_dir,plottype)
elif double_plot==True:
    if InitialPlotting==True:
        snap_to_plot(input_dir1,out_dir1,plottype1)
        snap_to_plot(input_dir2,out_dir2,plottype2)
    combine_snapshots(out_dir1, out_dir2, out_dir)
