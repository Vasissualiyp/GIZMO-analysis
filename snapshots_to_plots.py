#This code was written by Vasilii Pustovoit, CITA, 2023.02
#import libraries {{{
import os 
import numpy as np 
import yt 
import unyt
from PIL import Image
from scipy.optimize import curve_fit
from funcdef_snap_to_plot import *
from utils import *
from sph_plotter import *
from flags import get_flags_array
flags = get_flags_array()
import matplotlib.pyplot as plt #}}}

#--------------------------------START OF EDITABLE PART-------------------------------
# For flags, go to flags.py

# Choose what kind of a plot you want to create:
# Possibilities: density_profile; density; temperature
plottype='density' 
group_name='Cloud0000'

# In/Out Directories
input_dir='/fs/lustre/scratch/vpustovoit/GMC/CP3/CloudPhinder2023/phindertest609/'
#out_dir='./densityplots/'
out_dir='/cita/d/www/home/vpustovoit/'

#Units
time_units='Myr'
boxsize_units='Mpc'
density_units='g/cm**3'
temperature_units='K'
velocity_units='km/s'
smoothing_length_units='Mpc'

# For 2D plots (plottype = temperature, density)
axis_of_projection='z'

#color map limits
clrmin=2e-3
clrmax=1e-1

#For 2-plot mode, names of intermediate folders: 
#{{{
input_dir1  ='../starform_23-03-03/snapshot2/'
input_dir2  ='../starform_23-03-03/snapshot2/'
plottype1='smooth_length' #possibilities: density_profile; density
plottype2='smoothing_length_hist' #possibilities: density_profile; density
out_dir1    ='./smooth_length_plot/'
out_dir2    ='./smoothing_length/'
#}}}

#---------------------------------END OF EDITABLE PART--------------------------------
def linear_func(x, k, b):
    return k * x + b

# Array of units {{{
units = []
units.append(time_units)
units.append(boxsize_units)
units.append(density_units)
units.append(temperature_units)
units.append(velocity_units)
units.append(smoothing_length_units)
units.append(axis_of_projection)
units.append(group_name)
#}}}

if 'double_plot' in flags:
    if 'InitialPlotting' in flags:
        x1,y1 = snap_to_plot(flags, input_dir1,out_dir1,plottype1, units)
        #x2,y2 = snap_to_plot(flags, input_dir2,out_dir2,plottype2, units)
    combine_snapshots(out_dir, out_dir1, out_dir2)
    if 'Time_Dependent' in flags: #{{{
        plt.scatter(x2,y2[0], s = 3, label = 'Max gas velocity location')
        plt.scatter(x2,y2[1], s = 3, label = 'Max shock velocity location')
        #print(x2)
        #print(y2)
        plt.title('Shock max')
        plt.xlabel('time, ' + time_units ) 
        plt.ylabel('x, ' + boxsize_units)
        x = np.array(x2)
        y = np.array(y2[0])
        # find the index where x = 2000
        index_2000 = np.max(np.where(x < 2000))

        # fit the linear function to the data after x = 2000
        popt, pcov = curve_fit(linear_func, x[index_2000:], y[index_2000:])

        # the first parameter of popt is the slope k
        k = popt[0]*3.086e+19/3.154e+13

        print("The velocity of the shockwave is:", k, "km/s")
        plt.plot(x, linear_func(x, *popt), 'r-', label='fit')
        plt.legend()
        plt.savefig('shockmax.png')
    #}}}
else:
    x1,y1 = snap_to_plot(flags,input_dir,out_dir,plottype, units)

# Cleanup the global variables
for var in list(globals()):
    del globals()[var]
