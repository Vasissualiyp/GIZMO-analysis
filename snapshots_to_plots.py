#This code was written by Vasilii Pustovoit, CITA, 2023.02
#import libraries {{{
import os 
import sys
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
#plottype='density' 
#group_name='Cloud0000'
group_name=''

# Read system arguments
if len(sys.argv) > 1:
    day_attempt = sys.argv[1]
else:
    day_attempt = '2023.09.11:2/'

# For 2D plots (plottype = temperature, density)
axis_of_projection='z'

ParticleType = 'gas'
redshift = 199
redshift_parttype = str(int(redshift)) + '_' + ParticleType + '/' + axis_of_projection + '_temperature'+ '/'

# Set the plot types
if ParticleType in ['PartType0' , 'gas']:
    plottype = 'weighted_temperature'
    custom_center=[475,495,487]
    #ParticleType = 'PartType0'
elif ParticleType in ['PartType1']:
    #plottype = 'density'
    custom_center=[500,500,500]
    plottype = 'mass-gridded'
    #ParticleType = 'PartType1'
elif ParticleType in ['PartType2' , 'nbody' ]:
    #plottype = 'deposited_density'
    custom_center=[0,0,0]
    plottype = 'mass-gridded'
    flags.append('sph_plotter')
    flags.append('custom_loader')
    #ParticleType = 'PartType2'


# In/Out Directories
input_dir='/fs/lustre/scratch/vpustovoit/MUSIC2/output/' + day_attempt
#out_dir='./densityplots/'
out_dir='/cita/d/www/home/vpustovoit/plots/' + day_attempt + redshift_parttype 

#Units
time_units='redshift'
boxsize_units='Mpc'
density_units='g/cm**3'
temperature_units='K'
velocity_units='km/s'
smoothing_length_units='Mpc'
first_snapshot=0
zoom=20

#color map limits
clrmin=1e-8
clrmax=1e-5
colorbar_lims = (clrmin, clrmax)


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
