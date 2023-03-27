#This code was written by Vasilii Pustovoit, CITA, 2023.02
#import libraries {{{
import os 
import numpy as np 
import yt 
import unyt
from PIL import Image
from funcdef_snap_to_plot import *
from utils import *
from flags import get_flags_array
flags = get_flags_array()
import matplotlib.pyplot as plt #}}}

#--------------------------------START OF EDITABLE PART-------------------------------
# For flags, go to flags.py

# Choose what kind of a plot you want to create:
# Possibilities: density_profile; density; temperature
plottype='shock_velocity' 

# In/Out Directories
input_dir='../starform_23-03-03/snapshot2/'
out_dir='./shock_velocity/'

#Units
time_units='Myr'
boxsize_units='Mpc'
density_units='g/cm**3'
temperature_units='K'
velocity_units='km/s'

# For 2D plots (plottype = temperature, density)
axis_of_projection='y'

#color map limits
clrmin=2e-3
clrmax=1e-1

#For 2-plot mode, names of intermediate folders: 
#{{{
input_dir1  ='../starform_23-03-03/snapshot2/'
input_dir2  ='../starform_23-03-03/snapshot2/'
plottype1='shock_velocity' #possibilities: density_profile; density
plottype2='density' #possibilities: density_profile; density
out_dir1    ='./shockonly_velocity/'
out_dir2    ='./density/'
#}}}

#---------------------------------END OF EDITABLE PART--------------------------------

# Array of units {{{
units = []
units.append(time_units)
units.append(boxsize_units)
units.append(density_units)
units.append(temperature_units)
units.append(velocity_units)
#}}}

if 'double_plot' in flags:
    if 'InitialPlotting' in flags:
        snap_to_plot(flags, input_dir1,out_dir1,plottype1, units)
        #snap_to_plot(input_dir2,out_dir2,plottype2, units)
    combine_snapshots(out_dir, out_dir1, out_dir2)
else:
    snap_to_plot(flags,input_dir,out_dir,plottype, units)
