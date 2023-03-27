# This is a code that contains 90% of all the plotting 
# Libraries {{{
import os 
import numpy as np 
import yt 
import unyt
from PIL import Image
from flags import get_flags_array
flags = get_flags_array()
from utils import *
import matplotlib.pyplot as plt #}}}

#The function that creates plots for specified directories with the specified parameters 
def snap_to_plot(flags, input_dir, out_dir, plottype, units): 
    
    #Put the units in {{{
    time_units = units[0]
    boxsize_units = units[1]
    density_units = units[2]
    temperature_units = units[3]
    velocity_units = units[4]
    #}}}

    # List the contents of the input folder  {{{
    contents = os.listdir(input_dir) 
    
    # Count the number of snapshots by counting the number of files with a .hdf5 extension 
    num_snapshots = sum(1 for item in contents if item.endswith('.hdf5')) 
    #}}}
    
    #Working with the output directory {{{
    #Option for only working on a single snapshot
    if 'SinglePlotMode' in flags:
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
        if 'custom_center' in flags:
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
            if 'colorbarlims' in flags:
                p.set_zlim(("gas", "density"), zmin=(clrmin, "g/cm**2"), zmax=(clrmax, "g/cm**2"))
            #}}}
        
            annotate(ds, p, plottype, units)
        #}}}
    
        #2D Temperature plot {{{
        if plottype=='temperature':
            #Create Plot {{{
            p = yt.ProjectionPlot(ds, axis_of_projection,  ("gas", "temperature"), center=plot_center)
            
            #Set colorbar limits
            if 'colorbarlims' in flags:
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
            #print(density) 
            #Put the data into the appropriate units
            x_plot=np.array(x.to(boxsize_units))
            #y_plot=np.array(y.to(boxsize_units))     
            density_plot=np.array(density.to(density_units)) 
            
            #Sort the data in increasing order
            sorted_indecies=np.argsort(x_plot)
            x_plot=x_plot[sorted_indecies]
            density_plot=density_plot[sorted_indecies]
            if 'wraparound' in flags:
                x_plot = wrap_second_half(x_plot)
            #}}}
            #Create plot {{{
            #plt.scatter(x_plot, density_plot)  
            plt.plot(x_plot,density_plot)
            #plt.yscale('log')

            annotate(ds, plt, plottype, units)
            
        #}}}
       #}}}
    
        #1D shock velocity profile plot {{{
        elif plottype=='shock_velocity':
            ad=ds.r[:,1,1] #Only look at a slice in y=z=1
            #Load the data {{{
            x=ad[('gas','x')]
            #y=ad[('gas','y')]
            velocity_x=ad[('gas','velocity_x')]
            temperature=ad[('gas','temperature')]
            #Put the data into the appropriate units
            x_plot=np.array(x.to(boxsize_units))
            velocity_x_plot=np.array(velocity_x.to(velocity_units)) 
            
            #}}}
            #Calculate necessary pieces {{{ 
            v_1sq = (gamma + 1)/2 * temperature * R / unyt.g
            v_1sq = np.sqrt(v_1sq)
            v_1sq_plot=np.array(v_1sq.to(velocity_units)) 
            v_2sq = (gamma - 1)**2/2/(gamma + 1) * temperature * R / unyt.g
            v_2sq = np.sqrt(v_2sq)
            v_2sq_plot=np.array(v_2sq.to(velocity_units)) 
            velocity_calc = v_1sq - v_2sq
            #}}}
            # Sort the data in increasing order {{{
            # First sorting
            x_plot, velocity_x_plot, velocity_calc = sort_arrays(x_plot, velocity_x_plot, velocity_calc)
            if 'wraparound' in flags:
                x_plot = wrap_second_half(x_plot)
            # Sorting for nicer plotting
            x_plot, velocity_x_plot, velocity_calc = sort_arrays(x_plot, velocity_x_plot, velocity_calc)
            # Cut off the unnecessary data
            x_plot, velocity_x_plot, velocity_calc = cutoff_arrays(x_plot,0,velocity_x_plot, velocity_calc)
            #}}}
            #Create plot {{{
            #plt.scatter(x_plot, density_plot)  
            plt.plot(x_plot,[abs(v) for v in velocity_x_plot], label ='Measured')
            #plt.plot(x_plot,v_1sq,label ='v1')
            plt.plot(x_plot,[abs(v) for v in velocity_calc],label ='Calculated')
            #plt.yscale('log')
            plt.legend(loc = 'upper right')

            annotate(ds, plt, plottype, units)
            
        #}}}
        #}}}
        #}}}
       
        #Save the plot / Output {{{
        if 'plotting' in flags:
            if (plottype=='density' or plottype=='temperature'):
                p.save(out_dir+'plot'+snapno+'.png') 
            elif (plottype=='density_profile' or plottype=='shock_velocity'):
                plt.savefig(out_dir+'plot'+snapno+'.png')
                plt.clf()
                print(out_dir+'plot'+snapno+'.png')
        else:
            print("{:.2g}".format(time_yrs)," " + time_units) #}}}
    #}}}
#}}}
