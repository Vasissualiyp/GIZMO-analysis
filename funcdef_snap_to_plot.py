# This is a code that contains 90% of all the
# plotting 
# Libraries {{{
import os 
import numpy as np 
import yt 
import unyt
from PIL import Image
from flags import get_flags_array
flags = get_flags_array()
from utils import *
from sph_plotter import *
import matplotlib.pyplot as plt #}}}

#The function that creates plots for specified directories with the specified parameters 
def snap_to_plot(flags, input_dir, out_dir, plottype, units): 
    datax = []
    datay = [[],[]]
    
    #Put the units in {{{
    time_units = units[0]
    boxsize_units = units[1]
    density_units = units[2]
    temperature_units = units[3]
    velocity_units = units[4]
    smoothing_length_units = units[5]
    axis_of_projection = units[6]
    group_name = units[7]
    #}}}

    start = 0

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
    else:
        # Create the output folder if it does not exist 
        if not os.path.exists(out_dir): 
            os.makedirs(out_dir) 
    #}}}
    
    # Loop through every snapshot {{{
    for i in range(num_snapshots-start): 
         
        snapno=int_to_str(i+start,100) 
     
        # Load the snapshot {{{
        filename=input_dir+'snapshot_'+snapno+'.hdf5' 
        if 'sph_plotter' in flags:
            if 'custom_loader' in flags:
                #start_time = time.perf_counter()
                ds, plot_params = custom_load_all_data(filename, group_name)
                x = ds['Coordinates'][:,0]
                y = ds['Coordinates'][:,1]
                z = ds['Coordinates'][:,2]
                density = ds['Density']
                smoothing_lengths = ds['SmoothingLength']
                #end_time = time.perf_counter()
                #elapsed_time = end_time - start_time
                #print(f"Elapsed time for all-data loader: {elapsed_time} seconds")
        else:
            #{{{
            if 'custom_loader' in flags:
                ds, plot_params = custom_load(filename, group_name)
                left = plot_params[0,:]
                right = plot_params[1,:]
                origin = np.zeros(2)
                width = origin
                for i in range(0,2):
                    width[i] = (right[i] - left[i])
                    origin[i] = plot_params[2,i]
                    width[i] = width[i] * 2.1
                width = tuple(width)
                print(width)
            else:
                ds = yt.load_particles(filename) 
            
                # Adjust the plot center {{{
            if 'custom_center' in flags:
                # Compute the center of the sphere
                plot_center = ds.arr([0.5, 0.5, 0.5], "code_length")
            else:
                plot_center = ds.arr([0, 0, 0], "code_length") #}}}
            #}}}
            
        # Make a plot {{{
        #2D Density plot {{{
        if plottype=='density':
            #Create Plot {{{
            if 'sph_plotter' in flags:
               plt = sph_density_projection_optimized(x,y,z,density,smoothing_lengths, resolution=500) 
            else:
                #try:
                    #p = yt.ProjectionPlot(ds, axis_of_projection,  ("all", "density"), center=plot_center)
                #except:
                    #print("\nSPH plot failed. Attempting particle plot...\n")
                if 'custom_loader' in flags:
                    #p = yt.ParticlePlot(ds, 'particle_position_x', 'particle_position_y', ("all", "density"), origin=origin, width=(2,2))
                    print('width: ', width)
                    p = yt.ParticlePlot(ds, 'particle_position_x', 'particle_position_y', ("all", "density"), width=width, origin='upper-right-window')

                
                #Set colorbar limits
                if 'colorbarlims' in flags:
                    p.set_zlim(("gas", "density"), zmin=(clrmin, "g/cm**2"), zmax=(clrmax, "g/cm**2"))
                #}}}
            
                annotate(ds, p, plottype, units)
            dim = 2
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
            dim = 2
        #}}}

        #2D Smoothing Lengths plot {{{
        if plottype=='smooth_length':
            #Create Plot {{{
            p = yt.ProjectionPlot(ds, axis_of_projection,  ("gas", "smoothing_length"), center=plot_center)
            p.set_unit(("gas", "smoothing_length"), "Mpc**2" )
            
            #Set colorbar limits
            #if 'colorbarlims' in flags:
            #    p.set_zlim(("gas", "smoothing_length"), zmin=(clrmin, "K"), zmax=(clrmax, "K"))
            #}}}
            print(p)
        
            annotate(ds, p, plottype, units)
            dim = 2
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
            
            #}}}
            # Sort the data in increasing order {{{
            # First sorting
            x_plot, density_plot = sort_arrays(x_plot, density_plot)
            if 'wraparound' in flags:
                x_plot = wrap_second_half(x_plot)
            # Sorting for nicer plotting
            x_plot, density_plot = sort_arrays(x_plot, density_plot)
            # Cut off the unnecessary data
            x_plot, density_plot = cutoff_arrays(x_plot,0,density_plot)

            #}}}
            #Create plot {{{
            #plt.scatter(x_plot, density_plot)  
            plt.plot(x_plot,density_plot)
            #plt.yscale('log')

            annotate(ds, plt, plottype, units)
            
        #}}}
            dim = 1
       #}}}
    
        #1D shock velocity profile plot {{{
        elif plottype=='shock_velocity':
            ad=ds.r[:,1,1] #Only look at a slice in y=z=1
            # Set the time units {{{
            code_time = float(ds.current_time) 
            time_yrs=code_time * 0.978 / HubbleParam * unyt.Gyr
            time_yrs=time_yrs.to_value(time_units)

            #}}}
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
            v_1sq = (gamma + 1)/2 * temperature * R / unyt.g / 2
            v_1sq = np.sqrt(v_1sq)
            v_1sq_plot=np.array(v_1sq.to(velocity_units)) 
            v_2sq = (gamma - 1)**2/2/(gamma + 1) * temperature * R / unyt.g / 2
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
            # Get time-dependent data {{{
            if 'Time_Dependent' in flags:
                threshold = 4.4 * 10**2 * unyt.Myr
                threshold = threshold.to_value(time_units)
                if time_yrs > threshold:
                    print('Time after sorting is: ' + str(time_yrs))
                    datax.append(time_yrs)
                    print(datax)
                    #Find the max of gas velocity
                    x = x_plot
                    y = [abs(v) for v in velocity_x_plot]
                    max_idx = np.argmax(y)
                    max_x = x[max_idx] 
                    datay[0].append(max_x)
                    plt.axvline(x=max_x, color='r', linestyle='--', label='Max gas v') 
                    #Find the max of shock velocity
                    y = [abs(v) for v in velocity_calc]
                    max_idx = np.argmax(y)
                    max_x = x[max_idx] 
                    datay[1].append(max_x)
                    plt.axvline(x=max_x, color='r', linestyle='--', label='Max shock v') 
            #}}}
            plt.legend(loc = 'upper right')

            annotate(ds, plt, plottype, units)
            
        #}}}
            dim = 1
        #}}}

        #1D smoothing lengths histogram{{{
        elif plottype=='smoothing_length_hist':
            # Define gas slice
            gas_slice = ds.r[:, :, :]  # Change the region of interest

            # Get smoothing lengths
            smoothing_lengths = gas_slice[('gas', 'smoothing_length')]
            smoothing_lengths =np.array(smoothing_lengths.to(smoothing_length_units)) 
            #print(smoothing_lengths)

            # Create histogram
            plt.hist(smoothing_lengths, bins=50, color='blue', edgecolor='black')


            annotate(ds, plt, plottype, units)
            
            dim = 1
        #}}}
        #}}}
        #}}}
       
        #Save the plot / Output {{{
        if (('plotting' in flags) and ('sph_plotter' not in flags)):
            if dim == 2:
                print('p')
                p.save(out_dir+'2Dplot'+snapno+'.png') 
                print(out_dir+'2Dplot'+snapno+'.png')
            elif dim == 1:
                print('plt')
                plt.savefig(out_dir+'1Dplot'+snapno+'.png')
                plt.clf()
                print(out_dir+'1Dplot'+snapno+'.png')
            else:
                print("Dimensionality not given")
        elif 'sph_plotter' in flags:
            print('plt')
            plt.savefig(out_dir+'2Dplot'+snapno+'.png')
            plt.clf()
            print(out_dir+'2Dplot'+snapno+'.png')
        elif 'plotting' not in flags:
            print("{:.2g}".format(time_yrs)," " + time_units) #}}}
    #}}}

    return datax, datay
#}}}
