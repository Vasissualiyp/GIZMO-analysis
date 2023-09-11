# This is a code that contains 90% of all the
# plotting 
# Libraries {{{
import os 
import numpy as np 
import yt 
#import matplotlib.pyplot as plt
from hdf5converter import *
import unyt
from PIL import Image
from flags import get_flags_array
flags = get_flags_array()
from utils import *
from sk_upscaler import sk_upscaler_main as skup
from fftupscaler import *
from sph_plotter import *
import matplotlib.pyplot as plt 
#}}}

#The function that creates plots for specified directories with the specified parameters {{{
def snap_to_plot(flags, input_dir, out_dir, plottype, units): 
    datax = []
    datay = [[],[]]
    
    if 'units.custom_center' in flags:
        center_xyz = units.custom_center
    else:
        center_xyz = [0, 0, 0]

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
    for i in range(num_snapshots-units.start): 
        datax, datay = plot_for_single_snapshot(flags, input_dir, out_dir, plottype, units, i, (datax, datay))
    #}}}
    return datax, datay
#}}}

# A function that works on individual snapshots {{{
def plot_for_single_snapshot(flags, input_dir, out_dir, plottype, units, i, dataxy): 
    datax, datay = dataxy
    # Set the center {{{
    if 'units.custom_center' in flags:
        center_xyz = units.custom_center
    else:
        center_xyz = [0, 0, 0]
    snapno=int_to_str(i+units.start,100) 
    #}}}
 
    # Load the snapshot {{{
    filename=input_dir+'snapshot_'+snapno+'.hdf5' 
    units.file_name = filename
    if 'sph_plotter' in flags:
        if 'custom_loader' in flags:
            #{{{
            print('Started loading...')
            #units.start_time = time.perf_counter()
            #units.ParticleType = 'PartType1'
            ds, plot_params = custom_load_all_data(filename, units.group_name, units.ParticleType, flags)
            ds, BoxSize = center_and_find_box(ds)
            print(f'BoxSize is: {BoxSize}')
            # Upscaler {{{
            #n_increase=1
            #units.start_time = time.perf_counter()
            ##ds = increase_resolution_with_rbf(ds, n_increase, flags)
            ##print(f"Max smoothing length before upscaling: {max(ds['SmoothingLength'])}")
            #print('Started the upscaling')
            ##ds = skup(ds, n_increase, BoxSize, flags)
            #end_time = time.perf_counter()
            #elapsed_time = end_time - units.start_time
            #print(f"Elapsed time for smooth kernel upscaler: {elapsed_time} seconds")
            #}}}
            x = ds['Coordinates'][:,0]
            y = ds['Coordinates'][:,1]
            z = ds['Coordinates'][:,2]
            # Shift the center of the box using the plot_center array and the dimensions of the box
            # Define the maximum values for x, y, z
            max_x, max_y, max_z = (500, 500, 500 ) #np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z))
            
            # Shift and wrap the coordinates
            x = ((x + center_xyz[0]) + max_x) % (2 * max_x) - max_x
            y = ((y + center_xyz[1]) + max_y) % (2 * max_y) - max_y
            z = ((z + center_xyz[2]) + max_z) % (2 * max_z) - max_z
            
            # Proceed with the rest of your plotting code

            print(f'{np.size(x)} particles were detected of type {units.ParticleType}')
            # Attempt to get density and smoothing lengths {{{
            try:
                smoothing_lengths = ds['SmoothingLength']
                #print(f"Max smoothing length after upscaling: {max(ds['SmoothingLength'])}")
            except: 
                print("Smoothing length not found. Attempting to get the Softening Kernel Radius...")
                try:
                    smoothing_lengths = ds['Softening_KernelRadius']
                except: 
                    print("Softening Kernel Radius not found")
            try:
                density = ds['Density']
            except: 
                print("Density not found. Attempting to find the masses...")
                try:
                    mass = ds['Masses']
                    density = mass / ( smoothing_lengths ** 3 )
                except: 
                    print("Masses not found")
            #}}}
            
            print('Finished loading')
            #end_time = time.perf_counter()
            #elapsed_time = end_time - units.start_time
            #print(f"Elapsed time for all-data loader: {elapsed_time} seconds")
            #}}}
    else:
        if 'custom_loader' in flags:
            #{{{
            ds, plot_params = custom_load(filename, units.group_name)
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
            #}}}
        else:
            ds = yt.load(filename) 
            plot_center = ds.arr(center_xyz, "code_length")
    #}}}
        
    # Make a plot {{{
    units.start_time = time.perf_counter()
    #2D Density plot {{{
    if plottype=='density':
        if 'sph_plotter' in flags:
           plot = sph_density_projection_optimized(x,y,z,density,smoothing_lengths, flags, resolution=200, log_density=True) 
        else:
            #try:
            #ad = ds.all_data()
            #region1 = 'obj["'+ units.ParticleType +'", "density"] > 1e-60'
            #region2 = 'obj["'+ units.ParticleType +'", "density"] < 1e60'
            #region_not_nan = 'np.isfinite(obj["' + units.ParticleType + '", "density"])'
            #not_nan = ad.cut_region(region_not_nan)
            p = yt.ProjectionPlot(ds, units.axis_of_projection,  (units.ParticleType, "density"), center=plot_center)
            #p = yt.ProjectionPlot(ds, units.axis_of_projection, (units.ParticleType, "density"), data_source=ds.r[:], center=plot_center)
            #p.data_source(not_nan)
            p.zoom(units.zoom)
            """
            except KeyError:
                print("\nSPH plot failed. Attempting particle plot...\n")
                if 'custom_loader' in flags:
                    #p = yt.ParticlePlot(ds, 'particle_position_x', 'particle_position_y', ("all", "density"), origin=origin, width=(2,2))
                    print('width: ', width)
                    #p = yt.ParticlePlot(ds, 'particle_position_x', 'particle_position_y', ("all", "density"), width=width, origin='upper-right-window')
                p = yt.ParticlePlot(ds, (units.ParticleType, "particle_position_x"), (units.ParticleType, "particle_position_y"))

            
            """
            #Set colorbar limits
            if 'colorbarlims' in flags:
                p.set_zlim((units.ParticleType, "density"), zmin=(units.clrmin, "g/cm**2"), zmax=(units.clrmax, "g/cm**2"))
        
            annotate(ds, p, plottype, units, flags)
        dim = 2
    #}}}

    #2D Density plot 2. Not sure what this one does {{{
    if plottype=='density-2':
        #Create Plot {{{
        if 'sph_plotter' in flags:
           plot = sph_density_projection_optimized(x,y,z,density,smoothing_lengths, flags, resolution=500) 
        else:
            try:
                p = yt.ProjectionPlot(ds, units.axis_of_projection,  ("PartType2", "density"), center=plot_center)
            except:
                print("\nSPH plot failed. Attempting particle plot...\n")
                if 'custom_loader' in flags:
                    #p = yt.ParticlePlot(ds, 'particle_position_x', 'particle_position_y', ("all", "density"), origin=origin, width=(2,2))
                    print('width: ', width)
                    #p = yt.ParticlePlot(ds, 'particle_position_x', 'particle_position_y', ("all", "density"), width=width, origin='upper-right-window')
                p = yt.ParticlePlot(ds, ("PartType1", "particle_position_x"), ("PartType1", "particle_position_y"))

            
            #Set colorbar limits
            if 'colorbarlims' in flags:
                p.set_zlim(("gas", "density"), zmin=(units.clrmin, "g/cm**2"), zmax=(units.clrmax, "g/cm**2"))
            #}}}
        
            annotate(ds, p, plottype, units, flags)
        dim = 2
    #}}}

    #2D Density plot 3 - tidy (for grid plotting of particles){{{
    if plottype=='density-3':
        #Create Plot {{{
        if 'sph_plotter' and 'custom_loader'in flags:

            # Grid size (e.g., 50x50)
            N = 128
            
            # Determine the histogram of x and y coordinates
            if units.axis_of_projection == 'z':
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
            elif units.axis_of_projection == 'y':
                plt.xlabel('X Coordinate')
                plt.ylabel('Z Coordinate')
                y = z
                x = x
            elif units.axis_of_projection == 'x':
                plt.xlabel('Z Coordinate')
                plt.ylabel('Y Coordinate')
                x = z
                y = y
                
            hist, x_edges, y_edges = np.histogram2d(x, y, bins=N)
            
            #boxsize = 0.05
            # Get the minimum and maximum of x and y for the extent
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            # Create a meshgrid for the x and y coordinates
            #x_mesh, y_mesh = np.meshgrid(boxsize, boxsize)
            
            # Plot the 2D histogram as an image
            plt.imshow(hist.T, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='viridis', aspect='auto')
            #annotate(ds, plt, plottype, units)
            
            plt.title('Density of Points in the Grid')
            plt.colorbar(label='Number of Particles')
            plt.show()
            #sampling_fraction = 2
            #DPI = 100
            #x_resampled = x[::sampling_fraction]
            #y_resampled = y[::sampling_fraction]
            #fig = plt.figure(figsize=(20, 20), dpi=DPI)
            #plt.scatter(x_resampled, y_resampled, s=0.005, alpha=0.7)  # s controls the size of the points, alpha controls the transparency
            #
            #plt.xlabel('X Coordinate')
            #plt.ylabel('Y Coordinate')
            #plt.title('Projection of Points Along the Z Axis')
            #
            #plt.show()
            #plt.savefig(out_dir + )
            #plt.savefig(out_dir+'2Dplot-2'+snapno+'.png', dpi=DPI)
        else:
            try:
                p = yt.ProjectionPlot(ds, units.axis_of_projection,  (units.ParticleType, "density"), center=plot_center)
            except:
                print("\nSPH plot failed. Attempting particle plot...\n")
                if 'custom_loader' in flags:
                    #p = yt.ParticlePlot(ds, 'particle_position_x', 'particle_position_y', ("all", "density"), origin=origin, width=(2,2))
                    print('width: ', width)
                    #p = yt.ParticlePlot(ds, 'particle_position_x', 'particle_position_y', ("all", "density"), width=width, origin='upper-right-window')
                p = yt.ParticlePlot(ds, ("PartType1", "particle_position_x"), ("PartType1", "particle_position_y"))

            
            #Set colorbar limits
            if 'colorbarlims' in flags:
                p.set_zlim(("gas", "density"), zmin=(units.clrmin, "g/cm**2"), zmax=(units.clrmax, "g/cm**2"))
            #}}}
        
        dim = 2
    #}}}

    #2D Mass histogram - tidy (for grid plotting of particles){{{
    if plottype == 'mass-gridded':
        # Create Plot 
        if 'sph_plotter' and 'custom_loader'in flags:
    
            # Grid size (e.g., 128x128)
            N = 128
            plt.figure()
            #plt.title('Projected Mass in the Grid')
        
            # Determine which axis to project
            if units.axis_of_projection == 'z':
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
            elif units.axis_of_projection == 'y':
                plt.xlabel('X Coordinate')
                plt.ylabel('Z Coordinate')
                y = z
            elif units.axis_of_projection == 'x':
                plt.xlabel('Z Coordinate')
                plt.ylabel('Y Coordinate')
                x = z
        
            # Create weighted 2D histogram
            mass_hist, x_edges, y_edges = np.histogram2d(x, y, bins=N, weights=mass)
        
            # Get the minimum and maximum of x and y for the extent
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
        
            # Plot the 2D histogram as an image
            plt.imshow(mass_hist.T, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='viridis', aspect='auto')
        
            plt.colorbar(label='Mass projection, Msun/kpc$^2$')
            annotate(ds, input_dir, plottype, units, flags)
            plt.show()


        else:
            print('Plotting failed. Make sure that flags sph_plotter and custom_loader are enabled')  
            
            #Set colorbar limits
            if 'colorbarlims' in flags:
                p.set_zlim(("gas", "density"), zmin=(units.clrmin, "g/cm**2"), zmax=(units.clrmax, "g/cm**2"))
        
        dim = 2
    #}}}

    #2D density plot for DM - not fully working!{{{
    if plottype=='deposited_density':
        #Create Plot {{{
        if 'sph_plotter' in flags:
           plot = sph_density_projection_optimized(x,y,z,density,smoothing_lengths, flags, resolution=200, log_density=True) 
        else:
            #deposition_field = units.ParticleType + "_" + "mass"
            try:
                deposition_field = ds.add_deposited_particle_field((units.ParticleType, "Masses"), method="cic")
                #print('------------------------------------------------------------')
                #print(f'Deposition field: {deposition_field}')
                #print('------------------------------------------------------------')
                #print(ds.field_list)
                #print('------------------------------------------------------------')
                #p = yt.ProjectionPlot(ds, units.axis_of_projection, 'density', center=plot_center)
                #p = yt.ProjectionPlot(ds, units.axis_of_projection,  deposition_field)
                if units.axis_of_projection in ['x']:
                    p = yt.ParticlePlot(ds, 'particle_position_y', 'particle_position_z', 
                        (units.ParticleType, "Masses"), origin='upper-right-window', center=plot_center)
                elif units.axis_of_projection in ['y']:
                    p = yt.ParticlePlot(ds, 'particle_position_x', 'particle_position_z', 
                        (units.ParticleType, "Masses"), origin='upper-right-window', center=plot_center)
                elif units.axis_of_projection in ['z']:
                    p = yt.ParticlePlot(ds, 'particle_position_x', 'particle_position_y', 
                        (units.ParticleType, "Masses"), origin='upper-right-window', center=plot_center)
                p.zoom(units.zoom)
            except:
                print("Particle plot failed")
            # legacy exceptions handling {{{
            #except:
            #    print("\nSPH plot failed. Attempting particle plot...\n")
            #    if 'custom_loader' in flags:
            #        #p = yt.ParticlePlot(ds, 'particle_position_x', 'particle_position_y', ("all", "density"), origin=origin, width=(2,2))
            #        print('width: ', width)
            #        #p = yt.ParticlePlot(ds, 'particle_position_x', 'particle_position_y', ("all", "density"), width=width, origin='upper-right-window')
            #    p = yt.ParticlePlot(ds, (units.ParticleType1, "particle_position_x"), (units.ParticleType, "particle_position_y"))
            #}}}
            
            #Set colorbar limits
            if 'colorbarlims' in flags:
                p.set_zlim(("gas", "density"), zmin=(units.clrmin, "g/cm**2"), zmax=(units.clrmax, "g/cm**2"))
            #}}}
        
            annotate(ds, p, plottype, units, flags)
        dim = 2
    #}}}

    #2D Temperature plot {{{
    if plottype=='temperature':
        #Create Plot {{{
        p = yt.ProjectionPlot(ds, units.axis_of_projection,  ("gas", "temperature", flags), center=plot_center)
        
        #Set colorbar limits
        if 'colorbarlims' in flags:
            p.set_zlim(("gas", "temperature"), zmin=(units.clrmin, "K"), zmax=(units.clrmax, "K"))
        #}}}
    
        annotate(ds, p, plottype, units, flags)
        dim = 2
    #}}}

    #2D Smoothing Lengths plot {{{
    if plottype=='smooth_length':
        #Create Plot {{{
        p = yt.ProjectionPlot(ds, units.axis_of_projection,  ("gas", "smoothing_length"), center=plot_center)
        p.set_unit(("gas", "smoothing_length"), "Mpc**2" )
        
        #Set colorbar limits
        #if 'colorbarlims' in flags:
        #    p.set_zlim(("gas", "smoothing_length"), zmin=(units.clrmin, "K"), zmax=(units.clrmax, "K"))
        #}}}
        print(p)
    
        annotate(ds, p, plottype, units, flags)
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
        x_plot=np.array(x.to(units.boxsize_units))
        #y_plot=np.array(y.to(units.boxsize_units))     
        density_plot=np.array(density.to(units.density_units)) 
        
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

        annotate(ds, plt, plottype, units, flags)
        
    #}}}
        dim = 1
   #}}}

    #1D shock velocity profile plot {{{
    elif plottype=='shock_velocity':
        ad=ds.r[:,1,1] #Only look at a slice in y=z=1
        # Set the time units {{{
        code_time = float(ds.current_time) 
        time_yrs=code_time * 0.978 / HubbleParam * unyt.Gyr
        time_yrs=time_yrs.to_value(units.time_units)

        #}}}
        #Load the data {{{
        x=ad[('gas','x')]
        #y=ad[('gas','y')]
        velocity_x=ad[('gas','velocity_x')]
        temperature=ad[('gas','temperature')]
        #Put the data into the appropriate units
        x_plot=np.array(x.to(units.boxsize_units))
        velocity_x_plot=np.array(velocity_x.to(units.velocity_units)) 
        
        #}}}
        #Calculate necessary pieces {{{ 
        v_1sq = (gamma + 1)/2 * temperature * R / unyt.g / 2
        v_1sq = np.sqrt(v_1sq)
        v_1sq_plot=np.array(v_1sq.to(units.velocity_units)) 
        v_2sq = (gamma - 1)**2/2/(gamma + 1) * temperature * R / unyt.g / 2
        v_2sq = np.sqrt(v_2sq)
        v_2sq_plot=np.array(v_2sq.to(units.velocity_units)) 
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
            threshold = threshold.to_value(units.time_units)
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

        annotate(ds, plt, plottype, units, flags)
        
    #}}}
        dim = 1
    #}}}

    #1D smoothing lengths histogram{{{
    elif plottype=='smoothing_length_hist':
        # Define gas slice
        gas_slice = ds.r[:, :, :]  # Change the region of interest

        # Get smoothing lengths
        smoothing_lengths = gas_slice[('gas', 'smoothing_length')]
        smoothing_lengths =np.array(smoothing_lengths.to(units.smoothing_length_units)) 
        #print(smoothing_lengths)

        # Create histogram
        plt.hist(smoothing_lengths, bins=50, color='blue', edgecolor='black')


        annotate(ds, plt, plottype, units, flags)
        
        dim = 1
    #}}}
    end_time = time.perf_counter()
    elapsed_time = end_time - units.start_time
    print(f"Elapsed time for plotter: {elapsed_time} seconds")
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
        print("{:.2g}".format(time_yrs)," " + units.time_units) #}}}
    return datax, datay
#}}}
