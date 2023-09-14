# This is a code that contains 90% of all the
# plotting 
# Libraries {{{
from multiprocessing import Pool
import multiprocessing
import logging
import contextlib
import os 
import functools
import numpy as np 
import yt 
#import matplotlib.pyplot as plt
from hdf5converter import *
from yt.visualization.volume_rendering.api import (Camera, Scene, create_volume_source)
import unyt
from PIL import Image
from flags import get_flags_array
flags = get_flags_array()
from utils import *
from sk_upscaler import sk_upscaler_main as skup
from fftupscaler import *
from sph_plotter import *
import matplotlib.pyplot as plt 
import time
import sys
#}}}

# Assuming you want to use all available cores
num_cores = multiprocessing.cpu_count() # Set up for multiprocessing

logging.getLogger('yt').setLevel(logging.CRITICAL) # Disable yt messaging (pretty annoying)

# Class with all data, related to the snapshot {{{
class SnapshotData:
    def __init__(self, particle_positions=(None, None, None), dataxy=None, plot=None, plot_center=None, plottype=None, dim=None, time_yrs=None):
        self.x, self.y, self.z = particle_positions
        self.dataxy = dataxy
        self.plot = plot
        self.plot_center = plot_center
        self.plottype = plottype
        self.dim = dim
        self.time_yrs = time_yrs
#}}}

#The function that creates plots for specified directories with the specified parameters {{{
def snap_to_plot(flags, input_dir, out_dir, plottype, units): 
    datax = []
    datay = [[],[]]
    max_time = 6 * 60 * 60 # Define max time (in seconds) that
    snapdata = SnapshotData()
    snapdata.dataxy = (datax, datay)

    num_snapshots=get_number_of_snapshots(input_dir)
    if 'eternal_plotting' in flags:
    # Eternal plotting mode {{{
        i=0
        time_since_snap=0
        while True:
            num_snapshots=get_number_of_snapshots(input_dir)
            if i < num_snapshots - units.start:
                time.sleep(5)
                snapdata = plot_for_single_snapshot(flags, input_dir, out_dir, plottype, units, i, snapdata)
                i+=1
                time_since_snap=0
            else:
                print_time_since_last_snapshot(time_since_snap, max_time)
    #}}}
    else:
        # Loop through every snapshot {{{
        # Create a pool of worker processes
        with Pool(num_cores) as pool:
            # Use functools.partial or lambda to pass additional arguments to parallel_task
            func = functools.partial(parallel_task, flags=flags, input_dir=input_dir, out_dir=out_dir, plottype=plottype, units=units, snapdata=snapdata)
            results = pool.map(func, range(num_snapshots-units.start))
        
        # Process results to update datax and datay
        for datax_new, datay_new in results:
            # Update datax and datay if necessary
            # If you simply want to append or extend the arrays:
            datax.extend(datax_new)
            datay.extend(datay_new)
        #}}}
    return snapdata
#}}}

# Loop for parallelization {{{
def parallel_task(i, flags, input_dir, out_dir, plottype, units, snapdata):
    num_snapshots = get_number_of_snapshots(input_dir)
    snapdata_new = plot_for_single_snapshot(flags, input_dir, out_dir, plottype, units, i, snapdata)
    dataxy = snapdata_new.dataxy
    return dataxy
    #}}}

# A function that works on individual snapshots {{{
def plot_for_single_snapshot(flags, input_dir, out_dir, plottype, units, i, snapdata): 
    datax, datay = snapdata.dataxy
    snapdata.plottype = plottype
    # Set the center {{{
    if 'custom_center' in flags:
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
            # Load the snapshot
            ds = yt_snapshot_loader(filename, snapdata, snapno)
    #}}}
        
    # Make a plot
    plot_maker(ds, snapdata, units, flags, snapno)
    #}}}

    #Save the plot / Output {{{
    save_plot(snapdata, units, flags, snapno, out_dir)
    return snapdata
#}}}

# Get current number of snapshots in the folder {{{
def get_number_of_snapshots(input_dir):
    # List the contents of the input folder
    contents = os.listdir(input_dir) 
    
    # Count the number of snapshots by counting the number of files with a .hdf5 extension 
    num_snapshots = sum(1 for item in contents if item.endswith('.hdf5')) 

    return num_snapshots
#}}}

# Snapshot loaders {{{
def yt_snapshot_loader(filename, snapdata,snapno):
    ds = yt.load(filename) 
    print(f"Successfully loaded snapshot #{snapno}")
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            dd = ds.all_data()
            center_of_mass = dd.quantities["CenterOfMass"]()
            center_of_mass_kpc = center_of_mass.to(unyt.kpc)
    # Set the center of plot to the center of mass of the snapshot
    snapdata.plot_center = ds.arr(center_of_mass_kpc, "code_length")
    return ds
#}}}

# Plot distributor {{{
def plot_maker(ds, snapdata, units, flags, snapno):
    plottype = snapdata.plottype
    start_time = time.perf_counter()
    if plottype=='density': # TESTED
        density_plotting(ds, snapdata, units, flags)
    elif plottype=='density-2':
        density2_plotting(ds, snapdata, units, flags)
    elif plottype=='density-3':
        density3_plotting(ds, snapdata, units, flags)
    elif plottype == 'mass-gridded':
        mass_grid(ds, snapdata, units, flags)
    elif plottype=='temperature':
        temperature(ds, snapdata, units, flags)
    elif plottype=='weighted_temperature':
        weighted_temperature(ds, snapdata, units, flags)
    elif plottype=='volumetric_density':
        volumetric_density(ds, snapdata, units, flags, snapno)
    elif plottype=='deposited_density':
        deposition_density(ds, snapdata, units, flags)
    elif plottype=='smoothing_length':
        smoothing_length(ds, snapdata, units, flags)
    elif plottype=='density_profile':
        density_profile(ds, snapdata, units, flags)
    elif plottype=='shock_velocity':
        shock_velocity(ds, snapdata, units, flags)
    elif plottype=='smoothing_length_hist':
        smoothing_length_hist(ds, snapdata, units, flags)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for plotter: {elapsed_time} seconds on snapshot #{snapno}")
#}}}

# Different plot types {{{
# 2D Density plot {{{
def density_plotting(ds, snapdata, units, flags):
    if 'sph_plotter' in flags:
       plot = sph_density_projection_optimized(x,y,z,density,smoothing_lengths, flags, resolution=200, log_density=True) 
    else:
        p = yt.ProjectionPlot(ds, units.axis_of_projection,  (units.ParticleType, "density"), center=snapdata.plot_center)
        #p.set_cmap('inferno')
        p.zoom(units.zoom)

        #Set colorbar limits
        if 'colorbarlims' in flags:
            p.set_zlim((units.ParticleType, "density"), zmin=(units.clrmin, "g/cm**2"), zmax=(units.clrmax, "g/cm**2"))

        snapdata.plot = p
    
        annotate(ds, snapdata.plot, snapdata.plottype, units, flags)
    snapdata.dim = 2
#}}}
# 2D Density plot 2 (Have no idea what this one does) {{{
def density2_plotting(ds, snapdata, units, flags):
    if 'sph_plotter' in flags:
       plot = sph_density_projection_optimized(x,y,z,density,smoothing_lengths, flags, resolution=500) 
    else:
        try:
            p = yt.ProjectionPlot(ds, units.axis_of_projection,  ("PartType2", "density"), center=snapdata.plot_center)
        except:
            print("\nSPH plot failed. Attempting particle plot...\n")
            if 'custom_loader' in flags:
                #p = yt.ParticlePlot(ds, 'particle_position_x', 'particle_position_y', ("all", "density"), origin=origin, width=(2,2))
                print('width: ', width)
                #p = yt.ParticlePlot(ds, 'particle_position_x', 'particle_position_y', ("all", "density"), width=width, origin='upper-right-window')
            p = yt.ParticlePlot(ds, ("PartType1", "particle_position_x"), ("PartType1", "particle_position_y"))

        
        #Set colorbar limits
        if 'colorbarlims' in flags:
            p.set_zlim((units.ParticleType, "density"), zmin=(units.clrmin, "g/cm**2"), zmax=(units.clrmax, "g/cm**2"))
        snapdata.plot = p
    
        annotate(ds, snapdata.plot, snapdata.plottype, units, flags)
    snapdata.dim = 2
#}}}
# 2D Density plot 3 (For grid plotting of particles) {{{
def density3_plotting(ds, snapdata, units, flags):
     if 'sph_plotter' and 'custom_loader'in flags:

         x,y,z = snapdata.particle_positions
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
             snapdata.plot = yt.ProjectionPlot(ds, units.axis_of_projection,  (units.ParticleType, "density"), center=plot_center)
         except:
             print("\nSPH plot failed. Attempting particle plot...\n")
             if 'custom_loader' in flags:
                 #p = yt.ParticlePlot(ds, 'particle_position_x', 'particle_position_y', ("all", "density"), origin=origin, width=(2,2))
                 print('width: ', width)
                 #p = yt.ParticlePlot(ds, 'particle_position_x', 'particle_position_y', ("all", "density"), width=width, origin='upper-right-window')
             snapdata.plot = yt.ParticlePlot(ds, ("PartType1", "particle_position_x"), ("PartType1", "particle_position_y"))

         
         #Set colorbar limits
         if 'colorbarlims' in flags:
             snapdata.p.set_zlim(("gas", "density"), zmin=(units.clrmin, "g/cm**2"), zmax=(units.clrmax, "g/cm**2"))
     
     snapdata.dim = 2
#}}}
# 2D Mass histogram (tidy) {{{
def mass_grid(ds, snapdata, units, flags):
    # Create Plot 
    if 'sph_plotter' and 'custom_loader'in flags:

        x,y,z = snapdata.particle_positions
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
        annotate(ds, input_dir, snapdata.plottype, units, flags)
        plt.show()


    else:
        print('Plotting failed. Make sure that flags sph_plotter and custom_loader are enabled')  
        
    snapdata.dim = 2
#}}}
# 2D temperature plot (Should be working after refactorization){{{
def temperature(ds, snapdata, units, flags):
    snapdata.plot = yt.ProjectionPlot(ds, units.axis_of_projection,  ("gas", "temperature", flags), center=snapdata.plot_center)
    
    #Set colorbar limits
    if 'colorbarlims' in flags:
        snapdata.plot.set_zlim(("gas", "temperature"), zmin=(units.clrmin, "K"), zmax=(units.clrmax, "K"))
 
    annotate(ds, snapdata.plot, snapdata.plottype, units, flags)
    snapdata.dim = 2
#}}}
#2D Weighted temperature plot (Should be working after refactorization) {{{
def weighted_temperature(ds, snapdata, units, flags):
    snapdata.plot = yt.ProjectionPlot(  ds, units.axis_of_projection,  (units.ParticleType, "temperature"), 
                            center=snapdata.plot_center, weight_field=("gas", "density"))
    #p.set_cmap('inferno')
    snapdata.plot.zoom(units.zoom)

    #Set colorbar limits
    if 'colorbarlims' in flags:
        p.set_zlim((units.ParticleType, "temperature"), zmin=(units.clrmin, "K"), zmax=(units.clrmax, "K"))
    
    annotate(ds, snapdata.plot, snapdata.plottype, units, flags)
    snapdata.dim = 2
#}}}
# Volumetric density plot (definitely not working for now) {{{
def volumentric_density(ds, snapdata, units, flags, snapno):
        #p = yt.ProjectionPlot(ds, units.axis_of_projection,  (units.ParticleType, "density"), center=plot_center)
        sc = Scene()
        source = create_volume_source(ds, "density")
        sc.add_source(source)
        #sc = yt.create_scene(ds, lens_type='perspective')
        cam = sc.add_camera()
        im = sc.render()
        #source = sc[0]

        # Set the bounds of the transfer function
        source.tfh.set_bounds((3e-31, 5e-27))

        # set that the transfer function should be evaluated in log space
        source.tfh.set_log(True)

        # Make underdense regions appear opaque
        source.tfh.grey_opacity = True

        # Plot the transfer function, along with the CDF of the density field to
        # see how the transfer function corresponds to structure in the CDF
        source.tfh.plot("perspective_plot_"+snapno+".png", profile_field=("gas", "density"))

        # save the image, flooring especially bright pixels for better contrast
        sc.save("rendering.png", sigma_clip=6.0)

        #Set colorbar limits
        if 'colorbarlims' in flags:
            snapdata.plot.set_zlim((units.ParticleType, "density"), zmin=(units.clrmin, "g/cm**2"), zmax=(units.clrmax, "g/cm**2"))
        
        annotate(ds, snapdata.plot, snapdata.plottype, units, flags)
        snapdata.dim = 2
#}}}
# Deposition density plot (Not fully working) {{{
def deposition_density(ds, snapdata, units, flags, snapno):
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
        
            annotate(ds, p, plottype, units, flags)
        dim = 2
#}}}
# 2D smoothing lengths plot (Theoretically should be working) {{{
def smoothing_lengths(ds, snapdata, units, flags):
    snapdata.p = yt.ProjectionPlot(ds, units.axis_of_projection,  ("gas", "smoothing_length"), center=snapdata.plot_center)
    snapdata.p.set_unit(("gas", "smoothing_length"), "Mpc**2" )
    
    #Set colorbar limits
    #if 'colorbarlims' in flags:
    #    p.set_zlim(("gas", "smoothing_length"), zmin=(units.clrmin, "K"), zmax=(units.clrmax, "K"))

    annotate(ds, snapdata.plot, snapdata.plottype, units, flags)
    snapdata.dim = 2
#}}}
# 1D Density profile (Stopped working after refactorization) {{{
def density_profile(ds, snapdata, units, flags):
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
# 1D Shock velocity profile (Stopped working after refactorization) {{{
def shock_velocity(ds, snapdata, units, flags):
    ad=ds.r[:,1,1] #Only look at a slice in y=z=1
    # Set the time units {{{
    code_time = float(ds.current_time) 
    time_yrs=code_time * 0.978 / HubbleParam * unyt.Gyr
    snapdata.time_yrs=time_yrs.to_value(units.time_units)

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
# 1D Smoothing length histogram (Stopped working after refactorization) {{{
def smoothing_length_hist(ds, snapdata, units, flags):
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

# Save the plot {{{
def save_plot(snapdata, units, flags, snapno, out_dir):
    print(snapdata.dim)
    if (('plotting' in flags) and ('sph_plotter' not in flags)):
        if snapdata.dim == 2:
            snapdata.plot.save(out_dir+'2Dplot'+snapno+'.png') 
            print('Saved snapshot #' + snapno+ ' to: ' + out_dir+'2Dplot'+snapno+'.png')
        elif snapdata.dim == 1:
            print('plt')
            plt.savefig(out_dir+'1Dplot'+snapno+'.png')
            plt.clf()
            print('Saved #' + snapno+ ' to: ' +out_dir+'1Dplot'+snapno+'.png')
        else:
            print("Dimensionality not given")
    elif 'sph_plotter' in flags:
        print('plt')
        plt.savefig(out_dir+'2Dplot'+snapno+'.png')
        plt.clf()
        print(out_dir+'2Dplot'+snapno+'.png')
    elif 'plotting' not in flags:
        print("{:.2g}".format(snapdata.time_yrs)," " + units.time_units) #}}}
#}}}

# How much time passed since the last snapshot? {{{
def print_time_since_last_snapshot(time_since_snap, max_time):
    if time_since_snap < 300 :
        print(f'\rIt has been {time_since_snap} sec since the last snapshot', end='')
    elif time_since_snap < 7200 :
        time_since_snap_mins = time_since_snap // 60
        print(f'\rIt has been more than {time_since_snap_mins} mins since the last snapshot', end='')
    elif time_since_snap < max_time : 
        time_since_snap_hrs = time_since_snap // 3600
        print(f'\rIt has been more than {time_since_snap_hrs} hrs since the last snapshot', end='')
    else:
        sys.exit(f'Maximum time has been reached.\n max_time = {max_time} sec')
    time_since_snap+=5
    time.sleep(5)
#}}}
