**This documentation is very outdated!**

# GIZMO Analysis Library

This library is designed to analyze output from GIZMO, a flexible, multi-method magneto-hydrodynamics + gravity code for astrophysical simulations. It contains scripts for creating plots, upscaling data, converting simulations to movies, reading HDF5 files, analyzing CPU usage, and more.

## General Directory

### flags.py

This script contains a set of boolean flags used to control the behavior of various parts of the library. To modify these flags, open `flags.py` in a text editor and change the values as needed. Here is a list of the flags and their purposes:

- `SinglePlotMode`: Controls whether the library operates in single plot mode.
- `plotting`: Enables or disables plotting.
- `custom_loader`: Enables or disables a custom loader.
- `sph_plotter`: Enables or disables an SPH (Smoothed Particle Hydrodynamics) plotter.
- `colorbarlims`: Enables or disables color bar limits for 2D plots.
- `custom_center`: Enables or disables a custom center for 2D plots.
- `wraparound`: Enables or disables wraparound for 1D plots.
- `double_plot`: Enables or disables double plotting.
- `InitialPlotting`: Enables or disables initial plotting.
- `Time_Dependent`: Enables or disables time-dependent data analysis.
- `RestrictedLoad`: Enables or disables restricted load in the loader/upscaler.
- `debugging`: Enables or disables debugging.

This file also includes the `get_flags_array()` function, which retrieves the names of all flags currently set to `True`.

### utils.py

The `utils.py` file contains utility functions for various purposes:

- `annotate_plot`: This function annotates the yt plot with relevant information. Depending on the type of the plot (i.e., density, temperature, smoothing length, density profile, shock velocity, smoothing length histogram), it adds the appropriate title and labels to the plot.

- `wrap_second_half`: This function takes an array and wraps the second half of it around the maximum value in the array. This is used for data manipulation purposes.

- `cutoff_arrays`: This function cuts off the arrays based on a certain condition. It returns arrays with values that are greater than or equal to the cutoff value.

- `sort_arrays`: This function sorts arrays based on the values in the first array. It returns sorted arrays.

- `combine_snapshots`: This function combines multiple snapshots from different folders into a single image. The combined image contains one image on the left and another on the right.

- `center_and_find_box`: This function centers the distribution and calculates the size of the box containing the particles. It returns a DataFrame with centered coordinates and the size of the box.

- `custom_load_all_data`: This function loads all the data from an HDF5 file. It returns a dictionary containing the data and plot parameters.

- `custom_load_noyt`: This function loads only the necessary data for plotting from an HDF5 file, without using yt. It returns arrays of x, y, z coordinates, densities, smoothing lengths, and plot parameters.

- `custom_load`: This function loads only the necessary data for plotting from an HDF5 file, using yt. It returns a yt dataset and plot parameters.

- `increase_resolution_with_fft`: This function increases the resolution of the data using Fourier transformations. It returns a dictionary containing data with increased resolution.

- `increase_resolution_with_rbf`: This function increases the resolution of the data using Radial Basis Function interpolation. It returns a dictionary containing data with increased resolution.

Each function has a specific role in the library and is used in various parts of the GIZMO-analysis scripts. For a detailed understanding of each function, please refer to the comments and code in the `utils.py` file.

## Technical Analysis Directory

### cpu_usage_analysis.py

This script contains two functions for analyzing CPU usage during a simulation:

- `parse_cpu_file(file_path)`: This function reads a CPU usage file at the specified `file_path`. The function parses lines containing the simulation step, time, number of CPUs, task name, task time, and CPU usage percentage. It returns a pandas DataFrame with these data.

- `plot_cpu_usage(file_paths, labels, tasks, plot_by='time')`: This function generates a plot of CPU usage over time or simulation steps. The function takes a list of file paths (`file_paths`), a list of corresponding labels (`labels`), a list of tasks to include in the plot (`tasks`), and a string specifying whether to plot CPU usage over time or simulation steps (`plot_by`). The function uses `parse_cpu_file` to parse the CPU usage files and generates a line plot of CPU usage for each task.

To modify these functions, you may want to change the parsing logic in `parse_cpu_file` to match your specific CPU usage file format, or adjust the plotting parameters in `plot_cpu_usage` to suit your visualization needs.

## Snapshots to Movie Directory

### sph_plotter.py

This script contains four functions for creating 2D and 3D plots from Smoothed Particle Hydrodynamics (SPH) data. These functions are used by the `snapshots_to_plots.py` script to generate plots from simulation snapshots:

- `sph_density_projection_optimized(x, y, z, density, smoothing_lengths, resolution=100)`: This function creates a density projection plot of SPH data, optimized for large datasets. It uses adaptive kernel density estimation. The function takes arrays of x, y, z coordinates, density values, smoothing lengths, and an optional resolution parameter for the plot (default is 100).

- `sph_density_projection(x, y, z, density, smoothing_lengths, resolution=100)`: This function creates a density projection plot of SPH data. It uses a simpler kernel density estimation method that may be less efficient for large datasets.

- `sph_plotter2D(x,y,z,density,smoothing_lengths)`: This function creates a 2D density plot of SPH data using kernel density estimation in 2D.

- `sph_plotter3D(x,y,z,density,smoothing_lengths)`: This function creates a 3D density plot of SPH data using kernel density estimation in 3D.

Each of these functions can be modified to suit your specific data analysis and visualization needs. For instance, you might want to adjust the kernel density estimation methods, change the plot parameters, or add additional plot features.

### plots_to_movie.py

This script creates a movie from a series of plot images. The movie creation process depends on the specified output file extension:

- If the output file extension is 'gif', the script uses the `imageio` library to create a gif movie. It reads each plot file into an image and appends the images to a list. The list of images is then saved as a gif file in the output directory.

- If the output file extension is 'avi', the script uses the `cv2` (OpenCV) library to create an avi movie. It reads the first image file to get the frame size for the video, then writes each plot file into a frame of the video. The video is saved in the output directory.

- If the output file extension is neither 'gif' nor 'avi', the script prints an error message.

You can modify this script to add support for additional movie file formats, adjust the movie creation parameters, or change the way plot files are read and processed.

### snapshots_to_plots.py

This script creates plots from GIZMO simulation snapshots using the functions from the `sph_plotter.py` script. The following are the key variables and components that can be modified:

- `PlotType`: The type of plot to be created. It can be 2D or 3D.
- `InputDir`: The directory where the simulation snapshots are located.
- `OutputDir`: The directory where the generated plots will be saved.
- `TimeUnit`: The unit of time for the simulation.
- `SpatialUnit`: The unit of spatial dimensions for the simulation.
- `ProjectionAxis`: The axis along which to project the data for 2D plots.
- `ColorBarLims`: The limits for the color bar in the plots.
- `IntermediateFolders`: The intermediate folders to be created for the double plot mode.

After setting these variables, the script reads the snapshot files, generates plots for each snapshot, and saves the plots in the output directory. If double plot mode is enabled, the script generates two plots for each snapshot and saves them in separate folders.

## HDF5 Analysis Directory

### hdf5converter.py

This script contains a function for generating initial conditions (ICs) for GIZMO simulations from a pandas DataFrame. The following are the key variables and components that can be modified:

- `filename`: The name of the HDF5 file to be created.
- `df`: The DataFrame containing the initial conditions.
- `Boxsize`: The size of the simulation box.
- `NumFilesPerSnapshot`: The number of files per snapshot.
- `Time`: The time of the simulation.
- `Redshift`: The redshift of the simulation.
- `DoublePrecision`: A boolean flag indicating whether to use double precision.

The function `generate_GIZMO_IC()` creates an HDF5 file with the specified filename, adds the ICs from the DataFrame to the file, and sets various header attributes related to the simulation.

### hdf5_reader_header.py

This script reads the header data from a specified HDF5 file. You can modify the `filename` variable to change the HDF5 file that is read. The script prints information about the number of particles of each type in the simulation.

### fields_list.py

This script reads an HDF5 file and prints a list of all fields in the file. You can modify the `filename` variable to change the HDF5 file that is read.

## Upscalers Directory

### sk_upscaler.py

This script contains a variety of functions for creating and upscaling density fields from particle data. The following are the key variables and components that can be modified:

- `particle_positions`: The positions of the particles.
- `smoothing_lengths`: The smoothing lengths of the particles.
- `densities`: The densities of the particles.
- `box_size`: The size of the simulation box.
- `resolution`: The resolution of the grid.
- `field_values`: The field values of the particles.
- `grid_tuple`: The tuple representing the grid.
- `upscale_factor`: The upscale factor.
- `n_particles`: The number of particles to generate.

The main function of the script, `sk_upscaler_main()`, uses these inputs to upscale the density field, generate new particle positions, and upscale other data fields. 

### fftupscaler.py

This script contains functions for upscaling non-gridded data to gridded data using Fast Fourier Transforms (FFTs). The following are the key variables and components that can be modified:

- `field`: The field to be upscaled.
- `upscaled_field`: The upscaled field.
- `upscaled_tuple`: The tuple representing the upscaled grid.
- `original_data_dict`: The original data dictionary.
- `upscaled_data_dict`: The upscaled data dictionary.

The main function of the script, `upscale_with_fft_main()`, uses these inputs to upscale the data using FFTs.

## Customizing the Library

Most of the scripts in this library are designed to be run directly, but they can also be imported as modules and used in other scripts. To customize the library, you can edit the scripts directly or create new scripts that import and use the functions from the library. Remember to update the variables and flags as needed for your specific use case.

This library is designed to work with output from the GIZMO cosmological simulation code. The functionality and behavior of the scripts may depend on the specific output format and contents of your GIZMO simulations. Be sure to check that your simulations are compatible with the library before running the scripts.
















