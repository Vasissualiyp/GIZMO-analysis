**This README is very outdated!**

# GIZMO Analysis Library

This is a library designed to analyse output from the GIZMO cosmological simulation code. It contains various scripts for plotting, creating movies, upscaling simulations, and analyzing CPU usage, among other features.

## Contents

The library contains the following scripts:

- `snapshots_to_plots.py`: Creates plots from GIZMO simulation snapshots. Variables such as the plot type, input and output directory paths, time and spatial units, the projection axis for 2D plots, and color map limits can be edited directly within the script.

- `plots_to_movie.py`: Creates a movie from a series of plot images. The input and output directories, as well as the output movie file name and extension, can be specified within the script.

- `autogif.sh`: A bash script that automates the process of creating a movie from simulation snapshots. It runs `snapshots_to_plots.py` and `plots_to_movie.py` in sequence.

- `sph_plotter.py`: Contains functions for creating 2D and 3D plots from Smoothed Particle Hydrodynamics (SPH) data.

- `hdf5converter.py`: Generates initial conditions (ICs) for GIZMO simulations from a DataFrame.

- `sk_upscaler.py`: Contains functions for creating and upscaling density fields from particle data.

- `cpu_usage_analysis.py`: Contains functions for analyzing CPU usage during a simulation.

- `flags.py`: Contains a variety of flags that control the behavior of different parts of the library.

## Usage

To run any of the Python scripts, use a command of the form `python script_name.py`. Make sure to edit any necessary variables within the script before running it.

To run the `autogif.sh` bash script, use the command `bash autogif.sh`.

## Dependencies

The following Python libraries are required to run the scripts in this library:

- numpy
- meshoid
- yt
- unyt
- PIL (Pillow)
- scipy
- matplotlib
- imageio
- cv2 (OpenCV)
- numba
- h5py
- pandas

## Note

This library is designed to work with output from the GIZMO cosmological simulation code. The functionality and behavior of the scripts may depend on the specific output format and contents of your GIZMO simulations.
