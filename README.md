This is the combination of codes that I use in order to analyse output of GIZMO simulations, for plotting, etc.

Here is a description of what these codes do:

## snapshots_to_plots.py

This is the main script. It contains all the plotting. It has a lot of features. More about that in a separate section.

## plots_to_movie.py

This is the script that converts the plots made in the output folder of the snapshots_to_plots.py.

## autogif.sh

This bash script runs snapshots_to_plots and then plots_to_movie codes. You only have to run this script (after altering all the necessary info in the two python codes) if you want to get the movie as the output.

## fields_list.py

This small code I use for troubleshooting to see which fields I have in my hdf5 file.

## hdf5_reader_header.py

This script would read the header data of a specified hdf5 file. Useful to see how many particles of which type do I have in the simulation.
:
