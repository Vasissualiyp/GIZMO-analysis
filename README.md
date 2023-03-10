This is the combination of codes that I use in order to analyse output of GIZMO simulations, for plotting, etc.

Here is a description of what these codes do:

* snapshots\_to\_plots.py
This is the main script. It contains all the plotting. It has a lot of features. More about that in a separate section.

* plots\_to\_movie.py
This is the scipt that converts the plots made in the output folder of the snapshots\_to\_plots.py 

* autogif.sh
This bash script runs snapshots\_to\_plots and then plots\_to\_movie codes. You only have to run this script (after altering all the necessary info in these two python codes) if you want to get the movie as the output.

* fields\_list.py
This small code I use for troubleshooting to see which fields I have in my hdf5 file.

* hdf5\_reader\_header.py
This script would read the header data of a specified hdf5 file. Useful to see how many particles of which type do I have in the simulation.

* snapshots\_to\_plots.py
