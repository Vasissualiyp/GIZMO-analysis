#!/bin/bash

# Set the path to the SCRIPT_StP and SCRIPT_PtM files
SCRIPT_StP="./snapshots_to_plots.py"
SCRIPT_PtM="./plots_to_movie.py"

# Set the transmit_folder variable
transmit_folder=true

# Transmitting the name of the output folder from the first script into the input of another {{{
if [ "$transmit_folder" = true ]; then
    # Get the first line that starts with "out_dir" in SCRIPT_StP
    out_dir_line=$(grep "^out_dir" "$SCRIPT_StP" | head -n 1 | sed 's/\//\\\//g')

    # Replace the first line that starts with "input_dir" in SCRIPT_PtM with the out_dir line from SCRIPT_StP
    sed -i "0,/^input_dir/s/^input_dir.*/${out_dir_line}/" "$SCRIPT_PtM"

    # Replace the first occurrence of "out_dir" in SCRIPT_PtM with "input_dir"
    sed -i "0,/out_dir/s//input_dir/" "$SCRIPT_PtM"
fi #}}}

#Running the script {{{
source ~/research/pyenv/python3/bin/activate
python SCRIPT_StP
python SCRIPT_PtM
#}}}
