#!/bin/bash
# Run the script in the main git dir, not in the ./rockstar dir!

# Set the parths to all relevant files {{{
music_dir="../music"
python_dir="./rockstar"
backtrace_particles_path="$python_dir/backtrace_the_particles.py"
box_properties_file="$python_dir/boundbox_characteristics.txt"
music_dm_only_path="$music_dir/dm_only_ics.conf"
music_dm_withb_path="$music_dir/dm+b_ics.conf"
music_ics_script="./create_music_ics.sh" # This script needs to be ran from the MUSIC directory
#}}}

# Load python and setup the environment
module load python && source env/bin/activate || echo "Error in sourcing the environment"

# Run the script to generate the bounding box parameters
python $backtrace_particles_path

# Extract the line from the first file
ref_extent_line=$(grep '^ref_extent' "$box_properties_file")
echo "ref_extent line looks like: $ref_extent_line"
ref_center_line=$(grep '^ref_center' "$box_properties_file")
echo "ref_center line looks like: $ref_extent_line"

# Replace the line in the second file
sed -i "/^ref_extent/c\\$ref_extent_line" "$music_dm_withb_path"
sed -i "/^ref_center/c\\$ref_center_line" "$music_dm_withb_path"
