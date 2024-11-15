#!/bin/bash

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "run_and_postprocess_liggghts"

# Get the directory of the current script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set the base path
base_path="/home/arlenlex/LIGGGHTS_SEAICE/lexi_tests"

# Ask for the relative path to the simulation folder
read -p "Enter the path to the simulation folder (from $base_path): " rel_sim_folder

# Combine base path and relative path
sim_folder="$base_path/$rel_sim_folder"

# Ask for the script name
read -p "Enter the LIGGGHTS script name (e.g., in.shear): " script_name

# Ask for number of processors
read -p "Enter the number of processors: " num_processors

# Ask for the column_names (use default if none provided)
read -p "Enter the bond dump variable names (press Enter to use default): " column_names

output_dir="/mnt/c/Users/arlenlex/Documents/liggghts_data/$rel_sim_folder"

# Change to the simulation folder
cd "$sim_folder" || { echo "Simulation folder not found at $sim_folder!"; exit 1; }

# Delete the directory 'post' if it exists
if [ -d "post" ]; then
    rm -rf post
    echo "Deleted existing 'post' directory."
fi

# Run the LIGGGHTS script
mpirun --bind-to core -np $num_processors /usr/local/bin/liggghts -in "$script_name"

# # Make post writeable
# chmod -R +w "$sim_folder/post"

# After running the script, process the data in the directory 'post' and save two NetCDF files
python_script_path="$script_dir/dump2netcdf.py"
if [ -z "$column_names" ]; then
    python3 "$python_script_path" "$sim_folder/post" "$output_dir"
else
    python3 "$python_script_path" "$sim_folder/post" "$output_dir" --column_names "$column_names"
fi
