#!/bin/bash

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "ucs_viz"

# Get the directory of the current script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set the base path
base_path="/home/arlenlex/LIGGGHTS_SEAICE/lexi_tests/bateman/simulations/ensemble"
sim_folder="$base_path"

# Ask for the script name
read -p "Enter the LIGGGHTS script name (e.g., in.shear): " script_name

# Ask for number of processors
read -p "Enter the number of processors: " num_processors

output_dir="/mnt/c/Users/arlenlex/Documents/liggghts_data/bateman/simulations/ensemble"

# Change to the simulation folder
cd "$sim_folder" || { echo "Simulation folder not found at $sim_folder!"; exit 1; }

# Delete the directory 'post' if it exists
if [ -d "post" ]; then
    rm -rf post
    echo "Deleted existing 'post' directory."
fi

# Delete the files 'atoms.nc' and 'bonds.nc' in the output directory if they exist
if [ -f "$output_dir/all_atoms.nc" ]; then
    rm "$output_dir/all_atoms.nc"
    echo "Deleted existing 'all_atoms.nc' file."
fi

if [ -f "$output_dir/atoms_plate.nc" ]; then
    rm "$output_dir/atoms_plate.nc"
    echo "Deleted existing 'atoms_plate.nc' file."
fi


if [ -f "$output_dir/bonds.nc" ]; then
    rm "$output_dir/bonds.nc"
    echo "Deleted existing 'bonds.nc' file."
fi

# Run the LIGGGHTS script
mpirun --bind-to core -np $num_processors /usr/local/bin/liggghts -in "$script_name"

# After running the script, process the data in the directory 'post' and save two NetCDF files
python_script_path="$script_dir/dump2nc.py"
python3 "$python_script_path" "$sim_folder/post" "$output_dir"
