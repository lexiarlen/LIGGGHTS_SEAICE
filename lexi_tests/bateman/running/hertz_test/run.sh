#!/bin/bash

# Function to run LIGGGHTS simulation
run_liggghts() {
  local script_name="$1"
  local processors="$2"
  local sim_folder="$3"

  cd "$sim_folder" || exit 1
  mpirun --bind-to core -np "$processors" /usr/local/bin/liggghts -in "$script_name"
}

# Get the directory where the script was started
start_dir="$(pwd)"

# Workflow steps
base_path="/home/arlenlex/LIGGGHTS_SEAICE/lexi_tests/bateman/simulations/hertzian_contact"
output_path="/mnt/c/Users/arlenlex/Documents/liggghts_data/bateman/simulations/hertzian_contact"

output_dir=$output_path
mkdir -p "$output_dir"

# Step 0: Remove existing post directory & output files

if [ -d "$base_path/post" ]; then
    rm -rf $base_path/post
    echo "Deleted existing 'post' directory."
fi

if [ -f "$output_dir/all_atoms_final.nc" ]; then
    rm "$output_dir/all_atoms_final.nc"
    echo "Deleted existing 'all_atoms_final.nc' file."
fi

if [ -f "$output_dir/atoms_plate.nc" ]; then
    rm "$output_dir/atoms_plate.nc"
    echo "Deleted existing 'atoms_plate.nc' file."
fi

if [ -f "$output_dir/bonds_final.nc" ]; then
    rm "$output_dir/bonds_final.nc"
    echo "Deleted existing 'bonds_final.nc' file."
fi

if [ -f "$output_dir/stress_strain_data.nc" ]; then
    rm "$output_dir/stress_strain_data.nc"
    echo "Deleted existing 'stress_strain_data.nc' file."
fi

# Step 1: Run in.install_bonds
run_liggghts "in.bond" "1" "$base_path"

# Step 2: Run in.read_restart
run_liggghts "in.compress" "4" "$base_path"

# Step 3: Change back to the starting directory
cd "$start_dir" || exit 1

# Step 4: Run dump2nc.py
python3 dump2nc.py "$base_path/post" "$output_dir"

# Step 5: Run nc2figs.py
python3 nc2figs.py --output-dir "$output_dir" --dt 0.0000005

