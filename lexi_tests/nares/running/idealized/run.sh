#!/bin/bash

# Function to create unique experiment directories
create_output_dirs() {
  local base_dir="$1"
  local experiment_name="$2"

  local output_dir="${base_dir}/results/${experiment_name}"
  mkdir -p "$output_dir"
  echo "$output_dir"
}

# Function to run LIGGGHTS simulation
run_liggghts() {
  local script_name="$1"
  local processors="$2"
  local sim_folder="$3"

  cd "$sim_folder" || exit 1
  mpirun --bind-to core -np "$processors" --bind-to core /usr/local/bin/liggghts -in "$script_name"
}
# Get the directory where the script was started
start_dir="$(pwd)"

# Workflow steps
base_path="/home/arlenlex/LIGGGHTS_SEAICE/lexi_tests/nares/simulations/idealized" 
output_path="/mnt/c/Users/arlenlex/Documents/liggghts_data/nares/simulations/idealized"
experiment_name="15ms_max_wind_speed"
processors_install=1  
processors_load=2

# Step 1: Create output directory
output_dir=$(create_output_dirs "$output_path" "$experiment_name")
post_dir="post_${experiment_name}"

# Step 2: Remove existing post directory & output files

if [ -d "$base_path/$post_dir" ]; then
    rm -rf $base_path/$post_dir
    echo "Deleted existing 'post' directory."
fi

if [ -f "$output_dir/atoms.nc" ]; then
    rm "$output_dir/atoms.nc"
    echo "Deleted existing 'atoms.nc' file."
fi

if [ -d "$output_dir/bonds" ]; then
    rm -rf "$output_dir/bonds"
    echo "Deleted existing 'bonds' folder."
fi

if [ -d "$output_dir/vel_profs" ]; then
    rm -rf "$output_dir/vel_profs"
    echo "Deleted existing 'vel_profs' folder."
fi

# make new bonds folder
bond_dir="${output_dir}/bonds"
mkdir -p "$bond_dir"

# make new bonds folder
vel_dir="${output_dir}/vel_profs"
mkdir -p "$vel_dir"

# Step 3: Run in.bond
echo "Running in.add_bonds"
sed "s|write_restart .*|write_restart restarts/${experiment_name}.restart|" \
    "$base_path/in.add_bonds" > "$base_path/temp_${experiment_name}.add_bonds"
run_liggghts "temp_${experiment_name}.add_bonds" "$processors_install" "$base_path"
rm "$base_path/temp_${experiment_name}.add_bonds"

# Step 4: Run in.flow2d
echo "Running in.flow2d"
sed "s|read_restart .*|read_restart restarts/$experiment_name.restart|; s|variable post_dir .*|variable post_dir string "${post_dir}"|; \
    s|write_restart .*|write_restart restarts/${experiment_name}_increasing_u.restart|" \
    "$base_path/in.flow2d" > "$base_path/temp_$experiment_name.flow2d"
run_liggghts "temp_$experiment_name.flow2d" "$processors_load" "$base_path"
rm "$base_path/temp_$experiment_name.flow2d"

# Step 6: Change back to the starting directory
cd "$start_dir" || exit 1

if [ ! -f "$base_path/$post_dir" ]; then
    # Step 7: Run dump2nc.py
    python3 dump2nc.py "$base_path/$post_dir" "$output_dir"

    # Step 8: remove post directory
    #rm -rf "$base_path/$post_dir"
    
    # Step 9: Run nc2figs.py
    python3 nc2figs.py --output-dir "$output_dir" --dt 0.05
else 
    echo "Skipping processing, simulation error."
fi

# Step 10: Delete the output files to save space
