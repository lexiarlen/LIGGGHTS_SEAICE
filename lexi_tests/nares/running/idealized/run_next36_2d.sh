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
experiment_name="testing_postprocessing2d_from_restart"
processors_install=1  
processors_load=3

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

if [ -f "$output_dir/bonds_final.npz" ]; then
    rm "$output_dir/bonds_final.npz"
    echo "Deleted existing 'bonds_final.npz' file."
fi

# Step 3: Run in.flow2d
echo "Running in.flow2d_constant_u"
sed  "s|variable post_dir .*|variable post_dir string "$post_dir"|" \
    "$base_path/in.flow2d_constant_u" > "$base_path/temp_$experiment_name.flow2d_constant_u"
run_liggghts "temp_$experiment_name.flow2d_constant_u" "$processors_load" "$base_path"
rm "$base_path/temp_$experiment_name.flow2d_constant_u"

# Step 5: Change back to the starting directory
cd "$start_dir" || exit 1

if [ ! -f "$base_path/$post_dir" ]; then
    # Step 6: Run dump2nc.py
    python3 dump2nc.py "$base_path/$post_dir" "$output_dir"

    # Step 7: remove post directory
    #rm -rf "$base_path/$post_dir"
    
    # Step 8: Run nc2figs.py
    python3 nc2figs.py --output-dir "$output_dir" --dt 0.08
else 
    echo "Skipping processing, simulation error."
fi

# Step 9: Delete the output files to save space
