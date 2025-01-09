#!/bin/bash

# Function to create unique experiment directories
create_output_dirs() {
  local base_dir="$1"
  local experiment_name="$2"
  local variant="$3"

  local output_dir="${base_dir}/results/${experiment_name}/${variant}"
  mkdir -p "$output_dir"
  echo "$output_dir"
}

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
base_path="/home/arlenlex/LIGGGHTS_SEAICE/lexi_tests/bateman/simulations/ensemble"
experiment_name="compression"
processors=4  # Adjust as needed

for normal_strength in 5e5 1e6 5e6; do
  for shear_strength in 5e5 1e6 5e6; do
    variant="norm_${normal_strength}_shear_${shear_strength}"
    output_dir=$(create_output_dirs "$base_path" "$experiment_name" "$variant")
    
    # Step 1: Run in.install_bonds
    sed "s|variable normal_strength .*|variable normal_strength equal $normal_strength|; s|variable shear_strength .*|variable shear_strength equal $shear_strength|; s|write_restart .*|write_restart restarts/${variant}.restart|" \
        "$base_path/in.install_bonds" > "$base_path/temp.install_bonds"
    run_liggghts "temp.install_bonds" "$processors" "$base_path"
    rm "$base_path/temp.install_bonds"
    
    # Step 2: Run in.read_restart
    sed "s|read_restart .*|read_restart restarts/${variant}.restart|; s|write_restart .*|write_restart restarts/${variant}_final.restart|" \
        "$base_path/in.read_restart" > "$base_path/temp.read_restart"
    run_liggghts "temp.read_restart" "$processors" "$base_path"
    rm "$base_path/temp.read_restart"
    
    # Step 3: Change back to the starting directory
    cd "$start_dir" || exit 1
    
    # Step 4: Run dump2nc.py
    python3 dump2nc.py "$base_path/post" "$output_dir"
    
    # Step 5: Run nc2figs.py
    python3 nc2figs.py --output-dir "$output_dir" --dt 0.0000005
  done
done
