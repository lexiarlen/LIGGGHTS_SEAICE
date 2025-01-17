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
output_path="/mnt/c/Users/arlenlex/Documents/liggghts_data/bateman/simulations/ensemble"
experiment_name="compression"
processors=4  # Adjust as needed

for normal_strength in 3e5 7e5 1.1e6 1.5e6; do 
  for shear_strength in 3e5 7e5 1.1e6 1.5e6; do

    echo "RUNNING COMPRESSION SIMULATION: w/ normal_strength $normal_strength Pa and shear strength $shear_strength Pa"

    # Step 0: Delete existing post directory & output files
    
    variant="norm_${normal_strength}_shear_${shear_strength}"
    output_dir=$(create_output_dirs "$output_path" "$experiment_name" "$variant")
    
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
    sed "s|variable normal_strength .*|variable normal_strength equal $normal_strength|; s|variable shear_strength .*|variable shear_strength equal $shear_strength|; s|write_restart .*|write_restart restarts/${variant}.restart|" \
        "$base_path/in.install_bonds" > "$base_path/temp.install_bonds"
    run_liggghts "temp.install_bonds" "1" "$base_path"
    rm "$base_path/temp.install_bonds"
    
    # Step 2: Run in.read_restart
    sed "s|read_restart .*|read_restart restarts/${variant}.restart|; s|variable normal_strength .*|variable normal_strength equal $normal_strength|; s|variable shear_strength .*|variable shear_strength equal $shear_strength|" \
        "$base_path/in.read_restart" > "$base_path/temp.read_restart"
    run_liggghts "temp.read_restart" "4" "$base_path"
    rm "$base_path/temp.read_restart"
    
    # Step 3: Change back to the starting directory
    cd "$start_dir" || exit 1
    
    # Step 4: Run dump2nc.py
    python3 dump2nc.py "$base_path/post" "$output_dir"
    
    # Step 5: Run nc2figs.py
    python3 nc2figs.py --output-dir "$output_dir" --dt 0.0000005
  done
done
