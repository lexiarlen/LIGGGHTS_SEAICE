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
  mpirun --bind-to core -np "$processors" --bind-to core /usr/local/bin/liggghts -in "$script_name"
}
# Get the directory where the script was started
start_dir="$(pwd)"

# Workflow steps
base_path="/home/arlenlex/LIGGGHTS_SEAICE/lexi_tests/bateman/simulations/ensemble" 
output_path="/mnt/c/Users/arlenlex/Documents/liggghts_data/bateman/simulations/ensemble"
experiment_name="packings"
processors_install=1  
processors_load=4

for packing in dense1 dense2 dense3; do
    echo "RUNNING PACKING: $packing"

    # Step 1: Create output directory
    output_dir=$(create_output_dirs "$output_path" "$experiment_name" "$packing")
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

    # Step 2: Run in.install_bonds
    sed "s|read_data .*|read_data data/${packing}.data|; s|write_restart .*|write_restart restarts/${packing}.restart|" \
        "$base_path/in.install_bonds" > "$base_path/temp.install_bonds"
    run_liggghts "temp.install_bonds" "$processors_install" "$base_path"
    rm "$base_path/temp.install_bonds"
    
    # Step 3: Run in.read_restart #; s|write_restart .*|write_restart restarts/${packing}_final.restart|
    sed "s|read_restart .*|read_restart restarts/${packing}.restart|" \
        "$base_path/in.read_restart" > "$base_path/temp_$packing.read_restart"
    run_liggghts "temp_$packing.read_restart" "$processors_load" "$base_path"
    rm "$base_path/temp_$packing.read_restart"
    
    # Step 4: Change back to the starting directory
    cd "$start_dir" || exit 1
    
    # Step 5: Run dump2nc.py
    python3 dump2nc.py "$base_path/post" "$output_dir"
    
    # Step 6: Run nc2figs.py
    python3 nc2figs.py --output-dir "$output_dir" --dt 0.0000005
done