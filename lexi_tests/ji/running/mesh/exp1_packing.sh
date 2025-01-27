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
base_path="/home/arlenlex/LIGGGHTS_SEAICE/lexi_tests/ji/simulations/mesh" 
output_path="/mnt/c/Users/arlenlex/Documents/liggghts_data/ji/simulations/mesh"
experiment_name="packing"
processors_install=1  
processors_load=4

for packing in ji_mesh1 ji_mesh2; do 
    echo "RUNNING PACKING: $packing"

    # Step 1: Create output directory
    output_dir=$(create_output_dirs "$output_path" "$experiment_name" "$packing")

    # Step 2: Remove existing post directory & output files

    if [ -d "$base_path/post_${experiment_name}_${packing}" ]; then
        rm -rf $base_path/post_${experiment_name}_${packing}
        echo "Deleted existing 'post' directory."
    fi

    if [ -f "$output_dir/atoms.nc" ]; then
        rm "$output_dir/atoms.nc"
        echo "Deleted existing 'atoms.nc' file."
    fi

    if [ -f "$output_dir/plate.nc" ]; then
        rm "$output_dir/plate.nc"
        echo "Deleted existing 'plate.nc' file."
    fi

    if [ -f "$output_dir/bonds_final.nc" ]; then
        rm "$output_dir/bonds_final.nc"
        echo "Deleted existing 'bonds_final.nc' file."
    fi
    
    if [ -f "$output_dir/stress_strain_data.nc" ]; then
        rm "$output_dir/stress_strain_data.nc"
        echo "Deleted existing 'stress_strain_data.nc' file."
    fi

    # Step 3: Run in.bond
    sed "s|read_data .*|read_data data/${packing}.data|; s|write_restart .*|write_restart restarts/${packing}.restart|" \
        "$base_path/in.bond" > "$base_path/temp.bond"
    run_liggghts "temp.bond" "$processors_install" "$base_path"
    rm "$base_path/temp.bond"
    
    # Step 4: Run in.compress #; s|write_restart .*|write_restart restarts/${packing}_final.restart|
    sed "s|read_restart .*|read_restart restarts/${packing}.restart|; s|variable post_dir .*|variable post_dir string "post_${experiment_name}_${packing}"|;" \
        "$base_path/in.compress" > "$base_path/temp_$packing.compress"
    run_liggghts "temp_$packing.compress" "$processors_load" "$base_path"
    rm "$base_path/temp_$packing.compress"
    
    # Step 5: Change back to the starting directory
    cd "$start_dir" || exit 1
    
    if [ ! -f "$base_path/post_${experiment_name}_${packing}" ]; then
        # Step 6: Run dump2nc.py
        python3 dump2nc.py "$base_path/post_${experiment_name}_${packing}" "$output_dir"

        # Step 7: remove post directory
        rm -rf "$base_path/post_${experiment_name}_${packing}"
        
        # Step 8: Run nc2figs.py
        python3 nc2figs.py --output-dir "$output_dir" --dt 0.000001
    else 
        echo "Skipping processing, simulation error."
    fi

    # Step 9: Delete the output files to save space
    if [ -f "$output_dir/atoms.nc" ]; then
        rm "$output_dir/atoms.nc"
        echo "Deleted existing 'atoms.nc' file."
    fi

    if [ -f "$output_dir/plate.nc" ]; then
        rm "$output_dir/plate.nc"
        echo "Deleted existing 'plate.nc' file."
    fi

    if [ -f "$output_dir/bonds_final.nc" ]; then
        rm "$output_dir/bonds_final.nc"
        echo "Deleted existing 'bonds_final.nc' file."
    fi
done