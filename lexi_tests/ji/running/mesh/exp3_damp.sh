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
experiment_name="damp"
processors_install=1  
processors_load=4
packing="ji_mesh2"


for bond_damping in 5; do
    echo "RUNNING PACKING: $packing with bond damping: $bond_damping"

    # Step 1: Create output directory
    variant="damp_$bond_damping"
    output_dir=$(create_output_dirs "$output_path" "$experiment_name" "$variant")

    # Step 2: Remove existing post directory & output files
    if [ -d "$base_path/post_${experiment_name}_${variant}" ]; then
        rm -rf $base_path/post_${experiment_name}_${variant}
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
    sed "s|read_data .*|read_data data/${packing}.data|; s|write_restart .*|write_restart restarts/${variant}.restart|" \
        "$base_path/in.bond" > "$base_path/temp.bond"
    run_liggghts "temp.bond" "$processors_install" "$base_path"
    rm "$base_path/temp.bond"
    
    # Step 4: Run in.compress #; s|write_restart .*|write_restart restarts/${packing}_final.restart|
    sed "s|read_restart .*|read_restart restarts/${variant}.restart|; s|variable bond_damp_val .*|variable bond_damp_val equal $bond_damping|; \
        s|variable post_dir .*|variable post_dir string "post_${experiment_name}_${variant}"|;" \
        "$base_path/in.compress" > "$base_path/temp_$variant.compress"
    run_liggghts "temp_$variant.compress" "$processors_load" "$base_path"
    rm "$base_path/temp_$variant.compress"
    
    # Step 5: Change back to the starting directory
    cd "$start_dir" || exit 1
    
    if [ ! -f "$base_path/post_${experiment_name}_${variant}" ]; then
        # Step 6: Run dump2nc.py
        python3 dump2nc.py "$base_path/post_${experiment_name}_${variant}" "$output_dir"

        # Step 7: remove post directory
        #rm -rf "$base_path/post_${experiment_name}_${variant}"
        
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