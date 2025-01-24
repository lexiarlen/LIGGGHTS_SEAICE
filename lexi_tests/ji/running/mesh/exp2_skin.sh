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
base_path="/home/arlenlex/LIGGGHTS_SEAICE/lexi_tests/bateman/simulations/hertzian_contact" 
output_path="/mnt/c/Users/arlenlex/Documents/liggghts_data/bateman/simulations/hertzian_contact"
experiment_name="skin"
processors_install=1  
processors_load=5

for packing in poly20_1; do # already ran for mono1 so don't do again
    for bond_skin in 1.01 1.0001; do
        echo "RUNNING PACKING: $packing with bond skin: $bond_skin"

        # Step 1: Create output directory

        variant=$(printf "%.0e" "$(echo "$bond_skin" | awk '{print $1-1}')")  # Subtract 1 and format
        variant="${variant//-}$packing" # Remove negative signs in folder names
        output_dir=$(create_output_dirs "$output_path" "$experiment_name" "$variant")

        # Step 2: Remove existing post directory & output files
        if [ -d "$base_path/post_${experiment_name}_${variant}" ]; then
            rm -rf $base_path/post_${experiment_name}_${variant}
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


        # Step 3: Determine poly value: 0 for mono, 0.1 for poly10, 0.2 for poly20
        if [[ "$packing" == mono* ]]; then
        poly_value=0
        elif [[ "$packing" == poly10* ]]; then
        poly_value=0.1
        elif [[ "$packing" == poly20* ]]; then
        poly_value=0.2
        fi

        # Step 4: Run in.bond
        sed "s|variable bond_skin_multiplier .*|variable bond_skin_multiplier equal $bond_skin|; s|variable poly .*|variable poly equal $poly_value|; \
            s|read_data .*|read_data data/${packing}.data|; s|write_restart .*|write_restart restarts/${packing}_skin_${variant}.restart|" \
            "$base_path/in.bond" > "$base_path/temp.bond"
        run_liggghts "temp.bond" "$processors_install" "$base_path"
        rm "$base_path/temp.bond"
        
        # Step 5: Run in.compress
        sed "s|read_restart .*|read_restart restarts/${packing}_skin_${variant}.restart|; s|variable post_dir .*|variable post_dir string "post_${experiment_name}_${variant}"|" \
            "$base_path/in.compress" > "$base_path/temp.compress"
        run_liggghts "temp.compress" "$processors_load" "$base_path"
        rm "$base_path/temp.compress"
        
        # Step 6: Change back to the starting directory
        cd "$start_dir" || exit 1
        
        # Step 7: Run dump2nc.py
        python3 dump2nc.py "$base_path/post_${experiment_name}_${variant}" "$output_dir"

        # Step 8: remove post directory
        rm -rf "$base_path/post_${experiment_name}_${variant}"
        
        # Step 9: Run nc2figs.py
        python3 nc2figs.py --output-dir "$output_dir" --dt 0.000001

        # Step 10: Delete the output files to save space
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
    done
done