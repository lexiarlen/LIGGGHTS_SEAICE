#!/usr/bin/env bash
###############################################################################
# Bash script that prompts for domain boundaries and file info,
# then calls the permanent Python script fix_data.py.
###############################################################################

# Prompt user for domain boundaries
echo "Enter domain bounds in km (xlo xhi ylo yhi):"
read xlo xhi ylo yhi

# Prompt user for the repository containing the data file
base_path="/home/arlenlex/LIGGGHTS_SEAICE/lexi_tests/nares/simulations/create_packing/data"
read -p "Enter the name of the original *.data file (from $base_path): " rel_sim_folder
repo="$base_path/$rel_sim_folder"

read -p "Enter diameter: " d
read -p "Enter density: " density

# Check if bond_skin_thickness is provided
python fix_data_3d.py "$xlo" "$xhi" "$ylo" "$yhi" "$repo" "$d" "$density"
