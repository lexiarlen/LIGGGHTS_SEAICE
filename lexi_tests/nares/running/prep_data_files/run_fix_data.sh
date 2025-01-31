#!/usr/bin/env bash
###############################################################################
# Bash script that prompts for domain boundaries and file info,
# then calls the permanent Python script fix_data.py.
###############################################################################

# Prompt user for domain boundaries
echo "Enter domain bounds (xlo xhi ylo yhi zlo zhi):"
read xlo xhi ylo yhi zlo zhi

# Prompt user for the repository containing the data file
base_path="/home/arlenlex/LIGGGHTS_SEAICE/lexi_tests/ji/simulations/create_sample/data"
read -p "Enter the name of the original *.data file (from $base_path): " rel_sim_folder
repo="$base_path/$rel_sim_folder"

read -p "Bond skin thickness? (press Enter to use default = 0.005*0.015): " bond_skin_thickness

# Check if bond_skin_thickness is provided
if [[ -z "$bond_skin_thickness" ]]; then
    # Call Python script without bond_skin_thickness
    python fix_data.py "$xlo" "$xhi" "$ylo" "$yhi" "$zlo" "$zhi" "$repo"
else
    # Call Python script with bond_skin_thickness
    python fix_data.py "$xlo" "$xhi" "$ylo" "$yhi" "$zlo" "$zhi" "$repo" "$bond_skin_thickness"
fi
