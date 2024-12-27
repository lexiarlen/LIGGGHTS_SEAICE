#!/usr/bin/env bash
###############################################################################
# Bash script that prompts for domain boundaries and file info,
# then calls the permanent Python script fix_data.py.
###############################################################################

# Prompt user for domain boundaries
echo "Enter domain bounds (xlo xhi ylo yhi zlo zhi):"
read xlo xhi ylo yhi zlo zhi

# Prompt user for the repository containing the data file
base_path="/home/arlenlex/LIGGGHTS_SEAICE/lexi_tests/bateman/simulations"
read -p "Enter the path to the original *.data file (from $base_path): " rel_sim_folder
repo="$base_path/$rel_sim_folder"

# Now just call the permanent Python script with the collected args:
python fix_data.py "$xlo" "$xhi" "$ylo" "$yhi" "$zlo" "$zhi" "$repo"
