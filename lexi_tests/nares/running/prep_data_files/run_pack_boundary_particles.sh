#!/usr/bin/env bash
###############################################################################
# Bash script that prompts for domain boundaries and file info,
# then calls the permanent Python script fix_data.py.
###############################################################################

# Prompt user for domain boundaries
echo "Enter domain bounds in km (xlo xhi ylo yhi):"
read xlo xhi ylo yhi

# get diameter
read -p "Enter boundary particle diameter: " d

# get density
read -p "Enter boundary particle density: " density

# Prompt user for the repository containing the data file
out_path="/home/arlenlex/LIGGGHTS_SEAICE/lexi_tests/nares/simulations/create_packing_better/data"
read -p "Enter output path or press enter to use $out_path: " outpath

# Call Python script
# Check if repository is provided
if [[ -z "$outpath" ]]; then
    # Call Python script without repository
    python create_bdy_data.py "$xlo" "$xhi" "$ylo" "$yhi" "$d" "$density" "$out_path"
else
    # Call Python script with bond_skin_thickness
    python create_bdy_data.py "$xlo" "$xhi" "$ylo" "$yhi" "$d" "$density" "$out_path"
fi