#!/usr/bin/env bash

# This script automates running and post-processing of simulations from 
# ~/LIGGGHTS_SEAICE/lexi_tests/ucs/simulations/random with subfolders:
# phi60, phi70, phi80, phi90.
#
# Requirements:
#   - run_and_postprocess.sh in the current directory (the "running" folder)
#   - dump2netcdf.py in the current directory
#   - netcdf2figs.py in the current directory
#   - The netcdf2figs.py script originally asks for inputs; we will feed it "Y" using 'yes'
#
# Steps performed for each simulation directory:
#   1. Run and post-process the simulation using run_and_postprocess.sh (non-interactively).
#   2. Convert the resulting .nc files to figures by piping 'Y' into netcdf2figs.py.
#
# Adjust variables as needed.

# Set up conda environment for the first script
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "ucs_viz"

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="/home/arlenlex/LIGGGHTS_SEAICE/lexi_tests/ucs/simulations/random"
OUTPUT_BASE="/mnt/c/Users/arlenlex/Documents/liggghts_data/ucs/simulations/random"

# Simulation subdirectories
SIM_DIRS=("phi60" "phi70" "phi80" "phi90")

# LIGGGHTS script name and number of processors
LIGGGHTS_SCRIPT="in.compress" 
NUM_PROCESSORS=4

# Column names (if any) - if not, leave empty
COLUMN_NAMES=""

# Default dt for netcdf2figs.py
DT="0.00005"

# Loop over each simulation directory
for sim_subdir in "${SIM_DIRS[@]}"; do
    echo "Processing simulation: $sim_subdir"

    # Run and postprocess with run_and_postprocess.sh
    # The run_and_postprocess.sh script is interactive, so we provide inputs via a here-string.
    # Input order (based on the original script):
    # 1. relative simulation folder (from base_path)
    # 2. LIGGGHTS script name
    # 3. number of processors
    # 4. column_names (press Enter if not provided)
    #
    # If you have updated run_and_postprocess.sh to take arguments directly,
    # you can just pass them as arguments. Otherwise, we feed them with a here-string:
    cat <<EOF | bash "$SCRIPT_DIR/run_and_postprocess.sh"
random/$sim_subdir
$LIGGGHTS_SCRIPT
$NUM_PROCESSORS
$COLUMN_NAMES
EOF

    # After run_and_postprocess.sh finishes, we have atoms.nc and bonds.nc in $OUTPUT_BASE/$sim_subdir
    # Now run netcdf2figs.py with "Y" answers for all prompts.
    # netcdf2figs.py asks for:
    #   1. Enter base path (from ...): we pass 'random/phi...' 
    #   2. Enter dt (default: 0.00005): we pass $DT or leave blank
    #   3. Enter output directory if different from base path: leave blank to use the same
    #   4. Time average data? (Y/N): Y
    #   5. Create coordination number GIF? (Y/N): Y
    #   6. Plot the initial coordination numbers? (Y/N): Y
    #   7. Compute final floes? (Y/N): Y
    #   8. Compute FSD? (Y/N): Y
    #   9. Plot stress strain curve? (Y/N): Y

    # If you want all figures, we just say Y to all steps. If you want different behavior,
    # adjust the sequence of inputs below.

    yes Y | python3 "$SCRIPT_DIR/netcdf2figs.py" <<EOF2
random/$sim_subdir
$DT


EOF2

    echo "Finished processing $sim_subdir"
done

echo "All simulations processed."
