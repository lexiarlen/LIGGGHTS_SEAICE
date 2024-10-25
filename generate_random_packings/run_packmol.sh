#!/bin/bash

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "packmol_processing"

echo "Enter number of particles:"
read NUM_PARTICLES

echo "Enter domain bounds (xlo xhi ylo yhi):"
read XLO XHI YLO YHI

echo "Enter bond skin:"
read BOND_SKIN

echo "Enter output directory name:"
read OUTPUT_DIR

echo "Enter diameter:"
read DIAMETER

echo "Enter density:"
read DENSITY

# Compute radius & tolerance
RADIUS=$(awk "BEGIN {print $DIAMETER / 2}")
TOLERANCE=$(awk "BEGIN {print $DIAMETER * $BOND_SKIN}")

# Adjust domain bounds to account for particle radius and tolerance
ADJ_XLO=$(awk "BEGIN {print $XLO + $TOLERANCE / 2}")
ADJ_XHI=$(awk "BEGIN {print $XHI - $TOLERANCE / 2}")
ADJ_YLO=$(awk "BEGIN {print $YLO + $TOLERANCE / 2}")
ADJ_YHI=$(awk "BEGIN {print $YHI - $TOLERANCE / 2}")

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Write particle.xyz
cat > "$OUTPUT_DIR/particle.xyz" << EOF
1
Spherical particle
C 0.0 0.0 0.0
EOF

# Write particle.inp
cat > "$OUTPUT_DIR/particle.inp" << EOF
tolerance $TOLERANCE  # tolerance to avoid overlap

filetype xyz

output packed_particles.xyz  # output file

# Define the packing box with adjusted boundaries
structure particle.xyz  # single particle
  number $NUM_PARTICLES  # number of particles
  inside box $ADJ_XLO $ADJ_YLO 0.0 $ADJ_XHI $ADJ_YHI 0.0  # adjusted box dimensions
  radius $RADIUS  # set particle radius
end structure
EOF

# Run Packmol using Julia
julia -e "using Packmol; run_packmol(raw\"$(pwd)/$OUTPUT_DIR/particle.inp\")"

# Run the Python script to process the output
python process_packed_particles.py $XLO $XHI $YLO $YHI 0.0 0.0 $DIAMETER $DENSITY "$OUTPUT_DIR"

# Deactivate conda environment
conda deactivate
