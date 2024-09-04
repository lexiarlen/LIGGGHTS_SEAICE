#!/bin/bash

# Define the line numbers and the parameters you want to change
LINE_NUMBER_PARAM=14 
LINE_NUMBER_MKDIR=107  
LINE_NUMBER_DUMP=109    
LINE_NUMBER_BOND=112   

PARAMETER_NAME="bond_damp_val"  # Replace with the correct parameter name

# Define the range of values you want to test
VALUES=(0.0 0.05 5.0 25.0 50.0)  # Modify this list with the values you want to test

# Loop over each value
for VALUE in "${VALUES[@]}"; do
    # Define a specific directory name based on the parameter value
    DIR_NAME="post_${PARAMETER_NAME}_${VALUE}"
    
    # Create a temporary copy of the input file
    cp in.bond_contact in.bond_contact.tmp
    
    # Update the parameter value line
    sed -i "${LINE_NUMBER_PARAM}s/.*/variable ${PARAMETER_NAME}     equal ${VALUE}/" in.bond_contact.tmp
    
    # Update the mkdir command to create the specific directory
    sed -i "${LINE_NUMBER_MKDIR}s/post/${DIR_NAME}/" in.bond_contact.tmp
    
    # Update the dump commands to output to the specific directory
    sed -i "${LINE_NUMBER_DUMP}s|post/dump|${DIR_NAME}/dump|" in.bond_contact.tmp
    sed -i "${LINE_NUMBER_BOND}s|post/bfc|${DIR_NAME}/bfc|" in.bond_contact.tmp

    # Run the LIGGGHTS simulation with the modified input file
    mpirun -np 1 /usr/local/bin/liggghts -in in.bond_contact.tmp
    
    # Optionally, save the output with a specific name based on the parameter value
    mv log.liggghts "${DIR_NAME}/log_${PARAMETER_NAME}_${VALUE}.liggghts"
    
    # Remove the temporary file
    rm in.bond_contact.tmp
done