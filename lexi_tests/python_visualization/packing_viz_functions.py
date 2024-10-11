import numpy as np
import pandas as pd
import glob


def get_atom_df(directory, dt):
    
    atom_variable_names = ['id', 'mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'fx',
                            'fy', 'fz', 'radius', 'sxx', 'syy', 'szz', 'sxy', 'sxz', 'syz']

    # Initialize a dictionary to store DataFrames for each atom ID
    atom_data = {}
    timesteps = []


    # fpaths = sorted(glob.glob(directory + '/all*')) # olivia
    fpaths = sorted(glob.glob(directory + r'\d*.liggghts')) #lexi

    # Loop over all dump files in the simulation directory
    for fname in fpaths:
        # Open and read the file
        with open(fname, 'r') as file:
            lines = file.readlines()
            
            # Extract the timestep; on the second line
            timestep = int(lines[1].strip()) * dt
            timesteps.append(timestep)
            
            # Extract the number of atoms; on the fourth line
            num_atoms = int(lines[3].strip())
            
            # Extract the atom data starting from the ninth line
            for i in range(num_atoms):
                data_line = lines[9 + i].strip()
                vals = list(map(float, data_line.split()))
                
                # Atom ID is the first value in the data line; important not to hard code this because datalines are NOT in order
                atom_id = int(vals[0])
                
                # Calculate the force magnitude for the atom
                fx, fy, fz = vals[8], vals[9], vals[10]
                atom_force_magnitude = np.sqrt(fx**2 + fy**2 + fz**2)

                vx, vy, vz = vals[5], vals[6], vals[7]
                atom_velocity_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
                
                # Add force magnitude to the list of variables
                vals.append(atom_force_magnitude)
                vals.append(atom_velocity_magnitude)
                
                # If the atom ID doesn't have a DataFrame yet, create one
                if atom_id not in atom_data:
                    atom_data[atom_id] = {var: [] for var in atom_variable_names + ['fmag', 'vmag']}
                    atom_data[atom_id]['t'] = []
                
                # Append the values for this timestep
                for j, var in enumerate(atom_variable_names + ['fmag', 'vmag']):
                    atom_data[atom_id][var].append(vals[j])
                atom_data[atom_id]['t'].append(timestep)
                

    # Convert each atom's dictionary into a DataFrame and store in a new dictionary
    atom_dfs = {}
    for atom_id, data_dict in atom_data.items():
        atom_dfs[atom_id] = pd.DataFrame(data_dict).set_index('t')

    # Concatenate the atom DataFrames into a MultiIndex DataFrame
    final_df = pd.concat(atom_dfs, axis=1)
    return final_df