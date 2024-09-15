# functions for reading log, bond and dump files.
# used in the notebook python_plots.ipynb

import pandas as pd
import glob
import numpy as np

def get_dt(log_file_path):
    # Open and read the log file
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
        dt = None
        for i, line in enumerate(lines):
            if (line.strip().startswith('Step    Atoms numbonds             Dt')): 
                # Extract Dt from the first step line after 'Memory used'
                parts = lines[i+1].split()
                dt = parts[-1]  # Assuming Dt is always the last element
                break
    if dt == None:
        raise ValueError('No step size found')
    else:
        return float(dt)   
    
def create_bond_df(directory):
    '''
    Function for parsing LIGGGHTS bond files. 
    Computes bond force magnitude (b_fmag) in addition to other variables.
    Organizes data so that df['bfx'] gives an array of values across all timesteps.
    '''
    bond_variable_names = ["x1", "y1", "z1", "x2", "y2", "z2",
                           "batom1", "batom2", "bstatus", "bfx", "bfy", "bfz",
                           "btx", "bty", "btz", "beqdist"] 

    # Initialize a dictionary to store lists for each variable, including b_fmag
    data_dict = {var: [] for var in bond_variable_names}
    data_dict['b_fmag'] = []
    timesteps = []

    fpaths = sorted(glob.glob(directory + r'\*.bond'))

    # Loop over all files in the directory
    for fname in fpaths:
            
        # Open and read the file
        with open(fname, 'r') as file:
            lines = file.readlines()
            
            # Extract the timestep; on the second line
            timestep = int(lines[1].strip())
            timesteps.append(timestep)

            # Check if the last line starts with 'ITEM:'
            if lines[-1].strip().startswith('ITEM:'):
                for var in bond_variable_names:
                    data_dict[var].append(np.nan)
                data_dict['b_fmag'].append(np.nan)
                continue

            # Extract the data line; data is on the last line
            data_line = lines[-1]
            vals = list(map(float, data_line.split()))
            
            # Store each data value under its corresponding variable name
            for i, var in enumerate(bond_variable_names):
                data_dict[var].append(vals[i])

            # Calculate the force magnitude (b_fmag) and store it
            bfx, bfy, bfz = vals[9], vals[10], vals[11]
            bond_force_magnitude = np.sqrt(bfx**2 + bfy**2 + bfz**2)
            data_dict['b_fmag'].append(bond_force_magnitude)
                    
    # Create a DataFrame where each column contains an array of values across timesteps
    df = pd.DataFrame({var: pd.Series(data_dict[var], index=timesteps) for var in data_dict})
    
    return df

def create_atom_df(directory):
    atom_variable_names = ['id', 'mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'fx',
                            'fy', 'fz', 'omegax', 'omegay', 'omegaz', 'radius']
    
    # Initialize a dictionary to store DataFrames for each atom ID
    atom_data = {}
    timesteps = []

    fpaths = sorted(glob.glob(directory + r'\d*.liggghts'))

    # Loop over all dump files in the simulation directory
    for fname in fpaths:
        # Open and read the file
        with open(fname, 'r') as file:
            lines = file.readlines()
            
            # Extract the timestep; on the second line
            timestep = int(lines[1].strip())
            timesteps.append(timestep)
            
            # Extract the number of atoms; on the fourth line
            num_atoms = int(lines[3].strip())
            
            # Extract the atom data starting from the ninth line
            for i in range(num_atoms):
                data_line = lines[9 + i].strip()
                vals = list(map(float, data_line.split()))
                
                # Atom ID is the first value in the data line
                atom_id = int(vals[0])
                
                # Calculate the force magnitude for the atom
                fx, fy, fz = vals[8], vals[9], vals[10]
                atom_force_magnitude = np.sqrt(fx**2 + fy**2 + fz**2)
                
                # Add force magnitude to the list of variables
                vals.append(atom_force_magnitude)
                
                # If the atom ID doesn't have a DataFrame yet, create one
                if atom_id not in atom_data:
                    atom_data[atom_id] = {var: [] for var in atom_variable_names + ['atom_fmag']}
                    atom_data[atom_id]['timestep'] = []
                
                # Append the values for this timestep
                for j, var in enumerate(atom_variable_names + ['atom_fmag']):
                    atom_data[atom_id][var].append(vals[j])
                atom_data[atom_id]['timestep'].append(timestep)

    # Convert each atom's dictionary into a DataFrame and store in a new dictionary
    atom_dfs = {}
    for atom_id, data_dict in atom_data.items():
        atom_dfs[atom_id] = pd.DataFrame(data_dict).set_index('timestep')
    
    # Concatenate the atom DataFrames into a MultiIndex DataFrame
    final_df = pd.concat(atom_dfs, axis=1)
    
    return final_df
