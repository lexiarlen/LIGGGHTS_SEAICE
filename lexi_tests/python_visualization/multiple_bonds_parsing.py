import glob
import pandas as pd
import numpy as np
import networkx as nx

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

def compute_coordination_numbers(bond_directory, log_path, dt = None):
    if dt == None:
        dt = get_dt(log_path)

    bond_files = sorted(glob.glob(bond_directory + r'\*.bond'))
    coordination_numbers = {}
    num_bonds = {}

    for bond_file in bond_files:
        nbonds = 0
        with open(bond_file, 'r') as file:
            lines = file.readlines()
            
            timestep = int(lines[1].strip()) # Extract the timestep
            num_entries = int(lines[3].strip())  # Extract the number of bonds
            
            # Initialize a dictionary to count the coordination numbers at each timestep
            coord_num_at_timestep = {}
            
            # Read bond entries starting from line 9
            for i in range(9, 9 + num_entries):
                bond_data = list(map(float, lines[i].strip().split()))
                
                batom1 = int(bond_data[6])  # Atom 1 ID
                batom2 = int(bond_data[7])  # Atom 2 ID
                bbondbroken = bond_data[8]  # Bond status (0 = bonded, 1 = broken)
                
                # Only count the bond if it's not broken (bbondbroken = 0)
                if bbondbroken == 0:
                    # Update the coordination & nbonds counts for batom1 and batom2
                    nbonds += 1
                    if batom1 not in coord_num_at_timestep:
                        coord_num_at_timestep[batom1] = 0
                    if batom2 not in coord_num_at_timestep:
                        coord_num_at_timestep[batom2] = 0

                    coord_num_at_timestep[batom1] += 1
                    coord_num_at_timestep[batom2] += 1
            
            # Store coordination numbers at this timestep
            coordination_numbers[timestep] = coord_num_at_timestep
            num_bonds[timestep] = nbonds
    
    nbonds_df = pd.DataFrame(list(num_bonds.items()), columns=['t', 'nbonds']).set_index('t')

    time = nbonds_df.index * dt
    nbonds_df.index = time
    
    return coordination_numbers, nbonds_df

def create_atom_df(directory, log_path, dt = None):
    
    if dt == None:
        dt = get_dt(log_path)

    # atom_variable_names = ['id', 'mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'fx',
    #                         'fy', 'fz', 'omegax', 'omegay', 'omegaz', 'radius']

    atom_variable_names = ['id', 'mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'fx',
                            'fy', 'fz', 'radius', 'sxx', 'syy', 'szz', 'sxy', 'sxz', 'syz']
    
    # Initialize a dictionary to store DataFrames for each atom ID
    atom_data = {}
    timesteps = []

    # Compute coordination numbers from bond files
    coordination_numbers, num_bonds_df = compute_coordination_numbers(directory, log_path, dt)

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
                
                # Atom ID is the first value in the data line; important not to hard code this because datalines are NOT in order
                atom_id = int(vals[0])
                
                # Calculate the force magnitude for the atom
                fx, fy, fz = vals[8], vals[9], vals[10]
                atom_force_magnitude = np.sqrt(fx**2 + fy**2 + fz**2)
                
                # Add force magnitude to the list of variables
                vals.append(atom_force_magnitude)
                
                # If the atom ID doesn't have a DataFrame yet, create one
                if atom_id not in atom_data:
                    atom_data[atom_id] = {var: [] for var in atom_variable_names + ['atom_fmag', 'coord_num']}
                    atom_data[atom_id]['t'] = []
                
                # Append the values for this timestep
                for j, var in enumerate(atom_variable_names + ['atom_fmag']):
                    atom_data[atom_id][var].append(vals[j])
                atom_data[atom_id]['t'].append(timestep)

                # If bond exists at timestep, get coord num:
                coord_num = 0
                if np.isin(timestep, list(coordination_numbers.keys())):
                    coord_num = coordination_numbers[timestep].get(atom_id, 0)  # Default to 0 if no bonds found
                atom_data[atom_id]['coord_num'].append(coord_num)


    # Convert each atom's dictionary into a DataFrame and store in a new dictionary
    atom_dfs = {}
    for atom_id, data_dict in atom_data.items():
        atom_dfs[atom_id] = pd.DataFrame(data_dict).set_index('t')
    
    # Concatenate the atom DataFrames into a MultiIndex DataFrame
    final_df = pd.concat(atom_dfs, axis=1)
    
    time = final_df.index * dt
    final_df.index = time

    return final_df, num_bonds_df

def compute_fragments(bond_directory):
    bond_files = sorted(glob.glob(bond_directory + r'\*.bond'))
    fragments_per_timestep = {}

    Graphs = []
    G = nx.Graph()  # Main graph that keeps track of all atoms and bonds
    
    # Set to track all atoms that have appeared in the simulation
    all_atoms = set()

    for bond_file in bond_files:
        with open(bond_file, 'r') as file:
            g = nx.Graph()
            # Create a graph for each timestep
            lines = file.readlines()

            timestep = int(lines[1].strip())
            num_entries = int(lines[3].strip())

            # Add all atoms as nodes if not already added
            for i in range(9, 9 + num_entries):
                bond_data = list(map(float, lines[i].strip().split()))
                batom1 = int(bond_data[6])
                batom2 = int(bond_data[7])
                bond_broken = int(bond_data[8])  # 1 if bond is broken, 0 otherwise

                # Add atoms to the set of all atoms
                all_atoms.update([batom1, batom2])
                
                # Add atom nodes to both G and g
                G.add_node(batom1)
                G.add_node(batom2)
                g.add_node(batom1)
                g.add_node(batom2)

                # If bond is not broken, add the edge (bond)
                if bond_broken == 0:
                    G.add_edge(batom1, batom2)
                    g.add_edge(batom1, batom2)
                else:
                    # If the bond is broken, remove the edge if it exists
                    if G.has_edge(batom1, batom2):
                        G.remove_edge(batom1, batom2)
                    if g.has_edge(batom1, batom2):
                        g.remove_edge(batom1, batom2)

            # Ensure all atoms ever seen are still present in the graph
            for atom in all_atoms:
                G.add_node(atom)
                g.add_node(atom)

            # After processing all bonds, compute the number of connected components & save the graph
            Graphs.append(g)
            num_fragments = nx.number_connected_components(G)
            fragments_per_timestep[timestep] = num_fragments
    
    return fragments_per_timestep, Graphs


