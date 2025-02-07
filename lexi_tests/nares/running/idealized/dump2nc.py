#!/usr/bin/env python
# limited output for faster runs and only relevant figures
# we want atom stuff from every timestep, but bonds only from the final timestep to analyze the fracture pattern
# conda env = nares

import os
import glob
import pandas as pd
import numpy as np
import xarray as xr
import netCDF4 as nc
import argparse
from scipy.sparse import coo_matrix
from scipy.sparse import save_npz, load_npz


# Assume these functions exist from our previous code:
def process_bond_dump_file(filepath: os.PathLike, column_names=None):
    with open(filepath, 'r') as file:
        header_lines = [next(file).strip() for _ in range(9)]
    timestep = int(header_lines[1])
    if not column_names:
        column_names = 'batom1 batom2 bbondbroken'
    column_names = column_names.split()
    df = pd.read_csv(filepath, sep=r'\s+', skiprows=9, names=column_names)
    df = df[df['bbondbroken'] != 1]  # keep only bonds that exist
    df.drop('bbondbroken', axis=1, inplace=True)
    ds = df.to_xarray()
    ds = ds.assign_attrs(number_of_bonds=len(df), timestep=timestep)
    return ds

def create_sparse_coo_matrix(ds, all_atom_ids):
    """
    Create a sparse COO matrix from the xarray dataset `ds` containing bond information.
    """
    # Mapping from atom id to matrix index
    atom_to_index = {atom: idx for idx, atom in enumerate(all_atom_ids)}
    batom1 = ds['batom1'].values
    batom2 = ds['batom2'].values

    row_indices = []
    col_indices = []
    data = []
    for a1, a2 in zip(batom1, batom2):
        if a1 in atom_to_index and a2 in atom_to_index:
            row_indices.append(atom_to_index[a1])
            col_indices.append(atom_to_index[a2])
            data.append(1)
    n = len(all_atom_ids)
    return coo_matrix((data, (row_indices, col_indices)), shape=(n, n))


def process_bond_dump_and_create_sparse_matrix(filepath: os.PathLike, all_atom_ids, column_names=None):
    """
    Process a bond dump file and return a sparse COO matrix representing the bonds.
    
    Parameters:
      filepath (os.PathLike): Path to the bond dump file.
      all_atom_ids (list or array-like): List of all atom ids to include in the matrix.
      column_names (str, optional): Space-separated string of column names. Defaults to 'batom1 batom2 bbondbroken'.
      
    Returns:
      coo_matrix: A sparse matrix of shape (n, n) where n is len(all_atom_ids). For each bond 
                  (where bbondbroken is 0), the matrix entry (i, j) is set to 1. Atom ids not present 
                  in the file will have corresponding rows/columns filled with zeros.
    """
    # Read the header lines
    with open(filepath, 'r') as file:
        header_lines = [next(file).strip() for _ in range(9)]
    
    # Extract timestep (if needed for further processing or metadata)
    timestep = int(header_lines[1])
    
    # Use default column names if not provided
    if not column_names:
        column_names = 'batom1 batom2 bbondbroken'
    column_names = column_names.split()
    
    # Read the remainder of the file into a pandas DataFrame
    df = pd.read_csv(filepath, sep=r'\s+', skiprows=9, names=column_names)
    
    # Remove rows where the bond is broken (bbondbroken == 1)
    df = df[df['bbondbroken'] != 1]
    
    # Drop the bond status column as it is no longer needed
    df.drop('bbondbroken', axis=1, inplace=True)
    
    # Create a mapping from atom id to its corresponding index in the final matrix
    atom_to_index = {atom: idx for idx, atom in enumerate(all_atom_ids)}
    
    # Extract bond data
    batom1 = df['batom1'].values
    batom2 = df['batom2'].values

    # Prepare lists for the COO matrix format
    row_indices = []
    col_indices = []
    data = []
    
    # Populate the COO entries: only add bonds where both atom ids exist in the provided list
    for a1, a2 in zip(batom1, batom2):
        if a1 in atom_to_index and a2 in atom_to_index:
            row_indices.append(atom_to_index[a1])
            col_indices.append(atom_to_index[a2])
            data.append(1)  # bond exists
    
    # Define the matrix shape based on the number of atoms
    n = len(all_atom_ids)
    sparse_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(n, n))
    return sparse_matrix


def process_atom_dump_file(filepath: os.PathLike):
    """Read atom dump file and convert to xarray Dataset."""
    with open(filepath, 'r') as file:
        header_lines = [next(file).strip() for _ in range(9)]  

    # get relevant header data from first 9 lines
    timestep = int(header_lines[1])
    number_of_atoms = int(header_lines[3])
    column_names = header_lines[8].split()[2:]  

    # read into pandas dataframe
    df = pd.read_csv(filepath, index_col="id", sep=r'\s+', skiprows=9, names=column_names)
    
    # convert to xarray, add timestep dim and attributes 
    ds = df.to_xarray()
    ds = ds.expand_dims(timestep=[timestep])
    ds = ds.assign_attrs(number_of_atoms=number_of_atoms)

    return ds


def save_atom_ds_and_final_graph(post_dir: os.PathLike, atom_files: str, bond_files: str,
                             atom_ds_output_path: os.PathLike, bond_graph_output_path: os.PathLike):
    """
    Process atom and bond dump files, compute coordination numbers from bond files, and add them
    as a new variable to the atom dataset. For each timestep, the coordination number for an atom is
    computed as the sum of its occurrences in the bond matrix rows (batom1) and columns (batom2).
    
    Parameters:
      post_dir: post irectory containing the atom and bond dump files.
      atom_files: atom file names (commonly dump*.liggghts, where * is the timestep)
      bond_files: atom file names (commonly bfc*.bonds, where * is the timestep)
      output_path: Path to save the updated atom dataset (NetCDF format).
      
    Returns:
      The updated xarray Dataset with a new 'coordination' variable.
    """
    # process all atom dump files and concatenate along the timestep dimension
    atom_filepaths = sorted(glob.glob(os.path.join(post_dir, atom_files))) # sorting is key here!!!
    atom_datasets = [process_atom_dump_file(fp) for fp in atom_filepaths]
    atom_ds = xr.concat(atom_datasets, dim="timestep")
    
    all_atom_ids = list(atom_ds["id"].values)  # list of all atom ids
    
    # process bond dump files in order (make sure they correspond one-to-one with atom files!!!)
    bond_filepaths = sorted(glob.glob(os.path.join(post_dir, bond_files)))
    
    i = 0
    coordination_numbers = []  
    for bond_fp in bond_filepaths:
        #ds_bond = process_bond_dump_file(bond_fp)
        #sparse_mat = create_sparse_coo_matrix(ds_bond, all_atom_ids)
        sparse_mat = process_bond_dump_and_create_sparse_matrix(bond_fp, all_atom_ids)
        # compute coordination nums by summing over rows as matrix is symmetric 
        row_counts = np.array(sparse_mat.sum(axis=1)).flatten()
        coordination_numbers.append(row_counts)
        # save final graph
        i += 1
        if len(bond_filepaths) == i:
            save_npz(bond_graph_output_path, sparse_mat)
            print(f"Saved final graph to {os.path.basename(bond_graph_output_path)}.")
    
    # create an array of shape (n_atoms, n_timesteps)
    coord_array = np.array(coordination_numbers).T  # Transpose so rows correspond to atoms.
    
    # add the coordination numbers as a new variable "coordination" to the atom dataset.
    atom_ds = atom_ds.assign(coordination=(("id", "timestep"), coord_array))
    
    # save the updated dataset.
    atom_ds.to_netcdf(atom_ds_output_path)
    print(f"Saved updated atom dataset with coordination numbers to {os.path.basename(atom_ds_output_path)}.") 

def main():
    parser = argparse.ArgumentParser(description='Process LIGGGHTS simulation data.')
    parser.add_argument('post_directory', help='Path to the post directory containing simulation outputs')
    parser.add_argument('output_directory', help='Path to where you want to output data')
    args = parser.parse_args()

    post_dir = args.post_directory
    out_dir = args.output_directory

    # Check if post_directory exists and is a directory
    if not os.path.isdir(post_dir):
        print(f"Error: The post directory '{post_dir}' does not exist or is not a directory.")
        return

    # delete existing data files
    bond_output_path = os.path.join(out_dir, 'bonds_final.npz')
    if os.path.isfile(bond_output_path):
        os.remove(bond_output_path)
    atom_outpath = os.path.join(out_dir, 'atoms.nc')
    if os.path.isfile(atom_outpath):
        os.remove(atom_outpath) 

    # get new files
    save_atom_ds_and_final_graph(post_dir, atom_files="dump*.liggghts", bond_files="bfc*.bond",
                             atom_ds_output_path=atom_outpath, bond_graph_output_path=bond_output_path)


if __name__ == '__main__':
    main()
