#!/usr/bin/env python

import os
import glob
import pandas as pd
import numpy as np
import xarray as xr
import netCDF4 as nc
import argparse
import pyvista as pv

def process_atom_dump_file(filepath: os.PathLike):
    """Read atom dump file and convert to xarray Dataset."""
    with open(filepath, 'r') as file:
        header_lines = [next(file).strip() for _ in range(9)]  # Read only the first 9 lines

    # Extract specific information from the lines
    timestep = int(header_lines[1])
    number_of_atoms = int(header_lines[3])
    #xlim = (header_lines[5].split()).astype(int)
    column_names = header_lines[8].split()[2:]  # Assumes the 9th line has the relevant data

    # read into pandas dataframe
    df = pd.read_csv(filepath, index_col="id", sep=r'\s+', skiprows=9, names=column_names)
    
    # convert to xarray, add timestep dim and attributes 
    ds = df.to_xarray()
    ds = ds.expand_dims(timestep=[timestep])
    ds = ds.assign_attrs(number_of_atoms=number_of_atoms)

    return ds

def get_atom_ds(directory_path: os.PathLike, output_path: os.PathLike, return_ds=False):
    """Process all dump files from a directory and concatenate into xarray Dataset."""
    filepaths = sorted(glob.glob(os.path.join(directory_path, 'dump*.liggghts')))
    datasets = (process_atom_dump_file(fp) for fp in filepaths)
    ds = xr.concat(datasets, dim="timestep")  # Modify for lost atoms if needed
    ds.to_netcdf(output_path)
    print(f'Saved {os.path.basename(output_path)}.')
    if return_ds:
        return ds

def process_bond_dump_file(filepath: os.PathLike, column_names=None):
    with open(filepath, 'r') as file:
        header_lines = [next(file).strip() for _ in range(9)]  # Read only the first 9 lines

    # Extract specific information from the lines
    timestep = int(header_lines[1])
    if not column_names:
        column_names = 'batom1 batom2 bbondbroken bforceX bforceY bforceZ'
    column_names = column_names.split()

    # read into pandas dataframe
    df = pd.read_csv(filepath, sep=r'\s+', skiprows=9, names=column_names)

    # delete rows where bbondstatus is 1
    df = df[df['bbondbroken'] != 1]

    # delete the bond status column
    df.drop('bbondbroken', axis=1, inplace=True)

    # convert to xarray, add timestep dim and attributes 
    ds = df.to_xarray()
    ds = ds.assign_attrs(number_of_bonds=len(df))
    ds = ds.assign_attrs(timestep=timestep)
    return ds

def get_bond_ds(directory_path: os.PathLike, output_path: os.PathLike, column_names=None, return_ds=False):
    if not column_names:
        column_names = 'batom1 batom2 bbondbroken bforceX bforceY bforceZ'
    filepaths = sorted(glob.glob(os.path.join(directory_path, 'bfc*.bond')))
    with nc.Dataset(output_path, 'w', format='NETCDF4') as f:
        for fpath in filepaths:
            data = process_bond_dump_file(fpath, column_names)
            timestep = data.attrs['timestep']
            nbonds = data.attrs['number_of_bonds']
            group = f.createGroup(f'{timestep}')
            group.createDimension('index', nbonds)
            for v in data.variables:
                variable = group.createVariable(v, 'f4', ('index',))
                variable[:] = data[v]
    print(f'Saved {os.path.basename(output_path)}.')
    if return_ds:
        return nc.Dataset(output_path, 'r')
    
def process_mesh_dump_file(filepath_top_plate: os.PathLike, filepath_bot_plate: os.PathLike):
    """Read mesh dump files and extract the distance between plates."""
    with open(filepath_top_plate, 'r') as file:
        idx_found = -1
        found = False
        y_pos_top = np.nan
        for i, line in enumerate(file):
            if line.startswith("  facet normal 0 -1 0"):
                idx_found = i
                found = True
            if found & (i == idx_found + 2):
                y_pos_top = float(line.split()[2])
                break
    with open(filepath_bot_plate, 'r') as file:
        idx_found = -1
        y_pos_bot = np.nan
        found = False
        for i, line in enumerate(file):
            if line.startswith("  facet normal 0 1 0"):
                found = True
                idx_found = i
            if found & (i == idx_found + 2):
                y_pos_bot = float(line.split()[2])
                break
    timestep = int((os.path.splitext(os.path.basename(filepath_bot_plate))[0])[8:])
    distance_between_plates = y_pos_top - y_pos_bot
    return timestep, distance_between_plates

def get_distanes_btwn_plates(directory_path: os.PathLike, output_path: os.PathLike):
    """Process all dump files from a directory and concatenate into xarray Dataset."""
    filepaths_top = sorted(glob.glob(os.path.join(directory_path, 'mesh_top*.stl')))
    filepaths_bot = sorted(glob.glob(os.path.join(directory_path, 'mesh_bot*.stl')))
    data = (process_mesh_dump_file(fp_top, fp_bot) for fp_top, fp_bot in zip(filepaths_top, filepaths_bot))
    df = pd.DataFrame(data, columns=['t', 'distance'])
    df.set_index('t', inplace=True)
    ds = df.to_xarray()
    ds.to_netcdf(output_path)

def get_avg_stress_from_mesh_file(filepath: os.PathLike):
    '''gets average stress on the mesh from the particles
       definitely not sure if this is doing what it is supposed to
       also there's maybe a better way to get axial stress
    '''
    mesh = pv.read(filepath)
    normal_stress = mesh.cell_data['normal_stress_average']

    # get facets on bottom of plate only
    cell_centers = mesh.cell_centers()
    y_coords = cell_centers.points[:, 1]
    y_threshold = y_coords.min()
    downward_cells_indices = y_coords == y_threshold
    downward_normal_stresses = normal_stress[downward_cells_indices]

    # compute mean ignoring zeros
    non_zero_stresses = downward_normal_stresses[downward_normal_stresses != 0]
    if non_zero_stresses.size == 0:
        return 0.0  
    average_downward_stress = non_zero_stresses.mean()
    timestep = int((os.path.splitext(os.path.basename(filepath))[0])[10:])
    return timestep, average_downward_stress

def get_axial_stress(directory_path: os.PathLike, output_path: os.PathLike):
    """Process all dump files from a directory and concatenate into xarray Dataset."""
    filepaths = sorted(glob.glob(os.path.join(directory_path, 'get_stress*.vtk')))
    data = (get_avg_stress_from_mesh_file(fp) for fp in filepaths)
    df = pd.DataFrame(data, columns=['t', 'axial_stress'])
    df.set_index('t', inplace=True)
    ds = df.to_xarray()
    ds.to_netcdf(output_path)

def process_mesh_dump_file(filepath_top_plate: os.PathLike, filepath_bot_plate: os.PathLike):
    """Read mesh dump files and extract the distance between plates."""
    with open(filepath_top_plate, 'r') as file:
        idx_found = -1
        found = False
        y_pos_top = np.nan
        for i, line in enumerate(file):
            if line.startswith("  facet normal 0 -1 0"):
                idx_found = i
                found = True
            if found & (i == idx_found + 2):
                y_pos_top = float(line.split()[2])
                break
    with open(filepath_bot_plate, 'r') as file:
        idx_found = -1
        y_pos_bot = np.nan
        found = False
        for i, line in enumerate(file):
            if line.startswith("  facet normal 0 1 0"):
                found = True
                idx_found = i
            if found & (i == idx_found + 2):
                y_pos_bot = float(line.split()[2])
                break
    timestep = int((os.path.splitext(os.path.basename(filepath_bot_plate))[0])[8:])
    distance_between_plates = y_pos_top - y_pos_bot
    return timestep, distance_between_plates

def get_axial_strain(directory_path: os.PathLike, output_path: os.PathLike):
    """Process all dump files from a directory and concatenate into xarray Dataset."""
    filepaths_top = sorted(glob.glob(os.path.join(directory_path, 'mesh_top*.stl')))
    filepaths_bot = sorted(glob.glob(os.path.join(directory_path, 'mesh_bot*.stl')))
    data = (process_mesh_dump_file(fp_top, fp_bot) for fp_top, fp_bot in zip(filepaths_top, filepaths_bot))
    df = pd.DataFrame(data, columns=['t', 'strain'])
    df.set_index('t', inplace=True)
    # convert to strain
    df['strain'] = np.abs(df['strain'] - df['strain'].iloc[0])/df['strain'].iloc[0]
    ds = df.to_xarray()
    ds.to_netcdf(output_path)

def main():
    parser = argparse.ArgumentParser(description='Process LIGGGHTS simulation data.')
    parser.add_argument('post_directory', help='Path to the post directory containing simulation outputs')
    parser.add_argument('--column_names', default=None, help='Column names for bond data (default is "batom1 batom2 bbondbroken bforceX bforceY bforceZ")')
    parser.add_argument('output_directory', help='Path to where you want to output data')
    args = parser.parse_args()

    post_dir = args.post_directory
    out_dir = args.output_directory
    column_names = args.column_names

    # Check if post_directory exists and is a directory
    if not os.path.isdir(post_dir):
        print(f"Error: The post directory '{post_dir}' does not exist or is not a directory.")
        return

    # Check if post_directory contains expected files
    atom_files = sorted(glob.glob(os.path.join(post_dir, 'dump*.liggghts')))
    bond_files = sorted(glob.glob(os.path.join(post_dir, 'bfc*.bond')))

    if not atom_files:
        print(f"Error: No atom dump files (dump*.liggghts) found in the post directory '{post_dir}'.")
        return

    if not bond_files:
        print(f"Error: No bond dump files (bfc*.bond) found in the post directory '{post_dir}'.")
        return

    # Process atom data
    atom_output_path = os.path.join(out_dir, 'atoms.nc')
    if os.path.isfile(atom_output_path):
        os.remove(atom_output_path)
    get_atom_ds(post_dir, atom_output_path)

    # Process bond data
    bond_output_path = os.path.join(out_dir, 'bonds.nc')
    if os.path.isfile(bond_output_path):
        os.remove(bond_output_path)
    get_bond_ds(post_dir, bond_output_path, column_names=column_names)

    # Process stress & strain from meshes
    stress_output_path = os.path.join(out_dir, 'stress.nc')
    if os.path.isfile(stress_output_path):
        os.remove(stress_output_path)
    get_axial_stress(post_dir, stress_output_path)
    if os.path.isfile(strain_output_path):
        os.remove(strain_output_path)
    strain_output_path = os.path.join(out_dir, 'strain.nc')
    get_axial_strain(post_dir, strain_output_path)

if __name__ == '__main__':
    main()
