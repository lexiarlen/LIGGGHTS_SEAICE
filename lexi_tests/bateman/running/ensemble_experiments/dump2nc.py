#!/usr/bin/env python
# limited output for faster runs and only relevant figures
# we want atom stuff from every timestep, but bonds only from the final timestep to analyze the fracture pattern

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

def get_atom_ds(directory_path: os.PathLike, file_names, output_path: os.PathLike, return_ds=False):
    """Process all dump files from a directory and concatenate into xarray Dataset."""
    filepaths = sorted(glob.glob(os.path.join(directory_path, file_names)))
    datasets = (process_atom_dump_file(fp) for fp in filepaths)
    ds = xr.concat(datasets, dim="timestep")  # Modify for lost atoms if needed
    ds.to_netcdf(output_path)
    print(f'Saved {os.path.basename(output_path)}.')
    if return_ds:
        return ds

def get_final_bond_ds(directory_path: os.PathLike, output_path: os.PathLike, column_names=None):
    filepath = sorted(glob.glob(os.path.join(directory_path, 'bfc_final*.bond')))[-1]# take just the last one
    with open(filepath, 'r') as file:
        header_lines = [next(file).strip() for _ in range(9)]  # Read only the first 9 lines

    # Extract specific information from the lines
    timestep = int(header_lines[1])
    column_names = 'batom1 batom2 bbondbroken'
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
    ds.to_netcdf(output_path)
    print(f'Saved {os.path.basename(output_path)}.')

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

    # Process plate data
    plate_output_path = os.path.join(out_dir, 'atoms_plate.nc')
    if os.path.isfile(plate_output_path):
        os.remove(plate_output_path)
    get_atom_ds(post_dir, 'top_plate*.liggghts', plate_output_path)

    # Process bond data for final figure
    bond_output_path = os.path.join(out_dir, 'bonds_final.nc')
    if os.path.isfile(bond_output_path):
        os.remove(bond_output_path)
    get_final_bond_ds(post_dir, bond_output_path)

    # Process all atom data for final figure
    all_atoms_output_path = os.path.join(out_dir, 'all_atoms_final.nc')
    if os.path.isfile(all_atoms_output_path):
        os.remove(all_atoms_output_path)
    get_atom_ds(post_dir, 'all_atoms*.liggghts', all_atoms_output_path)

if __name__ == '__main__':
    main()
