#!/usr/bin/env python

import os
import glob
import pandas as pd
import xarray as xr
import netCDF4 as nc
import argparse

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
    get_atom_ds(post_dir, atom_output_path)

    # Process bond data
    bond_output_path = os.path.join(out_dir, 'bonds.nc')
    get_bond_ds(post_dir, bond_output_path, column_names=column_names)

if __name__ == '__main__':
    main()
