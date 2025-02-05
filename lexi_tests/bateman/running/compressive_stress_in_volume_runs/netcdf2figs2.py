#!/usr/bin/env python3
# limited output for faster runs and only relevant figures
# conda environment: netcdf2figs

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import xarray as xr
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import netCDF4 as nc
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import random

# backend stuff
import matplotlib
matplotlib.use('Agg')


# massive bond function to get bond stuff
def process_bond_file(ds_b, num_atoms):
    """
    returns bond data from bond netcdf file

    input:
        ds_b (netcdf4): bond netcdf file
        num_atoms (int): the total number of atoms.

    returns:
        n_bonds (np.ndarray): array of the number of bonds at each processed timestep
        average_bond_force_array (np.ndarray): Array of average bond forces at each processed timestep.
        adjacency_matrix (scipy.sparse.coo_matrix): Adjacency matrix for the last processed timestep.
        coordination_numbers_df (pd.DataFrame): DataFrame with coordination numbers for each atom at each timestep.

    TODO: this code can be easily modified to give the FSD at each timestep w/o storing a ton of data
    """

    # 1. compute number of bonds
    batom1 = ds_b['batom1'].values
    batom2 = ds_b['batom2'].values

    # 2. compute coordination numbers for each atom
    atom_ids = np.arange(1, num_atoms + 1)  
    atom_id_to_index = {atom_id: idx for idx, atom_id in enumerate(atom_ids)} # deals with indices

    # map batom1 and batom2 to indices
    try:
        batom1_indices = np.array([atom_id_to_index[atom_id] for atom_id in batom1])
        batom2_indices = np.array([atom_id_to_index[atom_id] for atom_id in batom2])
    except KeyError as e:
        raise ValueError(f"Atom ID {e.args[0]} not found in the expected range 1 to {num_atoms}.")

    # build the adjacency matrix
    data = np.ones(len(batom1_indices))
    row = batom1_indices
    col = batom2_indices
    adjacency = coo_matrix((data, (row, col)), shape=(num_atoms, num_atoms))

    # make the matrix symmetric -> don't need this step? since we parse all bonds?
    adjacency = adjacency + adjacency.transpose()
    adjacency.data = np.ones_like(adjacency.data) # ensure that duplicate entries are set to 1

    # compute & store coordination numbers for each atom
    coordination_numbers = np.array(adjacency.sum(axis=1)).flatten()

    # create dataframe with coordination numbers
    coordination_numbers_df = pd.DataFrame(
        coordination_numbers,
        index=atom_ids
    )
    coordination_numbers_df.index.name = 'id'

    return adjacency, coordination_numbers

def get_bond_fsd_from_graph(scipy_sparse_graph):
    number_of_connected_components, labels = connected_components(csgraph=scipy_sparse_graph, directed=False, return_labels=True)
    component_sizes = np.bincount(labels)
    return number_of_connected_components, labels, component_sizes

def value_to_color(value, norm, cmap):
    normed = norm(value)
    return cmap(normed)

def plot_final_coord_nums(ds, output_directory, coord_nums):
    """
    Plots the atom positions at the first timestep, colored by coordination numbers.

    Parameters:
        ds (xarray.Dataset): Dataset of atom states.
        output_directory (str): Directory where the plot should be saved.
        coordination_numbers_df (pd.DataFrame): DataFrame containing coordination numbers for each atom at each timestep.

    Returns:
        None (saves a plot to the output directory).
    """

    fig, ax = plt.subplots(figsize=(8, 8))

    # Get plot limits
    x_min = ds['x'].min().values
    x_max = ds['x'].max().values
    y_min = ds['z'].min().values
    y_max = ds['z'].max().values
    padding = 0.2  

    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_aspect('equal', 'box')
    ax.grid()

    # Atom IDs and initial positions
    atom_ids = ds['id'].values
    x0 = ds['x'].isel(timestep=0).values  
    y0 = ds['z'].isel(timestep=0).values  
    radius0 = ds['radius'].isel(timestep=0).values  


    # Set up colormap and normalization
    cmap = cm.jet
    max_coord_num = coord_nums.max()
    norm = mcolors.Normalize(vmin=0, vmax=max_coord_num)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Coordination Number')

    # Plot each atom as a circle colored by coordination number
    for idx in range(len(atom_ids)):
        x = x0[idx]
        y = y0[idx]
        radius = radius0[idx]
        coord_num = coord_nums[idx]
        color = cmap(norm(coord_num))

        circle = Circle((x, y), radius, alpha=0.5, color=color)
        ax.add_patch(circle)

    ax.set_title(f'Time = 0 s')

    # Save the figure
    os.makedirs(output_directory, exist_ok=True)
    fpath = os.path.join(output_directory, 'coordnum_final_timestep.jpg')
    plt.savefig(fpath, dpi=300)
    plt.close()


def plot_final_floes(t, ax, ds, labels, component_sizes):
    ax.clear()  # Clear the previous frame
    # TODO modify dump2netcdf to make these attributes of the data
    x_min = ds['x'].min().values
    x_max = ds['x'].max().values
    y_min = ds['z'].min().values
    y_max = ds['z'].max().values
    padding = 0.2 

    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_aspect('equal', 'box')
    ax.grid()

    # Determine the largest fragment by size
    largest_frag_id = np.argmax(component_sizes)

    # Map atom IDs to fragment IDs
    atom_ids = ds['id'].values
    # Ensure atom_ids are zero-based indices matching the graph nodes
    # If atom_ids are not zero-based or do not correspond to node indices, adjust accordingly
    atom_to_fragment = dict(zip(atom_ids, labels))

    # Fetch paired colors and prepare to assign them randomly
    tab20_colors = plt.get_cmap('Paired').colors
    random_colors = random.choices(tab20_colors, k=number_of_connected_components)

    # Assign colors: black for single-atom fragments, blue for the largest fragment
    fragment_colors = {}
    for frag_id in range(number_of_connected_components):
        size = component_sizes[frag_id]
        if size == 1:
            fragment_colors[frag_id] = (0, 0, 0, 1)  # black for single-atom fragments
        elif frag_id == largest_frag_id:
            fragment_colors[frag_id] = 'b'  # color large fragment blue
        else:
            fragment_colors[frag_id] = random_colors[frag_id]

    # Get positions and radii for all atoms at timestep t
    x0 = ds['x'].isel(timestep=t).values
    y0 = ds['z'].isel(timestep=t).values
    radius0 = ds['radius'].isel(timestep=t).values

    # Create a circle for each atom and add to the axes
    for idx, atom_id in enumerate(atom_ids):
        initial_x = x0[idx]
        initial_y = y0[idx]
        radius = radius0[idx]
        frag_id = atom_to_fragment.get(atom_id)
        if frag_id is not None:
            col = fragment_colors[frag_id]
        else:
            col = (0.5, 0.5, 0.5, 1)  # color unbonded particles black
        circle = Circle((initial_x, initial_y), radius, color=col, alpha=0.5)
        ax.add_patch(circle)

def plot_force_on_top_plate(ds_top_plate, dt, output_directory):
    fig = plt.figure()
    force = ds_top_plate['fz'].sum(dim='id').values
    plt.plot(ds_top_plate['timestep'].values*dt, force, color = 'maroon')
    plt.ylabel(r'$F_z$ [N]')
    plt.xlabel('Time [s]')
    plt.grid()
    outpath = os.path.join(output_directory, 'F_on_top_plate.jpg')
    plt.savefig(outpath, dpi = 300)
    plt.close()
    return force

def get_total_particle_volume(ds_a):
    first_step_radius = ds_a['radius'].isel(timestep=0) 
    # # Compute volume per ball 
    volume_per_ball = (4.0/3.0) * np.pi * (first_step_radius**3) 
    total_volume = volume_per_ball.sum(dim='id')
    return float(total_volume)

def get_bond_multiplier_factor(ds_a):
    total_volume = get_total_particle_volume(ds_a)
    max_radius = ds_a['radius'].max().max()
    domain_volume = 2*max_radius*100**2 # hard coding domain size, be careful!
    n = total_volume/domain_volume
    fac = -(1-n)/total_volume
    return fac.values

def get_compresive_stress_in_volume(ds_a, sig_zzs):
    '''
    uses the avg stress equation to get the avg compressive stress
    \sigma_zz in the volume. need to add to the parse bond function. 
    '''
    fac = get_bond_multiplier_factor(ds_a)
    return fac*np.array(sig_zzs)

# add main function to create figures from terminal
if __name__ == '__main__':
    # Interactive inputs
    rel_path = input(
        "Enter base path (from '/mnt/c/Users/arlenlex/Documents/liggghts_data/bateman/simulations'): "
    )
    base_path = os.path.join(r'/mnt/c/Users/arlenlex/Documents/liggghts_data/bateman/simulations', rel_path)

    dt_input = input("Enter dt (default: 0.0000005): ")
    dt = float(dt_input) if dt_input else 0.0000005

    output_directory = input("Enter output directory if different from base path: ")
    if not output_directory:
        output_directory = base_path

    coordnum_initial_input = input("Plot the final coordination numbers? (Y/N, default: N): ")
    coordnum_initial = coordnum_initial_input.lower() == 'y' if coordnum_initial_input else False

    final_floes_input = input("Compute final floes? (Y/N, default: N): ")
    final_floes = final_floes_input.lower() == 'y' if final_floes_input else False

    stress_strain_input = input("Plot force on top plate? (Y/N, default: N): ")
    stress_strain = stress_strain_input.lower() == 'y' if stress_strain_input else False

    # Open datasets
    fpath_a = os.path.join(base_path, 'all_atoms_final.nc')
    fpath_b = os.path.join(base_path, 'bonds_final.nc')
    ds_a = xr.open_dataset(fpath_a)
    ds_b = xr.open_dataset(fpath_b)

    # Define/get some arrays
    n_atoms = ds_a.attrs['number_of_atoms'].item()

    # Get bond data if any related option is True
    if final_floes or coordnum_initial:
        final_graph, coordination_numbers = process_bond_file(ds_b, n_atoms)
        number_of_connected_components, labels, component_sizes = get_bond_fsd_from_graph(final_graph)

    # Close bond dataset
    ds_b.close()

    if coordnum_initial:
        plot_final_coord_nums(ds_a, output_directory, coordination_numbers)

    if final_floes:
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_final_floes(-1, ax, ds_a, labels, component_sizes)
        outpath = os.path.join(output_directory, 'final_floes.jpg')
        plt.savefig(outpath, dpi = 300)
        plt.close()

    if stress_strain:
        ds_top_plate = xr.open_dataset(os.path.join(base_path, 'atoms_plate.nc'))
        force = plot_force_on_top_plate(ds_top_plate, dt, output_directory)
        ds_top_plate.close()
        np.save(os.path.join(output_directory, 'force_on_top_plate.npy'), force)

