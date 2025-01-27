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
import argparse

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


def plot_final_coord_nums(ds, output_directory, coordnums_df):
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
    y_min = ds['y'].min().values
    y_max = ds['y'].max().values
    padding = 5  

    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_aspect('equal', 'box')
    ax.grid()

    # Atom IDs and initial positions
    atom_ids = ds['id'].values
    x0 = ds['x'].isel(timestep=0).values  
    y0 = ds['y'].isel(timestep=0).values  
    radius0 = ds['radius'].isel(timestep=0).values  

    # Coordination numbers at the first timestep
    coord_nums = coordnums_df.iloc[:, 0].values  # Assuming columns are timesteps

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

    timestep_value = ds['timestep'].values[0]
    ax.set_title(f'Time = {timestep_value} s')

    # Save the figure
    os.makedirs(output_directory, exist_ok=True)
    fpath = os.path.join(output_directory, 'coordnum_first_timestep.jpg')
    plt.savefig(fpath, dpi=300)
    plt.close()


def value_to_color(value, norm, cmap):
    """Convert a scalar value to a matplotlib color."""
    return cmap(norm(value))

def create_atom_positions_animation(ds, output_directory, dt, frame_skip_value=1):
    '''
    in:
        ds (netcdf): dataset of atom states
        output_directory (string): directory where gif should be saved
    returns:
        saves gif of atoms with velocities every frame_skip_value timesteps
        to the output directory
    '''
    fig, ax = plt.subplots(figsize=(8, 8))

    # -- Determine the subset of atoms whose initial y < 0.05 --
    y0 = ds['y'].isel(timestep=0).values  # shape: (id,)
    mask = y0 < 0.05                      # boolean array
    # Filter all IDs using this mask
    atom_ids_full = ds['id'].values       # all IDs
    atom_ids = atom_ids_full[mask]

    # Similarly, filter x, y, z, radius, etc. at time 0
    x0 = ds['x'].isel(timestep=0).values[mask]  # shape: (#atoms that pass mask,)
    z0 = ds['z'].isel(timestep=0).values[mask]
    radius0 = ds['radius'].isel(timestep=0).values[mask]

    # Get min/max for x, z to define plot limits
    x_min = ds['x'].min().values
    x_max = ds['x'].max().values
    z_min = ds['z'].min().values
    z_max = ds['z'].max().values
    padding = 0.1  

    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(z_min - padding, z_max + padding)
    ax.set_aspect('equal', 'box')
    ax.grid()

    # Create circle patches for each valid atom
    circles = {}
    for idx, atom_id in enumerate(atom_ids):
        circle = Circle((x0[idx], z0[idx]), radius0[idx], alpha=0.9)
        ax.add_patch(circle)
        circles[atom_id] = circle

    # Set up colormap & colorbar for velocity magnitude
    cmap = cm.jet
    norm = mcolors.LogNorm(vmin=0.001, vmax=1)  # adjust if needed
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Velocity Magnitude [m/s]')

    # The update function for animation
    def update(frame):
        # Grab positions for all IDs, then slice by mask
        x = ds['x'].isel(timestep=frame).values[mask]
        z = ds['z'].isel(timestep=frame).values[mask]

        # Also need velocities for color
        vx = ds['vx'].isel(timestep=frame).values[mask]
        vy = ds['vy'].isel(timestep=frame).values[mask]
        vz = ds['vz'].isel(timestep=frame).values[mask]

        for idx, atom_id in enumerate(atom_ids):
            # Update position
            circles[atom_id].center = (x[idx], z[idx])

            # Compute velocity magnitude and set color
            vmag = np.sqrt(vx[idx]**2 + vy[idx]**2 + vz[idx]**2)
            color = value_to_color(vmag, norm, cmap)
            circles[atom_id].set_color(color)

            # Example fade-out if speed is large
            if vmag > 1:
                circles[atom_id].set_alpha(0.1)
            else:
                circles[atom_id].set_alpha(0.9)

        timestep_value = ds['timestep'].values[frame]
        ax.set_title(f'Time = {np.round(timestep_value*dt, decimals = 2)} s')
        return list(circles.values())

    # Build animation using only the frames you want
    frame_indices = range(0, len(ds['timestep']), frame_skip_value)
    ani = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        blit=False
    )

    # Save to GIF
    os.makedirs(output_directory, exist_ok=True)
    fpath = os.path.join(output_directory, 'sim.gif')
    ani.save(fpath, writer='Pillow', fps=5)



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

def plot_force_and_stress_strain_on_top_plate(ds_top_plate, dt, output_directory, l = 0.4,  strain_rate = 7.5e-3):
    fig1 = plt.figure()
    v_top = strain_rate * l
    force = ds_top_plate['force'].values
    time = ds_top_plate['timestep'].values*dt
    plt.plot(time, force, color = 'maroon')
    plt.ylabel(r'$F_z$ [N]')
    plt.xlabel('Time [s]')
    plt.grid()
    outpath = os.path.join(output_directory, 'F_on_top_plate.jpg')
    plt.savefig(outpath, dpi = 300)
    plt.close()

    fig2 = plt.figure()
    

    stress = force / (0.5*l)**2
    strain = v_top * time / l
    plt.plot(strain, stress, color = 'navy')
    plt.ylabel(r'Axial Stress [Pa]')
    plt.xlabel('Axial Strain [-]')
    plt.grid()
    outpath = os.path.join(output_directory, 'stress_v_strain.jpg')
    plt.savefig(outpath, dpi = 300)
    plt.close()
    
    max_stress = np.max(stress)
    print(f'max stress = {np.round(max_stress*1e-6, decimals = 2)} MPa')
    max_stress_idx = np.argmax(stress)
    strain_at_max_stress = strain[max_stress_idx]
    ds_top_plate['stress'] = stress
    ds_top_plate['strain'] = strain
    ds_top_plate.attrs['effective elastic modulus'] = max_stress/strain_at_max_stress
    ds_top_plate.attrs['max stress'] = max_stress
    print(f'effective elastic modulus = {np.round(max_stress/strain_at_max_stress*1e-9, decimals = 2)} GPa')
    ds_top_plate.to_netcdf(os.path.join(output_directory, 'stress_strain_data.nc'))



# add main function to create figures from terminal
if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate figures from LIGGGHTS simulation NetCDF outputs.")
    parser.add_argument("--output-dir", required=True, help="Path to the output directory containing NetCDF files.")
    parser.add_argument("--dt", type=float, default=0.000001, help="Timestep for the simulation.")
    args = parser.parse_args()

    # Extract arguments
    output_directory = args.output_dir
    dt = args.dt

    coordnum_initial = False
    final_floes = True
    stress_strain = True
    simulation_gif = False

    # Open datasets
    fpath_a = os.path.join(output_directory, 'atoms.nc')
    fpath_b = os.path.join(output_directory, 'bonds_final.nc')
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

    if simulation_gif:
        create_atom_positions_animation(ds_a, output_directory, dt, frame_skip_value=1)
    
    ds_a.close()

    if stress_strain:
        ds_top_plate = xr.open_dataset(os.path.join(output_directory, 'plate.nc'))
        plot_force_and_stress_strain_on_top_plate(ds_top_plate, dt, output_directory, l = 0.4,  strain_rate = 7.5e-3)
        ds_top_plate.close()