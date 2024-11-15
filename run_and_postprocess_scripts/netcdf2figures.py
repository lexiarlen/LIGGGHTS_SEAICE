#!/usr/bin/env python3

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
        ds_b (netcdf4 w/ groups): bond netcdf file
        num_atoms (int): the total number of atoms.

    returns:
        n_bonds (np.ndarray): array of the number of bonds at each processed timestep
        average_bond_force_array (np.ndarray): Array of average bond forces at each processed timestep.
        adjacency_matrix (scipy.sparse.coo_matrix): Adjacency matrix for the last processed timestep.
        coordination_numbers_df (pd.DataFrame): DataFrame with coordination numbers for each atom at each timestep.

    TODO: this code can be easily modified to give the FSD at each timestep w/o storing a ton of data
    """

    # get bond dataset group names
    group_names = list(ds_b.groups.keys())

    # initialize empty arrays to store data
    n_bonds_list = []
    average_bond_force_list = []
    adjacency_matrix = None
    coordination_numbers_dict = {}

    # parse bond dataset
    for idx in range(len(group_names)):
        group_name = group_names[idx]
        group = ds_b.groups[group_name]

        # 1. compute number of bonds
        batom1 = np.array(group.variables['batom1'][:], dtype=int)
        batom2 = np.array(group.variables['batom2'][:], dtype=int)
        n_bonds = len(batom1)
        n_bonds_list.append(n_bonds)

        # 2. compute average bond force
        bforcex = group.variables['bforceX'][:]
        bforcey = group.variables['bforceY'][:]
        bforcez = group.variables['bforceZ'][:]

        force_magnitudes = np.sqrt(bforcex**2 + bforcey**2 + bforcez**2)
        avg_force = np.mean(force_magnitudes)
        average_bond_force_list.append(avg_force)

        # 3. compute coordination numbers for each atom
        atom_ids = np.arange(1, num_atoms + 1)  
        atom_id_to_index = {atom_id: idx for idx, atom_id in enumerate(atom_ids)} # deals with indices

        # map batom1 and batom2 to indices
        try:
            batom1_indices = np.array([atom_id_to_index[atom_id] for atom_id in batom1])
            batom2_indices = np.array([atom_id_to_index[atom_id] for atom_id in batom2])
        except KeyError as e:
            raise ValueError(f"Atom ID {e.args[0]} not found in the expected range 1 to {num_atoms}.")

        # build the adjacency matrix for this timestep
        data = np.ones(len(batom1_indices))
        row = batom1_indices
        col = batom2_indices
        adjacency = coo_matrix((data, (row, col)), shape=(num_atoms, num_atoms))

        # make the matrix symmetric -> don't need this step? since we parse all bonds?
        adjacency = adjacency + adjacency.transpose()
        adjacency.data = np.ones_like(adjacency.data) # ensure that duplicate entries are set to 1

        # compute & store coordination numbers for each atom
        coordination_numbers = np.array(adjacency.sum(axis=1)).flatten()
        coordination_numbers_dict[group_name] = coordination_numbers

        # if this is the last processed timestep, store the adjacency matrix = final_graph
        if idx + 1 == len(group_names):
            adjacency_matrix = adjacency

    # convert lists to arrays
    n_bonds = np.array(n_bonds_list)
    average_bond_force_array = np.array(average_bond_force_list)

    # create dataframe with coordination numbers
    coordination_numbers_df = pd.DataFrame(
        coordination_numbers_dict,
        index=atom_ids
    )
    coordination_numbers_df.index.name = 'id'

    return n_bonds, average_bond_force_array, adjacency_matrix, coordination_numbers_df

def get_bond_fsd_from_graph(scipy_sparse_graph):
    number_of_connected_components, labels = connected_components(csgraph=scipy_sparse_graph, directed=False, return_labels=True)
    component_sizes = np.bincount(labels)
    return number_of_connected_components, labels, component_sizes

def value_to_color(value, norm, cmap):
    normed = norm(value)
    return cmap(normed)

def create_atom_positions_animation(ds, output_directory, coordnums_df, frame_skip_value=1):
    '''
    in:
        ds (netcdf): dataset of atom states
        output_directory (string): directory where gif should be saved
        coordnums_df (pandas dataframe): contains atom coordination numbers
    returns:
        saves gif of atoms with their coordination numbers every frame_skip_value timesteps to the output directory
    '''
    fig, ax = plt.subplots(figsize=(8, 8))

    # get plot limits
    # TODO modify dump2netcdf to make these attributes of the data
    x_min = ds['x'].min().values
    x_max = ds['x'].max().values
    y_min = ds['y'].min().values
    y_max = ds['y'].max().values
    padding = 5  

    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_aspect('equal', 'box')
    ax.grid()

    # initialize circles
    circles = {}
    atom_ids = ds['id'].values  

    # get initial positions and radii for all atoms at the first timestep
    x0 = ds['x'].isel(timestep=0).values  # Shape: (id,)
    y0 = ds['y'].isel(timestep=0).values  # Shape: (id,)
    radius0 = ds['radius'].isel(timestep=0).values  # Shape: (id,)

    # create a circle for each atom and add to the axes
    for idx, atom_id in enumerate(atom_ids):
        initial_x = x0[idx]
        initial_y = y0[idx]
        radius = radius0[idx]
        circle = Circle((initial_x, initial_y), radius, alpha=0.5)
        ax.add_patch(circle)
        circles[atom_id] = circle

    # set up colormap and normalization (adjust vmin and vmax as needed)
    cmap = cm.jet
    max_coord_num = coordnums_df.max().max().item()
    norm = mcolors.Normalize(vmin=0, vmax=max_coord_num)  
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Coordination Number')  # Adjust label if needed

    # update function for animation
    def update(frame):
        x = ds['x'].isel(timestep=frame).values
        y = ds['y'].isel(timestep=frame).values

        for idx, atom_id in enumerate(atom_ids):
            circles[atom_id].center = (x[idx], y[idx])
            coord_num = coordination_numbers_df.loc[atom_id].iloc[frame]
            color = value_to_color(coord_num, norm, cmap)
            circles[atom_id].set_color(color)

        timestep_value = ds['timestep'].values[frame]
        ax.set_title(f'Time = {timestep_value} s')
        return list(circles.values())

    frame_indices = range(0, len(ds['timestep']), frame_skip_value)

    # create animation
    ani = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        blit=False
    )

    # save gif
    os.makedirs(output_directory, exist_ok=True)
    fpath = os.path.join(output_directory, 'coordnum.gif')
    ani.save(fpath, writer='Pillow', fps=2)

def plot_initial_coord_nums(ds, output_directory, coordnums_df):
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
    coord_nums = coordination_numbers_df.iloc[:, 0].values  # Assuming columns are timesteps

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
    fpath = os.path.join(output_directory, 'coordnum_first_timestep.png')
    plt.savefig(fpath, dpi=300)
    plt.close()

# save figures for all time averaged quantities
def plot_and_save_quantity(x, y, units, title, color, output_directory, output_name):
    fig = plt.figure()
    plt.plot(x, y, color = color)
    plt.ylabel(units)
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.grid()

    outpath = os.path.join(output_directory, output_name)
    plt.savefig(outpath, dpi = 300)
    plt.close()


def plot_fsd(component_sizes, output_directory):
    fig = plt.figure()
    hist, bin_edges = np.histogram(component_sizes, bins=np.logspace(0, np.log2(np.max(component_sizes))+0.1, 10, base = 2))
    plt.scatter(bin_edges[1:], hist/bin_edges[1:], s = 40, color = 'g')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'DEs / floe')
    plt.ylabel(r'Floe # Density')
    plt.title('FSD')
    plt.grid()

    outpath = os.path.join(output_directory, 'fsd.png')
    plt.savefig(outpath, dpi = 300)
    plt.close()

def plot_final_floes(t, ax, ds, labels, component_sizes):
    ax.clear()  # Clear the previous frame
    # TODO modify dump2netcdf to make these attributes of the data
    x_min = ds['x'].min().values
    x_max = ds['x'].max().values
    y_min = ds['y'].min().values
    y_max = ds['y'].max().values
    padding = 5  

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
    y0 = ds['y'].isel(timestep=t).values
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


# add main function to create figures from terminal
if __name__ == '__main__':
    # Interactive inputs
    base_path = input(
        "Enter base path (default: '/mnt/c/Users/arlenlex/Documents/liggghts_data/dynamic_forcing/crystallized_floe/cyclone'): "
    )
    if not base_path:
        base_path = r'/mnt/c/Users/arlenlex/Documents/liggghts_data/dynamic_forcing/crystallized_floe/cyclone'

    dt_input = input("Enter dt (default: 0.0008): ")
    dt = float(dt_input) if dt_input else 0.0008

    output_directory = input("Enter output directory (default: '/mnt/c/Users/arlenlex/Documents/liggghts_data/dynamic_forcing/crystallized_floe/cyclone'): ")
    if not output_directory:
        output_directory = '/mnt/c/Users/arlenlex/Documents/liggghts_data/dynamic_forcing/crystallized_floe/cyclone'

    time_average_data_input = input("Time average data? (Y/N, default: Y): ")
    time_average_data = time_average_data_input.lower() == 'y' if time_average_data_input else True

    coordnum_gif_input = input("Create coordination number GIF? (Y/N, default: Y): ")
    coordnum_gif = coordnum_gif_input.lower() == 'y' if coordnum_gif_input else True

    coordnum_initial_input = input("Plot the initial coordination numbers? (Y/N, default: Y): ")
    coordnum_initial = coordnum_initial_input.lower() == 'y' if coordnum_initial_input else True

    final_floes_input = input("Compute final floes? (Y/N, default: Y): ")
    final_floes = final_floes_input.lower() == 'y' if final_floes_input else True

    fsd_input = input("Compute FSD? (Y/N, default: Y): ")
    fsd = fsd_input.lower() == 'y' if fsd_input else True

    # Open datasets
    fpath_a = os.path.join(base_path, 'atoms.nc')
    fpath_b = os.path.join(base_path, 'bonds.nc')
    ds_a = xr.open_dataset(fpath_a)
    ds_b = nc.Dataset(fpath_b, 'r')

    # Define/get some arrays
    n_atoms = ds_a.attrs['number_of_atoms'].item()

    # Get bond data if any related option is True
    if final_floes or time_average_data or coordnum_gif or fsd:
        nbonds, bond_force, final_graph, coordination_numbers_df = process_bond_file(ds_b, n_atoms)
        number_of_connected_components, labels, component_sizes = get_bond_fsd_from_graph(final_graph)

    # Close bond dataset
    ds_b.close()

    # Create coordination number animation if selected
    if coordnum_gif:
        create_atom_positions_animation(ds_a, output_directory, coordination_numbers_df, frame_skip_value=5)

    if coordnum_initial:
        plot_initial_coord_nums(ds_a, output_directory, coordination_numbers_df)

    if final_floes:
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_final_floes(-1, ax, ds_a, labels, component_sizes)
        outpath = os.path.join(output_directory, 'final_floes.png')
        plt.savefig(outpath, dpi = 300)

    # Compute and plot time-averaged data if selected
    if time_average_data:
        time = ds_a['timestep'].values * dt

        # Compute quantities
        ds_a['fmag'] = np.sqrt(ds_a['fx']**2 + ds_a['fy']**2 + ds_a['fz']**2)
        ds_a['vmag'] = np.sqrt(ds_a['vx']**2 + ds_a['vy']**2 + ds_a['vz']**2)

        avg_sxy = ds_a['c_stress[4]'].mean(dim='id')
        avg_vmag = ds_a['vmag'].mean(dim='id')
        avg_fmag = ds_a['fmag'].mean(dim='id')

        # Prepare data for plotting
        time_avg_quantities = [avg_sxy, avg_fmag, avg_vmag, nbonds, bond_force]
        units = ['[Pa]', '[N]', '[m/s]', '[#]', '[N]']
        titles = [r'$\overline{\sigma_{xy}}$', r'$|\overline{\bf{F}_a}|$', r'$|\overline{\bf{v}}|$',
                  '# bonds', r'$|\overline{\bf{F}_b}|$']
        output_names = ['shear_stress.png', 'a_fmag.png', 'vmag.png', 'nbonds.png', 'b_fmag.png']
        colors = ['r', 'indigo', 'b', 'tab:orange', 'hotpink']

        # Plot and save each quantity
        for i in range(len(time_avg_quantities)):
            plot_and_save_quantity(time, time_avg_quantities[i], units[i], titles[i], colors[i],
                                   output_directory, output_names[i])

    # Close atom dataset
    ds_a.close()

    # Compute and plot FSD if selected
    if fsd:
        plot_fsd(component_sizes, output_directory)
