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
from scipy.sparse import load_npz
import matplotlib.ticker as ticker
import glob

# backend stuff
import matplotlib
matplotlib.use('Agg')

def value_to_color(value, norm, cmap):
    normed = norm(value)
    return cmap(normed)

def plot_initial_coord_nums(ds, output_directory):
    '''
    IN:
        ds (netcdf): dataset of atom states
        output_directory (string): directory where gif should be saved
    OUT:
        saves gif of atoms with their coordination numbers every frame_skip_value timesteps to the output directory
    '''
    fig, ax = plt.subplots(figsize=(8, 8))

    # get plot limits
    x_min = ds['x'].min().values
    x_max = ds['x'].max().values
    y_min = ds['y'].min().values
    y_max = ds['y'].max().values
    padding = 5e3  

    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_aspect('equal', 'box')
    ax.grid()

    atom_ids = ds['id'].values  

    # get initial positions and radii for all atoms at the first timestep
    x0 = ds['x'].isel(timestep=0).values  # Shape: (id,)
    y0 = ds['y'].isel(timestep=0).values  # Shape: (id,)
    radius0 = ds['radius'].isel(timestep=0).values  # Shape: (id,)

    # set up colormap and normalization (adjust vmin and vmax as needed)
    cmap = cm.jet
    max_coord_num = ds['coordination'].max().max()
    norm = mcolors.Normalize(vmin=0, vmax=max_coord_num)  
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Coordination Number')  

    # create a circle for each atom and add to the axes
    for idx, atom_id in enumerate(atom_ids):
        circle = Circle((x0[idx], y0[idx]), radius0[idx], alpha=0.5)
        ax.add_patch(circle)
        coord_num = ds['coordination'].loc[atom_id].isel(timestep = 0)
        color = value_to_color(coord_num, norm, cmap)
        circle.set_color(color)

        ax.set_title(f'Time = 0 s')

    # save gif
    os.makedirs(output_directory, exist_ok=True)
    fpath = os.path.join(output_directory, 'coordnum.jpg')
    fig.savefig(fpath)

def m_to_km(x, pos):
    return f"{x/1000:g}"  # Remove trailing zeros if needed


def create_bond_broken_animation(ds, output_directory, dt, frame_skip_value=4):
    """
    IN:
        ds (netcdf): dataset of atom states containing at least the fields:
            'id'            : atom identifiers
            'x', 'y'        : atom positions (per timestep)
            'radius'        : atom radii (per timestep)
            'coordination'  : coordination number (per timestep)
            'timestep'      : timestep values
        output_directory (string): directory where gif should be saved
        dt (float): conversion factor to scale timestep values into seconds
        frame_skip_value (int): use every frame_skip_value-th timestep in the animation
    OUT:
        Saves a gif of atoms where each atom’s color represents the cumulative number
        of bonds broken (tracked in bbroken_colors) with a colorbar mapping 0 to 5.
        Atoms are outlined with a thin grey border.
    """

    fig, ax = plt.subplots(figsize=(8, 8))

    # Set plot limits with padding.
    x_min = ds['x'].min().values
    x_max = ds['x'].max().values
    y_min = ds['y'].min().values
    y_max = ds['y'].max().values
    padding = 5e3

    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)

    formatter = ticker.FuncFormatter(m_to_km)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # Add labels for the axes indicating kilometers.
    ax.set_xlabel("[km]")
    ax.set_ylabel("[km]")
    ax.set_aspect('equal', 'box')

    # Create a scalar mappable for bonds broken.
    cmap = cm.Reds  
    norm = mcolors.Normalize(vmin=0, vmax=6)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Bonds Broken')

    # Get atom ids.
    atom_ids = ds['id'].values

    # Precompute the cumulative number of bonds broken for each atom over time.
    # bbroken_colors is a dictionary with keys = atom id and values = array over timesteps.
    bbroken_colors = {}
    for atom_id in atom_ids:
        # Retrieve the coordination numbers for this atom as a 1D array over time.
        coord_array = ds['coordination'].loc[atom_id].values
        # Initialize an array of the same shape for the cumulative broken-bond count.
        broken = np.zeros_like(coord_array, dtype=float)
        broken[0] = 0.0  # At timestep 0, no bonds have been broken.
        for t in range(1, len(coord_array)):
            diff = coord_array[t - 1] - coord_array[t]
            # Only count a drop in coordination.
            if diff > 0:
                broken[t] = broken[t - 1] + diff
            else:
                broken[t] = broken[t - 1]
        bbroken_colors[atom_id] = broken

    # Initialize a dictionary to hold the Circle patches, keyed by atom id.
    circles = {}

    # Get initial positions and radii for all atoms at timestep 0.
    x0 = ds['x'].isel(timestep=0).values
    y0 = ds['y'].isel(timestep=0).values
    radius0 = ds['radius'].isel(timestep=0).values

    # Create a circle for each atom (with a thin grey border).
    for idx, atom_id in enumerate(atom_ids):
        circle = Circle((x0[idx], y0[idx]), radius0[idx], alpha=0.9,
                        edgecolor='grey', linewidth=0.5)
        ax.add_patch(circle)
        circles[atom_id] = circle

    # Update function for the animation.
    def update(frame):
        x = ds['x'].isel(timestep=frame).values
        y = ds['y'].isel(timestep=frame).values

        for idx, atom_id in enumerate(atom_ids):
            # Update atom position.
            circles[atom_id].center = (x[idx], y[idx])
            # Retrieve the precomputed bonds broken for this atom at the current timestep.
            bonds_broken = bbroken_colors[atom_id][frame]
            bonds_broken = min(bonds_broken, 6)
            color = sm.to_rgba(bonds_broken)
            circles[atom_id].set_facecolor(color)

        timestep_value = np.round(ds['timestep'].values[frame] * dt, decimals=2)
        ax.set_title(f'Time = {np.round(timestep_value/3600, decimals = 1)} hrs')
        #ax.set_facecolor('navy')
        return list(circles.values())

    frame_indices = range(0, len(ds['timestep']), frame_skip_value)

    ani = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        blit=False
    )

    os.makedirs(output_directory, exist_ok=True)
    fpath = os.path.join(output_directory, 'bbroken.gif')
    ani.save(fpath, writer='Pillow', fps=5, dpi = 100)


def plot_final_floes(t:int, ds:xr.Dataset, dt:float, output_dir:os.PathLike,
                     labels, component_sizes, number_of_connected_components):
    fig, ax = plt.subplots()
    ds = ds.isel(timestep = t)
    x_min = ds['x'].min().values
    x_max = ds['x'].max().values
    y_min = ds['y'].min().values
    y_max = ds['y'].max().values
    padding = 5e3 

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
    x0 = ds['x'].values
    y0 = ds['y'].values
    radius0 = ds['radius'].values

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

    ax.set_title(f'Floes at Time = {ds['timestep'].values*dt}')
    fig.savefig(os.path.join(output_dir, "final_floes.jpg"), dpi=300, bbox_inches="tight")

def create_coord_num_animation(ds, output_directory, dt, frame_skip_value=4):
    """
    IN:
        ds (netcdf): dataset of atom states containing at least the fields:
            'id'            : atom identifiers
            'x', 'y'        : atom positions (per timestep)
            'radius'        : atom radii (per timestep)
            'coordination'  : coordination number (per timestep)
            'timestep'      : timestep values
        output_directory (string): directory where gif should be saved
        dt (float): conversion factor to scale timestep values into seconds
        frame_skip_value (int): use every frame_skip_value-th timestep in the animation
    OUT:
        Saves a gif of atoms where each atom’s color represents the cumulative number
        of bonds broken (tracked in bbroken_colors) with a colorbar mapping 0 to 5.
        Atoms are outlined with a thin grey border.
    """

    fig, ax = plt.subplots(figsize=(8, 8))

    # Set plot limits with padding.
    x_min = ds['x'].min().values
    x_max = ds['x'].max().values
    y_min = ds['y'].min().values
    y_max = ds['y'].max().values
    padding = 5e3

    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)

    formatter = ticker.FuncFormatter(m_to_km)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # Add labels for the axes indicating kilometers.
    ax.set_xlabel("[km]")
    ax.set_ylabel("[km]")
    ax.set_aspect('equal', 'box')

    # Get atom ids.
    atom_ids = ds['id'].values

    # Initialize a dictionary to hold the Circle patches, keyed by atom id.
    circles = {}

    # set up colormap and normalization (adjust vmin and vmax as needed)
    cmap = cm.jet
    max_coord_num = ds['coordination'].max().max()
    norm = mcolors.Normalize(vmin=0, vmax=max_coord_num)  
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Coordination Number')  

    # Get initial positions and radii for all atoms at timestep 0.
    x0 = ds['x'].isel(timestep=0).values
    y0 = ds['y'].isel(timestep=0).values
    radius0 = ds['radius'].isel(timestep=0).values

    # Create a circle for each atom (with a thin grey border).
    for idx, atom_id in enumerate(atom_ids):
        circle = Circle((x0[idx], y0[idx]), radius0[idx], alpha=0.9,
                        edgecolor='grey', linewidth=0.5)
        ax.add_patch(circle)
        circles[atom_id] = circle

    # Update function for the animation.
    def update(frame):
        x = ds['x'].isel(timestep=frame).values
        y = ds['y'].isel(timestep=frame).values

        for idx, atom_id in enumerate(atom_ids):
            # Update atom position.
            circles[atom_id].center = (x[idx], y[idx])
            # Retrieve the precomputed bonds broken for this atom at the current timestep.

            # Map the bonds broken to a color.
            coord_num = ds['coordination'].loc[atom_id].isel(timestep = frame)
            color = value_to_color(coord_num, norm, cmap)
            circles[atom_id].set_facecolor(color)

        timestep_value = np.round(ds['timestep'].values[frame] * dt, decimals=2)
        ax.set_title(f'Time = {np.round(timestep_value/3600, decimals = 1)} hrs')
        #ax.set_facecolor('navy')
        return list(circles.values())

    frame_indices = range(0, len(ds['timestep']), frame_skip_value)

    ani = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        blit=False
    )

    os.makedirs(output_directory, exist_ok=True)
    fpath = os.path.join(output_directory, 'coordnum.gif')
    ani.save(fpath, writer='Pillow', fps=5, dpi = 100)


def plot_final_floes(t:int, ds:xr.Dataset, dt:float, output_dir:os.PathLike,
                     labels, component_sizes, number_of_connected_components):
    fig, ax = plt.subplots()
    ds = ds.isel(timestep = t)
    x_min = ds['x'].min().values
    x_max = ds['x'].max().values
    y_min = ds['y'].min().values
    y_max = ds['y'].max().values
    padding = 5e3 

    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_aspect('equal', 'box')

    formatter = ticker.FuncFormatter(m_to_km)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # Add labels for the axes indicating kilometers.
    ax.set_xlabel("[km]")
    ax.set_ylabel("[km]")

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
    x0 = ds['x'].values
    y0 = ds['y'].values
    radius0 = ds['radius'].values

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
    time = np.round((ds['timestep'].values*dt)/3600, decimals =1)
    ax.set_title(f'Floes at Time = {time} hrs')
    fig.savefig(os.path.join(output_dir, f"bonds/floes{time}.jpg"), dpi=300)

def get_bond_fsd_from_graph(scipy_sparse_graph):
    number_of_connected_components, labels = connected_components(csgraph=scipy_sparse_graph, directed=False, return_labels=True)
    component_sizes = np.bincount(labels)
    return number_of_connected_components, labels, component_sizes

def plot_fsd(component_sizes, time, output_directory):
    fig = plt.figure()
    hist, bin_edges = np.histogram(component_sizes, bins=np.logspace(0, np.log2(np.max(component_sizes))+0.1, 10, base = 2))
    plt.scatter(bin_edges[1:], hist/bin_edges[1:], s = 40, color = 'g')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'DEs / floe')
    plt.ylabel(r'Floe # Density')
    plt.title(f'FSD at Time = {time} hrs')
    plt.grid()
    outpath = os.path.join(output_directory, f'bonds/fsd{time}.jpg')
    plt.savefig(outpath, dpi = 300)
    plt.close()

def get_velocity_profile(v, y, bin_size = 5e3):
    """
    in: 
        v: (1d array) of velocity component (x or y) at timestep t
        y: (1d array) of atom y coordinates at timestep t

    """
    y_min = np.nanmin(y)
    y_max = np.nanmax(y)
    bins = np.arange(y_min, y_max + bin_size, bin_size)
    bin_indices = np.digitize(y, bins) - 1
    bin_indices[bin_indices == len(bins) - 1] = len(bins) - 2 # fix edge case
    v_means = np.full(len(bins-1), np.nan)

    for i in range(len(bins)-1):
        mask = (bin_indices == i)
        if np.any(mask):
            v_means[i] = np.mean(v[mask])

    #length_along_fjord = (bins[:-1] + bins[1:]) / 2
    return bins, v_means

def plot_velocity_transects(ds_a, dt, timesteps, output_directory):
    """
    in:
        ds_a: (netcdf) atom dataset
        dt: (float) timestep size
        timesteps: (1d np.array) of timesteps
        output_directory: (str) directory to save velocity profiles
    """
    for t in timesteps:
        time = dt*ds_a['timestep'].isel(timestep = t).values
        ux = ds_a['vx'].isel(timestep = t).values
        uy = ds_a['vy'].isel(timestep = t).values
        y = ds_a['y'].isel(timestep = t).values
        len_along_fjord, ux_means = get_velocity_profile(ux, y)
        ___, uy_means = get_velocity_profile(uy, y)
        fig, ax = plt.subplots()
        ax.axvline(0, color = 'gray', ls = '--')
        ax.plot(ux_means[1:-1], len_along_fjord[1:-1]*1e-3, label = r'$V_x$', color = 'k')
        ax.plot(uy_means[1:-1], len_along_fjord[1:-1]*1e-3, label = r'$V_y$', color = 'r')
        ax.set_ylim(-50, 250)
        min_val = np.minimum(np.nanmin(ux_means), np.nanmin(uy_means))
        if min_val != 0:
            ax.set_xlim(min_val , -1*min_val)
        ax.set_ylabel('[km]')
        ax.set_xlabel('[m/s]')
        #ax.set_xlim(-0.1, 0.1)
        ax.legend()
        ax.set_title(f'Time = {np.round(time/3600, decimals = 1)} hrs')
        fig.savefig(os.path.join(output_directory, f'vel_profs/vel_prof{np.round(time)}.jpg'), dpi = 300)
        plt.close()


# add main function to create figures from terminal
if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate figures from LIGGGHTS simulation NetCDF outputs.")
    parser.add_argument("--output-dir", required=True, help="Path to the output directory containing NetCDF files.")
    parser.add_argument("--dt", type=float, default=0.01, help="Timestep for the simulation.")
    args = parser.parse_args()

    # Extract arguments
    output_directory = args.output_dir
    dt = args.dt

    # Open datasets
    fpath_a = os.path.join(output_directory, 'atoms.nc')
    bdir_path = os.path.join(output_directory, 'bonds')
    fpaths_b = sorted(os.path.join(bdir_path, f) for f in os.listdir(bdir_path))
    ds_a = xr.open_dataset(fpath_a)

    # Define/get some arrays
    n_atoms = ds_a.attrs['number_of_atoms'].item()
    num_dumps = 200
    v_plot_freq = 40
    times_to_save = np.linspace(0, num_dumps, v_plot_freq).astype(int)

    
    # plot stuff
    plot_velocity_transects(ds_a, dt, times_to_save, output_directory)
    for fp_b in fpaths_b:
        basename = os.path.basename(fp_b)
        n = int(basename.split('_')[1].split('.')[0])
        graph = load_npz(fp_b)
        number_of_connected_components, labels, component_sizes = get_bond_fsd_from_graph(graph)
        time = dt*ds_a['timestep'].isel(timestep = n).values
        plot_final_floes(n, ds_a, dt, output_directory, labels, component_sizes, number_of_connected_components)
        plot_fsd(component_sizes, time, output_directory)
    create_bond_broken_animation(ds_a, output_directory, dt, frame_skip_value=1)
    #plot_initial_coord_nums(ds_a, output_directory)
    #create_coord_num_animation(ds_a, output_directory, dt, frame_skip_value=1)

    ds_a.close()