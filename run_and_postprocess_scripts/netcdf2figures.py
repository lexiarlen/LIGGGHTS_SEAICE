import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import xarray as xr
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import netCDF4 as nc
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import random

# inputs
base_path = r'c:\Users\arlenlex\Documents\liggghts_data\dynamic_forcing\crystallized_floe\cyclone'
dt = 0.001      
output_directory = 'test_output_data'
time_average_data = True
coordnum_gif = True
final_floes = True
fsd = True

# open datasets
fpath_a = os.path.join(base_path, 'atoms.nc')
fpath_b = os.path.join(base_path, 'bonds.nc')
ds_a = xr.open_dataset(fpath_a)
ds_b = nc.Dataset(fpath_b, 'r')

# define/get some arrays
n_atoms = ds_a.attrs['number_of_atoms'].item()

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

    # Create the animation
    ani = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        blit=False
    )

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    fpath = os.path.join(output_directory, 'coordnum.gif')

    # Save the animation as a GIF
    ani.save(fpath, writer='Pillow', fps=2)


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


# # get bond data
# if final_floes or time_average_data or coordnum_gif or fsd: # want to run if any is true
#     nbonds, bond_force, final_graph, coordination_numbers_df = process_bond_file(ds_b, n_atoms)

# # can now close bond dataset 
# ds_b.close()

# # make atom gif
# if coordnum_gif: 
#     create_atom_positions_animation(ds_a, output_directory, coordination_numbers_df, frame_skip_value=5) 

# if time_average_data:
#     time = ds_a['timestep'].values*dt

#     # compute quantities
#     ds_a['fmag'] = np.sqrt(ds_a['fx']**2 + ds_a['fy']**2 + ds_a['fz']**2)
#     ds_a['vmag'] = np.sqrt(ds_a['vx']**2 + ds_a['vy']**2 + ds_a['vz']**2)

#     avg_sxy = ds_a['c_stress[4]'].mean(dim='id')
#     avg_vmag = ds_a['vmag'].mean(dim='id')
#     avg_fmag = ds_a['fmag'].mean(dim='id')

#     # make array of time averaged quantities to loop plots
#     time_avg_quantities = [avg_sxy, avg_fmag, avg_vmag, nbonds, bond_force]
#     units = ['[Pa]', '[N]', '[m/s]', '[#]', '[N]']
#     titles = [r'$\overline{\sigma_{xy}}$', r'$|\overline{\textbf{F}_a}|$', r'$|\overline{\textbf{v}}|$', '# bonds', r'$|\overline{\textbf{F}_b}|$']
#     output_names = ['shear_stress.png', 'a_fmag.png', 'vmag.png', 'nbonds.png', 'b_fmag.png']
#     colors = ['r', 'indigo', 'b', 'tab:orange', 'hotpink']

#     for i in range(5):
#         plot_and_save_quantity(time, time_avg_quantities[i], units[i], titles[i], colors[i], output_directory, output_names[i])

# # now close atom dataset as we will no longer use
# ds_a.close()

# if fsd:
#     number_of_connected_components, labels, component_sizes = get_bond_fsd_from_graph(final_graph)
#     plot_fsd(component_sizes, output_directory)


# add main function to create figures based on input from bash file
if __name__ == '__main__':
    # Interactive inputs
    base_path = input(
        "Enter base path (default: 'c:\\Users\\arlenlex\\Documents\\liggghts_data\\dynamic_forcing\\crystallized_floe\\cyclone'): "
    )
    if not base_path:
        base_path = r'c:\Users\arlenlex\Documents\liggghts_data\dynamic_forcing\crystallized_floe\cyclone'

    dt_input = input("Enter dt (default: 0.001): ")
    dt = float(dt_input) if dt_input else 0.001

    output_directory = input("Enter output directory (default: 'test_output_data'): ")
    if not output_directory:
        output_directory = 'test_output_data'

    time_average_data_input = input("Time average data? (True/False, default: True): ")
    time_average_data = time_average_data_input.lower() == 'true' if time_average_data_input else True

    coordnum_gif_input = input("Create coordination number GIF? (True/False, default: True): ")
    coordnum_gif = coordnum_gif_input.lower() == 'true' if coordnum_gif_input else True

    final_floes_input = input("Compute final floes? (True/False, default: True): ")
    final_floes = final_floes_input.lower() == 'true' if final_floes_input else True

    fsd_input = input("Compute FSD? (True/False, default: True): ")
    fsd = fsd_input.lower() == 'true' if fsd_input else True

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

    # Close bond dataset
    ds_b.close()

    # Create coordination number animation if selected
    if coordnum_gif:
        create_atom_positions_animation(ds_a, output_directory, coordination_numbers_df, frame_skip_value=5)

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
        titles = [r'$\overline{\sigma_{xy}}$', r'$|\overline{\textbf{F}_a}|$', r'$|\overline{\textbf{v}}|$',
                  '# bonds', r'$|\overline{\textbf{F}_b}|$']
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
        number_of_connected_components, labels, component_sizes = get_bond_fsd_from_graph(final_graph)
        plot_fsd(component_sizes, output_directory)