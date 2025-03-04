#!/usr/bin/env python3
"""
fix_data.py

Usage:
    python fix_data.py xlo xhi ylo yhi zlo zhi repo

Description:
  Reads the input `.data` file from <repo>/<basename>.data,
  fixes overlaps by adjusting particle diameters, updates domain
  boundaries, and writes a new file: <repo>/<basename>_fixed.data.
"""

import sys
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import xarray as xr


def fix_overlaps_2d(df, repo, max_iters = 2000, ini_d_adj = 0.01, lr = 0.8):
    """
    Adjust diameters so that no two particles overlap. Before doing this, shrink the 
    particle diameter by the bond_skin_thickness. By design, in the packing process,
    particles are created too large by the bond_skin_thickness. 

    df must have columns: ['x', 'y', 'd', 'type']
    """
    # TODO: modify to make only adjust diameters for type 2 partilce
    # initial parameters
    coords = df[['x', 'y']].to_numpy()
    mean_d = df['d'].mean()
    particle_types = df['type'].to_numpy()
    diam_adj = mean_d*ini_d_adj
    overlaps_remaining = []

    # dynamic growth rate stuff
    gradient_flag=False
    min_adj = 1

    n = len(coords)
    for iteration in range(max_iters):
        if iteration >= 2 and gradient_flag == False:
            x = np.arange(iteration)
            doverlaps_dx = np.gradient(overlaps_remaining)
            if np.any(doverlaps_dx < 0):
                gradient_flag=True
        if gradient_flag:
            diam_adj *= lr
            diam_adj = np.maximum(diam_adj, min_adj)
        n_overlaps_this_pass = 0
        radii = 0.5 * df['d'].to_numpy()
        for i in range(n):
            distances = np.sqrt(np.sum((coords[i] - coords)**2, axis=1))
            # Indices of all particles that overlap with particle i
            overlap_indices = np.where((distances < (radii[i] + radii)) & (distances > 0))[0]
            if (overlap_indices.size > 0) and (particle_types[i] == 1): # if overlap and interior particle
                n_overlaps_this_pass += 1
                df.loc[i+1, 'd'] -= diam_adj + 1e-5 # adjust diameter at index label i+1 = coords[i]
        overlaps_remaining.append(n_overlaps_this_pass)
        if n_overlaps_this_pass == 0:
            print(f'[INFO] Converged at iteration # = {iteration}')
            break
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(np.arange(1, len(overlaps_remaining)+1), overlaps_remaining, color = 'b')
    ax.grid()
    ax.set_xlabel('Iteration #')
    ax.set_ylabel('# Fixed Overlaps')
    plt.show()
    plt.savefig(os.path.join(repo, f'overlaps_newfunc_{ini_d_adj}.jpg'), dpi=300, bbox_inches='tight')
    return df

def get_line(xi, xf, yi, yf, d):
    """
    Computes x and y coordinates of particles along a line from (x_i, y_i) to 
    (x_f, y_f) for particles of diameter size d.
    """
    lenx = xf - xi
    leny = yf - yi
    len = np.sqrt(lenx**2 + leny**2)
    n_atoms = int(np.floor(len/d))
    x = np.linspace(xi, xf, n_atoms)
    y = np.linspace(yi, yf, n_atoms)
    return np.array([x, y])

def get_bdy_particles_coords(d, density, include_top_bot_boundaries=False):
    """
    Returns a dataframe containing the coordinates of the boundary particles of
    the Danseraeu et al. (2017) fjord geometry. Wall #1 is bottom left wall and #s
    move clockwise around the geometry. Delete walls 5 and 10 for an open fjord.
    IN: d = diameter of boundary particles (m). If polydisperse, use the mean diameter.
    IN: include_top_bot_boundaries = boolean (opt) include top and bottom boundaries of fjord.
    OUT: pd.DataFrame with columns ['x', 'y'] containing particle coordinates.
    """
    # Convert diameter from m to km
    d_km = d / 1000.0
    r_km = d_km / 2.0 + 0.1 # radius in km

    # Define wall numbers
    wall_numbers = np.arange(1, 11)  # 1..10

    # Initial and final coordinates for each wall (in km)
    x_i = np.array([  0,  40,  40,   0,   0, 120, 120,  80,  80, 120], dtype=float)
    x_f = np.array([ 40,  40,   0,   0, 120, 120,  80,  80, 120,   0], dtype=float)
    y_i = np.array([  0,  80, 120, 160, 245, 247, 160, 120,  80,   0], dtype=float)
    y_f = np.array([ 80, 120, 160, 245, 245, 160, 120,  80,   0,   0], dtype=float)

    # Create an xarray dataset
    ds = xr.Dataset(
        {
            "x_i": ("wall", x_i),
            "x_f": ("wall", x_f),
            "y_i": ("wall", y_i),
            "y_f": ("wall", y_f),
        },
        coords={"wall": wall_numbers},
    )

    # Optionally exclude top and bottom walls
    if not include_top_bot_boundaries:
        ds = ds.sel(wall=ds.wall[np.isin(ds.wall, [5, 10], invert=True)])

    # Collect all x,y in lists
    all_x = []
    all_y = []
    for w in ds.wall.values:
        xi = ds.x_i.sel(wall=w).item()
        xf = ds.x_f.sel(wall=w).item()
        yi = ds.y_i.sel(wall=w).item()
        yf = ds.y_f.sel(wall=w).item()

        # 1) get the "center line" coordinates spaced by 'd_km'
        x_line, y_line = get_line(xi, xf, yi, yf, d_km)

        # 2) compute the direction vector and length
        dx = xf - xi
        dy = yf - yi
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            # check if wall length is degenerate
            continue

        # 3) shift by the radius (km) so that bdy particles are outside fjord bdy
        nx =  dy / length  
        ny = -dx / length 
        x_vals = x_line - r_km * nx
        y_vals = y_line - r_km * ny
        x_vals = x_vals[1:]
        y_vals = y_vals[1:]

        all_x.extend(x_vals)
        all_y.extend(y_vals)

    # fix units
    all_x = np.array(all_x)*1e3
    all_y = np.array(all_y)*1e3

    # create a dataframe
    df_bdy = pd.DataFrame({"x": all_x, "y": all_y})
    df_bdy["id"] = range(1, len(df_bdy) + 1)
    df_bdy = df_bdy.set_index("id")
    df_bdy["type"] = 2*np.ones_like(df_bdy["x"])
    df_bdy["density"] = density*np.ones_like(df_bdy["x"])
    df_bdy["d"] = d*np.ones_like(df_bdy["x"])
    df_bdy["z"] = np.zeros_like(df_bdy["x"])
    return df_bdy

def main():
    """
    Usage: python fix_data.py xlo xhi ylo yhi repo bond_skin_thickness
    """
    if len(sys.argv) == 7:
        _, xlo, xhi, ylo, yhi, original_fpath, bond_skin_thickness = sys.argv
        bond_skin_thickness = float(bond_skin_thickness)
    elif len(sys.argv) == 6:
        _, xlo, xhi, ylo, yhi, original_fpath = sys.argv
        bond_skin_thickness = 0.001*1500
    else:
        print("Usage: python fix_data.py xlo xhi ylo yhi repo bond_skin_thickness")
        sys.exit(1)

    # Convert domain size values to m and floats
    xlo = float(xlo)*1e3
    xhi = float(xhi)*1e3
    ylo = float(ylo)*1e3
    yhi = float(yhi)*1e3

    # HARD CODING BE CAREFUL
    d = 1000
    density = 920

    # Construct file paths
    basename = os.path.splitext(os.path.basename(original_fpath))[0]
    repo = os.path.dirname(original_fpath)
    fixed_file = os.path.join(repo, f"{basename}_fixed4.data")

    # (1) Read original .data file

    # Read atom data until we hit velocities
    column_names = ['type', 'x', 'y', 'z', 'd', 'density']
    rows_to_skip = 21
    cutoff_line = None
    with open(original_fpath, 'r') as f:
        for i, line in enumerate(f):
            if "Velocities" in line:
                cutoff_line = i - 1
                break
    rows_to_read = cutoff_line - rows_to_skip if cutoff_line is not None else None
    df_int = pd.read_csv(original_fpath, sep=r'\s+', usecols=[1, 2, 3, 4, 5, 6],
                    skiprows=rows_to_skip, nrows=rows_to_read,
                    names=column_names)

    # Reformat dataframe so we're ready to read it out
    df_int["id"] = range(1, len(df_int) + 1)
    df_int = df_int.set_index("id")

    # (2) Set parameters to install bonds on fake atoms to allow successful import
    x0 = xhi/2
    y0 = yhi/2
    z0 = 0.0

    xtri = d * np.cos(60*np.pi/180) 
    ytri = d * np.sin(60*np.pi/180)

    # (2) Get boundary particles
    df_bdy = get_bdy_particles_coords(d, density)

    # (3) Fix Overlaps

    # 3.1 Delete atoms at the very edge that got uplifted
    df_int = df_int[df_int['z'] <= -4000] # careful of hard coding here!!!! number from visual inspection in Ovito
    df_int['z'] = np.zeros_like(df_int['z'])
    #df_int.index = range(1, len(df_int) + 1) # reindex dataframe so that fix overlaps still works

    # 3.2 Run overlap code
    #df_int['d'] -= bond_skin_thickness # preliminary diameter adjustment

    # concatenate interior and boundary particle dataframes
    df = pd.concat([df_bdy, df_int], ignore_index=True)
    df.index=range(1,len(df)+1)
    df = fix_overlaps_2d(df, repo)
    num_atoms = len(df) 

    # Save a figure of these stats
    mean_val = df['d'].mean()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df['d'], bins=20, color='blue', edgecolor='black', alpha=0.7)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
            label=f'Mean: {mean_val:.2f}')
    ax.set_xlabel('d')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Diameters')
    ax.legend()
    plt.savefig(os.path.join(repo, 'histogram.jpg'), dpi=300, bbox_inches='tight')

    # (3) Build new .data content
    header_info = f"""header line input data

{num_atoms + 7} atoms
2 atom types
6 bonds
1 bond types

{xlo} {xhi} xlo xhi
{ylo} {yhi} ylo yhi
{-d} {d} zlo zhi

Atoms
"""

    # Initialize the list to store each line for the file
    output_lines = [header_info]

    # Generate the atoms section for the interior particles
    for index, row in df.iterrows():
        # format: id atom_type x y z d rho bond_type?
        atom_line = f"{index} {row['type']} {row['x']} {row['y']} 0 {row['d']} {row['density']} 1"
        output_lines.append(atom_line)

    # Pseudo bond info; add 7 extra atoms and install bond between them to implicitly set max bonds/atom
    bond_info = f"""{num_atoms + 1} 1 {x0} {y0} {z0} {d} {density} 1
{num_atoms + 2} 1 {x0 + d} {y0} {z0} {d} {density} 1
{num_atoms + 3} 1 {x0 + xtri} {y0 + ytri} {z0} {d} {density} 1
{num_atoms + 4} 1 {x0 - xtri} {y0 + ytri} {z0} {d} {density} 1
{num_atoms + 5} 1 {x0 - d} {y0} {z0} {d} {density} 1
{num_atoms + 6} 1 {x0 - xtri} {y0 - ytri} {z0} {d} {density} 1
{num_atoms + 7} 1 {x0 + xtri} {y0 - ytri} {z0} {d} {density} 1

Bonds

1 1 {num_atoms + 1} {num_atoms + 2}
2 1 {num_atoms + 1} {num_atoms + 3}
3 1 {num_atoms + 1} {num_atoms + 4}
4 1 {num_atoms + 1} {num_atoms + 5}
5 1 {num_atoms + 1} {num_atoms + 6}
6 1 {num_atoms + 1} {num_atoms + 7}

"""

    output_lines.append(bond_info)

    # Combine all lines into a single string
    output_content = "\n".join(output_lines)

    # Write out to the fixed file
    with open(fixed_file, 'w') as f:
        f.writelines(output_content)

    print(f"[INFO] New fixed data file created: {fixed_file}")

if __name__ == "__main__":
    main()
