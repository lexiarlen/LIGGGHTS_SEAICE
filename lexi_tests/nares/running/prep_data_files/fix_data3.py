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
from scipy.spatial import cKDTree

def update_growth_kdtree(df, df_bdy, growth_rate, bdy_radius, tol):
    """
    For each free particle, check (using cKDTree) if increasing its radius
    by 'growth_rate' would cause overlap with any other free particle or boundary.
    Returns a boolean array indicating which particles can grow.
    """
    positions = df[['x', 'y']].to_numpy()
    radii = df['radius'].to_numpy()
    n = len(df)
    can_grow = np.ones(n, dtype=bool)
    
    # build a KDTree for the current positions
    tree = cKDTree(positions)
    
    for i in range(n):
        new_r = radii[i] + growth_rate
        # get neighboring particles within the radius new_r + max(radii).
        indices = tree.query_ball_point(positions[i], new_r + np.max(radii))
        for j in indices:
            if i == j:
                continue
            d_ij = np.linalg.norm(positions[i] - positions[j])
            # check if growth would cause overlap with neighbor j allowing “near touching” by using a small tolerance.
            if d_ij <= (new_r + radii[j] - tol):
                can_grow[i] = False
                break
        # check against boundary particles.
        for _, bdy in df_bdy.iterrows():
            d_bdy = np.linalg.norm(positions[i] - np.array([bdy['x'], bdy['y']]))
            if d_bdy <= (new_r + bdy_radius - tol):
                can_grow[i] = False
                break
    return can_grow

def get_diameters(df, df_bdy, growth_rate = 100, bdy_radius = 1000, tol = 1e-5, min_growth_rate = 0.1):

    # get radius
    df['radius'] = df['d'] / 2.0

    # (1) pre-processing: remove initial overlaps by shrinking by 1/2 * maximum overlap

    # build a KDTree for the free particles
    positions = df[['x', 'y']].to_numpy()
    radii = df['radius'].to_numpy()
    tree = cKDTree(positions)
    delta_required = 0.0
    # search for pairs with overlaps 2x the diameter
    max_possible = np.max(radii) * 2
    overlap_pairs = tree.query_pairs(r=max_possible)

    for i, j in overlap_pairs:
        d_ij = np.linalg.norm(positions[i] - positions[j])
        overlap = (radii[i] + radii[j]) - d_ij
        if overlap > 0:
            delta_required = max(delta_required, overlap / 2)

    if delta_required > 0:
        shrink_amount = delta_required + tol  # add a small extra to be safe
        print(f"Initial overlap detected. Shrinking all radii by {shrink_amount:.4f} m.")
        df['radius'] = df['radius'] - shrink_amount
        df['d'] = 2 * df['radius']
    else:
        print("No initial overlaps detected.")

    # (2) iteratively grow particles until they touch or the growth rate is very small

    while growth_rate > min_growth_rate:
        grow_flags = update_growth_kdtree(df, df_bdy, growth_rate, bdy_radius, tol)
        
        if not np.any(grow_flags):
            # if no particles can safely grow, reduce the growth rate.
            growth_rate /= 2.0
            print(f"Reducing growth rate to {growth_rate:.4f} m")
        else:
            # grow only the particles that can safely grow
            df.loc[grow_flags, 'radius'] += growth_rate
            print(f"Grew {grow_flags.sum()} particles by {growth_rate:.4f} m")
        
        # if no particle can grow and the step is small, break.
        if not np.any(grow_flags) and growth_rate <= min_growth_rate:
            break

    df['d'] = 2 * df['radius']

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

def get_bdy_particles_coords(d, include_top_bot_boundaries=False):
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
    r_km = d_km / 2.0 #+ 0.15    # radius in km

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
        # 3) Get normals
        nx =  dy / length  
        ny = -dx / length  
        # 4) shift by the radius (km) so that bdy particles are outside fjord bdy
        x_vals = x_line - r_km * nx
        y_vals = y_line - r_km * ny
        # Delete overlapping particles at corners; will still have overlap if wall length % diameter != 0
        # This is ok because these boundary particles will not be integrated
        x_vals = x_vals[1:]
        y_vals = y_vals[1:]
        # Accumulate
        all_x.extend(x_vals)
        all_y.extend(y_vals)

    # Fix units
    all_x = np.array(all_x)*1e3
    all_y = np.array(all_y)*1e3
    # Create a DataFrame
    df_bdy = pd.DataFrame({"x": all_x, "y": all_y})
    df_bdy["id"] = range(1, len(df_bdy) + 1)
    df_bdy = df_bdy.set_index("id")
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
        bond_skin_thickness = 0.001*2200
    else:
        print("Usage: python fix_data.py xlo xhi ylo yhi repo bond_skin_thickness")
        sys.exit(1)

    # Convert domain size values to m and floats
    xlo = float(xlo)*1e3
    xhi = float(xhi)*1e3
    ylo = float(ylo)*1e3
    yhi = float(yhi)*1e3

    # Construct file paths
    basename = os.path.splitext(os.path.basename(original_fpath))[0]
    repo = os.path.dirname(original_fpath)
    fixed_file = os.path.join(repo, f"{basename}_fixed_by_dilation.data")

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
    df = pd.read_csv(original_fpath, sep=r'\s+', usecols=[1, 2, 3, 4, 5, 6],
                    skiprows=rows_to_skip, nrows=rows_to_read,
                    names=column_names)

    # Reformat dataframe so we're ready to read it out
    df["id"] = range(1, len(df) + 1)
    df = df.set_index("id")
    df["bond_type"] = 1

    # (2) Set parameters to install bonds on fake atoms to allow successful import
    # These will be deleted upon import so overlap is ok
    d = 2000               # set diameter of fake atoms to be that of the first atom
    density = 920   # set density of fake atoms to be that of the first atom

    x0 = xhi/2
    y0 = yhi/2
    z0 = 0.0

    xtri = d * np.cos(60*np.pi/180) 
    ytri = d * np.sin(60*np.pi/180)

    # (2) Get boundary particles
    mean_d = df['d'].mean() - bond_skin_thickness
    df_bdy = get_bdy_particles_coords(mean_d)
    num_bdy_particles = len(df_bdy['x'])
    print(f'num_bdy_particles = {num_bdy_particles}')

    # (3) Fix Overlaps

    # 3.1 Delete atoms at the very edge that got uplifted
    df = df[df['z'] <= -4000] # careful of hard coding here!!!! number from visual inspection in Ovito
    df.index = range(1, len(df) + 1) # reindex dataframe so that fix overlaps still works

    # 3.2 Run dilation code
    df['d'] -= bond_skin_thickness # preliminary diameter adjustment

    # Note: the new version of fix_overlaps_2d now accepts df_bdy so that interior particles are
    # adjusted to avoid overlaps with both other interior and the boundary particles.
    df = get_diameters(df, df_bdy)
    num_atoms = len(df) + num_bdy_particles
    print(f"[INFO] New mean diameter {df['d'].mean()} m, min diameter = {df['d'].min()}, and max diameter = {df['d'].max()}.")

    # Save a figure of these stats
    mean_val = df['d'].mean()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df['d'], bins=16, color='blue', edgecolor='black', alpha=0.7)
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
{-mean_d} {mean_d} zlo zhi

Atoms
"""

    # Initialize the list to store each line for the file
    output_lines = [header_info]

    # Add the boundary particles to the atoms section
    for index, row in df_bdy.iterrows():
        # format: id atom_type x y z d rho bond_type?
        atom_line = f"{index} 2 {row['x']} {row['y']} 0 {mean_d} 920 1" # hard coded density; be careful!!
        output_lines.append(atom_line)

    # Generate the atoms section for the interior particles
    for index, row in df.iterrows():
        # format: id atom_type x y z d rho bond_type?
        atom_line = f"{index+num_bdy_particles} 1 {row['x']} {row['y']} 0 {row['d']} {row['density']} 1"
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
