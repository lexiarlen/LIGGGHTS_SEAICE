#!/usr/bin/env python3
"""
fix_data.py

Usage:
    python fix_data.py xlo xhi ylo yhi zlo zhi repo

Description:
  Writes data file containing particles on the boundaries and writes a new file: <repo>/<basename>_fixed.data.
"""

import sys
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import xarray as xr

def fix_overlaps(df):
    """
    Adjust diameters so that no two particles overlap.
    df must have columns: ['x', 'y', 'z', 'd']
    """
    coords = df[['x', 'y', 'z']].to_numpy()
    radii = 0.5 * df['d'].to_numpy()
    n = len(coords)
    adjusted_radii_count = 0
    for i in range(n):
        distances = np.sqrt(np.sum((coords[i] - coords)**2, axis=1))
        # Indices of all particles that overlap with particle i
        overlap_indices = np.where((distances < (radii[i] + radii)) & (distances > 0))[0]
        if overlap_indices.size > 0:
            adjusted_radii_count += 1
            max_overlap = 0
            for j in overlap_indices:
                # Overlap amount
                overlap_amount = radii[i] + radii[j] - distances[j]
                max_overlap = np.maximum(overlap_amount, max_overlap)
            # Reduce diameter by the max overlap
            df.loc[i+1, 'd'] -= max_overlap + 1e-10
    return df, adjusted_radii_count

def get_line(xi, xf, yi, yf, d):
    lenx = xf - xi
    leny = yf - yi
    len = np.sqrt(lenx**2 + leny**2)
    n_atoms = int(np.floor(len/d))
    x = np.linspace(xi, xf, n_atoms)
    y = np.linspace(yi, yf, n_atoms)
    return np.array([x, y])

def get_bdy_particles_coords(d):
    wall_numbers = np.arange(1, 11) 

    x_i = np.array([ 0,  40,  40,   0,   0, 120, 120,  80,  80, 120])    # initial x-coordinates
    x_f = np.array([40,  40,   0,   0, 120, 120,  80,  80, 120,   0])    # final x-coordinates
    y_i = np.array([ 0,  80, 120, 160, 280, 280, 160, 120,  80,   0])    # initial y-coordinates
    y_f = np.array([80, 120, 160, 280, 280, 160, 120,  80,   0,   0])    # final y-coordinates

    # Create the dataset
    ds = xr.Dataset(
        {
            "x_i": ("wall", x_i),
            "x_f": ("wall", x_f),
            "y_i": ("wall", y_i),
            "y_f": ("wall", y_f),
        },
        coords={
            "wall": wall_numbers
        },
    )
    wall_dict_bad = {}
    wall_dict = {}
    for w in ds.wall.values:
        wall_dict_bad[w] = get_line(ds.x_i.sel(wall = w), ds.x_f.sel(wall = w), ds.y_i.sel(wall = w), ds.y_f.sel(wall = w), 2)
        wall_dict[w] = np.array([wall_dict_bad[w][0][1:-1], wall_dict_bad[w][1][1:-1]])

    return wall_dict
def main():
        
    # Construct file paths
    output_path = os.path.dirname('/LIGGGHTS_SEAICE/lexi_tests/nares/simulations/create_packing/data')
    fixed_file = os.path.join(output_path, f"bdy_particles.data")

    # (1) Get boundary particles:
    wall_dict = get_bdy_particles_coords()

    # (1.5) Hard code domain size
    pad1 = 5e3
    minx = -10e3-pad1
    maxx = 130e3+pad1
    miny = -10e3-pad1  
    maxy = 400e3+pad1
    # minz = -10e3-pad1
    # maxz = 10e3+pad1

    # read atom data until we hit velocities
    column_names = ['type', 'x', 'y', 'z', 'd', 'density']

    # reformat dataframe so we're ready to read it out
    df["id"] = range(1, len(df) + 1)
    df = df.set_index("id")
    df["bond_type"] = 1

    # Set parameters to install bonds on extra atoms to set bonds/atom
    d = df['d'].iloc[1]               # set diameter of fake atoms to be that of the first atom
    density = df['density'].iloc[1]   # set density of fake atoms to be that of the first atom

    x0 = maxx/2
    y0 = maxy/2
    z0 = 0.0
    z01 = zhi/2

    xtri = d * np.cos(60*np.pi/180) 
    ytri = d * np.sin(60*np.pi/180)


    # (2) Fix Overlaps
    # preliminary diameter adjustment
    df['d'] = df['d'] - bond_skin_thickness

    print(f"[INFO] Mean diameter {df['d'].mean()} m.")
    print("[INFO] Stats after fixing overlaps...")

    df, adjusted_radii_count = fix_overlaps(df)
    num_atoms = len(df)
    print(f"[INFO] Adjusted {adjusted_radii_count} diameters out of {num_atoms} total atoms.")
    print(f"[INFO] New mean diameter {df['d'].mean()} m, min diameter = {df['d'].min()}, and max diameter = {df['d'].max()}.")

    # (3) Build new .data content
    header_info = f"""header line input data

{num_atoms + 14} atoms
1 atom types
12 bonds
1 bond types

{xlo} {xhi} xlo xhi
{ylo} {yhi} ylo yhi
{zlo} {zhi} zlo zhi

Atoms
"""

    # Initialize the list to store each line for the file
    output_lines = [header_info]

    # Generate the atoms section
    for index, row in df.iterrows():
        atom_line = f"{index} 1 {row['x']} {row['y']} {row['z']} {row['d']} {row['density']} 1"
        output_lines.append(atom_line)

    # Pseudo bond info; add 14 extra atoms and install bond between them to implicitly set max bonds/atom
    bond_info = f"""{num_atoms + 1} 1 {x0} {y0} {z0} {d} {density} 1
{num_atoms + 2} 1 {x0 + d} {y0} {z0} {d} {density} 1
{num_atoms + 3} 1 {x0 + xtri} {y0 + ytri} {z0} {d} {density} 1
{num_atoms + 4} 1 {x0 - xtri} {y0 + ytri} {z0} {d} {density} 1
{num_atoms + 5} 1 {x0 - d} {y0} {z0} {d} {density} 1
{num_atoms + 6} 1 {x0 - xtri} {y0 - ytri} {z0} {d} {density} 1
{num_atoms + 7} 1 {x0 + xtri} {y0 - ytri} {z0} {d} {density} 1
{num_atoms + 8} 1 {x0 + d} {y0} {z01} {d} {density} 1
{num_atoms + 9} 1 {x0 + xtri} {y0 + ytri} {z01} {d} {density} 1
{num_atoms + 10} 1 {x0 - xtri} {y0 + ytri} {z01} {d} {density} 1
{num_atoms + 11} 1 {x0 - d} {y0} {z01} {d} {density} 1
{num_atoms + 12} 1 {x0 - xtri} {y0 - ytri} {z01} {d} {density} 1
{num_atoms + 13} 1 {x0 + xtri} {y0 - ytri} {z01} {d} {density} 1
{num_atoms + 14} 1 {x0} {y0} {z01} {d} {density} 1

Bonds

1 1 {num_atoms + 1} {num_atoms + 2}
2 1 {num_atoms + 1} {num_atoms + 3}
3 1 {num_atoms + 1} {num_atoms + 4}
4 1 {num_atoms + 1} {num_atoms + 5}
5 1 {num_atoms + 1} {num_atoms + 6}
6 1 {num_atoms + 1} {num_atoms + 7}
7 1 {num_atoms + 14} {num_atoms + 8}
8 1 {num_atoms + 14} {num_atoms + 9}
9 1 {num_atoms + 14} {num_atoms + 10}
10 1 {num_atoms + 14} {num_atoms + 11}
11 1 {num_atoms + 14} {num_atoms + 12}
12 1 {num_atoms + 14} {num_atoms + 13}

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
