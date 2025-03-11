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

def main():
    """
    Usage: python fix_data.py xlo xhi ylo yhi fpath diameter density
    Note that water level will be assumed to be diameter/2 and 
    water density will be 1027
    """
    if len(sys.argv) == 8:
        _, xlo, xhi, ylo, yhi, original_fpath, d, density = sys.argv
        d = float(d)
        density = float(density)
    else:
        print("Usage: python fix_data.py xlo xhi ylo yhi fpath diameter density")
        sys.exit(1)

    # Convert domain size values to m and floats
    xlo = float(xlo)*1e3
    xhi = float(xhi)*1e3
    ylo = float(ylo)*1e3
    yhi = float(yhi)*1e3

    # Construct file paths
    basename = os.path.splitext(os.path.basename(original_fpath))[0]
    repo = os.path.dirname(original_fpath)
    fixed_file = os.path.join(repo, f"{basename}_fixed_3d.data")

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

    # (2) Set parameters to install bonds on fake atoms to allow successful import
    x0 = xhi/2
    y0 = yhi/2
    z0 = 0.0

    xtri = d * np.cos(60*np.pi/180) 
    ytri = d * np.sin(60*np.pi/180)

    # (3) Fix Overlaps

    
    # shrink diameters
    df['d'] -= df['d']*0.15

    # Buoyancy correction
    water_level = d/2
    seawater_density=1027
    r = df['d']/2
    h = r * (1 + 2 * np.cos(1/3 * np.arccos(1 - 2*density/seawater_density) - 2*np.pi/3))
    z_coords = water_level - h + r
    df['z'] = z_coords


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
2 bond types

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
        atom_line = f"{index} {row['type']} {row['x']} {row['y']} {row['z']} {row['d']} {row['density']} 1"
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
