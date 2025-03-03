#!/usr/bin/env python3
"""
fix_data.py

Usage:
    python fix_data.py xlo xhi ylo yhi d out_path

Description:
  Create a data file containing the boundary particles. 
"""

import sys
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import xarray as xr


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

def get_bdy_particles_coords(d, density, include_top_boundaries=False):
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
    r_km = d_km / 2.0  # radius in km

    # Define wall numbers
    wall_numbers = np.arange(1, 11)  # 1..10

    # Initial and final coordinates for each wall (in km)
    x_i = np.array([  0,  40,  40,   0,   0, 120, 120,  80,  80, 120], dtype=float)
    x_f = np.array([ 40,  40,   0,   0, 120, 120,  80,  80, 120,   0], dtype=float)
    y_i = np.array([  0,  80, 120, 160, 275, 275, 160, 120,  80,   0], dtype=float)
    y_f = np.array([ 80, 120, 160, 275, 275, 160, 120,  80,   0,   0], dtype=float)

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

    # Optionally exclude top wall
    if not include_top_boundaries:
        ds = ds.sel(wall=ds.wall[np.isin(ds.wall, [5], invert=True)]) # wall 5 is top wall

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
    if len(sys.argv) == 8:
        _, xlo, xhi, ylo, yhi, d, density, out_path = sys.argv
        d = float(d)
        density = float(density)
    else:
        print("Usage: python fix_data.py xlo xhi ylo yhi d density out_path")
        sys.exit(1)

    # Convert domain size values to m and floats
    xlo = float(xlo)*1e3
    xhi = float(xhi)*1e3
    ylo = float(ylo)*1e3
    yhi = float(yhi)*1e3

    # Construct file paths
    new_file = os.path.join(out_path, f"boundary_particles_{int(d)}m.data")

    # (1) Set parameters to install bonds on fake atoms to allow successful import
    x0 = xhi/2
    y0 = yhi/2
    z0 = 0.0

    xtri = d * np.cos(60*np.pi/180) 
    ytri = d * np.sin(60*np.pi/180)

    # (2) Get boundary particles
    df = get_bdy_particles_coords(d, density)
    num_atoms = len(df)

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
    with open(new_file, 'w') as f:
        f.writelines(output_content)

    print(f"[INFO] New fixed data file created: {new_file}")

if __name__ == "__main__":
    main()
