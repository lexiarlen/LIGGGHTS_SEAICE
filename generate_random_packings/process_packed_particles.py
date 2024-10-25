#!/usr/bin/env python

import pandas as pd
import numpy as np
import sys
import os

def main():
    # Get arguments
    if len(sys.argv) != 10:
        print("Usage: process_packed_particles.py xlo xhi ylo yhi zlo zhi diameter density output_dir")
        sys.exit(1)

    _, xlo, xhi, ylo, yhi, zlo, zhi, diameter, density, output_dir = sys.argv

    xlo = float(xlo)
    xhi = float(xhi)
    ylo = float(ylo)
    yhi = float(yhi)
    zlo = -2.0
    zhi = 2.0
    d = float(diameter)
    density = float(density)

    # Set parameters to install bonds on extra atoms to set bonds/atom
    x0 = xhi/2
    y0 = yhi/2
    z0 = 0.0

    xtri = d * np.cos(60)
    ytri = d * np.sin(60)

    # Read packed_particles.xyz
    fname = os.path.join(output_dir, 'packed_particles.xyz')

    # Read xyz file into dataframe
    with open(fname, 'r') as f:
        lines = f.readlines()

    num_atoms = int(lines[0].strip())
    # Skip first two lines
    data_lines = lines[2:]

    # Read data into dataframe
    df = pd.DataFrame([line.strip().split() for line in data_lines], columns=['Element', 'x', 'y', 'z'])
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    df['z'] = df['z'].astype(float)

    # Build the header information
    header_info = f"""header line input data

{num_atoms + 7} atoms
1 atom types
6 bonds
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
        atom_line = f"{index + 1} 1 {row['x']} {row['y']} 0.0 {d} {density} 1"
        output_lines.append(atom_line)

    # Pseudo bond info; add 6 extra atoms and install bond between them to implicitly set max bonds/atom
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

    output_fname = os.path.join(output_dir, 'rcp.data')

    # Write the content to a text file
    with open(output_fname, 'w') as file:
        file.write(output_content)

    print(f"File {output_fname} has been created successfully.")

if __name__ == "__main__":
    main()
