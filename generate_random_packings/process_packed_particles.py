#!/usr/bin/env python

import pandas as pd
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
    zlo = float(zlo)
    zhi = float(zhi)
    diameter = float(diameter)
    density = float(density)

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

{num_atoms + 2} atoms
1 atom types
1 bonds
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
        atom_line = f"{index + 1} 1 {row['x']} {row['y']} 0.0 {diameter} {density} 1"
        output_lines.append(atom_line)

    # Pseudo bond info; add extra atoms and install bond between them
    bond_info = f"""{num_atoms + 1} 1 {xhi/2} {yhi/2} {zhi/2} {diameter} {density} 1
{num_atoms + 2} 1 {xhi/2} {yhi/2} {zhi/2} {diameter} {density} 1

Bonds

1 1 {num_atoms + 1} {num_atoms + 2}

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
