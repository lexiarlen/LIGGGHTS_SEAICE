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

def main():
    """
    Usage: python fix_data.py xlo xhi ylo yhi zlo zhi repo
    """
    if len(sys.argv) != 8:
        print("Usage: python fix_data.py xlo xhi ylo yhi zlo zhi repo")
        sys.exit(1)

    # Unpack arguments
    _, xlo, xhi, ylo, yhi, zlo, zhi, original_fpath = sys.argv

    xlo = float(xlo)
    xhi = float(xhi)
    ylo = float(ylo)
    yhi = float(yhi)
    zlo = float(zlo)
    zhi = float(zhi)

    # Construct file paths
    basename = os.path.splitext(os.path.basename(original_fpath))[0]
    repo = os.path.dirname(original_fpath)
    fixed_file = os.path.join(repo, f"{basename}_fixed.data")

    # (1) Read original .data file

    # read atom data until we hit velocities
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

    # reformat dataframe so we're ready to read it out
    df["id"] = range(1, len(df) + 1)
    df = df.set_index("id")
    df["bond_type"] = 1

    # Set parameters to install bonds on extra atoms to set bonds/atom
    d = df['d'].iloc[1]               # set diameter of fake atoms to be that of the first atom
    density = df['density'].iloc[1]   # set density of fake atoms to be that of the first atom

    x0 = xhi/2
    y0 = yhi/2
    z0 = 0.0
    z01 = zhi/2

    xtri = d * np.cos(60*np.pi/180) 
    ytri = d * np.sin(60*np.pi/180)


    # (2) Fix Overlaps
    print("[INFO] Stats after fixing overlaps...")

    df, adjusted_radii_count = fix_overlaps(df)
    num_atoms = len(df)
    print(f"[INFO] Adjusted {adjusted_radii_count} diameters out of {num_atoms} total atoms.")

    data = df['d'].to_numpy()
    a = np.min(data)
    b = np.max(data)
    # Kolmogorov-Smirnov test for uniform distribution
    ks_stat, ks_pvalue = stats.kstest(data, 'uniform', args=(a, b-a))
    print("[INFO] Kolmogorov-Smirnov statistic:", ks_stat)
    print("[INFO] P-value:", ks_pvalue)

    # Calculate mean and standard deviation for a uniform distribution
    mean_uniform = (a + b) / 2

    # Interpretation of results
    if ks_pvalue > 0.05:
        print(f"[INFO] K-S test gives that the data satisfies a uniform distribution with mean {mean_uniform}. The true mean is {np.mean(data)}.")
    else:
        print(f"[INFO] K-S test gives that the data does NOT satisfy a uniform distribution with mean {mean_uniform}. The true mean is {np.mean(data)}.")

    # fig1 = plt.figure()
    # plt.hist(data, bins=30, alpha=0.75, color='blue', edgecolor='black', density=True)
    # plt.title('Histogram of Data')
    # # Plot the normal distribution with the same mean and std dev
    # plt.show()

    # fig2 = plt.figure()
    # # Q-Q plot for uniform distribution
    # stats.probplot(data, dist="uniform", plot=plt)
    # plt.title('Q-Q plot against a Uniform distribution')
    # plt.show()

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
