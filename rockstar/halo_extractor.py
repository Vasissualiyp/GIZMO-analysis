import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Read Halo data {{{
def read_halo_data(file_path):
    """
    Reads a halo data file and returns a Pandas DataFrame.
    
    Parameters:
        file_path (str): The path to the halo data file.
        
    Returns:
        df (DataFrame): Pandas DataFrame containing the halo data.
    """
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
        
    # Remove the '#' and split the string to get the column names
    column_names = first_line[1:].split()
    
    # Read data into a Pandas DataFrame
    df = pd.read_csv(file_path, comment='#', delim_whitespace=True, header=None, names=column_names)
    
    return df
#}}}

# Plot 2D projection {{{
def plot_2d_projection(df, output_filename, scale_factor=10, alpha_value=0.2):
    """
    Plots a 2D projection of halo positions.
    
    Parameters:
        df (DataFrame): Pandas DataFrame containing the halo data.
        scale_factor (float): Factor by which to scale the halo radii for the plot.
        alpha_value (float): Transparency of the dots in the plot.
    """
    # Extract relevant columns for the 2D projection plot
    x_positions = df['x']
    y_positions = df['y']
    radii = df['rvir']
    
    # Create the scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(x_positions, y_positions, s=radii/scale_factor, alpha=alpha_value, edgecolors='none', marker='o')
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('2D Projection of Halo Positions')
    plt.colorbar(label='Radius')
    plt.savefig(output_filename)
    #plt.show()
#}}}

# Get largest halo {{{
def get_largest_halo(df):
    # Find the index of the halo with the maximum mass
    max_mass_idx = df['mvir'].idxmax()
    
    # Extract the details of this halo
    max_mass_halo = df.loc[max_mass_idx]
    
    # Get the position and mass of the halo
    max_mass_halo_position = max_mass_halo[['x', 'y', 'z']]
    max_mass_halo_mass = max_mass_halo['mvir']
    
    return max_mass_halo_position, max_mass_halo_mass
#}}}
    
# Restricting the halo based on the corners of region of interest {{{
def restrict_dataset(df, corner1, corner2):
    """
    Restricts the dataset to halos within a cubic region defined by two opposite corners.
    
    Parameters:
        df (DataFrame): Pandas DataFrame containing the halo data.
        corner1, corner2 (tuple): Coordinates of the two opposite corners of the cubic region.
        
    Returns:
        DataFrame: Restricted dataset containing only halos within the specified cubic region.
    """
    x1, y1, z1 = corner1
    x2, y2, z2 = corner2
    
    # Ensure the corners are ordered correctly (corner1 has smaller coordinates)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    z1, z2 = min(z1, z2), max(z1, z2)
    
    restricted_df = df[(df['x'] >= x1) & (df['x'] <= x2) &
                       (df['y'] >= y1) & (df['y'] <= y2) &
                       (df['z'] >= z1) & (df['z'] <= z2)]
    
    return restricted_df
#}}}

# Example usage
halo_file_path = '../../rockstar/halos_0.0.ascii'  # Replace with your file path
output_filename = '/cita/d/www/home/vpustovoit/plots/halos.png'

halo_data_df = read_halo_data(halo_file_path)

# Example usage: Restrict to a cubic region with corners at (25, 25, 25) and (75, 75, 75)
corner_min = 20
corner_max = 80
corner1 = (corner_min, corner_min, corner_min)
corner2 = (corner_max, corner_max, corner_max)

halo_data_df = restrict_dataset(halo_data_df, corner1, corner2)

plot_2d_projection(halo_data_df, output_filename)

max_mass_halo_position, max_mass_halo_mass = get_largest_halo(halo_data_df)
print("Position of the halo with largest mass:", max_mass_halo_position.to_dict(), " Mpc")
print("Mass of the halo with largest mass:", max_mass_halo_mass/1.e10 , 'x10^10 Msun')
