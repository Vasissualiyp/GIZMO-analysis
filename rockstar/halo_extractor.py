# Consolidating all the code into a single, copy-pastable block

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

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

# Example usage
output_filename = '/cita/d/www/home/vpustovoit/plots/halos.png'
file_path = '../../rockstar/halos_0.0.ascii'  # Replace with your file path
halo_data_df = read_halo_data(file_path)
plot_2d_projection(halo_data_df, output_filename)
