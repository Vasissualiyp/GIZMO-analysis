# Libraries {{{
from scipy.interpolate import griddata
from scipy.fftpack import fftn, ifftn
import numpy as np
#}}}

# Non-grid -> Grid {{{
def non_grid_to_grid(data_dict, grid_shape):
    """
    Convert non-gridded data to gridded data.
    """
    # Create an empty dictionary to store gridded data
    gridded_data_dict = {}

    # Generate a grid
    grid_x, grid_y, grid_z = np.mgrid[0:1:complex(grid_shape[0]), 0:1:complex(grid_shape[1]), 0:1:complex(grid_shape[2])]

    # Loop over the data dictionary and grid each dataset
    for key, values in data_dict.items():
        if key == 'Coordinates':
            continue
        gridded_data_dict[key] = griddata(data_dict['Coordinates'], values, (grid_x, grid_y, grid_z), method='linear')

    return gridded_data_dict, (grid_x, grid_y, grid_z)
#}}}

# Non-grid -> Grid (Chunked) {{{
def non_grid_to_grid_chunked(data_dict, grid_shape, chunk_size=100):
    """
    Convert non-gridded data to gridded data, in chunks to avoid memory issues.
    """

    # Extract the coordinates from the data
    coords = data_dict['Coordinates']
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # Normalize the coordinates
    x = (x - min(x)) / (max(x) - min(x))
    y = (y - min(y)) / (max(y) - min(y))
    z = (z - min(z)) / (max(z) - min(z))

    # Create an empty dictionary to store the gridded data
    gridded_data_dict = {}

    # Compute the grid in chunks
    for i in range(0, len(x), chunk_size):
        chunk_coords = np.column_stack((x[i:i+chunk_size], y[i:i+chunk_size], z[i:i+chunk_size]))
        grid_x, grid_y, grid_z = np.mgrid[0:1:complex(grid_shape[0]), 0:1:complex(grid_shape[1]), 0:1:complex(grid_shape[2])]
        grid_coords = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))
        for key in data_dict.keys():
            if key != 'Coordinates':
                gridded_data = griddata(chunk_coords, data_dict[key][i:i+chunk_size], grid_coords, method='linear', fill_value=0)
                if key not in gridded_data_dict:
                    gridded_data_dict[key] = gridded_data
                else:
                    gridded_data_dict[key] += gridded_data

    return gridded_data_dict, (grid_x, grid_y, grid_z)
#}}}

# Non-grid -> Grid (Sparse) {{{
from scipy.sparse import dok_matrix

def non_grid_to_grid_sparse(data_dict, grid_shape):
    """
    Convert non-gridded data to gridded data using a sparse matrix.
    """
    coords = data_dict['Coordinates']
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    x = (x - min(x)) / (max(x) - min(x))
    y = (y - min(y)) / (max(y) - min(y))
    z = (z - min(z)) / (max(z) - min(z))

    x_idx = (x * grid_shape[0]).astype(int)
    y_idx = (y * grid_shape[1]).astype(int)
    z_idx = (z * grid_shape[2]).astype(int)

    gridded_data_dict = {}
    for key in data_dict.keys():
        if key != 'Coordinates':
            sparse_matrix = dok_matrix(grid_shape)
            for xi, yi, zi, di in zip(x_idx, y_idx, z_idx, data_dict[key]):
                sparse_matrix[xi, yi, zi] = di
            gridded_data_dict[key] = sparse_matrix

    grid_coords = np.column_stack((x_idx, y_idx, z_idx))

    return gridded_data_dict, grid_coords
#}}}

# Non-grid -> Grid (Sparse) {{{
def non_grid_to_grid_sparse(data_dict, grid_shape):
    """
    Convert non-gridded data to gridded data using nested dictionaries.
    """
    coords = data_dict['Coordinates']
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    x = (x - min(x)) / (max(x) - min(x))
    y = (y - min(y)) / (max(y) - min(y))
    z = (z - min(z)) / (max(z) - min(z))

    x_idx = (x * grid_shape[0]).astype(int)
    y_idx = (y * grid_shape[1]).astype(int)
    z_idx = (z * grid_shape[2]).astype(int)

    gridded_data_dict = {}
    for key in data_dict.keys():
        if key != 'Coordinates':
            sparse_matrix = {}
            for xi, yi, zi, di in zip(x_idx, y_idx, z_idx, data_dict[key]):
                if xi not in sparse_matrix:
                    sparse_matrix[xi] = {}
                if yi not in sparse_matrix[xi]:
                    sparse_matrix[xi][yi] = {}
                sparse_matrix[xi][yi][zi] = di
            gridded_data_dict[key] = sparse_matrix

    grid_coords = np.column_stack((x_idx, y_idx, z_idx))

    return gridded_data_dict, grid_coords
#}}}

# FFT upscaler part {{{
def upscale_with_fft(gridded_data_dict, n):
    """
    Perform FFT-based interpolation (upscaling).
    """
    # Create an empty dictionary to store upscaled data
    upscaled_data_dict = {}

    # Loop over the gridded data dictionary and perform FFT interpolation on each dataset
    for key, values in gridded_data_dict.items():
        fft_vals = fftn(values)
        upscaled_data_dict[key] = np.real(ifftn(fft_vals, s=[n*i for i in fft_vals.shape]))

    return upscaled_data_dict
#}}}

# Grid -> Grid {{{
def grid_to_non_grid(upscaled_data_dict, new_coords, grid_coords):
    """
    Convert gridded data to non-gridded data.
    """
    # Create an empty dictionary to store non-gridded data
    non_gridded_data_dict = {}

    # Loop over the upscaled data dictionary and convert each gridded dataset back to non-gridded
    for key, values in upscaled_data_dict.items():
        non_gridded_data_dict[key] = griddata(grid_coords, values.flatten(), new_coords, method='linear')

    return non_gridded_data_dict
#}}}

# Helper function to filter out particles outside the original boundary {{{
def filter_particles(upscaled_data_dict, original_data_dict):
    """
    Remove particles from upscaled_data_dict that are located outside the boundary of original_data_dict.
    """
    # Get the boundary of the original data
    boundary = original_data_dict['Coordinates'].max(axis=0)

    # Get the coordinates of the upscaled data
    upscaled_coords = upscaled_data_dict['Coordinates']

    # Create a mask for particles within the boundary
    mask = np.all(upscaled_coords <= boundary, axis=1)

    # Apply the mask to the upscaled data
    for key in upscaled_data_dict.keys():
        upscaled_data_dict[key] = upscaled_data_dict[key][mask]

    return upscaled_data_dict
#}}}

# Main FFT upscaler function {{{
def upscale_with_fft_main(data_dict, n, flags):
    """
    Upscale the data using FFT.
    """
    # Check for 'Coordinates' and other necessary preprocessing
    if 'Coordinates' not in data_dict:
        print("Coordinates not found in the data dictionary.")
        return None

    # Extract the original coordinates
    coords = data_dict['Coordinates']
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # Generate new coordinates based on the increased resolution
    new_x = np.linspace(min(x), max(x), len(x) * n)
    new_y = np.linspace(min(y), max(y), len(y) * n)
    new_z = np.linspace(min(z), max(z), len(z) * n)

    # Initialize a new dictionary to store the interpolated data
    new_data_dict = {}

    # Store the new coordinates in the new dictionary
    new_coords = np.column_stack((new_x, new_y, new_z))
    new_data_dict['Coordinates'] = new_coords

    # Initialize new Particle IDs
    if 'ParticleIDs' in data_dict:
        max_original_id = max(data_dict['ParticleIDs'])
        new_id_counter = max_original_id + 1
        new_particle_ids = []

    # Initialize new ParticleIDGenerationNumber and ParticleChildIDsNumber
    if 'ParticleIDGenerationNumber' in data_dict:
        new_particle_id_gen_nums = []
    if 'ParticleChildIDsNumber' in data_dict:
        new_particle_child_ids_nums = []

    # Step 1: Convert non-gridded data to gridded data
    gridded_data_dict, grid_coords = non_grid_to_grid_sparse(data_dict, (n*len(x), n*len(y), n*len(z)))

    # Step 2: Perform FFT-based interpolation (upscaling)
    upscaled_gridded_data_dict = upscale_with_fft(gridded_data_dict, n)

    # Step 3: Convert the upscaled gridded data back into an upscaled version of the initial non-gridded data
    upscaled_data_dict = grid_to_non_grid(upscaled_gridded_data_dict, new_coords, grid_coords)

    # Step 4: Remove particles that are located outside the boundary of the original data
    upscaled_data_dict = filter_particles(upscaled_data_dict, data_dict)

    # Special handling for ParticleIDs, ParticleIDGenerationNumber, and ParticleChildIDsNumber
    if 'ParticleIDs' in data_dict:
        upscaled_data_dict['ParticleIDs'] = np.array([new_id_counter + i for i in range(len(upscaled_data_dict['Coordinates']))])

    if 'ParticleIDGenerationNumber' in data_dict:
        nearest_gen_nums = NearestNDInterpolator(coords, data_dict['ParticleIDGenerationNumber'])
        upscaled_data_dict['ParticleIDGenerationNumber'] = nearest_gen_nums(new_coords)

    if 'ParticleChildIDsNumber' in data_dict:
        nearest_child_ids_nums = NearestNDInterpolator(coords, data_dict['ParticleChildIDsNumber'])
        upscaled_data_dict['ParticleChildIDsNumber'] = nearest_child_ids_nums(new_coords)

    if 'debugging' in flags:
        print("Upscaling complete.")

    return upscaled_data_dict
#}}}
