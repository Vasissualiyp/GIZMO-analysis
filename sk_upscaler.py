from scipy.interpolate import RegularGridInterpolator as rgi
import numpy.fft as fft
import numpy as np
import tricubic

from scipy.stats import multivariate_normal

# Gaussian kernel {{{
def gaussian_kernel(grid, position, h, flags):
    """
    Computes the Gaussian kernel for a set of points.

    Parameters:
    grid (tuple): A tuple containing the x, y, z coordinates of the grid points.
    position (array-like): The position of the particle for which to compute the kernel.
    h (float): The smoothing length.

    Returns:
    kernel (array-like): The computed Gaussian kernel.
    """
    x, y, z = grid
    grid_points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    rv = multivariate_normal(mean=position, cov=[[h**2, 0, 0], [0, h**2, 0], [0, 0, h**2]])
    return rv.pdf(grid_points).reshape(x.shape)
#}}}

# Create the density field {{{
def create_density_field(particle_positions, smoothing_lengths, box_size, resolution, flags):
    """
    Create a density field from a set of particles.

    Parameters:
    particle_positions (array-like): Nx3 array of particle positions.
    smoothing_lengths (array-like): Nx1 array of smoothing lengths for each particle.
    box_size (array-like): Dimensions of the box that contains the particles.
    resolution (int): The resolution of the grid.

    Returns:
    field (array-like): The density field.
    x_grid, y_grid, z_grid (array-like): The coordinates of the grid points.
    """
    # Calculate the number of grid points in each dimension
    field_shape = tuple((np.array(box_size) * resolution).astype(int))

    # Initialize an empty field
    field = np.zeros(field_shape)

    # Create the grid
    x_grid, y_grid, z_grid = np.mgrid[0:box_size[0]:field_shape[0]*1j, 0:box_size[1]:field_shape[1]*1j, 0:box_size[2]:field_shape[2]*1j]

    # Create a 3D Gaussian kernel for each particle and add it to the field
    for position, h in zip(particle_positions, smoothing_lengths):
        kernel = gaussian_kernel((x_grid, y_grid, z_grid), position, h,flags)
        field += kernel

    return field, (x_grid, y_grid, z_grid)
#}}}

# Upscale field {{{
def upscale_field(field, x_grid, y_grid, z_grid, upscale_factor, flags):
    """
    Upscale a 3D field using FFTs.

    Parameters:
    field (3D numpy array): The input field to be upscaled.
    upscale_factor (int): The factor by which to upscale the field.

    Returns:
    upscaled_field (3D numpy array): The upscaled field.
    """
    # Perform the 3D FFT
    spectrum = fft.fftn(field)

    # Calculate the shape of the upscaled field
    upscale_shape = tuple(np.array(field.shape) * upscale_factor)

    # Create an array to hold the upscaled spectrum
    upscaled_spectrum = np.zeros(upscale_shape, dtype=complex)

    # The frequency bins are symmetric, with the low frequencies in the middle, so we need to
    # place the original spectrum in the middle of the larger array
    slices_original = [slice((upscale_shape[i] - field.shape[i]) // 2, (upscale_shape[i] + field.shape[i]) // 2) for i in range(3)]
    slices_upscaled = [slice(None)] * 3

    # Place the original spectrum at the center of the upscaled spectrum
    upscaled_spectrum[tuple(slices_original)] = spectrum

    # Perform the inverse FFT to get the upscaled field
    upscaled_field = np.real(fft.ifftn(upscaled_spectrum))

    # Create grids for the upscaled field
    x_upscaled = np.linspace(0, np.max(x_grid), upscale_shape[0])
    y_upscaled = np.linspace(0, np.max(y_grid), upscale_shape[1])
    z_upscaled = np.linspace(0, np.max(z_grid), upscale_shape[2])

    return upscaled_field, (x_upscaled, y_upscaled, z_upscaled)
#}}}

# Generate particle positions {{{
def generate_particle_positions(upscaled_field, x_upscaled, y_upscaled, z_upscaled, n_particles, flags):
    """
    Generate a set of particle positions based on an upscaled density field.

    Parameters:
    upscaled_field (3D numpy array): The upscaled density field.
    x_upscaled, y_upscaled, z_upscaled (1D numpy arrays): The grid points of the upscaled field.
    n_particles (int): The number of particles to generate.

    Returns:
    particle_positions (2D numpy array): The generated particle positions.
    """

    if 'debugging' in flags:
        print('Interpolator started')
    # Create an interpolator for the upscaled field
    interpolator = rgi((x_upscaled, y_upscaled, z_upscaled), upscaled_field)
    if 'debugging' in flags:
        print('Interpolator ended')

    # Initialize an empty array to hold the particle positions
    particle_positions = np.zeros((n_particles, 3))

    if 'debugging' in flags:
        print('Starting particle generation...')
        print()
    # Generate particles
    for i in range(n_particles):
        while True:
            if 'debugging' in flags:
                print(f'\rParticle number: {i} out of {n_particles}    ', end='', flush=True)

            # Generate a random position within the box
            pos = np.array([np.random.uniform(0, max(x_upscaled)), np.random.uniform(0, max(y_upscaled)), np.random.uniform(0, max(z_upscaled))])

            # Sample the density field at this position
            density = interpolator(pos)

            # Accept or reject the position based on the density
            if np.random.uniform(0, np.max(upscaled_field)) < density:
                particle_positions[i] = pos
                break

    return particle_positions
#}}}

# Vectorized particle positions generator {{{
def generate_particles_vectorized(upscaled_field, x_upscaled, y_upscaled, z_upscaled, n_particles, flags):
    """
    Generates particles within the box using the upscaled field.

    Parameters:
    upscaled_field (array-like): The upscaled field to use for generating particles.
    x_upscaled (array-like): The x coordinates of the upscaled field.
    y_upscaled (array-like): The y coordinates of the upscaled field.
    z_upscaled (array-like): The z coordinates of the upscaled field.
    n_particles (int): The number of particles to generate.

    Returns:
    particle_positions (array-like): The positions of the generated particles.
    """

    # Create a tricubic interpolator for the upscaled field.
    #interpolator = tricubic.tricubic(list(zip(x_upscaled.ravel(), y_upscaled.ravel(), z_upscaled.ravel())), upscaled_field.ravel())
    interpolator = rgi((x_upscaled, y_upscaled, z_upscaled), upscaled_field)

    if 'debugging' in flags:
        print('Interpolator finished')
        print(type(upscaled_field.ravel()))
        print(upscaled_field.ravel().shape)


    if 'debugging' in flags:
        print('Starting to generate the positions within the box... ')
    # Generate all positions at once within the box
    pos_all = np.random.uniform(0, np.max([x_upscaled, y_upscaled, z_upscaled]), size=(n_particles, 3))
    if 'debugging' in flags:
        print('Positions generated! ')

    # Sample the density field at all positions
    density_all = interpolator(pos_all)

    # Accept or reject all positions based on the density
    accepted_mask = np.random.uniform(0, np.max(upscaled_field), size=n_particles) < density_all

    # Keep only accepted positions
    particle_positions = pos_all[accepted_mask]

    if 'debugging' in flags:
        print('Repeat the process until you have enough accepted particles')
    # If there are not enough accepted particles, repeat the process
    while len(particle_positions) < n_particles:
        l = len(particle_positions)
        if 'debugging' in flags:
            print(f'\rParticle number: {l} out of {n_particles}    ', end='', flush=True)

        # Calculate how many more particles are needed
        n_more = n_particles - len(particle_positions)

        # Generate more positions
        pos_more = np.random.uniform(0, np.max([x_upscaled, y_upscaled, z_upscaled]), size=(n_more, 3))

        # Sample the density field at these positions
        density_more = interpolator(pos_more)

        # Accept or reject these positions
        accepted_mask_more = np.random.uniform(0, np.max(upscaled_field), size=n_more) < density_more

        # Add accepted positions to the particle positions
        particle_positions = np.concatenate([particle_positions, pos_more[accepted_mask_more]])

    return particle_positions
#}}}

# Interpolators {{{
def interpolate_data(ds, upscale_factor, box_size, x_upscale, y_upscale, z_upscale, upscaled_field, reshaped_field, new_particle_positions):
    data_type_lists = {
        'interpolation_upscaling': ['Density', 'ElectronAbundance', 'InternalEnergy', 'Potential', 'StarFormationRate'],
        'normalized_upscaling': ['Masses', 'SmoothingLength'],
        'name_additional_particles': ['ParticleIDs', 'ParticleIDGenerationNumber', 'ParticleChildIDsNumber'],
        'vector_interpolation': ['Velocities', 'Metallicity'],
    }

    upscaled_data_dict = {}

    for variable_name in ds.keys():
        for upscale_type, variable_list in data_type_lists.items():
            if variable_name in variable_list:
                if 'debugging' in flags:
                    print(f'Working with {variable_name}...')
                if upscale_type == 'interpolation_upscaling':
                    upscaled_data_dict[variable_name] = interpolation_upscaling(ds[variable_name], x_upscale, y_upscale, z_upscale, upscaled_field, reshaped_field, new_particle_positions)
                elif upscale_type == 'normalized_upscaling':
                    upscaled_data_dict[variable_name] = normalized_upscaling(ds[variable_name], upscale_factor)
                elif upscale_type == 'name_additional_particles':
                    upscaled_data_dict[variable_name] = name_additional_particles(ds[variable_name], upscale_factor)
                elif upscale_type == 'vector_interpolation':
                    upscaled_data_dict[variable_name] = vector_interpolation(ds[variable_name], x_upscale, y_upscale, z_upscale, upscaled_field, new_particle_positions)

    return upscaled_data_dict

def interpolation_upscaling(data, x_upscaled, y_upscaled, z_upscaled, upscaled_field, new_particle_positions, reshaped_field, flags):
    interpolator = rgi((x_upscaled, y_upscaled, z_upscaled), reshaped_field)
    return interpolator(new_particle_positions)

def normalized_upscaling(data, upscale_factor):
    return data / upscale_factor**3

def name_additional_particles(data, upscale_factor):
    return np.arange(1, len(data)*upscale_factor**3 + 1)

def vector_interpolation(data, x_grid, y_grid, z_grid, upscaled_field, new_particle_positions):
    reshaped_data = data.reshape(x_grid.shape)
    interpolated_data = []
    for i in range(data.shape[1]):
        interpolator = rgi((x_grid, y_grid, z_grid), reshaped_data[..., i])
        interpolated_data.append(interpolator(new_particle_positions))
    return np.array(interpolated_data).T
#}}}

# Main upscaler (By ChatGPT) {{{
def sk_upscaler_main(ds, upscale_factor, BoxSize, flags):
    """
    Perform data upscaling using smoothing kernel and generates new particle positions.

    Parameters:
    ds (dict): The dictionary containing original data.
    upscale_factor (int): The factor by which to increase the resolution.
    BoxSize (float): The size of the box.
    flags (dict): The dictionary containing flags for different options.

    Returns:
    upscaled_data_dict (dict): The dictionary containing upscaled data.
    """
    # Extract necessary data from the dictionary
    if 'debugging' in flags:
        print('Extracting data from the dictionary...')
    particle_positions = ds['Coordinates']
    smoothing_lengths = ds['SmoothingLength']
    if 'debugging' in flags:
        print()
        print('Old number of particles: ')
        print(len(smoothing_lengths))
    #masses = ds['Masses']
    density = ds['Density']
    if 'debugging' in flags:
        print('Data extracted from the dictionary')

    # Determine the size of the original box and the desired size of the upscaled box
    box_size = [BoxSize, BoxSize, BoxSize]
    field_shape = tuple((np.array(box_size) * upscale_factor).astype(int))

    # Create a density field from your particle data
    if 'debugging' in flags:
        print(np.shape(particle_positions))
    field, (x_grid, y_grid, z_grid) = create_density_field(particle_positions, smoothing_lengths, box_size, upscale_factor,flags)
    if 'debugging' in flags:
        print('Density field created')
        print(np.shape(x_grid))

    # Upscale the field
    upscaled_field, (x_upscaled, y_upscaled, z_upscaled) = upscale_field(field, x_grid, y_grid, z_grid, upscale_factor,flags)
    if 'debugging' in flags:
        print('Field upscaling complete')

    # Determine the number of particles to generate in the upscaled data
    n_particles = len(particle_positions) * upscale_factor**3

    # Generate new particle positions based on the upscaled field
    new_particle_positions = generate_particles_vectorized(upscaled_field, x_upscaled, y_upscaled, z_upscaled, n_particles,flags)
    if 'debugging' in flags:
        print('New particles positions generated')

    # Create the upscaled data dictionary
    upscaled_data_dict = {}
    upscaled_data_dict['Coordinates'] = new_particle_positions
    if 'debugging' in flags:
        print('Coordinates upscaling complete')
    # Since we are upscaling, the masses of the particles should decrease
    upscaled_data_dict['SmoothingLength'] = smoothing_lengths / upscale_factor**3
    # We need to interpolate the densities for the new particle positions

    #density = density.reshape(x_grid.shape)
    x_grid, y_grid, z_grid = np.meshgrid(x_upscaled, y_upscaled, z_upscaled, indexing='ij')
    
    # Reshape upscaled field to match the grid size
    reshaped_field = upscaled_field.reshape(x_grid.shape)

    # Create a regular grid interpolator for the reshaped field
    density_interpolator = rgi((x_upscaled, y_upscaled, z_upscaled), reshaped_field)

    #density_interpolator = rgi((x_grid, y_grid, z_grid), density)
    if 'debugging' in flags:
        print('Density upscaling complete')
    upscaled_data_dict['Density'] = density_interpolator(new_particle_positions)

    if 'debugging' in flags:
        print()
        print('New number of particles: ')
        print(len(upscaled_data_dict['Density']))

    return upscaled_data_dict


#}}}

def sk_upscaler_mainV2(ds, upscale_factor, BoxSize, flags):
    if 'debugging' in flags:
        print('Extracting data from the dictionary...')
    particle_positions = ds['Coordinates']
    smoothing_lengths = ds['SmoothingLength']

    if 'debugging' in flags:
        print('\nOld number of particles: ', len(smoothing_lengths))

    density = ds['Density']

    box_size = [BoxSize, BoxSize, BoxSize]

    field, (x_grid, y_grid, z_grid) = create_density_field(particle_positions, smoothing_lengths, box_size, upscale_factor,flags)

    upscaled_field, (x_upscaled, y_upscaled, z_upscaled) = upscale_field(field, x_grid, y_grid, z_grid, upscale_factor,flags)

    n_particles = len(particle_positions) * upscale_factor**3

    new_particle_positions = generate_particles_vectorized(upscaled_field, x_upscaled, y_upscaled, z_upscaled, n_particles,flags)

    if 'debugging' in flags:
        print('New particles positions generated')

    # Create the upscaled data dictionary
    upscaled_data_dict = {}
    upscaled_data_dict['Coordinates'] = new_particle_positions
    if 'debugging' in flags:
        print('Coordinates upscaling complete')

    # Reshape upscaled field to match the grid size
    reshaped_field = upscaled_field.reshape(x_grid.shape)

    # Obtain remaining upscaled data
    upscaled_data_dict.update(interpolate_data(ds, 
                                               upscale_factor, 
                                               box_size, 
                                               x_upscaled, 
                                               y_upscaled, 
                                               z_upscaled, 
                                               upscaled_field, 
                                               new_particle_positions, 
                                               reshaped_field,
                                               flags))

    if 'debugging' in flags:
        print('\nNew number of particles: ', len(upscaled_data_dict['Density']))

    return upscaled_data_dict

