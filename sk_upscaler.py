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

def gaussian_kernel_unnormalized(grid, position, h, density, flags):
    x, y, z = grid
    grid_points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    rv = multivariate_normal(mean=position, cov=[[h**2, 0, 0], [0, h**2, 0], [0, 0, h**2]])

    kernel = rv.pdf(grid_points) * density * (h**3)  # Unnormalize the kernel
    return kernel.reshape(x.shape)

#}}}

# Create the density field {{{
def create_density_field(particle_positions, smoothing_lengths, densities, box_size, resolution, flags):
    """
    Create a density field from a set of particles.

    Parameters:
    particle_positions (array-like): Nx3 array of particle positions.
    smoothing_lengths (array-like): Nx1 array of smoothing lengths for each particle.
    densities (array-like): Nx1 array of densities for each particle.
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
    grid_tuple = np.mgrid[-box_size[0]/2:box_size[0]/2:field_shape[0]*1j,
                      -box_size[1]/2:box_size[1]/2:field_shape[1]*1j,
                      -box_size[2]/2:box_size[2]/2:field_shape[2]*1j]

    # Create a 3D Gaussian kernel for each particle and add it to the field
    for position, h, density in zip(particle_positions, smoothing_lengths, densities):
        kernel = gaussian_kernel_unnormalized(grid_tuple, position, h, density, flags)
        field += density * kernel  # The kernel is now weighted by the density
        #field += kernel  # The kernel is now weighted by the density

    return field, grid_tuple
#}}}

# Create the scalar field {{{
def create_scalar_field(particle_positions, smoothing_lengths, field_values, box_size, grid_tuple, resolution, flags):
    """
    Create a scalar field from a set of particles.

    Parameters:
    particle_positions (array-like): Nx3 array of particle positions.
    smoothing_lengths (array-like): Nx1 array of smoothing lengths for each particle.
    field_values (array-like): Nx1 array of field_values for each particle.
    box_size (array-like): Dimensions of the box that contains the particles.
    resolution (int): The resolution of the grid.

    Returns:
    field (array-like): The scalar field.
    x_grid, y_grid, z_grid (array-like): The coordinates of the grid points.
    """
    # Calculate the number of grid points in each dimension
    field_shape = tuple((np.array(box_size) * resolution).astype(int))

    # Initialize an empty field
    field = np.zeros(field_shape)

    # Create a 3D Gaussian kernel for each particle and add it to the field
    for position, h, scalar in zip(particle_positions, smoothing_lengths, field_values):
        kernel = gaussian_kernel_unnormalized(grid_tuple, position, h, scalar, flags)
        field += scalar * kernel  # The kernel is now weighted by the scalar

    return field
#}}}

# Upscale density field {{{
def upscale_density_field(field, grid_tuple, upscale_factor, flags):
    """
    Upscale a 3D field using FFTs.

    Parameters:
    field (3D numpy array): The input field to be upscaled.
    upscale_factor (int): The factor by which to upscale the field.

    Returns:
    upscaled_field (3D numpy array): The upscaled field.
    """

    # Extract the grid
    x_grid, y_grid, z_grid = grid_tuple

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

# Upscale scalar field {{{
def upscale_scalar_field(field, grid_tuple, upscale_factor, flags):
    """
    Upscale a 3D field using FFTs.

    Parameters:
    field (3D numpy array): The input field to be upscaled.
    upscale_factor (int): The factor by which to upscale the field.

    Returns:
    upscaled_field (3D numpy array): The upscaled field.
    """

    # Extract the grid
    x_grid, y_grid, z_grid = grid_tuple

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

    return upscaled_field
#}}}

# Vectorized particle positions generator {{{
def generate_particles_vectorized(upscaled_field, upscaled_tuple, n_particles, flags):
    """
    Generates particles within the box using the upscaled field.

    Parameters:
    upscaled_field (array-like): The upscaled field to use for generating particles.
    upscaled_tuple (x, y, z - tuple): the coordinates of the upscaled field
    n_particles (int): The number of particles to generate.

    Returns:
    particle_positions (array-like): The positions of the generated particles.
    """

    # Extract the new positions
    x_upscaled, y_upscaled, z_upscaled = upscaled_tuple

    # Create a tricubic interpolator for the upscaled field.
    #interpolator = tricubic.tricubic(list(zip(x_upscaled.ravel(), y_upscaled.ravel(), z_upscaled.ravel())), upscaled_field.ravel())
    interpolator = rgi(upscaled_tuple, upscaled_field)

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
def upscale_data(ds, upscale_factor, box_size, grid_tuple, newgrid_tuple, upscaled_tuple, new_particle_positions, flags):
    data_type_lists = {
        'interpolation_upscaling': ['ElectronAbundance', 'InternalEnergy', 'Potential', 'StarFormationRate'],
        'normalized_upscaling':      ['Masses', 'SmoothingLength'],
        'name_additional_particles': ['ParticleIDs', 'ParticleIDGenerationNumber', 'ParticleChildIDsNumber'],
        'vector_interpolation':      ['Velocities', 'Metallicity'],
    }
    resolution = upscale_factor

    upscaled_data_dict = {}
    initial_positions = ds['Coordinates']
    smoothing_lengths = ds['SmoothingLength']
    for variable_name in ds.keys():
        print(variable_name)
        for upscale_type, variable_list in data_type_lists.items():
            #print(f'1. Upscaling type: {upscale_type}')
            if variable_name in variable_list:
                if 'debugging' in flags:
                    print(f'Working with {variable_name}...')
                if upscale_type == 'interpolation_upscaling':
                    upscaled_data_dict[variable_name] = interpolation_upscaling(initial_positions, smoothing_lengths, ds[variable_name], upscale_factor, box_size, grid_tuple, newgrid_tuple, upscaled_tuple, new_particle_positions, flags)
                    if 'debugging' in flags:
                        print(f'Upscaling for {variable_name} is complete!')
                elif upscale_type == 'normalized_upscaling':
                    upscaled_data_dict[variable_name] = normalized_upscaling(initial_positions, smoothing_lengths, ds[variable_name], upscale_factor, box_size, grid_tuple, newgrid_tuple, upscaled_tuple, new_particle_positions, flags)
                    if 'debugging' in flags:
                        print(f'Upscaling for {variable_name} is complete!')
                    """
                elif upscale_type == 'name_additional_particles':
                    upscaled_data_dict[variable_name] = name_additional_particles(ds[variable_name], upscale_factor)
                elif upscale_type == 'vector_interpolation':
                    upscaled_data_dict[variable_name] = vector_interpolation(ds[variable_name], x_upscale, y_upscale, z_upscale, upscaled_field, new_particle_positions)
                    """

    return upscaled_data_dict

def interpolation_upscaling(initial_particle_positions, smoothing_lengths, data, upscale_factor, box_size, grid_tuple, newgrid_tuple, upscaled_tuple, new_positions, flags):
    # Create the field
    field = create_scalar_field(initial_particle_positions, smoothing_lengths, data, box_size, grid_tuple, upscale_factor, flags)

    # Upscale the field with FFT
    upscaled_field = upscale_scalar_field(field, grid_tuple, upscale_factor, flags)

    # Interpolate the field at the new positions of particles
    x_newgrid, y_newgrid, z_newgrid = newgrid_tuple
    reshaped_field = upscaled_field.reshape(x_newgrid.shape)
    interpolator = rgi(upscaled_tuple, reshaped_field)
    return interpolator(new_positions)

def normalized_upscaling(initial_particle_positions, smoothing_lengths, data, upscale_factor, box_size, grid_tuple, newgrid_tuple, upscaled_tuple, new_positions, flags):
    data = interpolation_upscaling(initial_particle_positions, smoothing_lengths, data, upscale_factor, box_size, grid_tuple, newgrid_tuple, upscaled_tuple, new_positions, flags)
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

# The main upscaler function {{{
def sk_upscaler_main(ds, upscale_factor, BoxSize, flags):
    if 'debugging' in flags:
        print('Extracting data from the dictionary...')
    particle_positions = ds['Coordinates']
    densities = ds['Density']
    smoothing_lengths = ds['SmoothingLength']

    if 'debugging' in flags:
        print('\nOld number of particles: ', len(smoothing_lengths))
        print(f'Max of density: {max(densities)}')

    box_size = [BoxSize, BoxSize, BoxSize]

    #Create density field
    field, grid_tuple = create_density_field(particle_positions, smoothing_lengths, densities, box_size, upscale_factor, flags)
    if 'debugging' in flags:
        print(f'Max of density field: {np.max(field)}')

    upscaled_field, upscaled_tuple = upscale_density_field(field, grid_tuple, upscale_factor, flags)
    if 'debugging' in flags:
        print(f'Max of upscaled density field: {np.max(upscaled_field)}')

    n_particles = len(particle_positions) * upscale_factor**3

    new_particle_positions = generate_particles_vectorized(upscaled_field, upscaled_tuple, n_particles, flags)


    if 'debugging' in flags:
        print('New particles positions generated')

    # Create the upscaled data dictionary
    upscaled_data_dict = {}
    upscaled_data_dict['Coordinates'] = new_particle_positions
    if 'debugging' in flags:
        print('Coordinates upscaling complete')

    # Extract the grid
    x_upscaled, y_upscaled, z_upscaled = upscaled_tuple

    #density = density.reshape(x_grid.shape)

    x_newgrid, y_newgrid, z_newgrid = np.meshgrid(x_upscaled, y_upscaled, z_upscaled, indexing='ij')
    newgrid_tuple = (x_newgrid, y_newgrid, z_newgrid)

    # Reshape upscaled field to match the grid size
    reshaped_field = upscaled_field.reshape(x_newgrid.shape)

    interpolator = rgi(upscaled_tuple, reshaped_field)
    upscaled_data_dict['Density'] = interpolator(new_particle_positions)

    if 'debugging' in flags:
        print('Density complete!')

    # Reshape upscaled field to match the upscaled size
    #reshaped_field = upscaled_field.reshape(x_upscaled.shape)

    # Obtain remaining upscaled data
    upscaled_data_dict.update(upscale_data(ds, 
                                           upscale_factor, 
                                           box_size, 
                                           grid_tuple,
                                           newgrid_tuple,
                                           upscaled_tuple,
                                           new_particle_positions, 
                                           flags))

    if 'debugging' in flags:
        print('\nNew number of particles: ', len(upscaled_data_dict['Density']))
        print('\nNew max density: ', np.max(upscaled_data_dict['Density']))

    return upscaled_data_dict
#}}}
