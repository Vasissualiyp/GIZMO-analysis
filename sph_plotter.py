# Libraries {{{
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
#}}}

# Optimized density projection {{{
def sph_density_projection_optimized(x, y, z, density, smoothing_lengths, flags, resolution=100, log_density=False):
    """
    Creates a density projection plot of SPH data optimized for large datasets.
    Uses adaptive kernel density estimation

    Parameters:
        x, y, z: Arrays of x, y, z coordinates.
        density: Array of density values.
        smoothing_lengths: Array of smoothing lengths.
        resolution: The resolution of the grid for plotting. Default is 100.
    """
    small_positive_number = 1e-10
    # Convert input arrays to numpy arrays
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    density = np.array(density)
    smoothing_lengths = np.array(smoothing_lengths)

    # Create a grid for plotting
    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), resolution),
        np.linspace(y.min(), y.max(), resolution)
    )

    # Calculate the projected density
    projected_density = np.zeros((resolution, resolution))

    # Wrap your iterable with tqdm for a progress bar
    for xi, yi, zi, densi, hi in tqdm(zip(x, y, z, density, smoothing_lengths), total=len(x)):
        kernel = np.exp(-0.5 * ((grid_x - xi) ** 2 + (grid_y - yi) ** 2) / hi ** 2) / (2 * np.pi * hi ** 2)
        projected_density += densi * kernel

    # Plotting the result
    plt.figure()
    if log_density == True:
        projected_density = np.log10(projected_density + small_positive_number)
        plt.colorbar(label='log(Projected Density)')
    else:
        plt.colorbar(label='Projected Density')
    plt.imshow(np.rot90(projected_density), cmap=plt.cm.cubehelix,
               extent=[x.min(), x.max(), y.min(), y.max()],
               aspect='auto')
    plt.colorbar(label='Projected Density')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('SPH Density Projection Plot')
    plt.show()
    return plt
# }}}

# Density projection {{{
def sph_density_projection(x, y, z, density, smoothing_lengths, resolution=100):
    """
    Creates a density projection plot of SPH data.

    Parameters:
        x, y, z: Arrays of x, y, z coordinates.
        density: Array of density values.
        smoothing_lengths: Array of smoothing lengths.
        resolution: The resolution of the grid for plotting. Default is 100.
    """
    # Convert input arrays to numpy arrays
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    density = np.array(density)
    smoothing_lengths = np.array(smoothing_lengths)

    # Calculate the density field in 3D
    xyz = np.vstack([x, y, z])
    kde_3d = gaussian_kde(xyz, weights=density, bw_method=smoothing_lengths.mean())

    # Create a grid for plotting
    grid_x, grid_y, grid_z = np.mgrid[x.min():x.max():resolution*1j, y.min():y.max():resolution*1j, z.min():z.max():resolution*1j]
    positions_3d = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    density_field = kde_3d(positions_3d).reshape(grid_x.shape)

    # Sum the density along the z-axis
    projected_density = np.sum(density_field, axis=2)

    # Plotting the result
    plt.figure()
    plt.imshow(np.rot90(projected_density), cmap=plt.cm.plasma,
               extent=[x.min(), x.max(), y.min(), y.max()],
               aspect='auto')
    plt.colorbar(label='Projected Density')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('SPH Density Projection Plot')
    plt.show()
#}}}

# 2D plotter {{{
def sph_plotter2D(x,y,z,density,smoothing_lengths):
    # 2D case:
    # Kernel density estimation in 2D
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, weights=density, bw_method=smoothing_lengths.mean())

    # Creating a grid for plotting
    grid_x, grid_y = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
    positions = np.vstack([grid_x.ravel(), grid_y.ravel()])
    z_kde = kde(positions).reshape(grid_x.shape)

    # Plotting the result
    plt.figure()
    plt.imshow(np.rot90(z_kde), cmap=plt.cm.gist_earth_r,
               extent=[x.min(), x.max(), y.min(), y.max()])
    plt.colorbar(label='Density')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Smoothed Particle Hydrodynamics Plot')
    plt.show()
    return plt
# }}}

# 3D sph plotter {{{
def sph_plotter3D(x,y,z,density,smoothing_lengths):

    # 3D case (optional):
    # If you want to plot in 3D, you can use the following code instead of the 2D plotting code

    xyz = np.vstack([x, y, z])
    kde_3d = gaussian_kde(xyz, weights=density, bw_method=smoothing_lengths.mean())

    grid_x, grid_y, grid_z = np.mgrid[x.min():x.max():30j, y.min():y.max():30j, z.min():z.max():30j]
    positions_3d = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    z_kde_3d = kde_3d(positions_3d).reshape(grid_x.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(grid_x, grid_y, grid_z, c=z_kde_3d.ravel(), cmap=plt.cm.gist_earth_r, s=1)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    plt.colorbar(label='Density')
    plt.title('Smoothed Particle Hydrodynamics 3D Plot')
    plt.show()

    return plt
#}}}
