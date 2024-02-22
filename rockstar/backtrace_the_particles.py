# Run this script from the main git dir, not the rockstar dir!
import halo_extractor as HalExt
import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

#--------------------SOURCE CODE------------------------- {{{
# Mpc to Mpc/h {{{
def Mpc_to_Mpch(d_Mpc, z, H0, Om0):
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    Hz = cosmo.H(z)
    h = Hz / 100 
    h = h.value
    print(f"Hubble param: {h}")
    print(f"Type: {type(d_Mpc)}")
    d_Mpch = d_Mpc * h
    print(f"Type: {type(d_Mpch)}")
    return d_Mpch
#}}}

# Extract positions from the hdf5 file {{{
def extract_positions_IDs(filepath, ParticleType, IDs=None): 
    """
    Extracts raw HDF5 data for positions and IDs of the particles.

    Parameters:
        filepath (str): Path to the HDF5 file.
        ParticleType (str): Type of particles to consider (e.g., "PartType0").
        IDs (ndarray, optional): Numpy array containing the IDs of particles to be filtered.

    Returns:
        positions (ndarray): Numpy array containing the positions of the particles.
        ids (ndarray): Numpy array containing the IDs of the particles.
    """
    print(f'Extracting info about {ParticleType} from {filepath}...')
    with h5py.File(filepath, 'r') as f:
        # Import the data from hdf5 file
        hdf5_positions = f['/{}/Coordinates'.format(ParticleType)][:]
        hdf5_ids = f['/{}/ParticleIDs'.format(ParticleType)][:]

        # Set the mask only for the particles, whose IDs were passed to a function (if they were passed)
        if IDs is not None: # Tackling the case of only selected IDs
            if not isinstance(IDs, np.ndarray):
                raise ValueError("IDs must be a numpy array")
            mask = np.isin(hdf5_ids, IDs)
            hdf5_positions = hdf5_positions[mask]
            hdf5_ids = hdf5_ids[mask]

        # Extracting cosmological parameters and converting distance to Mpc/h
        z = f['Header'].attrs['Redshift']
        h = f['Header'].attrs['HubbleParam']
        H0 = h * 100
        OmO = f['Header'].attrs['Omega_Matter']

        #hdf5_positions = Mpc_to_Mpch(hdf5_positions, z, H0, OmO)
        #print()
        #print(f"Here are the boundaries of positions of {filepath}:")
        #print(f"{int(min(hdf5_positions[:, 0]))}, {int(min(hdf5_positions[:, 1]))}, {int(min(hdf5_positions[:, 2]))}")
        #print(f"{int(max(hdf5_positions[:, 0]))}, {int(max(hdf5_positions[:, 1]))}, {int(max(hdf5_positions[:, 2]))}")
        #print(np.max(hdf5_positions))

    return hdf5_positions, hdf5_ids
#}}}

# Filter particles to the padding box {{{
def filter_particles_in_padded_box(positions, ids, box_size, padding):
    """
    Filters particles that are within a cubic box considering a padding from the edges.

    Parameters:
        positions (np.ndarray): A numpy array of shape (n, 3) containing the positions of the particles.
        ids (np.ndarray): A numpy array containing the IDs of the particles.
        box_size (float): The size of the cubic box in which the particles reside.
        padding (float): The padding value inside the box boundaries to filter the particles.

    Returns:
        np.ndarray: Filtered positions of the particles within the padded box.
        np.ndarray: IDs of the particles within the padded box.

    Raises:
        ValueError: If `positions` and `ids` arrays do not have the same length.
        ValueError: If `padding` is negative or greater than half of the `box_size`.
    """

    if len(positions) != len(ids):
        raise ValueError("The lengths of positions and IDs must match.")

    if not (0 <= padding < box_size / 2):
        raise ValueError("Padding must be non-negative and less than half of the box size.")

    # Define the lower and upper bounds considering the padding
    lower_bound = padding
    upper_bound = box_size - padding

    # Create a mask for particles within the padded boundaries
    within_bounds_mask = np.all((positions >= lower_bound) & (positions <= upper_bound), axis=1)

    # Apply the mask to filter positions and IDs
    filtered_positions = positions[within_bounds_mask]
    filtered_ids = ids[within_bounds_mask]

    return filtered_positions, filtered_ids
#}}}

# Get IDs of particles that are inside of a given halo {{{
def get_particle_ids_in_halo(halo_center, halo_radius, particle_positions, particle_ids, halo_radius_fraction): 
    """
    Returns the IDs of particles that are within a given halo.

    Parameters:
        halo_center (tuple): Coordinates of the halo center.
        halo_radius (float): Radius of the halo.
        particle_positions (ndarray): Numpy array containing the positions of the particles.
        particle_ids (ndarray): Numpy array containing the IDs of the particles.

    Returns:
        ndarray: Numpy array containing the IDs of particles inside the halo.
    """
    print('Getting particle IDs in the halo...')

    # Calculate the distance of each particle from the halo center
    distances = np.linalg.norm(particle_positions - np.array(halo_center), axis=1)

    # Find the particles within the halo radius
    halo_radius = halo_radius_fraction * halo_radius
    #print(f"Halo virial radius: {halo_radius} kpc") 
    #print(f"Halo center (kpc):")
    #print(f"{halo_center}")
    inside_halo_mask = distances < halo_radius

    # Extract the IDs of these particles
    inside_halo_ids = particle_ids[inside_halo_mask]

    return inside_halo_ids
#}}}

# Calculate bounding box from positions {{{
def calculate_bounding_box(positions):
    """
    Finds the bounding box for a set of particles.

    Parameters:
        positions (ndarray): An N x 3 array of particle positions (x, y, z).

    Returns:
        bounding_box_min (ndarray): 1 x 3 array indicating the minimum coordinates (x_min, y_min, z_min) of the bounding box.
        bounding_box_max (ndarray): 1 x 3 array indicating the maximum coordinates (x_max, y_max, z_max) of the bounding box.
        bounding_box_size (ndarray): 1 x 3 array indicating the size (dx, dy, dz) of the bounding box.
    """
    # Find the minimum and maximum coordinates along each axis
    bounding_box_min = np.min(positions, axis=0)
    bounding_box_max = np.max(positions, axis=0)
    
    # Calculate the size of the bounding box
    bounding_box_size = bounding_box_max - bounding_box_min
    bounding_box_pos = (bounding_box_max + bounding_box_min) / 2
    
    return bounding_box_pos, bounding_box_size
#}}}

# Master function to get the parameters of the bounding box {{{
def get_bounding_box_parameters(start_snapshot, last_snapshot, halo_file_path, box_padding, halo_radius_fraction):
    
    # Extract relevant info from the last snapshot
    particle_positions, particle_IDs = extract_positions_IDs(last_snapshot, "PartType1")
    
    # Print data {{{
    #print("positions:")
    #print(particle_positions)
    #print(f"Max position: {max(particle_positions[0])}")
    #print(f"Min position: {min(particle_positions[0])}")
    #
    #print()
    #print("IDs:")
    #print(particle_IDs)
    #print(f"Max ID: {max(particle_IDs)}")
    #print(f"Min ID: {min(particle_IDs)}")
    #print(f"Length: {len(particle_IDs)}")
    #}}}

    # Create a dataframe with all halo data
    halo_data_df = HalExt.read_halo_data(halo_file_path)
    halo_data_df = HalExt.convert_halo_df_to_gizmo_units(halo_data_df)

    # Restrict the halo-finding region to the padded region (to prevent getting out of boundaries for the box)
    corner1_x = np.min(particle_positions)  + box_padding
    corner2_x = np.max(particle_positions) - box_padding
    corner1 = (corner1_x, corner1_x, corner1_x)
    corner2 = (corner2_x, corner2_x, corner2_x)
    #print(f"corner 1: {corner1}")
    #print(f"corner 2: {corner2}")

    padded_halo_data_df = HalExt.restrict_dataset(halo_data_df, corner1, corner2)
    #print("Padded halo dataset:")
    #print(padded_halo_data_df)
    #print()

    # Obtain the relevant data about most massive halo
    halo_position, halo_radius, halo_mass = HalExt.get_largest_halo(padded_halo_data_df)
    
    # Get the IDs of the particles inside of halos
    inside_halo_IDs = get_particle_ids_in_halo(halo_position, 
                                               halo_radius, 
                                               particle_positions,                                               particle_IDs,
                                               halo_radius_fraction)

    # Get the initial positions of the particles inside of the halo 
    initial_particle_positions, particle_IDs = extract_positions_IDs(start_snapshot, 
                                                                     "PartType1", 
                                                                     inside_halo_IDs)
    
    b_box_pos, b_box_size = calculate_bounding_box(initial_particle_positions)

    return  b_box_pos, b_box_size
#}}}

#--------------------MAIN-------------------------}}}

if __name__ == "__main__":

    # Set the parameters
    snapshots_dir='2024.01.22:2' # Directory with all snapshot files
    today_snap = '008' # The last snapshot file (where the halo has been identified)
    halo_file_path = '../rockstar/halos_0.0.ascii'  # Replace with your file path
    box_file_path =  './rockstar/boundbox_characteristics.txt'
    padding = 6000 # Padding in kpc - Only haloes in a box, padded with this value are considered
    halo_radius_fraction = 2
    
    # Working with directories {{{
    snapshots_dir = '../output/' + snapshots_dir + '/'
    last_snapshot= snapshots_dir + 'snapshot_' + today_snap + '.hdf5'
    start_snapshot= snapshots_dir + 'snapshot_000.hdf5'
    #}}}

    boundbox_pos, boundbox_size = get_bounding_box_parameters(start_snapshot, last_snapshot, halo_file_path, padding, halo_radius_fraction)

    # Print the output {{{
    boundbox_size = boundbox_size / 1000 # convert to Mpc
    boundbox_size_max = max(boundbox_size)
    boundbox_size_full = np.ones(3) * boundbox_size_max
    #boundbox_size = boundbox_size_full # Enable this if you want a cube box
    boundbox_pos = boundbox_pos / 1000 # convert to Mpc

    #print(f'Box Size: {boundbox_size} Mpc')
    #print(f'Box Position: {boundbox_pos} Mpc')

    # Formatting each element to a specific number of decimal places
    formatted_size = ', '.join(f'{size:.3f}' for size in boundbox_size)
    formatted_pos = ', '.join(f'{pos:.3f}' for pos in boundbox_pos)
    
    # Printing in the desired format
    with open(box_file_path, 'w') as file:
        file.write(f'ref_extent= \t\t{formatted_size}\n')
        file.write(f'ref_center= \t\t{formatted_pos}\n')
    #}}}
