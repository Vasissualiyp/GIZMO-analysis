import halo_extractor
import h5py
import numpy as np

#--------------------SOURCE CODE------------------------- {{{
# Extract positions from the hdf5 file {{{
def extract_positions_IDs(filepath, ParticleType): 
    """
    Extracts raw HDF5 data for positions and IDs of the particles.

    Parameters:
        filepath (str): Path to the HDF5 file.

    Returns:
        positions (ndarray): Numpy array containing the positions of the particles.
        ids (ndarray): Numpy array containing the IDs of the particles.
    """
    with h5py.File(filepath, 'r') as f:
        positions = f['/{}/Coordinates'.format(ParticleType)][:]
        ids = f['/{}/ParticleIDs'.format(ParticleType)][:]
    return positions, ids
#}}}

# Get IDs of particles that are inside of a given halo {{{
def get_particle_ids_in_halo(halo_center, halo_radius, particle_positions, particle_ids): 
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
    # Calculate the distance of each particle from the halo center
    distances = np.linalg.norm(particle_positions - np.array(halo_center), axis=1)

    # Find the particles within the halo radius
    inside_halo_mask = distances < halo_radius

    # Extract the IDs of these particles
    inside_halo_ids = particle_ids[inside_halo_mask]

    return inside_halo_ids
#}}}
#--------------------MAIN-------------------------}}}

# Set the parameters
snapshots_dir='2023.10.27:6' # Directory with all snapshot files
today_snap = '000' # The last snapshot file (where the halo has been identified)

# Working with directories {{{
snapshots_dir = '../../output/' + snapshots_dir + '/'
today_snap= snapshots_dir + 'snapshot_' + today_snap + '.hdf5'
starting_snap= snapshots_dir + 'snapshot_000.hdf5'
#}}}

positions, IDs = extract_positions_IDs(today_snap, "PartType2")

print("positions:")
print(positions)
print(f"Max position: {max(positions[0])}")
print(f"Min position: {min(positions[0])}")

print()
print("IDs:")
print(IDs)
print(f"Max ID: {max(IDs)}")
print(f"Min ID: {min(IDs)}")
print(f"Length: {len(IDs)}")

