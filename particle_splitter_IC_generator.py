import h5py
import numpy as np
import shutil

def add_parttype3_at_center_of_mass_final(input_filepath, output_filepath):
    """
    Final function to add a new PartType3 particle at the center of mass of PartType0 within a snapshot.
    
    Parameters:
    - filepath: Path to the snapshot file.
    """
    # Fields for PartType3, as provided with reference to PartType1
    parttype3_fields = [
        'AGS-KernelRadius', 'Coordinates', 'Masses', 'ParticleChildIDsNumber',
        'ParticleIDGenerationNumber', 'ParticleIDs', 'Softening_KernelRadius', 'Velocities'
    ]
    shutil.copy(input_filepath, output_filepath)
    
    with h5py.File(output_filepath, 'a') as f:
        # List of all headers in the snapshot
        all_headers = list(f.keys())
        #print(all_headers)
        
        # Compute the center of mass of PartType0 from the first header
        coordinates = f['PartType0']['Coordinates'][:]
        masses = f['PartType0']['Masses'][:]
        center_of_mass = sum(masses[:, None] * coordinates) / sum(masses)
        center_of_mass = np.array(center_of_mass)
        print(center_of_mass)
        
        # If PartType3 doesn't exist, create it
        if 'PartType3' not in f:
            f.create_group('PartType3')
        
        # For each field, create a dataset with a single value (initialized with zeros for simplicity)
        for field in parttype3_fields:
            if field not in f['PartType3']:
                # Determine the correct shape for the dataset
                data_shape = (1, 3) if field in ['Coordinates', 'Velocities'] else (1,)
                #print('shape:')
                #print(np.shape(center_of_mass))

                dtype = float  # Default dtype
        
                # For 'Coordinates', set the value to center_of_mass
                if field == 'Coordinates':
                    data = np.array([center_of_mass])
                elif field == 'Velocities':
                    data = np.array([[0.0, 0.0, 0.0]])
                elif field == 'Masses':
                    data = np.array([10.0])
                else:
                    data = np.array([0.0])

                print(f'Shape of {field} is {data_shape}')
        
                # Create the dataset for PartType3
                f['PartType3'].create_dataset(field, shape=data_shape, dtype=dtype)
                f['PartType3'][field][...] = data

in_file_path = '/fs/lustre/scratch/vpustovoit/MUSIC2/output/2023.09.15:1/snapshot_031.hdf5'
out_file_path = '/fs/lustre/scratch/vpustovoit/MUSIC2/output/2023.09.15:1/snapshot_031_modified.hdf5'

# Test the final function on the provided snapshot
add_parttype3_at_center_of_mass_final(in_file_path, out_file_path)

# Check if the PartType3 particle was created successfully after using the final function
with h5py.File(out_file_path, 'r') as f:
    datasets_after_final_function = list(f.keys())
    parttype3_fields_after_final_function = list(f['PartType3'].keys())


