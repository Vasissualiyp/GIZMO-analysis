import h5py
import numpy as np
import shutil

import numpy as np
import shutil
import h5py

def add_parttype3_at_center_of_mass_final(input_filepath, output_filepath):
    """
    Final function to add a new PartType3 particle at the center of mass of PartType0 within a snapshot.

    Parameters:
    - input_filepath: Path to the snapshot file.
    - output_filepath: Path to the output file where the new particle will be added.
    """
    # Fields for PartType3, as provided with reference to PartType1
    parttype3_fields = [
        'AGS-KernelRadius', 'Coordinates', 'Masses', 'ParticleChildIDsNumber',
        'ParticleIDGenerationNumber', 'ParticleIDs', 'Softening_KernelRadius', 'Velocities'
    ]

    shutil.copy(input_filepath, output_filepath)

    with h5py.File(output_filepath, 'a') as f:
        # Compute the center of mass of PartType0
        coordinates = f['PartType0']['Coordinates'][:]
        masses = f['PartType0']['Masses'][:]
        center_of_mass = sum(masses[:, None] * coordinates) / sum(masses)
        center_of_mass = np.array(center_of_mass)

        # If PartType3 doesn't exist, create it
        if 'PartType3' not in f:
            f.create_group('PartType3')

        # For each field, create a dataset with a single value (initialized with zeros for simplicity)
        for field in parttype3_fields:
            if field not in f['PartType3']:
                data_shape = (1, 3) if field in ['Coordinates', 'Velocities'] else (1,)
                dtype = float  # Default dtype

                # Initialize the data based on the field
                if field == 'Coordinates':
                    data = np.array([center_of_mass])
                elif field == 'Velocities':
                    data = np.array([[0.0, 0.0, 0.0]])
                elif field == 'Masses':
                    data = np.array([1e-3])
                elif field == 'ParticleIDs':
                    data = np.array([int(32768)])
                else:
                    data = np.array([0.0])

                # Create the dataset for PartType3
                f['PartType3'].create_dataset(field, shape=data_shape, dtype=dtype)
                f['PartType3'][field][...] = data

        # Update NumPart_ThisFile and NumPart_Total attributes in the header to reflect the new particle
        # Assuming the order is PartType0, PartType1, ... , PartType5
        #print(f['Header'].attrs['NumPart_ThisFile'])

        # Create numpy arrays from the attributes
        num_part_this_file = np.array(f['Header'].attrs['NumPart_ThisFile'])
        num_part_total = np.array(f['Header'].attrs['NumPart_Total'])
        
        # Modify the numpy arrays
        num_part_this_file[3] += 1
        num_part_total[3] += 1
        
        # Set the modified values back to the attributes
        f['Header'].attrs['NumPart_ThisFile'] = num_part_this_file
        f['Header'].attrs['NumPart_Total'] = num_part_total
        
        #print(f['Header'].attrs['NumPart_ThisFile'])
        
        

in_file_path = '/home/vasilii/Software/GIZMO/output/2024.01.22:2/snapshot_008.hdf5'

out_file_path = '/home/vasilii/Software/GIZMO/ICs/PartSplitIC.hdf5'

# Test the final function on the provided snapshot
add_parttype3_at_center_of_mass_final(in_file_path, out_file_path)

# Check if the PartType3 particle was created successfully after using the final function
#with h5py.File(out_file_path, 'r') as f:
#    datasets_after_final_function = list(f.keys())
#    parttype3_fields_after_final_function = list(f['PartType3'].keys())


