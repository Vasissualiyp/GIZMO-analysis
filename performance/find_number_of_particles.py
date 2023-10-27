import h5py

def get_number_of_elements_in_masses(file_path):
    with h5py.File(file_path, 'r') as f:
        # Access the "Masses" dataset within the "PartType0" group
        dataset_name = 'Masses'
        conversion_arg = 1
        units = 'Code units'
        masses_dataset = f['PartType2'][dataset_name]
        #maxmass = max(masses_dataset) * conversion_arg
        #print(f"Maximum "+dataset_name+f" of a particle is: {maxmass} {units}")
        #minmass = min(masses_dataset) * conversion_arg
        #print(f"Minimum "+dataset_name+f" of a particle is: {minmass} {units}")
        
        # Return the number of elements
        return masses_dataset.size

def print_contents_of_parttype3(file_path):
    with h5py.File(file_path, 'r') as f:
        # Access the "PartType3" group
        #parttype3_group = f['PartType3']
        parttype3_group = f['PartType0']

        for dataset_name, dataset in parttype3_group.items():
            print(f"Dataset Name: {dataset_name}")
            print("Contents:")
            print(dataset[:])  # This prints the entire dataset
            print("-" * 50)  # Separator for clarity

file_path1 = "../output/2023.10.17:5/snapshot_052.hdf5"
#file_path2 = "../output/2023.10.20:9/snapshot_001.hdf5"
#file_path = "../IC.hdf5"

#print_contents_of_parttype3(file_path2)

#print_contents_of_parttype3(file_path1)
num_elements = get_number_of_elements_in_masses(file_path1)
print(f"Number of elements in the 'Masses' dataset (1st file): {num_elements}")
#num_elements = get_number_of_elements_in_masses(file_path2)
#print(f"Number of elements in the 'Masses' dataset (2nd file): {num_elements}")

