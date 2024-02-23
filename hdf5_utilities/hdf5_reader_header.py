import h5py
import sys

def main(hdf5_file_location):
    print(f"Reading snapshot file {hdf5_file_location}...")
    with h5py.File(hdf5_file_location, "r") as f:
        
        print("Successfully opened the file")
        # Read the header information
        header = f["Header"]
        box_size = header.attrs["BoxSize"]
        num_particles = header.attrs["NumPart_ThisFile"]
        redshift = header.attrs["Redshift"]
        #omega_m = header.attrs["Omega0"]
        #omega_l = header.attrs["OmegaLambda"]
        hubble_param = header.attrs["HubbleParam"]
        # Read the mass of type 1 particles
        type_1_masses = f["PartType1/Masses"]
        mass_of_type_1_particle_gizmo = type_1_masses[0] # GIZMO mass units: 10^10 Msun
        mass_of_type_1_particle_msun = mass_of_type_1_particle_gizmo / 10**10 # Rockstar mass units: Msun
        # Output the summary to a text file
        output_file = "zeldovich_ics_summary.txt"
        with open(output_file, "w") as out_file:
            out_file.write(f"File name: {hdf5_file_location}\n")
            out_file.write(f"Box size: {box_size}\n")
            out_file.write("Number of particles type 0 (gas): {}\n".format(num_particles[0]))
            out_file.write("Number of particles type 1 (dark matter): {}\n".format(num_particles[1]))
            out_file.write("Number of particles type 2 (collisionless): {}\n".format(num_particles[2]))
            out_file.write("Number of particles type 3 (grains/PIC particles): {}\n".format(num_particles[3]))
            out_file.write("Number of particles type 4 (stars): {}\n".format(num_particles[4]))
            out_file.write("Number of particles type 5 (black holes/sinks): {}\n".format(num_particles[5]))
            out_file.write(f"Redshift: {redshift}\n")
            out_file.write(f"PartType1 Mass: {mass_of_type_1_particle_msun}\n")
            #out_file.write(f"Omega_m: {omega_m}\n")
            #out_file.write(f"Omega_L: {omega_l}\n")
            out_file.write(f"Hubble parameter: {hubble_param}\n")
        print(f"Output file saved in {output_file}")
if __name__ == "__main__":
    
    # Check if at least one argument is provided (first argument is the script name)
    if len(sys.argv) > 1:
        hdf5_file_location = sys.argv[1]
    else:
        print("No HDF5 file location provided.")
        sys.exit(1)  # Exit the script with an error code

    main(hdf5_file_location) 

