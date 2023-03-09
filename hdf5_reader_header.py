#import h5py {{{
#
## Open the initial conditions file
#with h5py.File("zeldovich_ics.hdf5", "r") as f:
#
#    # Read the header information
#    header = f["Header"]
#    box_size = header.attrs["BoxSize"]
#    num_particles = header.attrs["NumPart_Total"]
#    redshift = header.attrs["Redshift"]
#    Omega_m = header.attrs["Omega0"]
#    Omega_L = header.attrs["OmegaLambda"]
#    Hubble_param = header.attrs["HubbleParam"]
#
#    # Output the summary to a text file
#    with open("zeldovich_ics_summary.txt", "w") as out_file:
#        out_file.write(f"Box size: {box_size}\n")
#        out_file.write(f"Number of particles: {num_particles}\n")
#        out_file.write(f"Redshift: {redshift}\n")
#        out_file.write(f"Omega_m: {Omega_m}\n")
#        out_file.write(f"Omega_L: {Omega_L}\n")
#        out_file.write(f"Hubble parameter: {Hubble_param}\n")
#}}}

import h5py

# Open the file
with h5py.File("snapshot1/snapshot_000.hdf5", "r") as f:
    
    # Read the header information
    header = f["Header"]
    box_size = header.attrs["BoxSize"]
    num_particles = header.attrs["NumPart_ThisFile"]
    redshift = header.attrs["Redshift"]
    #omega_m = header.attrs["Omega0"]
    #omega_l = header.attrs["OmegaLambda"]
    hubble_param = header.attrs["HubbleParam"]
  # Output the summary to a text file
    with open("zeldovich_ics_summary.txt", "w") as out_file:
        out_file.write(f"Box size: {box_size}\n")
        out_file.write("Number of particles type 0 (gas): {}\n".format(num_particles[0]))
        out_file.write("Number of particles type 1 (dark matter): {}\n".format(num_particles[1]))
        out_file.write("Number of particles type 2 (collisionless): {}\n".format(num_particles[2]))
        out_file.write("Number of particles type 3 (grains/PIC particles): {}\n".format(num_particles[3]))
        out_file.write("Number of particles type 4 (stars): {}\n".format(num_particles[4]))
        out_file.write("Number of particles type 5 (black holes/sinks): {}\n".format(num_particles[5]))
        out_file.write(f"Redshift: {redshift}\n")
        #out_file.write(f"Omega_m: {omega_m}\n")
        #out_file.write(f"Omega_L: {omega_l}\n")
        out_file.write(f"Hubble parameter: {hubble_param}\n")
    
