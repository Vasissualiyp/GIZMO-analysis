import h5py
import numpy as np

# GIZMO ICs from the dataframe {{{
def generate_GIZMO_IC(filename, 
                        df,
                        Boxsize, 
                        NumFilesPerSnapshot=1, 
                        Time=0.00990099, 
                        Redshift=100, 
                        DoublePrecision=False):
    OmegaMatter = 1.0
   
    # Write hdf5 file
    IC = h5py.File(filename, 'w')

    # Create hdf5 groups
    header = IC.create_group("Header")
    part0 = IC.create_group("PartType0")

    # Copy datasets
    for subgroup in df[group_name]['PartType0']:
        # Store the data in the dictionary
        part0.create_dataset(subgroup, data=df[subgroup])

    # Header entries
    header.attrs.create("BoxSize", Boxsize)
    header.attrs.create("ComovingIntegrationOn", 1)
    header.attrs.create("Effective_Kernel_NeighborNumber", 32)
    header.attrs.create("Fixed_ForceSoftening_Keplerian_Kernel_Extent", np.array([280, 0, 0, 0, 0, 0], dtype=np.float64))
    header.attrs.create("Flag_Cooling", 0)
    header.attrs.create("Flag_DoublePrecision", int(DoublePrecision))
    header.attrs.create("Flag_Feedback", 0)
    header.attrs.create("Flag_IC_Info", 3)
    header.attrs.create("Flag_Metals", 0)
    header.attrs.create("Flag_Sfr", 0)
    header.attrs.create("Flag_StellarAge", 0)
    header.attrs.create("GIZMO_version", 2022)
    header.attrs.create("Gravitational_Constant_In_Code_Inits", 43007.1)
    header.attrs.create("HubbleParam", 1)
    header.attrs.create("Kernel_Function_ID", 3)
    header.attrs.create("MassTable", np.array([0, 0, 0, 0, 0, 0], dtype=np.float64))
    header.attrs.create("Maximum_Mass_For_Cell_Split", 85.8911)
    header.attrs.create("Minimum_Mass_For_Cell_Merge", 8.6866)
    header.attrs.create("NumFilesPerSnapshot", NumFilesPerSnapshot)
    header.attrs.create("NumPart_ThisFile", np.array([NumberOfCells, 0, 0, 0, 0, 0], dtype=np.int32))
    header.attrs.create("NumPart_Total", np.array([NumberOfCells, 0, 0, 0, 0, 0], dtype=np.uint32))
    header.attrs.create("NumPart_Total_HighWord", np.array([0, 0, 0, 0, 0, 0], dtype=np.uint32))
    header.attrs.create("Omega_Baryon", 1)
    header.attrs.create("Omega_Lambda", 0)
    header.attrs.create("Omega_Matter", OmegaMatter)
    header.attrs.create("Omega_Radiation", 0)
    header.attrs.create("Redshift", Redshift)
    header.attrs.create("Time", Time)
    header.attrs.create("UnitLength_In_CGS", 3.08568e+21)
    header.attrs.create("UnitMass_In_CGS", 1.989e+43)
    header.attrs.create("UnitVelocity_In_CGS", 100000)

    # Close hdf5 file
    IC.close()
#}}}
