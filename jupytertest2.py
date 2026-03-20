
#%matplotlib inline
#%config InlineBackend.figure_format = 'png'
from IPython.display import display
import sys, os, importlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import h5py
from scipy import stats
from scipy.fft import fftn, fftfreq
from scipy.spatial import cKDTree


# Setup paths
scratch_analysis_path = "/scratch/vasissua/SHIVAN/analysis/"
sys.path.insert(0, scratch_analysis_path)
import meshoid_plotting.starforge_plot as sfp
import meshoid_plotting.utility_funcs as utilf
#from jupytertest import plot_zooms

importlib.reload(sfp)
importlib.reload(utilf)

from vasthemer import set_theme
#set_theme("stylix_transparent")
plt.style.use('dark_background')
        
# ============================================================================
# MAIN ANALYSIS SCRIPT
# ============================================================================

# Setup simulation path
run_name = "m12f"  # Adjust as needed
mainsnapstr = "225"  # Number for which angular momentum and recenter is calculated for 

scratch_path = "/scratch/vasissua/"
run_out_path = os.path.join(scratch_path, "COPY/2026-03/m12f/output_jeans_refinement")

# Load snapshot
#data_dict = utilf.load_snapshot_full(run_path, center_on_stars=True)
kpc_to_au = 206266.3 * 1e3
r_max_au = 1e4 # Radius for which to calculate the angular momentum


#Extract snapshot numbers
snap_nos = sorted([ a.split("_")[1].split(".")[0] 
                    for a in os.listdir(run_out_path) 
                    if "snapshot_" in a and a[-4:] == "hdf5" ])
    
    
def plot_faceon_and_edgeon_for(run_out_path, snapstr, out_path,
                               plot_quantities, mainsnap=True,
                               external_data=None):
    snap_hdf5 = "snapshot_" + snapstr + ".hdf5"
    #run_path = os.path.join(run_out_path, run_name, snap_hdf5) # SHIVAN2 PATH
    snap_hdf5 = "snapshot_" + snapstr + ".hdf5"
    run_path = os.path.join(run_out_path, snap_hdf5)

    print(f"Analysis for snapshot: {run_path}")
    print("─"*80)
    
    extra_rotations = [0, np.pi/2]
    suffixes = ["faceon", "edgeon"]
    data_dicts, external_data = sfp.setup_meshoid(run_path,
                                                  center_type="potential",
                                                  rotate_type="L",
                                                  L_calc_radius=r_max_au /
                                                  kpc_to_au, recenter=True,
                                                  calculate_h2_quantities=False,
                                                  extra_rotation=extra_rotations,
                                                  mainsnap=mainsnap,
                                                  external_data=external_data)

    if type(data_dicts) != list:
        print(f"WARNING: Only a single element found in data_dicts, setup_meshoid output!")
        data_dicts = [data_dicts]

    if len(data_dicts) != len(suffixes):
        raise ValueError(f"Dataset mismatch: {len(data_dicts)} elements in data_dicts, and {len(suffixes)} in suffixes!")

    for data_dict, suffix in zip(data_dicts, suffixes):
        print(f"Data loaded successfully!")
        print("─"*80)
        
        #pdata = data_dict["pdata"]
        #center = data_dict["center"]
        #l_check = sfp.angular_momentum(pdata["Coordinates"], pdata["Velocities"],
        #                           center, r_max_au / kpc_to_au)
        #l_check = l_check / np.linalg.norm(l_check)
        #print(f"L after rotation (should be [0,0,1]): {l_check}")
        
        #================================================
        
        def plot_value(plot_quantity):
            
            plot_quantity_str = plot_quantity
            if plot_quantity == None:
                plot_quantity_str = "density"
            plot_quantity_str = plot_quantity_str.lower()
            print(f"Printing {plot_quantity_str} with suffix {suffix}...")

            fig = sfp.plot_zooms(data_dict, plot_quantity=plot_quantity, 
                                 xplots=4, yplots=2, init_auscale=10, init_pcscale=5, 
                                 projection=False, mainsnap=mainsnap)
            #display(fig)
            
            # Set up filenames for plots
            outname = plot_quantity_str + "_" + suffix + ".png"
            out_save_path = os.path.join(out_path, "8x_zoom_m12f", snapstr)
            out_save_file = os.path.join(out_save_path, outname)

            os.makedirs(out_save_path, exist_ok=True)
            fig.savefig(out_save_file)
            print(f"SUCCESS: Saved file at: {out_save_file}")
            plt.close(fig)
        
        for plot_quantity in plot_quantities:
            plot_value(plot_quantity)
    return external_data
        #plot_value("Potential")


plot_quantities = [ None, "Potential" ]
out_path = os.path.join(scratch_path, "SHIVAN", "analysis")
#external_data = plot_faceon_and_edgeon_for(run_out_path, mainsnapstr, out_path,
#                                           plot_quantities, mainsnap=True)
#snap_nos.remove(mainsnapstr)
for snapstr in snap_nos:
    plot_faceon_and_edgeon_for(run_out_path, snapstr, out_path,
                               plot_quantities, mainsnap=True) #, external_data=external_data)
