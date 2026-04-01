
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
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo


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
mainsnapstr = "225"  # Number for which angular momentum and recenter is calculated for 

scratch_path = "/scratch/vasissua/"
run_out_path = os.path.join(scratch_path, "COPY/2026-03/m12f/output_jeans_refinement")

# Load snapshot
#data_dict = utilf.load_snapshot_full(run_path, center_on_stars=True)
kpc_to_au = 206266.3 * 1e3
r_max_au = 1e5 # Radius for which to calculate the angular momentum


#Extract snapshot numbers
snap_nos = sorted([ a.split("_")[1].split(".")[0] 
                    for a in os.listdir(run_out_path) 
                    if "snapshot_" in a and a[-4:] == "hdf5" ])

snap_nos = snap_nos[::-1] # Reverse the list 

def plot_mass_vs_r(data_dict, num_bins, out_save_path):
    pdata = data_dict["pdata"]
    center = data_dict["center"]
    pos = pdata["Coordinates"]
    mass = pdata["Masses"]

    time = cosmo.age(pdata["Redshift"]).to(u.kyr)

    # Relative distances
    r_vec = pos - center
    dist = np.linalg.norm(r_vec, axis=1)
    
    # Define log bins based on data range
    rmin, rmax = dist[dist > 0].min(), dist.max()
    bins = np.logspace(np.log10(rmin), np.log10(rmax), num_bins + 1)
    
    # Digitize distances into bins
    indices = np.digitize(dist, bins)
    
    # Sum masses in each bin
    bin_masses = np.array([mass[indices == i].sum() for i in range(1, len(bins))])
    
    # Cumulative mass M(<r)
    cumulative_mass = np.cumsum(bin_masses)
    
    # Bin centers for plotting
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    out_save_file = os.path.join(out_save_path, "m_vs_r.png")
    data_save_file = os.path.join(out_save_path, "m_vs_r.txt")
    center_save_file = os.path.join(out_save_path, "center.txt")
    l_save_file = os.path.join(out_save_path, "angular_momentum.txt")

    plt.title(f"Cumulative mass vs radius. t = {time}")
    plt.plot(np.log10(bin_centers), np.log10(cumulative_mass/10**10))
    plt.xlabel("log(r), log(kpc)")
    plt.ylabel("log(M), log(Msun)")
    plt.savefig(out_save_file)
    plt.close()

    print(f"Saved M vs R plot in {out_save_file}")
    data_to_save = np.column_stack((bin_centers, cumulative_mass/10**10))
    np.savetxt(data_save_file, data_to_save,
           header="radius_kpc cumulative_mass_msun",
           fmt="%.8e")
    # Save original center (before recentering) so it's not [0,0,0]
    center_to_save = data_dict.get("original_center", data_dict["center"])
    np.savetxt(center_save_file, center_to_save)
    np.savetxt(l_save_file, data_dict["L"])
    print(f"Saved M vs R data in {data_save_file}")
    print(f"Saved center at {center_to_save} in {center_save_file}")
    

def plot_faceon_and_edgeon_for(run_out_path, snapstr, out_path,
                               plot_quantities, mainsnap=True,
                               external_data=None, precalculated_center=None):
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
                                                  external_data=external_data,
                                                  precalculated_center=precalculated_center)

    if type(data_dicts) != list:
        print(f"WARNING: Only a single element found in data_dicts, setup_meshoid output!")
        data_dicts = [data_dicts]

    if len(data_dicts) != len(suffixes):
        raise ValueError(f"Dataset mismatch: {len(data_dicts)} elements in data_dicts, and {len(suffixes)} in suffixes!")

    num_bins = 100
    out_save_path = os.path.join(out_path, "8x_zoom_m12f", snapstr)
    os.makedirs(out_save_path, exist_ok=True)
    plot_mass_vs_r(data_dicts[0], num_bins, out_save_path)

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
            print(f"Plotting {plot_quantity_str} with suffix {suffix}...")

            fig = sfp.plot_zooms(data_dict, plot_quantity=plot_quantity, 
                                 xplots=4, yplots=2, init_auscale=10, init_pcscale=5, 
                                 projection=False, mainsnap=mainsnap)
            #display(fig)
            
            # Set up filenames for plots
            outname = plot_quantity_str + "_" + suffix + ".png"
            out_save_file = os.path.join(out_save_path, outname)

            fig.savefig(out_save_file)
            print(f"SUCCESS: Saved file at: {out_save_file}")
            plt.close(fig)
        
        #if not mainsnap:
        for plot_quantity in plot_quantities:
            plot_value(plot_quantity)
    return external_data
        #plot_value("Potential")


plot_quantities = [ None, "Potential" ]
out_path = os.path.join(scratch_path, "SHIVAN", "analysis")
external_data = None
#snap_nos.remove(mainsnapstr)
#exit(0)

# If restarting, set first snapshot to restart from
final_snap = len(snap_nos)
first_snap = final_snap
delta_snap = final_snap - first_snap
snap_nos = snap_nos[delta_snap:]

# Two-pass approach with smoothing (set to False to disable)
use_smoothing = True
smoothing_sigma = 2.0  # Gaussian kernel width in snapshot units
calculate_centers_only = False  # Set to True to only calculate centers (first pass only)
recalculate_centers = True  # Set to True to recalculate centers from scratch

if calculate_centers_only:
    # Only calculate and save centers (useful for long runs)
    sfp.calculate_all_centers(
        run_out_path, snap_nos, center_type="potential",
        L_calc_radius=r_max_au, kpc_to_au=kpc_to_au,
        out_path=out_path, run_subdir="8x_zoom_m12f"
    )
    print("\nCenter calculation complete. Set calculate_centers_only=False to plot with smoothing.")
    exit(0)

if use_smoothing:
    # Check if we need to calculate centers or load from files
    if recalculate_centers:
        # First pass: Calculate and save all centers to files
        sfp.calculate_all_centers(
            run_out_path, snap_nos, center_type="potential",
            L_calc_radius=r_max_au, kpc_to_au=kpc_to_au,
            out_path=out_path, run_subdir="8x_zoom_m12f"
        )

    # Load centers from saved files
    centers, velocities, times = sfp.load_centers_from_files(
        out_path, snap_nos, run_subdir="8x_zoom_m12f"
    )

    # Smooth centers
    smoothed_centers = sfp.smooth_centers(centers, sigma=smoothing_sigma)

    print("\n" + "="*80)
    print("SECOND PASS: Plotting with smoothed centers...")
    print("="*80)

    # Second pass: Plot with smoothed centers
    for i, snapstr in enumerate(snap_nos):
        external_data = plot_faceon_and_edgeon_for(
            run_out_path, snapstr, out_path, plot_quantities,
            mainsnap=True, external_data=external_data,
            precalculated_center=smoothed_centers[i]
        )
else:
    # Single pass without smoothing (original behavior)
    for snapstr in snap_nos:
        external_data = plot_faceon_and_edgeon_for(
            run_out_path, snapstr, out_path, plot_quantities,
            mainsnap=True, external_data=external_data
        )
