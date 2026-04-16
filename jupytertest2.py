
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
import meshoid_plotting.notebook_method as nbm
#from jupytertest import plot_zooms

importlib.reload(sfp)
importlib.reload(utilf)
importlib.reload(nbm)

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

# ALL DISTANCES BELOW ARE IN KPC (GIZMO UNITS)
# Define list of radii for angular momentum calculation (in kpc)
# This will test which radius gives the most stable L vector
r_max_kpc = 1e-3
L_radii_list = [
    r_max_kpc * 1e-2,  # 0.01x
    r_max_kpc * 5e-2,  # 0.05x
    r_max_kpc * 1e-1,  # 0.1x
    r_max_kpc * 5e-1,  # 0.5x
    r_max_kpc * 1e0 ,  # 1.0x
    r_max_kpc * 5e0 ,  # 5x
    r_max_kpc * 1e1 ,  # 10x
    r_max_kpc * 5e1 ,  # 50x
    r_max_kpc * 1e2 ,  # 100x
    r_max_kpc * 5e2 ,  # 100x
]


#Extract snapshot numbers
snap_nos = sorted([ a.split("_")[1].split(".")[0] 
                    for a in os.listdir(run_out_path) 
                    if "snapshot_" in a and a[-4:] == "hdf5" ])

snap_nos = snap_nos[::-1] # Reverse the list

def calculate_density_weighted_center(pdata, reference_center=None, search_radius_kpc=None):
    """
    Calculate center of mass weighted by density instead of mass.

    This gives more weight to high-density regions while considering the
    distribution, rather than just picking the single densest particle.

    Parameters:
    -----------
    pdata : dict
        Gas particle data containing 'Coordinates' and 'Density'
    reference_center : ndarray, shape (3,), optional
        Reference point for spatial filtering. If provided with search_radius_kpc,
        only considers particles within search_radius_kpc of this point.
    search_radius_kpc : float, optional
        Radius around reference_center to consider for calculation.

    Returns:
    --------
    center : ndarray, shape (3,)
        Density-weighted center position in kpc
    """
    pos = pdata['Coordinates']
    density = pdata['Density']

    # Apply spatial filter if reference center is provided
    if reference_center is not None and search_radius_kpc is not None:
        dist_from_ref = np.linalg.norm(pos - reference_center, axis=1)
        mask = dist_from_ref < search_radius_kpc

        if mask.sum() == 0:
            print(f"  WARNING: No gas particles within {search_radius_kpc:.3e} kpc of reference center!")
            print(f"  Falling back to global density-weighted center")
            mask = np.ones(len(pos), dtype=bool)
        else:
            print(f"  Using {mask.sum()} particles within {search_radius_kpc:.3e} kpc of reference")
    else:
        mask = np.ones(len(pos), dtype=bool)

    # Calculate density-weighted center
    # center = sum(density_i * position_i) / sum(density_i)
    pos_masked = pos[mask]
    density_masked = density[mask]

    center = np.sum(density_masked[:, None] * pos_masked, axis=0) / np.sum(density_masked)

    print(f"  Density-weighted center at {center}")
    print(f"  Total density weight: {np.sum(density_masked):.3e}")

    return center

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

    sphere_volumes = 4 / 3 * np.pi * bin_centers**3
    densities = cumulative_mass * 10**10 / sphere_volumes * u.solMass / u.kpc**3
    densities = densities.to(u.g / u.cm**3).value

    # Perform plotting
    fig, ax = plt.subplots()
    ax.set_title(f"Density vs radius. t = {time}")
    ax.plot(np.log10(bin_centers), np.log10(densities))
    ax.set_xlabel("log(r), log(kpc)")
    ax.set_ylabel(r'log($\rho$), log(g/cm$^3$)')
    fig.savefig(out_save_file)
    plt.close(fig)
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
                               external_data=None, precalculated_center=None,
                               precalculated_L=None, L_use_stars=False, L_use_weighted=True,
                               fallback_to_density=True, center_method="potential"):
    snap_hdf5 = "snapshot_" + snapstr + ".hdf5"
    #run_path = os.path.join(run_out_path, run_name, snap_hdf5) # SHIVAN2 PATH
    snap_hdf5 = "snapshot_" + snapstr + ".hdf5"
    run_path = os.path.join(run_out_path, snap_hdf5)

    print(f"Analysis for snapshot: {run_path}")
    print("─"*80)

    # If using density_weighted center and no precalculated center, calculate it
    if center_method == "density_weighted" and precalculated_center is None:
        print(f"Using density-weighted center calculation method")
        # Load snapshot data to calculate center
        pdata, _, _, _ = sfp.load_snapshot_data(run_path, mainsnap, False)
        precalculated_center = calculate_density_weighted_center(
            pdata,
            reference_center=reference_center,
            search_radius_kpc=reference_search_radius_kpc
        )
        print(f"Calculated density-weighted center: {precalculated_center}")

    extra_rotations = [0, np.pi/2]
    suffixes = ["faceon", "edgeon"]
    data_dicts, external_data = sfp.setup_meshoid(run_path,
                                                  center_type="potential" if center_method != "density_weighted" else "potential",
                                                  rotate_type="L",
                                                  L_calc_radius=r_max_kpc,
                                                  recenter=True,
                                                  calculate_h2_quantities=False,
                                                  extra_rotation=extra_rotations,
                                                  mainsnap=mainsnap,
                                                  external_data=external_data,
                                                  precalculated_center=precalculated_center,
                                                  precalculated_L=precalculated_L,
                                                  L_use_stars=L_use_stars,
                                                  L_use_weighted=L_use_weighted,
                                                  fallback_to_density=fallback_to_density)

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
                                 init_boxsize=1e-6, # Length of plotted box in terms of units of total boxsize
                                 xplots=2, yplots=2, init_auscale=6, init_pcscale=1, 
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
smoothing_method = 'polynomial'  # 'gaussian' or 'polynomial'
smoothing_sigma = 15.0  # Gaussian kernel width in snapshot units (only used if smoothing_method='gaussian')
smoothing_sigma_L = 15.0  # Gaussian kernel width for L vector smoothing (only used if smoothing_method='gaussian')
polynomial_degree = 1  # Polynomial degree for fitting (1=linear, 2=quadratic, etc.) (only used if smoothing_method='polynomial')
polynomial_degree_L = 1  # Polynomial degree for L vector fitting (only used if smoothing_method='polynomial')
L_radius_index = 0  # Index of radius to use for L vector (see L_radii_list above)
calculate_centers_only = False  # Set to True to only calculate centers (first pass only)
recalculate_centers = True  # Set to True to recalculate centers from scratch
L_use_stars = False  # Set to True to calculate angular momentum from STARFORGE stars (PartType5) instead of gas
L_use_weighted = False  # Set to True to use weighted angular momentum (density & potential). Default: True
skip_snaps_for_center_calc = 0 # How many snapshots to skip for center calculations
fallback_to_density = True  # Set to True to use densest gas when no stars (notebook method, more stable). Set to False for extrapolation (old method)
center_method = "potential"  # Method for center calculation: "potential" (default), "density_weighted" (new), or use center_type in setup_meshoid

# ============================================================================
# IMPORTANT: Reference center for full simulation data
# ============================================================================
# If using FULL simulation data (not cutouts), you MUST specify a reference center!
# The notebook works because it uses spatial cutouts - only particles near the region
# of interest are in the data. With full simulation data, the "densest gas" might
# be anywhere in the galaxy, not in your refinement region.
#
# Set this to the approximate center of your refinement region (in kpc).
# You can find this by:
# 1. Looking at the InitCondFile or IC setup for your zoom-in
# 2. Looking at where sinks form at late times
# 3. Looking at the high-density region in your visualization
#
# Set to None if using cutout data (like the notebook's output_cutout).
reference_center = None  # e.g., np.array([41.75, 44.22, 46.01])  # kpc - SET THIS!
reference_search_radius_kpc = 0.1  # Search for densest gas within this radius of reference_center

if calculate_centers_only:
    # Only calculate and save centers (useful for long runs)
    # Use pure notebook method
    nbm.calculate_all_centers_notebook(
        run_out_path, snap_nos, r_search_kpc=r_max_kpc,
        out_path=out_path, run_subdir="8x_zoom_m12f",
        L_radii_list=L_radii_list,
        reference_center=reference_center,
        search_radius_kpc=reference_search_radius_kpc
    )
    print("\nCenter calculation complete. Set calculate_centers_only=False to plot with smoothing.")
    exit(0)

if use_smoothing:
    # Check if we need to calculate centers or load from files
    if recalculate_centers:
        # First pass: Calculate and save all centers to files
        # Use pure notebook method
        nbm.calculate_all_centers_notebook(
            run_out_path, snap_nos[skip_snaps_for_center_calc:],
            r_search_kpc=r_max_kpc,
            out_path=out_path, run_subdir="8x_zoom_m12f",
            L_radii_list=L_radii_list,
            reference_center=reference_center,
            search_radius_kpc=reference_search_radius_kpc
        )

    # Load centers from saved files
    centers, velocities, times, valid_snap_nos_centers = sfp.load_centers_from_files(
        out_path, snap_nos, run_subdir="8x_zoom_m12f"
    )

    # Load angular momentum vectors from saved files
    L_vectors, L_radii, valid_snap_nos_L = sfp.load_angular_momentum_vectors(
        out_path, snap_nos, run_subdir="8x_zoom_m12f"
    )
    
    # Ensure both have the same valid snapshots
    if valid_snap_nos_centers != valid_snap_nos_L:
        print("Warning: Centers and L vectors have different valid snapshots")
        # Use intersection of both
        valid_snap_nos = [s for s in valid_snap_nos_centers if s in valid_snap_nos_L]
        print(f"Using intersection: {len(valid_snap_nos)} snapshots")
    else:
        valid_snap_nos = valid_snap_nos_L
    
    # Update snap_nos to only include valid snapshots for plotting
    snap_nos = valid_snap_nos
    print(f"\nProcessing {len(snap_nos)} snapshots with valid data")

    # Extract L vectors for the selected radius
    print(f"\nUsing L vectors from radius index {L_radius_index}: R = {L_radii[L_radius_index]:.3e} kpc")
    L_vectors_selected = L_vectors[:, L_radius_index, :]  # Shape: (n_snapshots, 3)

    # Smooth centers using selected method
    if smoothing_method == 'polynomial':
        print(f"\nSmoothing centers with polynomial fit (degree={polynomial_degree})...")
        smoothed_centers = sfp.fit_polynomial_trajectory(centers, times=times, degree=polynomial_degree)
    else:
        print(f"\nSmoothing centers with Gaussian filter (sigma={smoothing_sigma})...")
        smoothed_centers = sfp.smooth_vector_evolution(centers, sigma=smoothing_sigma)

    # Smooth L vectors using selected method
    if smoothing_method == 'polynomial':
        print(f"\nSmoothing L vectors with polynomial fit (degree={polynomial_degree_L})...")
        smoothed_L = sfp.fit_polynomial_trajectory(L_vectors_selected, times=times, degree=polynomial_degree_L)
    else:
        print(f"\nSmoothing L vectors with Gaussian filter (sigma={smoothing_sigma_L})...")
        smoothed_L = sfp.smooth_vector_evolution(L_vectors_selected, sigma=smoothing_sigma_L)

    print("\n" + "="*80)
    print("SECOND PASS: Plotting with smoothed centers and L vectors...")
    print("="*80)

    # Second pass: Plot with smoothed centers and L vectors
    for i, snapstr in enumerate(snap_nos):
        external_data = plot_faceon_and_edgeon_for(
            run_out_path, snapstr, out_path, plot_quantities,
            mainsnap=True, external_data=external_data,
            precalculated_center=smoothed_centers[i],
            precalculated_L=smoothed_L[i],
            L_use_stars=L_use_stars,
            L_use_weighted=L_use_weighted,
            fallback_to_density=fallback_to_density,
            center_method=center_method
        )
else:
    # Single pass without smoothing (original behavior)
    for snapstr in snap_nos:
        external_data = plot_faceon_and_edgeon_for(
            run_out_path, snapstr, out_path, plot_quantities,
            mainsnap=True, external_data=external_data,
            L_use_stars=L_use_stars,
            L_use_weighted=L_use_weighted,
            fallback_to_density=fallback_to_density,
            center_method=center_method
        )
