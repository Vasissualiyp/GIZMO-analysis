
"""
Jupyter-style analysis script for GIZMO+FIRE+STARFORGE snapshots.
Plots various astrophysical quantities:
- Toomre-Q vs R
- Rotational KE vs total KE (and virial parameter, Mach number, mass-to-flux ratio)
- Turbulent KE power spectrum
- Multiplicity vs Mass
- IMF, SFE, SFR

PartType5: STARFORGE particles (what we care about)
PartType4: FIRE particles (should not be present)
"""

%matplotlib inline
%config InlineBackend.figure_format = 'png'
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

# ============================================================================
# MAIN ANALYSIS SCRIPT
# ============================================================================

# Setup simulation path
run_name = "m12f"  # Adjust as needed
snapstr = "225"  # Adjust snapshot number

scratch_path = "/scratch/vasissua/"
run_out_path = os.path.join(scratch_path, "COPY/2026-03/m12f/output_jeans_refinement")
snap_hdf5 = "snapshot_" + snapstr + ".hdf5"
run_path = os.path.join(run_out_path, snap_hdf5)

print(f"Analysis for snapshot: {run_path}")
print("="*80)

# Load snapshot
#data_dict = utilf.load_snapshot_full(run_path, center_on_stars=True)
kpc_to_au = 206266.3 * 1e3
r_max_au = 1e4 # Radius for which to calculate the angular momentum

data_dict = sfp.setup_meshoid(run_path, center_type="potential", 
                              rotate_type = "L", L_calc_radius = r_max_au / kpc_to_au,
                              recenter = True, calculate_h2_quantities=False)

print(f"Data loaded successfully!")
print("─"*80)

pdata = data_dict["pdata"]
center = data_dict["center"]
l_check = sfp.angular_momentum(pdata["Coordinates"], pdata["Velocities"],
                           center, r_max_au / kpc_to_au)
l_check = l_check / np.linalg.norm(l_check)
print(f"L after rotation (should be [0,0,1]): {l_check}")

#================================================


from vasthemer import set_theme
set_theme("stylix_transparent")
plt.style.use('dark_background')

fig = sfp.plot_zooms(data_dict, plot_quantity="Potential", 
                     xplots=4, yplots=2, init_auscale=10, init_pcscale=5)
display(fig)

outname = "8x_zoom_m12f_potential.png"
out_save_path = os.path.join(scratch_path, "SHIVAN", "analysis", outname)
fig.savefig(out_save_path)




"""
# ============================================================================
# ANALYSIS 1: Toomre-Q vs R
# ============================================================================
print("\n1. Computing Toomre-Q vs R...")
r_centers, Q, sigma_gas, omega, c_s = compute_toomre_q(data_dict, n_bins=50)
fig_toomre = utilf.plot_toomre_q(r_centers, Q, sigma_gas, 
                           output_path="toomre_q_vs_r.png")
display(fig_toomre)
print(f"   Mean Q = {np.nanmean(Q):.2f}")
print(f"   Stable (Q>1) fraction: {np.sum(Q > 1) / len(Q):.2%}")

# ============================================================================
# ANALYSIS 2: Energy Ratios and Global Properties
# ============================================================================
print("\n2. Computing energy ratios and global properties...")
energy_dict = utilf.compute_energy_ratios(data_dict)
fig_energy = utilf.plot_energy_ratios(energy_dict, 
                                output_path="energy_ratios.png")
#display(fig_energy)
print(f"   Rotational KE fraction: {energy_dict['KE_rotational']/energy_dict['KE_total']:.3f}")
print(f"   Virial parameter: {energy_dict['virial_parameter']:.3f}")
print(f"   Mach number: {energy_dict['mach_number']:.3f}")

# ============================================================================
# ANALYSIS 3: Mass-to-Flux Ratio
# ============================================================================
print("\n3. Computing mass-to-flux ratio...")
mtf_dict = utilf.compute_mass_to_flux_ratio(data_dict)
if mtf_dict:
    print(f"   Dimensionless mass-to-flux ratio (mu): {mtf_dict['mu']:.3f}")
    print(f"   (Compare with Yasmine's paper)")
else:
    print("   No magnetic field data available!")

# ============================================================================
# ANALYSIS 4: Turbulent Power Spectrum
# ============================================================================
print("\n4. Computing turbulent power spectrum...")
k, power = utilf.compute_turbulent_power_spectrum(data_dict, grid_size=128)
fig_ps = utilf.plot_power_spectrum(k, power, 
                             output_path="turbulent_power_spectrum.png")
#display(fig_ps)

# ============================================================================
# ANALYSIS 5: Stellar Properties (IMF, SFE, SFR, Multiplicity)
# ============================================================================
print("\n5. Computing stellar properties...")
stellar_props = utilf.compute_stellar_properties(data_dict)
if stellar_props:
    fig_imf = utilf.plot_imf(stellar_props, output_path="imf.png")
    #display(fig_imf)
    
    fig_stellar = utilf.plot_stellar_summary(stellar_props, 
                                      output_path="stellar_summary.png")
    #display(fig_stellar)
    
    print(f"   Number of stars: {stellar_props['n_stars']}")
    print(f"   Total stellar mass: {stellar_props['total_stellar_mass']:.2e} M_sun")
    print(f"   SFE: {stellar_props['star_formation_efficiency']:.3f}")
    print(f"   Multiplicity fraction: {stellar_props['multiplicity_fraction']:.3f}")
    if stellar_props['star_formation_rate']:
        print(f"   SFR: {stellar_props['star_formation_rate']:.3e} stars/Myr")

print("\n" + "="*80)
print("Analysis complete! Plots saved to current directory.")
print("="*80)
"""

