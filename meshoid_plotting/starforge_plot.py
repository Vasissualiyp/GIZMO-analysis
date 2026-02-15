#!/usr/bin/env python
"""
Example script showing how to use starforge_movies.py code to plot 
density and stars for a single snapshot.

Usage:
    python plot_single_snapshot.py <snapshot_path> [options]
    
Options:
    snapshot_path              Path to snapshot file (e.g., /path/to/snapshot_010.hdf5)
    --output-dir=<dir>        Directory to save output [default: ./]
    --box-size=<size>          Fraction of BoxSize to show [default: 0.4]
    --resolution=<res>        Resolution of surface density map [default: 1000]
    --vmin=<vmin>             Minimum value for colorbar [default: 1]
    --vmax=<vmax>             Maximum value for colorbar [default: 2000]
    --center-on-stars         Center on stars instead of box center [default: False]
"""

import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src/'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../scripts/movies/'))

from docopt import docopt
import h5py
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import colorcet as cc
matplotlib.use('Agg')
from matplotlib import colors
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib.cm import get_cmap
from meshoid import Meshoid

def calculate_h2_rates(
    T, # Temperature
    nH_cgs, # Total H number density in cm^-3
    xH0, # Neutral H fraction
    xe, # Free electron fraction
    xHp, # Ionized H fraction
    nHe0, 
    nHep, 
    nHepp, 
    metallicity_solar, # Z/Z_sun
    urad_G0, # strength ouf the local FUV radiation in Habing units
    xi_cr_h2, # Cosmic ray ionizaton rate
    clumping_factor=1.0,
    Tdust=None
):
    """
    Calculates H2 formation/dissociation rates based on FIRE-2/3 physics.
    
    Returns a dict containing individual coefficients and the net rate dfH2/dt.
    """
    sqrt_T = np.sqrt(T)
    log_T = np.log10(T)
    ln_T = np.log(T)
    nH0 = xH0 * nH_cgs
    
    # 1. Clumping Factors
    # Note: the code uses clumping_factor^3 for 3-body processes
    c_fac = clumping_factor
    c_fac3 = clumping_factor**3

    # 2. Critical Densities for Collisional Dissociation (Glover & Abel 2008)
    logT4 = log_T - 4.0
    ncr_H = 10**(3.0 - 0.416*logT4 - 0.327*logT4**2)
    ncr_H2 = 10**(4.845 - 1.3*logT4 + 1.62*logT4**2)
    ncr_He = 10**(5.0792 * (1. - 1.23e-5 * (T - 2000.)))
    
    # Simple guess for LTE vs non-LTE interpolation
    # (Assuming xH ~ 1, xH2 ~ 0 for the rate coefficient itself)
    ncrit = 1.0 / (1.0/ncr_H + 0.0/ncr_H2 + 1.0/ncr_He) 
    n_ncrit = nH_cgs / ncrit
    f_v0_LTE = 1. / (1. + n_ncrit)
    f_LTE_v0 = 1. - f_v0_LTE

    # 3. Formation: Dust Surface (a_Z)
    if Tdust is None:
        Tdust = 30.0 # Default fallback
    
    a_Z_coeff = (3.e-18 * sqrt_T / 
                ((1. + 4.e-2*np.sqrt(T + Tdust) + 2.e-3*T + 8.e-6*T**2) * (1. + 1.e4/np.exp(np.clip(600./Tdust, None, 90.)))))
    a_Z = a_Z_coeff * metallicity_solar * nH0 * c_fac

    # 4. Formation: Gas Phase (H- process)
    # Simplified equilibrium H- calculation
    lnTeV = ln_T - 9.35915
    # k1: H + e -> H- + photon
    if T <= 6000.:
        k1 = -17.845 + 0.762*log_T + 0.1523*log_T**2 - 0.03274*log_T**3
    else:
        k1 = -16.420 + 0.1998*log_T**2 - 5.447e-3*log_T**4 + 4.0415e-5*log_T**6
    k1 = 10**np.maximum(k1, -50.)
    
    # k2: H- + H -> H2 + e
    k2 = 1.5e-9 if T <= 300. else 4.0e-9 * T**-0.17
    
    # In equilibrium, x_Hminus ~ (k1 * xH0 * xe) / (k2 * xH0 + ...)
    # Here we simplify to the primary formation term
    a_GP = k2 * (k1 * xH0 * xe / k2) * nH0 * c_fac 

    # 5. Formation: 3-Body
    b_3B = (6.0e-32/np.sqrt(sqrt_T) + 2.0e-31/sqrt_T) * nH0 * nH0 * xH0 * c_fac3

    # 6. Dissociation: Collisional (H2 + X)
    # H2 + H
    b_H2HI_v0 = 6.67e-12 * sqrt_T * np.exp(-np.clip(1. + 63593./T, None, 90.))
    b_H2HI_LTE = 3.52e-9 * np.exp(-np.clip(43900./T, None, 90.))
    b_H2HI = 10**(f_v0_LTE*np.log10(b_H2HI_v0) + f_LTE_v0*np.log10(b_H2HI_LTE)) * (xH0*nH_cgs) * c_fac

    # H2 + H+
    b_H2Hp = np.maximum(0., -3.323e-7 + 3.373e-7*ln_T - 1.449e-7*ln_T**2 + 3.417e-8*ln_T**3 
                       - 4.781e-9*ln_T**4 + 3.973e-10*ln_T**5 - 1.817e-11*ln_T**6 
                       + 3.531e-13*ln_T**7) * np.exp(-np.clip(21237.15/T, None, 90.)) * (xHp*nH_cgs) * c_fac

    # 7. Dissociation: Radiative (LW) & Cosmic Rays
    G_LW = 3.3e-11 * urad_G0 * 0.5 # 0.5 factor because solving for mass fraction fH2
    xi_cr = xi_cr_h2 * 0.5

    # Net Rate Calculation (dfH2/dt)
    # Following the comment: dot[nH2]/nH0
    # term1: Formation
    formation = a_Z + a_GP + b_3B
    # term2: Destruction (Linear in fH2)
    destruction = (b_H2HI*0.5 + b_H2Hp*0.5 + G_LW + xi_cr)
    
    return {
        "rate_net": formation - destruction, # This assumes fH2 is small/initial
        "formation_dust": a_Z,
        "formation_gas_phase": a_GP,
        "formation_3body": b_3B,
        "dissoc_collisional_HI": b_H2HI * 0.5,
        "dissoc_collisional_Hplus": b_H2Hp * 0.5,
        "dissoc_photodiss_LW": G_LW,
        "dissoc_cosmic_rays": xi_cr
    }

def extract_gas_parameters(snapshot_path, calculate_h2_quantities=False):
    with h5py.File(snapshot_path, 'r') as F:
        # Load standard fields provided in your example
        pdata = {}
        gas = F["PartType0"]
        
        # Core fields
        fields = ["Masses", "Coordinates", "SmoothingLength", 
                  "MolecularMassFraction", "Temperature", "Density"]
        if calculate_h2_quantities: 
            fields = ["Masses", "Coordinates", "SmoothingLength", 
                      "InternalEnergy", "Density", "Temperature", 
                      "NeutralHydrogenAbundance", "ElectronAbundance", 
                      "HII", "Dust_Temperature", "SoundSpeed", "Velocities"]
        for field in fields:
            if field in gas:
                pdata[field] = gas[field][:]
        
        # Load Header for unit conversions
        header = F['Header'].attrs
        for key in header.keys():
            pdata[key] = header[key]
            
        # --- Handle Missing/Multidimensional Fields ---
        
        if calculate_h2_quantities:
            # 1. Metallicity (Index 0 is usually total Z)
            if "Metallicity" in gas:
                # Shape is (N, 11) or (N, 15). [0] is total, [1] is He, [2] is C...
                metallicity_all = gas["Metallicity"][:]
                pdata["Z_total"] = metallicity_all[:, 0]
                pdata["xHe_total"] = metallicity_all[:, 1] # Mass fraction of Helium
                
            # 2. Photon Flux (LW Band)
            # In FIRE, LW is often in a specific bin of PhotonFluxDensity 
            # (Usually bin 0 or 1 depending on the RT_LYMAN_WERNER flag)
            if "PhotonFluxDensity" in gas:
                pdata["LW_Flux"] = gas["PhotonFluxDensity"][:, 0] # Check your specific bin mapping
                
            # 3. Unit Conversions (Snapshot units to CGS)
            # Note: GIZMO units are typically 1e10 M_sun, kpc, and km/s
            h = pdata.get('HubbleParam', 1.0)
            a = pdata.get('Time', 1.0) # Expansion factor
            
            # Conversion to nH (cm^-3)
            # Assuming typical GIZMO unit_density = 6.77e-22 g/cm^3 / h^2
            # This part requires your specific All.UNIT_DENSITY_IN_NHCGS constant
            # For standard FIRE-2:
            unit_density_cgs = 6.767e-22 # g/cm^3
            mp = 1.67e-24 # proton mass
            xh_mass_frac = 0.76 # typical hydrogen mass fraction
            
            # Physical density in nH [cm^-3]
            pdata["nH_cgs"] = (pdata["Density"] * (unit_density_cgs * h**2 / a**3)) * xh_mass_frac / mp
        
    return pdata

def load_snapshot_data(snapshot_path, load_parttype4=False, 
                       calculate_h2_quantities=False):
    """
    Load snapshot data from an HDF5 file.
    
    Returns:
        pdata: Gas particle data dictionary
        star_data: STARFORGE stars (PartType5)
        fire_star_data: FIRE stars (PartType4)
    """
    
    with h5py.File(snapshot_path, 'r') as F:
        # Load gas particle data (PartType0)
        pdata = extract_gas_parameters(snapshot_path, calculate_h2_quantities)
        
        # Add header info
        for key in F['Header'].attrs.keys():
            pdata[key] = F['Header'].attrs[key]
        
        # Load STARFORGE stars (PartType5)
        star_data = {}
        if 'PartType5' in F.keys() and len(F['PartType5']) > 0:
            for field in ["Masses", "Coordinates"]:
                if field in F["PartType5"]:
                    star_data[field] = F["PartType5"][field][:]
        
        # Load FIRE stars (PartType4)
        fire_star_data = {}
        if load_parttype4:
            if 'PartType4' in F.keys() and len(F['PartType4']) > 0:
                for field in ["Masses", "Coordinates"]:
                    if field in F["PartType4"]:
                        fire_star_data[field] = F["PartType4"][field][:]
    
    return pdata, star_data, fire_star_data

def calculate_h2_rates_vectorized(pdata, G0_field=None, xi_cr=None):
    """
    Calculates H2 rates for all particles in the snapshot simultaneously.
    
    Args:
        pdata (dict): Dictionary of numpy arrays from the h5py extractor.
        G0_field (array/float): Habing field. If None, uses LW_Flux if available.
        xi_cr (array/float): CR ionization rate. If None, uses FIRE default.
    """
    T = pdata["Temperature"]
    nH_cgs = pdata["nH_cgs"]
    xH0 = pdata["NeutralHydrogenAbundance"]
    xe = pdata["ElectronAbundance"]
    xHp = pdata["HII"]
    Z_solar = pdata["Z_total"] / 0.02  # Assuming 0.02 is solar Z
    Tdust = pdata.get("Dust_Temperature", np.full_like(T, 30.0))
    
    # 1. Precompute temperature-dependent terms
    sqrt_T = np.sqrt(T)
    log_T = np.log10(T)
    ln_T = np.log(T)
    nH0 = xH0 * nH_cgs
    
    # 2. Clumping Factor (Simple estimate if not in snapshot)
    # FIRE often uses (1 + 0.5 * Mach^2). 
    # Here we assume 1.0 unless you've calculated it from Velocities/SoundSpeed.
    c_fac = 1.0 
    c_fac3 = 1.0

    # 3. Collisional Dissociation Coefficients (GA08)
    logT4 = log_T - 4.0
    ncr_H = 10**(3.0 - 0.416*logT4 - 0.327*logT4**2)
    ncr_He = 10**(5.0792 * (1. - 1.23e-5 * (T - 2000.)))
    
    # Assuming xH~1, xHe~0.1 for critical density calc
    ncrit = 1.0 / (1.0/ncr_H + 0.1/ncr_He) 
    n_ncrit = nH_cgs / ncrit
    f_v0_LTE = 1. / (1. + n_ncrit)
    f_LTE_v0 = 1. - f_v0_LTE

    # 4. Formation: Dust Surface (a_Z)
    exp_term = np.exp(np.clip(600./Tdust, None, 90.))
    a_Z_coeff = (3.e-18 * sqrt_T / 
                ((1. + 4.e-2*np.sqrt(T + Tdust) + 2.e-3*T + 8.e-6*T**2) * (1. + 1.e4/exp_term)))
    a_Z = a_Z_coeff * Z_solar * nH0 * c_fac

    # 5. Formation: Gas Phase (H- process equilibrium)
    # k1: H + e -> H- + photon
    k1 = np.where(T <= 6000.,
                  10**(-17.845 + 0.762*log_T + 0.1523*log_T**2 - 0.03274*log_T**3),
                  10**(-16.420 + 0.1998*log_T**2 - 5.447e-3*log_T**4 + 4.0415e-5*log_T**6))
    k1 = np.maximum(k1, 1e-50)
    k2 = np.where(T <= 300., 1.5e-9, 4.0e-9 * T**-0.17)
    
    # Equilibrium H- approximation: a_GP ~ k1 * xH0 * xe * nH0
    a_GP = k1 * xH0 * xe * nH0 * c_fac 

    # 6. Formation: 3-Body
    b_3B = (6.0e-32/np.sqrt(sqrt_T) + 2.0e-31/sqrt_T) * nH0 * nH0 * xH0 * c_fac3

    # 7. Dissociation: Collisional
    # H2 + H (HI)
    b_H2HI_v0 = 6.67e-12 * sqrt_T * np.exp(-np.clip(1. + 63593./T, None, 90.))
    b_H2HI_LTE = 3.52e-9 * np.exp(-np.clip(43900./T, None, 90.))
    b_H2HI = 10**(f_v0_LTE*np.log10(b_H2HI_v0) + f_LTE_v0*np.log10(b_H2HI_LTE)) * (xH0*nH_cgs) * c_fac

    # 8. Dissociation: Radiative & CR
    if G0_field is None:
        G0_field = pdata.get("LW_Flux", 1e-10) # Fallback to a floor
    
    G_LW = 3.3e-11 * G0_field * 0.5
    xi_cr_val = 7.525e-16 * 0.5 if xi_cr is None else xi_cr * 0.5

    # 9. Pack results into a dictionary of arrays
    results = {
        "formation_dust": a_Z,
        "formation_gas_phase": a_GP,
        "formation_3body": b_3B,
        "dissoc_collisional_HI": b_H2HI * 0.5,
        "dissoc_photodiss_LW": G_LW,
        "dissoc_cosmic_rays": xi_cr_val,
        "total_formation": a_Z + a_GP + b_3B,
        "total_destruction_coeff": (b_H2HI * 0.5 + G_LW + xi_cr_val)
    }
    
    return results

def plot_h2_rate_map(data_dict, rate_key="total_formation", output_dir='./', 
                     box_size=0.4, resolution=1000, pc_scale=0, au_scale=0, plot_fire_stars=False):
    """
    Plots a mass-weighted map of a specific H2 rate/coefficient.
    """
    M = data_dict["M"]
    center = data_dict["center"]
    pdata_boxsize = data_dict["boxsize"]
    snapshot_path = data_dict["snapshot_path"]
    h2_results = data_dict["h2_results"]
    
    actual_box_size = pdata_boxsize * box_size
    
    # 1. Select the quantity to plot from your results dict
    if rate_key not in h2_results:
        print(f"Key {rate_key} not found in h2_results!")
        return
    
    quantity = h2_results[rate_key]
    
    # 2. Use meshoid to calculate the mass-weighted projected average
    # This represents <Rate>_mass along the line of sight
    rate_map = M.ProjectedAverage(quantity, center=center, size=actual_box_size, res=resolution)
    
    # Grid for plotting
    min_pos = center - actual_box_size / 2
    max_pos = center + actual_box_size / 2
    X = np.linspace(min_pos[0], max_pos[0], resolution)
    Y = np.linspace(min_pos[1], max_pos[1], resolution)
    X, Y = np.meshgrid(X, Y, indexing='ij')

    # Setup Figure
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Plot using LogNorm (rates vary by many orders of magnitude)
    # We use a different colormap (e.g., 'viridis' or 'magma') to distinguish from gas density
    cmap = 'cet_linear_kryw_5_100_c75' if 'formation' in rate_key else 'cet_linear_benw_5_100_c86'
    
    pcm = ax.pcolormesh(X, Y, rate_map, 
                        norm=colors.LogNorm(vmin=np.percentile(rate_map, 5), 
                                            vmax=np.percentile(rate_map, 99)), 
                        cmap=cmap)
    
    # Add Colorbar for the rate
    cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(f'{rate_key} (Mass-Weighted Avg)', color='white', size=15)
    cb.ax.yaxis.set_tick_params(color='white', labelcolor='white')

    # Formatting
    ax.set_aspect('equal')
    ax.set_xlim([min_pos[0], max_pos[0]])
    ax.set_ylim([min_pos[1], max_pos[1]])
    ax.axis('off')

    # Scalebar Logic (reusing your snippet)
    vertical_size = actual_box_size / 300
    fontprops = fm.FontProperties(size=15)
    scalebar_pc = AnchoredSizeBar(ax.transData, 1.0, "1 pc", 
                                  loc='upper left', pad=1, color='white', 
                                  frameon=False, size_vertical=vertical_size, 
                                  fontproperties=fontprops)
    ax.add_artist(scalebar_pc)

    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_h2_{rate_key}.png"
    output_path = os.path.join(output_dir, Path(snapshot_path).stem + suffix)
    plt.savefig(output_path, dpi=300, facecolor='black')
    
    return fig

def setup_meshoid(snapshot_path, center_on_stars=False, 
                  calculate_h2_quantities=False):
    """
    Plot surface density and stars for a single snapshot.
    """
    
    # Load data
    print(f"Loading snapshot: {snapshot_path}")
    pdata, star_data, fire_star_data = load_snapshot_data(snapshot_path, True, 
                                                          calculate_h2_quantities)
    
    # Determine center
    if center_on_stars:
        if star_data and 'Coordinates' in star_data:
            star_coords = star_data['Coordinates']
            star_masses = star_data['Masses']
            center = np.average(star_coords, axis=0, weights=star_masses)
            print(f"Centered on STARFORGE stars at {center}")
        elif fire_star_data and 'Coordinates' in fire_star_data:
            fire_star_coords = fire_star_data['Coordinates']
            fire_star_masses = fire_star_data['Masses']
            center = np.average(fire_star_coords, axis=0, weights=fire_star_masses)
            print(f"Centered on FIRE stars at {center}")
        else:
            print("Warning: No stars found, centering on box center")
            box_size_val = pdata['BoxSize']
            center = np.array([box_size_val / 2, box_size_val / 2, box_size_val / 2])
    else:
        box_size_val = pdata['BoxSize']
        center = np.array([box_size_val / 2, box_size_val / 2, box_size_val / 2])
        print(f"Centered on box center at {center}")
    
    # Extract gas particle data
    pos = pdata["Coordinates"]
    mass = pdata["Masses"]
    hsml = pdata["SmoothingLength"]
    if calculate_h2_quantities: h2_results = calculate_h2_rates_vectorized(pdata)
    else: h2_results = {}
    
    # Create Meshoid object for surface density calculation
    print("Creating surface density map...")
    M = Meshoid(pos, mass, hsml)
    dictionary = {
        "M": M,
        "center": center,
        "star_data": star_data,
        "snapshot_path": snapshot_path,
        "fire_star_data": fire_star_data,
        "boxsize": pdata["BoxSize"],
        "h2_results": h2_results,
        "pdata": pdata
    }
    return dictionary
    
def plot_single_snapshot(dictionary, output_dir='./', box_size=0.4, resolution=1000, 
                         vmin=1, vmax=2000, pc_scale=3, au_scale=6, plot_fire_stars=False):
    M = dictionary["M"]
    star_data = dictionary["star_data"]
    fire_star_data = dictionary["fire_star_data"]
    center = dictionary["center"]
    pdata_boxsize = dictionary["boxsize"]
    snapshot_path = dictionary["snapshot_path"]

    
    # Convert box_size from fraction to actual size
    actual_box_size = pdata_boxsize * box_size
    print(f"actual box size: {actual_box_size}")
    # Set up coordinate grid
    min_pos = center - actual_box_size / 2
    max_pos = center + actual_box_size / 2
    X = np.linspace(min_pos[0], max_pos[0], resolution)
    Y = np.linspace(min_pos[1], max_pos[1], resolution)
    X, Y = np.meshgrid(X, Y, indexing='ij')
    
    # Calculate surface density
    sigma_gas = M.SurfaceDensity(M.m, center=center, 
                                 size=actual_box_size, res=resolution)
    # Automatically set the boundaries for gas density
    vmin = np.min(sigma_gas)
    vmax = np.max(sigma_gas)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Plot surface density
    p = ax.pcolormesh(X, Y, sigma_gas, 
                      norm=colors.LogNorm(vmin=vmin, vmax=vmax), 
                      cmap='cet_fire')
    print(f"vmin: {vmin}, vmax: {vmax}")
    
    # Plot STARFORGE stars if available
    if star_data and 'Coordinates' in star_data and len(star_data['Coordinates']) > 0:
        star_coords = star_data['Coordinates']
        star_masses = star_data['Masses']
        marker_size = 20 * (star_masses / np.max(star_masses))
        ax.scatter(star_coords[:, 0], star_coords[:, 1], 
                  c='white', s=marker_size, alpha=0.8, 
                  edgecolors='white', linewidths=0.5, label='STARFORGE stars')
        print(f"Plotted {len(star_coords)} STARFORGE stars")
    
    # Plot FIRE stars if available
    if fire_star_data and 'Coordinates' in fire_star_data \
                      and len(fire_star_data['Coordinates']) > 0 \
                      and plot_fire_stars:
        fire_star_coords = fire_star_data['Coordinates']
        fire_star_masses = fire_star_data['Masses']
        marker_size = 100 * (fire_star_masses / np.max(fire_star_masses))
        ax.scatter(fire_star_coords[:, 0], fire_star_coords[:, 1], 
                  c='cyan', s=marker_size, alpha=0.8, 
                  edgecolors='white', linewidths=0.5, label='FIRE stars')
        print(f"Plotted {len(fire_star_coords)} FIRE stars")
    
    # Set plot properties
    ax.set_aspect('equal')
    ax.set_xlim([min_pos[0], max_pos[0]])
    ax.set_ylim([min_pos[1], max_pos[1]])
    plt.xticks([])
    plt.yticks([])
    
    # Add scalebar
    fontprops = fm.FontProperties(size=20)
    scale_bar_size = actual_box_size / 10
    au_to_pc = 4.848102e-6
    pc_power = pc_scale
    if au_scale > -10:
        au_power = au_scale
        vertical_size = actual_box_size / 300
        scalebar = AnchoredSizeBar(ax.transData,
                                   au_to_pc*10**au_power, 
                                   f"""$10^{{{au_power}}}$ AU""",
                                   loc = 'upper left',
                                   pad=1,
                                   color='white',
                                   frameon=False,
                                   size_vertical=vertical_size,
                                   fontproperties=fontprops)
        ax.add_artist(scalebar)
    scalebar_pc = AnchoredSizeBar(ax.transData,
                               10**pc_power, f"""$10^{{{pc_power}}}$ pc""", 
                               loc = 'upper left',
                               pad=3,
                               color='white',
                               frameon=False,
                               size_vertical=vertical_size,
                               fontproperties=fontprops)
    ax.add_artist(scalebar_pc)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    
    snap_name = Path(snapshot_path).stem + '.png'
    output_path = os.path.join(output_dir, snap_name)
    #plt.savefig(output_path, dpi=300, facecolor='black')
    #print(f"Saved: {output_path}")
    #plt.show()
    return fig



if __name__ == '__main__':
    # For simple usage without docopt
    if len(sys.argv) < 2:
        print("Usage: python plot_single_snapshot.py <snapshot_path> [--output-dir=<dir>] [--box-size=<size>] [--resolution=<res>] [--vmin=<vmin>] [--vmax=<vmax>] [--center-on-stars]")
        sys.exit(1)
    
    snapshot_path = sys.argv[1]
    
    # Parse simple kwargs
    kwargs = {'output_dir': './', 'box_size': 0.4, 'resolution': 1000, 
              'vmin': 1e-4, 'vmax': 1e-3, 'center_on_stars': True, 
              'load_parttype4':True}
    
    for arg in sys.argv[2:]:
        if arg.startswith('--'):
            if '=' in arg:
                key, value = arg[2:].split('=')
                if key == 'center-on-stars':
                    kwargs['center_on_stars'] = True
                elif key in ['resolution', 'vmin', 'vmax']:
                    kwargs[key.replace('-', '_')] = int(value)
                elif key == 'box-size':
                    kwargs['box_size'] = float(value)
                else:
                    kwargs[key.replace('-', '_')] = value
    
    dict = setup_meshoid(snapshot_path, **kwargs)
    plot_single_snapshot(dict, **kwargs)
