#!/r/bin/env python
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

import sys, os, time, h5py, matplotlib

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src/'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../scripts/movies/'))

from docopt import docopt
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import colorcet as cc
matplotlib.use('Agg')
from matplotlib import colors
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib.cm import get_cmap
from meshoid import Meshoid
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from pytreegrav import Potential
from scipy.ndimage import gaussian_filter1d

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

def extract_gas_parameters(snapshot_path, calculate_h2_quantities=False, parttype=0):
    with h5py.File(snapshot_path, 'r') as F:
        # Load standard fields provided in your example
        pdata = {}
        gas = F["PartType" + str(parttype)]
        
        # Core fields
        fields = ["Masses", "Coordinates", "SmoothingLength", "Potential",
                  "MolecularMassFraction", "Temperature", "Density", "Velocities"]
        if calculate_h2_quantities: 
            fields.extend(["InternalEnergy", "NeutralHydrogenAbundance",
                           "ElectronAbundance", "HII", "Dust_Temperature",
                           "SoundSpeed"])
        if parttype == 1: # high-res DM
            fields = ["Masses", "Coordinates", "SmoothingLength"]
        for field in fields:
            if field in gas:
                pdata[field] = gas[field][:]


        print(f"fields to obtain for parttype {parttype}: {fields}")
        
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

def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Return the rotation matrix for a rotation about a unit vector `axis` by `angle` radians.
    """
    # Ensure axis is a unit vector
    axis = axis / np.linalg.norm(axis)
    kx, ky, kz = axis
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    # Rodrigues' rotation formula
    R = np.array([
        [c + t*kx*kx,      t*kx*ky - s*kz,   t*kx*kz + s*ky],
        [t*ky*kx + s*kz,   c + t*ky*ky,      t*ky*kz - s*kx],
        [t*kz*kx - s*ky,   t*kz*ky + s*kx,   c + t*kz*kz]
    ])
    return R


def load_snapshot_data(snapshot_path, load_parttype4=False,
                       calculate_h2_quantities=False):
    """
    Load snapshot data from an HDF5 file.

    Returns:
        pdata: Gas particle data dictionary
        pdata_dm: DM particle data dictionary
        star_data: STARFORGE stars (PartType5)
        fire_star_data: FIRE stars (PartType4)
    """

    with h5py.File(snapshot_path, 'r') as F:
        # Load gas particle data (PartType0)
        pdata = extract_gas_parameters(snapshot_path, calculate_h2_quantities)

        # Check if potential is present, but DON'T calculate it here
        # (will be calculated later for subset if needed)
        if "Potential" in pdata:
            print(f"Potential is in the dataset")
        else:
            print(f"Potential data not found - will calculate for local region if needed")

        # Try importing parttype 1's if possible
        #try:
        #    pdata_dm = extract_gas_parameters(snapshot_path, False, 1)
        #except:
        #    print(f"PartType1 not in the dataset.")
        pdata_dm = {}
        
        # Add header info
        for key in F['Header'].attrs.keys():
            pdata[key] = F['Header'].attrs[key]
        
        # Load STARFORGE stars (PartType5)
        star_data = {}
        if 'PartType5' in F.keys() and len(F['PartType5']) > 0:
            for field in ["Masses", "Coordinates", "Velocities"]:
                if field in F["PartType5"]:
                    star_data[field] = F["PartType5"][field][:]
        
        # Load FIRE stars (PartType4)
        fire_star_data = {}
        if load_parttype4:
            if 'PartType4' in F.keys() and len(F['PartType4']) > 0:
                for field in ["Masses", "Coordinates"]:
                    if field in F["PartType4"]:
                        fire_star_data[field] = F["PartType4"][field][:]
    
    return pdata, pdata_dm, star_data, fire_star_data

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


def v_com(pos, vel, mass, center, R):
    """
    Compute velocity of the center of mass

    Parameters:
    pos : ndarray, shape (N, 3)  -> positions (kpc)
    vel : ndarray, shape (N, 3)  -> velocities (km/s)
    mass : ndarray, shape (N)    -> masses (GIZMO units)
    center : ndarray, shape (3,) -> reference point (kpc)
    R : float                     -> cutoff radius (kpc)

    Returns:
    L : ndarray, shape (3,)       -> velocity of center of mass (km/s)
    """
    # Vectors from center to points
    r = pos - center               # broadcasting, shape (N,3)
    
    # Distances from center
    dist = np.linalg.norm(r, axis=1)   # shape (N,)
    
    # Select points within the sphere
    mask = dist <= R
    m_total = np.sum(mass[mask])
    
    # Handle case where no particles are in the sphere
    if m_total == 0 or np.sum(mask) == 0:
        return np.zeros(3)
    
    p_in = vel[mask] * mass[mask][:, np.newaxis]
    p_total = np.sum(p_in, axis=0)
    
    # Compute velocity of center of mass
    v_com = p_total / m_total
    return v_com

def angular_momentum(pos, vel, mass, center, R):
    """
    Compute total angular momentum for points within distance R from center.

    Parameters:
    pos  : ndarray, shape (N, 3) -> positions (kpc)
    vel  : ndarray, shape (N, 3) -> velocities (km/s)
    mass : ndarray, shape (N,)   -> masses (GIZMO units: 10^10 Msun)
    center : ndarray, shape (3,) -> reference point (kpc)
    R : float                    -> cutoff radius (kpc)

    Returns:
    L : ndarray, shape (3,)      -> total angular momentum vector (10^10 Msun * kpc * km/s)
    """
    # Vectors from center to points
    r = pos - center               # broadcasting, shape (N,3)

    # Distances from center
    dist = np.linalg.norm(r, axis=1)   # shape (N,)

    # Select points within the sphere
    mask = dist <= R
    r_in = r[mask]                     # (M,3)
    v_in = vel[mask]                   # (M,3)

    # Compute cross product and sum over selected points
    L = np.sum(np.cross(r_in, v_in * mass[mask][:, None]), axis=0)
    return L

def angular_momentum_weighted(pos, vel, mass, center, R, density, potential, pdata=None):
    """
    Compute weighted angular momentum for points within distance R from center.
    Uses density and potential as weights to emphasize denser, more bound material.

    Parameters:
    pos       : ndarray, shape (N, 3) -> positions (kpc)
    vel       : ndarray, shape (N, 3) -> velocities (km/s)
    mass      : ndarray, shape (N,)   -> masses (GIZMO units: 10^10 Msun)
    center    : ndarray, shape (3,)   -> reference point (kpc)
    R         : float                 -> cutoff radius (kpc)
    density   : ndarray, shape (N,)   -> density values
    potential : ndarray, shape (N,)   -> potential values (can contain NaNs)
    pdata     : dict, optional        -> full particle data (for calculating potential if needed)

    Returns:
    L : ndarray, shape (3,)      -> weighted angular momentum vector
    """
    # Vectors from center to points
    r = pos - center
    dist = np.linalg.norm(r, axis=1)

    # Select points within the sphere
    mask = dist <= R
    
    # If potential has NaNs in the selected region and pdata is provided, calculate it
    if np.any(np.isnan(potential[mask])) and pdata is not None:
        print(f"  Potential has NaNs in selected region, calculating locally...")
        potential, _ = calculate_potential_local(pdata, center, R)
    
    # Filter to valid (non-NaN) potential values within radius
    valid_mask = mask & ~np.isnan(potential)
    
    if np.sum(valid_mask) == 0:
        print("  Warning: No valid potential values for weighted L, falling back to mass-weighted")
        return angular_momentum(pos, vel, mass, center, R)
    
    r_in = r[valid_mask]
    v_in = vel[valid_mask]
    m_in = mass[valid_mask]
    rho_in = density[valid_mask]
    pot_in = potential[valid_mask]
    
    # Weight by density and inverse potential (more negative = more bound = higher weight)
    # Normalize potential to avoid numerical issues
    pot_min = np.min(pot_in)
    pot_max = np.max(pot_in)
    if pot_max > pot_min:
        pot_normalized = (pot_in - pot_max) / (pot_max - pot_min)  # Range [0, 1], more negative = closer to 1
    else:
        pot_normalized = np.ones_like(pot_in)
    
    # Combine density and potential weighting
    weights = rho_in * pot_normalized * m_in
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Calculate weighted angular momentum
    L = np.sum(np.cross(r_in, v_in) * weights[:, None], axis=0)
    
    return L

def calculate_angular_momentum_multiple_radii(pos, vel, mass, center, radii):
    """
    Calculate angular momentum for multiple sphere radii.

    This allows testing which radius gives the most stable angular momentum vector.

    Parameters:
    pos    : ndarray, shape (N, 3) -> positions (kpc)
    vel    : ndarray, shape (N, 3) -> velocities (km/s)
    mass   : ndarray, shape (N,)   -> masses (GIZMO units: 10^10 Msun)
    center : ndarray, shape (3,)   -> reference point (kpc)
    radii  : list or ndarray       -> list of radii to test (kpc)

    Returns:
    L_vectors : ndarray, shape (n_radii, 3) -> angular momentum for each radius (10^10 Msun * kpc * km/s)
    """
    # Precompute distances once
    r = pos - center
    dist = np.linalg.norm(r, axis=1)

    L_vectors = np.zeros((len(radii), 3))

    for i, R in enumerate(radii):
        # Select points within this radius
        mask = dist <= R
        r_in = r[mask]
        v_in = vel[mask]
        m_in = mass[mask]

        # Calculate angular momentum
        L_vectors[i] = np.sum(np.cross(r_in, v_in * m_in[:, None]), axis=0)

        # Normalize for comparison
        L_mag = np.linalg.norm(L_vectors[i])
        if L_mag > 0:
            L_norm = L_vectors[i] / L_mag
        else:
            L_norm = np.array([0, 0, 0])

        print(f"  R = {R:.3e} kpc: |L| = {L_mag:.3e}, L_norm = [{L_norm[0]:.3f}, {L_norm[1]:.3f}, {L_norm[2]:.3f}], N_particles = {np.sum(mask)}")

    return L_vectors

def calculate_angular_momentum_multiple_radii_weighted(pos, vel, mass, center, radii, 
                                                       density, potential, pdata=None):
    """
    Calculate weighted angular momentum for multiple sphere radii.
    Uses density and potential as weights.

    Parameters:
    pos       : ndarray, shape (N, 3) -> positions (kpc)
    vel       : ndarray, shape (N, 3) -> velocities (km/s)
    mass      : ndarray, shape (N,)   -> masses (GIZMO units: 10^10 Msun)
    center    : ndarray, shape (3,)   -> reference point (kpc)
    radii     : list or ndarray       -> list of radii to test (kpc)
    density   : ndarray, shape (N,)   -> density values
    potential : ndarray, shape (N,)   -> potential values (can contain NaNs)
    pdata     : dict, optional        -> full particle data (for calculating potential if needed)

    Returns:
    L_vectors : ndarray, shape (n_radii, 3) -> weighted angular momentum for each radius
    """
    r = pos - center
    dist = np.linalg.norm(r, axis=1)

    L_vectors = np.zeros((len(radii), 3))

    for i, R in enumerate(radii):
        L_vectors[i] = angular_momentum_weighted(pos, vel, mass, center, R, 
                                                 density, potential, pdata)
        
        # Normalize for comparison
        L_mag = np.linalg.norm(L_vectors[i])
        if L_mag > 0:
            L_norm = L_vectors[i] / L_mag
        else:
            L_norm = np.array([0, 0, 0])
        
        mask = dist <= R
        print(f"  R = {R:.3e} kpc: |L| = {L_mag:.3e}, L_norm = [{L_norm[0]:.3f}, {L_norm[1]:.3f}, {L_norm[2]:.3f}], N_particles = {np.sum(mask)}")

    return L_vectors

def filter_pdata(pdata: dict, maxboxsize, delta, cutoff_type="box"):
    # Maxboxsize is the fraction of total boxsize
    coords = pdata["Coordinates"]
    limit = pdata["BoxSize"] * maxboxsize * (1 + delta)

    if cutoff_type == "box":
        mask = (coords[:, 0] < limit) & (coords[:, 0] > -limit) & \
               (coords[:, 1] < limit) & (coords[:, 1] > -limit) & \
               (coords[:, 2] < limit) & (coords[:, 2] > -limit)
    elif cutoff_type == "sphere":
        mask = np.linalg.norm(coords, axis=1) < limit / 2
    else:
        raise ValueError(f"Unknown cutoff type: {cutoff_type}. Allowed: box, sphere.")

    for key, data in pdata.items():
        if isinstance(data, np.ndarray) and len(data) == len(coords):
            pdata[key] = data[mask]
            
    return pdata

def get_com_from_stars(star_data, stars_name):
    star_coords = star_data['Coordinates']
    star_masses = star_data['Masses']
    center = np.average(star_coords, axis=0, weights=star_masses)
    print(f"Centered on {stars_name} stars at {center}")
    return center

def smooth_vector_evolution(vectors, sigma=2.0, edge_mode='extend'):
    """
    Smooth vectors evolving in time using Gaussian convolution.

    This reduces "jerking" between snapshots caused by vector jumping between sinks.

    Args:
        vectors: Array of shape (n_snapshots, 3) with vector positions
        sigma: Standard deviation of Gaussian kernel in snapshot units (default: 2.0)
        edge_mode: How to handle edges. Options:
            'extend' (default): Blend to original values near edges (within 2*sigma)
            'nearest': Use scipy's nearest mode only (may still pull toward edge mean)

    Returns:
        smoothed_vectors: Smoothed vector evolution
    """
    print(f"Smoothing {len(vectors)} vector positions with Gaussian kernel (sigma={sigma}, edge_mode={edge_mode})...")

    n = len(vectors)

    # Apply Gaussian filter to each coordinate independently
    smoothed_vectors = np.zeros_like(vectors)
    for i in range(3):
        smoothed_vectors[:, i] = gaussian_filter1d(vectors[:, i], sigma=sigma, mode='nearest')

    # Handle edge cases by blending back to original values
    if edge_mode == 'extend':
        edge_width = int(np.ceil(2 * sigma))
        if edge_width > 0 and n > 2 * edge_width:
            for i in range(edge_width):
                # Blend factor: 0 at edge (use original), 1 at edge_width (use smoothed)
                blend = i / edge_width

                # Start edge
                smoothed_vectors[i] = (1 - blend) * vectors[i] + blend * smoothed_vectors[i]

                # End edge
                j = n - 1 - i
                smoothed_vectors[j] = (1 - blend) * vectors[j] + blend * smoothed_vectors[j]

    # Report average displacement
    displacements = np.linalg.norm(vectors - smoothed_vectors, axis=1)
    print(f"  Average displacement from smoothing: {np.mean(displacements):.3e} kpc")
    print(f"  Maximum displacement from smoothing: {np.max(displacements):.3e} kpc")

    return smoothed_vectors


def fit_polynomial_trajectory(vectors, times=None, degree=1):
    """
    Fit polynomial trajectory to vector evolution in time.

    This is more physically motivated than Gaussian smoothing - assumes
    the center moves with constant velocity (degree=1), constant acceleration
    (degree=2), or higher order motion.

    Args:
        vectors: Array of shape (n_snapshots, 3) with vector positions
        times: Array of shape (n_snapshots,) with times. If None, uses snapshot indices.
        degree: Polynomial degree (1=linear, 2=quadratic, 3=cubic, etc.)

    Returns:
        fitted_vectors: Smoothed vector evolution from polynomial fit
    """
    print(f"Fitting degree-{degree} polynomial trajectory to {len(vectors)} vector positions...")

    n = len(vectors)

    # Use snapshot index as time if not provided
    if times is None:
        times = np.arange(n)

    # Fit polynomial to each coordinate independently
    fitted_vectors = np.zeros_like(vectors)

    for i in range(3):
        # Fit polynomial: x(t) = a_n*t^n + ... + a_1*t + a_0
        coeffs = np.polyfit(times, vectors[:, i], degree)
        fitted_vectors[:, i] = np.polyval(coeffs, times)

        coord_name = ['x', 'y', 'z'][i]
        print(f"  {coord_name}: coeffs = {coeffs} (highest power first)")

    # Report statistics
    displacements = np.linalg.norm(vectors - fitted_vectors, axis=1)
    residuals_rms = np.sqrt(np.mean(displacements**2))

    print(f"  RMS residual: {residuals_rms:.3e} kpc")
    print(f"  Mean displacement: {np.mean(displacements):.3e} kpc")
    print(f"  Max displacement: {np.max(displacements):.3e} kpc")

    return fitted_vectors


def calculate_all_centers(run_out_path, snap_nos, center_type, L_calc_radius_kpc,
                          out_path, run_subdir="8x_zoom_m12f",
                          L_radii_list=None, L_use_stars=False, L_use_weighted=True,
                          fallback_to_density=True):
    """
    First pass: Calculate centers for all snapshots and save to files.

    Args:
        run_out_path: Path to snapshot directory
        snap_nos: List of snapshot numbers
        center_type: Type of centering ("potential", "star", etc.)
        L_calc_radius_kpc: Radius for angular momentum/potential calculation (in kpc)
        out_path: Base output path for saving centers
        run_subdir: Subdirectory name for this run (default: "8x_zoom_m12f")
        L_radii_list: List of radii (in kpc) for angular momentum calculation.
                      If None, uses [L_calc_radius_kpc] (single radius)
        L_use_stars: If True, calculate angular momentum from STARFORGE stars (PartType5) instead of gas.
                    Defaults to False (use gas particles). Overrides L_use_weighted.
        L_use_weighted: If True, calculate weighted angular momentum using density and potential.
                       Defaults to True. Only used if L_use_stars=False.
        fallback_to_density: If True (default), use densest gas particle when no stars found.
                            If False, use extrapolation from previous snapshot.
                            Recommended: True (notebook behavior - more stable).

    Returns:
        None (centers and angular momentum vectors are saved to files)
    """
    print("="*80)
    print("FIRST PASS: Calculating and saving centers for all snapshots...")
    print("="*80)

    # Setup radii for angular momentum calculation
    if L_radii_list is None:
        L_radii_list = [L_calc_radius_kpc]

    print(f"\nAngular momentum will be calculated for {len(L_radii_list)} radii:")
    for j, r in enumerate(L_radii_list):
        print(f"  [{j}] R = {r:.3e} kpc")
    
    if L_use_stars:
        print("\n*** Using STARFORGE stars (PartType5) for angular momentum calculation ***\n")
    elif L_use_weighted:
        print("\n*** Using weighted angular momentum (density & potential) for calculation ***\n")
    else:
        print("\n*** Using standard mass-weighted angular momentum for calculation ***\n")

    external_data = None

    for i, snapstr in enumerate(snap_nos):
        snap_hdf5 = "snapshot_" + snapstr + ".hdf5"
        run_path = os.path.join(run_out_path, snap_hdf5)

        print(f"\n[{i+1}/{len(snap_nos)}] Processing {snapstr}...")

        # Load data
        pdata, pdata_dm, star_data, fire_star_data = load_snapshot_data(
            run_path, True, calculate_h2_quantities=False
        )

        box_size_val = pdata['BoxSize']
        time_current = cosmo.age(pdata["Redshift"]).to(u.second)

        # Get center for this snapshot
        center = get_center(pdata, star_data, fire_star_data, external_data,
                           time_current, center_type, L_calc_radius_kpc,
                           box_size_val, fallback_to_density)

        # Calculate velocity of center
        v_tot = v_com(pdata["Coordinates"], pdata["Velocities"], pdata["Masses"],
                     center, L_calc_radius_kpc)

        # Calculate angular momentum for multiple radii
        print(f"  Calculating angular momentum for {len(L_radii_list)} radii:")
        if L_use_stars and star_data and len(star_data.get("Masses", [])) > 0:
            print("    Using STARFORGE stars (PartType5)")
            L_vectors = calculate_angular_momentum_multiple_radii(
                star_data["Coordinates"], star_data["Velocities"], star_data["Masses"],
                center, L_radii_list
            )
        elif L_use_weighted:
            print("    Using weighted angular momentum (density & potential)")
            L_vectors = calculate_angular_momentum_multiple_radii_weighted(
                pdata["Coordinates"], pdata["Velocities"], pdata["Masses"],
                center, L_radii_list, pdata["Density"],
                pdata.get("Potential", np.full(len(pdata["Masses"]), np.nan)),
                pdata
            )
        else:
            if L_use_stars:
                print("    Warning: L_use_stars=True but no star data available, using gas particles")
            print("    Using standard mass-weighted angular momentum")
            L_vectors = calculate_angular_momentum_multiple_radii(
                pdata["Coordinates"], pdata["Velocities"], pdata["Masses"],
                center, L_radii_list
            )

        # Save center, velocity, time, and L vectors to files
        out_save_path = os.path.join(out_path, run_subdir, snapstr)
        L_save_path = os.path.join(out_save_path, "L_vectors")
        os.makedirs(out_save_path, exist_ok=True)
        os.makedirs(L_save_path, exist_ok=True)

        center_save_file = os.path.join(out_save_path, "center.txt")
        velocity_save_file = os.path.join(out_save_path, "velocity.txt")
        time_save_file = os.path.join(out_save_path, "time.txt")
        radii_save_file = os.path.join(L_save_path, "radii.txt")

        np.savetxt(center_save_file, center)
        np.savetxt(velocity_save_file, v_tot)
        np.savetxt(time_save_file, np.array([time_current.to(u.second).value]))
        np.savetxt(radii_save_file, L_radii_list)  # Save radii list for reference

        # Save each L vector
        for j, (r, L_vec) in enumerate(zip(L_radii_list, L_vectors)):
            L_file = os.path.join(L_save_path, f"L_r{j:02d}_{r:.3e}kpc.txt")
            np.savetxt(L_file, L_vec)

        # Update external_data for next snapshot
        if external_data is None:
            external_data = [None, None, None, None]
        external_data[0] = center
        external_data[1] = v_tot
        external_data[2] = time_current.to(u.second).value

        print(f"  Center: {center}")
        print(f"  Velocity: {v_tot}")
        print(f"  Saved to: {center_save_file}")
        print(f"  Angular momentum vectors saved to: {L_save_path}")

    print("\n" + "="*80)
    print("FIRST PASS COMPLETE - All centers and L vectors saved to files")
    print("="*80)

def load_centers_from_files(out_path, snap_nos, run_subdir="8x_zoom_m12f"):
    """
    Load previously calculated centers from files.

    Args:
        out_path: Base output path where centers were saved
        snap_nos: List of snapshot numbers
        run_subdir: Subdirectory name for this run (default: "8x_zoom_m12f")

    Returns:
        centers: Array of shape (n_valid_snapshots, 3) with center positions
        velocities: Array of shape (n_valid_snapshots, 3) with center velocities
        times: Array of times for each snapshot
        valid_snap_nos: List of snapshot numbers that were successfully loaded
    """
    print("="*80)
    print("Loading centers from saved files...")
    print("="*80)

    centers = []
    velocities = []
    times = []
    valid_snap_nos = []

    for i, snapstr in enumerate(snap_nos):
        out_save_path = os.path.join(out_path, run_subdir, snapstr)
        center_save_file = os.path.join(out_save_path, "center.txt")
        velocity_save_file = os.path.join(out_save_path, "velocity.txt")
        time_save_file = os.path.join(out_save_path, "time.txt")

        if not os.path.exists(center_save_file):
            print(f"[{i+1}/{len(snap_nos)}] Skipping snapshot {snapstr} - center file not found")
            continue

        center = np.loadtxt(center_save_file)

        # Handle optional velocity and time files
        if os.path.exists(velocity_save_file):
            velocity = np.loadtxt(velocity_save_file)
        else:
            velocity = np.zeros(3)

        if os.path.exists(time_save_file):
            time = np.loadtxt(time_save_file)
        else:
            time = 0.0

        centers.append(center)
        velocities.append(velocity)
        times.append(time)
        valid_snap_nos.append(snapstr)

        print(f"[{i+1}/{len(snap_nos)}] Loaded snapshot {snapstr}: center = {center}")

    if len(centers) == 0:
        raise ValueError("No valid center data found for any snapshot!")

    centers = np.array(centers)
    velocities = np.array(velocities)
    times = np.array(times)

    print(f"\nLoaded {len(centers)}/{len(snap_nos)} centers from files")
    print(f"Valid snapshots: {valid_snap_nos[0]} to {valid_snap_nos[-1]}")
    print("="*80)

    return centers, velocities, times, valid_snap_nos


def load_angular_momentum_vectors(out_path, snap_nos, run_subdir="8x_zoom_m12f"):
    """
    Load previously calculated angular momentum vectors for all radii.

    Args:
        out_path: Base output path where L vectors were saved
        snap_nos: List of snapshot numbers
        run_subdir: Subdirectory name for this run (default: "8x_zoom_m12f")

    Returns:
        L_vectors: Array of shape (n_valid_snapshots, n_radii, 3) with L vectors
        radii: Array of radii used for calculation
        valid_snap_nos: List of snapshot numbers that were successfully loaded
    """
    print("="*80)
    print("Loading angular momentum vectors from saved files...")
    print("="*80)

    # First, get the radii list from the first snapshot
    first_snap = snap_nos[0]
    L_save_path = os.path.join(out_path, run_subdir, first_snap, "L_vectors")
    radii_file = os.path.join(L_save_path, "radii.txt")

    if not os.path.exists(radii_file):
        raise FileNotFoundError(f"Radii file not found: {radii_file}")

    radii = np.loadtxt(radii_file)
    # Ensure radii is always a 1D array, even if file contains single value
    radii = np.atleast_1d(radii)
    n_radii = len(radii)

    print(f"Found {n_radii} radii:")
    for i, r in enumerate(radii):
        print(f"  [{i}] R = {r:.3e} kpc")

    # Load L vectors for all snapshots and radii
    L_vectors = []
    valid_snap_nos = []

    for i, snapstr in enumerate(snap_nos):
        L_save_path = os.path.join(out_path, run_subdir, snapstr, "L_vectors")
        
        # Skip if L_vectors directory doesn't exist
        if not os.path.exists(L_save_path):
            print(f"[{i+1}/{len(snap_nos)}] Skipping snapshot {snapstr} - L_vectors directory not found")
            continue
            
        L_snap = []
        all_files_found = True

        for j, r in enumerate(radii):
            L_file = os.path.join(L_save_path, f"L_r{j:02d}_{r:.3e}kpc.txt")

            if not os.path.exists(L_file):
                print(f"  Warning: L vector file not found: {L_file}")
                # Use NaN placeholder to maintain array shape
                L_vec = np.array([np.nan, np.nan, np.nan])
                all_files_found = False
            else:
                L_vec = np.loadtxt(L_file)
                # Ensure L_vec is 1D array with 3 elements
                L_vec = np.atleast_1d(L_vec)
            
            L_snap.append(L_vec)

        if all_files_found:
            L_vectors.append(L_snap)
            valid_snap_nos.append(snapstr)
            print(f"[{i+1}/{len(snap_nos)}] Loaded L vectors for snapshot {snapstr}")
        else:
            print(f"[{i+1}/{len(snap_nos)}] Skipping snapshot {snapstr} - missing L vector files")

    if len(L_vectors) == 0:
        raise ValueError("No valid L vector data found for any snapshot!")

    L_vectors = np.array(L_vectors)

    print(f"\nLoaded L vectors with shape: {L_vectors.shape} (snapshots, radii, 3)")
    print(f"Successfully loaded {len(L_vectors)}/{len(snap_nos)} snapshots")
    print(f"Valid snapshots: {valid_snap_nos[0]} to {valid_snap_nos[-1]}")
    print("="*80)

    return L_vectors, radii, valid_snap_nos

def calculate_potential_local(pdata, center, radius):
    """
    Calculate gravitational potential for particles within a sphere.

    This avoids expensive full-domain potential calculations by only
    computing potential for a local region around the center.

    Args:
        pdata: Particle data dictionary
        center: Center position (3D array, kpc)
        radius: Radius within which to calculate potential (kpc)

    Returns:
        potential: Full potential array (only local region calculated, rest is NaN)
        mask: Boolean mask of particles within radius
    """
    print(f"Calculating potential for local region (r < {radius:.3e} kpc) around {center}...")
    t = time.time()

    # Find particles within radius
    r = pdata["Coordinates"] - center
    dist = np.linalg.norm(r, axis=1)
    mask = dist <= radius

    n_local = np.sum(mask)
    n_total = len(pdata["Coordinates"])
    print(f"  Selected {n_local}/{n_total} particles ({100*n_local/n_total:.1f}%)")

    # Calculate potential only for local particles
    potential_local = Potential(
        pdata["Coordinates"][mask],
        pdata["Masses"][mask],
        pdata["SmoothingLength"][mask]
    )

    # Create full potential array (NaN for particles outside region)
    potential_full = np.full(n_total, np.nan)
    potential_full[mask] = potential_local

    print(f"  Potential calculation complete in {time.time() - t:.2f} seconds")

    return potential_full, mask

def get_center(pdata, star_data, fire_star_data, external_data, time_current,
               center_type, L_calc_radius, box_size_val, fallback_to_density=True):
    """
    Find center position for snapshot.

    Parameters
    ----------
    fallback_to_density : bool
        If True (default), use densest gas particle when no stars found.
        If False, use extrapolation from previous snapshot (old behavior).

        Recommended: True (notebook behavior - more stable, no error accumulation)
    """
    center = np.array([box_size_val / 2, box_size_val / 2, box_size_val / 2])
    has_stars = False

    # Check if we have stars
    if star_data and 'Coordinates' in star_data and len(star_data['Coordinates']) > 0:
        has_stars = True
        center = get_com_from_stars(star_data, "STARFORGE")
    elif fire_star_data and 'Coordinates' in fire_star_data and len(fire_star_data['Coordinates']) > 0:
        has_stars = True
        center = get_com_from_stars(fire_star_data, "FIRE")

    # If no stars found, choose fallback method
    if not has_stars:
        if fallback_to_density:
            # Notebook approach: use densest gas particle
            idx = np.argmax(pdata['Density'])
            center = pdata['Coordinates'][idx].copy()
            print(f"No stars found - using densest gas particle at {center}")
        else:
            # Old approach: extrapolate from previous snapshot
            if external_data is not None and external_data[0] is not None:
                # Extrapolate center from previous snapshot using velocity
                c_init = external_data[0] * u.kpc
                v_init = external_data[1] * u.km / u.second
                t_init = external_data[2] * u.second

                dt = time_current - t_init

                # Calculate physical displacement
                dr = (v_init * dt).decompose().to(u.kpc).value
                center = c_init.value + dr
                print(f"No stars found - extrapolating from previous snapshot:")
                print(f"  Previous center: {c_init}")
                print(f"  Time delta: {dt}")
                print(f"  Displacement: {dr}")
                print(f"  New center: {center}")
            else:
                print(f"Warning: No stars found and no previous data, centering on box center at {center}")

    if center_type == "potential": # Find minimum potential within sphere around initial center
        # Calculate potential locally if not present
        if "Potential" not in pdata or np.all(np.isnan(pdata["Potential"])):
            print("Potential not in dataset, calculating for local region...")
            pdata["Potential"], _ = calculate_potential_local(pdata, center, L_calc_radius)

        # Find minimum potential within radius
        r = pdata["Coordinates"] - center
        dist = np.linalg.norm(r, axis=1)   # shape (N,)
        mask = dist <= L_calc_radius
        pot_data = pdata["Potential"]

        # Only consider valid (non-NaN) potential values
        valid_mask = mask & ~np.isnan(pot_data)
        if np.sum(valid_mask) == 0:
            print("Warning: No valid potential values found, using initial center")
        else:
            pot_idx = np.argmin(pot_data[valid_mask])
            center = pdata["Coordinates"][valid_mask][pot_idx].copy()
            print(f"Centered on minimum of potential at {center}")
    return center

def get_rotation_matrix(pdata, first_sim_flag, external_data, center, L_calc_radius, 
                       precalculated_L=None, star_data=None, L_use_stars=False, 
                       L_use_weighted=True):
    if precalculated_L is not None:
        l = precalculated_L
        print(f"Using pre-calculated angular momentum vector: {l}")
        external_data[3] = l
    elif first_sim_flag:
        if L_use_stars and star_data and len(star_data.get("Masses", [])) > 0:
            print("Calculating angular momentum from STARFORGE stars (PartType5)")
            l = angular_momentum(star_data["Coordinates"], star_data["Velocities"],
                                 star_data["Masses"], center, L_calc_radius)
        elif L_use_weighted:
            print("Calculating weighted angular momentum (density & potential)")
            l = angular_momentum_weighted(
                pdata["Coordinates"], pdata["Velocities"], pdata["Masses"], 
                center, L_calc_radius, pdata["Density"], 
                pdata.get("Potential", np.full(len(pdata["Masses"]), np.nan)), 
                pdata
            )
        else:
            if L_use_stars:
                print("Warning: L_use_stars=True but no star data available, using gas particles")
            print("Calculating standard mass-weighted angular momentum")
            l = angular_momentum(pdata["Coordinates"], pdata["Velocities"],
                                 pdata["Masses"], center, L_calc_radius)
        print(f"Angular momentum vector was found to be {l}")
        external_data[3] = l
    else:
        l = external_data[3]
    if l is None or isinstance(l, int):
        raise ValueError("Angular momentum vector (index 3) is missing or invalid in external_data.")
    l_norm = l / np.linalg.norm(l)
    print(f"DEBUG: external_data: {external_data}")
    print(f"After normalization, angular momentum vector is: {l_norm}")
    angle = np.arccos(l_norm[2]) # arccos(L_z / |L|)
    z_axis = np.array([0,0,1])
    axis_vec = np.cross(l, z_axis)
    print(f"Will now perform rotation with angle {angle} aroud axis {axis_vec}...")
    return rotation_matrix(axis_vec, angle), l

def setup_meshoid(snapshot_path, center_type="none", rotate_type="none",
                  L_calc_radius = 1e20, calculate_h2_quantities=False,
                  recenter=False, extra_rotation=[0], mainsnap=False,
                  maxboxsize=1e-2, external_data = None, precalculated_center=None,
                  precalculated_L=None, L_use_stars=False, L_use_weighted=True,
                  fallback_to_density=True):
    """
    Sets up data containers for a single snapshot

    Args:
        snapshot_path (str): Path to snapshot data
        center_type (str): "none" (Default): do not recenter the data
                           "potential": recenter to potential minimum
                           "star": recenter to average of star postitions
        rotate_type (str): "none" (Default): do not rotate the data
                           "L": recenter and then rotate so that angular momentum vector is aligned with z
        L_calc_radius (float): radius inside of which to calculate angular
                           momentum for rotation (kpc). If set to 1e20 (default),
                           effectively uses a very large radius
        calculate_h2_quantities (bool): Whether to calculate the H2 quantites. Defaults to False.
        recenter (bool): Whether to recenter to box center.
        extra_rotation (float): Can set to pi/2 to get edge-on plot
        mainsnap (bool): Whether rotation/recenter quantites are calculated from
                         this snapshot, or are taken from the file. Default
                         False (quantites taken from external file)
        maxboxsize (float): Size of the largest box where plotting will be happening.
        external_data (list): List that is used if mainsnap=False. Contents:
                              [center_of_box (kpc), velocity_com (km/s), reference_time (s), normalized_angular_momentum_vec]
        precalculated_center (ndarray): Pre-calculated (and possibly smoothed) center position (kpc).
                              If provided, overrides center calculation. Shape: (3,)
        precalculated_L (ndarray): Pre-calculated (and possibly smoothed) angular momentum vector.
                              If provided, overrides L calculation. Shape: (3,). Does not need to be normalized.
        L_use_stars (bool): If True, calculate angular momentum from STARFORGE stars (PartType5) instead of gas.
                           Defaults to False (use gas particles). Overrides L_use_weighted.
        L_use_weighted (bool): If True, calculate weighted angular momentum using density and potential.
                              Defaults to True. Only used if L_use_stars=False.
        fallback_to_density (bool): If True (default), use densest gas particle when no stars found.
                                   If False, use extrapolation from previous snapshot.
                                   Recommended: True (notebook behavior - more stable, no error accumulation).

    Returns:
        dict: Dictionary with snapshot data
    """

    #if external_data == None and not mainsnap:
    #    raise ValueError(f"When setting mainsnap=False in setup_meshoid, you MUST populate external_data")
    
    # Load data
    print(f"Loading snapshot: {snapshot_path}")
    pdata, pdata_dm, star_data, fire_star_data = load_snapshot_data(snapshot_path, True, 
                                                                    calculate_h2_quantities)
    
    # Needed to only calculate angular momentum vector when needed
    first_sim_flag = False
    if not external_data:
        first_sim_flag = True
        external_data = [None, None, None, None]

    box_size_val = pdata['BoxSize']
    v_tot = 0
    a = pdata["Time"]
    time_current = cosmo.age(pdata["Redshift"]).to(u.second)

    # Determine center
    if precalculated_center is not None:
        center = precalculated_center.copy()
        print(f"Using pre-calculated center: {center}")
    else:
        center = get_center(pdata, star_data, fire_star_data, external_data,
                            time_current, center_type, L_calc_radius, box_size_val,
                            fallback_to_density)

    v_tot = v_com(pdata["Coordinates"], pdata["Velocities"], pdata["Masses"],
                  center, L_calc_radius)
    external_data[0] = center
    external_data[1] = v_tot
    external_data[2] = time_current.to(u.second).value

    # Save original center before recentering (for saving to file later)
    original_center = center.copy()

    # Recentering
    #if rotate_type == "L":
    #    recenter = True
    #print(f"Coords before recentering:")
    #print(pdata["Coordinates"])
    if recenter:
        pdata["Coordinates"] -= center
        #pdata_dm["Coordinates"] -= center
        if star_data: star_data["Coordinates"] -= center
        if fire_star_data: fire_star_data["Coordinates"] -= center
        center = np.zeros([3])
    print("─"*80)

    # This defines radius of cutoff sphere to be distance from center to far-edge of the cube
    delta = np.sqrt(3) - 1 
    pdata = filter_pdata(pdata, maxboxsize, delta, cutoff_type="sphere")

    #print(f"Coords after recentering:")
    #print(pdata["Coordinates"])

    data_dicts = []
    prev_angle = 0
    extra_rotation_axis = [1, 0, 0]
    # For each extra_rotation angle, output a data_dict
    for x_rotation in extra_rotation:
        # Rotate to make z align with angular momentum
        l = np.array([0, 0, 1])
        if rotate_type == "L" and L_calc_radius != 1e20:

            # If this is a first extra rotation, then align z with angular
            # momentum, and update angular momentum value
            if len(data_dicts) == 0:
                R, l = get_rotation_matrix(pdata, first_sim_flag,
                                           external_data, center,
                                           L_calc_radius, precalculated_L,
                                           star_data, L_use_stars, L_use_weighted)
            else:
                R = np.identity(3)

            # Undo previous extra rotation, and then do the next rotation
            prev_R = rotation_matrix(extra_rotation_axis, -prev_angle) # Undo previous rotation
            extra_R = rotation_matrix(extra_rotation_axis, x_rotation) # For edge-on projection
            prev_angle = x_rotation
            R = extra_R @ prev_R @ R

            # Rotate the data
            pdata["Coordinates"] = pdata["Coordinates"] @ R.T
            pdata["Velocities" ] = pdata["Velocities" ] @ R.T
            #if pdata_dm["Coordinates"]: pdata_dm["Coordinates"] = pdata_dm["Coordinates"] @ R.T 
            print(f"Rotation matrix: {R}")
            if star_data: 
                star_data["Coordinates"] = star_data["Coordinates"] @ R.T
            if fire_star_data: 
                fire_star_data["Coordinates"] = fire_star_data["Coordinates"] @ R.T
        
        # H2 calculations
        if calculate_h2_quantities: h2_results = calculate_h2_rates_vectorized(pdata)
        else: h2_results = {}
        
        # Create Meshoid object for surface density calculation
        print("Creating Meshoid object...")
        M = Meshoid(pdata["Coordinates"], pdata["Masses"], pdata["SmoothingLength"])
        dictionary = {
            "M": M,
            "L": l,
            "center": center.copy(),
            "original_center": original_center.copy(),  # Center before recentering
            "star_data": star_data.copy(),
            "snapshot_path": snapshot_path,
            "fire_star_data": fire_star_data.copy(),
            "boxsize": pdata["BoxSize"].copy(),
            "h2_results": h2_results,
            "pdata": pdata.copy(),
            "pdata_dm": pdata_dm.copy()
        }
        data_dicts.append(dictionary)
    return data_dicts, external_data

def add_zoomboxes(ax, actual_box_size, center, plot_zoombox):
    # Centered white box (side = 1/10th of current actual_box_size)
    s = actual_box_size / 20  # Half-side length
    ax.plot([center[0]-s, center[0]+s, center[0]+s, center[0]-s, center[0]-s], 
            [center[1]-s, center[1]-s, center[1]+s, center[1]+s, center[1]-s], 
            color='white', lw=1.5, zorder=10)
    add_NE_line = False
    add_NW_line = False
    add_SE_line = False
    add_SW_line = False
    if plot_zoombox == 1: # Zoom-in is eastward
        add_NE_line, add_SE_line = True, True
    elif plot_zoombox == 2: # Zoom-in is southward
        add_SW_line, add_SE_line = True, True
    elif plot_zoombox == 3: # Zoom-in is westward
        add_SW_line, add_NW_line = True, True
    elif plot_zoombox == 4: # Zoom-in is northward
        add_NE_line, add_NW_line = True, True
    # Define limits for clarity
    x_left, x_right = center[0] - actual_box_size/2, center[0] + actual_box_size/2
    y_bottom, y_top = center[1] - actual_box_size/2, center[1] + actual_box_size/2

    if add_SE_line:
        ax.plot([center[0] + s, x_right], [center[1] - s, y_bottom], color='white', lw=1.5, zorder=10)
    if add_NE_line:
        ax.plot([center[0] + s, x_right], [center[1] + s, y_top], color='white', lw=1.5, zorder=10)
    if add_NW_line:
        ax.plot([center[0] - s, x_left], [center[1] + s, y_top], color='white', lw=1.5, zorder=10)
    if add_SW_line:
        ax.plot([center[0] - s, x_left], [center[1] - s, y_bottom], color='white', lw=1.5, zorder=10)

def add_scalebars(ax, actual_box_size, au_scale, pc_scale):
    # +- 3's you see here in exponents are from the fact that GIZMO units are
    # in kpc, not pc
    fontprops = fm.FontProperties(size=20)
    scale_bar_size = actual_box_size / 10
    au_to_pc = 4.848102e-6
    pc_power = pc_scale
    if au_scale > -7:
        au_power = au_scale
        vertical_size = actual_box_size / 300
        scalebar = AnchoredSizeBar(ax.transData,
                                   au_to_pc*10**au_power*1e-3, 
                                   f"""$10^{{{au_power}}}$ AU""",
                                   loc = 'upper left',
                                   pad=1,
                                   color='white',
                                   frameon=False,
                                   size_vertical=vertical_size,
                                   fontproperties=fontprops)
        ax.add_artist(scalebar)

    scalebar_pc = AnchoredSizeBar(ax.transData,
                               10**(pc_power-3), f"""$10^{{{pc_power}}}$ pc""", 
                               loc = 'upper left',
                               pad=3,
                               color='white',
                               frameon=False,
                               size_vertical=vertical_size,
                               fontproperties=fontprops)
    ax.add_artist(scalebar_pc)
    
def plot_single_snapshot(dictionary, ax, plot_quantity, box_size, resolution,
                         pc_scale, au_scale, plot_fire_stars, plot_zoombox,
                         projection, mainsnap, vmin=1, vmax=2000,
                         output_dir="./"):
    M = dictionary["M"]
    star_data = dictionary["star_data"]
    fire_star_data = dictionary["fire_star_data"]
    center = dictionary["center"]
    pdata_boxsize = dictionary["boxsize"]
    snapshot_path = dictionary["snapshot_path"]

    if plot_quantity == None:
        quantity_data = M.m
    elif plot_quantity in dictionary["pdata"]:
        quantity_data = dictionary["pdata"][plot_quantity]
    else:
        print(f"No {plot_quantity} found in this snapshot. Will not be plotting it.")
        return

    if not projection:
        Plotter = lambda x: M.SurfaceDensity(x, center=center,
                                             size=actual_box_size,
                                             res=resolution)
    else:
        Plotter = lambda x: M.Projection(x, center=center,
                                         size=actual_box_size, res=resolution)

    
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
    sigma_gas = Plotter(quantity_data)
    # Automatically set the boundaries for gas density
    vmin = np.min(sigma_gas)+1e-16
    vmax = np.max(sigma_gas)+1e-16 # Adding here in case they are both 0
    
    # Create figure
    ax.set_facecolor('black')
    
    # Set colorbar to log-normal if everything is positive
    if vmin > 0:
        norm_colors = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm_colors = colors.Normalize(vmin=vmin, vmax=vmax)

    # Plot surface density
    p = ax.pcolormesh(X, Y, sigma_gas, norm=norm_colors, cmap='cet_fire')
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
    add_scalebars(ax, actual_box_size, au_scale, pc_scale)

    # Adding zoom-boxes which show in which direction your zooms are happening.
    if plot_zoombox: add_zoomboxes(ax, actual_box_size, center, plot_zoombox)
    
    plt.tight_layout()
    
    # Save figure
    #os.makedirs(output_dir, exist_ok=True)
    
    #snap_name = Path(snapshot_path).stem + '.png'
    #output_path = os.path.join(output_dir, snap_name)
    #plt.savefig(output_path, dpi=300, facecolor='black')
    #print(f"Saved: {output_path}")
    #plt.show()
    #return fig


def plot_single_zoom(data_dict, ax, kwargs):
    plot_single_snapshot(data_dict, ax, **kwargs)
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def plot_zooms(data_dict, plot_quantity=None, resolution=1000, xplots = 2, 
               yplots = 3, init_boxsize = 1e-2, projection = False, 
               init_pcscale = 5, init_auscale = 10, mainsnap = False):
    print(f"Started plotting zoom-ins...")
    fig, axs = plt.subplots(yplots, xplots, figsize=(xplots*10, yplots*10))
    boxsize_zooms = xplots * yplots
    #print(np.shape(axs))
    plot_zoombox = 1
    plot_fire_stars = False
    for i in range(boxsize_zooms):
        new_boxsize = init_boxsize / 10**i
        new_pcscale = init_pcscale - i
        new_auscale = init_auscale - i
        xplot_id = i % xplots
        yplot_id = i // xplots
        plot_zoombox_l = 1
        if yplot_id % 2 == 1:
            xplot_id = xplots - xplot_id - 1
            plot_zoombox_l = 3
            if xplot_id == 0: plot_zoombox_l = 2
        elif (i + 1) % xplots == 0:
            plot_zoombox_l = 2  # End of row: next plot is South
        #print(f"x,y: {xplot_id}, {yplot_id}")
        ax = axs[yplot_id][xplot_id]
        if i == boxsize_zooms-1: plot_zoombox_l = 0
        kwargs = {'box_size': new_boxsize, 'output_dir': './', 
                  'resolution': resolution, "pc_scale": new_pcscale, 
                  "au_scale": new_auscale, "plot_fire_stars": plot_fire_stars,
                  "plot_quantity": plot_quantity, "projection": projection,
                  "mainsnap": mainsnap, "plot_zoombox": plot_zoombox_l} #, 'vmin': 1e-4, 'vmax': 1e-3}
        plot_single_zoom(data_dict, ax, kwargs)
    plt.subplots_adjust(wspace=0, hspace=0)
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
