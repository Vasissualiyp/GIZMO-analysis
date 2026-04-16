from IPython.display import display
import sys, os, importlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import h5py
from scipy import stats
from scipy.fft import fftn, fftfreq
from scipy.spatial import cKDTree

# Physical constants (CGS)
G_GRAV = 6.674e-8  # gravitational constant in cm^3/g/s^2
K_BOLTZ = 1.381e-16  # Boltzmann constant in erg/K
M_PROTON = 1.673e-24  # proton mass in g
M_SUN = 1.989e33  # solar mass in g
PC_TO_CM = 3.086e18  # parsec to cm
KM_TO_CM = 1e5  # km to cm
YEAR_TO_SEC = 3.154e7  # year to seconds

# Unit conversions (GIZMO units)
CODE_MASS = 2e43  # 10^10 Msun in grams
CODE_LENGTH = 3.0857e+21  # kpc in cm
CODE_VELOCITY = 1e5  # km/s in cm/s
CODE_TIME = CODE_LENGTH / CODE_VELOCITY  # time in seconds

# ============================================================================
# SNAPSHOT LOADING AND SETUP
# ============================================================================

def load_snapshot_full(snapshot_path, center_on_stars=True):
    """
    Load snapshot with all particle types and compute basic properties.
    Returns dictionary with all data needed for analysis.
    """
    print(f"Loading snapshot: {snapshot_path}")
    
    with h5py.File(snapshot_path, 'r') as F:
        # Load header
        header = F['Header'].attrs
        time = header['Time']
        boxsize = header['BoxSize']
        hubble = header.get('HubbleParam', 1.0)
        
        # Load gas particles (PartType0)
        gas_data = {}
        if 'PartType0' in F.keys():
            pt0 = F['PartType0']
            for field in ['Coordinates', 'Masses', 'Velocities', 'Density', 
                          'Potential', 'Temperature', 'SmoothingLength', 
                          'MagneticField']:
                if field in pt0.keys():
                    gas_data[field] = pt0[field][:]
        
        # Load STARFORGE stars (PartType5) - what we care about
        star_data = {}
        if 'PartType5' in F.keys():
            pt5 = F['PartType5']
            for field in ['Coordinates', 'Masses', 'Velocities', 'ParticleIDs',
                         'StellarFormationTime']:
                if field in pt5.keys():
                    star_data[field] = pt5[field][:]
            print(f"Found {len(star_data.get('Masses', []))} STARFORGE stars (PartType5)")
        else:
            print("Warning: No PartType5 (STARFORGE stars) found!")
        
        # Check for FIRE stars (PartType4) - should not be present
        if 'PartType4' in F.keys() and len(F['PartType4']['Masses']) > 0:
            print(f"WARNING: Found {len(F['PartType4']['Masses'])} FIRE stars (PartType4)!")
    
    # Determine center
    if center_on_stars and star_data and 'Coordinates' in star_data:
        star_coords = star_data['Coordinates']
        star_masses = star_data['Masses']
        center = np.average(star_coords, axis=0, weights=star_masses)
        print(f"Centered on STARFORGE stars at {center}")
    else:
        center = np.array([boxsize/2, boxsize/2, boxsize/2])
        print(f"Centered on box center at {center}")
    
    return {
        'gas_data': gas_data,
        'star_data': star_data,
        'center': center,
        'time': time,
        'boxsize': boxsize,
        'hubble': hubble,
        'snapshot_path': snapshot_path
    }

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def compute_cylindrical_coords(coords, center, velocities=None):
    """
    Compute cylindrical coordinates (R, phi, z) centered on given point.
    Assumes z-axis is vertical.
    """
    rel_coords = coords - center
    R = np.sqrt(rel_coords[:, 0]**2 + rel_coords[:, 1]**2)
    phi = np.arctan2(rel_coords[:, 1], rel_coords[:, 0])
    z = rel_coords[:, 2]
    
    if velocities is not None:
        v_R = (rel_coords[:, 0] * velocities[:, 0] + 
               rel_coords[:, 1] * velocities[:, 1]) / R
        v_phi = (-rel_coords[:, 1] * velocities[:, 0] + 
                  rel_coords[:, 0] * velocities[:, 1]) / R
        v_z = velocities[:, 2]
        return R, phi, z, v_R, v_phi, v_z
    
    return R, phi, z

def compute_toomre_q(data_dict, n_bins=50, r_max=None):
    """
    Compute Toomre Q parameter as function of radius.
    Q = c_s * kappa / (pi * G * Sigma)
    where kappa is epicyclic frequency (~ sqrt(2) * Omega for thin disk)
    """
    gas = data_dict['gas_data']
    center = data_dict['center']
    
    coords = gas['Coordinates']
    masses = gas['Masses']
    velocities = gas['Velocities']
    temps = gas.get('Temperature', np.ones(len(masses)) * 100)
    
    # Compute cylindrical coordinates
    R, phi, z, v_R, v_phi, v_z = compute_cylindrical_coords(coords, center, velocities)
    
    # Convert units
    #R_pc = R * CODE_LENGTH / PC_TO_CM
    R_pc = R * 1000
    masses_g = masses * CODE_MASS
    
    if r_max is None:
        r_max = np.percentile(R_pc, 95)
    
    # Radial bins
    r_bins = np.linspace(0, r_max, n_bins)
    r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
    
    # Compute surface density in each bin
    sigma_gas = np.zeros(len(r_centers))
    omega = np.zeros(len(r_centers))
    c_s = np.zeros(len(r_centers))
    
    for i in range(len(r_centers)):
        mask = (R_pc >= r_bins[i]) & (R_pc < r_bins[i+1])
        if np.sum(mask) > 0:
            # Surface density (mass per unit area)
            annulus_area = np.pi * (r_bins[i+1]**2 - r_bins[i]**2) * PC_TO_CM**2
            sigma_gas[i] = np.sum(masses_g[mask]) / annulus_area
            
            # Angular velocity (from v_phi)
            omega[i] = np.median(v_phi[mask] * CODE_VELOCITY / (R_pc[mask] * PC_TO_CM))
            
            # Sound speed (from temperature)
            mean_temp = np.mean(temps[mask])
            c_s[i] = np.sqrt(K_BOLTZ * mean_temp / (2.33 * M_PROTON))  # mean molecular weight ~ 2.33
    
    # Epicyclic frequency (for thin disk, kappa ~ sqrt(2) * Omega)
    # Inside compute_toomre_q
    L = (r_centers * PC_TO_CM)**2 * omega  # Angular momentum R^2 * Omega
    d_L2_dr = np.gradient(L, r_centers * PC_TO_CM)
    kappa = np.sqrt((2 * omega / (r_centers * PC_TO_CM)) * d_L2_dr)
    #kappa = np.sqrt(2) * np.abs(omega)
    
    # Toomre Q parameter
    Q = c_s * kappa / (np.pi * G_GRAV * sigma_gas)
    Q[sigma_gas == 0] = np.nan
    
    return r_centers, Q, sigma_gas, omega, c_s

def compute_energy_ratios(data_dict):
    """
    Compute kinetic energy components:
    - Rotational KE
    - Total KE
    - Turbulent KE
    Also compute virial parameter, Mach number
    """
    gas = data_dict['gas_data']
    center = data_dict['center']
    
    coords = gas['Coordinates']
    masses = gas['Masses']
    velocities = gas['Velocities']
    temps = gas.get('Temperature', np.ones(len(masses)) * 100)
    
    # Compute cylindrical coordinates
    R, phi, z, v_R, v_phi, v_z = compute_cylindrical_coords(coords, center, velocities)
    
    # Convert to physical units
    masses_g = masses * CODE_MASS
    v_R_cgs = v_R * CODE_VELOCITY
    v_phi_cgs = v_phi * CODE_VELOCITY
    v_z_cgs = v_z * CODE_VELOCITY
    
    # Kinetic energies
    KE_rot = 0.5 * np.sum(masses_g * v_phi_cgs**2)
    KE_radial = 0.5 * np.sum(masses_g * v_R_cgs**2)
    KE_vertical = 0.5 * np.sum(masses_g * v_z_cgs**2)
    KE_total = KE_rot + KE_radial + KE_vertical
    KE_turbulent = KE_radial + KE_vertical  # non-rotational
    
    # Gravitational energy (approximate - only gas self-gravity)
    total_mass = np.sum(masses_g)
    rel_coords = coords - center
    r = np.sqrt(np.sum(rel_coords**2, axis=1)) * CODE_LENGTH
    mean_r = np.mean(r)
    PE_grav = -G_GRAV * total_mass**2 / mean_r  # very rough estimate
    
    # Virial parameter: alpha_vir = 5 * sigma_v^2 * R / (G * M)
    v_R_disp = np.std(v_R_cgs) # Better: calculate per-bin to avoid radial gradients
    v_z_disp = np.std(v_z_cgs)
    sigma_v = np.sqrt(v_R_disp**2 + v_z_disp**2)
    #sigma_v = np.sqrt(np.sum(masses_g * (v_R_cgs**2 + v_z_cgs**2)) / total_mass)
    alpha_vir = 5 * sigma_v**2 * mean_r / (G_GRAV * total_mass)
    
    # Mach number
    mean_temp = np.mean(temps)
    c_s = np.sqrt(K_BOLTZ * mean_temp / (2.33 * M_PROTON))
    mach_number = sigma_v / c_s
    
    results = {
        'KE_rotational': KE_rot,
        'KE_total': KE_total,
        'KE_turbulent': KE_turbulent,
        'KE_radial': KE_radial,
        'KE_vertical': KE_vertical,
        'PE_gravitational': PE_grav,
        'virial_parameter': alpha_vir,
        'mach_number': mach_number,
        'velocity_dispersion': sigma_v,
        'sound_speed': c_s
    }
    
    return results

def compute_mass_to_flux_ratio(data_dict):
    """
    Compute dimensionless mass-to-flux ratio: mu = (M/Phi) / (M/Phi)_crit
    where (M/Phi)_crit = 1/(2*pi*sqrt(G)) in CGS
    
    This should be comparable to Yasmine's paper results.
    """
    gas = data_dict['gas_data']
    
    if 'MagneticField' not in gas:
        print("Warning: No magnetic field data available!")
        return None
    
    masses = gas['Masses']
    B_field = gas['MagneticField']  # in code units
    coords = gas['Coordinates']
    hsml = gas.get('SmoothingLength', np.ones(len(masses)) * 0.01)
    
    # Convert to physical units
    masses_g = masses * CODE_MASS
    B_cgs = B_field * np.sqrt(4 * np.pi)  # assuming B is in Gauss in snapshot
    
    # Total mass
    total_mass = np.sum(masses_g)
    
    # Estimate magnetic flux (rough estimate using B_z and area)
    # Better: integrate B·dA over a surface
    B_z = B_cgs[:, 2]
    areas = np.pi * (hsml * CODE_LENGTH)**2  # approximate area per particle
    magnetic_flux = np.sum(np.abs(B_z) * areas)
    
    # Critical mass-to-flux ratio
    critical_ratio = 1.0 / (2 * np.pi * np.sqrt(G_GRAV))  # in CGS
    
    # Dimensionless mass-to-flux ratio
    mass_to_flux = total_mass / magnetic_flux
    mu = mass_to_flux / critical_ratio
    
    return {
        'mu': mu,
        'mass_to_flux': mass_to_flux,
        'critical_mass_to_flux': critical_ratio,
        'total_mass': total_mass,
        'magnetic_flux': magnetic_flux
    }

def compute_turbulent_power_spectrum(data_dict, grid_size=128):
    """
    Compute turbulent kinetic energy power spectrum from velocity field.
    Returns k and E(k).
    """
    gas = data_dict['gas_data']
    center = data_dict['center']
    boxsize = data_dict['boxsize']
    
    coords = gas['Coordinates']
    velocities = gas['Velocities']
    masses = gas['Masses']
    
    # Subtract bulk velocity
    v_bulk = np.average(velocities, axis=0, weights=masses)
    v_turb = velocities - v_bulk
    
    # Create velocity field on grid using SPH kernel
    # Simplified: bin particles into grid cells
    rel_coords = coords - (center - boxsize/2)
    
    # Normalized coordinates [0, 1]
    norm_coords = rel_coords / boxsize
    
    # Grid indices
    i_grid = (norm_coords[:, 0] * grid_size).astype(int)
    j_grid = (norm_coords[:, 1] * grid_size).astype(int)
    k_grid = (norm_coords[:, 2] * grid_size).astype(int)
    
    # Mask particles outside box
    mask = ((i_grid >= 0) & (i_grid < grid_size) & 
            (j_grid >= 0) & (j_grid < grid_size) & 
            (k_grid >= 0) & (k_grid < grid_size))
    
    # Velocity field (mass-weighted)
    v_field = np.zeros((grid_size, grid_size, grid_size, 3))
    mass_field = np.zeros((grid_size, grid_size, grid_size))
    
    for idx in range(len(coords)):
        if mask[idx]:
            i, j, k = i_grid[idx], j_grid[idx], k_grid[idx]
            mass_field[i, j, k] += masses[idx]
            v_field[i, j, k] += masses[idx] * v_turb[idx]
    
    # Average velocity in each cell
    nonzero = mass_field > 0
    for dim in range(3):
        v_field[:, :, :, dim][nonzero] /= mass_field[nonzero]
    
    # Compute FFT of velocity components
    v_k = np.zeros((grid_size, grid_size, grid_size, 3), dtype=complex)
    for dim in range(3):
        v_k[:, :, :, dim] = fftn(v_field[:, :, :, dim])
    
    # Power spectrum: |v_k|^2
    power = np.sum(np.abs(v_k)**2, axis=-1)
    
    # Radial binning in k-space
    kx = fftfreq(grid_size, d=boxsize/grid_size)
    ky = fftfreq(grid_size, d=boxsize/grid_size)
    kz = fftfreq(grid_size, d=boxsize/grid_size)
    
    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
    
    # Bin power by k magnitude
    k_bins = np.logspace(np.log10(k_mag[k_mag > 0].min()), 
                        np.log10(k_mag.max()), 50)
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
    
    power_spectrum = np.zeros(len(k_centers))
    for i in range(len(k_centers)):
        mask_k = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        if np.sum(mask_k) > 0:
            power_spectrum[i] = np.mean(power[mask_k])
    
    return k_centers, power_spectrum

def compute_stellar_properties(data_dict):
    """
    Compute stellar multiplicity, IMF, SFE, SFR from PartType5 stars.
    """
    star_data = data_dict['star_data']
    time = data_dict['time']
    
    if not star_data or 'Masses' not in star_data:
        print("No star data available!")
        return None
    
    star_masses = star_data['Masses'] * CODE_MASS / M_SUN  # in solar masses
    star_coords = star_data['Coordinates']
    
    n_stars = len(star_masses)
    total_stellar_mass = np.sum(star_masses)
    
    print(f"Found {n_stars} stars with total mass {total_stellar_mass:.2e} M_sun")
    
    # IMF: histogram of stellar masses
    mass_bins = np.logspace(np.log10(star_masses.min()), 
                           np.log10(star_masses.max()), 30)
    hist, bin_edges = np.histogram(star_masses, bins=mass_bins)
    
    # Multiplicity: find stellar systems (stars close together)
    # Use spatial clustering
    if n_stars > 1:
        tree = cKDTree(star_coords)
        separation_threshold = 0.01  # code units (adjust based on resolution)
        pairs = tree.query_pairs(r=separation_threshold)
        
        # Count systems
        # Simple approach: stars within threshold are in a system
        single_stars = n_stars - 2 * len(pairs)  # rough estimate
        multiple_systems = len(pairs)
        
        multiplicity_fraction = multiple_systems / n_stars if n_stars > 0 else 0
    else:
        multiplicity_fraction = 0
        multiple_systems = 0
    
    # Star Formation Rate (need formation times)
    if 'StellarFormationTime' in star_data:
        formation_times = star_data['StellarFormationTime']
        # Compute SFR over some time window
        time_window = 0.1  # Myr in code units (adjust)
        recent_stars = np.sum(formation_times > (time - time_window))
        sfr = recent_stars / time_window if time_window > 0 else 0
    else:
        sfr = None
    
    # Star Formation Efficiency (need gas mass)
    gas_mass = np.sum(data_dict['gas_data']['Masses']) * CODE_MASS / M_SUN
    sfe = total_stellar_mass / (total_stellar_mass + gas_mass)
    
    return {
        'n_stars': n_stars,
        'total_stellar_mass': total_stellar_mass,
        'mean_stellar_mass': np.mean(star_masses),
        'median_stellar_mass': np.median(star_masses),
        'imf_bins': bin_edges,
        'imf_hist': hist,
        'multiplicity_fraction': multiplicity_fraction,
        'n_multiple_systems': multiple_systems,
        'star_formation_rate': sfr,
        'star_formation_efficiency': sfe,
        'gas_mass': gas_mass
    }

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_toomre_q(r_centers, Q, sigma_gas, output_path=None):
    """Plot Toomre Q parameter vs radius."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Toomre Q
    ax1.plot(r_centers, Q, 'o-', linewidth=2)
    ax1.axhline(y=1, color='r', linestyle='--', label='Q = 1 (marginal stability)')
    ax1.set_ylabel('Toomre Q', fontsize=14)
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Surface density
    ax2.plot(r_centers, sigma_gas, 'o-', linewidth=2, color='orange')
    ax2.set_xlabel('Radius [pc]', fontsize=14)
    ax2.set_ylabel('Surface Density [g/cm²]', fontsize=14)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_energy_ratios(energy_dict, output_path=None):
    """Plot kinetic energy components and other global properties."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # KE components
    ke_labels = ['Rotational', 'Turbulent', 'Radial', 'Vertical', 'Total']
    ke_values = [energy_dict['KE_rotational'], energy_dict['KE_turbulent'],
                energy_dict['KE_radial'], energy_dict['KE_vertical'], 
                energy_dict['KE_total']]
    
    ax1.bar(ke_labels, ke_values, color=['blue', 'orange', 'green', 'red', 'purple'])
    ax1.set_ylabel('Kinetic Energy [erg]', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_title('Kinetic Energy Components', fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    
    # KE ratios
    ke_rot_frac = energy_dict['KE_rotational'] / energy_dict['KE_total']
    ke_turb_frac = energy_dict['KE_turbulent'] / energy_dict['KE_total']
    
    ax2.bar(['Rotational', 'Turbulent'], [ke_rot_frac, ke_turb_frac],
           color=['blue', 'orange'])
    ax2.set_ylabel('Fraction of Total KE', fontsize=12)
    ax2.set_title('KE Fractions', fontsize=14)
    ax2.set_ylim(0, 1)
    
    # Virial parameter and Mach number
    params = ['Virial\nParameter', 'Mach\nNumber']
    values = [energy_dict['virial_parameter'], energy_dict['mach_number']]
    
    ax3.bar(params, values, color=['cyan', 'magenta'])
    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_title('Cloud Properties', fontsize=14)
    ax3.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    
    # Velocity dispersion vs sound speed
    ax4.bar(['Velocity\nDispersion', 'Sound\nSpeed'], 
           [energy_dict['velocity_dispersion'], energy_dict['sound_speed']],
           color=['green', 'brown'])
    ax4.set_ylabel('Velocity [cm/s]', fontsize=12)
    ax4.set_title('Velocities', fontsize=14)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_power_spectrum(k, power, output_path=None):
    """Plot turbulent KE power spectrum."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.loglog(k, power, 'o-', linewidth=2, markersize=4)
    
    # Add Kolmogorov reference line (E(k) ~ k^(-5/3))
    k_ref = k[len(k)//4:3*len(k)//4]
    power_ref = power[len(k)//4] * (k_ref / k[len(k)//4])**(-5./3.)
    ax.loglog(k_ref, power_ref, '--', color='red', linewidth=2, 
             label='Kolmogorov: $k^{-5/3}$', alpha=0.7)
    
    ax.set_xlabel('Wavenumber k [1/code units]', fontsize=14)
    ax.set_ylabel('Power Spectrum E(k)', fontsize=14)
    ax.set_title('Turbulent Kinetic Energy Power Spectrum', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_imf(stellar_props, output_path=None):
    """Plot Initial Mass Function."""
    if stellar_props is None:
        print("No stellar properties to plot!")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot IMF
    bin_centers = 0.5 * (stellar_props['imf_bins'][1:] + stellar_props['imf_bins'][:-1])
    bin_widths = stellar_props['imf_bins'][1:] - stellar_props['imf_bins'][:-1]
    
    # Normalize: dN/dlog(M)
    imf_normalized = stellar_props['imf_hist'] / (bin_widths / bin_centers)
    
    ax.loglog(bin_centers, imf_normalized, 'o-', linewidth=2, markersize=8)
    
    # Add Salpeter reference (dN/dM ~ M^-2.35)
    m_ref = bin_centers
    salpeter = imf_normalized[len(bin_centers)//2] * (m_ref / bin_centers[len(bin_centers)//2])**(-1.35)
    ax.loglog(m_ref, salpeter, '--', color='red', linewidth=2, 
             label='Salpeter: $M^{-1.35}$', alpha=0.7)
    
    ax.set_xlabel('Stellar Mass [M$_\\odot$]', fontsize=14)
    ax.set_ylabel('dN/dlog(M)', fontsize=14)
    ax.set_title(f'Initial Mass Function (N={stellar_props["n_stars"]} stars)', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_stellar_summary(stellar_props, output_path=None):
    """Plot summary of stellar properties."""
    if stellar_props is None:
        print("No stellar properties to plot!")
        return None
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Number of stars and multiplicity
    ax1.bar(['Single\nStars', 'Multiple\nSystems'], 
           [stellar_props['n_stars'] - stellar_props['n_multiple_systems'],
            stellar_props['n_multiple_systems']],
           color=['blue', 'orange'])
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Stellar Multiplicity', fontsize=14)
    
    # Mass distribution
    ax2.text(0.5, 0.8, f"Total Stars: {stellar_props['n_stars']}", 
            transform=ax2.transAxes, ha='center', fontsize=14)
    ax2.text(0.5, 0.6, f"Total Stellar Mass: {stellar_props['total_stellar_mass']:.2e} M$_\\odot$",
            transform=ax2.transAxes, ha='center', fontsize=12)
    ax2.text(0.5, 0.4, f"Mean Mass: {stellar_props['mean_stellar_mass']:.2f} M$_\\odot$",
            transform=ax2.transAxes, ha='center', fontsize=12)
    ax2.text(0.5, 0.2, f"Median Mass: {stellar_props['median_stellar_mass']:.2f} M$_\\odot$",
            transform=ax2.transAxes, ha='center', fontsize=12)
    ax2.axis('off')
    ax2.set_title('Mass Statistics', fontsize=14)
    
    # Star formation efficiency
    ax3.bar(['Gas', 'Stars'], 
           [stellar_props['gas_mass'], stellar_props['total_stellar_mass']],
           color=['green', 'red'])
    ax3.set_ylabel('Mass [M$_\\odot$]', fontsize=12)
    ax3.set_yscale('log')
    ax3.set_title(f'SFE = {stellar_props["star_formation_efficiency"]:.3f}', fontsize=14)
    
    # Multiplicity fraction
    ax4.pie([stellar_props['multiplicity_fraction'], 
            1 - stellar_props['multiplicity_fraction']],
           labels=['Multiple', 'Single'],
           colors=['orange', 'blue'],
           autopct='%1.1f%%',
           startangle=90)
    ax4.set_title('Multiplicity Fraction', fontsize=14)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig
