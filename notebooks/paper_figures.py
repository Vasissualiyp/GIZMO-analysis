"""
paper_figures.py
----------------
Multi-epoch figure assembly for SHIVAN paper.
Loads 6 key snapshots, extracts all quantities, and produces publication-quality
multi-panel figures combining all epochs.

Called by generate_paper_plots.py — not intended for standalone use.
"""

import os
import sys
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import LogLocator, NullFormatter, AutoMinorLocator, MultipleLocator
from meshoid import Meshoid

try:
    import cmasher as cmr
    _ECLIPSE_CMAP = 'cmr.eclipse'
except ImportError:
    cmr = None
    # Register eclipse colormap manually from embedded CMasher data so the
    # fallback is the real colormap, not just inferno.
    # Source: https://github.com/1313e/CMasher colormaps/eclipse/eclipse_norm.txt
    _eclipse_rgb = np.array([
        [0.000000,0.000000,0.000000],[0.000208,0.000211,0.000265],[0.000713,0.000731,0.000949],
        [0.001462,0.001514,0.002030],[0.002431,0.002542,0.003507],[0.003601,0.003801,0.005384],
        [0.004958,0.005285,0.007671],[0.006490,0.006987,0.010378],[0.008189,0.008902,0.013516],
        [0.010043,0.011028,0.017097],[0.012046,0.013362,0.021134],[0.014187,0.015903,0.025640],
        [0.016460,0.018650,0.030628],[0.018856,0.021601,0.036113],[0.021368,0.024757,0.042062],
        [0.023987,0.028117,0.048063],[0.026707,0.031682,0.054053],[0.029519,0.035452,0.060036],
        [0.032415,0.039429,0.066015],[0.035387,0.043493,0.071994],[0.038427,0.047512,0.077975],
        [0.041499,0.051500,0.083961],[0.044483,0.055460,0.089953],[0.047384,0.059395,0.095954],
        [0.050201,0.063308,0.101964],[0.052934,0.067200,0.107984],[0.055584,0.071076,0.114016],
        [0.058149,0.074935,0.120061],[0.060629,0.078782,0.126118],[0.063021,0.082617,0.132188],
        [0.065326,0.086444,0.138272],[0.067540,0.090262,0.144370],[0.069662,0.094075,0.150480],
        [0.071689,0.097884,0.156603],[0.073619,0.101691,0.162739],[0.075449,0.105498,0.168885],
        [0.077175,0.109305,0.175043],[0.078795,0.113115,0.181209],[0.080304,0.116929,0.187383],
        [0.081697,0.120750,0.193563],[0.082970,0.124577,0.199747],[0.084119,0.128414,0.205932],
        [0.085137,0.132262,0.212117],[0.086019,0.136121,0.218297],[0.086758,0.139995,0.224469],
        [0.087347,0.143884,0.230631],[0.087779,0.147791,0.236776],[0.088045,0.151716,0.242901],
        [0.088137,0.155662,0.249000],[0.088045,0.159629,0.255067],[0.087760,0.163620,0.261095],
        [0.087271,0.167637,0.267075],[0.086566,0.171679,0.273001],[0.085633,0.175750,0.278862],
        [0.084461,0.179851,0.284648],[0.083035,0.183981,0.290348],[0.081344,0.188142,0.295949],
        [0.079373,0.192336,0.301436],[0.077111,0.196561,0.306796],[0.074548,0.200818,0.312013],
        [0.071673,0.205106,0.317069],[0.068481,0.209423,0.321947],[0.064972,0.213767,0.326629],
        [0.061149,0.218134,0.331097],[0.057031,0.222522,0.335336],[0.052642,0.226924,0.339329],
        [0.048024,0.231334,0.343065],[0.043242,0.235746,0.346534],[0.038380,0.240152,0.349732],
        [0.033798,0.244545,0.352657],[0.029720,0.248916,0.355314],[0.026254,0.253259,0.357713],
        [0.023498,0.257566,0.359867],[0.021536,0.261833,0.361791],[0.020438,0.266055,0.363505],
        [0.020258,0.270228,0.365029],[0.021035,0.274350,0.366383],[0.022795,0.278419,0.367589],
        [0.025554,0.282436,0.368664],[0.029318,0.286400,0.369628],[0.034085,0.290313,0.370498],
        [0.039850,0.294175,0.371288],[0.046249,0.297988,0.372012],[0.052931,0.301754,0.372683],
        [0.059808,0.305475,0.373312],[0.066813,0.309153,0.373907],[0.073895,0.312791,0.374478],
        [0.081015,0.316390,0.375031],[0.088146,0.319953,0.375573],[0.095269,0.323481,0.376109],
        [0.102368,0.326976,0.376644],[0.109434,0.330441,0.377183],[0.116459,0.333877,0.377728],
        [0.123437,0.337287,0.378282],[0.130366,0.340671,0.378849],[0.137243,0.344031,0.379431],
        [0.144066,0.347369,0.380029],[0.150835,0.350687,0.380645],[0.157550,0.353986,0.381280],
        [0.164211,0.357266,0.381936],[0.170819,0.360530,0.382613],[0.177374,0.363778,0.383313],
        [0.183878,0.367013,0.384035],[0.190332,0.370234,0.384780],[0.196736,0.373443,0.385548],
        [0.203093,0.376641,0.386340],[0.209403,0.379829,0.387155],[0.215667,0.383008,0.387994],
        [0.221886,0.386178,0.388856],[0.228063,0.389341,0.389741],[0.234197,0.392498,0.390648],
        [0.240291,0.395648,0.391578],[0.246346,0.398794,0.392529],[0.252362,0.401935,0.393501],
        [0.258340,0.405073,0.394494],[0.264283,0.408208,0.395506],[0.270191,0.411341,0.396536],
        [0.276065,0.414471,0.397584],[0.281906,0.417601,0.398649],[0.287715,0.420731,0.399729],
        [0.293495,0.423860,0.400823],[0.299245,0.426990,0.401930],[0.304967,0.430121,0.403048],
        [0.310662,0.433253,0.404175],[0.316332,0.436388,0.405311],[0.321978,0.439524,0.406453],
        [0.327601,0.442665,0.407600],[0.333203,0.445806,0.408748],[0.338786,0.448951,0.409896],
        [0.344350,0.452100,0.411043],[0.349899,0.455252,0.412185],[0.355433,0.458408,0.413321],
        [0.360955,0.461568,0.414446],[0.366467,0.464731,0.415560],[0.371972,0.467898,0.416659],
        [0.377472,0.471068,0.417741],[0.382969,0.474241,0.418802],[0.388466,0.477417,0.419840],
        [0.393967,0.480596,0.420853],[0.399473,0.483776,0.421838],[0.404989,0.486958,0.422791],
        [0.410518,0.490140,0.423710],[0.416063,0.493323,0.424593],[0.421626,0.496504,0.425437],
        [0.427213,0.499684,0.426241],[0.432826,0.502861,0.427003],[0.438468,0.506035,0.427721],
        [0.444143,0.509204,0.428393],[0.449853,0.512368,0.429019],[0.455602,0.515526,0.429598],
        [0.461392,0.518676,0.430128],[0.467226,0.521817,0.430611],[0.473107,0.524950,0.431047],
        [0.479035,0.528072,0.431434],[0.485013,0.531183,0.431774],[0.491043,0.534282,0.432070],
        [0.497124,0.537282,0.432320],[0.503258,0.540443,0.432525],[0.509446,0.543550,0.432687],
        [0.515688,0.546550,0.432809],[0.521982,0.549583,0.432891],[0.528330,0.552601,0.432933],
        [0.534731,0.555595,0.432939],[0.541184,0.558594,0.432909],[0.547688,0.561568,0.432844],
        [0.554242,0.564528,0.432746],[0.560844,0.567474,0.432616],[0.567495,0.570405,0.432454],
        [0.574192,0.573323,0.432262],[0.580934,0.576227,0.432040],[0.587720,0.579118,0.431789],
        [0.594549,0.581996,0.431507],[0.601419,0.584862,0.431198],[0.608329,0.587715,0.430858],
        [0.615276,0.590557,0.430490],[0.622261,0.593389,0.430092],[0.629282,0.596209,0.429664],
        [0.636338,0.599020,0.429204],[0.643426,0.601821,0.428713],[0.650547,0.604614,0.428188],
        [0.657698,0.607175,0.427032],[0.664879,0.610095,0.426438],[0.672088,0.612901,0.425798],
        [0.679324,0.615868,0.425124],[0.686586,0.618468,0.424006],[0.693872,0.621222,0.423244],
        [0.701180,0.623973,0.422435],[0.708510,0.626721,0.421574],[0.715860,0.629468,0.420658],
        [0.723226,0.632214,0.419684],[0.730609,0.634962,0.418647],[0.738004,0.637713,0.417543],
        [0.745410,0.640468,0.416368],[0.752823,0.643229,0.415115],[0.760240,0.646000,0.413778],
        [0.767657,0.648781,0.412352],[0.775067,0.651578,0.410829],[0.782467,0.654393,0.409201],
        [0.789848,0.657230,0.406462],[0.797203,0.660095,0.406602],[0.804520,0.662994,0.404615],
        [0.811788,0.665933,0.402492],[0.818992,0.668922,0.400226],[0.826114,0.671971,0.397813],
        [0.833133,0.675090,0.395251],[0.840025,0.678293,0.392542],[0.846762,0.681594,0.389695],
        [0.853314,0.685010,0.386726],[0.859652,0.688554,0.383662],[0.865746,0.692242,0.380538],
        [0.871572,0.696083,0.377396],[0.877114,0.700086,0.374283],[0.882365,0.704250,0.371243],
        [0.887329,0.708574,0.368318],[0.892015,0.712950,0.365536],[0.896442,0.717667,0.362920],
        [0.900631,0.722414,0.360481],[0.904604,0.727280,0.358222],[0.908385,0.732252,0.356139],
        [0.911985,0.737320,0.354229],[0.915434,0.742475,0.352479],[0.918744,0.747708,0.350879],
        [0.921930,0.753011,0.349421],[0.925004,0.758379,0.348092],[0.927979,0.763807,0.346881],
        [0.930862,0.769289,0.345777],[0.933662,0.774822,0.344773],[0.936388,0.780401,0.343859],
        [0.939045,0.786025,0.343027],[0.941638,0.791695,0.342271],[0.944173,0.797395,0.341581],
        [0.946653,0.803137,0.340954],[0.949080,0.808916,0.340380],[0.951461,0.814728,0.339859],
        [0.953795,0.820572,0.339379],[0.956088,0.826453,0.338942],[0.958340,0.832362,0.338542],
        [0.960552,0.838303,0.338168],[0.962728,0.844273,0.337826],[0.964869,0.850273,0.337508],
        [0.966975,0.856301,0.337211],[0.969048,0.862358,0.336932],[0.971088,0.868443,0.336667],
        [0.973097,0.874557,0.336414],[0.975025,0.880695,0.336170],[0.976944,0.886865,0.335933],
        [0.978834,0.893060,0.335700],[0.980694,0.899282,0.335468],[0.982544,0.905532,0.335236],
        [0.984385,0.911808,0.335002],[0.986175,0.918111,0.334765],[0.987862,0.924441,0.334519],
        [0.989860,0.930800,0.334263],[0.991584,0.937184,0.334001],[0.993278,0.943596,0.333723],
        [0.994943,0.950036,0.333428],
    ])
    _ec_cmap = matplotlib.colors.ListedColormap(_eclipse_rgb, name='cmr.eclipse')
    _ec_cmap_r = _ec_cmap.reversed()
    try:
        matplotlib.colormaps.register(_ec_cmap)
        matplotlib.colormaps.register(_ec_cmap_r)
    except Exception:
        pass
    _ECLIPSE_CMAP = 'cmr.eclipse'

from generic_utils.constants import kpc, AU, Msun, G
from hybrid_sims_utils.read_snap import get_snap_data_hybrid, convert_units_to_physical
from notebooks.make_disk_movie_frames import (
    identify_disk, rotation_matrix_to_z, _save_fig_dual, _scale_to_Myr,
)

# ── Global font scaling ──
# Doubled for publication readability (2× previous sizes)
_ORIG_RC = {
    'font.size': 20, 'axes.labelsize': 22, 'axes.titlesize': 22,
    'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 16,
}
_REDUCED_BORDER_RC = {'axes.linewidth': 1.4}

def _use_orig_rc(func):
    import functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with plt.rc_context(_ORIG_RC):
            return func(*args, **kwargs)
    return wrapper

def _use_thin_border(func):
    import functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with plt.rc_context(_REDUCED_BORDER_RC):
            return func(*args, **kwargs)
    return wrapper

plt.rcParams.update({
    'font.size': 40,
    'axes.labelsize': 44,
    'axes.titlesize': 44,
    'xtick.labelsize': 36,
    'ytick.labelsize': 36,
    'legend.fontsize': 32,
    'xtick.major.size': 8,
    'xtick.minor.size': 4,
    'ytick.major.size': 8,
    'ytick.minor.size': 4,
    'xtick.major.width': 2.4,
    'xtick.minor.width': 1.6,
    'ytick.major.width': 2.4,
    'ytick.minor.width': 1.6,
    'axes.linewidth': 2.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
})

# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

EPOCHS = [
    {"snap": 28,  "label": "Pre-stellar"},
    {"snap": 50,  "label": "First sink"},
    {"snap": 100, "label": "Early disk", "alt_snap": 96},
    {"snap": 193, "label": "GI onset"},
    {"snap": 429, "label": "Late-time"},
    {"snap": 646, "label": "Extended"},
]

EPOCH_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

N_BINS = 20       # coarse radial bins (velocities, Q, mu, dx)
N_RHO  = 100      # fine radial bins (density profile)

# Full-simulation path for phase diagram background
_FULLSIM_PATH = None  # set by generate_paper_plots.py before calling make_all_figures
# Wide-sim path for 500 pc profiles (cutout sim with small files)
_WIDESIM_PATH = None  # set by generate_paper_plots.py; falls back to _FULLSIM_PATH


def _load_fullsim_wide_gas(snap_num, fullsim_dir, hdr, com, com_vel, r_max_kpc=2.0):
    """Load gas particles from the full simulation for wide-radius velocity profiles.

    Returns dict with keys 'coords_kpc', 'vel_kms', 'mass', 'u' (internal energy),
    already with com subtracted from coords and com_vel subtracted from velocities.
    Only particles within r_max_kpc of com are returned. Returns None on failure.
    """
    import h5py as _h5
    snap_path = None
    for fmt in [f'snapshot_{snap_num:04d}.hdf5', f'snapshot_{snap_num:03d}.hdf5']:
        _p = os.path.join(fullsim_dir, fmt)
        if os.path.exists(_p):
            snap_path = _p
            break
    if snap_path is None:
        return None

    try:
        with _h5.File(snap_path, 'r') as f:
            if 'PartType0' not in f:
                return None
            coords_raw = f['PartType0/Coordinates'][:].astype(np.float64)
            vel_raw    = f['PartType0/Velocities'][:].astype(np.float64)
            mass_raw   = f['PartType0/Masses'][:].astype(np.float64)
            u_raw = (f['PartType0/InternalEnergy'][:].astype(np.float64)
                     if 'InternalEnergy' in f['PartType0'] else None)
    except Exception as e:
        print(f"    _load_fullsim_wide_gas: failed {snap_path}: {e}")
        return None

    # Unit conversion: comoving kpc/h → physical kpc, velocities → km/s
    a = float(hdr['Time'])
    h = float(hdr.get('HubbleParam', 0.7))
    coords_kpc = coords_raw * (a / h)
    vel_kms    = vel_raw * np.sqrt(a)
    mass       = mass_raw / h  # 1e10 Msun

    # Select within r_max_kpc of com
    dists = np.linalg.norm(coords_kpc - com, axis=1)
    mask = dists < r_max_kpc
    if mask.sum() < 5:
        return None

    n_sel = mask.sum()
    r_max_sel = np.percentile(np.linalg.norm(coords_kpc[mask] - com, axis=1), 99) * 1e3  # pc
    print(f"    _load_fullsim_wide_gas: {n_sel} particles within {r_max_kpc:.1f} kpc "
          f"(99th pct r = {r_max_sel:.1f} pc)")

    _com_vel = com_vel if com_vel is not None else np.zeros(3)
    return {
        'coords_kpc': coords_kpc[mask] - com,
        'vel_kms': vel_kms[mask] - _com_vel,
        'mass': mass[mask],
        'u': u_raw[mask] if u_raw is not None else None,
    }


def load_fullsim_phase(snap_num, fullsim_dir):
    """Load T, n, and optionally H₂ fraction from full simulation for phase diagram.

    Returns (n_arr, T_arr, fh2_arr) subsampled to ~200K particles.
    fh2_arr is None if MolecularMassFraction is not in the snapshot.
    Returns (None, None, None) on failure.
    """
    import h5py as _h5
    snap_path = os.path.join(fullsim_dir, f'snapshot_{snap_num:03d}.hdf5')
    if not os.path.exists(snap_path):
        snap_path = os.path.join(fullsim_dir, f'snapshot_{snap_num}.hdf5')
    if not os.path.exists(snap_path):
        print(f"    Full-sim snapshot not found: {snap_path}")
        return None, None, None

    try:
        with _h5.File(snap_path, 'r') as f:
            n_total = f['PartType0/Density'].shape[0]
            stride = max(1, n_total // 200000)
            rho = f['PartType0/Density'][::stride].astype(np.float64)
            u = f['PartType0/InternalEnergy'][::stride].astype(np.float64)
            hdr = dict(f['Header'].attrs)

            # Try to load H₂ fraction
            fh2_raw = None
            if 'MolecularMassFraction' in f['PartType0']:
                fh2_raw = f['PartType0/MolecularMassFraction'][::stride].astype(np.float64)

        # Convert units: density to cgs
        rho_cgs = rho * 1e10 * Msun / kpc**3
        n_arr = rho_cgs / _m_p

        # Temperature: T = mu * m_p * (gamma-1) * u_cgs / k_B
        k_B = 1.381e-16
        mu_mol = 1.22
        u_cgs = u * 1e10  # (km/s)^2 → cm^2/s^2
        T_arr = mu_mol * _m_p * (_GAMMA - 1.0) * u_cgs / k_B

        valid = (n_arr > 0) & (T_arr > 0)
        fh2_out = fh2_raw[valid] if fh2_raw is not None else None
        return n_arr[valid], T_arr[valid], fh2_out
    except Exception as e:
        print(f"    Error loading full-sim phase data: {e}")
        return None, None, None
_GAMMA = 5.0 / 3.0
_m_p   = 1.673e-24  # proton mass [g]

# ═════════════════════════════════════════════════════════════════════════════
# Data extraction
# ═════════════════════════════════════════════════════════════════════════════

def extract_epoch_data(snap_num, args, sink_form_time_Myr=None):
    """Load one snapshot and extract all quantities needed for paper figures.

    Parameters
    ----------
    snap_num : int
    args : Defaults-like object with path, sim, image_box, r_search, r_max, etc.
    sink_form_time_Myr : float or None
        Time of first sink formation (Myr). If known, used for label.

    Returns
    -------
    dict with all 2D maps, 1D profiles, phase data, metadata.
    """
    res = args.res
    image_box_kpc = args.image_box

    gas_fields = ['Masses', 'Coordinates', 'SmoothingLength',
                  'Velocities', 'Density', 'ParticleIDs', 'InternalEnergy',
                  'MagneticField']
    # Try to load H2 field
    h2_field = None
    for candidate in ['MolecularMassFraction', 'Molecular_Fraction',
                      'MolecularHydrogenFraction', 'H2Fraction']:
        gas_fields.append(candidate)

    hdr, pdata, stardata, fsd, _, _ = get_snap_data_hybrid(
        args.sim, args.path, snap_num, snapshot_suffix='', snapdir=False,
        refinement_tag=False, verbose=False, custom_gas_fields=gas_fields)
    hdr, pdata, stardata, fsd = convert_units_to_physical(hdr, pdata, stardata, fsd)

    time_Myr = _scale_to_Myr(hdr['Time'])

    # Identify disk
    is_disk, com, L_hat, r_cyl, z, v_phi_disk, v_K, com_vel = identify_disk(
        pdata, stardata,
        r_search_kpc=args.r_search, r_max_kpc=args.r_max,
        rho_threshold_cgs=args.rho_thresh, aspect_ratio=args.aspect,
        f_kep=args.f_kep, use_bounds=True,
        reference_center=getattr(args, 'reference_center', None),
        reference_search_radius=getattr(args, 'reference_search_radius', 0.1),
    )

    rot = rotation_matrix_to_z(L_hat)

    # Co-rotating frame
    phi_ref = 0.0
    if getattr(args, 'corotate', True) and stardata and len(stardata.get('Masses', [])) > 0:
        idx_ms = np.argmax(stardata['Masses'])
        ref_disk = (stardata['Coordinates'][idx_ms] - com) @ rot.T
        phi_ref = np.arctan2(ref_disk[1], ref_disk[0])
    c_phi, s_phi = np.cos(-phi_ref), np.sin(-phi_ref)
    R_ip = np.array([[c_phi, -s_phi, 0.], [s_phi, c_phi, 0.], [0., 0., 1.]])
    rot_fo = R_ip @ rot

    gas_dists = np.linalg.norm(pdata['Coordinates'] - com, axis=1)

    # Small-box particles
    cut_small = gas_dists < image_box_kpc * 0.75
    coords_s = pdata['Coordinates'][cut_small] - com
    pos_fo = coords_s @ rot_fo.T
    pos_small = coords_s @ rot.T
    mass_small = pdata['Masses'][cut_small]
    hsml_small = pdata['SmoothingLength'][cut_small]
    pos_edge = pos_small[:, [0, 2, 1]]  # swap y<->z for edge-on

    center0 = np.zeros(3)
    extent_AU = image_box_kpc * kpc / AU
    half_AU = extent_AU / 2
    ax_AU = np.linspace(-half_AU, half_AU, res)
    X, Y = np.meshgrid(ax_AU, ax_AU, indexing='ij')

    # ── Surface density projections ──
    def _surf(pos, mass, hsml, size):
        if len(pos) == 0:
            return np.zeros((res, res)), None
        M = Meshoid(pos, mass, hsml)
        return M.SurfaceDensity(M.m * 1e10, center=center0, size=size, res=res) / 1e6, M

    sig_fo, M_fo = _surf(pos_fo, mass_small, hsml_small, image_box_kpc)
    sig_eo, M_eo = _surf(pos_edge, mass_small, hsml_small, image_box_kpc)

    # ── Velocity decomposition ──
    vel_raw = pdata['Velocities'][cut_small]
    vel_com = vel_raw - (com_vel if com_vel is not None else np.zeros(3))
    vel_rot = vel_com @ rot.T

    r_xy = np.linalg.norm(pos_small[:, :2], axis=1)
    safe_rxy = np.maximum(r_xy, 1e-30)
    e_r_x = pos_small[:, 0] / safe_rxy
    e_r_y = pos_small[:, 1] / safe_rxy
    v_r =  vel_rot[:, 0] * e_r_x + vel_rot[:, 1] * e_r_y
    v_phi = -vel_rot[:, 0] * e_r_y + vel_rot[:, 1] * e_r_x
    v_z = vel_rot[:, 2]

    r_outer = np.percentile(r_xy, 95) if len(r_xy) > 0 else 1.0
    r_outer = max(r_outer, 1e-20)
    bins = np.linspace(0.0, r_outer, N_BINS + 1)
    bidx = np.clip(np.digitize(r_xy, bins) - 1, 0, N_BINS - 1)
    bin_centers_kpc = (bins[:-1] + bins[1:]) / 2

    vr_prof = np.zeros(N_BINS)
    vphi_prof = np.zeros(N_BINS)
    for b in range(N_BINS):
        mb = bidx == b
        if mb.sum() > 0:
            w = mass_small[mb]; wsum = w.sum()
            vr_prof[b] = np.dot(v_r[mb], w) / wsum
            vphi_prof[b] = np.dot(v_phi[mb], w) / wsum

    dv_r = v_r - vr_prof[bidx]
    dv_phi = v_phi - vphi_prof[bidx]
    dv_z = v_z
    v_rest = np.sqrt(dv_r**2 + dv_phi**2 + dv_z**2)

    # ── Sound speed (computed early — needed for Q_combined) ──
    if 'InternalEnergy' in pdata:
        u_small = pdata['InternalEnergy'][cut_small]
        cs_small = np.sqrt(_GAMMA * (_GAMMA - 1.0) * np.maximum(u_small, 0.0))
    else:
        cs_small = np.zeros(len(mass_small))

    # ── Face-on sound-speed map (mass-weighted, for Q_fo_combined) ──
    if M_fo is not None and len(cs_small) > 0:
        _sm_cs = np.maximum(M_fo.SurfaceDensity(M_fo.m, center=center0, size=image_box_kpc, res=res), 1e-40)
        cs_fo_map = M_fo.SurfaceDensity(M_fo.m * cs_small, center=center0, size=image_box_kpc, res=res) / _sm_cs
    else:
        cs_fo_map = np.zeros((res, res))

    # ── Toomre Q = σ_r · κ / (π · G · Σ) ──
    # κ computed numerically: κ² = (2Ω/r) · d(r²Ω)/dr
    # where Ω = |vphi(r)|/r from the measured rotation profile.
    Sigma_prof = np.zeros(N_BINS)
    sigma_r_prof = np.zeros(N_BINS)
    Omega_prof = np.zeros(N_BINS)

    for b in range(N_BINS):
        mb = bidx == b
        if mb.sum() == 0:
            continue
        r_lo, r_hi = bins[b], bins[b + 1]
        area_kpc2 = np.pi * max(r_hi**2 - r_lo**2, 1e-40)
        w = mass_small[mb]; wsum = w.sum()
        Sigma_prof[b] = wsum * 1e10 * Msun / (area_kpc2 * kpc**2)
        vr2_mw = np.dot(v_r[mb]**2, w) / wsum
        sigma_r_prof[b] = np.sqrt(max(vr2_mw - vr_prof[b]**2, 0.0))
        r_mid = bin_centers_kpc[b]
        if r_mid > 0 and vphi_prof[b] != 0:
            Omega_prof[b] = abs(vphi_prof[b]) / r_mid
        else:
            Omega_prof[b] = 0.0

    # Epicyclic frequency: κ² = (2Ω/r) · d(r²Ω)/dr
    kappa_prof = np.zeros(N_BINS)
    r2Omega = bin_centers_kpc**2 * Omega_prof
    for b in range(N_BINS):
        r_b = bin_centers_kpc[b]
        if r_b <= 0 or Omega_prof[b] <= 0:
            continue
        if b > 0 and b < N_BINS - 1 and Omega_prof[b-1] > 0 and Omega_prof[b+1] > 0:
            dr = bin_centers_kpc[b+1] - bin_centers_kpc[b-1]
            d_r2Omega = r2Omega[b+1] - r2Omega[b-1]
        elif b < N_BINS - 1 and Omega_prof[b+1] > 0:
            dr = bin_centers_kpc[b+1] - bin_centers_kpc[b]
            d_r2Omega = r2Omega[b+1] - r2Omega[b]
        elif b > 0 and Omega_prof[b-1] > 0:
            dr = bin_centers_kpc[b] - bin_centers_kpc[b-1]
            d_r2Omega = r2Omega[b] - r2Omega[b-1]
        else:
            kappa_prof[b] = Omega_prof[b]
            continue
        kappa_sq = (2.0 * Omega_prof[b] / r_b) * (d_r2Omega / max(dr, 1e-30))
        kappa_prof[b] = np.sqrt(max(kappa_sq, 0.0))

    with np.errstate(divide='ignore', invalid='ignore'):
        Q_prof = np.where(
            (Sigma_prof > 0) & (kappa_prof > 0),
            (sigma_r_prof * 1e5) * (kappa_prof * 1e5 / kpc) / (np.pi * G * Sigma_prof),
            np.nan)

    # cs_prof (needed for Q_combined; cs_small already computed above)
    cs_prof = np.zeros(N_BINS)
    for b in range(N_BINS):
        mb = bidx == b
        if mb.sum() > 0:
            w = mass_small[mb]; wsum = w.sum()
            cs_prof[b] = np.dot(cs_small[mb], w) / wsum

    with np.errstate(divide='ignore', invalid='ignore'):
        _kappa_1d = kappa_prof * 1e5 / kpc
        Q_prof_combined = np.where(
            (Sigma_prof > 0) & (kappa_prof > 0),
            (np.sqrt(sigma_r_prof**2 + cs_prof**2) * 1e5) * _kappa_1d / (np.pi * G * Sigma_prof),
            np.nan)
        Q_prof_therm = np.where(
            (Sigma_prof > 0) & (kappa_prof > 0),
            (cs_prof * 1e5) * _kappa_1d / (np.pi * G * Sigma_prof),
            np.nan)

    # Face-on Q map
    if M_fo is not None and len(v_r) > 0:
        _sm = np.maximum(M_fo.SurfaceDensity(M_fo.m, center=center0, size=image_box_kpc, res=res), 1e-40)
        _vrm = M_fo.SurfaceDensity(M_fo.m * v_r, center=center0, size=image_box_kpc, res=res) / _sm
        _v2m = M_fo.SurfaceDensity(M_fo.m * v_r**2, center=center0, size=image_box_kpc, res=res) / _sm
        sigma_r_fo = np.sqrt(np.maximum(_v2m - _vrm**2, 0.0))
    else:
        sigma_r_fo = np.zeros((res, res))

    r_px_kpc = np.sqrt(X**2 + Y**2) * AU / kpc
    kappa_px_cgs = np.interp(r_px_kpc, bin_centers_kpc, kappa_prof,
                              left=0.0, right=0.0) * 1e5 / kpc
    pc_cm = kpc / 1e3
    Sigma_fo_gcm2 = np.maximum(sig_fo, 0.0) * Msun / pc_cm**2
    with np.errstate(divide='ignore', invalid='ignore'):
        Q_fo = np.where(Sigma_fo_gcm2 > 0,
                        (sigma_r_fo * 1e5) * kappa_px_cgs / (np.pi * G * Sigma_fo_gcm2),
                        np.nan)
    Q_fo = np.where(np.isfinite(Q_fo), Q_fo, 0.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_eff_fo = np.sqrt(np.maximum(sigma_r_fo**2 + cs_fo_map**2, 0.0))
        Q_fo_combined = np.where(Sigma_fo_gcm2 > 0,
                                 (sigma_eff_fo * 1e5) * kappa_px_cgs / (np.pi * G * Sigma_fo_gcm2),
                                 np.nan)
    Q_fo_combined = np.where(np.isfinite(Q_fo_combined), Q_fo_combined, 0.0)

    # ── Velocity dispersion maps (mass-weighted |δv|) ──
    if M_fo is not None and len(v_rest) > 0:
        _sm_fo = np.maximum(M_fo.SurfaceDensity(M_fo.m, center=center0, size=image_box_kpc, res=res), 1e-40)
        vdisp_fo = M_fo.SurfaceDensity(M_fo.m * v_rest, center=center0, size=image_box_kpc, res=res) / _sm_fo
    else:
        vdisp_fo = np.zeros((res, res))
    if M_eo is not None and len(v_rest) > 0:
        _sm_eo = np.maximum(M_eo.SurfaceDensity(M_eo.m, center=center0, size=image_box_kpc, res=res), 1e-40)
        vdisp_eo = M_eo.SurfaceDensity(M_eo.m * v_rest, center=center0, size=image_box_kpc, res=res) / _sm_eo
    else:
        vdisp_eo = np.zeros((res, res))

    # ── Density profile (fine bins) ──
    bins_rho = np.linspace(0.0, r_outer, N_RHO + 1)
    bidx_rho = np.clip(np.digitize(r_xy, bins_rho) - 1, 0, N_RHO - 1)
    bin_ctr_rho_AU = (bins_rho[:-1] + bins_rho[1:]) / 2 * kpc / AU

    rho_cgs_small = pdata['Density'][cut_small].astype(np.float64) * 1e10 * Msun / kpc**3
    rho_prof = np.zeros(N_RHO)
    for b in range(N_RHO):
        mb = bidx_rho == b
        if mb.sum() > 0:
            w = mass_small[mb]; wsum = w.sum()
            rho_prof[b] = np.dot(rho_cgs_small[mb], w) / wsum

    # ── Turbulent velocity profile ──
    vturb_prof = np.zeros(N_BINS)
    for b in range(N_BINS):
        mb = bidx == b
        if mb.sum() > 0:
            w = mass_small[mb]; wsum = w.sum()
            vturb_prof[b] = np.dot(v_rest[mb], w) / wsum

    # ── Resolution profile dx = (m/rho)^{1/3} ──
    _mass_g = mass_small * 1e10 * Msun
    _dx_AU = (_mass_g / rho_cgs_small)**(1.0 / 3.0) / AU
    # Coarse bins (legacy)
    dx_prof = np.zeros(N_BINS)
    for b in range(N_BINS):
        mb = bidx == b
        if mb.sum() > 0:
            w = mass_small[mb]; wsum = w.sum()
            dx_prof[b] = np.dot(_dx_AU[mb], w) / wsum
    # Fine log-spaced bins for resolution profile — use ALL particles
    _cut_res = np.ones(len(gas_dists), dtype=bool)  # no radial cut
    _res_coords = pdata['Coordinates'][_cut_res] - com
    _res_pos = _res_coords @ rot.T
    _res_r_xy = np.sqrt(_res_pos[:, 0]**2 + _res_pos[:, 1]**2)
    _res_r_AU = _res_r_xy * kpc / AU
    _res_mass_g = pdata['Masses'][_cut_res] * 1e10 * Msun
    _res_rho_cgs = pdata['Density'][_cut_res].astype(np.float64) * 1e10 * Msun / kpc**3
    _res_dx_AU = (_res_mass_g / _res_rho_cgs)**(1.0 / 3.0) / AU
    _res_mass = pdata['Masses'][_cut_res]
    _N_DX = 100
    _dx_bin_edges = np.logspace(0, np.log10(1e3 * 206265.0), _N_DX + 1)  # 1 AU to 10^3 pc
    _dx_bin_ctr = np.sqrt(_dx_bin_edges[:-1] * _dx_bin_edges[1:])
    _dx_bidx = np.clip(np.digitize(_res_r_AU, _dx_bin_edges) - 1, 0, _N_DX - 1)
    dx_fine_prof = np.full(_N_DX, np.nan)
    for b in range(_N_DX):
        mb = _dx_bidx == b
        if mb.sum() > 0:
            w = _res_mass[mb]; wsum = w.sum()
            dx_fine_prof[b] = np.dot(_res_dx_AU[mb], w) / wsum
    # Mass profile in Msun (per-particle Δm)
    _res_mass_Msun = _res_mass * 1e10  # Msun
    mass_fine_prof = np.full(_N_DX, np.nan)
    for b in range(_N_DX):
        mb = _dx_bidx == b
        if mb.sum() > 0:
            w = _res_mass[mb]; wsum = w.sum()
            mass_fine_prof[b] = np.dot(_res_mass_Msun[mb], w) / wsum
    # Also save wider particle scatter for plotting
    r_xy_AU = _res_r_AU
    _dx_AU_wide = _res_dx_AU

    # ── Mass-to-flux ratio ──
    has_bfield = 'MagneticField' in pdata
    mf_prof = np.full(N_BINS, np.nan)
    Bz_fo_map = None; Br_fo_map = None; Btor_fo_map = None; Bpol_fo_map = None
    Bz_eo_map = None; Br_eo_map = None; Btor_eo_map = None; Bpol_eo_map = None
    Bmag_fo_map = None; Bmag_eo_map = None
    mu_fo_map = None; mu_eo_map = None

    if has_bfield:
        _B_raw = pdata['MagneticField'][cut_small]
        _B_rot = _B_raw @ rot.T  # disk frame
        _Bz = _B_rot[:, 2]

        # Cylindrical B decomposition (same as velocity)
        _Br = _B_rot[:, 0] * e_r_x + _B_rot[:, 1] * e_r_y
        _Bphi = -_B_rot[:, 0] * e_r_y + _B_rot[:, 1] * e_r_x  # toroidal
        _Bpol = np.sqrt(_Br**2 + _Bz**2)  # poloidal

        _sqrt_G = np.sqrt(G)
        for b in range(N_BINS):
            mb = bidx == b
            if mb.sum() == 0:
                continue
            w = mass_small[mb]; wsum = w.sum()
            _Bz_mw = np.dot(np.abs(_Bz[mb]), w) / wsum
            if _Bz_mw > 0 and Sigma_prof[b] > 0:
                mf_prof[b] = 2.0 * np.pi * _sqrt_G * Sigma_prof[b] / _Bz_mw

        # B-field 2D maps — use thin midplane slices instead of line-of-sight projections.
        # Face-on slice: particles within ±slice_thick of the disk plane (|z_disk| < thick).
        # Edge-on slice: particles within ±slice_thick of the y=0 plane (|y_disk| < thick).
        # slice_thick = 5% of the image box half-width (~one disk scale height).
        _slice_thick = image_box_kpc * 0.05

        def _bfield_slice_map(pos_sl, mass_sl, hsml_sl, B_sl, size):
            """Mass-weighted mean B-field from a thin midplane slab."""
            if len(mass_sl) < 5 or mass_sl.sum() < 1e-40:
                return None
            M_sl = Meshoid(pos_sl, mass_sl, hsml_sl)
            sm = np.maximum(M_sl.SurfaceDensity(M_sl.m, center=center0, size=size, res=res), 1e-40)
            return M_sl.SurfaceDensity(M_sl.m * B_sl, center=center0, size=size, res=res) / sm

        # Face-on maps: slice at z_disk ≈ 0 (pos_fo[:, 2] = disk-normal coord)
        _B_rot_fo = _B_raw @ rot_fo.T
        _Bz_fo = _B_rot_fo[:, 2]
        pos_fo_xy = pos_fo[:, :2]
        r_fo_xy = np.linalg.norm(pos_fo_xy, axis=1)
        safe_r_fo = np.maximum(r_fo_xy, 1e-30)
        e_r_x_fo = pos_fo_xy[:, 0] / safe_r_fo
        e_r_y_fo = pos_fo_xy[:, 1] / safe_r_fo
        _Br_fo = _B_rot_fo[:, 0] * e_r_x_fo + _B_rot_fo[:, 1] * e_r_y_fo
        _Bphi_fo = -_B_rot_fo[:, 0] * e_r_y_fo + _B_rot_fo[:, 1] * e_r_x_fo
        _Bpol_fo = np.sqrt(_Br_fo**2 + _Bz_fo**2) * np.sign(_Br_fo + 1e-30)
        Bmag_small = np.sqrt(np.sum(_B_raw**2, axis=1))
        _Bpol_signed = np.sqrt(_Br**2 + _Bz**2) * np.sign(_Br + 1e-30)

        # Face-on slice mask: |z_disk| < _slice_thick
        _fo_sl = np.abs(pos_fo[:, 2]) < _slice_thick
        _fo_pos  = pos_fo[_fo_sl];     _fo_mass = mass_small[_fo_sl]; _fo_hsml = hsml_small[_fo_sl]
        Bz_fo_map   = _bfield_slice_map(_fo_pos, _fo_mass, _fo_hsml, _Bz_fo[_fo_sl],   image_box_kpc)
        Br_fo_map   = _bfield_slice_map(_fo_pos, _fo_mass, _fo_hsml, _Br_fo[_fo_sl],   image_box_kpc)
        Btor_fo_map = _bfield_slice_map(_fo_pos, _fo_mass, _fo_hsml, _Bphi_fo[_fo_sl], image_box_kpc)
        Bpol_fo_map = _bfield_slice_map(_fo_pos, _fo_mass, _fo_hsml, _Bpol_fo[_fo_sl], image_box_kpc)
        Bmag_fo_map = _bfield_slice_map(_fo_pos, _fo_mass, _fo_hsml, Bmag_small[_fo_sl], image_box_kpc)

        # Edge-on maps (pos_edge = pos_small[:, [0,2,1]], so pos_edge[:,2] = y_disk)
        # Edge-on slice: |y_disk| < _slice_thick (xz-plane cross-section)
        _eo_sl = np.abs(pos_edge[:, 2]) < _slice_thick
        _eo_pos  = pos_edge[_eo_sl];   _eo_mass = mass_small[_eo_sl]; _eo_hsml = hsml_small[_eo_sl]
        Bz_eo_map   = _bfield_slice_map(_eo_pos, _eo_mass, _eo_hsml, _Bz[_eo_sl],          image_box_kpc)
        Br_eo_map   = _bfield_slice_map(_eo_pos, _eo_mass, _eo_hsml, _Br[_eo_sl],          image_box_kpc)
        Btor_eo_map = _bfield_slice_map(_eo_pos, _eo_mass, _eo_hsml, _Bphi[_eo_sl],        image_box_kpc)
        Bpol_eo_map = _bfield_slice_map(_eo_pos, _eo_mass, _eo_hsml, _Bpol_signed[_eo_sl], image_box_kpc)
        Bmag_eo_map = _bfield_slice_map(_eo_pos, _eo_mass, _eo_hsml, Bmag_small[_eo_sl],   image_box_kpc)

        # Mass-to-flux ratio μ = 2π√G Σ / |Bz|  (dimensionless, in critical units)
        _eps_b = 1e-30
        _pc_cm = kpc / 1e3
        if Bz_fo_map is not None:
            mu_fo_map = (2.0 * np.pi * np.sqrt(G) * Sigma_fo_gcm2
                         / np.maximum(np.abs(Bz_fo_map), _eps_b))
            mu_fo_map = np.where(Sigma_fo_gcm2 > 0, mu_fo_map, np.nan)
        if Bz_eo_map is not None:
            Sigma_eo_gcm2 = np.maximum(sig_eo, 0.0) * Msun / _pc_cm**2
            mu_eo_map = (2.0 * np.pi * np.sqrt(G) * Sigma_eo_gcm2
                         / np.maximum(np.abs(Bz_eo_map), _eps_b))
            mu_eo_map = np.where(Sigma_eo_gcm2 > 0, mu_eo_map, np.nan)

    # ── Wide-radius Mach number profile ──
    # Use full-sim particles when available (extends to ~2 kpc); otherwise fall back
    # to cutout particles (max ~0.5 pc).
    N_MACH_BINS = 40
    r_wide_kpc = 2.0  # target 2 kpc if full-sim available
    _fs_wide = None
    _wide_dir = _WIDESIM_PATH if _WIDESIM_PATH is not None else _FULLSIM_PATH
    if _wide_dir is not None:
        _fs_wide = _load_fullsim_wide_gas(snap_num, _wide_dir, hdr, com, com_vel,
                                          r_max_kpc=r_wide_kpc)
    if _fs_wide is not None:
        coords_w_raw = _fs_wide['coords_kpc']   # already com-subtracted
        vel_w_raw    = _fs_wide['vel_kms']       # already com_vel-subtracted
        mass_w       = _fs_wide['mass']
        u_w_fs       = _fs_wide['u']
        n_wide_found = len(mass_w)
    else:
        # Fall back to cutout particles
        r_wide_kpc = max(args.r_max * 5, args.r_search * 2, 5e-4)
        cut_wide = gas_dists < r_wide_kpc
        _com_vel_sub = com_vel if com_vel is not None else np.zeros(3)
        coords_w_raw = pdata['Coordinates'][cut_wide] - com
        vel_w_raw    = pdata['Velocities'][cut_wide] - _com_vel_sub
        mass_w       = pdata['Masses'][cut_wide]
        u_w_fs       = pdata['InternalEnergy'][cut_wide] if 'InternalEnergy' in pdata else None
        n_wide_found = cut_wide.sum()

    if n_wide_found > 10:
        pos_w = coords_w_raw @ rot.T
        vel_w = vel_w_raw @ rot.T
        r_xy_w = np.linalg.norm(pos_w[:, :2], axis=1)

        # Log-spaced bins from 50 AU to max radius
        r_min_mach = 1 * AU / kpc  # 1 AU in kpc
        r_max_mach = r_wide_kpc
        mach_bin_edges = np.logspace(np.log10(r_min_mach), np.log10(r_max_mach), N_MACH_BINS + 1)
        mach_bidx = np.clip(np.digitize(r_xy_w, mach_bin_edges) - 1, 0, N_MACH_BINS - 1)
        mach_bin_ctr_AU = np.sqrt(mach_bin_edges[:-1] * mach_bin_edges[1:]) * kpc / AU

        # Velocity decomposition (wide)
        safe_rxy_w = np.maximum(r_xy_w, 1e-30)
        e_rx_w = pos_w[:, 0] / safe_rxy_w
        e_ry_w = pos_w[:, 1] / safe_rxy_w
        vr_w =  vel_w[:, 0] * e_rx_w + vel_w[:, 1] * e_ry_w
        vphi_w = -vel_w[:, 0] * e_ry_w + vel_w[:, 1] * e_rx_w
        vz_w = vel_w[:, 2]

        # Streaming subtraction
        vr_stream_w = np.zeros(N_MACH_BINS)
        vphi_stream_w = np.zeros(N_MACH_BINS)
        for b in range(N_MACH_BINS):
            mb = mach_bidx == b
            if mb.sum() > 0:
                w = mass_w[mb]; wsum = w.sum()
                vr_stream_w[b] = np.dot(vr_w[mb], w) / wsum
                vphi_stream_w[b] = np.dot(vphi_w[mb], w) / wsum

        dvr_w = vr_w - vr_stream_w[mach_bidx]
        dvphi_w = vphi_w - vphi_stream_w[mach_bidx]
        vrest_w = np.sqrt(dvr_w**2 + dvphi_w**2 + vz_w**2)

        # Sound speed (wide)
        if u_w_fs is not None:
            cs_w = np.sqrt(_GAMMA * (_GAMMA - 1.0) * np.maximum(u_w_fs, 0.0))
        else:
            cs_w = np.zeros(n_wide_found)

        # Bin: mass-weighted Mach + wide kinematic profiles
        mach_wide_prof    = np.full(N_MACH_BINS, np.nan)
        vturb_wide_prof   = np.full(N_MACH_BINS, np.nan)
        cs_wide_prof      = np.full(N_MACH_BINS, np.nan)
        sigma_r_wide_prof = np.full(N_MACH_BINS, np.nan)
        for b in range(N_MACH_BINS):
            mb = mach_bidx == b
            if mb.sum() > 0:
                w = mass_w[mb]; wsum = w.sum()
                vt_b   = np.dot(vrest_w[mb], w) / wsum
                cs_b   = np.dot(cs_w[mb], w) / wsum
                vr2_mw = np.dot(vr_w[mb]**2, w) / wsum
                sig_r_b = np.sqrt(max(vr2_mw - vr_stream_w[b]**2, 0.0))
                vturb_wide_prof[b]   = vt_b
                cs_wide_prof[b]      = cs_b
                sigma_r_wide_prof[b] = sig_r_b
                if cs_b > 0:
                    mach_wide_prof[b] = vt_b / cs_b
        vr_wide_prof   = vr_stream_w
        vphi_wide_prof = vphi_stream_w
    else:
        mach_bin_ctr_AU   = np.array([])
        mach_wide_prof    = np.array([])
        vturb_wide_prof   = np.array([])
        cs_wide_prof      = np.array([])
        sigma_r_wide_prof = np.array([])
        vr_wide_prof      = np.array([])
        vphi_wide_prof    = np.array([])

    # ── Spherical profiles (0 → 500 pc): M_enc, M_shell, ρ ──
    _N_SPH = 80
    _r_sph_max_kpc = 0.5   # 500 pc = 0.5 kpc
    _r_sph_min_kpc = 1.0 * AU / kpc
    _sph_edges = np.logspace(
        np.log10(_r_sph_min_kpc), np.log10(_r_sph_max_kpc), _N_SPH + 1)
    sph_ctr_AU = np.sqrt(_sph_edges[:-1] * _sph_edges[1:]) * kpc / AU

    # Combine cutout + wide gas for spherical profiles
    if _fs_wide is not None:
        _r_sph_all = np.linalg.norm(_fs_wide['coords_kpc'], axis=1)
        _mass_sph_all = _fs_wide['mass'] * 1e10   # Msun
    else:
        _r_sph_all = gas_dists
        _mass_sph_all = pdata['Masses'] * 1e10

    m_enc_prof = np.zeros(_N_SPH)
    rho_sph_prof = np.zeros(_N_SPH)
    if len(_r_sph_all) > 5:
        for _b in range(_N_SPH):
            m_enc_prof[_b] = _mass_sph_all[_r_sph_all < _sph_edges[_b + 1]].sum()
            _in_shell = (_r_sph_all >= _sph_edges[_b]) & (_r_sph_all < _sph_edges[_b + 1])
            _vol_cm3 = (4.0 / 3.0) * np.pi * ((_sph_edges[_b + 1] * kpc)**3 - (_sph_edges[_b] * kpc)**3)
            _shell_mass_g = _mass_sph_all[_in_shell].sum() * Msun
            rho_sph_prof[_b] = _shell_mass_g / _vol_cm3 if _vol_cm3 > 0 else 0.0
    m_shell_prof = np.diff(m_enc_prof, prepend=0.0)

    # ── Virial parameter profile: α(r) = 5 σ²(<r) r / (3 G M(<r)) ──
    # σ² = mass-weighted 3D velocity dispersion within sphere of radius r
    virial_sph_prof = np.full(_N_SPH, np.nan)
    if _fs_wide is not None:
        _vel_sph = _fs_wide['vel_kms']   # already com_vel-subtracted, km/s
        _r_sph_3d = np.linalg.norm(_fs_wide['coords_kpc'], axis=1)
        _mass_sph_w = _fs_wide['mass'] * 1e10  # Msun
    else:
        _com_vel_sub = com_vel if com_vel is not None else np.zeros(3)
        _vel_sph = pdata['Velocities'] - _com_vel_sub
        _r_sph_3d = gas_dists
        _mass_sph_w = pdata['Masses'] * 1e10
    if len(_r_sph_3d) > 5:
        for _b in range(_N_SPH):
            _in_r = _r_sph_3d < _sph_edges[_b + 1]
            _n_in = _in_r.sum()
            if _n_in < 5 or m_enc_prof[_b] <= 0:
                continue
            _w = _mass_sph_w[_in_r]
            _v = _vel_sph[_in_r]  # (N, 3) km/s
            _wsum = _w.sum()
            _v_mean = np.sum(_v * _w[:, None], axis=0) / _wsum
            _dv = _v - _v_mean
            _sigma2 = np.sum(_w * np.sum(_dv**2, axis=1)) / _wsum  # km²/s²
            _r_cm = _sph_edges[_b + 1] * kpc
            _M_g = m_enc_prof[_b] * Msun
            virial_sph_prof[_b] = (5.0 * _sigma2 * 1e10 * _r_cm) / (3.0 * G * _M_g)

    # ── Keplerian velocity profile v_K = sqrt(G M_disk_enc(<r_cyl) / r) ──
    # Uses cumulative disk mass within cylindrical radius r_cyl (disk particles only),
    # matching the cylindrical bin grid used for the kinematic profiles.
    _G_cgs = G   # cm^3 g^-1 s^-2 (from generic_utils.constants)
    _km_s  = 1e5  # cm/s per km/s

    # Disk particles in the cylindrical frame
    _disk_mask = is_disk[cut_small]
    if _disk_mask.any():
        _disk_rcyl_kpc = r_xy[_disk_mask]            # cylindrical r [kpc]
        _disk_mass_Msun = mass_small[_disk_mask] * 1e10  # Msun
        # Sort by cylindrical radius for cumulative sum
        _sort = np.argsort(_disk_rcyl_kpc)
        _r_sorted_kpc = _disk_rcyl_kpc[_sort]
        _M_sorted     = _disk_mass_Msun[_sort]
        _M_cum        = np.cumsum(_M_sorted)          # cumulative disk mass [Msun]
    else:
        _r_sorted_kpc = np.array([])
        _M_cum        = np.array([])

    def _vK_from_disk(r_kpc):
        """v_K [km/s] from cumulative disk mass within cylindrical radius r_kpc."""
        r_cm = r_kpc * kpc
        if r_cm <= 0 or len(_M_cum) == 0:
            return 0.0
        idx = np.searchsorted(_r_sorted_kpc, r_kpc)
        M_enc = _M_cum[idx - 1] if idx > 0 else 0.0
        if M_enc <= 0:
            return 0.0
        return np.sqrt(_G_cgs * M_enc * Msun / r_cm) / _km_s

    vK_prof = np.array([_vK_from_disk(r) for r in bin_centers_kpc])

    # Wide-grid v_K: use all particles within wide region (not disk-only,
    # as the wide profiles cover environment gas well beyond the disk)
    vK_wide_prof = np.array([])
    if len(mach_bin_ctr_AU) > 0 and n_wide_found > 0:
        _wide_rcyl_kpc = r_xy_w                            # cylindrical r [kpc]
        _wide_mass_Msun = mass_w * 1e10                    # Msun
        _sort_w = np.argsort(_wide_rcyl_kpc)
        _r_sort_w = _wide_rcyl_kpc[_sort_w]
        _M_cum_w  = np.cumsum(_wide_mass_Msun[_sort_w])

        def _vK_wide(r_kpc):
            r_cm = r_kpc * kpc
            if r_cm <= 0:
                return 0.0
            idx = np.searchsorted(_r_sort_w, r_kpc)
            M_enc = _M_cum_w[idx - 1] if idx > 0 else 0.0
            if M_enc <= 0:
                return 0.0
            return np.sqrt(_G_cgs * M_enc * Msun / r_cm) / _km_s

        vK_wide_prof = np.array([_vK_wide(_r * AU / kpc) for _r in mach_bin_ctr_AU])

    # ── Phase diagram data ──
    rho_local = rho_cgs_small.copy()
    n_local = rho_local / _m_p  # number density [cm^-3]

    if 'InternalEnergy' in pdata:
        T_local = pdata['InternalEnergy'][cut_small] * 1e10 / (1.5 * 1.381e-16 / _m_p)
        # T = (gamma-1) * u * mu * m_p / k_B; u in (km/s)^2 = 1e10 cm^2/s^2
        # More careful: T = mu * m_p * (gamma-1) * u / k_B
        k_B = 1.381e-16  # erg/K
        mu_mol = 1.22  # mean molecular weight for primordial neutral gas
        u_cgs = pdata['InternalEnergy'][cut_small] * 1e10  # cm^2/s^2
        T_local = mu_mol * _m_p * (_GAMMA - 1.0) * u_cgs / k_B
    else:
        T_local = np.zeros(len(mass_small))

    # H2 fraction
    fh2_local = None
    for candidate in ['MolecularMassFraction', 'Molecular_Fraction',
                      'MolecularHydrogenFraction', 'H2Fraction']:
        if candidate in pdata:
            fh2_local = pdata[candidate][cut_small]
            break

    # |B| magnitude for B vs rho phase diagram
    B_local = None
    if 'MagneticField' in pdata:
        B_small_ph = pdata['MagneticField'][cut_small]
        B_local = np.sqrt(np.sum(B_small_ph**2, axis=1))  # |B| [Gauss]

    mass_local = mass_small * 1e10  # Msun
    is_disk_local = is_disk[cut_small]

    # Disk mass and mass infall rate for infall timescale
    M_disk_Msun = float(np.sum(mass_small[is_disk[cut_small]] * 1e10))  # Msun
    # Mdot ≈ 4π r_max² ρ v_r (inward), estimated at outer disk edge
    # Use mass flux through outermost radial bin with disk particles
    _r_disk = r_xy[is_disk[cut_small]] if is_disk[cut_small].any() else np.array([])
    _vr_disk = v_r[is_disk[cut_small]] if is_disk[cut_small].any() else np.array([])
    Mdot_Msun_yr = np.nan
    if len(_r_disk) > 5:
        r_edge = np.percentile(_r_disk, 90)
        shell = (_r_disk > r_edge * 0.8) & (_r_disk < r_edge * 1.2)
        if shell.sum() > 2:
            _m_sh = mass_small[is_disk[cut_small]][shell] * 1e10  # Msun
            _vr_sh = _vr_disk[shell]  # km/s (negative = infall)
            # mass-weighted radial velocity
            _vr_mw = np.dot(_vr_sh, _m_sh) / _m_sh.sum()
            # Shell area: 4π r²
            _A_cm2 = 4.0 * np.pi * (r_edge * kpc)**2
            # Surface density in shell
            _r_lo = r_edge * 0.8 * kpc; _r_hi = r_edge * 1.2 * kpc
            _A_shell = 4.0 * np.pi * (_r_hi**2 - _r_lo**2)
            _Sigma_sh = _m_sh.sum() * Msun / _A_shell  # g/cm²
            # Mdot = Sigma * v_r * 4π r (negative vr = infall = positive Mdot)
            _Mdot_cgs = -_Sigma_sh * (_vr_mw * 1e5) * _A_cm2 / _A_shell * _A_shell
            _Mdot_cgs2 = -_m_sh.sum() * Msun * (_vr_mw * 1e5) / (r_edge * kpc)
            # Simpler: Mdot = -sum(m_i * v_r_i / r_i) for infall
            _Mdot_cgs3 = -np.sum(_m_sh * Msun * _vr_sh * 1e5 / (_r_disk[shell] * kpc))
            Mdot_Msun_yr = _Mdot_cgs3 / Msun * (3.156e7)  # cgs → Msun/yr

    # Sink positions
    n_stars = 0
    star_fo_AU = np.empty((0, 2))
    star_eo_AU = np.empty((0, 2))
    if stardata and len(stardata.get('Masses', [])) > 0:
        n_stars = len(stardata['Masses'])
        scoords = stardata['Coordinates'] - com
        s_fo = scoords @ rot_fo.T
        s_eo = (scoords @ rot.T)[:, [0, 2, 1]]
        star_fo_AU = s_fo[:, :2] * kpc / AU
        star_eo_AU = s_eo[:, :2] * kpc / AU

    # Time label
    if sink_form_time_Myr is not None:
        dt_kyr = (time_Myr - sink_form_time_Myr) * 1e3
        time_label = f"$t - t_1$ = {dt_kyr:.1f} kyr"
    else:
        time_label = f"t = {time_Myr:.3f} Myr"

    bin_AU = bin_centers_kpc * kpc / AU

    return {
        'snap_num': snap_num,
        'time_Myr': time_Myr,
        'time_label': time_label,
        't1_Myr': sink_form_time_Myr,
        # 2D maps
        'sig_fo': sig_fo, 'sig_eo': sig_eo,
        'vdisp_fo': vdisp_fo, 'vdisp_eo': vdisp_eo,
        'Q_fo': Q_fo,
        'Bz_fo': Bz_fo_map, 'Br_fo': Br_fo_map, 'Btor_fo': Btor_fo_map, 'Bpol_fo': Bpol_fo_map,
        'Bz_eo': Bz_eo_map, 'Br_eo': Br_eo_map, 'Btor_eo': Btor_eo_map, 'Bpol_eo': Bpol_eo_map,
        'Bmag_fo': Bmag_fo_map, 'Bmag_eo': Bmag_eo_map,
        'mu_fo': mu_fo_map, 'mu_eo': mu_eo_map,
        'X': X, 'Y': Y, 'half_AU': half_AU,
        # 1D profiles
        'bin_AU': bin_AU,
        'bin_ctr_rho_AU': bin_ctr_rho_AU,
        'vr_prof': vr_prof, 'neg_vr_prof': -vr_prof,
        'vphi_prof': vphi_prof, 'vturb_prof': vturb_prof,
        'cs_prof': cs_prof,
        'rho_prof': rho_prof, 'dx_prof': dx_prof,
        'dx_fine_prof': dx_fine_prof, 'dx_fine_bin_AU': _dx_bin_ctr,
        'mass_fine_prof': mass_fine_prof,
        'dx_AU_particles': _dx_AU_wide, 'r_AU_particles': r_xy_AU,
        'mass_Msun_particles': _res_mass * 1e10,
        'Q_prof': Q_prof, 'Q_prof_combined': Q_prof_combined, 'Q_prof_therm': Q_prof_therm,
        'Q_fo_combined': Q_fo_combined,
        'mf_prof': mf_prof,
        'sigma_r_prof': sigma_r_prof,
        'Sigma_prof': Sigma_prof,
        'Omega_prof': Omega_prof,
        # Phase data
        'rho_local': rho_local, 'n_local': n_local,
        'T_local': T_local, 'fh2_local': fh2_local, 'B_local': B_local,
        'mass_local': mass_local, 'is_disk_local': is_disk_local,
        # Mach profile + wide kinematic profiles
        'mach_wide_prof': mach_wide_prof,
        'mach_wide_bin_AU': mach_bin_ctr_AU,
        'vr_wide_prof': vr_wide_prof, 'neg_vr_wide_prof': -vr_wide_prof,
        'vphi_wide_prof': vphi_wide_prof,
        'cs_wide_prof': cs_wide_prof, 'vturb_wide_prof': vturb_wide_prof,
        'sigma_r_wide_prof': sigma_r_wide_prof,
        # Sinks
        'n_stars': n_stars,
        'star_fo_AU': star_fo_AU, 'star_eo_AU': star_eo_AU,
        # Disk mass and infall rate
        'M_disk_Msun': M_disk_Msun,
        'Mdot_Msun_yr': Mdot_Msun_yr,
        # Spherical shell profiles
        'sph_ctr_AU': sph_ctr_AU,
        'm_enc_prof': m_enc_prof,
        'm_shell_prof': m_shell_prof,
        'rho_sph_prof': rho_sph_prof,
        'virial_sph_prof': virial_sph_prof,
        # Keplerian velocity profiles
        'vK_prof': vK_prof,
        'vK_wide_prof': vK_wide_prof,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2D Grid Figures
# ═════════════════════════════════════════════════════════════════════════════

def _add_scale_bar(ax, length_AU, label=None, color='white'):
    """Add a 1 kAU scale bar to bottom-left of a projection panel."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]

    if label is None:
        if length_AU >= 1000:
            label = f'{length_AU:.0f} AU'
        else:
            label = f'{length_AU:.0f} AU'

    x1 = xlim[0] + 0.05 * xspan
    x2 = x1 + length_AU
    y0 = ylim[0] + 0.07 * yspan

    ax.plot([x1, x2], [y0, y0], color=color, lw=2.5, solid_capstyle='butt')
    ax.text((x1 + x2) / 2, y0 + 0.02 * yspan, label,
            color=color, ha='center', va='bottom', fontsize=18,
            fontweight='bold')


def _prune_shared_ticks(axes_2d):
    """Hide edge tick labels that clash at shared panel boundaries.

    For a 2-D array of axes with wspace=0/hspace=0, wraps each axis's
    formatter so that the outermost tick label on interior edges is blanked.
    Uses FuncFormatter which survives bbox_inches='tight' re-renders.
    """
    from matplotlib.ticker import FuncFormatter
    nrows, ncols = axes_2d.shape

    def _wrap_formatter(ax, axis, hide_low, hide_high):
        axis_obj = ax.xaxis if axis == 'x' else ax.yaxis
        orig_fmt = axis_obj.get_major_formatter()

        def _fmt(val, pos):
            locs = axis_obj.get_major_locator()()
            vmin, vmax = ax.get_xlim() if axis == 'x' else ax.get_ylim()
            visible = sorted(v for v in locs if vmin <= v <= vmax)
            if visible:
                if hide_low and abs(val - visible[0]) < 1e-6 * max(abs(visible[0]), 1):
                    return ''
                if hide_high and abs(val - visible[-1]) < 1e-6 * max(abs(visible[-1]), 1):
                    return ''
            # ScalarFormatter needs set_locs to produce correct labels
            orig_fmt.set_locs(locs)
            return orig_fmt(val, pos)

        axis_obj.set_major_formatter(FuncFormatter(_fmt))

    for r in range(nrows):
        for c in range(ncols):
            ax = axes_2d[r, c]
            # x-axis: hide leftmost tick only (keep tick on earlier column)
            hide_x_low  = c > 0
            if hide_x_low:
                _wrap_formatter(ax, 'x', hide_low=True, hide_high=False)
            # y-axis: hide topmost tick only (keep tick on earlier row)
            if r > 0:
                _wrap_formatter(ax, 'y', hide_low=False, hide_high=True)


@_use_orig_rc
def plot_grid_faceon(epoch_data_list, field_key, outdir, cmap, norm,
                     label, filename, contour_level=None):
    """2x3 grid of face-on projections, one per epoch."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10),
                             gridspec_kw={'wspace': 0, 'hspace': 0,
                                          'left': 0.02, 'right': 0.94, 'top': 0.98, 'bottom': 0.02})
    axes_flat = axes.flatten()

    for i, ed in enumerate(epoch_data_list):
        ax = axes_flat[i]
        data = ed.get(field_key)
        if data is None:
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', va='center')
            ax.set_xlim(-ed['half_AU'], ed['half_AU'])
            ax.set_ylim(-ed['half_AU'], ed['half_AU'])
        else:
            im = ax.pcolormesh(ed['X'], ed['Y'], data, cmap=cmap, norm=norm,
                               rasterized=True)
            if contour_level is not None and np.any(np.isfinite(data)):
                try:
                    ax.contour(ed['X'], ed['Y'], data, levels=[contour_level],
                               colors='k', linewidths=0.8)
                except Exception:
                    pass
        # Sink markers (only those inside image box)
        if ed['n_stars'] > 0:
            sx, sy = ed['star_fo_AU'][:, 0], ed['star_fo_AU'][:, 1]
            in_box = (np.abs(sx) < ed['half_AU']) & (np.abs(sy) < ed['half_AU'])
            if in_box.any():
                ax.scatter(sx[in_box], sy[in_box],
                           marker='*', s=30, c='white', edgecolors='k', linewidths=0.5, zorder=5)
        ax.set_xlim(-ed['half_AU'], ed['half_AU'])
        ax.set_ylim(-ed['half_AU'], ed['half_AU'])
        ax.set_aspect('equal')
        ax.text(0.03, 0.95, ed['time_label'], transform=ax.transAxes,
                va='top', color='white',
                bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
        _add_scale_bar(ax, 1000, '1000 AU')
        ax.set_xticks([])
        ax.set_yticks([])

    cbar_ax = fig.add_axes([0.945, 0.04, 0.015, 0.92])
    fig.colorbar(im, cax=cbar_ax, label=label)

    _save_fig_dual(fig, os.path.join(outdir, 'light', filename))


@_use_orig_rc
def plot_grid_edgeon(epoch_data_list, field_key, outdir, cmap, norm,
                     label, filename, x_lim=2500, y_lim=625):
    """6x1 vertical stack of edge-on projections."""
    # Compute figure height from aspect ratio so panels touch with no vertical gap
    panel_w = 10.0  # inches for the plot area
    panel_h = panel_w * (y_lim / x_lim)  # height per panel preserving aspect
    fig_h = panel_h * 6
    fig, axes = plt.subplots(6, 1, figsize=(panel_w + 1, fig_h),
                             gridspec_kw={'hspace': 0,
                                          'left': 0.02, 'right': 0.93, 'top': 0.99, 'bottom': 0.02})

    for i, ed in enumerate(epoch_data_list):
        ax = axes[i]
        data = ed.get(field_key)
        if data is None:
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', va='center')
        else:
            im = ax.pcolormesh(ed['X'], ed['Y'], data, cmap=cmap, norm=norm,
                               rasterized=True)
        # Sink markers (only those inside plot region)
        if ed['n_stars'] > 0:
            sx, sy = ed['star_eo_AU'][:, 0], ed['star_eo_AU'][:, 1]
            in_box = (np.abs(sx) < x_lim) & (np.abs(sy) < y_lim)
            if in_box.any():
                ax.scatter(sx[in_box], sy[in_box],
                           marker='*', s=30, c='white', edgecolors='k', linewidths=0.5, zorder=5)
        _eo_xlim = min(x_lim, ed['half_AU'])
        _eo_ylim = min(y_lim, ed['half_AU'])
        ax.set_xlim(-_eo_xlim, _eo_xlim)
        ax.set_ylim(-_eo_ylim, _eo_ylim)
        ax.set_aspect('equal')
        ax.text(0.01, 0.85, ed['time_label'], transform=ax.transAxes,
                va='top', color='white',
                bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
        ax.set_xticks([])
        ax.set_yticks([])

    cbar_ax = fig.add_axes([0.935, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=label)

    _save_fig_dual(fig, os.path.join(outdir, 'light', filename))

@_use_orig_rc
def plot_grid_combined(epoch_data_list, field_fo, field_eo, outdir, cmap, norm,
                       label, filename, x_lim=2500, y_lim=625, contour_level=None):
    """Combined grid: rows alternate face-on / edge-on for snaps 1-3, then 4-6.

    Layout (5 rows × 3 cols, row 2 is a small visible gap):
      Row 0: face-on  snaps 0,1,2
      Row 1: edge-on  snaps 0,1,2
      Row 2: (gap)
      Row 3: face-on  snaps 3,4,5
      Row 4: edge-on  snaps 3,4,5
    """
    from matplotlib.gridspec import GridSpec

    eo_aspect = y_lim / x_lim  # e.g. 500/2000 = 0.25
    gap_h = 0.018              # gap row height relative to face-on row (25% of original)
    col_w = 5.0  # inches per column
    fig_w = col_w * 3 + 0.5  # +0.5 for colorbar
    fig_h = 2 * (col_w + col_w * eo_aspect) + col_w * gap_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(5, 3,
                  height_ratios=[1, eo_aspect, gap_h, 1, eo_aspect],
                  hspace=0, wspace=0,
                  left=0.02, right=0.95, top=0.98, bottom=0.04)

    im_ref = None  # for colorbar

    # Track axes in creation order: all_axes[i] = (ax_fo, ax_eo) for epoch i
    all_ax_fo = {}
    all_ax_eo = {}

    for i, ed in enumerate(epoch_data_list):
        half = ed['half_AU']
        X, Y = ed['X'], ed['Y']
        fo_data = ed.get(field_fo)
        eo_data = ed.get(field_eo)
        fo_stars = ed['star_fo_AU'] if ed['n_stars'] > 0 else None
        eo_stars = ed['star_eo_AU'] if ed['n_stars'] > 0 else None

        group = i // 3  # 0 = snaps 0-2, 1 = snaps 3-5
        col   = i % 3
        # Skip row 2 (gap); group 0 → rows 0,1; group 1 → rows 3,4
        row_fo = 0 if group == 0 else 3
        row_eo = 1 if group == 0 else 4

        def _plot_panel(ax, data, X_p, Y_p, xlim_lo, xlim_hi, ylim_lo, ylim_hi, stars_xy):
            nonlocal im_ref
            if data is None:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', va='center')
            else:
                im = ax.pcolormesh(X_p, Y_p, data, cmap=cmap, norm=norm, rasterized=True)
                im_ref = im
                if contour_level is not None and np.any(np.isfinite(data)):
                    try:
                        ax.contour(X_p, Y_p, data, levels=[contour_level],
                                   colors='k', linewidths=0.8)
                    except Exception:
                        pass
            ax.set_xlim(xlim_lo, xlim_hi)
            ax.set_ylim(ylim_lo, ylim_hi)
            ax.set_aspect('auto')
            ax.tick_params(direction='in', which='both', top=True, right=True)
            if stars_xy is not None and len(stars_xy) > 0:
                sx, sy = stars_xy[:, 0], stars_xy[:, 1]
                in_box = (sx > xlim_lo) & (sx < xlim_hi) & (sy > ylim_lo) & (sy < ylim_hi)
                if in_box.any():
                    ax.scatter(sx[in_box], sy[in_box],
                               marker='*', s=30, c='white', edgecolors='k', linewidths=0.5, zorder=5)

        # Face-on panel
        ax_fo = fig.add_subplot(gs[row_fo, col])
        _plot_panel(ax_fo, fo_data, X, Y, -half, half, -half, half, fo_stars)
        ax_fo.text(0.03, 0.95, ed['time_label'], transform=ax_fo.transAxes,
                   va='top', color='white',
                   bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
        ax_fo.text(0.97, 0.05, 'face-on', transform=ax_fo.transAxes,
                   ha='right', va='bottom', color='white',
                   bbox=dict(facecolor='black', alpha=0.4, boxstyle='round,pad=0.15'))
        _add_scale_bar(ax_fo, 1000, '1000 AU')
        ax_fo.set_xticks([])
        ax_fo.set_yticks([])
        all_ax_fo[i] = ax_fo

        # Edge-on panel — clamp limits to data extent
        _eo_xlim = min(x_lim, half)
        _eo_ylim = min(y_lim, half)
        ax_eo = fig.add_subplot(gs[row_eo, col])
        _plot_panel(ax_eo, eo_data, X, Y, -_eo_xlim, _eo_xlim, -_eo_ylim, _eo_ylim, eo_stars)
        ax_eo.text(0.97, 0.05, 'edge-on', transform=ax_eo.transAxes,
                   ha='right', va='bottom', color='white',
                   bbox=dict(facecolor='black', alpha=0.4, boxstyle='round,pad=0.15'))
        ax_eo.set_xticks([])
        ax_eo.set_yticks([])
        all_ax_eo[i] = ax_eo

    # Colorbar — extend nearly full height, minimal whitespace
    if im_ref is not None:
        cbar_ax = fig.add_axes([0.955, 0.04, 0.015, 0.94])
        fig.colorbar(im_ref, cax=cbar_ax, label=label)

    _save_fig_dual(fig, os.path.join(outdir, 'light', filename))


# ═════════════════════════════════════════════════════════════════════════════
# 1D Profile Overlays
# ═════════════════════════════════════════════════════════════════════════════

def plot_velocity_profiles(epoch_data_list, outdir):
    """3x1 shared-x plot: v_r, v_phi, delta_v with 6 epoch curves each."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             gridspec_kw={'hspace': 0})
    fields = [('vr_prof', r'$v_r$ [km/s]'),
              ('vphi_prof', r'$v_\phi$ [km/s]'),
              ('vturb_prof', r'$|\delta v|$ [km/s]')]

    for ax_idx, (field, ylabel) in enumerate(fields):
        ax = axes[ax_idx]
        for i, ed in enumerate(epoch_data_list):
            valid = ed[field] != 0
            ax.plot(ed['bin_AU'][valid], ed[field][valid],
                    color=EPOCH_COLORS[i], lw=3.0,
                    label=ed['time_label'] if ax_idx == 0 else None)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color='gray', ls=':', lw=1.0)
        ax.tick_params(direction='in', which='both', top=True, right=True)
        if ax_idx == 0:
            ax.legend(loc='upper right')

    axes[-1].set_xlabel('r [AU]')

    # Prune clashing edge tick labels
    _prune_shared_ticks(axes.reshape(-1, 1))

    _save_fig_dual(fig, os.path.join(outdir, 'light', 'velocity_profiles.png'))


def plot_profile_overlay(epoch_data_list, field_key, bin_key, outdir,
                         ylabel, filename, log_x=False, log_y=False,
                         ref_line=None, ref_label=None,
                         power_law_fit=False):
    """Single panel with 6 epoch curves overlaid."""
    fig, ax = plt.subplots(figsize=(12, 9))

    for i, ed in enumerate(epoch_data_list):
        x = ed[bin_key]
        y = ed[field_key]
        valid = np.isfinite(y) & (y != 0)
        if log_x:
            valid &= (x > 0)
        if log_y:
            valid &= (y > 0)
        if valid.sum() == 0:
            continue

        # Compute power-law fit first so slope can go in the label
        _slope = None
        _intercept = None
        if power_law_fit and log_x and log_y and valid.sum() >= 4:
            fit_mask = valid.copy()
            first_valid = np.where(valid)[0]
            if len(first_valid) > 1:
                fit_mask[first_valid[0]] = False
            if fit_mask.sum() >= 3:
                _lr = np.log10(x[fit_mask])
                _lrho = np.log10(y[fit_mask])
                _slope, _intercept = np.polyfit(_lr, _lrho, 1)

        # Build label: include fit slope if available
        lbl = ed['time_label']
        if _slope is not None:
            lbl += rf'  ($\rho \propto r^{{{_slope:.1f}}}$)'

        if log_x and log_y:
            ax.loglog(x[valid], y[valid], color=EPOCH_COLORS[i], lw=3.0, label=lbl)
        elif log_y:
            ax.semilogy(x[valid], y[valid], color=EPOCH_COLORS[i], lw=3.0, label=lbl)
        elif log_x:
            ax.semilogx(x[valid], y[valid], color=EPOCH_COLORS[i], lw=3.0, label=lbl)
        else:
            ax.plot(x[valid], y[valid], color=EPOCH_COLORS[i], lw=3.0, label=lbl)

        # Draw fit line (no legend entry)
        if _slope is not None:
            _fit_r = x[fit_mask]
            ax.loglog(_fit_r, 10**_intercept * _fit_r**_slope,
                      color=EPOCH_COLORS[i], ls='--', lw=1.6, alpha=0.6)

    if ref_line is not None:
        ax.axhline(ref_line, color='gray', ls='--', lw=2.0, alpha=0.7)
        if ref_label:
            ax.text(0.98, ref_line, ref_label, transform=ax.get_yaxis_transform(),
                    ha='right', va='bottom', color='gray')

    ax.set_xlabel('r [AU]')
    ax.set_ylabel(ylabel)
    ax.tick_params(direction='in', which='both', top=True, right=True)
    ax.legend(loc='best', fontsize=int(plt.rcParams['legend.fontsize'] * 0.75))

    _save_fig_dual(fig, os.path.join(outdir, 'light', filename))


# ═════════════════════════════════════════════════════════════════════════════
# Resolution Profile (scatter + median band)
# ═════════════════════════════════════════════════════════════════════════════

def plot_resolution_profile(epoch_data_list, outdir):
    """2-panel: Δm vs r (top) and (m/ρ)^{1/3} vs r (bottom).

    Both panels show:
      - scatter plot of individual particles for epoch index 1 (dt=0)
      - median line with ±1σ shaded band from epochs 1-5
    Single shared legend between the two panels.
    """
    from matplotlib.lines import Line2D

    fig, (ax_m, ax_dx) = plt.subplots(2, 1, figsize=(12, 18), sharex=True,
                                       gridspec_kw={'hspace': 0})
    fig.patch.set_facecolor('w')

    bin_key = 'dx_fine_bin_AU'
    _AU_per_pc = 206265.0

    def _collect_profiles(field_key):
        profiles = []
        ref_bins = None
        for i in range(1, min(6, len(epoch_data_list))):
            ed = epoch_data_list[i]
            x = ed[bin_key]
            y = ed.get(field_key)
            if y is None:
                continue
            if ref_bins is None:
                ref_bins = x
            y_interp = np.interp(ref_bins, x, y, left=np.nan, right=np.nan)
            profiles.append(y_interp)
        return ref_bins, profiles

    def _plot_median_band(ax, ref_bins, profiles, color='steelblue'):
        if ref_bins is None or len(profiles) == 0:
            return
        prof_arr = np.array(profiles)
        log_prof = np.where(prof_arr > 0, np.log10(prof_arr), np.nan)
        median_log = np.nanmedian(log_prof, axis=0)
        std_log = np.nanstd(log_prof, axis=0)
        valid = np.isfinite(median_log) & (ref_bins > 0)
        median_prof = 10**median_log
        lo = 10**(median_log - std_log)
        hi = 10**(median_log + std_log)
        ax.fill_between(ref_bins[valid], lo[valid], hi[valid],
                        color=color, alpha=0.25)
        ax.loglog(ref_bins[valid], median_prof[valid], color=color, lw=4.0)

    # Top panel: Δm vs r
    ref_bins_m, profs_m = _collect_profiles('mass_fine_prof')
    _plot_median_band(ax_m, ref_bins_m, profs_m)

    if len(epoch_data_list) > 1:
        ed0 = epoch_data_list[1]
        r_pts = ed0.get('r_AU_particles')
        m_pts = ed0.get('mass_Msun_particles')
        if r_pts is not None and m_pts is not None:
            valid_pts = (r_pts > 0) & (m_pts > 0)
            ax_m.scatter(r_pts[valid_pts], m_pts[valid_pts],
                         s=0.3, alpha=0.15, c='k', rasterized=True)

    ax_m.set_ylabel(r'$\Delta m$ [$M_\odot$]')
    ax_m.set_xscale('log')
    ax_m.set_yscale('log')
    ax_m.set_xlim(1, 1e3 * _AU_per_pc)
    ax_m.tick_params(direction='in', which='both', top=True, right=True,
                     labelbottom=False)

    ax_pc = ax_m.secondary_xaxis('top', functions=(lambda x: x / _AU_per_pc,
                                                    lambda x: x * _AU_per_pc))
    ax_pc.set_xlabel('r [pc]')
    ax_pc.tick_params(direction='in', which='both')

    # Bottom panel: (m/ρ)^{1/3} vs r
    ref_bins_dx, profs_dx = _collect_profiles('dx_fine_prof')
    _plot_median_band(ax_dx, ref_bins_dx, profs_dx)

    if len(epoch_data_list) > 1:
        ed0 = epoch_data_list[1]
        r_pts = ed0.get('r_AU_particles')
        dx_pts = ed0.get('dx_AU_particles')
        if r_pts is not None and dx_pts is not None:
            valid_pts = (r_pts > 0) & (dx_pts > 0)
            ax_dx.scatter(r_pts[valid_pts], dx_pts[valid_pts],
                          s=0.3, alpha=0.15, c='k', rasterized=True)

    ax_dx.set_xlabel('r [AU]')
    ax_dx.set_ylabel(r'$(\Delta m/\rho)^{1/3}$ [AU]')
    ax_dx.set_xscale('log')
    ax_dx.set_yscale('log')
    ax_dx.tick_params(direction='in', which='both', top=True, right=True)

    # Shared legend between panels
    _dt0_label = epoch_data_list[1]['time_label'] + ' (particles)' if len(epoch_data_list) > 1 else ''
    legend_handles = [
        Line2D([0], [0], color='steelblue', lw=4.0, label=r'median $(t - t_1 > 0)$'),
        Line2D([], [], color='steelblue', lw=8, alpha=0.25, label=r'$\pm 1\sigma\; (t - t_1 > 0)$'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k',
               markersize=10, label=_dt0_label),
    ]
    ax_m.legend(handles=legend_handles, loc='upper left',
                fontsize=int(plt.rcParams['legend.fontsize'] * 0.75))

    _save_fig_dual(fig, os.path.join(outdir, 'light', 'profile_resolution.png'))


# ═════════════════════════════════════════════════════════════════════════════
# Mach Number Profile
# ═════════════════════════════════════════════════════════════════════════════

def plot_mach_profile(epoch_data_list, outdir):
    """Mach number vs radius (log x, 50 AU to ~1 pc) for 6 epochs."""
    fig, ax = plt.subplots(figsize=(12, 9))

    for i, ed in enumerate(epoch_data_list):
        x = ed['mach_wide_bin_AU']
        y = ed['mach_wide_prof']
        if len(x) == 0:
            continue
        valid = np.isfinite(y)
        if valid.sum() == 0:
            continue
        ax.semilogx(x[valid], y[valid], color=EPOCH_COLORS[i], lw=3.0,
                    label=ed['time_label'])

    ax.axhline(1.0, color='gray', ls='--', lw=2.0, alpha=0.7)
    ax.text(0.98, 1.0, 'Ma = 1', transform=ax.get_yaxis_transform(),
            ha='right', va='bottom', color='gray')
    ax.set_xlim(50, 250000)
    ax.set_xlabel('r [AU]')
    ax.set_ylabel(r'$\mathcal{M} = |\delta v| / c_s$')
    ax.tick_params(direction='in', which='both', top=True, right=True)
    ax.legend(loc='best')

    _save_fig_dual(fig, os.path.join(outdir, 'light', 'profile_mach.png'))


# ═════════════════════════════════════════════════════════════════════════════
# Phase Diagrams (individual per-snapshot, matching diagnostic style)
# ═════════════════════════════════════════════════════════════════════════════

def _load_phase_stats(frames_dir):
    """Load phase_stats.npz if available. Returns dict or None."""
    if frames_dir is None:
        return None
    p = os.path.join(frames_dir, 'phase_stats.npz')
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return {k: d[k] for k in d.files}


def _phase_cumulative_overlay(ax, ps, snap_lo, snap_hi, n_ctr, row,
                               color, label=None):
    """
    Overlay mean ± std of the per-bin median T (or f_H2) for all snapshots
    with snap number in [snap_lo, snap_hi].

    ps     : phase_stats dict
    snap_lo, snap_hi : inclusive snap number range
    row    : 'T' or 'fH2'
    color  : line color
    """
    snaps = ps['snap_nums']
    mask = (snaps >= snap_lo) & (snaps <= snap_hi)
    if mask.sum() == 0:
        return

    key_med = 'T_med' if row == 'T' else 'fH2_med'
    mat = ps[key_med][mask]           # (n_snaps_in_range, N_BINS)

    # For each bin, mean and std of the per-snapshot medians
    with np.errstate(all='ignore'):
        mn  = np.nanmean(mat, axis=0)
        std = np.nanstd(mat,  axis=0)

    valid = np.isfinite(mn)
    if not valid.any():
        return

    xv, yv, ylo, yhi = (n_ctr[valid], mn[valid],
                         mn[valid] - std[valid], mn[valid] + std[valid])
    ax.plot(xv, yv, color=color, lw=3.6, ls='-', zorder=5,
            label=label, alpha=0.9)
    ax.fill_between(xv, ylo, yhi, color=color, alpha=0.15, zorder=4)


def _phase_instant_overlay(ax, ps, snap_num, n_ctr, row, color, label=None):
    """Overlay instantaneous binned median ± 1σ for a single snapshot."""
    snaps = ps['snap_nums']
    idx_arr = np.where(snaps == snap_num)[0]
    if len(idx_arr) == 0:
        # Use nearest snapshot
        idx_arr = [int(np.argmin(np.abs(snaps - snap_num)))]
    i = idx_arr[0]

    key_med = 'T_med' if row == 'T' else 'fH2_med'
    key_p16 = 'T_p16' if row == 'T' else 'fH2_p16'
    key_p84 = 'T_p84' if row == 'T' else 'fH2_p84'

    med = ps[key_med][i]
    p16 = ps[key_p16][i]
    p84 = ps[key_p84][i]

    valid = np.isfinite(med)
    if not valid.any():
        return

    xv = n_ctr[valid]
    ax.plot(xv, med[valid], color=color, lw=3.6, ls='-', zorder=5,
            label=label, alpha=0.9)
    ax.fill_between(xv, p16[valid], p84[valid], color=color, alpha=0.15, zorder=4)


@_use_thin_border
def plot_phase_diagrams(epoch_data_list, outdir, rho_thresh=1e-15,
                        frames_dir=None, kg_data=None):
    """6-panel T vs n + H₂ vs n scatter phase diagram for first 3 epochs.

    Layout: 2 rows × 3 columns
      Row 0: T vs n for epochs 0, 1, 2
      Row 1: H₂ vs n for epochs 0, 1, 2

    Full-sim data (if available) is plotted as a faint background on T-n panels.

    Overlays (if frames_dir given and phase_stats.npz exists):
      - Cumulative: mean ± std of per-snapshot binned medians between epoch pairs
        (nothing on panel 0; between epochs 0-1 on panel 1; between 1-2 on panel 2)
      - A second figure ('phase_instant_median.png') with instantaneous
        binned median ± 1σ for each of the 3 epochs.

    kg_data: optional dict with keys 'T_n_logn', 'T_n_logT_Z0', 'T_n_logT_Z1em4',
             'fH2_n_logn', 'fH2_Z0', 'fH2_Z1em4' for KG2023 reference curves.
    """
    _tick_kw = dict(colors='k', which='both', direction='in', right=True, top=True,
                    labelcolor='k')

    # Use first 3 epochs only
    eds = epoch_data_list[:3]

    # Determine axis bounds from the data
    all_logn_cutout, all_logT, all_logfh2 = [], [], []
    for ed in eds:
        n = ed['n_local']; T = ed['T_local']
        v = (n > 0) & (T > 0)
        if v.any():
            all_logn_cutout.append(np.log10(n[v]))
            all_logT.append(np.log10(T[v]))
        fh2 = ed.get('fh2_local')
        if fh2 is not None:
            vf = v & (fh2 > 0)
            if vf.any():
                all_logfh2.append(np.log10(fh2[vf]))

    if not all_logn_cutout:
        print("  No valid phase data, skipping.")
        return

    all_logn_cutout = np.concatenate(all_logn_cutout)
    all_logT = np.concatenate(all_logT)
    # Tight bounds with 5% padding
    def _bounds(arr, pad=0.05):
        lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
        span = max(hi - lo, 0.5)
        return lo - pad * span, hi + pad * span

    # Cutout-only bounds (panels 1,2) and full-range bounds (panel 0)
    n_lo_cutout, n_hi = _bounds(all_logn_cutout)
    n_hi = max(n_hi, 14.5)  # extend to show highest densities
    T_lo, T_hi = _bounds(all_logT)

    # For panel 0, include full-sim range
    n_lo_full = n_lo_cutout
    _n_fs_h2, _fh2_fs_h2 = None, None
    if _FULLSIM_PATH is not None:
        n_fs0, T_fs0, fh2_fs0 = load_fullsim_phase(eds[0]['snap_num'], _FULLSIM_PATH)
        if n_fs0 is not None and len(n_fs0) > 0:
            v_fs = (n_fs0 > 0) & (T_fs0 > 0)
            if v_fs.any():
                all_logn_full = np.concatenate([all_logn_cutout, np.log10(n_fs0[v_fs])])
                n_lo_full, _ = _bounds(all_logn_full)

    # Load H₂ background from full-sim snap 27 (last snap with 28-field output)
    # Fallback to snap 0 if snap 27 has no useful H₂ data
    if _FULLSIM_PATH is not None:
        _n_fs_h2, _, _fh2_fs_h2 = load_fullsim_phase(27, _FULLSIM_PATH)
        if _fh2_fs_h2 is None or not np.any(_fh2_fs_h2 > 1e-6):
            print("    snap 27 H₂ data absent or all < 1e-6; trying snap 0...")
            _n_fs_h2, _, _fh2_fs_h2 = load_fullsim_phase(0, _FULLSIM_PATH)
    else:
        _n_fs_h2, _fh2_fs_h2 = None, None

    has_h2 = len(all_logfh2) > 0
    if has_h2:
        all_logfh2 = np.concatenate(all_logfh2)
        fh2_lo, fh2_hi = _bounds(all_logfh2)
    else:
        fh2_lo, fh2_hi = -6.0, 0.0

    n_rows = 2 if has_h2 else 1
    # 4:3 subplots for full-page figure; 75% font sizes
    _ph_fs = {k: v * 0.75 for k, v in {
        'axes.labelsize': plt.rcParams['axes.labelsize'],
        'xtick.labelsize': plt.rcParams['xtick.labelsize'],
        'ytick.labelsize': plt.rcParams['ytick.labelsize'],
        'legend.fontsize': plt.rcParams['legend.fontsize'],
    }.items()}
    _ph_fs['legend.fontsize'] *= 1.25
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 9),
                             squeeze=False,
                             gridspec_kw={'hspace': 0, 'wspace': 0})
    fig.patch.set_facecolor('w')
    for ax_row in axes:
        for ax in ax_row:
            ax.tick_params(labelsize=_ph_fs['xtick.labelsize'])

    for j, ed in enumerate(eds):
        n_dens = ed['n_local']
        T = ed['T_local']
        mass = ed['mass_local']
        disk = ed['is_disk_local']
        fh2 = ed.get('fh2_local')

        valid = (n_dens > 0) & (T > 0)
        logn = np.log10(n_dens[valid])
        logT = np.log10(T[valid])

        # ── Row 0: T vs n ──
        ax = axes[0, j]
        ax.set_facecolor('w')

        # Full-sim background scatter (first panel only)
        if j == 0 and _FULLSIM_PATH is not None:
            # Reuse cached data from bounds computation if available
            if 'n_fs0' in dir() and n_fs0 is not None and len(n_fs0) > 0:
                _n_fs, _T_fs, _fh2_fs = n_fs0, T_fs0, fh2_fs0
            else:
                _n_fs, _T_fs, _fh2_fs = load_fullsim_phase(ed['snap_num'], _FULLSIM_PATH)
            if _n_fs is not None and len(_n_fs) > 0:
                ax.scatter(np.log10(_n_fs), np.log10(_T_fs),
                           s=1.0, alpha=0.5, c='blue', rasterized=True,
                           label='full sim')

        # Non-disk cutout particles
        nondisk = valid & ~disk
        if nondisk.any():
            ax.scatter(np.log10(n_dens[nondisk]), np.log10(T[nondisk]),
                       s=0.5, alpha=0.4, c='red', rasterized=True,
                       label='cutout (non-disk)' if j == 0 else None)
        # Disk particles on top
        if disk.sum() > 0 and (valid & disk).any():
            d_logn = np.log10(n_dens[valid & disk])
            d_logT = np.log10(T[valid & disk])
            ax.scatter(d_logn, d_logT, s=0.5, alpha=0.3, c='green', rasterized=True,
                       label='disk' if j == 0 else None)

        # Per-panel x-limits: panel 0 includes full sim range, others cutout only
        _n_lo_j = n_lo_full if j == 0 else n_lo_cutout
        ax.set_xlim(_n_lo_j, n_hi)
        ax.set_ylim(T_lo, T_hi)
        ax.minorticks_on()
        ax.tick_params(**_tick_kw)
        for sp in ax.spines.values(): sp.set_edgecolor('k')
        ax.text(0.03, 0.95, ed['time_label'], transform=ax.transAxes,
                va='top', color='k', fontsize=_ph_fs['legend.fontsize'],
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        if j == 0:
            ax.set_ylabel(r'$\log_{10}\ T\ \mathrm{(K)}$', fontsize=_ph_fs['axes.labelsize'])
            _legend_handles, _legend_labels = ax.get_legend_handles_labels()
        else:
            ax.set_yticklabels([])
        # x-labels only on bottom row
        if has_h2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r'$\log_{10}\ n$ (cm$^{-3}$)', fontsize=_ph_fs['axes.labelsize'])

        # ── Row 1: H₂ vs n ──
        if has_h2 and fh2 is not None:
            ax2 = axes[1, j]
            ax2.set_facecolor('w')

            # Full-sim H₂ background (all panels)
            if _fh2_fs_h2 is not None:
                _v_fh2 = (_n_fs_h2 > 0) & (_fh2_fs_h2 > 1e-6)
                if _v_fh2.any():
                    ax2.scatter(np.log10(_n_fs_h2[_v_fh2]), np.log10(_fh2_fs_h2[_v_fh2]),
                                s=1.0, alpha=0.5, c='blue', rasterized=True)

            valid2 = valid & (fh2 > 0)
            if valid2.any():
                # Non-disk cutout
                nd2 = valid2 & ~disk
                if nd2.any():
                    ax2.scatter(np.log10(n_dens[nd2]), np.log10(fh2[nd2]),
                                s=0.5, alpha=0.25, c='red', rasterized=True)
                # Disk
                if disk.sum() > 0 and (valid2 & disk).any():
                    d2 = valid2 & disk
                    ax2.scatter(np.log10(n_dens[d2]), np.log10(fh2[d2]),
                                s=0.5, alpha=0.15, c='green', rasterized=True)
            ax2.set_xlim(_n_lo_j, n_hi)
            ax2.set_ylim(max(fh2_lo, -6.0), fh2_hi)
            ax2.set_xlabel(r'$\log_{10}\ n$ (cm$^{-3}$)', fontsize=_ph_fs['axes.labelsize'])
            ax2.minorticks_on()
            ax2.tick_params(**_tick_kw, labelsize=_ph_fs['xtick.labelsize'])
            for sp in ax2.spines.values(): sp.set_edgecolor('k')
            if j == 0:
                ax2.set_ylabel(r'$\log_{10}\ f_{\rm H_2}$', fontsize=_ph_fs['axes.labelsize'])
            elif j == 1:
                ax2.set_yticklabels([])
                if _legend_handles:
                    ax2.legend(_legend_handles, _legend_labels, loc='lower right',
                               markerscale=15, framealpha=0.7, fontsize=_ph_fs['legend.fontsize'])
            else:
                ax2.set_yticklabels([])

    # ── KG2023 reference curves (drawn before overlays so scatter sits on top) ──
    # Each metallicity may have its own n-grid
    _kg_n_Z0  = kg_data.get('T_n_logn_Z0')    if kg_data else None
    _kg_T_Z0  = kg_data.get('T_n_logT_Z0')    if kg_data else None
    _kg_n_Z1  = kg_data.get('T_n_logn_Z1em4') if kg_data else None
    _kg_T_Z1  = kg_data.get('T_n_logT_Z1em4') if kg_data else None
    _kg_f_n   = kg_data.get('fH2_n_logn')     if kg_data else None
    _kg_f_Z0  = kg_data.get('fH2_Z0')         if kg_data else None
    _kg_f_Z1  = kg_data.get('fH2_Z1em4')      if kg_data else None
    if kg_data is not None:
        for j in range(len(eds)):
            if _kg_n_Z0 is not None and _kg_T_Z0 is not None:
                axes[0, j].plot(_kg_n_Z0, _kg_T_Z0, 'k-',  lw=2.4, alpha=0.7, zorder=2,
                                label='KG23 Z=0'           if j == 0 else None)
            if has_h2:
                if _kg_f_n is not None and _kg_f_Z0 is not None:
                    axes[1, j].plot(_kg_f_n, _kg_f_Z0, 'k-',  lw=2.4, alpha=0.7, zorder=2,
                                    label='KG23 Z=0'          if j == 0 else None)

    # ── Phase stats overlays (cumulative mean±std between epoch pairs) ──
    ps = _load_phase_stats(frames_dir)
    if ps is not None:
        n_ctr = ps['n_ctr']
        _ov_colors = ['#555555', '#aa3333', '#3366aa']   # grey, dark-red, dark-blue
        # Resolve epoch times for legend labels
        _t1_ps = float(ps['t1_Myr'][0]) if not np.isnan(ps['t1_Myr'][0]) else None
        for j, ed in enumerate(eds):
            if j == 0:
                continue   # Panel 0: no overlay (nothing before first epoch)
            snap_prev = eds[j - 1]['snap_num']
            snap_curr = ed['snap_num']
            col = _ov_colors[j % len(_ov_colors)]
            # Build a time-range label using epoch time_labels
            t_prev_lbl = eds[j - 1]['time_label']
            t_curr_lbl = ed['time_label']
            lbl = f'median {t_prev_lbl}–{t_curr_lbl}'
            _phase_cumulative_overlay(axes[0, j], ps, snap_prev, snap_curr,
                                      n_ctr, 'T', col, label=lbl)
            axes[0, j].legend(loc='lower right', fontsize=_ph_fs['legend.fontsize'] * 0.8,
                              framealpha=0.7)
            if has_h2:
                _phase_cumulative_overlay(axes[1, j], ps, snap_prev, snap_curr,
                                          n_ctr, 'fH2', col, label=lbl)
                axes[1, j].legend(loc='lower right',
                                  fontsize=_ph_fs['legend.fontsize'] * 0.8, framealpha=0.7)

    # KG legend on panel 0 (T row) and panel 0 (fH2 row)
    if kg_data is not None:
        axes[0, 0].legend(loc='lower right', fontsize=_ph_fs['legend.fontsize'] * 0.8,
                          framealpha=0.7)
        if has_h2:
            axes[1, 0].legend(loc='lower right', fontsize=_ph_fs['legend.fontsize'] * 0.8,
                              framealpha=0.7)

    _save_fig_dual(fig, os.path.join(outdir, 'light', 'phase_combined.png'))

    # ── Second figure: instantaneous binned median ± 1σ per epoch ──
    if ps is not None:
        n_ctr = ps['n_ctr']
        fig2, axes2 = plt.subplots(n_rows, 3, figsize=(18, 9),
                                   squeeze=False,
                                   gridspec_kw={'hspace': 0, 'wspace': 0})
        fig2.patch.set_facecolor('w')
        for j, ed in enumerate(eds):
            col = EPOCH_COLORS[j % len(EPOCH_COLORS)]
            lbl = ed['time_label']
            n_dens = ed['n_local']
            T_ep   = ed['T_local']
            mass   = ed['mass_local']
            disk   = ed['is_disk_local']
            fh2_ep = ed.get('fh2_local')
            valid  = (n_dens > 0) & (T_ep > 0)

            ax = axes2[0, j]
            ax.set_facecolor('w')
            # Scatter (same as main plot) — drawn first so median sits on top
            nondisk = valid & ~disk
            if nondisk.any():
                ax.scatter(np.log10(n_dens[nondisk]), np.log10(T_ep[nondisk]),
                           s=0.5, alpha=0.4, c='red', rasterized=True,
                           label='cutout (non-disk)' if j == 0 else None)
            if disk.sum() > 0 and (valid & disk).any():
                ax.scatter(np.log10(n_dens[valid & disk]), np.log10(T_ep[valid & disk]),
                           s=0.5, alpha=0.3, c='green', rasterized=True,
                           label='disk' if j == 0 else None)
            # KG reference
            if kg_data is not None:
                if _kg_n_Z0 is not None and _kg_T_Z0 is not None:
                    ax.plot(_kg_n_Z0, _kg_T_Z0, 'k-',  lw=2.4, alpha=0.7, zorder=3,
                            label='KG23 Z=0'           if j == 0 else None)
            # Instantaneous binned median on top
            _phase_instant_overlay(ax, ps, ed['snap_num'], n_ctr, 'T', col,
                                   label=lbl + ' median')

            _n_lo_j = n_lo_full if j == 0 else n_lo_cutout
            ax.set_xlim(_n_lo_j, n_hi); ax.set_ylim(T_lo, T_hi)
            ax.minorticks_on(); ax.tick_params(**_tick_kw)
            for sp in ax.spines.values(): sp.set_edgecolor('k')
            ax.text(0.03, 0.95, lbl, transform=ax.transAxes, va='top', color='k',
                    fontsize=_ph_fs['legend.fontsize'],
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
            ax.legend(loc='lower right', fontsize=_ph_fs['legend.fontsize'] * 0.75,
                      framealpha=0.7, markerscale=8)
            if j == 0:
                ax.set_ylabel(r'$\log_{10}\ T\ \mathrm{(K)}$',
                              fontsize=_ph_fs['axes.labelsize'])
            else:
                ax.set_yticklabels([])
            if has_h2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(r'$\log_{10}\ n$ (cm$^{-3}$)',
                              fontsize=_ph_fs['axes.labelsize'])

            if has_h2 and fh2_ep is not None:
                ax2 = axes2[1, j]
                ax2.set_facecolor('w')
                # Scatter
                valid2 = valid & (fh2_ep > 0)
                if valid2.any():
                    nd2 = valid2 & ~disk
                    if nd2.any():
                        ax2.scatter(np.log10(n_dens[nd2]), np.log10(fh2_ep[nd2]),
                                    s=0.5, alpha=0.25, c='red', rasterized=True)
                    if disk.sum() > 0 and (valid2 & disk).any():
                        ax2.scatter(np.log10(n_dens[valid2 & disk]),
                                    np.log10(fh2_ep[valid2 & disk]),
                                    s=0.5, alpha=0.15, c='green', rasterized=True)
                # KG reference
                if kg_data is not None:
                    if _kg_f_n is not None and _kg_f_Z0 is not None:
                        ax2.plot(_kg_f_n, _kg_f_Z0, 'k-',  lw=2.4, alpha=0.7, zorder=3)
                    if _kg_f_n is not None and _kg_f_Z1 is not None:
                        ax2.plot(_kg_f_n, _kg_f_Z1, 'k--', lw=2.4, alpha=0.7, zorder=3)
                # Instantaneous binned median
                _phase_instant_overlay(ax2, ps, ed['snap_num'], n_ctr, 'fH2', col)
                ax2.set_xlim(_n_lo_j, n_hi)
                ax2.set_ylim(max(fh2_lo, -6.0), fh2_hi)
                ax2.set_xlabel(r'$\log_{10}\ n$ (cm$^{-3}$)',
                               fontsize=_ph_fs['axes.labelsize'])
                ax2.minorticks_on()
                ax2.tick_params(**_tick_kw, labelsize=_ph_fs['xtick.labelsize'])
                for sp in ax2.spines.values(): sp.set_edgecolor('k')
                if j == 0:
                    ax2.set_ylabel(r'$\log_{10}\ f_{\rm H_2}$',
                                   fontsize=_ph_fs['axes.labelsize'])
                else:
                    ax2.set_yticklabels([])

        _save_fig_dual(fig2, os.path.join(outdir, 'light', 'phase_instant_median.png'))
        print("  Instantaneous phase median figure saved.")


@_use_thin_border
def plot_bfield_phase(epoch_data_list, outdir):
    """6-panel |B| vs n scatter phase diagram for first 3 epochs.

    Layout: 2 rows × 3 columns
      Row 0: |B| vs n for first 3 epochs
      Row 1: |B| vs n for next 3 epochs (if available)
    Uses all 6 epochs in a 2×3 grid.
    """
    _tick_kw = dict(colors='k', which='both', direction='in', right=True, top=True)
    _bf_fs = {k: v * 0.5 for k, v in {
        'axes.labelsize': plt.rcParams['axes.labelsize'],
        'xtick.labelsize': plt.rcParams['xtick.labelsize'],
        'ytick.labelsize': plt.rcParams['ytick.labelsize'],
        'legend.fontsize': plt.rcParams['legend.fontsize'],
        'annotation.fontsize': plt.rcParams.get('font.size', 20),
    }.items()}
    _bf_fs['legend.fontsize'] *= 1.25

    eds = epoch_data_list[:6]
    # Check if any have B data
    has_any_B = any(ed.get('B_local') is not None and len(ed.get('B_local', [])) > 0
                    for ed in eds)
    if not has_any_B:
        print("  No B-field data, skipping B-field phase diagram.")
        return

    # Determine bounds from all epochs
    all_logn, all_logB = [], []
    for ed in eds:
        n = ed['n_local']; B = ed.get('B_local')
        if B is None:
            continue
        v = (n > 0) & (B > 0)
        if v.any():
            all_logn.append(np.log10(n[v]))
            all_logB.append(np.log10(B[v]))
    if not all_logn:
        return
    all_logn = np.concatenate(all_logn)
    all_logB = np.concatenate(all_logB)

    def _bounds(arr, pad=0.05):
        lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
        span = max(hi - lo, 0.5)
        return lo - pad * span, hi + pad * span

    n_lo, n_hi = _bounds(all_logn)
    B_lo, B_hi = _bounds(all_logB)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), squeeze=False,
                             gridspec_kw={'hspace': 0, 'wspace': 0})
    fig.patch.set_facecolor('w')

    for i, ed in enumerate(eds):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        ax.set_facecolor('w')

        n_dens = ed['n_local']
        B_mag = ed.get('B_local')
        disk = ed['is_disk_local']

        if B_mag is None or len(B_mag) == 0:
            ax.text(0.5, 0.5, 'No B data', transform=ax.transAxes, ha='center', va='center')
            continue

        valid = (n_dens > 0) & (B_mag > 0)
        non_disk_plotted = disk_plotted = False
        if valid.any():
            logn = np.log10(n_dens[valid])
            logB = np.log10(B_mag[valid])
            nd = valid & ~disk
            if nd.any():
                ax.scatter(np.log10(n_dens[nd]), np.log10(B_mag[nd]),
                           s=0.3, alpha=0.05, c='tab:red', rasterized=True,
                           label='non-disk')
                non_disk_plotted = True
            if disk.sum() > 0 and (valid & disk).any():
                d = valid & disk
                ax.scatter(np.log10(n_dens[d]), np.log10(B_mag[d]),
                           s=0.5, alpha=0.15, c='tab:green', rasterized=True,
                           label='disk')
                disk_plotted = True

        ax.set_xlim(n_lo, n_hi)
        ax.set_ylim(B_lo, B_hi)
        ax.tick_params(**_tick_kw, labelsize=_bf_fs['xtick.labelsize'])
        for sp in ax.spines.values(): sp.set_edgecolor('k')
        ax.text(0.03, 0.95, ed['time_label'], transform=ax.transAxes,
                va='top', color='k', fontsize=_bf_fs['annotation.fontsize'],
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        if col == 0:
            ax.set_ylabel(r'$\log_{10}\ |B|\ \mathrm{(G)}$',
                          fontsize=_bf_fs['axes.labelsize'])
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(r'$\log_{10}\ n$ (cm$^{-3}$)', fontsize=_bf_fs['axes.labelsize'])

        # Legend in bottom-centre panel
        if row == 1 and col == 1:
            from matplotlib.lines import Line2D
            legend_handles = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:red',
                       markersize=6, label='non-disk (cutout)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:green',
                       markersize=6, label='disk'),
            ]
            ax.legend(handles=legend_handles, loc='lower right',
                      fontsize=_bf_fs['legend.fontsize'],
                      framealpha=0.8)

    # Hide unused panels if < 6 epochs
    for i in range(len(eds), 6):
        axes[i // 3, i % 3].set_visible(False)

    _save_fig_dual(fig, os.path.join(outdir, 'light', 'phase_Bfield.png'))


# ═════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════
# Infall timescale: M_disk / Mdot
# ═════════════════════════════════════════════════════════════════════════════

def plot_infall_timescale(epoch_data_list, outdir, frames_dir=None):
    """Continuous t-series: M_disk, |dM_disk/dt|, t_infall = M_disk/|dMdot|.

    Loads mass_evolution.npz from frames_dir for continuous curves.
    Falls back to 6-point scatter if file not found.
    """
    from scipy.ndimage import gaussian_filter1d

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                         gridspec_kw={'hspace': 0})
    fig.patch.set_facecolor('w')

    me_path = os.path.join(frames_dir, 'mass_evolution.npz') if frames_dir else None
    used_continuous = False

    if me_path and os.path.exists(me_path):
        me     = np.load(me_path)
        t_abs  = me['times_Myr']
        M_disk = me['M_disk']      # (N,) Msun
        n_sinks = me['n_sinks']
        t1 = float(me['t1_Myr'][0]) if not np.isnan(me['t1_Myr'][0]) else None

        if t1 is None:
            idx1 = np.where(n_sinks > 0)[0]
            t1 = float(t_abs[idx1[0]]) if len(idx1) > 0 else float(t_abs[0])

        t_kyr = (t_abs - t1) * 1e3

        # Smooth M_disk and compute dM/dt [Msun/yr]
        M_sm   = gaussian_filter1d(M_disk, sigma=3.0)
        n_t    = len(t_abs)
        dMdt   = np.full(n_t, np.nan)
        for i in range(n_t):
            i_lo = max(i - 1, 0)
            i_hi = min(i + 1, n_t - 1)
            dt   = (t_abs[i_hi] - t_abs[i_lo]) * 1e6
            if dt > 0:
                dMdt[i] = (M_sm[i_hi] - M_sm[i_lo]) / dt
        dMdt = gaussian_filter1d(np.where(np.isfinite(dMdt), dMdt, 0.0), sigma=3.0)

        t_infall = np.full(n_t, np.nan)
        nz = np.abs(dMdt) > 0
        t_infall[nz] = M_sm[nz] / np.abs(dMdt[nz]) / 1e3   # yr → kyr

        fin = np.isfinite(t_infall) & (t_infall > 0) & (M_sm > 0)

        ax1.plot(t_kyr, M_sm, color='k', lw=3.0, zorder=4)
        ax1.set_yscale('log')

        ax2.plot(t_kyr[np.isfinite(dMdt)], dMdt[np.isfinite(dMdt)], color='k', lw=3.0)
        ax2.axhline(0, color='k', lw=1.0, ls=':')
        ax2.set_yscale('symlog', linthresh=max(np.nanpercentile(np.abs(dMdt[dMdt != 0]), 5)
                                               if np.any(dMdt != 0) else 1e-5, 1e-6))

        if fin.any():
            ax3.plot(t_kyr[fin], t_infall[fin], color='k', lw=3.0)
        ax3.set_yscale('log')

        # Mark epoch timestamps — look up by snap_num to avoid ΛCDM time mismatch
        _snap_arr = me['snap_nums'] if 'snap_nums' in me.files else None
        for j, ed in enumerate(epoch_data_list):
            if _snap_arr is not None:
                idx_ep = int(np.argmin(np.abs(_snap_arr - ed['snap_num'])))
                t_ep   = t_kyr[idx_ep]
            else:
                t_ep   = (ed['time_Myr'] - t1) * 1e3
            for ax in (ax1, ax2, ax3):
                ax.axvline(t_ep, color=EPOCH_COLORS[j], lw=1.6, ls=':', alpha=0.8)

        ax3.set_xlabel(r'$t - t_1$ [kyr]')
        used_continuous = True

    if not used_continuous:
        # Fallback: scatter for 6 epochs
        for i, ed in enumerate(epoch_data_list):
            M    = ed.get('M_disk_Msun', np.nan)
            Mdot = ed.get('Mdot_Msun_yr', np.nan)
            c    = EPOCH_COLORS[i]
            lbl  = ed['time_label']
            if np.isfinite(M) and M > 0:
                ax1.scatter(i, M, color=c, s=80)
                ax1.annotate(lbl, (i, M), textcoords='offset points', xytext=(5, 5), color=c)
            if np.isfinite(Mdot) and abs(Mdot) > 0:
                ax2.scatter(i, abs(Mdot), color=c, s=80,
                            marker='^' if Mdot > 0 else 'v')
            if np.isfinite(M) and np.isfinite(Mdot) and Mdot > 0 and M > 0:
                ax3.scatter(i, M / Mdot / 1e3, color=c, s=80)
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('Epoch index')

    ax1.set_ylabel(r'$M_{\rm disk}$ [M$_\odot$]')
    ax1.tick_params(direction='in', which='both', right=True, top=True, labelbottom=False)

    ax2.set_ylabel(r'$\dot{M}_{\rm disk}$ [M$_\odot$ yr$^{-1}$]')
    ax2.tick_params(direction='in', which='both', right=True, top=True, labelbottom=False)

    ax3.set_ylabel(r'$t_{\rm infall}$ [kyr]')
    ax3.tick_params(direction='in', which='both', right=True, top=True)

    _save_fig_dual(fig, os.path.join(outdir, 'light', 'infall_timescale.png'))


# ═════════════════════════════════════════════════════════════════════════════
# Merged Toomre Q: face-on grid + heatmap
# ═════════════════════════════════════════════════════════════════════════════

@_use_orig_rc
def plot_toomre_Q_merged(epoch_data_list, frames_dir, outdir,
                          merge_data=None, pos_history=None):
    """Merged figure: 2×3 face-on Toomre Q maps (top) + Q heatmap (bottom).

    Uses GridSpec with height_ratios to give the heatmap appropriate space.
    """
    from matplotlib.gridspec import GridSpec
    from notebooks.make_disk_movie_frames import _add_pos_lines, _add_merge_markers, _add_formation_markers

    Q_cmap = plt.colormaps.get_cmap('RdYlGn').copy()
    Q_cmap.set_under(Q_cmap(0.0))   # values < vmin → lowest color
    Q_cmap.set_over(Q_cmap(1.0))    # values > vmax → highest color
    Q_cmap.set_bad('white', alpha=0) # NaN → transparent
    Q_norm = colors.LogNorm(vmin=0.1, vmax=10)

    # Load heatmap data
    profile_files = sorted(glob.glob(os.path.join(frames_dir, 'qprofiles', 'qprofile_*.npz')))
    if not profile_files:
        print("  No qprofile data for merged Q figure, skipping.")
        return

    times, Q_rows, n_sinks_list = [], [], []
    _seen_form_Myr = {}
    for f in sorted(profile_files):
        d = np.load(f)
        t = float(np.atleast_1d(d['time_Myr'])[0])
        # Prefer combined (thermal+turbulent) Q if present, else fall back to turbulent
        Q = d['Q_combined'].copy() if 'Q_combined' in d.files else d['Q'].copy()
        r = d['r_AU'].copy()
        times.append(t)
        Q_rows.append((r, Q))
        n_sinks_list.append(int(d['n_sinks'][0]) if 'n_sinks' in d else 0)
        if 'sink_form_Myr' in d and 'sink_r_AU' in d:
            for tf, rf in zip(d['sink_form_Myr'], d['sink_r_AU']):
                tf_key = round(float(tf), 6)
                if tf_key not in _seen_form_Myr:
                    _seen_form_Myr[tf_key] = float(rf)

    r_AU_ref = max(Q_rows, key=lambda x: float(x[0].max()) if len(x[0]) > 0 else 0.0)[0]
    sort_idx = np.argsort(times)
    times_arr = np.array(times)[sort_idx]
    n_sinks_arr = np.array(n_sinks_list)[sort_idx]
    t1_Myr = float(min(_seen_form_Myr.keys())) if _seen_form_Myr else None
    if t1_Myr is None:
        sink_snaps = np.where(n_sinks_arr > 0)[0]
        t1_Myr = float(times_arr[sink_snaps[0]]) if len(sink_snaps) > 0 else None

    Q_mat = np.full((len(times_arr), len(r_AU_ref)), np.nan)
    for i, idx in enumerate(sort_idx):
        r_i, Q_i = Q_rows[idx]
        valid = np.isfinite(Q_i) & (Q_i > 0)
        if valid.sum() >= 2:
            # Interpolate only within valid range; clamp edges to nearest value
            Q_mat[i] = np.interp(r_AU_ref, r_i[valid], Q_i[valid])
        elif valid.sum() == 1:
            Q_mat[i] = Q_i[valid][0]
    # Forward-fill fully-empty rows
    for i in range(1, Q_mat.shape[0]):
        if np.all(np.isnan(Q_mat[i])):
            Q_mat[i] = Q_mat[i - 1]

    t_shifted = (times_arr - t1_Myr if t1_Myr is not None else times_arr) * 1e3

    # Build figure: 3 rows (2 for faceon Q grid, 1 for heatmap)
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 3, height_ratios=[1, 1, 0.8],
                  hspace=0, wspace=0,
                  left=0.08, right=0.92, top=0.97, bottom=0.06)

    # Top 2×3: face-on Q maps
    im_ref = None
    fo_axes = {}
    for i, ed in enumerate(epoch_data_list[:6]):
        row, col = i // 3, i % 3
        ax = fig.add_subplot(gs[row, col])
        Q_data = ed.get('Q_fo_combined', ed.get('Q_fo'))
        X, Y = ed['X'], ed['Y']
        half = ed['half_AU']

        if Q_data is not None:
            im = ax.pcolormesh(X, Y, Q_data, cmap=Q_cmap, norm=Q_norm, rasterized=True)
            im_ref = im
            try:
                ax.contour(X, Y, Q_data, levels=[1.0], colors='k', linewidths=0.8)
            except Exception:
                pass
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect('equal')
        ax.text(0.03, 0.95, ed['time_label'], transform=ax.transAxes,
                va='top', color='k',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

        if ed['n_stars'] > 0:
            sx, sy = ed['star_fo_AU'][:, 0], ed['star_fo_AU'][:, 1]
            in_box = (np.abs(sx) < half) & (np.abs(sy) < half)
            if in_box.any():
                ax.scatter(sx[in_box], sy[in_box], marker='*', s=30,
                           c='white', edgecolors='k', linewidths=0.5, zorder=5)

        _add_scale_bar(ax, 1000, '1000 AU', color='black')
        ax.tick_params(direction='in', which='both', top=True, right=True)
        ax.set_xticks([])  # no x ticks on face-on panels
        if col == 0:
            ax.set_ylabel('y [AU]')
        else:
            ax.set_yticklabels([])
        fo_axes[i] = ax

    # Prune overlapping y-ticks between row 0 and row 1
    _prune_shared_ticks(np.array([
        [fo_axes[0], fo_axes[1], fo_axes[2]],
        [fo_axes[3], fo_axes[4], fo_axes[5]],
    ]))

    # Colorbar for Q — spans full figure height
    if im_ref is not None:
        cbar_ax = fig.add_axes([0.925, 0.06, 0.015, 0.91])
        fig.colorbar(im_ref, cax=cbar_ax, label=r'Toomre $Q$', extend='both')

    # Bottom row: Q heatmap spanning all 3 columns
    ax_hm = fig.add_subplot(gs[2, :])
    ax_hm.set_facecolor('w')

    dt = np.diff(t_shifted)
    t_lo = np.concatenate([[t_shifted[0] - dt[0]/2], t_shifted[:-1] + dt/2])
    t_hi = np.concatenate([t_shifted[:-1] + dt/2, [t_shifted[-1] + dt[-1]/2]])
    T = np.concatenate([t_lo, [t_hi[-1]]])
    dr = np.diff(r_AU_ref)
    r_lo = np.concatenate([[r_AU_ref[0] - dr[0]/2], r_AU_ref[:-1] + dr/2])
    r_hi = np.concatenate([r_AU_ref[:-1] + dr/2, [r_AU_ref[-1] + dr[-1]/2]])
    R = np.concatenate([r_lo, [r_hi[-1]]])

    Tg, Rg = np.meshgrid(T, R, indexing='ij')
    im_hm = ax_hm.pcolormesh(Tg, Rg, Q_mat, norm=Q_norm,
                              cmap=Q_cmap, rasterized=True)

    Tc, Rc = np.meshgrid(t_shifted, r_AU_ref, indexing='ij')
    Q_filled = np.where(np.isfinite(Q_mat), Q_mat, 1.0)
    try:
        ax_hm.contour(Tc, Rc, Q_filled, levels=[1.0], colors='k', linewidths=1.5)
    except Exception:
        pass

    _add_formation_markers(ax_hm, merge_data, r_AU_ref)

    _add_merge_markers(ax_hm, merge_data, r_AU_ref, t1_Myr)
    if pos_history is not None:
        _add_pos_lines(ax_hm, pos_history)

    xlabel = (r'$t - t_1$ (kyr)' if t1_Myr is not None else 'Time (kyr)')
    ax_hm.set_xlabel(xlabel)
    ax_hm.set_ylabel('r (AU)')
    ax_hm.set_ylim(0, 2500)
    ax_hm.set_yticks([0, 500, 1000, 1500, 2000])
    ax_hm.tick_params(direction='in', which='both', right=True, top=True)
    for sp in ax_hm.spines.values(): sp.set_edgecolor('k')

    n_finite = np.isfinite(Q_mat).sum(axis=1)
    peak_cov = n_finite.max()
    if peak_cov > 0:
        # Find last row with actual data (not just forward-filled NaN rows)
        has_data = n_finite > 0
        last_data_idx = np.where(has_data)[0][-1] if has_data.any() else len(t_shifted) - 1
        dense_rows = np.where(n_finite >= peak_cov * 0.25)[0]
        t_dense = t_shifted[dense_rows]
        t_lo_lim = np.percentile(t_dense, 5)
        t_hi_lim = t_shifted[last_data_idx]
        span = max(t_hi_lim - t_lo_lim, 1.0)
        ax_hm.set_xlim([t_lo_lim - span * 0.05, t_hi_lim])

    ax_hm.legend()

    _save_fig_dual(fig, os.path.join(outdir, 'light', 'toomre_Q_merged.png'))
    print(f"  Merged Toomre Q figure saved.")


# ═════════════════════════════════════════════════════════════════════════════
# Shell mass and accretion rate profiles
# ═════════════════════════════════════════════════════════════════════════════

def plot_shell_mass_accretion(epoch_data_list, outdir, frames_dir=None):
    """4-panel vertical: ρ(r), M_sphere(r), Mdot_shell(r), α_vir(r).

    Uses all 6 epoch snapshots with viridis colorbar for time.
    Mdot computed from centered finite differences of M_enc between epochs.
    """
    import matplotlib.cm as cm

    _t1 = next((ed['t1_Myr'] for ed in epoch_data_list
                if ed.get('t1_Myr') is not None), None)
    _AU_per_pc = 206265.0

    eds = epoch_data_list[:6]
    dt_vals = []
    for ed in eds:
        dt = ((ed['time_Myr'] - _t1) * 1e3) if _t1 is not None else 0.0
        dt_vals.append(dt)

    _cmap = plt.colormaps.get_cmap('viridis')
    dt_min = min(dt_vals) if dt_vals else 0
    dt_max = max(dt_vals) if dt_vals else 0
    dt_span = max(dt_max - dt_min, 0.1)

    def _color(dt):
        return _cmap(0.1 + 0.8 * (dt - dt_min) / dt_span)

    fig, (ax_rho, ax_m, ax_md, ax_vir) = plt.subplots(
        4, 1, figsize=(12, 36), sharex=True,
        gridspec_kw={'hspace': 0})
    fig.patch.set_facecolor('w')

    for i, ed in enumerate(eds):
        r = ed.get('sph_ctr_AU')
        if r is None or len(r) == 0:
            continue
        c = _color(dt_vals[i])
        r_pc = r / _AU_per_pc

        rho = ed.get('rho_sph_prof')
        if rho is not None:
            v = rho > 0
            if v.any():
                ax_rho.loglog(r_pc[v], rho[v], color=c, lw=3.0)

        m_enc = ed.get('m_enc_prof')
        if m_enc is not None:
            v = m_enc > 0
            if v.any():
                ax_m.loglog(r_pc[v], m_enc[v], color=c, lw=3.0)

        vir = ed.get('virial_sph_prof')
        if vir is not None:
            v = np.isfinite(vir) & (vir > 0)
            if v.any():
                ax_vir.semilogx(r_pc[v], vir[v], color=c, lw=3.0)

    # Mdot from centered finite differences of m_enc between adjacent epochs
    for i, ed in enumerate(eds):
        r = ed.get('sph_ctr_AU')
        m_enc = ed.get('m_enc_prof')
        if r is None or m_enc is None:
            continue
        i_lo = max(i - 1, 0)
        i_hi = min(i + 1, len(eds) - 1)
        if i_lo == i_hi:
            continue
        m_lo = eds[i_lo].get('m_enc_prof')
        m_hi = eds[i_hi].get('m_enc_prof')
        t_lo = eds[i_lo].get('time_Myr', 0)
        t_hi = eds[i_hi].get('time_Myr', 0)
        if m_lo is None or m_hi is None:
            continue
        dt_yr = (t_hi - t_lo) * 1e6
        if dt_yr <= 0 or len(m_lo) != len(m_hi):
            continue
        dm_dt = (m_hi - m_lo) / dt_yr
        r_pc = r / _AU_per_pc
        pos = np.isfinite(dm_dt) & (dm_dt > 0)
        if pos.any():
            ax_md.loglog(r_pc[pos], dm_dt[pos], color=_color(dt_vals[i]), lw=3.0)

    _xlim = (1e-6, 600)
    _tick_kw = dict(direction='in', which='both', top=True, right=True,
                    labelsize=plt.rcParams['xtick.labelsize'])
    _lbl_fs = plt.rcParams['axes.labelsize']

    ax_rho.set_ylabel(r'$\rho$ [g cm$^{-3}$]', fontsize=_lbl_fs)
    ax_rho.set_xlim(*_xlim)
    ax_rho.tick_params(**_tick_kw, labelbottom=False)

    ax_m.set_ylabel(r'$M(<r)$ [M$_\odot$]', fontsize=_lbl_fs)
    ax_m.tick_params(**_tick_kw, labelbottom=False)

    ax_md.set_ylabel(r'$\dot{M}_{\rm shell}$ [M$_\odot$ yr$^{-1}$]', fontsize=_lbl_fs)
    ax_md.set_ylim(bottom=4e-7)
    ax_md.tick_params(**_tick_kw, labelbottom=False)

    ax_vir.set_ylabel(r'$\alpha_{\rm vir}$', fontsize=_lbl_fs)
    ax_vir.set_xscale('log')
    ax_vir.set_ylim(0, 10)
    ax_vir.axhline(1.0, color='grey', ls='--', lw=1.5, alpha=0.6)
    ax_vir.set_xlabel('r [pc]', fontsize=_lbl_fs)
    ax_vir.tick_params(**_tick_kw)

    sm = cm.ScalarMappable(cmap=_cmap,
                           norm=plt.Normalize(vmin=dt_min, vmax=dt_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_rho, ax_m, ax_md, ax_vir],
                        location='right', pad=0.02, aspect=60, shrink=0.6)
    cbar.set_label(r'$t - t_1$ [kyr]', fontsize=plt.rcParams['axes.labelsize'])
    cbar.ax.tick_params(labelsize=plt.rcParams['xtick.labelsize'])

    _save_fig_dual(fig, os.path.join(outdir, 'light', 'profile_shell_mass_accretion.png'))
    print("  Radial profiles (ρ, M, Mdot, α_vir) saved.")


def plot_mass_accretion_heatmaps(frames_dir, t1_Myr, outdir, merge_data=None):
    """2-row heatmap figure: shell mass (top) and dM/dt (bottom) vs (t, r).

    Loads mass_evolution.npz (single pre-computed file). Falls back to
    per-snapshot massprofile_*.npz files if not found.
    """
    from notebooks.make_disk_movie_frames import _add_formation_markers
    from scipy.ndimage import gaussian_filter1d

    me_path = os.path.join(frames_dir, 'mass_evolution.npz')
    if os.path.exists(me_path):
        me        = np.load(me_path)
        times_arr = me['times_Myr']
        r_AU_ref  = me['r_AU']
        M_mat     = me['M_shell']          # (N, N_r)
        n_sinks_s = me['n_sinks']
        _t1 = float(me['t1_Myr'][0]) if not np.isnan(me['t1_Myr'][0]) else None
        # Always use npz-internal t1 (externally-passed t1_Myr uses a different
        # ΛCDM integration giving ~6 Myr offset → wrong t_kyr axis)
        if _t1 is not None:
            t1_Myr = _t1
    else:
        # Fallback: load individual massprofile files
        mp_files = sorted(glob.glob(
            os.path.join(frames_dir, 'massprofiles', 'massprofile_*.npz')))
        if not mp_files:
            print("  No mass_evolution.npz or massprofile data found.")
            return
        times, m_rows, n_sinks_list = [], [], []
        for f in mp_files:
            d = np.load(f)
            times.append(float(np.atleast_1d(d['time_Myr'])[0]))
            m_rows.append((d['r_AU'].copy(), d['m_shell'].copy()))
            n_sinks_list.append(int(np.atleast_1d(d['n_sinks'])[0]) if 'n_sinks' in d else 0)
        sort_idx  = np.argsort(times)
        times_arr = np.array(times)[sort_idx]
        n_sinks_s = np.array(n_sinks_list)[sort_idx]
        r_AU_ref  = max(m_rows, key=lambda x: float(x[0].max()) if len(x[0]) > 0 else 0)[0]
        n_t, n_r  = len(times_arr), len(r_AU_ref)
        M_mat = np.full((n_t, n_r), np.nan)
        for i, idx in enumerate(sort_idx):
            r_i, m_i = m_rows[idx]
            valid = np.isfinite(m_i) & (m_i > 0)
            if valid.sum() >= 2:
                M_mat[i] = np.interp(r_AU_ref, r_i[valid], m_i[valid])

    if t1_Myr is None:
        sink_snaps = np.where(np.array(n_sinks_s) > 0)[0]
        t1_Myr = float(times_arr[sink_snaps[0]]) if len(sink_snaps) > 0 else times_arr[0]
    t_kyr = (times_arr - t1_Myr) * 1e3
    n_t, n_r = M_mat.shape

    # ── Central finite difference dm/dt along time axis ──
    dt_yr      = np.zeros(n_t)
    dt_yr[1:-1] = (times_arr[2:] - times_arr[:-2]) * 1e6
    dt_yr[0]   = (times_arr[1]  - times_arr[0])   * 1e6
    dt_yr[-1]  = (times_arr[-1] - times_arr[-2])  * 1e6

    Md_mat = np.full_like(M_mat, np.nan)
    for i in range(n_t):
        i_lo = max(i - 1, 0)
        i_hi = min(i + 1, n_t - 1)
        dt = (times_arr[i_hi] - times_arr[i_lo]) * 1e6
        if dt > 0 and np.any(np.isfinite(M_mat[i_lo])) and np.any(np.isfinite(M_mat[i_hi])):
            Md_mat[i] = (M_mat[i_hi] - M_mat[i_lo]) / dt

    # Gaussian smoothing along time axis (sigma=10 → ~270 yr, enough to
    # suppress sink orbital noise while preserving kyr-scale structure)
    for b in range(n_r):
        col = Md_mat[:, b]
        finite = np.isfinite(col)
        if finite.sum() > 5:
            col[finite] = gaussian_filter1d(col[finite], sigma=10.0)
            Md_mat[:, b] = col

    fig, (ax_m, ax_md) = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                       gridspec_kw={'hspace': 0})

    # ── Top: shell mass ──
    m_vals = M_mat[np.isfinite(M_mat) & (M_mat > 0)]
    m_norm = (colors.LogNorm(vmin=max(np.percentile(m_vals, 2), 1e-4),
                             vmax=np.percentile(m_vals, 99))
              if len(m_vals) > 0 else colors.LogNorm(1e-4, 1e3))
    im_m = ax_m.pcolormesh(t_kyr, r_AU_ref, M_mat.T,
                            cmap='plasma', norm=m_norm, shading='auto')
    cb_m = fig.colorbar(im_m, ax=ax_m, pad=0.01)
    cb_m.set_label(r'$M_{\rm shell}$ [M$_\odot$]')
    ax_m.set_yscale('log')
    ax_m.set_ylabel('r [AU]')
    ax_m.set_ylim(r_AU_ref[0], r_AU_ref[-1])
    ax_m.tick_params(labelbottom=False, direction='in', which='both', right=True)
    _add_formation_markers(ax_m, merge_data, r_AU_ref)

    # ── Bottom: dM/dt ──
    md_fin = Md_mat[np.isfinite(Md_mat)]
    if len(md_fin) > 0:
        vmax_md = np.percentile(np.abs(md_fin), 98)
        nz = md_fin[md_fin != 0]
        linthresh_md = max(np.percentile(np.abs(nz), 10), 1e-8) if len(nz) > 0 else 1e-5
    else:
        vmax_md, linthresh_md = 1e-3, 1e-5
    md_norm  = colors.SymLogNorm(linthresh=linthresh_md, vmin=-vmax_md, vmax=vmax_md)
    dm_cmap  = plt.colormaps.get_cmap('RdBu_r').copy()
    dm_cmap.set_bad('grey', alpha=0.3)

    im_md = ax_md.pcolormesh(t_kyr, r_AU_ref, Md_mat.T,
                              cmap=dm_cmap, norm=md_norm, shading='auto')
    cb_md = fig.colorbar(im_md, ax=ax_md, pad=0.01, extend='both')
    cb_md.set_label(r'$\dot{M}$ [M$_\odot$ yr$^{-1}$]')
    ax_md.set_yscale('log')
    ax_md.set_ylabel('r [AU]')
    ax_md.set_xlabel(r'$t - t_1$ [kyr]')
    ax_md.set_ylim(r_AU_ref[0], r_AU_ref[-1])
    ax_md.tick_params(direction='in', which='both', right=True)
    _add_formation_markers(ax_md, merge_data, r_AU_ref)

    _save_fig_dual(fig, os.path.join(outdir, 'light', 'heatmap_mass_accretion.png'))
    print("  Mass + accretion heatmaps saved.")


# ═════════════════════════════════════════════════════════════════════════════
# Kinematic radial profiles (2×3 grid per figure, 2 figures)
# ═════════════════════════════════════════════════════════════════════════════

@_use_thin_border
def plot_kinematic_radial_profiles(epoch_data_list, outdir):
    """Three kinematic profile figures: disk, wide, and combined (symlog).

    Figure 1 (disk-only, cylindrical, small box):
      Each panel shows v_r, v_phi, c_s, δv, σ_r vs r_cyl [AU] for one epoch.

    Figure 2 (disk+environment, wide cylindrical, ~50 kAU):
      Same quantities but using wide-radius bins (mach_wide_bin_AU).

    Figure 3 (combined): symlog x-axis merging disk and wide data, 3×2 layout.
    """
    from matplotlib.lines import Line2D
    from scipy.interpolate import UnivariateSpline
    _tick_kw = dict(colors='k', which='both', direction='in', right=True, top=True)
    _tick_major_w = 1.0
    _tick_minor_w = 1.0
    eds = epoch_data_list[:6]

    # 75% font sizes for full-page kinematic figures
    _kin_fs = {k: v * 0.75 for k, v in {
        'axes.labelsize': plt.rcParams['axes.labelsize'],
        'xtick.labelsize': plt.rcParams['xtick.labelsize'],
        'ytick.labelsize': plt.rcParams['ytick.labelsize'],
        'legend.fontsize': plt.rcParams['legend.fontsize'],
    }.items()}

    def _smooth(r, v, s_factor=None):
        """Smooth a profile with a spline. Returns (r_smooth, v_smooth)."""
        finite = np.isfinite(v) & np.isfinite(r)
        if finite.sum() < 4:
            return r[finite], v[finite]
        r_f, v_f = r[finite], v[finite]
        sort = np.argsort(r_f)
        r_f, v_f = r_f[sort], v_f[sort]
        if len(r_f) < 4:
            return r_f, v_f
        if s_factor is None:
            s_factor = len(r_f) * 1.0
        try:
            spl = UnivariateSpline(r_f, v_f, s=s_factor, k=3)
            r_dense = np.linspace(r_f.min(), r_f.max(), max(len(r_f) * 3, 100))
            return r_dense, spl(r_dense)
        except Exception:
            return r_f, v_f

    def _smooth_log(r, v, s_factor=None):
        """Smooth a profile in log(r) space for log-x plots."""
        finite = np.isfinite(v) & np.isfinite(r) & (r > 0)
        if finite.sum() < 4:
            return r[finite], v[finite]
        r_f, v_f = r[finite], v[finite]
        sort = np.argsort(r_f)
        r_f, v_f = r_f[sort], v_f[sort]
        if s_factor is None:
            s_factor = len(r_f) * 2.0
        try:
            spl = UnivariateSpline(np.log10(r_f), v_f, s=s_factor, k=3)
            lr_dense = np.linspace(np.log10(r_f.min()), np.log10(r_f.max()),
                                   max(len(r_f) * 3, 100))
            return 10**lr_dense, spl(lr_dense)
        except Exception:
            return r_f, v_f

    def _make_kin_fig(eds, lines, r_key, title, outname, xscale='linear',
                      xlim=None, row0_ymax=None):
        """Build one 2×3 kinematic profile figure with shared axes + no whitespace."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 9), squeeze=False,
                                 sharex='col', sharey='row',
                                 gridspec_kw={'hspace': 0, 'wspace': 0})
        fig.patch.set_facecolor('w')
        _lh = [Line2D([0], [0], color=c, lw=1.8, label=lbl) for _, c, lbl in lines]

        _x_lo = xlim[0] if xlim is not None else -np.inf
        _smoother = _smooth_log if xscale == 'log' else _smooth

        for i, ed in enumerate(eds):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            ax.set_facecolor('w')
            r = ed.get(r_key)
            if r is None or len(r) == 0:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
                continue

            _x_mask = r >= _x_lo
            for prof_key, color, lbl in lines:
                v = ed.get(prof_key)
                if v is not None and len(v) == len(r):
                    finite = np.isfinite(v) & _x_mask
                    if finite.any():
                        ax.plot(r[finite], v[finite], color=color, lw=0.6, alpha=0.35)
                        rs, vs = _smoother(r[finite], v[finite])
                        ax.plot(rs, vs, color=color, lw=1.5)

            ax.axhline(0, color='k', lw=0.5, ls=':')
            ax.axhline(5, color='grey', lw=1.2, ls=':', alpha=0.5)
            ax.yaxis.set_major_locator(MultipleLocator(3))
            ax.yaxis.set_minor_locator(MultipleLocator(1))
            if xscale == 'log':
                ax.set_xscale('log')
            if xlim is not None:
                ax.set_xlim(*xlim)
            ax.tick_params(**_tick_kw, labelsize=_kin_fs['xtick.labelsize'])
            ax.tick_params(which='major', width=_tick_major_w)
            ax.tick_params(which='minor', width=_tick_minor_w)
            for sp in ax.spines.values(): sp.set_edgecolor('k')

            ax.text(0.03, 0.97, ed['time_label'], transform=ax.transAxes,
                    va='top', fontsize=_kin_fs['legend.fontsize'],
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

            if row == 1:
                ax.set_xlabel('r [AU]', fontsize=_kin_fs['axes.labelsize'])
            else:
                ax.xaxis.set_tick_params(labelbottom=False)
            if col == 0:
                ax.set_ylabel('velocity [km/s]', fontsize=_kin_fs['axes.labelsize'])
            else:
                ax.yaxis.set_tick_params(labelleft=False)

            if row == 0 and col == 2:
                ax.legend(handles=_lh, loc='upper right',
                          fontsize=_kin_fs['legend.fontsize'])

        for i in range(len(eds), 6):
            axes[i // 3, i % 3].set_visible(False)

        for _row in range(2):
            _yvals = []
            for _col in range(3):
                for _ln in axes[_row, _col].get_lines():
                    try:
                        _yd = np.asarray(_ln.get_ydata(), dtype=float).ravel()
                        _yd = _yd[np.isfinite(_yd)]
                        if len(_yd):
                            _yvals.extend(_yd.tolist())
                    except Exception:
                        pass
            if _yvals:
                _yhi = max(_yvals)
                _ylo = min(min(_yvals), 0.0)
                _yhi_tick = np.ceil(max(_yhi, 3) / 3) * 3
                if _row == 0 and row0_ymax is not None:
                    _yhi_tick = row0_ymax
                axes[_row, 0].set_ylim(float(_ylo) - 0.5, float(_yhi_tick) + 0.3)

        _save_fig_dual(fig, os.path.join(outdir, 'light', outname))

    _DISK_LINES = [
        ('neg_vr_prof',  'tab:blue',   r'$-v_r$'),
        ('vphi_prof',    'tab:orange', r'$v_\phi$'),
        ('cs_prof',      'tab:green',  r'$c_s$'),
        ('vturb_prof',   'tab:red',    r'$|\delta v|$'),
        ('sigma_r_prof', 'purple',     r'$\sigma_r$'),
        ('vK_prof',      'black',      r'$v_K$'),
    ]
    _WIDE_LINES = [
        ('neg_vr_wide_prof',  'tab:blue',   r'$-v_r$'),
        ('vphi_wide_prof',    'tab:orange', r'$v_\phi$'),
        ('cs_wide_prof',      'tab:green',  r'$c_s$'),
        ('vturb_wide_prof',   'tab:red',    r'$|\delta v|$'),
        ('sigma_r_wide_prof', 'purple',     r'$\sigma_r$'),
        ('vK_wide_prof',      'black',      r'$v_K$'),
    ]

    _make_kin_fig(eds, _DISK_LINES, 'bin_AU',
                  'Kinematic radial profiles — disk region',
                  'profile_kinematics_disk.png',
                  xlim=(0, None))
    _make_kin_fig(eds, _WIDE_LINES, 'mach_wide_bin_AU',
                  'Kinematic radial profiles — disk + environment (wide)',
                  'profile_kinematics_wide.png',
                  xscale='log', xlim=(100, 500 * 206265.0))

    # ── Combined symlog figure (3 rows × 2 cols, landscape 4:3 subplots) ──
    _COMBINED_DISK = [
        ('neg_vr_prof', 'tab:blue', r'$-v_r$'),
        ('vphi_prof',   'tab:orange', r'$v_\phi$'),
        ('cs_prof',     'tab:green', r'$c_s$'),
        ('vturb_prof',  'tab:red', r'$|\delta v|$'),
        ('sigma_r_prof','purple', r'$\sigma_r$'),
        ('vK_prof',     'black', r'$v_K$'),
    ]
    _COMBINED_WIDE = [
        ('neg_vr_wide_prof', 'tab:blue', None),
        ('vphi_wide_prof',   'tab:orange', None),
        ('cs_wide_prof',     'tab:green', None),
        ('vturb_wide_prof',  'tab:red', None),
        ('sigma_r_wide_prof','purple', None),
        ('vK_wide_prof',     'black', None),
    ]

    _lh_comb = [Line2D([0], [0], color=c, lw=1.8, label=lbl)
                for _, c, lbl in _COMBINED_DISK]
    _linthresh = 3000.0
    fig_c, axes_c = plt.subplots(3, 2, figsize=(24, 27), squeeze=False,
                                  sharey='all',
                                  gridspec_kw={'hspace': 0.08, 'wspace': 0.05})
    fig_c.patch.set_facecolor('w')

    for i, ed in enumerate(eds):
        row, col = i // 2, i % 2
        ax = axes_c[row, col]
        ax.set_facecolor('w')

        r_disk = ed.get('bin_AU')
        r_wide = ed.get('mach_wide_bin_AU')

        for (dk, dc, _), (wk, wc, _) in zip(_COMBINED_DISK, _COMBINED_WIDE):
            vd = ed.get(dk)
            vw = ed.get(wk)
            if r_disk is not None and vd is not None and len(vd) == len(r_disk):
                fin = np.isfinite(vd)
                if fin.any():
                    ax.plot(r_disk[fin], vd[fin], color=dc, lw=0.6, alpha=0.35)
                    rs, vs = _smooth(r_disk[fin], vd[fin])
                    ax.plot(rs, vs, color=dc, lw=1.5)
            if r_wide is not None and vw is not None and len(vw) == len(r_wide):
                in_range = r_wide >= _linthresh
                fin = np.isfinite(vw) & in_range
                if fin.any():
                    ax.plot(r_wide[fin], vw[fin], color=wc, lw=0.6, alpha=0.35)
                    rs, vs = _smooth_log(r_wide[fin], vw[fin])
                    ax.plot(rs, vs, color=wc, lw=1.5)

        ax.set_xscale('symlog', linthresh=_linthresh)
        ax.set_xlim(0, 500 * 206265.0)
        ax.axhline(0, color='k', lw=0.5, ls=':')
        ax.axhline(5, color='grey', lw=1.2, ls=':', alpha=0.5)
        ax.yaxis.set_major_locator(MultipleLocator(3))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.tick_params(**_tick_kw, labelsize=_kin_fs['xtick.labelsize'])
        ax.tick_params(which='major', width=_tick_major_w)
        ax.tick_params(which='minor', width=_tick_minor_w)
        for sp in ax.spines.values(): sp.set_edgecolor('k')

        ax.text(0.03, 0.97, ed['time_label'], transform=ax.transAxes,
                va='top', fontsize=_kin_fs['legend.fontsize'],
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

        if row == 2:
            ax.set_xlabel('r [AU]', fontsize=_kin_fs['axes.labelsize'])
        else:
            ax.xaxis.set_tick_params(labelbottom=False)
        if col == 0:
            ax.set_ylabel('velocity [km/s]', fontsize=_kin_fs['axes.labelsize'])

        if row == 0 and col == 1:
            ax.legend(handles=_lh_comb, loc='upper right',
                      fontsize=_kin_fs['legend.fontsize'])

    for i in range(len(eds), 6):
        axes_c[i // 2, i % 2].set_visible(False)

    _all_y = []
    for _r in range(3):
        for _c in range(2):
            for _ln in axes_c[_r, _c].get_lines():
                try:
                    _yd = np.asarray(_ln.get_ydata(), dtype=float).ravel()
                    _yd = _yd[np.isfinite(_yd)]
                    if len(_yd):
                        _all_y.extend(_yd.tolist())
                except Exception:
                    pass
    if _all_y:
        _yhi = max(_all_y)
        _ylo = min(min(_all_y), 0.0)
        _yhi_tick = np.ceil(max(_yhi, 3) / 3) * 3
        axes_c[0, 0].set_ylim(float(_ylo) - 0.5, float(_yhi_tick) + 0.3)

    _save_fig_dual(fig_c, os.path.join(outdir, 'light', 'profile_kinematics_combined.png'))


# ═════════════════════════════════════════════════════════════════════════════
# Master orchestrator
# ═════════════════════════════════════════════════════════════════════════════

def make_all_figures(epoch_data_list, outdir, frames_dir=None, alt_epoch_data_list=None):
    """Generate all multi-epoch paper figures from pre-extracted epoch data."""

    os.makedirs(os.path.join(outdir, 'light'), exist_ok=True)

    # ── Determine shared colorbars ──
    # Surface density
    all_sig = np.concatenate([ed['sig_fo'][ed['sig_fo'] > 0] for ed in epoch_data_list
                              if np.any(ed['sig_fo'] > 0)])
    sd_norm = colors.LogNorm(vmin=max(np.percentile(all_sig, 1), 1e3),
                             vmax=np.percentile(all_sig, 99.99) * 2.0)

    # B-field range (signed)
    def _bfield_range(key):
        vals = []
        for ed in epoch_data_list:
            d = ed.get(key)
            if d is not None:
                vals.append(d[np.isfinite(d)])
        if not vals:
            return None
        all_v = np.concatenate(vals)
        if len(all_v) == 0:
            return None
        vmax = np.percentile(np.abs(all_v), 99)
        linthresh = max(np.percentile(np.abs(all_v[all_v != 0]), 10), 1e-10) if np.any(all_v != 0) else 1e-5
        return colors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)

    print("  Generating combined density grid...")
    plot_grid_combined(epoch_data_list, 'sig_fo', 'sig_eo', outdir, _ECLIPSE_CMAP, sd_norm,
                       r'$\Sigma$ [M$_\odot$/pc$^2$]', 'combined_density.png')

    # Velocity dispersion maps
    all_vd = np.concatenate([ed['vdisp_fo'][ed['vdisp_fo'] > 0] for ed in epoch_data_list
                             if np.any(ed['vdisp_fo'] > 0)])
    vd_norm = colors.LogNorm(vmin=max(np.percentile(all_vd, 5), 0.1),
                             vmax=np.percentile(all_vd, 99.5)) if len(all_vd) > 0 else colors.LogNorm(0.1, 100)

    print("  Generating combined velocity dispersion grid...")
    plot_grid_combined(epoch_data_list, 'vdisp_fo', 'vdisp_eo', outdir, 'viridis', vd_norm,
                       r'$|\delta v|$ [km/s]', 'combined_vdisp.png')

    print("  Generating face-on Toomre Q grid (thermal+turbulent)...")
    Q_norm = colors.LogNorm(vmin=0.1, vmax=10)
    plot_grid_faceon(epoch_data_list, 'Q_fo_combined', outdir, 'RdYlGn', Q_norm,
                     r'Toomre $Q$ (combined)', 'faceon_toomre_Q.png', contour_level=1.0)

    # B-field grids (combined faceon+edgeon per component)
    for bcomp, blabel in [('Bz', r'$B_z$'), ('Br', r'$B_r$'),
                          ('Btor', r'$B_\phi$'), ('Bpol', r'$B_{\rm pol}$')]:
        fo_key = f'{bcomp}_fo'
        eo_key = f'{bcomp}_eo'
        b_norm_fo = _bfield_range(fo_key)
        b_norm_eo = _bfield_range(eo_key)
        b_norm = b_norm_fo or b_norm_eo
        if b_norm is None:
            print(f"  Skipping {bcomp} (no data)")
            continue
        print(f"  Generating combined {bcomp} grid...")
        plot_grid_combined(epoch_data_list, fo_key, eo_key, outdir, 'RdBu_r', b_norm,
                           f'{blabel} [G]', f'combined_{bcomp}.png')

    # |B| magnitude (unsigned)
    all_bmag_vals = np.concatenate([
        ed['Bmag_fo'][ed['Bmag_fo'] > 0] for ed in epoch_data_list
        if ed.get('Bmag_fo') is not None and np.any(ed['Bmag_fo'] > 0)])
    if len(all_bmag_vals) > 0:
        bmag_norm = colors.LogNorm(vmin=max(np.percentile(all_bmag_vals, 5), 1e-10),
                                   vmax=np.percentile(all_bmag_vals, 99.5))
        print("  Generating combined |B| magnitude grid...")
        plot_grid_combined(epoch_data_list, 'Bmag_fo', 'Bmag_eo', outdir,
                           'plasma', bmag_norm,
                           r'$|B|$ [G]', 'combined_Bmag.png')

    # Mass-to-flux ratio μ
    all_mu_vals_list = [
        ed['mu_fo'][np.isfinite(ed['mu_fo']) & (ed['mu_fo'] > 0)]
        for ed in epoch_data_list
        if ed.get('mu_fo') is not None and np.any(np.isfinite(ed['mu_fo']) & (ed['mu_fo'] > 0))]
    if all_mu_vals_list:
        all_mu_vals = np.concatenate(all_mu_vals_list)
        mu_norm = colors.LogNorm(vmin=max(np.percentile(all_mu_vals, 1), 0.01),
                                 vmax=min(np.percentile(all_mu_vals, 99), 1e4))
        print("  Generating combined μ (mass-to-flux) grid...")
        plot_grid_combined(epoch_data_list, 'mu_fo', 'mu_eo', outdir,
                           'RdYlBu_r', mu_norm,
                           r'$\mu_\Phi$', 'combined_mu.png', contour_level=1.0)

    # 1D profiles
    print("  Generating velocity profiles...")
    plot_velocity_profiles(epoch_data_list, outdir)

    print("  Generating density profile overlay...")
    # Only first 3 epochs (t - t1 < 1.3 kyr) for density profile
    _t1_dens = next((ed['t1_Myr'] for ed in epoch_data_list if ed.get('t1_Myr') is not None), None)
    _dens_epochs = [ed for ed in epoch_data_list
                    if _t1_dens is None or (ed['time_Myr'] - _t1_dens) * 1e3 < 1.3]
    plot_profile_overlay(_dens_epochs, 'rho_prof', 'bin_ctr_rho_AU', outdir,
                         r'$\rho$ [g/cm$^3$]', 'profile_density.png',
                         log_x=True, log_y=True, power_law_fit=True)

    print("  Generating resolution profile...")
    plot_resolution_profile(epoch_data_list, outdir)

    print("  Generating Toomre Q profile overlay (combined thermal+turbulent)...")
    plot_profile_overlay(epoch_data_list, 'Q_prof_combined', 'bin_AU', outdir,
                         r'Toomre $Q$', 'profile_toomre_Q.png',
                         log_x=False, log_y=True, ref_line=1.0, ref_label='Q=1')

    print("  Generating mass-to-flux ratio overlay...")
    plot_profile_overlay(epoch_data_list, 'mf_prof', 'bin_AU', outdir,
                         r'$\mu_\Phi$', 'profile_mass_to_flux.png',
                         log_x=False, log_y=True, ref_line=1.0, ref_label=r'$\mu=1$')

    # Mach number profile
    print("  Generating Mach number profile...")
    plot_mach_profile(epoch_data_list, outdir)

    print("  Generating shell mass + accretion profiles...")
    plot_shell_mass_accretion(epoch_data_list, outdir, frames_dir=frames_dir)

    # alt epoch list (used for phase diagrams and B-field phase)
    _alt_eds = alt_epoch_data_list if alt_epoch_data_list is not None else epoch_data_list

    # Phase diagrams (combined 6-panel: first 3 epochs, T + H2)
    print("  Generating combined phase diagram...")
    # Load KG2023 reference data if available
    _kg_data = None
    _kg_path = os.path.join(outdir, 'kg2023_phase_data.npz')
    if not os.path.exists(_kg_path):
        _kg_path = os.path.join(outdir, '..', 'paper_plots', 'kg2023_phase_data.npz')
    if os.path.exists(_kg_path):
        try:
            _kgd = np.load(_kg_path)
            _kg_data = {k: _kgd[k] for k in _kgd.files}
            print(f"    Loaded KG2023 reference data from {_kg_path}")
        except Exception as _e:
            print(f"    WARNING: could not load KG2023 data: {_e}")
    plot_phase_diagrams(_alt_eds, outdir,
                        frames_dir=frames_dir, kg_data=_kg_data)

    # Kinematic radial profiles (disk + wide)
    print("  Generating kinematic radial profiles...")
    plot_kinematic_radial_profiles(epoch_data_list, outdir)

    # B-field phase diagram (6-panel: all 6 epochs, |B| vs n)
    print("  Generating B-field phase diagram...")
    plot_bfield_phase(_alt_eds, outdir)

    # Infall timescale: M_disk / Mdot
    print("  Generating infall timescale plot...")
    plot_infall_timescale(epoch_data_list, outdir, frames_dir=frames_dir)

    print("  All multi-epoch figures complete.")
