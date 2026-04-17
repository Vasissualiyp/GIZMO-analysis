# 3RD version of script to plot everything. Just gave up and using Shivan's method at this point



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

# Set up GUAC
guac_src_path = "/home/vasissua/PYTHON/GUAC/src/"
pfp_src_path = "/home/vasissua/PYTHON/pfh_python/gizmopy/"
sys.path.insert(0, guac_src_path)
sys.path.insert(0, pfp_src_path)
import hybrid_sims_utils.read_snap as _rsnap
importlib.reload(_rsnap)
import notebooks.make_disk_movie_frames as ntbk
#from jupytertest import plot_zooms


importlib.reload(sfp)
importlib.reload(utilf)
importlib.reload(ntbk)

from vasthemer import set_theme
#set_theme("stylix_transparent")
plt.style.use('dark_background')

class Defaults():
    def __init__(self):
        #self.path = '/scratch/vasissua/COPY/2026-03/m12f/'
        self.path = '/scratch/vasissua/COPY/2026-03/m12f_cutout/'
        self.sim = 'output_jeans_refinement'
        #self.sim = 'output_cutout'
        self.outdir = '/scratch/vasissua/SHIVAN/analysis/frames/'
        self.snap_start = None
        self.snap_end = None
        self.res = 400
        self.image_box = 2e-5
        self.r_search = 1e-5
        self.r_max = 1e-5
        self.rho_thresh = 1e-15
        self.aspect = 0.3
        self.f_kep = 0.3
        self.vmin = 1e5
        self.vmax = 1e8
        self.ncores = 1
        self.cmap = 'inferno'
        # Set to the approximate center of the jeans-refinement region (physical kpc).
        # When no sinks exist, find_center will search for the densest gas within
        # reference_search_radius of this point instead of searching globally.
        # Set to None to fall back to the global densest-particle behavior.
        self.reference_center = None          # e.g. np.array([41.75, 44.22, 46.01])
        self.reference_search_radius = 0.1    # kpc
        self.corotate = True                  # pin face-on to disk's rotating frame
        self.vmax_vel = None                  # km/s; set e.g. 5.0 to fix flicker
        self.min_gas_particles = 0        # skip frame if n_gas < this (after snap min_gas_snap)
        #self.min_gas_particles = 171000        # skip frame if n_gas < this (after snap min_gas_snap)
        self.min_gas_snap = 150               # snap number at which the gas-count check activates

main_func = lambda: ntbk.main(Defaults())

main_func()
ntbk.make_Q_heatmap(os.path.join(scratch_analysis_path, "frames"))


