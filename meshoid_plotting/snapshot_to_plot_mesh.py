from meshoid import Meshoid
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
sys.path.append('/cita/h/home-2/vpustovoit/.local/lib/python3.10/site-packages')
from meshoid import Meshoid
from ..utils import *

group_name=''
# Read system arguments
if len(sys.argv) > 1:
    day_attempt = sys.argv[1]
else:
    day_attempt = '2023.10.20:17/'

snapno = '001'

ParticleType = 'PartType1'
plottype = 'density'

clrmax = 1e1
clrmin = 1e-5

SizeOfShownBox = 100000

# For 2D plots (plottype = temperature, density)
axis_of_projection='y'


# Getting the in/out directories{{{
name_appendix = ParticleType + '/' + axis_of_projection + '_' + plottype + '/'
input_file = 'snapshot_'+snapno+'.hdf5'
output_file = '2Dplot'+snapno+'.png'
# In/Out Directories
input_dir='/fs/lustre/scratch/vpustovoit/STARFORGE/output/' + day_attempt
#out_dir='./densityplots/'
output_dir='/cita/d/www/home/vpustovoit/plots/' + day_attempt + name_appendix

input_file = input_dir + input_file
output_file = output_dir + output_file
#}}}

def plot_for_single_snapshot_mesh(i): #{{{
    snapno=int_to_str(i+units.start,100)
    print('snapno: ', snapno)

    F = h5py.File(input_file,"r")
    rho = F["PartType1"]["Masses"][:]
    #density_cut = (rho*1e-2 > clrmin)
    density_cut = (rho*1e-2 > 0)
    pdata = {}
    #for field in "Masses", "Coordinates", "SmoothingLength", "Velocities":
    for field in "Masses", "Coordinates", "Velocities":
        pdata[field] = F["PartType1"][field][:][density_cut]
    F.close()
    
    #print(pdata["Coordinates"])
    
    pos = pdata["Coordinates"]
    center = np.median(pos,axis=0)
    pos -= center
    radius_cut = np.sum(pos*pos,axis=1) < SizeOfShownBox*SizeOfShownBox
    # Below for PartType0
    #pos, mass, hsml, v = pos[radius_cut], pdata["Masses"][radius_cut], pdata["Softening_KernelRadius"][radius_cut], pdata["Velocities"][radius_cut]
    #M = Meshoid(pos, mass, hsml)
    # Below for PartType1
    pos, mass, v = pos[radius_cut], pdata["Masses"][radius_cut], pdata["Velocities"][radius_cut]
    M = Meshoid(pos, mass, verbose=True)
    
    rmax = SizeOfShownBox
    res = 800
    X = Y = np.linspace(-rmax, rmax, res)
    X, Y = np.meshgrid(X, Y)
    fig, ax = plt.subplots(figsize=(6,6))
    sigma_gas_msun_pc2 = M.SurfaceDensity(M.m,center=np.array([0,0,0]),size=SizeOfShownBox,res=res)*1e4
    p = ax.pcolormesh(X, Y, sigma_gas_msun_pc2, norm=colors.LogNorm(vmin=clrmin*100,vmax=clrmax))
    ax.set_aspect('equal')
    fig.colorbar(p,label=r"$\Sigma_{gas}$ $(\rm M_\odot\,pc^{-2})$")
    ax.set_xlabel("X (kpc)")
    ax.set_ylabel("Y (kpc)")
    #plt.show()
    print(output_file)
    plt.savefig(output_file)
#}}}

def snap_to_plot_mesh(input_dir, output_dir):
    #datax = []
    #datay = [[],[]]
    #snapdata = SnapshotData()
    #snapdata.dataxy = (datax, datay)
    max_time = 6 * 60 * 60 # Define max time (in seconds) that

    num_snapshots=get_number_of_snapshots(input_dir)
    # Eternal plotting mode {{{
    i=0
    time_since_snap=0
    num_snapshots=get_number_of_snapshots(input_dir)
    if i < num_snapshots - units.start:
        time.sleep(5)
        plot_for_single_snapshot_mesh(i)
        i+=1
        time_since_snap=0
    else:
        exit()
    #    print_time_since_last_snapshot(time_since_snap, max_time)
    #    time_since_snap+=5
    #    time.sleep(5)
    #}}}
snap_to_plot_mesh(input_dir, output_dir)
