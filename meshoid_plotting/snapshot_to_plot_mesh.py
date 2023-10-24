from meshoid import Meshoid
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
sys.path.append('/cita/h/home-2/vpustovoit/.local/lib/python3.10/site-packages')
from meshoid import Meshoid


group_name=''
# Read system arguments
if len(sys.argv) > 1:
    day_attempt = sys.argv[1]
else:
    day_attempt = '2023.10.19:2/'

snapno = '094'

ParticleType = 'gas'
plottype = 'density'

clrmax = 1e1
clrmin = 1e-5

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

F = h5py.File(input_file,"r")
rho = F["PartType0"]["Density"][:]
density_cut = (rho*300 > clrmin)
pdata = {}
for field in "Masses", "Coordinates", "SmoothingLength", "Velocities":
    pdata[field] = F["PartType0"][field][:][density_cut]
F.close()

pos = pdata["Coordinates"]
center = np.median(pos,axis=0)
pos -= center
radius_cut = np.sum(pos*pos,axis=1) < 20*20
pos, mass, hsml, v = pos[radius_cut], pdata["Masses"][radius_cut], pdata["SmoothingLength"][radius_cut], pdata["Velocities"][radius_cut]


M = Meshoid(pos, mass, hsml)
rmax = 20
res = 800
X = Y = np.linspace(-rmax, rmax, res)
X, Y = np.meshgrid(X, Y)
fig, ax = plt.subplots(figsize=(6,6))
sigma_gas_msun_pc2 = M.SurfaceDensity(M.m,center=np.array([0,0,0]),size=20.,res=res)*1e4
p = ax.pcolormesh(X, Y, sigma_gas_msun_pc2, norm=colors.LogNorm(vmin=clrmin*100,vmax=clrmax))
ax.set_aspect('equal')
fig.colorbar(p,label=r"$\Sigma_{gas}$ $(\rm M_\odot\,pc^{-2})$")
ax.set_xlabel("X (kpc)")
ax.set_ylabel("Y (kpc)")
#plt.show()
print(output_file)
plt.savefig(output_file)
