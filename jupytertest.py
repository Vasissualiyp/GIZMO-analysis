
%matplotlib inline
%config InlineBackend.figure_format = 'png'
from IPython.display import display
import sys, os, importlib
scratch_analysis_path="/scratch/vasissua/SHIVAN/analysis/"
sys.path.insert(0,scratch_analysis_path)
import meshoid_plotting.starforge_plot as sfp
import matplotlib.pyplot as plt
importlib.reload(sfp)



run_name = "2026-01-20_m12i_snaps1"
#run_name = "2026-01-20_m12i"
snapstr = "060"

#run_name = "2026-01-30_m12f_shivan"
#snapstr = "005"

scratch_path = "/scratch/vasissua/"
run_out_path = os.path.join(scratch_path, "SHIVAN2/output")
snap_hdf5 = "snapshot_" + snapstr + ".hdf5"

run_path = os.path.join(run_out_path, run_name, snap_hdf5)

kwargs1 = {'center_on_stars': False}
data_dict  = sfp.setup_meshoid(run_path, **kwargs1)
print(f"Data setup complete!")

code_mass = 2e43 # 10^10 Msun in grams
code_length = 3.0857e+21 # kpc in cm
m_proton = 1.673e-24 # g


pdata = data_dict["pdata"]
a = pdata["Time"]
density = pdata["Density"]

density *= code_mass / code_length**3 / a**3
number_density = density / (0.76 * m_proton)

fig, (ax1, ax2) = plt.subplots(figsize=(13,7), ncols=2)
ax1.plot(number_density, pdata["MolecularMassFraction"], ".", markersize=0.5)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("log Density, $cm^{{-3}}$")
ax1.set_ylabel("H$_2$ fraction")
ax1.set_xlim(1e-5,1e20)
#fig.savefig(scratch_analysis_path + "rho_H2.png")
#plt.close()



ax2.plot(number_density, pdata["Temperature"], ".", markersize=0.5)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("log Density, $cm^{{-3}}$")
ax2.set_ylabel("log Temperature, $K$")
ax2.set_xlim(1e-5,1e20)
display(fig)
fig.savefig(scratch_analysis_path + "rho_T_H2.png")
plt.close()




boxsize = 1e-7
pc_scale_power = -3
au_scale_power = 3
plot_fire_stars = False
kwargs2 = {'box_size': boxsize, 'output_dir': './', 'resolution': 500,
           "pc_scale": pc_scale_power, "au_scale": au_scale_power, 
           "plot_fire_stars": plot_fire_stars} #, 'vmin': 1e-4, 'vmax': 1e-3}
#fig = sfp.plot_single_snapshot(data_dict, **kwargs2)
print(f"Started plotting...")
#fig = sfp.plot_h2_rate_map(data_dict, rate_key="total_formation", **kwargs2)
display(fig)



kwargs2["plot_fire_stars"] = False
fig = sfp.plot_single_snapshot(data_dict, **kwargs2)
display(fig)





import numpy as np

x = np.logspace(1e-2, 1e2, 1000)
def f(T):
    return ((1.555 + 0.1272 * T**0.77) * np.exp(-128 / T) + (2.406 + 0.1232 * T**0.92) * np.exp(-255/T) ) * np.exp(- T**2 / 10**6 / 25)

y = f(x)
plt.plot(x,y)
plt.savefig(scratch_path + "testplot.png")

