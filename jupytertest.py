
#%matplotlib inline
#%config InlineBackend.figure_format = 'png'
from IPython.display import display
import sys, os, importlib
scratch_analysis_path="/scratch/vasissua/SHIVAN/analysis/"
sys.path.insert(0,scratch_analysis_path)
import meshoid_plotting.starforge_plot as sfp
import matplotlib.pyplot as plt


importlib.reload(sfp)
import numpy as np



run_name = "2026-01-20_m12i_snaps1"
run_name = "2026-01-30_m12f_shivan"
#run_name = "2026-01-20_m12i"
snapstr = "005"

#run_name = "2026-01-30_m12f_shivan"
#snapstr = "005"

run_name = "m12f"  # Adjust as needed
snapstr = "225"  # Adjust snapshot number

scratch_path = "/scratch/vasissua/"
#run_out_path = os.path.join(scratch_path, "SHIVAN2/output")
run_out_path = os.path.join(scratch_path, "COPY/2026-03/m12f/output_jeans_refinement")
snap_hdf5 = "snapshot_" + snapstr + ".hdf5"

run_path = os.path.join(run_out_path, run_name, snap_hdf5) # SHIVAN2 PATH
run_path = os.path.join(run_out_path, snap_hdf5) # COPY PATH

center_on_stars = True
data_dict  = sfp.setup_meshoid(run_path, center_type="potential", recenter=False,
                               rotate_type="L")
print(f"Data setup complete!")

code_mass = 2e43 # 10^10 Msun in grams
code_length = 3.0857e+21 # kpc in cm
m_proton = 1.673e-24 # g


pdata = data_dict["pdata"]
pdata_dm = data_dict["pdata_dm"]
a = pdata["Time"]
density = pdata["Density"]

density *= code_mass / code_length**3 / a**3
number_density = density / (0.76 * m_proton)


#coords_dm = pdata_dm["Coordinates"]
#print(coords_dm)
#
#
#coords_dm_t = np.transpose(coords_dm)
#r = np.sqrt(coords_dm_t[0]**2 + coords_dm_t[1]**2 + coords_dm_t[2]**2)
#print(r)


from matplotlib.ticker import LogLocator, NullFormatter
from vasthemer import set_theme
set_theme("stylix_transparent")
#fig, ax = plt.subplots(figsize=(7,7))
#ax.plot(r, pdata_dm["Masses"])
#fig.set_layout_engine("compressed")
#display(fig)

#print(data_dict["pdata"]["MolecularMassFraction"])



"""
fig, (ax1, ax2) = plt.subplots(figsize=(7,7), nrows=2, sharex=True)
ax1.plot(number_density, pdata["MolecularMassFraction"], ".", markersize=0.5)
ax1.set_xscale("log")
ax1.set_yscale("log")
#ax1.set_xlabel("log Density, $cm^{{-3}}$")
ax1.set_ylabel("H$_2$ fraction")
ax1.set_xlim(1e-5,1e20)
#ax1.set_xticks(np.arange(0, 2*np.pi, 0.5), minor=True)
#fig.savefig(scratch_analysis_path + "rho_H2.png")
#plt.close()



ax2.plot(number_density, pdata["Temperature"], ".", markersize=0.5)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("log Density, $cm^{{-3}}$")
ax2.set_ylabel("log Temperature, $K$")
ax2.set_xlim(1e-5,1e20)
ax2.xaxis.set_major_locator(LogLocator(base=10, numticks=6))  # every 5 decades
ax2.xaxis.set_minor_locator(LogLocator(base=10, subs='all', numticks=50))
ax2.xaxis.set_minor_formatter(NullFormatter()) # No labels for minor ticks

fig.set_layout_engine("compressed")

display(fig)


"""

print(pdata["Coordinates"])

plt.style.use('dark_background')

def plot_single_zoom(data_dict, resolution, boxsize, pc_scale_power, au_scale_power, ax, plot_zoombox):
    plot_fire_stars = False
    kwargs2 = {'box_size': boxsize, 'output_dir': './', 'resolution': resolution,
               "pc_scale": pc_scale_power, "au_scale": au_scale_power, 
               "plot_fire_stars": plot_fire_stars,
               "plot_zoombox": plot_zoombox} #, 'vmin': 1e-4, 'vmax': 1e-3}
    sfp.plot_single_snapshot(data_dict, ax, **kwargs2)
    ax.plot(data_dict["center"][0], data_dict["center"][1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def plot_zooms(data_dict, resolution=1000, xplots = 2, yplots = 2, init_boxsize = 1e-2,
               init_pcscale = 5, init_auscale = 10):
    print(f"Started plotting zoom-ins...")
    fig, axs = plt.subplots(yplots, xplots, figsize=(xplots*10, yplots*10))
    boxsize_zooms = xplots * yplots
    #print(np.shape(axs))
    plot_zoombox = 1
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
        plot_single_zoom(data_dict, resolution, new_boxsize, new_pcscale, new_auscale, ax, 
                         plot_zoombox=plot_zoombox_l)
    plt.subplots_adjust(wspace=0, hspace=0)
    return fig

#fig = sfp.plot_h2_rate_map(data_dict, rate_key="total_formation", **kwargs2)
fig = plot_zooms(data_dict)
display(fig)

outname = "8x_zoom_shivan_m12f.png"
out_save_path = os.path.join(scratch_path, "SHIVAN", "analysis", outname)
fig.savefig(out_save_path)



