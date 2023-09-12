import yt
import unyt

# 1. Load the Data
filename="../output/2023.09.11:2/snapshot_075.hdf5"
ds = yt.load(filename)
dd = ds.all_data()
dense_ad = dd.cut_region(['obj["gas", "density"] > 5e-30'])

# 2. Define the Region 
center_of_mass = dd.quantities["CenterOfMass"]()
center_of_mass_kpc = center_of_mass.to(unyt.kpc)
print(f"Ceneter of Mass: {center_of_mass}")    
plot_center = ds.arr(center_of_mass_kpc, "code_length")

sp = ds.sphere(plot_center, (10, "kpc"))

# 3. Extract Mass Data
mass_data = sp[("gas","mass")]

# 4. Sum the Mass
total_mass = mass_data.sum()
total_mass = total_mass.to(unyt.Msun)

print(f"Total mass inside the sphere: {total_mass}")

# Density Plot 
"""
prj = yt.ProjectionPlot(
    ds, "z", ("gas", "density"), center=plot_center, width=(30, "kpc"), data_source=sp
)
clrmin, clrmax = (1e-9, 1e-5)
prj.set_zlim(("gas", "density"), zmin=(clrmin, "g/cm**2"), zmax=(clrmax, "g/cm**2"))
savedir='/cita/d/www/home/vpustovoit/plots/object_only.png' 
prj.save(savedir)
"""

# Temperature plot
prj = yt.ProjectionPlot(ds, "z", ("gas", "temperature"), center=plot_center, width=(100, "kpc"), weight_field=("gas", "density"))
#clrmin, clrmax = (9e25, 1e26)
#prj.set_zlim(("gas", "temperature"), zmin=(clrmin, "K*cm"), zmax=(clrmax, "K*cm"))
savedir='/cita/d/www/home/vpustovoit/plots/object_only_t.png' 
prj.save(savedir)
