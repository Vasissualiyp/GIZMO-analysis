import yt
ds = yt.load("../starform_23-03-03/snapshot2/snapshot_142.hdf5")
for i in sorted(ds.field_list):
  print(i)
