import pathlib

import matplotlib.pyplot as pyplot
import numpy
import xarray


# %%
data_root_directory = pathlib.Path('/Volumes/Rayleigh/cruise-oc4920/steamy_data')
bathymetry_file = data_root_directory / 'bathymetry' / 'gebco_2022_n60.0_s54.0_w7.5_e15.0.nc'


# %%
bathymetry = xarray.open_dataarray(bathymetry_file)
bathymetry.name = 'bathymetry'


# %%
max_depth = 500
min_depth = 0
delta_depths = 50

number_of_isobaths = (max_depth - min_depth)// delta_depths + 1

isobaths = numpy.linspace(-max_depth, -min_depth, number_of_isobaths)

fig, ax = pyplot.subplots(1, 1, figsize=(12, 8))
bathymetry.plot.contour(ax=ax, levels=isobaths, linewidths=0.7, colors='tab:grey')
ax.set_xlabel('˚E')
ax.set_ylabel('˚N')
