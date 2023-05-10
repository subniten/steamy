# %%
import pathlib

import matplotlib.pyplot as pyplot
import numpy
import xarray


# %%
data_root_directory = pathlib.Path('/Volumes/Rayleigh/cruise-oc4920/steamy_data')
bathymetry_file = (
    data_root_directory / 'bathymetry' / 'gebco_2022_n60.0_s54.0_w7.5_e15.0.nc'
)


# %%
bathymetry = xarray.open_dataarray(bathymetry_file)
bathymetry.name = 'bathymetry'


# %%
max_depth = 500
min_depth = 0
delta_depths = 50

number_of_isobaths = (max_depth - min_depth) // delta_depths + 1

isobaths = numpy.linspace(-max_depth, -min_depth, number_of_isobaths)

# %%
fig, ax = pyplot.subplots(1, 1, figsize=(12, 8))
bathymetry.plot.contour(ax=ax, levels=isobaths, linewidths=0.7, colors='tab:grey')
ax.set_xlabel('˚E')
ax.set_ylabel('˚N')


# %%
# Cut it down to smaller area
# Zoom in in the plot to the desired area, then get the axes limits
# ax.get_xlim()
# ax.get_ylim()
lon_indices = (bathymetry.lon >= 10.0) & (bathymetry.lon <= 13.0)
lat_indices = (bathymetry.lat >= 57.0) & (bathymetry.lat <= 59.0)
iselector = dict(
    lon=numpy.arange(len(bathymetry.lon))[lon_indices],
    lat=numpy.arange(len(bathymetry.lat))[lat_indices],
)
# .isel == index selector, like the indices in a numpy.array
bathymetry = bathymetry.isel(iselector)

fig, ax = pyplot.subplots(1, 1, figsize=(12, 8))
bathymetry.plot.contour(ax=ax, levels=isobaths, linewidths=0.7, colors='tab:grey')
ax.set_xlabel('˚E')
ax.set_ylabel('˚N')

bathymetry.to_netcdf(data_root_directory / 'bathymetry' / 'steamy_bathymetry.nc')
