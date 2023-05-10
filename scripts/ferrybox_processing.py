# %%
import datetime
import math

import cartopy.crs as ccrs
import cmocean.cm as cmap
from geopy.distance import geodesic
import matplotlib.dates as mdates
import matplotlib.pyplot as pyplot
import numpy
import xarray

# For this to work, make sure steamy_utils directory is in your PYTHONPATH
#  i) notebook userds
#      start your notebook in a directory containing steamy_utilities directory by
#      PYTHONPATH=`pwd` jupyterlab-lab
# ii) python/ipython users
#      PYTHONPATH=`pwd` ipython --pylab 
import steamy_utilities


# %%
_figsize = (14, 7)

steamy_utilities.set_steamy_data_root_path(
    '/Volumes/Rayleigh/cruise-oc4920/steamy_data'
)

bathymetry_file = steamy_utilities.get_bathymetry_file()
ctd_directory = steamy_utilities.get_ctd_directory()
ferrybox_directory = steamy_utilities.get_ferrybox_directory()


# %%
tsg_data = steamy_utilities.read_ferrybox_directory(ferrybox_directory)
bathy = steamy_utilities.load_bathymetry(bathymetry_file)

ctd_files = [s for s in sorted(ctd_directory.glob('*.cnv')) if 'yoyo' not in s.name]
station_ids = numpy.array([s.stem.split('_')[-1] for s in ctd_files], dtype=str)

ctd_casts = steamy_utilities.read_ctd_files(
    ctd_files, station_identifiers=station_ids
)  # to get station positions


# %%
station_02_time = datetime.datetime(2023, 5, 3, 9, 29)
timestamp_station_02 = numpy.datetime64(station_02_time)

selector = dict(index=tsg_data.index[tsg_data.time > timestamp_station_02].values)
_data = tsg_data.sel(selector)
distances = steamy_utilities.cumulative_distance(_data)
_data = _data.assign_coords(
    dict(
        distance=xarray.DataArray(
            distances,
            coords=dict(index=_data.index.values),
            dims=['index'],
            attrs=dict(units='km'),
        )
    )
)


# %%
ctd_times_minutes = ctd_casts.time.values.astype('datetime64[m]')
ctd_station_names = [f'{nr:03}' for nr in ctd_casts.cast.values]

indices = [
    (_data.index[_ctd_time == _data.time].values, sn)
    for _ctd_time, sn in zip(ctd_times_minutes, ctd_station_names)
]
indices = [(idx[0], sn) for idx, sn in indices if len(idx) > 0]
ctd_station_selector = dict(index=[idx[0] for idx in indices])
station_data = _data.sel(ctd_station_selector)
station_data['station_name'] = xarray.DataArray(
    [sn for idx, sn in indices], coords=dict(index=station_data.index), dims=['index']
)


# %%
fig_time, axes_time = steamy_utilities.plot_tsg_with_respect_to_x_variable(
    tsg_data, x_variable='time'
)


# %%
fig_distance, axes_distance = steamy_utilities.plot_tsg_with_respect_to_x_variable(
    _data, x_variable='distance'
)
ax_chlorophyll = axes_distance['ax_chlorophyll']
ax_oxygen = axes_distance['ax_oxygen']
ax_oxygen.set_ylim((9.7, 10.7))
ax_chlorophyll.set_ylim((-0.2, 0.8))

# ax_oxygen.set_xlim((0, 70))


ax_temperature = axes_distance['ax_temp']
for distance, sn in zip(station_data.distance.values, station_data.station_name.values):
    ax_temperature.annotate(
        sn.split('_')[-1], xy=(distance, 1.05), xycoords=('data', 'axes fraction')
    )
    ax_temperature.axvline(distance, color='tab:grey')
ax_temperature_time = axes_time['ax_temp']
for time, sn in zip(station_data.time.values, station_data.station_name.values):
    ax_temperature_time.annotate(
        sn.split('_')[-1], xy=(time, 1.05), xycoords=('data', 'axes fraction')
    )
    ax_temperature_time.axvline(time, color='tab:grey')


# %%
fig = pyplot.figure(figsize=(16, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=11.0))
img = tsg_data.plot.scatter(
    x='longitude',
    y='latitude',
    c=tsg_data.Salinity_SBE45.where(tsg_data.Salinity_SBE45 > 29.4),
    cmap=cmap.haline,
    ax=ax,
    s=10,
    ec=None,
    vmin=29.5,
    vmax=35.,
)
bathy.plot.contour(
    x='lon',
    y='lat',
    ax=ax,
    levels=numpy.linspace(-500.0, 0, 21),
    linewidths=0.7,
    colors='tab:grey',
)
pyplot.colorbar(img)
ax.set_xlim((10, 11.94780225))
ax.set_ylim((57.3, 58.5))

station_data.plot.scatter(x='longitude', y='latitude', s=4, marker='+', c='tab:red', alpha=0.6)
