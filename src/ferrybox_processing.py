# %%
# The `# %%` denotes a so called cell, a concept taken from matlab. IF you are using an integrated
# developer environment that supports it, it will be similar to jupyter cells, but more 
# true pythonesque. If you're using an editor that doesn't recognise `# %%`, nothing will
# happen because it will treat is as a comment. 


# %%
import datetime
import pathlib

import cartopy.crs as ccrs
import cmocean.cm as cmap
from geopy.distance import geodesic
import matplotlib.dates as mdates
import matplotlib.pyplot as pyplot
import numpy
import pandas

import xarray


# %%
steamy_data_directory_root = pathlib.Path('/Volumes/Rayleigh/cruise-oc4920/steamy_data')

# %%
# DRY (don't repeat yourself) paradigm (google it)
ferry_directory = steamy_data_directory_root / 'tsg'
ferry_file_path = ferry_directory / '123_2023-05-03_09.49.44_Amerikakajen_to_New harbor_741601'
ferry_file_path = ferry_directory / '124_2023-05-04_08.04.29_Skagen_to_Skagen_741601.tsv'
ferry_file_path = ferry_directory / '125_2023-05-05_09.06.10_Skagen_to_Nya_varvet_741601.tsv'


# This is optional. Set to None if you don't have it or want to use it.
ctd_positions_file = pathlib.Path(
    '/Volumes/Rayleigh/cruise-oc4920/data/ctd_positions_04.tsv'
)
ctd_begin_time = numpy.datetime64(datetime.datetime(2023, 5, 4, 8, 0))


bathymetry_file = steamy_data_directory_root / 'bathymetry' / 'gebco_2022_n60.0_s54.0_w7.5_e15.0.nc'


# %%
# %%
# Plotting parameters
_figsize = (14, 7)  # Size of your plot window in inches (hey, american made these, ok)

temperature_colour = 'tab:orange'
temperature_inlet_colour = 'tab:purple'

salinity_colour = 'tab:cyan'
chlorophyll_colour = 'tab:green'
oxygen_colour = 'tab:red'


# %%
# This is the gist of the this script, here the ferrybox data is loaded.
# But, the ferrybox file also contains the units of each columns, and these will not be loaded here.
# So further belof there is a function for this.
df = pandas.read_csv(ferry_file_path, skiprows=16, encoding='iso8859-1', sep='\t', parse_dates=[['Date', 'Time']], keep_date_col=True)

# I you prefer working with xarray:s than panda. xarray is numpy with many panda features on top,
# so therefor the .to_xarray method exists in pandas.
data = df.to_xarray()

# With xarray can only use variables denoted as coordinates as independent 
# variables in your axes. So for a line plot, the independent variable has to be on of the coordinates.
# For a colormesh or contour plot, the x- and y-axis has to be chosen
# from the coordinates section of an xarray.
data = data.assign_coords(dict(
    time=data.Date_Time,
    latitude=data.Latitude, 
    longitude=data.Longitude
    )
)

if bathymetry_file.exists():
    bathy = xarray.open_dataarray(bathymetry_file)
    bathy.name = 'bathymetry'


# %%
def get_units_per_colum_in_ferry_box_file(path):
    """Read out the units for each column in the ferrybox output file

    Args:
        path (str or pathlib.Path): path to your ferrybox file
    """
    def _get_names(_line):
        return _line.strip().split('\t')[1:]
    
    skip_rows = 15
    with open(path, 'r', encoding='iso8859-1') as buffer:
        [None for _, __ in zip(range(skip_rows), buffer)]
        units = buffer.readline()
        columns = buffer.readline()
    
    return dict(
        [(col, uni) 
         for col, uni in 
         zip(_get_names(columns), _get_names(units))])

        
# %%
# Convenience functions to pretty print when the independent variable in a plot is 
# time. 
def rot_ticks(ax, rotation=0, horizontal_alignment='center'):
    for xlabels in ax.get_xticklabels():
        xlabels.set_rotation(rotation)
        xlabels.set_ha(horizontal_alignment)
        
        
def time_axis_formatter(ax, interval=None):
    if interval is not None:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval))
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


# %%
def plot_wrt_x(_data, *, x_variable):
    fig, axes = pyplot.subplots(3, 1, figsize=_figsize, sharex='all')

    ax_temp = axes[0]

    ax_salinity = axes[1]
    ax_chlorophyll = axes[2]
    ax_oxygen = ax_chlorophyll.twinx()
    
    temperature = _data['Temp_SBE45']
    temperature_inlet = _data['Temp_in_SBE38']
    
    salinity = _data['Salinity_SBE45']
    chlorophyll = _data['Chlorophyll']
    oxygen = _data['Oxygen']
    
    temperature.plot(x=x_variable, ax=ax_temp, color=temperature_colour)
    ax_temp.tick_params(axis='y', labelcolor=temperature_colour)

    
    _y_min = min(min(temperature.values), min(temperature_inlet.values))
    _y_max = max(max(temperature.values), max(temperature_inlet.values))
    
    salinity.plot(x=x_variable, ax=ax_salinity, color=salinity_colour)
    ax_salinity.tick_params(axis='y', labelcolor=salinity_colour)
    
    chlorophyll.plot(x=x_variable, ax=ax_chlorophyll, color=chlorophyll_colour)
    oxygen.plot(x=x_variable, ax=ax_oxygen, color=oxygen_colour)
    ax_chlorophyll.set_ylim((0, 2.5))
    ax_oxygen.set_ylim((7, 12))
    ax_chlorophyll.tick_params(axis='y', labelcolor=chlorophyll_colour)
    ax_oxygen.tick_params(axis='y', labelcolor=oxygen_colour)
    
    for ax in fig.axes: ax.grid(True)
    
    axs = [ax_temp, ax_salinity]
    for ax in axs: ax.set_xlabel('')
    
    if 'Time' in x_variable:
        time_axis_formatter(ax_chlorophyll)
        rot_ticks(ax_chlorophyll)

    return (
        fig, 
        dict(
            ax_temp=ax_temp, 
            ax_salinity=ax_salinity,
            ax_chlorophyll=ax_chlorophyll,
            ax_oxygen=ax_oxygen
        )
    )

    
# %%
def cumulative_distance(dataset):
    latitudes = dataset.Latitude.values
    longitudes = dataset.Longitude.values
    
    positions = [(lat, lon) for lat, lon in zip(latitudes, longitudes)]
    start_pos = positions[0]
    positions = [start_pos,] + positions
    
    diffs = numpy.array(
        [geodesic(pos0, pos1).km 
         for pos0, pos1 in zip(positions[:-1], positions[1:])]
    )
    return numpy.cumsum(diffs)


# %%
def time_box_data(dataset, *, begin=None, end=None):
    indices = numpy.ones(dataset.index.shape, dtype=bool)
    if begin is not None:
        indices = (
            indices & (dataset.time >= begin).values
        )
    if end is not None:
        indices = (
            indices & (dataset.time <= end).values
        )
    selector = dict(index=data.index[indices].values)
    _dataset = dataset.sel(selector)
    distances = cumulative_distance(_dataset)
    return _dataset.assign_coords(
        dict(distance=xarray.DataArray(
                distances, 
                coords=dict(index=_dataset.index.values),
                dims=['index'],
                attrs=dict(units='km')   
            )
        )
    )


# %%
# In an xarray.DataArray, you can add attributes, like units for instance.
# Here we add the units parsed from the ferrybox file and add them to our loaded dataset
# that we assigned to the variable data
units = get_units_per_colum_in_ferry_box_file(ferry_file_path)
for var, unit in units.items():
    if len(unit) > 0:
        data[var].attrs = dict(units=unit)
    

# %%
_data = time_box_data(data, begin=ctd_begin_time)


# %%
if ctd_positions_file.exists():
    ctd_positions = pandas.read_csv(ctd_positions_file, sep='\t', parse_dates=[['date', 'time']]).to_xarray()    
    selector = dict(
        index=ctd_positions.index[[('CTD' in sn) for sn in ctd_positions.station_name.values]].values
    )
    ctd_positions = ctd_positions.sel(selector)

    ctd_times = ctd_positions.date_time.values
    ctd_station_names = ctd_positions.station_name.values
    indices = [
        (_data.index[_ctd_time == _data.time].values, sn)
        for _ctd_time, sn in zip(ctd_times, ctd_station_names)
    ]
    indices = [(idx[0], sn) for idx, sn in indices if len(idx) > 0]
    ctd_station_selector = dict(index=[idx[0] for idx in indices])
    station_data = _data.sel(ctd_station_selector)
    station_data['station_name'] = xarray.DataArray(
        [sn for idx, sn in indices],
        coords=dict(index=station_data.index),
        dims=['index']
    )
    
    
# %%
res = plot_wrt_x(data, x_variable='time')


# %%
fig, axes = plot_wrt_x(_data, x_variable='distance')
ax_chlorophyll = axes['ax_chlorophyll']
ax_oxygen = axes['ax_oxygen']
ax_oxygen.set_ylim((9.7, 10.7))
ax_chlorophyll.set_ylim((-0.2, .8))

ax_oxygen.set_xlim((0, 70))


if ctd_positions_file.exists():
    ax_temperature = axes['ax_temp']
    for distance, sn in zip(station_data.distance.values, station_data.station_name.values):
        ax_temperature.annotate(sn.split('_')[-1], xy=(distance, 1.05), xycoords=('data', 'axes fraction'))
        ax_temperature.axvline(
            distance, color='tab:grey'
        )
    ax_temperature_time = res[1]['ax_temp']
    for time, sn in zip(station_data.time.values, station_data.station_name.values):
        ax_temperature_time.annotate(sn.split('_')[-1], xy=(time, 1.05), xycoords=('data', 'axes fraction'))
        ax_temperature_time.axvline(
            time, color='tab:grey'
        )


# %%
fig = pyplot.figure(figsize=(16, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=11.))
img = data.plot.scatter(x='longitude', y='latitude', c=data.Salinity_SBE45, cmap=cmap.haline, ax=ax, s=2, ec=None)
if bathymetry_file.exists():
    bathy.plot.contour(x='lon', y='lat', ax=ax, levels=numpy.linspace(-500., 0, 21), linewidths=0.7, colors='tab:grey')
pyplot.colorbar(img)
ax.set_xlim((10, 11.94780225))
ax.set_ylim((57.3, 58.5))

if ctd_positions_file.exists():
    station_data.plot.scatter(x='longitude', y='latitude', s=9, marker='+', c='tab:red')
