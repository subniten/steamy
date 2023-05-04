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
_figsize = (14, 7)


# %%
ferry_file = pathlib.Path('/Volumes/Rayleigh/cruise-oc4920/data/Ferrybox/2023-05-03_09.49.44_Amerikakajen_to__741601.csv')
ctd_positions_file = pathlib.Path(
    '/Volumes/Rayleigh/cruise-oc4920/data/ctd_positions.tsv'
)
bathymetry_file = '/Volumes/Rayleigh/cruise-oc4920/meta/gebco_2022_n60.0_s54.0_w7.5_e15.0.nc'


# %%
temperature_colour = 'tab:orange'
temperature_inlet_colour = 'tab:purple'

salinity_colour = 'tab:cyan'
chlorophyll_colour = 'tab:green'
oxygen_colour = 'tab:red'


# %%
df = pandas.read_csv(ferry_file, skiprows=16, encoding='iso8859-1', sep='\t', parse_dates=[['Date', 'Time']], keep_date_col=True)

data = df.to_xarray()
data = data.assign_coords(dict(
    time=data.Date_Time,
    latitude=data.Latitude, 
    longitude=data.Longitude
    )
)

bathy = xarray.open_dataarray(bathymetry_file)
bathy.name = 'bathymetry'


# %%

# %%
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
def get_units_per_colum_in_ferry_box_file(path):
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
units = get_units_per_colum_in_ferry_box_file(ferry_file)
for var, unit in units.items():
    if len(unit) > 0:
        data[var].attrs = dict(units=unit)
    

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
station_02_time = datetime.datetime(2023, 5, 3, 9, 29)
timestamp_station_02 = numpy.datetime64(station_02_time)

selector = dict(index=data.index[data.time > timestamp_station_02].values)
_data = data.sel(selector)
distances = cumulative_distance(_data)
_data = _data.assign_coords(
    dict(distance=xarray.DataArray(
            distances, 
            coords=dict(index=_data.index.values),
            dims=['index'],
            attrs=dict(units='km')   
        )
    )
)

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
bathy.plot.contour(x='lon', y='lat', ax=ax, levels=numpy.linspace(-500., 0, 21), linewidths=0.7, colors='tab:grey')
pyplot.colorbar(img)
ax.set_xlim((10, 11.94780225))
ax.set_ylim((57.3, 58.5))

if ctd_positions_file.exists():
    station_data.plot.scatter(x='longitude', y='latitude', s=9, marker='+', c='tab:red')
