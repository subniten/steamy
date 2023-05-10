import pathlib

import cartopy.crs as ccrs
import cmocean.cm as cmap
from geopy.distance import geodesic
import matplotlib.dates as mdates
import matplotlib.pyplot as pyplot
import pandas

from .steamy_common import (
    figsize,
    chlorophyll_colour,
    oxygen_colour,
    salinity_colour,
    temperature_colour,
)
from .steamy_plotting_utilities import (
    rotate_xtick_labels,
    time_axis_formatter,
)


def read_ferrybox_directory(ferrybox_directory):
    def read_ferrybox_tsv(ferrybox_file_path):
        return pandas.read_csv(
            ferrybox_file_path,
            skiprows=16,
            encoding='iso8859-1',
            sep='\t',
            parse_dates=[['Date', 'Time']],
            keep_date_col=True,
        )

    ferrybox_files = sorted(pathlib.Path(ferrybox_directory).glob('*.tsv'))
    _ = [read_ferrybox_tsv(fp) for fp in ferrybox_files]
    units = get_units_per_column_in_ferry_box_file(ferrybox_files[0])
    tsg_dataframe = pandas.concat(_, ignore_index=True)
    data = tsg_dataframe.to_xarray()
    for varname, units in units.items():
        if varname in data:
            if data[varname].attrs is None:
                data[varname].attrs = dict(units=units)
            else:
                data[varname].attrs['units'] = units

    return data.assign_coords(
        dict(time=data.Date_Time, latitude=data.Latitude, longitude=data.Longitude)
    )


def get_units_per_column_in_ferry_box_file(path):
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
        [(col, uni) for col, uni in zip(_get_names(columns), _get_names(units))]
    )


def plot_tsg_with_respect_to_x_variable(_data, *, x_variable, figsize=figsize):
    fig, axes = pyplot.subplots(3, 1, figsize=figsize, sharex='all')

    ax_temp = axes[0]

    ax_salinity = axes[1]
    ax_chlorophyll = axes[2]
    ax_oxygen = ax_chlorophyll.twinx()

    temperature = _data['Temp_SBE45']

    salinity = _data['Salinity_SBE45']
    chlorophyll = _data['Chlorophyll']
    oxygen = _data['Oxygen']

    temperature.plot(x=x_variable, ax=ax_temp, color=temperature_colour)
    ax_temp.tick_params(axis='y', labelcolor=temperature_colour)

    salinity.plot(x=x_variable, ax=ax_salinity, color=salinity_colour)
    ax_salinity.tick_params(axis='y', labelcolor=salinity_colour)

    chlorophyll.plot(x=x_variable, ax=ax_chlorophyll, color=chlorophyll_colour)
    oxygen.plot(x=x_variable, ax=ax_oxygen, color=oxygen_colour)
    ax_chlorophyll.set_ylim((0, 2.5))
    ax_oxygen.set_ylim((7, 12))
    ax_chlorophyll.tick_params(axis='y', labelcolor=chlorophyll_colour)
    ax_oxygen.tick_params(axis='y', labelcolor=oxygen_colour)

    for ax in fig.axes:
        ax.grid(True)

    axs = [ax_temp, ax_salinity]
    for ax in axs:
        ax.set_xlabel('')

    if 'Time' in x_variable:
        time_axis_formatter(ax_chlorophyll)
        rotate_xtick_labels(ax_chlorophyll)

    return (
        fig,
        dict(
            ax_temp=ax_temp,
            ax_salinity=ax_salinity,
            ax_chlorophyll=ax_chlorophyll,
            ax_oxygen=ax_oxygen,
        ),
    )
