import pathlib

from geopy.distance import geodesic
import numpy
import xarray

_steamy_paths = {}

figsize = (12, 8)  # default figure size in inches

chlorophyll_colour = 'tab:green'
oxygen_colour = 'tab:red'
salinity_colour = 'tab:cyan'
temperature_colour = 'tab:orange'


# Steamy data paths
def set_steamy_data_root_path(path):
    """Set the root directory path for steamy data

    Args:
        path (str or pathlib.Path): path to root directory of steamy data

    Raises:
        ValueError: if provided path is not a directory
    """
    p = pathlib.Path(path)
    if p.is_dir():
        _steamy_paths['root_path'] = pathlib.Path(path)
    else:
        raise ValueError(f'Given path {path} is not a directory')
    _initalise_sub_directories()


def get_adcp_directory():
    if 'adcp_directory' in _steamy_paths:
        return _steamy_paths['adcp_directory']


def get_bathymetry_directory():
    if 'bathymetry_directory' in _steamy_paths:
        return _steamy_paths['bathymetry_directory']


def get_bathymetry_file():
    return get_bathymetry_directory() / 'steamy_bathymetry.nc'


def get_ctd_directory():
    if 'ctd_directory' in _steamy_paths:
        return _steamy_paths['ctd_directory']


def get_ferrybox_directory():
    if 'ferrybox_directory' in _steamy_paths:
        return _steamy_paths['ferrybox_directory']


def _initalise_sub_directories():
    if 'root_path' in _steamy_paths:
        root_path = _steamy_paths['root_path']
        _steamy_paths['bathymetry_directory'] = root_path / 'bathymetry'
        _steamy_paths['ctd_directory'] = root_path / 'steamy_ctd_data'
        _steamy_paths['ferrybox_directory'] = root_path / 'tsg'
        _steamy_paths['adcp_directory'] = (
            root_path / 'steamy_adcp_data/Single Date Files'
        )


def cumulative_distance(dataset, latitude_name='Latitude', longitude_name='Longitude'):
    """Calculates the accumulated distance from the first position


    Args:
        dataset (xarray.Dataset or pandas.Dataframe): dataset containing latitude and longitude values for each index
        latitude_name (str, optional): name of the latitude variable in the provided dataset. Defaults to 'Latitude'.
        longitude_name (str, optional): name of the longitude variable in the provided dataset. Defaults to 'Longitude'.

    Returns:
        numpy.array: the accumulated distance in (km) from the first point
    """
    latitudes = dataset[latitude_name].values
    longitudes = dataset[longitude_name].values

    positions = [(lat, lon) for lat, lon in zip(latitudes, longitudes)]
    start_pos = positions[0]
    positions = [
        start_pos,
    ] + positions

    diffs = numpy.array(
        [geodesic(pos0, pos1).km for pos0, pos1 in zip(positions[:-1], positions[1:])]
    )
    return numpy.cumsum(diffs)


def load_bathymetry(bathymetry_file_path):
    bathy = xarray.open_dataarray(bathymetry_file_path)
    bathy.name = 'bathymetry'
    bathy.attrs = dict(units='m')

    return bathy
