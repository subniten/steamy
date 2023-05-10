import datetime
import pathlib
import warnings

import numpy
import scipy
import xarray

from .steamy_common import cumulative_distance


# warnings.filterwarnings('ignore')

_nan_int_value = -32768


def read_adcp_file(adcp_file_path):
    def _adcp_int_to_floats(mat_array):
        indices = mat_array == _nan_int_value
        _array = mat_array.astype(float)
        _array[indices] = float('nan')
        return _array

    def _get_depth_bins(_data):
        first_bin_depth = _adcp_int_to_floats(_data['RDIBin1Mid'])
        bin_size = float(_data['RDIBinSize'][0][0])
        return (first_bin_depth + (_data['SerBins'][0] - 1) * bin_size)[0]

    def _velocity_to_floats(mat_array):
        _velocities = _adcp_int_to_floats(mat_array) * 1e-3
        return _velocities

    def _get_times(_data):
        def _fix_year(*t):
            return (t[0] + 2000,) + t[1:]

        time_keys = dict(
            year='SerYear',
            month='SerMon',
            day='SerDay',
            hour='SerHour',
            minutes='SerMin',
            seconds='SerSec',
        )
        return numpy.array(
            [
                datetime.datetime(*(_fix_year(*timestamp)))
                for timestamp in zip(
                    *[data[_][:, 0].astype(int) for _ in time_keys.values()]
                )
            ],
            dtype='datetime64[ns]',
        )

    adcp_file_path = pathlib.Path(adcp_file_path)
    data = scipy.io.loadmat(adcp_file_path.open('rb'))

    depth_bins = _get_depth_bins(data)
    time_stamps = _get_times(data)

    time_1d_kwargs = dict(coords=dict(time=time_stamps), dims=['time'])
    positions = dict(
        latitude=xarray.DataArray(
            _adcp_int_to_floats(data['AnFLatDeg']).T[0], **time_1d_kwargs
        ),
        longitude=xarray.DataArray(
            _adcp_int_to_floats(data['AnFLonDeg']).T[0], **time_1d_kwargs
        ),
    )
    distance = xarray.DataArray(
        cumulative_distance(
            xarray.Dataset(positions),
            latitude_name='latitude',
            longitude_name='longitude',
        ),
        attrs=dict(units='km'),
        **time_1d_kwargs,
    )

    time_depth_kwargs = dict(
        coords=dict(time=time_stamps, depth=depth_bins, distance=distance),
        dims=['time', 'depth'],
    )
    vel_attrs = dict(units='m·s$^{-1}$')
    zonal_velocities = xarray.DataArray(
        _velocity_to_floats(data['SerEmmpersec']), attrs=vel_attrs, **time_depth_kwargs
    )
    meridional_velocity = xarray.DataArray(
        _velocity_to_floats(data['SerNmmpersec']), attrs=vel_attrs, **time_depth_kwargs
    )
    vertical_velocity = xarray.DataArray(
        _velocity_to_floats(data['SerVmmpersec']), attrs=vel_attrs, **time_depth_kwargs
    )

    return xarray.Dataset(
        dict(
            u=_set_bottom_bin_to_nan(zonal_velocities),
            v=_set_bottom_bin_to_nan(meridional_velocity),
            w=_set_bottom_bin_to_nan(vertical_velocity),
            **positions,
        )
    )


def _set_bottom_bin_to_nan(_velocity):
    bottom_bin_index = numpy.array(
        [(numpy.sum(~numpy.isnan(_.values)) - 1) for _ in _velocity]
    )
    for idx, vel in zip(bottom_bin_index, _velocity):
        if idx > -1:
            vel[idx] = float('nan')
    return _velocity
