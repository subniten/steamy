# %%
import datetime

from cmocean import cm as cmo
import ctd
import gsw
import matplotlib.pyplot as pyplot
import numpy
from scipy.interpolate import griddata
import xarray


def get_position_in_cnv_file(cnv_file_path):
    def _parse_position(_line):
        _sign_hemispheres = dict(N=1, S=-1, E=1, W=-1)
        left, right = _line.strip().split(' = ')
        lat_lon = left.split(' ')[-1].strip().lower()
        deg, dec_min, hemisphere = right.split(' ')
        return (
            lat_lon,
            (float(deg) + float(dec_min) / 60.0) * _sign_hemispheres[hemisphere],
        )

    with open(cnv_file_path, 'r') as ctd_file:
        return dict(
            [
                _parse_position(line)
                for line in ctd_file
                if 'NMEA' in line and 'itude' in line
            ]
        )


def get_cast_time_in_cnv_file(cnv_file_path):
    format = '%b %d %Y %H:%M:%S'
    with open(cnv_file_path, 'r') as ctd_file:
        return [
            datetime.datetime.strptime(line.split('=')[1].strip(), format)
            for line in ctd_file
            if 'NMEA UTC (Time)' in line
        ][0]


def mixed_layer_depth(_ctd_casts, *, mld_threshold, reference_depth):
    idx = numpy.argmin(numpy.abs((_ctd_casts.depth.values - reference_depth)))
    mixed_layer_depths = float('nan') * numpy.ones(_ctd_casts.dims['cast'])
    
    for cast_nr in _ctd_casts.cast.values:
        cast = _ctd_casts.sel(dict(cast=cast_nr))
        reference_density = cast.density[idx].values
        for depth_index in range(idx, _ctd_casts.dims['depth']):
            if abs(reference_density - cast.density[depth_index]) >= mld_threshold:
                mixed_layer_depths[cast_nr] = cast.depth[depth_index].values
                break
    
    return xarray.DataArray(
        mixed_layer_depths,
        dims=['cast'],
        coords=dict(cast=_ctd_casts.cast),
        attrs=dict(units='m', long_name='mld', short_name='mixed_layer_depth'),
        name='mld',
    )


def read_ctd_files(ctd_files, station_identifiers=None):
    ctd_casts = [ctd.from_cnv(ctd_file) for ctd_file in ctd_files]
    ctd_cast_positions = [get_position_in_cnv_file(ctd_file) for ctd_file in ctd_files]

    max_depth = max([cast.index.max() for cast in ctd_casts])

    depth = numpy.arange(0, max_depth, 0.2)

    conservative_temperature = numpy.ndarray([numpy.size(depth), numpy.size(ctd_files)])
    absolute_salinity = numpy.ndarray([numpy.size(depth), numpy.size(ctd_files)])
    density = numpy.ndarray([numpy.size(depth), numpy.size(ctd_files)])
    sigma_0 = numpy.ndarray([numpy.size(depth), numpy.size(ctd_files)])
    for cast_nr, (cast, position) in enumerate(zip(ctd_casts, ctd_cast_positions)):
        pressure = cast.index
        salinity_cast = gsw.SA_from_SP(
            cast['sal00'], pressure, position['longitude'], position['latitude']
        ).values
        temperature_cast = gsw.CT_from_t(salinity_cast, cast['t090C'], pressure).values
        density_cast = gsw.rho(salinity_cast, temperature_cast, pressure).values
        sigma_cast = gsw.sigma0(salinity_cast, temperature_cast)
        depth_cast = cast['depSM'].values

        conservative_temperature[:, cast_nr] = griddata(
            depth_cast, temperature_cast, depth
        )
        absolute_salinity[:, cast_nr] = griddata(depth_cast, salinity_cast, depth)
        density[:, cast_nr] = griddata(depth_cast, density_cast, depth)
        sigma_0[:, cast_nr] = griddata(depth_cast, sigma_cast, depth)

    if station_identifiers is None:
        station_name = [f'{nr:03}' for nr in numpy.arange(len(ctd_casts), dtype=int)]
    else:
        station_name = station_identifiers
    station_name = numpy.array(station_name, dtype=str)

    casts = numpy.arange(len(ctd_casts), dtype=int)
    kwargs = dict(dims=['cast'], coords=dict(cast=casts))

    ctd_cast_times = xarray.DataArray(
        numpy.array(
            [
                numpy.datetime64(get_cast_time_in_cnv_file(ctd_file), 'ns')
                for ctd_file in ctd_files
            ],
            dtype=numpy.datetime64,
        ),
        **kwargs,
    )
    latitudes = xarray.DataArray(
        numpy.array([el['latitude'] for el in ctd_cast_positions]), **kwargs
    )
    longitudes = xarray.DataArray(
        numpy.array([el['longitude'] for el in ctd_cast_positions]), **kwargs
    )
    station_names = xarray.DataArray(station_name, **kwargs)

    depth = xarray.DataArray(
        depth, dims=['depth'], coords=dict(depth=depth), attrs=dict(units='m')
    )
    kwargs = dict(
        dims=['depth', 'cast'],
        coords=dict(
            depth=depth,
            cast=casts,
            latitude=latitudes,
            longitude=longitudes,
            time=ctd_cast_times,
        ),
    )

    rho_name = '$\\rho$'
    ct_name = '$\\Theta$'
    sa_name = 'S$_A$'
    si_name = '$\\sigma_0$'
    rho_unit = 'kg·m$^{-3}$'
    arrays = dict(
        absolute_salinity=xarray.DataArray(
            absolute_salinity,
            attrs=dict(units='g·kg$^{-1}$', long_name=sa_name),
            **kwargs,
        ),
        conservative_temperature=xarray.DataArray(
            conservative_temperature,
            attrs=dict(units='˚C', long_name=ct_name),
            **kwargs,
        ),
        density=xarray.DataArray(
            density, attrs=dict(units=rho_unit, long_name=rho_name), **kwargs
        ),
        sigma_0=xarray.DataArray(
            sigma_0, attrs=dict(units=rho_unit, long_name=si_name), **kwargs
        ),
        station_name=station_names,
    )

    return xarray.Dataset(arrays)


def buoyancy_frequency_squared_from_density_profile(sigma_or_rho, *, rho_0, g=9.81, z_var='depth', cut_off_per_m=25.):
    if sigma_or_rho[z_var][4] > 0:
        z = -sigma_or_rho[z_var].values
    else:
        z = sigma_or_rho[z_var].values
    _axis = sigma_or_rho.dims.index(z_var)
    gradient = numpy.gradient(sigma_or_rho.values, z, axis=_axis)
    gradient = numpy.where(numpy.abs(gradient) < cut_off_per_m, gradient, float('nan'))
    n_squared = (-1 * g / rho_0) * gradient
    return xarray.DataArray(
        n_squared,
        coords=sigma_or_rho.coords,
        dims=sigma_or_rho.dims,
        attrs=dict(units='s$^{-2}$', long_name='N$^2$'),
        name='buoyancy_freqency_sq'
    )
