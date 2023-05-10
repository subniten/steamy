# %%
import pathlib

from cmocean import cm as cmo
import gsw
import matplotlib.pyplot as pyplot
import numpy
from scipy.interpolate import griddata
import xarray

# For this to work, make sure steamy_utils directory is in your PYTHONPATH
#  i) notebook userds
#      start your notebook in a directory containing steamy_utilities directory by
#      PYTHONPATH=`pwd` jupyterlab-lab
# ii) python/ipython users
#      PYTHONPATH=`pwd` ipython --pylab
import steamy_utilities


# %%
reference_depth = 5  # (m)
mld_threshold = 0.03  # (density)

steamy_root_directory = pathlib.Path('/Volumes/Rayleigh/cruise-oc4920/steamy_data')
output_directory = steamy_root_directory.parent / 'output_figures'


# %%
output_directory.mkdir(exist_ok=True)
figsize = steamy_utilities.figsize


# %%
steamy_utilities.set_steamy_data_root_path(steamy_root_directory)

ctd_directory = steamy_utilities.get_ctd_directory()

ctd_files = [fp for fp in sorted(ctd_directory.glob('*.cnv'))]
station_ids = numpy.array([s.stem.split('_')[1] for s in ctd_files], dtype=str)

ctd_casts = steamy_utilities.read_ctd_files(ctd_files, station_identifiers=station_ids)

conservative_temperature = ctd_casts.conservative_temperature
conservative_temperature = conservative_temperature.where(
    conservative_temperature > 1.0
)

absolute_salinity = ctd_casts.absolute_salinity
absolute_salinity = absolute_salinity.where(absolute_salinity > 1.0)

sigma_0 = ctd_casts.sigma_0
sigma_0 = sigma_0.where(sigma_0 > 10.0)

# %%
fig, axes = pyplot.subplots(1, 3, figsize=figsize, sharey='all', sharex='all')

plt_kwargs = dict(x='cast', y='depth')
plot_temperature = conservative_temperature.plot(
    ax=axes[0], cmap=cmo.thermal, **plt_kwargs
)
plot_salinity = absolute_salinity.plot(ax=axes[1], cmap=cmo.haline, **plt_kwargs)

plot_density = sigma_0.plot(ax=axes[2], cmap=cmo.dense, **plt_kwargs)
axes[0].invert_yaxis()

fig.tight_layout()


# %%
labels = ctd_casts.station_name.values


# %%
fig, axes = pyplot.subplots(1, 3, figsize=figsize, sharey='all')

for ct, sa, sigma, lbl in zip(
    conservative_temperature.T, absolute_salinity.T, sigma_0.T, labels
):
    ct.plot(ax=axes[0], y='depth', label=lbl)

    sa.plot(ax=axes[1], y='depth', label=lbl)
    sigma.plot(ax=axes[2], y='depth', label=lbl)

axes[0].set_ylim((0, 100))
axes[0].invert_yaxis()
axes[0].set_xlim((6.0, 10.0))
axes[1].set_xlim((29.5, 35.0))
axes[2].set_xlim((22.5, 27.5))

axes[0].legend()
axes[1].legend()
axes[2].legend()

axes[0].invert_yaxis()

for ax, title in zip(fig.axes[:3], ['$\\Theta$', 'S$_A$', '$\\sigma_0$']):
    ax.set_title(title)


# %%
for ct, sa, sigma, lbl in zip(
    conservative_temperature.T, absolute_salinity.T, sigma_0.T, labels
):
    fig, axes = pyplot.subplots(1, 3, figsize=figsize, sharey='all')

    ct.plot(ax=axes[0], y='depth', label=lbl)
    sa.plot(ax=axes[1], y='depth', label=lbl)
    sigma.plot(ax=axes[2], y='depth', label=lbl)
    axes[0].set_ylim((0, 100))
    axes[0].invert_yaxis()
    axes[0].set_xlim((6.0, 10.0))
    axes[1].set_xlim((29.5, 35.0))
    axes[2].set_xlim((22.5, 27.5))
    for ax, title in zip(fig.axes[:3], ['$\\Theta$', 'S$_A$', '$\\sigma_0$']):
        ax.legend()
        ax.grid(True)
        ax.set_title(title)
    fig.savefig(output_directory / f'ctd_{lbl}.png', dpi=200)


# %%
idx = numpy.argmin(numpy.abs((ctd_casts.depth.values - reference_depth)))
mixed_layer_depths = float('nan') * numpy.ones(ctd_casts.dims['cast'])

for cast_nr in ctd_casts.cast.values:
    cast = ctd_casts.sel(dict(cast=cast_nr))
    reference_density = cast.density[idx].values
    for depth_index in range(idx, ctd_casts.dims['depth']):
        if abs(reference_density - cast.density[depth_index]) >= mld_threshold:
            mixed_layer_depths[cast_nr] = cast.depth[depth_index].values
            break

mixed_layer_depth = xarray.DataArray(
    mixed_layer_depths,
    dims=['cast'],
    coords=dict(cast=ctd_casts.cast),
    attrs=dict(units='m'),
    name='mld',
)


# %%
fig, axes = pyplot.subplots(1, 1, figsize=figsize)

mixed_layer_depth.plot(ax=axes, x='cast', marker='x', linestyle='-')
axes.grid(True)
axes.invert_yaxis()
