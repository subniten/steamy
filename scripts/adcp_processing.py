import cmocean.cm as cmo
import matplotlib.pyplot as pyplot
import numpy

import steamy_utilities

steamy_utilities.set_steamy_data_root_path(
    '/Volumes/Rayleigh/cruise-oc4920/steamy_data'
)

adcp_files = sorted(steamy_utilities.get_adcp_directory().glob('*.mat'))
file = '230504_STA.mat'

adcp_file = [s for s in adcp_files if file in s.name][0]

_figsize = steamy_utilities.figsize

#
adcp_data = steamy_utilities.read_adcp_file(adcp_file)

# Plot velocity contours as fucntion of distance and depth
current_velocities_levels = numpy.linspace(-0.5, 0.5, 11)


fig, axes = pyplot.subplots(1, 2, figsize=_figsize, sharex='all', sharey='all')
plot_kwargs = dict(y='depth', cmap=cmo.balance, levels=current_velocities_levels)
u_img = adcp_data.u.plot(ax=axes[0], **plot_kwargs)
v_img = adcp_data.v.plot(ax=axes[1], **plot_kwargs)

axes[0].set_ylim(0, 100)
axes[0].invert_yaxis()

fig.tight_layout()


# Plot vectors in map

# Read Bathymetry File
bathymetry = steamy_utilities.load_bathymetry(steamy_utilities.get_bathymetry_file())

depth_level = 10
fig, ax2 = pyplot.subplots(1, 1, figsize=_figsize)
bathymetry.plot.contour(ax=ax2, x='lon', y='lat', levels=[0], colors='grey')
bathymetry.plot.contour(
    ax=ax2,
    x='lon',
    y='lat',
    levels=numpy.arange(-500, 0, 20),
    colors='grey',
    linewidths=0.7,
)

img_quiv = ax2.quiver(
    adcp_data.longitude,
    adcp_data.latitude,
    adcp_data.u[:, depth_level],
    adcp_data.v[:, depth_level],
    scale=35,
    width=0.002,
)
ax2.quiverkey(img_quiv, 0.9, 0.9, 1, '1 ms$^{-1}$', coordinates='axes')

ax2.set_ylim(57.2, 58.5)
ax2.set_xlim(10.3, 12.3)
ax2.set(
    title=f'Currents at depth: {adcp_data.depth[depth_level].values:.04} m',
    xlabel='Latitude (˚)',
    ylabel='Longitude (˚)',
)

pyplot.tight_layout()
pyplot.show()


# Depth averaged currents
Umean = adcp_data.u.mean(dim='depth')
Vmean = adcp_data.v.mean(dim='depth')

fig, ax3 = pyplot.subplots(1, 1, figsize=(8, 6))
bathymetry.plot.contour(ax=ax3, x='lon', y='lat', levels=[0], colors='grey')
bathymetry.plot.contour(
    ax=ax3,
    x='lon',
    y='lat',
    levels=numpy.arange(-500, 0, 20),
    colors='grey',
    linewidths=0.7,
)

img_quiv_2 = ax3.quiver(
    adcp_data.longitude, adcp_data.latitude, Umean, Vmean, scale=35, width=0.002
)
ax3.quiverkey(img_quiv_2, 0.9, 0.9, 1, '1 ms$^{-1}$', coordinates='axes')
ax3.set_ylim(57.2, 58.5)
ax3.set_xlim(10.3, 12.3)
ax3.set(
    title='Depth Averaged Currents', xlabel='Latitude [deg]', ylabel='Longitude [deg]'
)
fig.tight_layout()
