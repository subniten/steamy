# %%
from glob import glob

from cmocean import cm as cmo
import gsw
import matplotlib.pyplot as pyplot
import numpy
import pandas
from scipy.interpolate import griddata
from seabird.cnv import fCNV


# %%


max_z = 70

files = sorted(glob('*.cnv'))

z = numpy.arange(0, max_z, 0.2)

temp_grid = numpy.ndarray([numpy.size(z), numpy.size(files)])
sal_grid = numpy.ndarray([numpy.size(z), numpy.size(files)])

for i, fname in enumerate(files):
    ctd_cast = fCNV(fname)

    temp_cast = ctd_cast['TEMP'].data
    sal_cast = ctd_cast['PSAL'].data

    depth_cast = ctd_cast['DEPTH'].data

    temp_grid[:, i] = griddata(depth_cast, temp_cast, z)
    sal_grid[:, i] = griddata(depth_cast, sal_cast, z)


# %%


fig, ax = pyplot.subplots(figsize=[10, 6], ncols=2)

fig1 = ax[0].pcolormesh(numpy.arange(numpy.size(files)), z, temp_grid, cmap=cmo.thermal)
ax[0].invert_yaxis()
cb1 = pyplot.colorbar(fig1, ax=ax[0], label='temperature [°C]')
ax[0].set_xticks(numpy.arange(0, 7, 1))
ax[0].set(title='temperature [°C]', xlabel='CTD cast', ylabel='depth [m]')

fig2 = ax[1].pcolormesh(numpy.arange(numpy.size(files)), z, sal_grid, cmap=cmo.haline)
ax[1].invert_yaxis()
cb2 = pyplot.colorbar(fig2, ax=ax[1], label='salinity [psu]')
ax[1].set_xticks(numpy.arange(0, 7, 1))
ax[1].set(title='salinity [psu]', xlabel='CTD cast', ylabel=None)

pyplot.tight_layout()
pyplot.show()


# %%
# data = [ctd_001 ,ctd_002 ,ctd_003 ,ctd_004 ,ctd_005 ,ctd_006 ,ctd_007]
labels = ['ctd_001', 'ctd_002', 'ctd_003', 'ctd_004', 'ctd_005', 'ctd_006', 'ctd_007']


# %%
temp_grid.shape


# %%
fig, ax = pyplot.subplots(figsize=[10, 6], ncols=2)

for i in range(numpy.size(files)):
    ax[0].plot(temp_grid[:, i], z, label=labels[i])
    ax[0].invert_yaxis()
    ax[0].legend()

    ax[1].plot(sal_grid[:, i], z, label=labels[i])
    ax[1].invert_yaxis()
    ax[1].legend()

pyplot.show()


# %%
rho = []
mld = []

for j, dat in enumerate(data):
    for z in range(0, len(ctd_cast['DEPTH']) + 1):
        mld_cast = fCNV(fname)
        rho.apend(gsw.sigma0(mld_cast['PSAL'], mld_cast['TEMP']))
