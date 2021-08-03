import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.colors import BoundaryNorm
import matplotlib as mpl
import xarray as xr
import numpy as np
from netCDF4 import Dataset as netcdf_dataset
from cartopy.io import shapereader
import geopandas

# get country borders
resolution = '10m'
category = 'cultural'
name = 'admin_0_countries'

shpfilename = shapereader.natural_earth(resolution, category, name)
df = geopandas.read_file(shpfilename)
poly = df.loc[df['ADMIN'] == 'Australia']['geometry'].values[0]

def outsider(model,file,var):
    if file == 'NEE_cumsum':
        ensstd = ('../CMIP6/CTRL/'+file+'_LPJ-GUESS_1901-2018_ensstd.nc')
        ensmean = ('../CMIP6/CTRL/'+file+'_LPJ-GUESS_1901-2018_ensmean.nc')
        model = ('../CMIP6/CTRL/'+model+'/'+file+'_LPJ-GUESS_1901-2018.nc')
    else:
        ensstd = ('../CMIP6/CTRL/'+file+'_LPJ-GUESS_1850-2100_ensstd.nc')
        ensmean = ('../CMIP6/CTRL/'+file+'_LPJ-GUESS_1850-2100_ensmean.nc')
        model = ('../CMIP6/CTRL/'+model+'/'+file+'_LPJ-GUESS_1850-2100.nc')

    ds_ensstd = xr.open_dataset(ensstd)
    ds_ensmean = xr.open_dataset(ensmean)
    ds_model = xr.open_dataset(model)

    ds_upper = ds_ensmean + ds_ensstd
    ds_lower = ds_ensmean - ds_ensstd

    cond1_exspread=ds_model[var]>ds_lower[var]
    cond2_exspread=ds_model[var]<ds_upper[var]

    cond1_inspread=ds_model[var]<ds_lower[var]
    cond2_inspread=ds_model[var]>ds_upper[var]

    exspread = ds_model[var].where(cond1_exspread&cond2_exspread,10000)
    youknowit = exspread.where(cond1_inspread|cond2_inspread,0)

    youknowit_sum = youknowit.sel(Time=slice('1901-01-01',
                                             '2018-12-31')).sum(dim='Time')/10000
    youknowit_percentage = youknowit_sum/118 * 100

    upper = youknowit_percentage.where(youknowit_percentage>50,0)
    lower = upper.where(upper<50,1)
    return(lower)

nbp = outsider('CanESM5','cflux','NEE')
nbp_cum = outsider('CanESM5','NEE_cumsum','NEE')
veg = outsider('CanESM5','cpool','VegC')
soil = outsider('CanESM5','cpool','SoilC')

nbp_exspread = nbp*0
nbp_cum_exspread = nbp_cum*0
veg_exspread = veg*0
soil_exspread = soil*0

modelz = ['CanESM5', 'CESM2-WACCM', 'CMCC-CM2-SR5', 'EC-Earth3', 'EC-Earth3-Veg',
          'GFDL-CM4', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KIOST-ESM',
          'MIROC6', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0',
          'NorESM2-LM', 'NorESM2-MM']

for m in modelz:
    nbp = outsider(m,'cflux','NEE')
    nbp_cum = outsider(m,'NEE_cumsum','NEE')
    veg = outsider(m,'cpool','VegC')
    soil = outsider(m,'cpool','SoilC')

    nbp_exspread = nbp_exspread+nbp
    nbp_cum_exspread = nbp_cum_exspread+nbp_cum
    veg_exspread = veg_exspread+veg
    soil_exspread = soil_exspread+soil

cflux_ensmax = ('../CMIP6/CTRL/cflux_LPJ-GUESS_1850-2100_ensmax.nc')
ds_cflux_ensmax = xr.open_dataset(cflux_ensmax)

lat = ds_cflux_ensmax['Lat'].values
lon = ds_cflux_ensmax['Lon'].values

data_list = [soil_exspread.where(ds_cflux_ensmax['NEE'].isel(Time=0)>=0),
             soil_exspread.where(ds_cflux_ensmax['NEE'].isel(Time=0)>=0),
             nbp_exspread.where(ds_cflux_ensmax['NEE'].isel(Time=0)>=0),
             nbp_cum_exspread.where(ds_cflux_ensmax['NEE'].isel(Time=0)>=0),
             veg_exspread.where(ds_cflux_ensmax['NEE'].isel(Time=0)>=0),
             soil_exspread.where(ds_cflux_ensmax['NEE'].isel(Time=0)>=0)]

exp_names = ['NBP', 'NBP', 'CVeg', 'CSoil','CVeg', 'CSoil']

projection = ccrs.PlateCarree()
axes_class = (GeoAxes,
              dict(map_projection=projection))

fig = plt.figure(figsize=(8,10))

axgr = AxesGrid(fig, 111, axes_class=axes_class,
                nrows_ncols=(3, 2),
                axes_pad=0.3,
                label_mode='')

levels = np.arange(0,11,1)

cmap = plt.cm.viridis_r
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist[0] = 'lightgrey'
cmap = mpl.colors.LinearSegmentedColormap.from_list('mcm', cmaplist,
                                                    cmap.N)

norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

for (i, ax), en in zip(enumerate(axgr), exp_names):
    # ax.coastlines()

    masked = np.ma.masked_where(data_list[i] == 0, data_list[i])
    p = ax.pcolormesh(lon, lat, data_list[i], cmap=cmap, norm=norm)
    ax.add_geometries(poly, crs=ccrs.PlateCarree(), facecolor='none',
                      edgecolor='0.0')
    ax.set_extent([112.25,153.75,-43.75,-10.75], crs=ccrs.PlateCarree())
    print(np.nanmin(data_list[i]))
    print(np.nanmax(data_list[i]))

    ax.axis('off')

cax = plt.axes([0.1, 0.05, 0.8, 0.04])
fig.colorbar(p, cax=cax, ticks=levels, orientation='horizontal',
             extend='neither', label = '# models')
plt.subplots_adjust(top=0.98, left=0.03, right=0.97, bottom=0.08,
                    wspace=0.0, hspace=0.0)

plt.show()
