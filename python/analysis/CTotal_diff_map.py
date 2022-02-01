import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import matplotlib as mpl
import xarray as xr
import numpy as np
from cartopy.io import shapereader
import geopandas

resolution = '10m'
category = 'cultural'
name = 'admin_0_countries'

shpfilename = shapereader.natural_earth(resolution, category, name)
df = geopandas.read_file(shpfilename)
poly = df.loc[df['ADMIN'] == 'Australia']['geometry'].values[0]

def wurst(method,model,file,var,selection):
    CRUJRA = ('../reanalysis/CTRL/CRUJRA/'+file+'_LPJ-GUESS_1901-2018.nc')

    if method in ('original', 'QM', 'CDFt', 'MRec'):
        COR = ('../LPJ_monthly_corrected/'+method+'/'+model+'/'+file+'_'+
               model+'_1850-2100.nc')
    elif method in ('SCALING', 'MVA'):
        COR = ('../LPJ_monthly_corrected/'+method+'/'+model+'/'+file+'_'+
                     model+'_1851-2100.nc')
    elif method == 'dOTC':
        COR = ('../LPJ_monthly_corrected/'+method+'/'+model+'/'+file+'_'+
                     model+'_1851-2025.nc')
    elif method == 'Uniform':
        COR = ('../LPJ_ensemble_averages/Uniform/cpool_'+selection+
                     '_1850-2100.nc')
    elif method == 'Random Forest':
        COR = ('../LPJ_ensemble_averages/Random_Forest/'+var+'_'+
                     selection+'_1850-2100.nc')
    elif method == 'Weighted':
        COR = ('../LPJ_ensemble_averages/Weighted/'+var+
                     '_weighted_1850-2100.nc')

    ds_CRUJRA = xr.open_dataset(CRUJRA)
    ds_COR = xr.open_dataset(COR)

    ds_COR_diff_raw = ds_COR - ds_CRUJRA

    ds_COR_diff=ds_COR_diff_raw.sel(Time=slice('1989-01-01',
                                               '2018-12-31')).mean(dim='Time',
                                                                   skipna=False)

    return(ds_COR_diff[var]*1000,
           ds_CRUJRA['Lat'].values,
           ds_CRUJRA['Lon'].values)

fig, axs = plt.subplots(nrows=7,ncols=7,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=(11,9.5))
axs=axs.flatten()

def plot_wurst(method, model, fname, var, position, selection):
    ds, lat, lon = wurst(method, model, fname, var, selection)

    projection = ccrs.PlateCarree()

    levels = [-5000,-2000,-1000,-500,-200,-100,-50,
              50,100,200,500,1000,2000,5000]

    cmap_cbar = 'BrBG'
    cmap_colors = plt.cm.get_cmap(cmap_cbar,len(levels)+1)
    colors = list(cmap_colors(np.arange(len(levels)+1)))

    middle = (len(levels)/2)
    colors[int(middle)] = 'lightgrey'

    cmap = mpl.colors.ListedColormap(colors[1:-1], '')
    cmap.set_over(colors[-1])
    cmap.set_under(colors[0])

    norm = mpl.colors.BoundaryNorm(levels, ncolors=len(levels)-1, clip=False)

    p = axs[position].pcolormesh(lon, lat, ds, cmap=cmap, norm=norm)
    axs[position].add_geometries(poly, crs=ccrs.PlateCarree(), facecolor='none',
                      edgecolor='0.0')
    axs[position].set_extent([112.25,153.75,-43.75,-10.75],
                             crs=ccrs.PlateCarree())
    axs[position].axis('off')

    cax = plt.axes([0.1, 0.06, 0.8, 0.035])
    label_name = '$\Delta$ CTotal$\mathrm{_{CMIP-CRUJRA}}$ (1989-2018) [gC]'

    fig.colorbar(p, cax=cax, ticks=levels, orientation='horizontal',
                 extend='both', label = label_name)

#     # print(np.nanmin(data_list[i]))
#     # print(np.nanmax(data_list[i]))

model_names=['EC-Earth3-Veg', 'INM-CM4-8', 'KIOST-ESM', 'MPI-ESM1-2-HR',
             'NorESM2-MM', 'Ensemble mean']
pos_titles=[0,7,14,21,28,35]

for mn, pt in zip(model_names, pos_titles):
    axs[pt].text(105, -25, mn, ha='center', va='center', rotation=90,
                size=12)

pos_ec=[0,1,2,3,4,5,6]
pos_inm=[7,8,9,10,11,12,13]
pos_kiost=[14,15,16,17,18,19,20]
pos_mpi=[21,22,23,24,25,26,27]
pos_nor=[28,29,30,31,32,33,34]

pos_list=[pos_ec,pos_inm,pos_kiost,pos_mpi,pos_nor]
model_names_bounding=['EC-Earth3-Veg', 'INM-CM4-8', 'KIOST-ESM', 'MPI-ESM1-2-HR',
                      'NorESM2-MM']

bc_methods=['original', 'SCALING', 'MVA', 'QM', 'CDFt', 'MRec', 'dOTC']
bc_methods_title=['original', 'Scaling', 'MAV', 'QM', 'CDF-t', 'MRec', 'dOTC']

for pl, mn in zip(pos_list,model_names_bounding):
    for pe,bm,bmt in zip(pl,bc_methods,bc_methods_title):
        plot_wurst(bm, mn, 'cpool', 'Total', pe, '')
        if mn == 'EC-Earth3-Veg':
            axs[pe].set_title(bmt)

pos_ens=[36,38,40]
ens_methods=['Uniform', 'Weighted', 'Random Forest']

for pe,em in zip(pos_ens,ens_methods):
    plot_wurst(em, '', 'cpool', 'Total', pe, 'full')
    axs[pe].set_title(em)

for i in [35,37,39,41,45]:
    fig.delaxes(axs[i])

pos_uniform=[42,43,44]
pos_rf=[46,47,48]
sel_methods=['skill', 'independence','bounding']
sel_methods_title=['Skill', 'Independence', 'Bounding']

for pu,sm,smt in zip(pos_uniform, sel_methods, sel_methods_title):
    plot_wurst('Uniform', '', 'cpool', 'Total', pu, sm)
    axs[pu].set_title(smt)

for pr,sm,smt in zip(pos_rf, sel_methods, sel_methods_title):
    plot_wurst('Random Forest', '', 'cpool', 'Total', pr, sm)
    axs[pr].set_title(smt)

figure_name = 'CTotal_diff_map.png'

plt.subplots_adjust(top=0.97, left=0.05, right=0.97, bottom=0.1,
                    wspace=0.08, hspace=0.2)

# plt.subplot_tool()
# plt.show()
plt.savefig(figure_name, dpi=400)
