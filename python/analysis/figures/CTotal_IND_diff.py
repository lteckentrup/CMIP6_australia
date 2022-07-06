import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import matplotlib as mpl
import xarray as xr
import numpy as np
import matplotlib.patches as mpatches

def wurst(method,model,file,var,selection):
    CRUJRA = ('../reanalysis/CTRL/CRUJRA/'+file+'_LPJ-GUESS_1901-2018.nc')

    if method in ('original', 'SCALING', 'MVA', 'QM', 'CDFt', 'R2D2', 'dOTC'):
        GCM = ('../LPJ_monthly_Corrected/'+method+'/'+model+'/'+file+'_'+
               model+'_1850-2100.nc')
    elif method == 'Uniform':
        GCM = ('../LPJ_ensemble_averages/Uniform/'+file+'_'+selection+
               '_1850-2100.nc')
    elif method == 'Random Forest':
        GCM = ('../LPJ_ensemble_averages/Random_Forest/'+var+'_'+
                     selection+'_1850-2100.nc')
    elif method == 'Weighted':
        GCM = ('../LPJ_ensemble_averages/Weighted/'+var+'_weighted_1850-2100.nc')

    ds_CRUJRA = xr.open_dataset(CRUJRA)
    ds_GCM = xr.open_dataset(GCM)

    ds_GCM_mean = ds_GCM.sel(Time=slice('1989-01-01',
                                        '2018-12-31')).mean(dim='Time',
                                                           skipna=False)
    ds_CRUJRA_mean = ds_CRUJRA.sel(Time=slice('1989-01-01',
                                              '2018-12-31')).mean(dim='Time',
                                                                 skipna=False)

    ds_GCM_diff = (ds_GCM_mean - ds_CRUJRA_mean)/ds_CRUJRA_mean

    return(ds_GCM_diff[var]*100,
           ds_CRUJRA['Lat'].values,
           ds_CRUJRA['Lon'].values)

fig, axs = plt.subplots(nrows=5,ncols=7,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=(11,7.5))
axs=axs.flatten()

def plot_wurst(method, model, fname, var, position, selection):
    ds, lat, lon = wurst(method, model, fname, var, selection)

    projection = ccrs.PlateCarree()
    levels = [-70,-60,-50,-40,-30,-20,-10,-5,5,10,20,30,40,50,60,70]

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
    axs[position].set_extent([112.25,153.75,-43.75,-10.75],
                             crs=ccrs.PlateCarree())
    axs[position].axis('off')
    axs[position].coastlines()


    axs[position].add_patch(mpatches.Rectangle(xy=[118, -12], width=10, height=6,
                                               facecolor='white',
                                               zorder=12,
                                               transform=ccrs.PlateCarree())
                 )
    axs[position].add_patch(mpatches.Rectangle(xy=[146, -14], width=20, height=10,
                                               facecolor='white',
                                               zorder=12,
                                               transform=ccrs.PlateCarree())
                 )

    cax = plt.axes([0.1, 0.06, 0.8, 0.035])
    label_name = '$\Delta$ C$\mathrm{_{Total,bias}}$ [%]'

    fig.colorbar(p, cax=cax, ticks=levels, orientation='horizontal',
                 extend='both', label = label_name)

model_names=['LG$_\mathrm{EC-Earth3-Veg}$',
             'LG$_\mathrm{INM-CM4-8}$',
             'LG$_\mathrm{KIOST-ESM}$',
             'LG$_\mathrm{MPI-ESM1-2-HR}$',
             'LG$_\mathrm{NorESM2-MM}$']

pos_titles=[0,7,14,21,28]

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

bc_methods=['original', 'SCALING', 'MVA', 'QM', 'CDFt', 'R2D2', 'dOTC']
bc_methods_title=['original', 'Scaling', 'MAV', 'QM', 'CDF-t', 'R2D2', 'dOTC']

for pl, mn in zip(pos_list,model_names_bounding):
    for pe,bm,bmt in zip(pl,bc_methods,bc_methods_title):
        plot_wurst(bm, mn, 'cpool', 'Total', pe, '')
        if mn == 'EC-Earth3-Veg':
            axs[pe].set_title(bmt)

figure_name = 'CTotal_IND_diff.png'

plt.subplots_adjust(top=0.97, left=0.05, right=0.97, bottom=0.1,
                    wspace=0.08, hspace=0.2)

# plt.subplot_tool()
plt.show()
# plt.savefig(figure_name, dpi=400)
