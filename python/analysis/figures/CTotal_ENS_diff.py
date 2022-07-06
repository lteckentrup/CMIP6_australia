import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import matplotlib as mpl
import xarray as xr
import numpy as np
import xclim as xc
from xclim import ensembles
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

GCM_full = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5',
               'CESM2-WACCM', 'CMCC-CM2-SR5', 'EC-Earth3', 'EC-Earth3-Veg',
               'GFDL-CM4', 'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR',
               'KIOST-ESM', 'MIROC6', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR',
               'MRI-ESM2-0', 'NESM3', 'NorESM2-LM', 'NorESM2-MM']
GCM_skill = ['GFDL-CM4', 'GFDL-ESM4', 'KIOST-ESM', 'MPI-ESM1-2-HR',
                'MPI-ESM1-2-LR', 'MRI-ESM2-0']
GCM_independence = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'CESM2-WACCM',
                       'CMCC-CM2-SR5', 'MIROC6', 'MPI-ESM1-2-HR', 'NESM3',
                       'NorESM2-MM']
GCM_bounding = ['EC-Earth3-Veg', 'INM-CM4-8', 'KIOST-ESM', 'MPI-ESM1-2-HR',
                   'NorESM2-MM']

def wurst(BC_method,file,selection,Weighting):
    suffix = '_1850-2100.nc'

    data_list = []
    if Weighting == 'Uniform':
        pathway = '../LPJ_monthly_corrected/'+BC_method+'/'
        if selection == 'Full':
            GCM_list =  GCM_full
        elif selection == 'Skill':
            GCM_list =  GCM_skill
        elif selection == 'Independence':
            GCM_list =  GCM_independence
        elif selection == 'Bounding':
            GCM_list = GCM_bounding

        for GCM in GCM_list:
            data_list.append(pathway+GCM+'/'+file+'_'+GCM+suffix)

        ens = ensembles.create_ensemble(data_list).load()
        stats = ensembles.ensemble_mean_std_max_min(ens)
        ens_mean = stats['Total_mean']

        ens_mean['Time'] = pd.date_range(start='1850-01-01', end='2100-12-31',
                                         freq='A')

    elif Weighting == 'Weighted':
        ens_mean = xr.open_dataset('../LPJ_ensemble_averages/Weighted/Total_weighted_1850-2100.nc')
        ens_mean = ens_mean['Total']
    elif Weighting == 'Random Forest':
        ens_mean = xr.open_dataset('../LPJ_ensemble_averages/Random_Forest/Total_full_1850-2100.nc')
        ens_mean = ens_mean['Total']

    ENS_mean = ens_mean.sel(Time=slice('1989','2018')).mean(dim='Time', skipna=False)

    ds_CRUJRA = xr.open_dataset('../reanalysis/CTRL/CRUJRA/'+file+'_LPJ-GUESS_1901-2018.nc')
    ds_CRUJRA_mean = ds_CRUJRA.sel(Time=slice('1989','2018')).mean(dim='Time', skipna=False)

    return(((ENS_mean-ds_CRUJRA_mean['Total'])/ds_CRUJRA_mean['Total'])*100,
           ens_mean['Lat'].values,
           ens_mean['Lon'].values)

fig, axs = plt.subplots(nrows=4,ncols=3,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=(8,11))
axs=axs.flatten()

def plot_wurst(method, fname, var, position, selection, Weighting):
    ds, lat, lon = wurst(method, fname, selection, Weighting)

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

    p = axs[position].pcolormesh(lon, lat, ds.values, cmap=cmap, norm=norm)
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
    label_name = 'C$_\mathrm{Total,bias}$ [%]'

    fig.colorbar(p, cax=cax, ticks=levels, orientation='horizontal',
                 extend='both', label = label_name)

    print(np.nanmin(ds.values))
    print(np.nanmax(ds.values))

plot_wurst('original', 'cpool', 'Total', 0, 'Full', 'Uniform')
plot_wurst('', 'cpool', 'Total', 1, 'Bounding', 'Weighted')
plot_wurst('', 'cpool', 'Total', 2, 'Bounding', 'Random Forest')

plot_wurst('original', 'cpool', 'Total', 3, 'Skill', 'Uniform')
plot_wurst('original', 'cpool', 'Total', 4, 'Independence', 'Uniform')
plot_wurst('original', 'cpool', 'Total', 5, 'Bounding', 'Uniform')

plot_wurst('SCALING', 'cpool', 'Total', 6, 'Bounding', 'Uniform')
plot_wurst('MVA', 'cpool', 'Total', 7, 'Bounding', 'Uniform')
plot_wurst('QM', 'cpool', 'Total', 8, 'Bounding', 'Uniform')

plot_wurst('CDFt', 'cpool', 'Total', 9, 'Bounding', 'Uniform')
plot_wurst('R2D2', 'cpool', 'Total', 10, 'Bounding', 'Uniform')
plot_wurst('dOTC', 'cpool', 'Total', 11, 'Bounding', 'Uniform')

titles = ['ENS$_\mathrm{Arithmetic,Full}$',
          'ENS$_\mathrm{Weighted}$',
          'ENS$_\mathrm{RF}$',
          'ENS$_\mathrm{Arithmetic,Skill}$',
          'ENS$_\mathrm{Arithmetic,Independence}$',
          'ENS$_\mathrm{Arithmetic,Bounding}$',
          'ENS$_\mathrm{Arithmetic,Bounding,Scaling}$',
          'ENS$_\mathrm{Arithmetic,Bounding,MAV}$',
          'ENS$_\mathrm{Arithmetic,Bounding,QM}$',
          'ENS$_\mathrm{Arithmetic,Bounding,CDF-t}$',
          'ENS$_\mathrm{Arithmetic,Bounding,R2D2}$',
          'ENS$_\mathrm{Arithmetic,Bounding,dOTC}$']
titles_num = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 'k)',
              'l)']
axes_pos = [0,1,2,3,4,5,6,7,8,9,10,11]

for i, tn, t in zip(axes_pos, titles_num, titles):
    axs[i].set_title(t)
    axs[i].set_title(tn, loc='left')

figure_name = 'CTotal_diff_avg.png'

plt.subplots_adjust(top=0.96, left=0.05, right=0.97, bottom=0.1,
                    wspace=0.0, hspace=0.2)

# plt.subplot_tool()
plt.show()
# plt.savefig(figure_name, dpi=400)
