import os
import xarray as xr
import pandas as pd
import numpy as np
from matplotlib.pyplot import cm

### Grab GCM names
path_full='../LPJ_monthly_corrected/original/'
path_bounding='../LPJ_monthly_corrected/dOTC/'

GCM_names_unsorted = [name for name in os.listdir(path_full)
                        if os.path.isdir(os.path.join(path_full, name))]

GCM_names_bounding_unsorted = [name for name in os.listdir(path_bounding)
                                 if os.path.isdir(os.path.join(path_bounding, name))]

GCM_names_full = sorted(GCM_names_unsorted, key=str.lower)
GCM_names_full.remove('CRUJRA')
GCM_names_full.append('CRUJRA')

GCM_names_bounding = sorted(GCM_names_bounding_unsorted, key=str.lower)

### Generate colormap
color_20=cm.tab20(np.arange(0,20,1))
color_add=cm.tab20b([0])
black = np.array([0,0,0,1], ndmin=2)
cmap=np.vstack((color_20,color_add,black))

### Bias correction methods
methods=['original', 'SCALING', 'MVA', 'QM', 'CDFt', 'dOTC', 'MRec', 'R2D2']

### Calculate area weighted averages and sums
path_BC='../LPJ_monthly_corrected/'
path_avg='../LPJ_ensemble_averages/'

## Area weighted sum
def area_weighted_sum(method, GCM_name, fname, var, selection):
    if method in ('original', 'SCALING', 'MVA', 'QM', 'CDFt', 'dOTC', 'MRec', 'R2D2'):
        path=path_BC
    else:
        path=path_avg

    gridarea = xr.open_dataset(path_BC+'gridarea_mask.nc')
    if GCM_name == 'CRUJRA':
        suffix='_1901-2018.nc'
    else:
        suffix='_1850-2100.nc'

    if method in ('original', 'SCALING', 'MVA', 'QM', 'CDFt', 'dOTC', 'MRec', 'R2D2'):
        GCM = xr.open_dataset(path+method+'/'+GCM_name+'/'+fname+'_'+
                                GCM_name+suffix)
    elif method == 'Uniform':
        GCM = xr.open_dataset(path+method+'/'+fname+'_'+selection+suffix)
    elif method == 'Weighted':
        GCM = xr.open_dataset(path+method+'/'+var+'_weighted'+suffix)
    elif method == 'Random_Forest':
        GCM = xr.open_dataset(path+method+'/'+var+'_'+selection+suffix)

    GCM_sel = GCM.sel(Time=slice('1901','2018'))

    weighted = GCM_sel[var] * gridarea
    weighted = weighted.rename({'Total':var})
    sum = weighted[var].sum(dim=['Lat', 'Lon']) / 1e12
    return(sum.values)

## Area weighted average
def area_weighted_avg(method, GCM_name, fname, var, selection):
    if method in ('original', 'SCALING', 'MVA', 'QM', 'CDFt', 'dOTC', 'MRec', 'R2D2'):
        path=path_BC
    else:
        path=path_avg

    gridarea = xr.open_dataset(path_BC+'gridarea_mask.nc')
    if GCM_name == 'CRUJRA':
        suffix='_1901-2018.nc'
    else:
        suffix='_1850-2100.nc'

    if method in ('original', 'SCALING', 'MVA', 'QM', 'CDFt', 'dOTC', 'MRec', 'R2D2'):
        GCM = xr.open_dataset(path+method+'/'+GCM_name+'/'+fname+'_'+
                                GCM_name+suffix)
    elif method == 'Uniform':
        GCM = xr.open_dataset(path+method+'/'+fname+'_'+selection+suffix)
    elif method == 'Weighted':
        GCM = xr.open_dataset(path+method+'/'+var+'_weighted'+suffix)
    elif method == 'Random_Forest':
        GCM = xr.open_dataset(path+method+'/'+var+'_'+selection+suffix)

    if fname in ('temp', 'prec', 'insol'):
        GCM_sel = GCM.sel(time=slice('1901','2018'))

        if fname in ('temp', 'insol'):
            GCM_annual = GCM_sel.groupby('time.year').mean('time')
            GCM_annual = GCM_annual.rename({'lat':'Lat', 'lon':'Lon'})
        elif fname == 'prec':
            GCM_annual = GCM_sel.groupby('time.year').sum('time')
            GCM_annual = GCM_annual.rename({'lat':'Lat', 'lon':'Lon'})

        if method == 'Weighted':
            GCM_annual = GCM_annual.rename({'lat':'Lat', 'lon':'Lon'})
        else:
            pass

    else:
        GCM_sel = GCM.sel(Time=slice('1901','2018'))
        GCM_annual = GCM_sel

    weighted = GCM_annual[var] * gridarea
    weighted = weighted.rename({'Total':var})
    avg = weighted[var].sum(dim=['Lat', 'Lon']) / gridarea.Total.sum(dim=['Lat', 'Lon'])
    return(avg.values)

## Area weighted seasonal sum
def area_weighted_seasonal_sum(method, GCM, var, PFT, selection):
    if method in ('original', 'SCALING', 'MVA', 'QM', 'CDFt', 'dOTC', 'MRec', 'R2D2'):
        path=path_BC
    else:
        path=path_avg

    gridarea = xr.open_dataset(path_BC+'gridarea_mask.nc')

    if GCM == 'CRUJRA':
        suffix='_1901-2018.nc'
    else:
        suffix='_1850-2100.nc'

    if method in ('original', 'SCALING', 'MVA', 'QM', 'CDFt', 'dOTC', 'MRec', 'R2D2'):
        GCM = xr.open_dataset(path+method+'/'+GCM+'/'+var+'_'+PFT+'_'+GCM+suffix)
    elif method == 'Uniform':
        GCM = xr.open_dataset(path+method+'/'+var+'_'+PFT+'_'+selection+suffix)
    elif method == 'Weighted':
        GCM = xr.open_dataset(path+method+'/'+var+'_'+PFT+'_weighted'+suffix)
    elif method == 'Random_Forest':
        GCM = xr.open_dataset(path+method+'/'+var+'_'+PFT+'_full'+suffix)

    GCM_sel = GCM.sel(Time=slice('1989','2010'))
    clim = GCM_sel.groupby('Time.month').mean('Time')

    weighted =  clim[var+'_'+PFT] * gridarea['Total']
    sum = weighted.sum(dim=['Lat', 'Lon']) / 1e12
    return(sum.values)

### Generate output files
def generate_dataframe_BC(method, fname, var, PFT):
    df = pd.DataFrame()

    if var == 'mpgpp':
        df['month'] = np.arange(1,13,1)
    else:
        df['Year'] = np.arange(1901,2019,1)

    if method == 'original':
        GCM_names = GCM_names_full
    else:
        GCM_names = GCM_names_bounding

    for mn in GCM_names:
        if var in ('Total', 'NEE'):
            df[mn] = area_weighted_sum(method, mn, fname, var, '')
        elif var in ('temp', 'prec', 'insol'):
            df[mn] = area_weighted_avg(method, mn, fname, var, '')
        elif var == 'mpgpp':
            df[mn] = area_weighted_seasonal_sum(method, mn, var, PFT, '')

    if var == 'mpgpp':
        return(df.set_index('month'))
    else:
        return(df.set_index('Year'))

### Generate output files
def generate_dataframe_avg(method, fname, var, PFT):
    df = pd.DataFrame()

    if var == 'mpgpp':
        df['month'] = np.arange(1,13,1)
    else:
        df['Year'] = np.arange(1901,2019,1)

    selection_methods = ['full', 'skill', 'independence', 'bounding']

    if method == 'Uniform':
        for sm in selection_methods:
            if var in ('Total', 'NEE'):
                df[sm] = area_weighted_sum(method, '', fname, var, sm)
            elif var in ('temp', 'prec', 'insol'):
                df[sm] = area_weighted_avg(method, '', fname, var, sm)
            elif var == 'mpgpp':
                df[sm] = area_weighted_seasonal_sum(method, '', var, PFT, sm)
    else:
        if var in ('Total', 'NEE'):
            df['full'] = area_weighted_sum(method, '', fname, var, '')
        elif var in ('temp', 'prec', 'insol'):
            df['full'] = area_weighted_avg(method, '', fname, var, '')
        elif var == 'mpgpp':
            df['full'] = area_weighted_seasonal_sum(method, '', var, PFT, '')

    if var == 'mpgpp':
        return(df.set_index('month'))
    else:
        return(df.set_index('Year'))

df_prec = generate_dataframe_BC('R2D2', 'temp', 'temp', '')
df_prec.to_csv(path_BC+'R2D2_csv/temp_full.csv')

