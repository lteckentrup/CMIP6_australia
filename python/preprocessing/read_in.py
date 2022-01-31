import os
import xarray as xr
import pandas as pd
import numpy as np
from matplotlib.pyplot import cm

### Grab model names
path_full='../LPJ_monthly_corrected/original/'
path_bounding='../LPJ_monthly_corrected/dOTC/'

model_names_unsorted = [name for name in os.listdir(path_full)
                        if os.path.isdir(os.path.join(path_full, name))]

model_names_bounding_unsorted = [name for name in os.listdir(path_bounding)
                                 if os.path.isdir(os.path.join(path_bounding, name))]

model_names = sorted(model_names_unsorted, key=str.lower)
model_names.remove('CRUJRA')
model_names.append('CRUJRA')

model_names_bounding = sorted(model_names_bounding_unsorted, key=str.lower)

### Generate colormap
color_20=cm.tab20(np.arange(0,20,1))
color_add=cm.tab20b([0])
black = np.array([0,0,0,1], ndmin=2)
cmap=np.vstack((color_20,color_add,black))

### Calculate area weighted averages and sums
path='../LPJ_monthly_corrected/'

## Area weighted sum
def area_weighted_sum(method, model, fname, var):
    gridarea = xr.open_dataset(path+'gridarea_mask.nc')
    if model == 'CRUJRA':
        suffix='_1901-2018.nc'
    else:
        suffix='_1850-2100.nc'

    model = xr.open_dataset(path+method+'/'+model+'/'+fname+'_'+
                            model+suffix)

    model_sel = model.sel(Time=slice('1901','2018'))

    weighted = model_sel[var] * gridarea
    weighted = weighted.rename({'Total':var})
    sum = weighted[var].sum(dim=['Lat', 'Lon']) / 1e12
    return(sum.values)

## Area weighted average
def area_weighted_avg(method, model, fname, var):
    gridarea = xr.open_dataset(path+'gridarea_mask.nc')
    if model == 'CRUJRA':
        suffix='_1901-2018.nc'
    else:
        suffix='_1850-2100.nc'

    model = xr.open_dataset(path+method+'/'+model+'/'+fname+'_'+
                            model+suffix)

    if fname in ('temp', 'prec', 'insol'):
        model_sel = model.sel(time=slice('1901','2018'))

        if fname in ('temp', 'insol'):
            model_annual = model_sel.groupby('time.year').mean('time')
        elif fname == 'prec':
            model_annual = model_sel.groupby('time.year').sum('time')

        model_annual = model_annual.rename({'lat':'Lat', 'lon':'Lon'})

    else:
        model_sel = model.sel(Time=slice('1901','2018'))
        model_annual = model_sel

    weighted = model_annual[var] * gridarea
    weighted = weighted.rename({'Total':var})
    avg = weighted[var].sum(dim=['Lat', 'Lon']) / gridarea.Total.sum(dim=['Lat', 'Lon'])
    return(avg.values)

## Area weighted seasonal sum
def area_weighted_seasonal_sum(method, model, var, PFT):
    gridarea = xr.open_dataset(path+'gridarea_mask.nc')

    if model == 'CRUJRA':
        suffix='_1901-2018.nc'
    else:
        suffix='_1850-2100.nc'

    model = xr.open_dataset(path+method+'/'+model+'/'+var+'_'+PFT+'_'+model+suffix)

    model_sel = model.sel(Time=slice('1901','2018'))

    clim = model_sel.groupby('Time.month').mean('Time')

    weighted = gridarea * clim[var+'_'+PFT]
    weighted = weighted.rename({'Total':var+'_'+PFT})
    sum = weighted[var+'_'+PFT].sum(dim=['Lat', 'Lon']) / 1e12
    return(sum.values)

### Generate output files
def generate_dataframe(method, fname, var):
    df = pd.DataFrame()

    if var == 'mpgpp':
        df['month'] = np.arange(1,13,1)
    else:
        df['Year'] = np.arange(1901,2019,1)

    for mn in model_names:
        if var in ('Total', 'NEE'):
            df[mn] = area_weighted_sum(method, mn, fname, var)
        elif var in ('temp', 'prec', 'insol'):
            df[mn] = area_weighted_avg(method, mn, fname, var)
        elif var == 'mpgpp':
            df[mn] = area_weighted_seasonal_sum(method, model, var, PFT)

    if var == 'mpgpp':
        return(df.set_index('month'))
    else:
        return(df.set_index('Year'))

df_temp = generate_dataframe('original', 'temp', 'temp')
df_temp.to_csv(path+'original_csv/temp_full.csv')

df_prec = generate_dataframe('original', 'prec', 'prec')
df_prec.to_csv(path+'original_csv/prec_full.csv')

df_Total = generate_dataframe('original', 'cpool', 'Total')
df_Total.to_csv(path+'original_csv/CTotal_full.csv')

df_NEE = generate_dataframe('original', 'cflux', 'NEE')
df_NEE.to_csv(path+'original_csv/NEE_full.csv')
