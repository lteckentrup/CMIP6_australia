import os
import xarray as xr
import pandas as pd
import numpy as np

path='../LPJ_monthly_corrected/'
path_full='../LPJ_monthly_corrected/original/'
path_bounding='../LPJ_monthly_corrected/dOTC/'

### Grab model names
model_names_unsorted = [name for name in os.listdir(path_full)
                        if os.path.isdir(os.path.join(path_full, name))]
model_names_unsorted.remove('CRUJRA')

model_names_bounding_unsorted = [name for name in os.listdir(path_bounding)
                                 if os.path.isdir(os.path.join(path_bounding, name))]

model_names = sorted(model_names_unsorted, key=str.lower)
model_names_bounding = sorted(model_names_bounding_unsorted, key=str.lower)


## Calculate area weighted sum
def area_weighted_sum(method, model, fname, var):
    gridarea = xr.open_dataset(path+'gridarea_mask.nc')
    model = xr.open_dataset(path+method+'/'+model+'/'+fname+'_'+
                            model+'_1850-2100.nc')

    model_sel = model.sel(Time=slice('1901','2018'))

    weighted = model_sel[var] * gridarea
    weighted = weighted.rename({'Total':var})
    sum = weighted[var].sum(dim=['Lat', 'Lon']) / 1e12
    return(sum.values)

## Calculate area weighted average
def area_weighted_avg(method, model, fname, var):
    gridarea = xr.open_dataset(path+'gridarea_mask.nc')
    model = xr.open_dataset(path+method+'/'+model+'/'+fname+'_'+
                            model+'_1850-2100.nc')

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

## Calculate area weighted seasonal sum
def area_weighted_seasonal_sum(method, model, var, PFT):
    gridarea = xr.open_dataset(path+'gridarea_mask.nc')
    model = xr.open_dataset(path+method+'/'+model+'/'+var+'_'+PFT+'_'+model+
                            '_1850-2100.nc')

    model_sel = model.sel(Time=slice('1901','2018'))

    clim = model_sel.groupby('Time.month').mean('Time')

    weighted = gridarea * clim[var+'_'+PFT]
    weighted = weighted.rename({'Total':var+'_'+PFT})
    sum = weighted[var+'_'+PFT].sum(dim=['Lat', 'Lon']) / 1e12
    return(sum.values)

df_temp = pd.DataFrame()
df_temp['Year'] = np.arange(1901,2019,1)
# df_temp = df_temp.set_index('Year')

for mn in model_names:
    df_temp[mn] = area_weighted_sum('original', mn, 'temp', 'temp')

df_temp.to_csv('temp_full_original.csv')
