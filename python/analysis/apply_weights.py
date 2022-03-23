import xarray as xr
import pandas as pd
import numpy as np

models = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2-WACCM',
          'CMCC-CM2-SR5', 'EC-Earth3', 'EC-Earth3-Veg', 'GFDL-CM4', 'GFDL-ESM4',
          'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KIOST-ESM', 'MIROC6',
          'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM',
          'NorESM2-MM']

weights = pd.read_csv('weights.csv')
df_bc = pd.read_csv('bc.csv')

def apply(model, fname, var):
    ds = xr.open_dataset('../CMIP6/CTRL/'+model+'/'+fname+
                         '_LPJ-GUESS_1850-2100.nc')

    ds['Time'] = pd.date_range(start='1850-01-01', end='2100-12-31', freq='A')
    ds_var = ds[var]
    ds_weight = weights[model].values*(ds_var-df_bc[model].values)

    return(ds_weight)

empty = np.zeros([251,68,84])
Time = pd.date_range(start='1850-01-01', end='2100-12-31', freq='A')
Lat = np.arange(-43.75,-9.75,0.5)
Lon = np.arange(112.25,154.25,0.5)

da_hybrid = xr.DataArray(empty,
                         coords={'Time': Time,'Lat': Lat, 'Lon': Lon},
                         dims=['Time', 'Lat', 'Lon'])

for m in models:
    da_hybrid = da_hybrid + apply(m, 'cpool', 'Total')

ds_new = da_hybrid.to_dataset(name='Total')

ds_new.to_netcdf('../LPJ_ensemble_averages/Weighted/Total_weighted_1850-2100.nc',
                 encoding={'Time':{'dtype': 'double'},
                           'Lat':{'dtype': 'double'},
                           'Lon':{'dtype': 'double'},
                           'Total':{'dtype': 'float32'}})
