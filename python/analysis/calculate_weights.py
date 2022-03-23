import numpy as np
import xarray as xr
import pandas as pd
from numpy.linalg import multi_dot
import os, psutil
from datetime import datetime

startTime = datetime.now()

def readin(file,var,model):
    if model == 'CRUJRA':
        fname = ('../../monthly_lpj_guess/runs_CRUJRA/'+file+'.out')
    else:
        fname = ('../../monthly_lpj_guess/runs_original/'+model+'/'+file+'.out')

    df = pd.read_csv(fname, delim_whitespace=True)

    df = df.loc[df['Year'].isin(np.arange(1989, 2011))]

    return(df[var].values)

models = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2-WACCM',
          'CMCC-CM2-SR5', 'EC-Earth3', 'EC-Earth3-Veg', 'GFDL-CM4', 'GFDL-ESM4',
          'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KIOST-ESM', 'MIROC6',
          'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM',
          'NorESM2-MM']

def weighting(file, var):
    df_obs = pd.DataFrame()
    df_sim = pd.DataFrame()

    df_obs['CRUJRA'] = readin(file,var,'CRUJRA')

    for m in models:
        df_sim[m] = readin(file,var,m)

    le_df_error = df_sim.sub(df_obs['CRUJRA'], axis=0)
    bc_term = le_df_error.mean(axis=0, skipna=True)
    df_sim_bc = df_sim.copy()
    df_sim_bc = df_sim_bc - bc_term

    error_df = df_sim_bc.sub(df_obs['CRUJRA'], axis=0)

    M_cov = error_df.cov()
    model_count = len(M_cov.columns)

    unit_col = np.ones((model_count,1))
    M_cov_inv = np.linalg.pinv(M_cov)

    unit_transpose = unit_col.transpose()
    weights = np.matmul(M_cov_inv, unit_col)/multi_dot([unit_transpose,
                                                        M_cov_inv,
                                                        unit_col])

    df_bc = pd.DataFrame(bc_term).transpose()
    df_weights = pd.DataFrame(weights.transpose(),columns = models)

    df_bc.to_csv('bc.csv')
    df_weights.to_csv('weights.csv')

weighting('cpool', 'Total')

process = psutil.Process(os.getpid())
print(process.memory_info().rss/(1024 ** 2))
print(datetime.now() - startTime)
