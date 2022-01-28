from write_netcdf import (
    convert_ascii_netcdf_monthly,
    convert_ascii_netcdf_annual
    )

'''
convert_ascii_netcdf_monthly(pathway, var, experiment)
convert_ascii_netcdf_annual(pathway, var, experiment)
'''

methods = ['original', 'SCALING', 'MVA', 'QM', 'CDFt', 'MRec', 'dOTC']

model_names = ['EC-Earth3-Veg',
               'INM-CM4-8',
               'KIOST-ESM',
               'MPI-ESM1-2-HR',
               'NorESM2-MM']

vars_annual = ['cflux', 'cpool', 'fpc']
vars_monthly = ['mpgpp']
PFTs = ['C4G']

for va in vars_annual:
    print(va)
    convert_ascii_netcdf_annual('original', va, 'CRUJRA', 'daily')
    for mn in model_names:
        print(mn)
        for m in methods:
            convert_ascii_netcdf_annual(m, va, mn, 'daily')

for va in vars_monthly:
    print(va)
    for PFT in PFTs:
        print(PFT)
        convert_ascii_netcdf_monthly('original', va, 'CRUJRA', 'daily', 'C4G')
        for mn in model_names:
            print(mn)
            for m in methods:
                convert_ascii_netcdf_monthly(m, va, mn, 'daily', 'C4G')
