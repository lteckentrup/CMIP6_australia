import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import pandas as pd
from pylab import text
import glob

fig = plt.figure(figsize=(10.0,10))

fig.subplots_adjust(hspace=0.12)
fig.subplots_adjust(wspace=0.18)
fig.subplots_adjust(right=0.98)
fig.subplots_adjust(left=0.08)
fig.subplots_adjust(bottom=0.17)
fig.subplots_adjust(top=0.94)

plt.rcParams['text.usetex'] = False
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.size'] = 11
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

df_NBP = pd.DataFrame()
df_CVeg = pd.DataFrame()
df_CSoil = pd.DataFrame()
df_CRUNCEP = pd.DataFrame()
df_CRUJRA = pd.DataFrame()
df_GSWP3 = pd.DataFrame()

vars = ['nbp', 'VegC', 'SoilC']

suffix_cmip6= 'LPJ-GUESS_1850-2100_annual_australia_oz.nc'
suffix_cruncep= 'LPJ-GUESS_1901-2015_annual_australia_oz.nc'
suffix_crujra= 'LPJ-GUESS_1901-2018_annual_australia_oz.nc'
suffix_gswp3= 'LPJ-GUESS_1901-2010_annual_australia_oz.nc'

cmip6_cflux=glob.glob('../CMIP6/CTRL/**/cflux_'+suffix_cmip6, recursive=True)
cmip6_cpool=glob.glob('../CMIP6/CTRL/**/cpool_'+suffix_cmip6, recursive=True)

reanalysis_cflux=glob.glob('../reanalysis/CTRL/**/cflux_*annual_australia_oz.nc',
                           recursive=True)
reanalysis_cpool=glob.glob('../reanalysis/CTRL/**/cpool_*_annual_australia_oz.nc',
                           recursive=True)

names_prel = [w.replace('../CMIP6/CTRL/', '') for w in cmip6_cflux]
model_names = [w.replace('/cflux_'+suffix_cmip6, '') for w in names_prel]

names_prel = [w.replace('../reanalysis/CTRL/', '') for w in reanalysis_cflux]
names_prel2  = [sub.replace('/cflux_'+suffix_cruncep, '') for sub in names_prel]
names_prel3  = [sub.replace('/cflux_'+suffix_crujra, '') for sub in names_prel2]
reanalysis_names = [sub.replace('/cflux_'+suffix_gswp3, '') for sub in names_prel3]

length_cmip6 = np.arange(0,len(cmip6_cflux))
length_reanalysis = np.arange(0,len(reanalysis_cflux))

for i, mn in zip(length_cmip6, model_names):
    Cflux_CMIP6 = nc.Dataset(cmip6_cflux[i])
    Cpool_CMIP6 = nc.Dataset(cmip6_cpool[i])

    df_NBP[mn] = Cflux_CMIP6.variables['NEE'][51:169,0,0]*(-1)
    df_CVeg[mn] = Cpool_CMIP6.variables['VegC'][51:169,0,0]
    df_CSoil[mn] = Cpool_CMIP6.variables['SoilC'][51:169,0,0]

dataframes_reanalysis = [df_CRUJRA, df_CRUNCEP, df_GSWP3]
for i, rn, df in zip(length_reanalysis, reanalysis_names, dataframes_reanalysis):
    Cflux_reanalysis = nc.Dataset(reanalysis_cflux[i])
    Cpool_reanalysis = nc.Dataset(reanalysis_cpool[i])

    df['NBP'] = Cflux_reanalysis.variables['NEE'][:,0,0]*(-1)
    df['CVeg'] = Cpool_reanalysis.variables['VegC'][:,0,0]
    df['CSoil'] = Cpool_reanalysis.variables['SoilC'][:,0,0]
    df['year'] = np.arange(1901,1901+len(df),1)

df_NBP['max'] = df_NBP.max(axis=1)
df_NBP['min'] = df_NBP.min(axis=1)

df_NBP['mean'] = df_NBP.mean(axis=1)
df_NBP['std'] = df_NBP.std(axis=1)
df_NBP['mean+std'] = df_NBP['mean'] + df_NBP['std']
df_NBP['mean-std'] = df_NBP['mean'] - df_NBP['std']

dataframes_cmip6 = [df_NBP, df_CVeg, df_CSoil]
for df in dataframes_cmip6:
    df['year'] = np.arange(1901,2019,1)

print(df_NBP.tail())
ax1.plot(df_NBP['year'], df_NBP['mean'],lw=3.0, ls="-",
         label='CMIP6 ensemble mean', alpha = 1, color='tab:green')

ax1.fill_between(df_NBP['year'], df_NBP['min'], df_NBP['max'],
                 color='tab:green', alpha=0.15, label='Model spread')

# ax1.set_ylim([-0.95,1.7])

ax1.axhline(linewidth=2, color='k', alpha=0.5)
ax1.legend(loc='lower left', ncol=1, fancybox=False, frameon=False, fontsize=12)

ax1.set_ylabel('NBP [PgC yr-1]')
ax1.set_title('Annual NBP Australia')

## Cumulative sum individual models
for mn in model_names:
    ax2.plot(df_NBP['year'], df_NBP[mn].cumsum(), lw=2.0, alpha=0.4, label=mn)
    ax3.plot(df_NBP['year'], df_CVeg[mn], lw=2.0, alpha=0.4, label=mn)
    ax4.plot(df_NBP['year'], df_CSoil[mn], lw=2.0, alpha=0.4, label=mn)

colors = ['k', '#5a5a5a', '#c0c0c0']
for rn, c, df in zip(reanalysis_names, colors, dataframes_reanalysis):
    if rn == 'GSWP3':
        pass
    else:
        ax1.plot(df['year'], df['NBP'], color=c, lw=2.0, label='LPJ GUESS '+rn)
        ax2.plot(df['year'], df['NBP'].cumsum(), color=c, lw=3.0,
                 label='LPJ GUESS '+rn)
        ax3.plot(df['year'], df['CVeg'], color=c, lw=3.0,
                 label='LPJ GUESS '+rn)
        ax4.plot(df['year'], df['CSoil'], color=c, lw=3.0,
                 label='LPJ GUESS '+rn)

vars = ['nbp', 'cVeg', 'cSoil']
df_TRENDY = pd.DataFrame()

for v in vars:
    lpj_guess = nc.Dataset('../TRENDY/'+v+'/processed/LPJ-GUESS_S2_'+v+
                           '_australia_annual_oz.nc')
    df_TRENDY[v] = lpj_guess.variables[v][:,0,0]

df_TRENDY['year'] = np.arange(1901,2019,1)
ax1.plot(df_TRENDY['year'], df_TRENDY['nbp'], color='r', lw=2.0,
         label='LPJ GUESS TRENDY v.8')
ax2.plot(df_TRENDY['year'], df_TRENDY['nbp'].cumsum(), color='r', lw=3.0,
         label='LPJ GUESS TRENDY v.8')
ax3.plot(df_TRENDY['year'], df_TRENDY['cVeg'], color='r', lw=3.0,
         label='LPJ GUESS TRENDY v.8')
ax4.plot(df_TRENDY['year'], df_TRENDY['cSoil'], color='r', lw=3.0,
         label='LPJ GUESS TRENDY v.8')

ax1.set_xticklabels([])
ax1.set_xlim(1970,2018)

ax2.axhline(linewidth=1, color='k', alpha=0.5)
ax2.set_title('Cumulative NBP Australia')
ax2.set_ylabel('Cumulative NBP [PgC]')
ax2.set_xticklabels([])

ax3.legend(loc='upper center', bbox_to_anchor=(1.1, -0.1), ncol=4)
ax3.set_title('Carbon stored in vegetation Australia')
ax3.set_ylabel('$\mathrm{C_{Veg}}$ [PgC]')

ax4.set_title('Carbon stored in soil Australia')
ax4.set_ylabel('$\mathrm{C_{Soil}}$ [PgC]')

text(0.04, 1.02, 'a)', ha='center',transform=ax1.transAxes, fontsize=14)
text(0.04, 1.02, 'b)', ha='center',transform=ax2.transAxes, fontsize=14)
text(0.04, 1.02, 'c)', ha='center',transform=ax3.transAxes, fontsize=14)
text(0.04, 1.02, 'd)', ha='center',transform=ax4.transAxes, fontsize=14)

plt.show()
