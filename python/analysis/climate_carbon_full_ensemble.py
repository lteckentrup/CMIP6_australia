import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import pandas as pd
from pylab import text
import glob
from matplotlib.pyplot import cm
import matplotlib.ticker as ticker
import xarray as xr
from matplotlib.gridspec import GridSpec
from read_in import model_names

fig = plt.figure(figsize=(9,9))

fig.subplots_adjust(hspace=0.22)
fig.subplots_adjust(wspace=11.0)
fig.subplots_adjust(right=0.75)
fig.subplots_adjust(left=0.1)
fig.subplots_adjust(bottom=0.05)
fig.subplots_adjust(top=0.95)

plt.rcParams['text.usetex'] = False
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.size'] = 11
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

gs=GridSpec(3,5)

ax1=fig.add_subplot(gs[0,:2])
ax2=fig.add_subplot(gs[0,2:])
ax3=fig.add_subplot(gs[1,:2])
ax4=fig.add_subplot(gs[1,2:])
ax5=fig.add_subplot(gs[2,:2])
ax6=fig.add_subplot(gs[2,2:])

ax1_twin = ax1.twinx()
ax3_twin = ax3.twinx()
ax5_twin = ax5.twinx()

df_temp_K = pd.read_csv('../LPJ_monthly_corrected/original_csv/temp_full.csv',
                        index_col='Year')
df_prec = pd.read_csv('../LPJ_monthly_corrected/original_csv/prec_full.csv',
                      index_col='Year')
df_CTotal = pd.read_csv('../LPJ_monthly_corrected/original_csv/CTotal_full.csv',
                      index_col='Year')
df_NEE = pd.read_csv('../LPJ_monthly_corrected/original_csv/NEE_full.csv',
                      index_col='Year')

df_temp = df_temp_K - 273.15
df_NBP = df_NEE * (-1)

def rolling_avg(df):
    df_anomaly = df-df[:30].mean(axis=0)
    rolling = df_anomaly.rolling(window=30,center=True).mean()
    return(rolling)

color_20=cm.tab20(np.arange(0,20,1))
color_add=cm.tab20b([0])
black = np.array([0,0,0,1], ndmin=2)

color=np.vstack((color_20,color_add,black))

dataframes = [df_temp,df_prec,df_CTotal]
axes=[ax2,ax4,ax6]

for df,a in zip(dataframes,axes):
    rolling = rolling_avg(df)
    a.axhline(0, linewidth=1,  color='k', alpha=0.5)
    for mn, c in zip(model_names, color):
        if mn == 'CRUJRA':
            lw=3.0
        else:
            lw=2.0
        a.plot(rolling[mn], color=c, lw=lw, label=mn)

def boxplot_plot(df,ax,stat,position,facecolor):
    if stat=='avg':
        df_boxplot = df[-30:].drop(columns=['CRUJRA']).mean(axis=0)
    elif stat=='std':
        df_boxplot = df[-30:].drop(columns=['CRUJRA']).std(axis=0)

    boxplot=ax.boxplot(df_boxplot,
                       positions=[position],
                       patch_artist=True,
                       widths = .5,
                       medianprops = dict(linestyle='-',
                                          linewidth=2,
                                          color='Yellow'),
                       whiskerprops = dict(linestyle='-',
                                           linewidth=1.5,
                                           color='k'),
                       capprops = dict(linestyle='-',
                                       linewidth=1.5,
                                       color='k'),
                       boxprops = dict(linestyle='-',
                                       linewidth=2,
                                       color='Black',
                                       facecolor=facecolor,
                                       alpha=.7))

    if stat=='avg':
        scatter=df['CRUJRA'][-30:].mean()
    elif stat=='std':
        scatter=df['CRUJRA'][-30:].std()

    ax.scatter(position,scatter, marker='*',c='k',s=160,zorder=3,label='CRUJRA')

boxplot_plot(df_temp,ax1,'avg',0.5,'#088da5')
boxplot_plot(df_temp,ax1_twin,'std',1.5,'#ed1556')
boxplot_plot(df_prec,ax3,'avg',0.5,'#088da5')
boxplot_plot(df_prec,ax3_twin,'std',1.5,'#ed1556')
boxplot_plot(df_CTotal,ax5,'avg',0.5,'#088da5')
boxplot_plot(df_NBP,ax5_twin,'std',1.5,'#ed1556')

for a in (ax1,ax2,ax3,ax4):
    ax1.set_xticklabels([])

ax5_twin.set_xticklabels(['Avg', 'IAV'])
    
for a in (ax1,ax3,ax5):
    a.axvline(1, linewidth=1,  color='k', alpha=0.5)

titles=['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
axes=[ax1,ax2,ax3,ax4,ax5,ax6]

for a, t in zip(axes, titles):
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.set_title(t, loc = 'left')

for at in (ax1_twin,ax3_twin,ax5_twin):
    at.spines['top'].set_visible(False)

ax1.set_ylabel('$\mathrm{T_{\mu,1989-2018}}$ [$^\circ$C]')
ax3.set_ylabel('$\mathrm{PPT_{\mu,1989-2018}}$ [mm]')
ax5.set_ylabel('$\mathrm{C_{Total,\mu,1989-2018}}$ [PgC]')

ax1_twin.set_ylabel('T$_{\sigma,1989-2018}$ [$^\circ$C]')
ax3_twin.set_ylabel('PPT$_{\sigma,1989-2018}$ [mm yr$^{-1}$]')
ax5_twin.set_ylabel('$\mathrm{NBP_{\sigma,1989-2018}}$ [PgC]')

ax2.set_ylabel('$\Delta$ T [$^\circ$C]')
ax4.set_ylabel('$\Delta$ PPT [mm]')
ax6.set_ylabel('$\Delta \mathrm{C_{Total}}$ [PgC]')

ax2.legend(loc='upper center', bbox_to_anchor=(1.4, 0.7), ncol=1, frameon=False)
ax3.legend(loc='upper center', bbox_to_anchor=(4.305, -0.37), ncol=1, frameon=False)


plt.show()
# plt.savefig('climate_carbon_change.pdf')
