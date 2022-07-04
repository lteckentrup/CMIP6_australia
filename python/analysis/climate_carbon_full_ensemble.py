import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from matplotlib.gridspec import GridSpec
from scipy import signal
from read_in import (model_names_full,
                     cmap)

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

### Read in averages from csv
def read_csv(var):
    df = pd.read_csv('../LPJ_monthly_corrected/original_csv/'+var+'_full.csv',
                     index_col='Year')

    if var == 'temp':
        df = df-273.15
    elif var == 'NEE':
        df = df*(-1)

    return(df)

### Calculate 30 year moving average
def rolling_avg(var):
    df = read_csv('temp')
    df_anomaly = df-df[:30].mean(axis=0)
    rolling = df_anomaly.rolling(window=30,center=True).mean()
    return(rolling)

vars = ['prec', 'temp', 'CTotal']
axes=[ax2,ax4,ax6]

### Plot change in variables over time
for v,a in zip(vars,axes):
    rolling = rolling_avg(v)
    a.axhline(0, linewidth=1,  color='k', alpha=0.5)
    for mn, c in zip(model_names_full, cmap):
        if mn == 'CRUJRA':
            lw=3.0
        else:
            lw=2.0
        a.plot(rolling[mn], color=c, lw=lw, label=mn)

### Plot boxplots with CMIP6 spread of total values and IAV in variables
def boxplot_plot(var,ax,stat,position,facecolor):
    if stat=='avg':
        df = read_csv(var)
        df_boxplot = df[-30:].drop(columns=['CRUJRA']).mean(axis=0)
        scatter=df['CRUJRA'][-30:].mean()
    elif stat=='std':
        df = read_csv(var)
        df_notrend = df.apply(signal.detrend)
        df_boxplot = df_notrend[-30:].drop(columns=['CRUJRA']).std(axis=0)
        scatter=df_notrend['CRUJRA'][-30:].std()
        print(df_boxplot.min())
        print(df_boxplot.max())
        print(scatter)

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

    ax.scatter(position,scatter, marker='*',c='k',s=160,zorder=3,label='CRUJRA')

boxplot_plot('prec',ax1,'avg',0.5,'#088da5')
boxplot_plot('prec',ax1_twin,'std',1.5,'#ed1556')
boxplot_plot('temp',ax3,'avg',0.5,'#088da5')
boxplot_plot('temp',ax3_twin,'std',1.5,'#ed1556')
boxplot_plot('CTotal',ax5,'avg',0.5,'#088da5')
boxplot_plot('NEE',ax5_twin,'std',1.5,'#ed1556')

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

ax1.set_ylabel('$\mathrm{PPT_{\mu,1989-2018}}$ [mm]')
ax3.set_ylabel('$\mathrm{T_{\mu,1989-2018}}$ [$^\circ$C]')
ax5.set_ylabel('$\mathrm{C_{Total,\mu,1989-2018}}$ [PgC]')

ax1_twin.set_ylabel('PPT$_{\sigma,1989-2018}$ [mm yr$^{-1}$]')
ax3_twin.set_ylabel('T$_{\sigma,1989-2018}$ [$^\circ$C]')
ax5_twin.set_ylabel('$\mathrm{NBP_{\sigma,1989-2018}}$ [PgC]')

ax2.set_ylabel('$\Delta$ PPT [mm]')
ax4.set_ylabel('$\Delta$ T [$^\circ$C]')
ax6.set_ylabel('$\Delta \mathrm{C_{Total}}$ [PgC]')

ax2.legend(loc='upper center', bbox_to_anchor=(1.4, 0.7), ncol=1,
           frameon=False)
ax3.legend(loc='upper center', bbox_to_anchor=(4.305, -0.37), ncol=1,
           frameon=False)

fig.align_ylabels()

# plt.show()
plt.savefig('climate_carbon_full_ensemble.pdf')
