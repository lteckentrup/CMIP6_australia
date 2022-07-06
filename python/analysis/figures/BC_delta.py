import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from scipy import signal

fig=plt.figure(figsize=(7.5,9))

fig.subplots_adjust(hspace=0.17)
fig.subplots_adjust(wspace=0.15)
fig.subplots_adjust(right=0.95)
fig.subplots_adjust(left=0.1)
fig.subplots_adjust(bottom=0.19)
fig.subplots_adjust(top=0.95)

plt.rcParams['text.usetex']=False
plt.rcParams['axes.labelsize']=12
plt.rcParams['font.size']=11
plt.rcParams['legend.fontsize']=12
plt.rcParams['xtick.labelsize']=11
plt.rcParams['ytick.labelsize']=11

ax1=fig.add_subplot(3,2,1)
ax2=fig.add_subplot(3,2,2)
ax3=fig.add_subplot(3,2,3)
ax4=fig.add_subplot(3,2,4)
ax5=fig.add_subplot(3,2,5)
ax6=fig.add_subplot(3,2,6)

axes=[ax1,ax2,ax3,ax4,ax5,ax6]
axes_BC=[ax1,ax2,ax3,ax4,ax5]
color=['tab:blue', 'tab:orange','tab:green','tab:red','tab:purple','tab:brown',
       '#515151']
GCMs_bounding = ['INM-CM4-8', 'NorESM2-MM', 'MPI-ESM1-2-HR', 'KIOST-ESM',
                 'EC-Earth3-Veg']

def readin(var, method):
    if method == 'Reanalysis':
        df = pd.read_csv('../LPJ_monthly_corrected/original_csv/'+var+'_full.csv',
                         index_col='Year')
        return(df_CRUJRA)
    else:
        if method in ['original', 'Scaling', 'MAV', 'QM', 'CDF-t', 'R2D2', 'dOTC']:
            df = pd.read_csv('../LPJ_monthly_corrected/'+method+'_csv/'+var+'_full.csv',
                             index_col='Year')
        elif method == 'Weighting':
            df = pd.read_csv('../LPJ_ensemble_averages/'+method+'_csv/'+var+'.csv',
                             index_col='Year')
        else:
            df = pd.read_csv('../LPJ_ensemble_averages/'+method+'_csv/'+var+'.csv')
            df['Year'] = np.arange(1901,2019)
            df = df.set_index('Year')

        return(df)

def rolling_avg(df):
    df_anomaly = df-df[:30].mean(axis=0)
    rolling = df_anomaly.rolling(window=30,center=True).mean()
    return(rolling)

def spread(var):
    df = readin(var, 'original')

    rolling = rolling_avg(df)
    rolling['min'] = rolling.drop(columns=['CRUJRA']).min(axis=1)
    rolling['max'] = rolling.drop(columns=['CRUJRA']).max(axis=1)

    for ax in axes:
        label_cru='LG$_\mathrm{CRUJRA}$'
        label_spread='Raw ensemble\nspread'

        ax.fill_between(rolling.reset_index()['Year'], rolling['min'], 
                        rolling['max'], label=label_spread, color='tab:grey', 
                        alpha=0.15)
        ax.plot(rolling['CRUJRA'],label=label_cru,color='k',lw=3,ls='-')

def plot_BC(ax, GCM, var):
    BC_methods = ['original', 'Scaling', 'MAV', 'QM', 'CDF-t', 'dOTC', 'R2D2']
    position = [1,2.5,3.5,4.5,5.5,6.5,7.5]

    for mt,c,p in zip(BC_methods,color,position):
        df = readin(var, mt)
        rolling = rolling_avg(df[GCM])

        if mt == 'original':
            lw = 3
            ls = '-'
            label = 'Raw'
        else:
            lw = 1.5
            ls = '--'
            label=mt

        ax.plot(rolling,label=label,color=c,lw=lw,ls=ls)

### Plot ensemble averages
def plot_ENS(method, selection, color, var):
    df = readin(var, method)
    rolling = rolling_avg(df['full'])

    if method == 'Random_Forest':
        label = 'ENS$_\mathrm{RF}$'
    elif method == 'Weighted':
        label = 'ENS$_\mathrm{Weighted}$'
    elif method == 'Uniform':
        label = 'ENS$_\mathrm{Arithmetic,Full}$'

    ax6.plot(rolling,label=label,color=color,lw=1.5,ls='--')

ens_methods = ['Uniform', 'Weighted', 'Random_Forest']
ens_colors = ['tab:olive', 'tab:pink', 'tab:cyan']

var='prec'

### Plot change in in variable
spread(var)
for ax, mn in zip(axes_BC, GCMs_bounding):
    plot_BC(ax, mn, var)

for em, ec in zip(ens_methods, ens_colors):
    plot_ENS(em, 'full', ec, var)

axes_all = [ax1,ax2,ax3,ax4,ax5,ax6]
title_num = ['a)','b)','c)','d)','e)','f)']

if var == 'CTotal':
    ylabel = '$\Delta \mathrm{C_{Total}}$ [PgC]'
    title_name = ['LG$_\mathrm{INM-CM4-8}$',
                  'LG$_\mathrm{NorESM2-MM}$',
                  'LG$_\mathrm{MPI-ESM1-2-HR}$',
                  'LG$_\mathrm{KIOST-ESM}$',
                  'LG$_\mathrm{EC-Earth3-Veg}$',
                  'Ensemble Average']
else:
    if var == 'temp':
        ylabel = '$\Delta$ T [$^\circ$C]'
    elif var == 'prec':
        ylabel = '$\Delta$ PPT [mm]'

    title_name = ['INM-CM4-8',
                  'NorESM2-MM',
                  'MPI-ESM1-2-HR',
                  'KIOST-ESM',
                  'EC-Earth3-Veg',
                  'Ensemble Average']

for aa,tnu,tna in zip(axes_all,title_num,title_name):
    aa.set_title(tnu, loc='left')
    aa.set_title(tna)

    aa.spines['right'].set_visible(False)
    aa.spines['top'].set_visible(False)

    aa.axhline(color='k', alpha=0.5,lw=0.5)

for a in (ax1,ax2,ax3,ax4):
    a.set_xticklabels([])

for a in (ax1,ax3,ax5):
    a.set_ylabel(ylabel)

custom_markers = [Patch(facecolor='tab:grey', edgecolor='tab:grey', alpha=0.15),
                  Line2D([0], [0], linestyle='-', lw=3, color='k'),
                  Line2D([0], [0], linestyle='-', lw=3, color='tab:blue'),
                  Line2D([0], [0], linestyle='--', lw=1.5, color='tab:orange'),
                  Line2D([0], [0], linestyle='--', lw=1.5, color='tab:green'),
                  Line2D([0], [0], linestyle='--', lw=1.5, color='tab:red'),
                  Line2D([0], [0], linestyle='--', lw=1.5, color='tab:purple'),
                  Line2D([0], [0], linestyle='--', lw=1.5, color='tab:brown'),
                  Line2D([0], [0], linestyle='--', lw=1.5, color='tab:grey'),
                  Line2D([0], [0], linestyle='--', lw=1.5, color='tab:olive'),
                  Line2D([0], [0], linestyle='--', lw=1.5, color='tab:pink'),
                  Line2D([0], [0], linestyle='--', lw=1.5, color='tab:cyan')]

ax5.legend(custom_markers, ['Raw ensemble spread',
                            'LG$_\mathrm{CRUJRA}$',
                            'Raw',
                            'Scaling',
                            'MAV',
                            'QM',
                            'CDF-t',
                            'R2D2',
                            'dOTC',
                            'ENS$_\mathrm{Arithmetic,Full}$',
                            'ENS$_\mathrm{Weighted}$',
                            'ENS$_\mathrm{RF}$'],

           loc='upper center', bbox_to_anchor=(1.0, -0.25),ncol=4, frameon=False)

fig.align_ylabels()

plt.show()
# plt.savefig('BC_delta.pdf')
