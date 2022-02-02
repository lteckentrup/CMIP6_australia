import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from read_in import model_names_bounding
from matplotlib.gridspec import GridSpec

fig=plt.figure(figsize=(10,12))

fig.subplots_adjust(hspace=0.8)
fig.subplots_adjust(wspace=0.28)
fig.subplots_adjust(right=0.8)
fig.subplots_adjust(left=0.10)
fig.subplots_adjust(bottom=0.05)
fig.subplots_adjust(top=0.95)

plt.rcParams['text.usetex']=False
plt.rcParams['axes.labelsize']=12
plt.rcParams['font.size']=11
plt.rcParams['legend.fontsize']=12
plt.rcParams['xtick.labelsize']=11
plt.rcParams['ytick.labelsize']=11

gs=GridSpec(17,2)

ax1=fig.add_subplot(gs[0:4,:1])
ax2=fig.add_subplot(gs[0:4,1:])
ax3=fig.add_subplot(gs[5:9,:1])
ax4=fig.add_subplot(gs[5:9,1:])
ax5=fig.add_subplot(gs[9:13,:1])
ax6=fig.add_subplot(gs[9:13,1:])
ax7=fig.add_subplot(gs[13:,:1])
ax8=fig.add_subplot(gs[13:,1:])

axes=[ax3,ax4,ax5,ax6,ax7,ax8]

color=['tab:blue', 'tab:orange','tab:green','tab:red','tab:purple','tab:brown',
       '#515151']

def rolling_avg(df):
    df_anomaly = df-df[:30].mean(axis=0)
    rolling = df_anomaly.rolling(window=30,center=True).mean()
    return(rolling)

df_CTotal = pd.read_csv('../LPJ_monthly_corrected/original_csv/CTotal_full.csv',
                        index_col='Year')
df_NEE = pd.read_csv('../LPJ_monthly_corrected/original_csv/NEE_full.csv',
                     index_col='Year')

df_CTotal_CRUJRA = df_CTotal['CRUJRA'][-30:].mean()
df_NEE_CRUJRA = df_NEE['CRUJRA'][-30:].std()

rolling = rolling_avg(df_CTotal)
rolling['min'] = rolling.drop(columns=['CRUJRA']).min(axis=1)
rolling['max'] = rolling.drop(columns=['CRUJRA']).max(axis=1)

for ax in axes:
    if ax == ax8:
        label_cru='_nolegend_'
        label_spread='_nolegend_'
    else:
        label_cru='CRUJRA'
        label_spread='Model spread'

    ax.fill_between(rolling.reset_index()['Year'], rolling['min'], rolling['max'],
                    label=label_spread, color='tab:grey', alpha=0.15)
    ax.plot(rolling['CRUJRA'],label=label_cru,color='k',lw=3,ls='-')

def plot_wurst_BC(ax, model,marker, addition):
    BC_methods = ['original', 'Scaling', 'MAV', 'QM', 'CDF-t', 'dOTC', 'MRec']
    position = [1,2.5,3.5,4.5,5.5,6.5,7.5]

    for mt,c,p in zip(BC_methods,color,position):
        df_CTotal = pd.read_csv('../LPJ_monthly_corrected/'+mt+'_csv/CTotal_full.csv',
                                index_col='Year')
        df_NEE = pd.read_csv('../LPJ_monthly_corrected/'+mt+'_csv/NEE_full.csv',
                             index_col='Year')

        df_CTotal_diff = df_CTotal[model][-30:].mean()-df_CTotal_CRUJRA
        df_NEE_diff = df_NEE[model][-30:].std()-df_NEE_CRUJRA

        rolling = rolling_avg(df_CTotal[model])

        if mt == 'original':
            lw = 3
            ls = '-'
        else:
            lw = 1.5
            ls = '--'

        ax.plot(rolling,label=mt,color=c,lw=lw,ls=ls)
        ax1.scatter(p+addition,df_CTotal_diff,color=c,marker=marker)
        ax2.scatter(p+addition,df_NEE_diff,color=c,marker=marker)

axes_BC = [ax3,ax4,ax5,ax6,ax7]
marker = ['o', 'x', 's', '+', 'v']
addition = [0.3,-0.3,0,0.1,-0.1]

for ax, mn, m, a in zip(axes_BC, model_names_bounding, marker, addition):
    plot_wurst_BC(ax, mn, m, a)

### Plot ensemble averages
def plot_wurst_ens(method, selection, color, position, marker):
    df_CTotal = pd.read_csv('../LPJ_ensemble_averages/'+method+'_csv/CTotal.csv',
                            index_col='Year')
    df_NEE = pd.read_csv('../LPJ_ensemble_averages/'+method+'_csv/NEE.csv',
                            index_col='Year')

    df_CTotal_diff = df_CTotal[-30:].mean()-df_CTotal_CRUJRA
    df_NEE_diff = df_NEE[-30:].std()-df_NEE_CRUJRA

    print(df_CTotal_diff)
    print(df_NEE)

    rolling = rolling_avg(df_CTotal[selection])

    if method == 'Random_Forest':
        label = 'Random Forest'
    else:
        label = method

    ax8.plot(rolling,label=label,color=color,lw=1.5,ls='--')

    selection_methods = ['full', 'skill', 'independence', 'bounding']
    selection_markers = ['o', 'D', '_', 'h']

    for sme, sma in zip(selection_methods, selection_markers):
        if sme == 'full':
            ax1.scatter(position,df_CTotal_diff[sme],color=color,
                        marker=sma,facecolors='none')
            ax2.scatter(position,df_NEE_diff[sme],color=color,
                        marker=sma,facecolors='none')
        else:
            if method == 'Weighted':
                pass
            else:
                if sme in ('skill', 'independence'):
                    addition=-0.2
                else:
                    addition=0.2
                ax1.scatter(position+addition,df_CTotal_diff[sme],
                            color=color,marker=sma)
                ax2.scatter(position+addition,df_NEE_diff[sme],
                            color=color,marker=sma)

ens_methods = ['Uniform', 'Weighted', 'Random_Forest']
ens_colors = ['tab:olive', 'tab:pink', 'tab:cyan']
ens_pos = [9, 10, 11]

for em, ec, ep in zip(ens_methods, ens_colors, ens_pos):
    plot_wurst_ens(em, 'full', ec, ep, 'o')

axes_all = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]
title_num = ['a)','b)','c)','d)','e)','f)','g)','h)']
title_name = ['Average $\mathrm{C_{Total}}$', 'IAV NBP', 'EC-Earth3-Veg',
              'INM-CM4-8', 'KIOST-ESM', 'MPI-ESM1-2-HR', 'NorESM2-MM',
              'Ensemble average']

for aa,tnu,tna in zip(axes_all,title_num,title_name):
    aa.set_title(tnu, loc='left')
    aa.set_title(tna)

    aa.spines['right'].set_visible(False)
    aa.spines['top'].set_visible(False)

    aa.axhline(color='k', alpha=0.5,lw=0.5)

for a in (ax3,ax5,ax7):
    a.set_ylabel('$\Delta \mathrm{C_{Total}}$ [PgC]')

ax1.set_ylabel('$\mathrm{\Delta C_{\mathrm{Total,\mu,CMIP6-CRUJRA}}}$ [PgC]')
ax2.set_ylabel('$\mathrm{\Delta NBP_{\mathrm{\sigma,CMIP6-CRUJRA}}}$ [PgC]')

ax6.legend(loc='upper center', bbox_to_anchor=(1.3, 1.7), ncol=1, frameon=False)
ax8.legend(loc='upper center', bbox_to_anchor=(1.32, 0.8), ncol=1, frameon=False)

custom_markers = [Line2D([0], [0], marker='o', linestyle='', color='k'),
                  Line2D([0], [0], marker='x', linestyle='', color='k'),
                  Line2D([0], [0], marker='s', linestyle='', color='k'),
                  Line2D([0], [0], marker='+', linestyle='', color='k'),
                  Line2D([0], [0], marker='v', linestyle='', color='k'),
                  Line2D([0], [0], marker='o', markerfacecolor='none',
                         linestyle='', color='k'),
                  Line2D([0], [0], marker='D', linestyle='', color='k'),
                  Line2D([0], [0], marker='_', linestyle='', color='k'),
                  Line2D([0], [0], marker='h', linestyle='', color='k')]

ax2.legend(custom_markers, ['EC-Earth3-Veg', 'INM-CM4-8', 'KIOST-ESM',
                            'MPI-ESM1-2-HR', 'NorESM2-MM', 'Full', 'Skilled',
                            'Independence', 'Bounding'],
           loc='upper center', bbox_to_anchor=(1.33, 1.1),frameon=False)

bc_methods = ['original', 'SCALING', 'MAV', 'QM', 'CDFt', 'dOTC', 'MRec',
              'Uniform', 'I+S', 'RF']

for a in (ax3,ax4,ax5,ax6):
    a.set_xticklabels([])
for a in (ax1, ax2):
    a.set_xticks([1,2.5,3.5,4.5,5.5,6.5,7.5,9,10,11])
    a.set_xticklabels(bc_methods,rotation = 45, ha='right')

    a.axvline(x=1.75,color='k', alpha=0.5)
    a.axvline(x=8.25,color='k', alpha=0.5)

fig.align_ylabels()

# plt.show()
plt.savefig('carbon_BC_ens.pdf')
