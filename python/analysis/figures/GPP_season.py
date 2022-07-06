import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.pyplot import cm
from matplotlib.patches import Rectangle
from matplotlib.pyplot import cm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

fig=plt.figure(figsize=(8,9))

fig.subplots_adjust(hspace=0.2)
fig.subplots_adjust(wspace=0.15)
fig.subplots_adjust(right=0.98)
fig.subplots_adjust(left=0.10)
fig.subplots_adjust(bottom=0.15)
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

methods=['original', 'Scaling', 'MAV', 'QM', 'CDF-t', 'R2D2', 'dOTC']
axes=[ax1,ax2,ax3,ax4,ax5]
GCMs_extremes = ['EC-Earth3-Veg', 'INM-CM4-8', 'KIOST-ESM',
                    'MPI-ESM1-2-HR','NorESM2-MM']

xlabels=['N', 'D', 'J', 'F', 'M', 'A','M', 'J', 'J', 'A', 'S', 'O']

color_20=cm.tab20(np.arange(0,20,1))
color_add=cm.tab20b([0,2,4])
color=np.vstack((color_20,color_add))

def readin(method):
    if method in ('original', 'Scaling', 'MAV', 'QM', 'CDF-t', 'R2D2', 'dOTC'):
        df = pd.read_csv('../LPJ_monthly_corrected/'+method+'_csv/mpgpp_C4G_full.csv')
    else:
        df = pd.read_csv('../LPJ_ensemble_averages/'+method+'_csv/mpgpp_C4G.csv')

    return(df)

idx=[10,11,0,1,2,3,4,5,6,7,8,9]

color=['tab:blue','tab:orange','tab:green','tab:red','tab:purple',
       'tab:brown','#515151']

def plot_wurst(var, veg_type):
    if var == 'mpgpp':
        label = 'GPP [PgC mon$^{-1}$]'
    elif var == 'mpnpp':
        label = 'NPP [PgC mon$^{-1}$]'
    elif var == 'mplai':
        label = 'LAI [m$^2$ m$^{-2}$]'

    ax1.set_ylabel(label)
    ax3.set_ylabel(label)
    ax5.set_ylabel(label)

    for gcm, ax in zip(GCMs_extremes, axes):
        if gcm == 'NorESM2-MM':
            label = 'LG$_\mathrm{CRUJRA}$'
        else:
            label = '_nolegend_'

        for mt, c in zip(methods,color):
            if mt == 'original':
                lw=3.0
                ls='-'
            else:
                lw=1.5
                ls='--'
            df=readin(mt)

            df_new=pd.DataFrame()
            df_new[gcm]=df[gcm]
            df_new['month']=np.arange(1,13)
            if mt == 'MVA':
                label='MAV'
            elif mt == 'CDFt':
                label='CDF-t'
            elif mt == 'original':
                label = 'Raw'
            else:
                label=mt
            ax.plot(df_new['month'], df_new[gcm][idx], label=label, color=c,
                    lw=lw,ls=ls,zorder=10)
        else:
            pass
        ax.set_xticks(np.arange(1,13))

        if ax in (ax1,ax2,ax3,ax4):
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(xlabels)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        rect_wet = Rectangle((1,0),6,25, edgecolor='none', facecolor='tab:blue', alpha=0.2)
        rect_dry = Rectangle((7,0),6,25, edgecolor='none', facecolor='tab:red', alpha=0.2)
        ax.add_patch(rect_wet)
        ax.add_patch(rect_dry)
        ax.set_ylim(-0.01,0.39)

df_original=readin('original')
df_original['month']=np.arange(1,13)

df_original['min'] = df_original.drop(columns=['month', 'CRUJRA']).min(axis=1)
df_original['max'] = df_original.drop(columns=['month', 'CRUJRA']).max(axis=1)

for a in (ax1,ax2,ax3,ax4,ax5,ax6):
    if a == ax5:
        label_cru = 'LG$_\mathrm{CRUJRA}$'
        label_spread = 'Raw ensemble Spread'
    else:
        label_cru = '_nolegend_'
        label_spread = '_nolegend_'

    a.plot(df_original['month'], df_original['CRUJRA'][idx], label=label_cru,
           color='k',lw=3.0,ls='-',zorder=10)
    a.fill_between(df_original['month'], df_original['min'][idx],df_original['max'][idx],
                   color='tab:grey',alpha=0.2,label=label_spread)


df_uniform = readin('Uniform')
df_weighted = readin('Weighted')
df_rf = readin('Random_Forest')

ax6.plot(df_uniform['month'], df_uniform['full'][idx],
         label='ENS$_\mathrm{Arithmetic,Full}$',
         color='tab:olive',
         ls='--',
         lw=1.5,
         zorder=10)
ax6.plot(df_weighted['month'],
         df_weighted['full'][idx],
         label='ENS$_\mathrm{Weighted}$',
         color='tab:pink',
         ls='--',
         lw=1.5,
         zorder=10)
ax6.plot(df_rf['month'],
         df_rf['full'][idx],
         label='ENS$_\mathrm{RF}$',
         color='tab:cyan',
         ls='--',
         lw=1.5,
         zorder=10)

ax6.set_xticks(np.arange(1,13))
ax6.set_xticklabels(xlabels)

rect_wet = Rectangle((1,0),6,25, edgecolor='none', facecolor='tab:blue', alpha=0.2)
rect_dry = Rectangle((7,0),6,25, edgecolor='none', facecolor='tab:red', alpha=0.2)
ax6.add_patch(rect_wet)
ax6.add_patch(rect_dry)

ax6.set_ylim(-0.01,0.39)

plot_wurst('mpgpp', 'C4G')

ax1.set_title('a)', loc='left')
ax2.set_title('b)', loc='left')
ax3.set_title('c)', loc='left')
ax4.set_title('d)', loc='left')
ax5.set_title('e)', loc='left')
ax6.set_title('f)', loc='left')

ax1.set_title('LG$_\mathrm{EC-Earth3-Veg}$')
ax2.set_title('LG$_\mathrm{INM-CM4-8}$')
ax3.set_title('LG$_\mathrm{KIOST-ESM}$')
ax4.set_title('LG$_\mathrm{MPI-ESM1-2-HR}$')
ax5.set_title('LG$_\mathrm{NorESM2-MM}$')
ax6.set_title('Ensemble average')

ax5.spines['right'].set_visible(False)
ax5.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.spines['top'].set_visible(False)

ax6.legend(loc='best', ncol=1, frameon=False)
ax5.legend(loc='upper center', bbox_to_anchor=(1.0, -0.15), ncol=3, frameon=False)


ax5.set_xticks(np.arange(1,13))
ax5.set_xticklabels(xlabels)

ax5.set_ylim(-0.01,0.39)

ax2.set_yticklabels([])
ax4.set_yticklabels([])
ax6.set_yticklabels([])

# plt.show()
plt.savefig('gpp_seasonal_C4G.pdf')
# plt.savefig('gpp_seasonal_C4G_BC_monthly.pdf')
