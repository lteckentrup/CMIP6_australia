import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from read_in import GCM_names_bounding
from matplotlib.gridspec import GridSpec
from scipy import signal

fig=plt.figure(figsize=(7.5,9))

fig.subplots_adjust(hspace=0.2)
fig.subplots_adjust(wspace=0.4)
fig.subplots_adjust(right=0.95)
fig.subplots_adjust(left=0.15)
fig.subplots_adjust(bottom=0.2)
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

color=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown',
       '#515151']
GCM_skill = ['GFDL-CM4', 'GFDL-ESM4', 'KIOST-ESM', 'MPI-ESM1-2-HR',
                'MPI-ESM1-2-LR', 'MRI-ESM2-0']
GCM_independence = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'CESM2-WACCM',
                       'CMCC-CM2-SR5', 'MIROC6', 'MPI-ESM1-2-HR', 'NESM3',
                       'NorESM2-MM']
GCM_bounding = ['EC-Earth3-Veg', 'INM-CM4-8', 'KIOST-ESM', 'MPI-ESM1-2-HR',
                   'NorESM2-MM']

BC_methods = ['original', 'Scaling', 'MAV', 'QM', 'CDF-t', 'R2D2', 'dOTC']

Selection_methods = ['Full', 'Skill', 'Independence']
selection_markers = ['o', 'D', '_']

for a in (ax1,ax3,ax5):
    a.axhline(color='k', alpha=0.5,lw=0.5)

for a in (ax1,ax2,ax3,ax4,ax5,ax6):
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.axvline(x=1.75,color='k', alpha=0.5)
    a.axvline(x=8.25,color='k', alpha=0.5)
    a.set_xticks([1,2.5,3.5,4.5,5.5,6.5,7.5,9,10,11])
    a.set_xticklabels([])

def readin_CRU(var):
    df = pd.read_csv('../LPJ_monthly_corrected/original_csv/'+var+'_full.csv',
                     index_col='Year')
    df_CRUJRA = df['CRUJRA'][-30:].mean()
    return(df_CRUJRA)

def readin_BC(method, var):
    df = pd.read_csv('../LPJ_monthly_corrected/'+method+'_csv/'+var+'_full.csv',
                     index_col='Year')
    if method == 'original':
        df = df.drop(columns=['CRUJRA'])

    return(df)

def readin_ENS(method, var):
    if method == 'Weighting':
        df = pd.read_csv('../LPJ_ensemble_averages/'+method+'_csv/'+var+'.csv',
                         index_col='Year')
    else:
        df = pd.read_csv('../LPJ_ensemble_averages/'+method+'_csv/'+var+'.csv')

    return(df)

def calc_bias_cv(method, selection, var):
    df_CRUJRA = readin_CRU(var)
    df = readin_BC(method, var)

    if method == 'original':
        if selection == 'Full':
            df = df
        elif selection == 'Skill':
            df = df[GCM_skill]
        elif selection == 'Independence':
            df = df[GCM_independence]
    else:
        df = df[GCM_names_bounding]

    ### Calculate bias
    df_diff = df[-30:].mean()-df_CRUJRA
    ### Calculate ensemble average of bias
    df_ens_diff = df_diff.mean()
    ### Calculate coefficient of variance over ensemble
    df_CV = df.std(axis=1)/df.mean(axis=1)
    ### Average coefficient of variance over the last 30 years
    df_ens_CV = df_CV[-30:].mean()

    return(df_diff, df_ens_diff, df_ens_CV)

def plot_BC(ax_GCM, ax_CV, GCM, var, marker, addition):
    bias_GCM = []
    bias_ENS = []
    CV_ENS = []

    ### Create list with values for GCM bias, ensemble mean bias, and ensemble CV
    for bcm in BC_methods:
        bias_gcm, bias_ens, cv_ens = calc_bias_cv(bcm, 'Full', var)
        bias_GCM.append(bias_gcm[GCM])
        bias_ENS.append(bias_ens)
        CV_ENS.append(cv_ens)

    pos_BC = [1,2.5,3.5,4.5,5.5,6.5,7.5]

    ### Plot individual GCM bias
    ax_GCM.scatter(np.array(pos_BC)+addition,bias_GCM,color=color,marker=marker)

    ### Plot ensemble bias and CV
    if GCM=='EC-Earth3-Veg':
        ax_GCM.scatter(pos_BC,bias_ENS,color='k',marker='h',zorder=10)
        ax_CV.scatter(pos_BC,CV_ENS,color=color,marker='h',zorder=10)

def plot_ENS_sel(ax_ENS, ax_CV, var):
    bias_ENS = []
    CV_ENS = []

    for sm in Selection_methods:
        _, bias_ens, cv_ens = calc_bias_cv('original', sm, var)
        bias_ENS.append(bias_ens)
        CV_ENS.append(cv_ens)

    pos_ENS = [8.8,9,9.2]
    facecolors='none'
    for i, sm in zip(range(0,3), selection_markers):
        if i == 0:
            ax_ENS.scatter(pos_ENS[i], bias_ENS[i], color='tab:olive',
                           marker=sm, facecolors='none')
            ax_CV.scatter(pos_ENS[i], CV_ENS[i], color='tab:olive',
                           marker=sm, facecolors='none')
        else:
            ax_ENS.scatter(pos_ENS[i], bias_ENS[i], color='tab:olive',
                           marker=sm)
            ax_CV.scatter(pos_ENS[i], CV_ENS[i], color='tab:olive',
                           marker=sm)

def plot_ENS_method(var,method, color, position, axis):
    df_CRUJRA = readin_CRU(var)
    df = readin_ENS(method, var)

    bias_ENS = df['full'][-30:].mean() - df_CRUJRA
    axis.scatter(position,bias_ENS,color=color, marker='o',facecolors='none')

bc_methods = ['Raw', 'Scaling', 'MAV', 'QM', 'CDFt', 'R2D2', 'dOTC',
              'ENS$_\mathrm{Arithmetic}$', 'ENS$_\mathrm{Weighted}$',
              'ENS$_\mathrm{RF}$']

marker = ['o', 'x', 's', '+', 'v']
addition = [0.3,-0.3,0,0.1,-0.1]

for mn, m, a in zip(GCM_names_bounding, marker, addition):
    plot_BC(ax1,ax2, mn, 'prec', m, a)

for mn, m, a in zip(GCM_names_bounding, marker, addition):
    plot_BC(ax3,ax4, mn, 'temp', m, a)

for mn, m, a in zip(GCM_names_bounding, marker, addition):
    plot_BC(ax5,ax6, mn, 'CTotal', m, a)

plot_ENS_sel(ax1,ax2, 'prec')
plot_ENS_sel(ax3,ax4, 'temp')
plot_ENS_sel(ax5,ax6, 'CTotal')

vars = ['prec', 'temp', 'CTotal']
axes = [ax1,ax3,ax5]

for v,a in zip(vars,axes):
    plot_ENS_method(v, 'Weighted', 'tab:pink', 10, a)
    plot_ENS_method(v', 'Random_Forest', 'tab:cyan', 11, a)

ax5.set_xticklabels(bc_methods,rotation = 45, ha='right')
ax6.set_xticklabels(bc_methods,rotation = 45, ha='right')

ax1.set_ylabel('$\mathrm{\Delta PPT_{\mathrm{\mu,CMIP6-CRUJRA}}}$ [mm]')
ax2.set_ylabel('CV PPT [mm]')
ax3.set_ylabel('$\mathrm{\Delta T_{\mathrm{\mu,CMIP6-CRUJRA}}}$ [$^\circ$C]')
ax4.set_ylabel('CV T [$^\circ$C]')
ax5.set_ylabel('$\mathrm{\Delta C_{\mathrm{Total,\mu,CMIP6-CRUJRA}}}$ [PgC]')
ax6.set_ylabel('CV $\mathrm{C_{\mathrm{Total}}}$ [PgC]')

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

ax6.legend(custom_markers, ['LG$_\mathrm{EC-Earth3-Veg}$',
                            'LG$_\mathrm{INM-CM4-8}$',
                            'LG$_\mathrm{KIOST-ESM}$',
                            'LG$_\mathrm{MPI-ESM1-2-HR}$',
                            'LG$_\mathrm{NorESM2-MM}$',
                            'Full',
                            'Skill',
                            'Independence',
                            'Bounding'],

           loc='upper center', bbox_to_anchor=(-0.4, -0.4), ncol=3, frameon=False)

labels = ['a)','b)','c)','d)','e)','f)']
axes = [ax1,ax2,ax3,ax4,ax5,ax6]

for a,l in zip(axes,labels):
    a.set_title(l,loc='left')

ax1.set_title('Bias\n')
ax2.set_title('Coefficient of variation\n')

fig.align_ylabels()
# plt.show()
plt.savefig('BC_scatter.pdf')
