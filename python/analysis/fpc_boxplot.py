import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import xarray as xr
from matplotlib.gridspec import GridSpec
sns.set_theme(style='ticks')

fig = plt.figure(figsize=(11,9))

plt.rcParams['text.usetex'] = False
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.size'] = 11
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

gs=GridSpec(2,19,hspace=0.2,wspace=0.0,right=0.98,left=0.08,bottom=0.25,top=0.97)

ax1=fig.add_subplot(gs[0,:6])
ax2=fig.add_subplot(gs[0,6:7])
ax3=fig.add_subplot(gs[0,7:8])
ax4=fig.add_subplot(gs[0,10:16])
ax5=fig.add_subplot(gs[0,16:17])
ax6=fig.add_subplot(gs[0,17:18])

ax7=fig.add_subplot(gs[1,:6])
ax8=fig.add_subplot(gs[1,6:7])
ax9=fig.add_subplot(gs[1,7:8])

ax10=fig.add_subplot(gs[1,10:16])
ax11=fig.add_subplot(gs[1,16:17])
ax12=fig.add_subplot(gs[1,17:18])

def readin(model,method,selection):
    if method == 'CRUJRA':
        fname = ('../reanalysis/CTRL/CRUJRA/fpc_LPJ-GUESS_1901-2018.nc')
    elif method == 'Weighted':
        fname = ('../LPJ_ensemble_averages/Weighted/fpc_weighted_1850-2100.nc')
    elif method in ('Uniform', 'Random_Forest'):
        fname = ('../LPJ_ensemble_averages/'+method+'/fpc_'+selection+'_1850-2100.nc')
    elif method in ('original', 'QM', 'CDFt', 'MRec'):
        fname = ('../LPJ_monthly_corrected/'+method+'/'+model+'/fpc_'+model+'_1850-2100.nc')
    elif method in ('SCALING', 'MVA'):
        fname = ('../LPJ_monthly_corrected/'+method+'/'+model+'/fpc_'+model+'_1851-2100.nc')
    elif method == 'dOTC':
        fname = ('../LPJ_monthly_corrected/'+method+'/'+model+'/fpc_'+model+'_1851-2025.nc')

    ds = xr.open_dataset(fname)
    ds = ds.sel(Time='2018')

    Temperate = ds['TeNE']+ds['TeBS']+ds['IBS']+ds['TeBE']
    Tropical = ds['TrBE']+ds['TrIBE']+ds['TrBR']
    C3G = ds['C3G']
    C4G = ds['C4G']

    return(Temperate.values.flatten(),
           Tropical.values.flatten(),
           C3G.values.flatten(),
           C4G.values.flatten())

methods = ['original', 'SCALING', 'MVA', 'QM', 'CDFt', 'dOTC', 'MRec']
methods_legend = ['original', 'Scaling', 'MAV', 'QM', 'CDF-t', 'dOTC',
                  'MRec', 'Uniform', 'Weighted', 'Random Forest']
methods_sel=['full', 'skill', 'independence', 'bounding']

### Generate dataframes for bias corrected models
def generate_dataframe(model):
    df_temperate = pd.DataFrame()
    df_tropical = pd.DataFrame()
    df_C3G = pd.DataFrame()
    df_C4G = pd.DataFrame()

    for m in methods:
        Temperate, Tropical, C3G, C4G = readin(model, m, '')
        df_temperate[m] = pd.Series(Temperate)
        df_tropical[m] = pd.Series(Tropical)
        df_C3G[m] = pd.Series(C3G)
        df_C4G[m] = pd.Series(C4G)

    return(df_temperate.assign(Model=model),
           df_tropical.assign(Model=model),
           df_C3G.assign(Model=model),
           df_C4G.assign(Model=model))

### Generate grouped dataframes for bias corrected models
def grouped_dataframes(index):
    df = pd.concat([generate_dataframe('EC-Earth3-Veg')[index],
                    generate_dataframe('INM-CM4-8')[index],
                    generate_dataframe('KIOST-ESM')[index],
                    generate_dataframe('MPI-ESM1-2-HR')[index],
                    generate_dataframe('NorESM2-MM')[index]])

    df_long = pd.melt(df, 'Model', var_name='Method',value_name='FPC')
    return(df_long.replace(0, np.nan))

### Generate dataframes for ensemble averages
def generate_dataframe_ens(method):
    df_temperate = pd.DataFrame()
    df_tropical = pd.DataFrame()
    df_C3G = pd.DataFrame()
    df_C4G = pd.DataFrame()

    for m in methods_sel:
        Temperate, Tropical, C3G, C4G = readin('', method, m)
        df_temperate[m] = pd.Series(Temperate)
        df_tropical[m] = pd.Series(Tropical)
        df_C3G[m] = pd.Series(C3G)
        df_C4G[m] = pd.Series(C4G)

    return(df_temperate.assign(Model=method),
           df_tropical.assign(Model=method),
           df_C3G.assign(Model=method),
           df_C4G.assign(Model=method))

### Generate grouped dataframes for ensemble averages
def grouped_dataframes_ens(index, method):
    df = generate_dataframe_ens(method)[index]
    df_long = pd.melt(df, 'Model', var_name='Method', value_name='FPC')
    return(df_long.replace(0, np.nan))

### Calculate reference stats
def reference_stats(veg_type, ax):
    median = df_CRUJRA[veg_type].replace(0, np.nan).median()
    quant_low = df_CRUJRA[veg_type].replace(0, np.nan).quantile(.25)
    quant_high = df_CRUJRA[veg_type].replace(0, np.nan).quantile(.75)

    if ax==ax10:
        label_median='Median$\mathrm{_{CRUJRA}}$'
        label_quant_low='Q1$\mathrm{_{CRUJRA}}$'
        label_quant_high='Q3$\mathrm{_{CRUJRA}}$'
    else:
        label_median='_nolegend_'
        label_quant_low='_nolegend_'
        label_quant_high='_nolegend_'

    ax.axhline(quant_low,color='k',lw=2,ls='-.',alpha=0.7,label=label_quant_low)
    ax.axhline(quant_high,color='k',lw=2,ls=':',alpha=0.7,label=label_quant_high)
    ax.axhline(median,color='k',lw=2,ls='--',alpha=0.7,label=label_median)

### Readin reanalysis
df_CRUJRA = pd.DataFrame()
Temperate, Tropical, C3G, C4G = readin('','CRUJRA','')
df_CRUJRA['Temperate'] = Temperate
df_CRUJRA['Tropical'] = Tropical
df_CRUJRA['C3G'] = C3G
df_CRUJRA['C4G'] = C4G

axes_temp=[ax1,ax2,ax3]
axes_tropical=[ax4,ax5,ax6]
axes_C3G=[ax7,ax8,ax9]
axes_C4G=[ax10,ax11,ax12]
axes=[axes_temp,axes_tropical,axes_C3G,axes_C4G]

veg_types=['Temperate', 'Tropical', 'C3G', 'C4G']

### Plot reference stats
for al,vt in zip(axes,veg_types):
    for a in al:
        reference_stats(vt, a)

### Get grouped dataframes for bias corrected models
df_BC_temperate = grouped_dataframes(0)
df_BC_tropical = grouped_dataframes(1)
df_BC_C3G = grouped_dataframes(2)
df_BC_C4G = grouped_dataframes(3)

### Get grouped dataframes for ensemble averaging methods
df_UMMM_temperate = grouped_dataframes_ens(0, 'Uniform')
df_UMMM_tropical = grouped_dataframes_ens(1, 'Uniform')
df_UMMM_C3G = grouped_dataframes_ens(2, 'Uniform')
df_UMMM_C4G = grouped_dataframes_ens(3, 'Uniform')

df_RF_temperate = grouped_dataframes_ens(0, 'Random_Forest')
df_RF_tropical = grouped_dataframes_ens(1, 'Random_Forest')
df_RF_C3G = grouped_dataframes_ens(2, 'Random_Forest')
df_RF_C4G = grouped_dataframes_ens(3, 'Random_Forest')

### Plot bias corrected models
axes_BC = [ax1,ax4,ax7,ax10]
df_BC = [df_BC_temperate, df_BC_tropical, df_BC_C3G, df_BC_C4G]

colors = ['#a1c9f4', '#ffb482', '#8de5a1', '#ff9f9b', '#d0bbff',
          '#debb9b', '#cfcfcf', '#fffea3', '#fab0e4', '#b9f2f0']
sns.set_palette(sns.color_palette(colors))

for ab, df in zip(axes_BC, df_BC):
    ab = sns.boxplot(x='Model', hue='Method', y='FPC',
                      data=df, showfliers=False, whis=0,
                      ax=ab)

### Plot uniform averages
axes_UMMM = [ax2,ax5,ax8,ax11]
df_UMMM = [df_UMMM_temperate, df_UMMM_tropical, df_UMMM_C3G, df_UMMM_C4G]

colors = ['#fffea3', '#fffea3', '#fffea3', '#fffea3']
sns.set_palette(sns.color_palette(colors))

for au, df in zip(axes_UMMM, df_UMMM):
    au = sns.boxplot(x='Model', hue='Method', y='FPC',
                      data=df, showfliers=False, whis=0, width=0.6,
                      ax=au)

    hatches = ['', '', '//',  '//', '..', '..', '\\\\', '\\\\']
    # Loop over the bars
    for i,thisbar in enumerate(au.patches):
        # Set a different hatch for each bar
        thisbar.set_hatch(hatches[i])

### Plot random forest averages

axes_RF = [ax3,ax6,ax9,ax12]
df_RF = [df_RF_temperate, df_RF_tropical, df_RF_C3G, df_RF_C4G]

colors = ['#b9f2f0', '#b9f2f0', '#b9f2f0', '#b9f2f0']
sns.set_palette(sns.color_palette(colors))

for ar, df in zip(axes_RF, df_RF):
    ar = sns.boxplot(x='Model', hue='Method', y='FPC',
                      data=df, showfliers=False, whis=0, width=0.6,
                      ax=ar)

    hatches = ['', '', '//',  '//', '..', '..', '\\\\', '\\\\']
    # Loop over the bars
    for i,thisbar in enumerate(ar.patches):
        # Set a different hatch for each bar
        thisbar.set_hatch(hatches[i])

for a in (ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12):
    a.legend_.remove()

# ax7.legend(handles=ax7.legend_.legendHandles[:len(methods_legend)],
#            loc='upper center', bbox_to_anchor=(-0.7,-0.45), ncol=5, frameon=False)
# ax8.legend(handles=ax8.legend_.legendHandles[:len(methods_legend)],
#            loc='upper center', bbox_to_anchor=(-0.8,-0.46), ncol=2, frameon=False)


for a in (ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12):
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.set_xlabel('')

for a in (ax2,ax3,ax5,ax6,ax8,ax9,ax11,ax12):
    a.spines['left'].set_visible(False)
    a.set_yticks([])

for a in (ax1,ax2,ax3,ax4,ax5,ax6):
    a.set_xticklabels([])

for a in (ax2,ax3,ax5,ax6,ax8,ax9,ax11,ax12):
    a.set_yticklabels([])
    a.set_ylabel('')

for a in (ax7,ax8,ax9,ax10,ax11,ax12):
    a.tick_params(axis='x', labelrotation=90)

for a in (ax1,ax2,ax3):
    a.set_ylim(-0.005,0.3315)
for a in (ax4,ax5,ax6):
    a.set_ylim(-0.005,0.58)
for a in (ax7,ax8,ax9):
    a.set_ylim(-0.005,0.43)
for a in (ax10,ax11,ax12):
    a.set_ylim(0.15,0.83)

title_left=['a)', 'b)', 'c)', 'd)']
title_right=['Temperate trees', 'Tropical trees', 'C3 grasses', 'C4 grasses']
axes=[ax1,ax4,ax7,ax10]

for a,tl,tr in zip(axes,title_left,title_right):
    a.set_title(tl, loc='left')
    a.set_title(tr,loc='right')

ax2.axvline(-0.35, color='k', lw=2, alpha=0.7)
ax5.axvline(-0.35, color='k', lw=2, alpha=0.7)
ax8.axvline(-0.35, color='k', lw=2, alpha=0.7)
ax11.axvline(-0.35, color='k', lw=2, alpha=0.7)

plt.show()

# plt.savefig('the_ultimate_boxplot_alternative.pdf')
