import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sns.set_theme(style='ticks')

fig = plt.figure(figsize=(9,9))

plt.rcParams['text.usetex'] = False
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.size'] = 11
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

fig.subplots_adjust(hspace=0.19)
fig.subplots_adjust(wspace=0.12)
fig.subplots_adjust(top=0.95)
fig.subplots_adjust(bottom=0.28)
fig.subplots_adjust(right=0.97)
fig.subplots_adjust(left=0.1)

ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

def readin(model,method,selection):
    if method in ('Weighted', 'Uniform', 'Random_Forest', 'original', 'SCALING',
                  'MVA', 'QM', 'CDFt', 'MRec', 'dOTC', 'R2D2'):
        suffix='_1850-2100.nc'

    if method == 'CRUJRA':
        fname = ('../reanalysis/CTRL/CRUJRA/fpc_LPJ-GUESS_1901-2018.nc')
    elif method == 'Weighted':
        fname = ('../LPJ_ensemble_averages/Weighted/fpc_weighted_1850-2100.nc')
    elif method in ('Uniform', 'Random_Forest'):
        fname = ('../LPJ_ensemble_averages/'+method+'/fpc_'+selection+suffix)
    elif method in ('original', 'SCALING', 'MVA', 'QM', 'CDFt', 'MRec', 'dOTC'):
        fname = ('../LPJ_monthly_corrected/'+method+'/'+model+'/fpc_'+model+suffix)

    ds = xr.open_dataset(fname)
    ds = ds.sel(Time=slice('1989', '2018'))

    Temperate = ds['TeNE']+ds['TeBS']+ds['IBS']+ds['TeBE']
    Tropical = ds['TrBE']+ds['TrIBE']+ds['TrBR']

    C3G = ds['C3G']
    C4G = ds['C4G']

    return(Temperate.values.flatten(),
           Tropical.values.flatten(),
           C3G.values.flatten(),
           C4G.values.flatten())

methods = ['original', 'SCALING', 'MVA', 'QM', 'CDFt', 'MRec', 'dOTC']
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

    if model == 'EC-Earth3-Veg':
        model_num = 2
    elif model == 'INM-CM4-8':
        model_num = 4
    elif model == 'KIOST-ESM':
        model_num = 6
    elif model == 'MPI-ESM1-2-HR':
        model_num = 8
    elif model == 'NorESM2-MM':
        model_num = 10

    return(df_temperate.assign(Model=model_num),
           df_tropical.assign(Model=model_num),
           df_C3G.assign(Model=model_num),
           df_C4G.assign(Model=model_num))

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

    if method in ('Weighted', 'Random_Forest'):
        Temperate, Tropical, C3G, C4G = readin('', method, 'full')
        df_temperate['full'] = pd.Series(Temperate)
        df_tropical['full'] = pd.Series(Tropical)
        df_C3G['full'] = pd.Series(C3G)
        df_C4G['full'] = pd.Series(C4G)
    else:
        for m in methods_sel:
            Temperate, Tropical, C3G, C4G = readin('', method, m)
            df_temperate[m] = pd.Series(Temperate)
            df_tropical[m] = pd.Series(Tropical)
            df_C3G[m] = pd.Series(C3G)
            df_C4G[m] = pd.Series(C4G)

    if method == 'Uniform':
        method_num = 12
    elif method == 'Weighted':
        method_num = 13
    elif method == 'Random_Forest':
        method_num = 14

    return(df_temperate.assign(Model=method_num),
           df_tropical.assign(Model=method_num),
           df_C3G.assign(Model=method_num),
           df_C4G.assign(Model=method_num))

### Generate grouped dataframes for ensemble averages
def grouped_dataframes_ens(index, method):
    df = generate_dataframe_ens(method)[index]
    df_long = pd.melt(df, 'Model', var_name='Method', value_name='FPC')
    return(df_long.replace(0, np.nan))
    # return(df_long)

### Calculate reference stats
def reference_stats(veg_type, ax):
    median = df_CRUJRA[veg_type].replace(0, np.nan).median()
    quant_low = df_CRUJRA[veg_type].replace(0, np.nan).quantile(.25)
    quant_high = df_CRUJRA[veg_type].replace(0, np.nan).quantile(.75)

    ax.axhline(quant_low,color='k',lw=2,ls='-.',alpha=0.7)
    ax.axhline(quant_high,color='k',lw=2,ls=':',alpha=0.7)
    ax.axhline(median,color='k',lw=2,ls='--',alpha=0.7)

### Readin reanalysis
df_CRUJRA = pd.DataFrame()
Temperate, Tropical, C3G, C4G = readin('', 'CRUJRA', '')
df_CRUJRA['Temperate'] = Temperate
df_CRUJRA['Tropical'] = Tropical
df_CRUJRA['C3G'] = C3G
df_CRUJRA['C4G'] = C4G

axes=[ax1,ax2,ax3,ax4]

veg_types=['Temperate', 'Tropical', 'C3G', 'C4G']

### Plot reference stats
for a,vt in zip(axes,veg_types):
        reference_stats(vt, a)

def boxplot(method):
    ### Get grouped dataframes for bias corrected models
    if method == 'BC':
        dataframes = [grouped_dataframes(0),
                      grouped_dataframes(1),
                      grouped_dataframes(2),
                      grouped_dataframes(3)]
    else:
        dataframes = [grouped_dataframes_ens(0, method),
                      grouped_dataframes_ens(1, method),
                      grouped_dataframes_ens(2, method),
                      grouped_dataframes_ens(3, method)]

    ### Define colors and boxplot width
    if method == 'BC':
        colors = ['#a1c9f4', '#ffb482', '#8de5a1', '#ff9f9b', '#d0bbff',
                  '#debb9b', '#cfcfcf']
        width=1.7
    elif method=='Uniform':
        colors = ['#fffea3', '#fffea3', '#fffea3', '#fffea3']
        width=1.1
    elif method == 'Weighted':
        colors = ['#fab0e4', '#fab0e4', '#fab0e4', '#fab0e4']
        width=0.3
    elif method == 'Random_Forest':
        colors = ['#b9f2f0', '#b9f2f0', '#b9f2f0', '#b9f2f0']
        width=0.3

    sns.set_palette(sns.color_palette(colors))
    for a, df in zip(axes, dataframes):
        a = sns.boxplot(x='Model', hue='Method', y='FPC',
                        data=df, showfliers=False, whis=0, width=width,
                        ax=a, order = np.arange(1,16))

boxplot('BC')
boxplot('Uniform')
boxplot('Weighted')
boxplot('Random_Forest')

for a in (ax1,ax2,ax3,ax4):
    a.legend_.remove()

legend_elements = [Line2D([0], [0], color='k', lw=2, ls='-'),
                   Line2D([0], [0], color='k', lw=2, ls=':'),
                   Line2D([0], [0], color='k', lw=2, ls='--'),
                   Patch(facecolor='#a1c9f4', edgecolor='w'),
                   Patch(facecolor='#ffb482', edgecolor='w'),
                   Patch(facecolor='#8de5a1', edgecolor='w'),
                   Patch(facecolor='#ff9f9b', edgecolor='w'),
                   Patch(facecolor='#d0bbff', edgecolor='w'),
                   Patch(facecolor='#debb9b', edgecolor='w'),
                   Patch(facecolor='#cfcfcf', edgecolor='w'),
                   Patch(facecolor='#fffea3', edgecolor='w'),
                   Patch(facecolor='#fab0e4', edgecolor='w'),
                   Patch(facecolor='#b9f2f0', edgecolor='w')
                   ]

legend_labels = ['Q$_{\mathrm{low}}$ LG$_\mathrm{CRUJRA}$',
                 'Q$_{\mathrm{high}}$ LG$_\mathrm{CRUJRA}$',
                 'Median LG$\mathrm{_{CRUJRA}}$', 'Raw', 'Scaling', 'MAV',
                 'QM', 'CDF-t', 'R2D2', 'dOTC', 'Uniform', 'Weighted',
                 'Random Forest']

ax4.legend(legend_elements, legend_labels, loc='upper center',
           bbox_to_anchor=(-0.1, -0.5), ncol=5, frameon=False)

hatches = ['']
hatches = hatches * 47
hatches.extend(['//', '..', '\\\\', '', '', '', '', '', '', '',
                '//', '..', '\\\\'])

def hatching(axis):
    for i,thisbar in enumerate(axis.patches):
        # Boxes from left to right
        thisbar.set_hatch(hatches[i])

for a in (ax1,ax2,ax3,ax4):
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.axvline(10.13, color='k', lw=2, alpha=0.7)
    a.set_xticks([1,3,5,7,9,11,12,13])
    a.set_xticklabels(['LG$_\mathrm{EC-Earth3-Veg}$', 'LG$_\mathrm{INM-CM4-8}$',
                       'LG$_\mathrm{KIOST-ESM}$', 'LG$_\mathrm{MPI-ESM1-2-HR}$',
                       'LG$_\mathrm{NorESM2-MM}$', 'ENS$_\mathrm{Arithmetic}$',
                       'ENS$_\mathrm{Weighted}$', 'ENS$_\mathrm{RF}$'])

    a.tick_params(axis='x', labelrotation=90)
    a.set_xlabel('')
    a.set_xlim(0,14)
    hatching(a)

for a in (ax1,ax2):
    a.set_xticklabels([])
for a in (ax2,ax4):
    a.set_ylabel('')
for a in (ax3,ax4):
    a.tick_params(axis='x', labelrotation=90)

title_left=['a)', 'b)', 'c)', 'd)']
title_right=['Temperate trees', 'Tropical trees', 'C3 grasses', 'C4 grasses']

for a,tl,tr in zip(axes,title_left,title_right):
    a.set_title(tl, loc='left')
    a.set_title(tr,loc='center')

# plt.show()

plt.savefig('the_ultimate_boxplot.pdf')
