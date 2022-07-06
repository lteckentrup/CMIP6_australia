import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import xarray as xr
from scipy import signal

model_names = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5',
               'CESM2-WACCM', 'CMCC-CM2-SR5', 'EC-Earth3', 'EC-Earth3-Veg',
               'GFDL-CM4', 'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0',
               'IPSL-CM6A-LR', 'KIOST-ESM', 'MIROC6', 'MPI-ESM1-2-HR',
               'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM',
               'NorESM2-MM']

def rank_skill(var):
    monthly = pd.read_csv('../Selection_skill/csv_files/monthly_'+var+
                          '_skill.csv',index_col='Model')
    annual = pd.read_csv('../Selection_skill/csv_files/annual_'+var+
                         '_skill.csv',index_col='Model')

    monthly['Average'] = monthly.mean(axis=1)
    annual['Average'] = annual.mean(axis=1)

    df=pd.DataFrame()
    df['Monthly']=monthly['Average'].rank(method='max', ascending=True)
    df['Annual']=annual['Average'].rank(method='max', ascending=True)
    df_transpose = df.T

    return(df_transpose)

def rank_independence(var):
    df_annual=pd.read_csv('../Selection/csv_files/annual_'+var+
                          '_independent_pearson.csv')
    df_monthly=pd.read_csv('../Selection/csv_files/monthly_'+var+
                           '_independent_pearson.csv')

    annual=[]
    monthly=[]

    for mn in model_names:
        monthly.append(df_monthly[mn][df_monthly[mn] < 0.3].count())
        annual.append(df_annual[mn][df_annual[mn] < 0.3].count())

    df_annual_independent = pd.DataFrame()
    df_monthly_independent = pd.DataFrame()

    df_annual_independent['Annual'] = annual
    df_monthly_independent['Monthly'] = monthly

    df_rank = pd.DataFrame()
    df_rank['Model'] = model_names
    df_rank['Monthly'] = df_monthly_independent.rank(method='max', ascending=False)
    df_rank['Annual'] = df_annual_independent.rank(method='max', ascending=False)
    df_rank = df_rank.set_index('Model')

    df_transpose = df_rank.T

    return(df_transpose)

def rank_extremes(var):
    avg=[]
    change=[]
    iav=[]

    for mn in model_names:
        df = pd.read_csv('../LPJ_monthly_corrected/original_csv/'+var+'_full.csv')
        avg.append(df[mn][-30:].mean())
        delta = df[mn]-df[mn][:30].mean(axis=0)
        change.append(delta[-30:].mean())
        df_notrend = df.apply(signal.detrend)
        iav.append(df_notrend[mn][-30:].std())

    df_avg = pd.DataFrame()
    df_change = pd.DataFrame()
    df_IAV = pd.DataFrame()
    df_rank=pd.DataFrame()

    df_avg['Average'] = avg
    df_change['Change'] = change
    df_IAV['IAV'] = iav

    df_rank['Model'] = model_names
    df_rank['Average'] = df_avg['Average'].rank(ascending=False)
    df_rank['Change'] = df_change['Change'].rank(ascending=False)
    df_rank['IAV'] = df_IAV['IAV'].rank(ascending=False)

    df_avg_transpose = df_rank.set_index('Model').T
    return(df_avg_transpose)

df_rank_skill_temp=rank_skill('temp')
df_rank_skill_prec=rank_skill('prec')
df_rank_independence_temp=rank_independence('temp')
df_rank_independence_prec=rank_independence('prec')
df_rank_extreme_temp=rank_extremes('temp')
df_rank_extreme_prec=rank_extremes('prec')

color_20=cm.tab20(np.arange(0,20,1))
color_add=cm.tab20b([0,2,4])
color=np.vstack((color_20,color_add))

fig = plt.figure(figsize=(8,10))

fig.subplots_adjust(hspace=0.12)
fig.subplots_adjust(wspace=0.2)
fig.subplots_adjust(right=0.98)
fig.subplots_adjust(left=0.09)
fig.subplots_adjust(bottom=0.2)
fig.subplots_adjust(top=0.97)

plt.rcParams['text.usetex'] = False
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.size'] = 11
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

ax1 = fig.add_subplot(3,2,1)
ax2 = fig.add_subplot(3,2,2)
ax3 = fig.add_subplot(3,2,3)
ax4 = fig.add_subplot(3,2,4)
ax5 = fig.add_subplot(3,2,5)
ax6 = fig.add_subplot(3,2,6)

for mn, c in zip(model_names, color):
    if mn in ['ACCESS-CM2', 'CESM2-WACCM', 'EC-Earth3-Veg', 'INM-CM5-0',
              'MPI-ESM1-2-HR', 'NorESM2-LM']:
        mt='o'
        ax1.scatter(x=np.array([1.2,2.2]), y=df_rank_skill_temp[mn].values,
                    color=c, marker=mt, label=mn)
        ax3.scatter(x=np.array([1.2,2.2]), y=df_rank_independence_temp[mn].values,
                    color=c, marker=mt, label=mn)
        ax2.scatter(x=np.array([1.2,2.2]), y=df_rank_skill_prec[mn].values,
                    color=c, marker=mt, label=mn)
        ax4.scatter(x=np.array([1.2,2.2]), y=df_rank_independence_prec[mn].values,
                    color=c, marker=mt, label=mn)
        ax5.scatter(x=np.array([0.7,1.7,2.7]), y=df_rank_extreme_temp[mn].values,
                    color=c, marker=mt, label=mn)
        ax6.scatter(x=np.array([0.7,1.7,2.7]), y=df_rank_extreme_prec[mn].values,
                    color=c, marker=mt, label=mn)
    elif mn in ['ACCESS-ESM1-5', 'CMCC-CM2-SR5', 'GFDL-CM4', 'IPSL-CM6A-LR',
                'MPI-ESM1-2-LR', 'NorESM2-MM']:
        mt='^'
        ax1.scatter(x=np.array([1.4,2.4]), y=df_rank_skill_temp[mn].values,
                    color=c, marker=mt, label=mn)
        ax3.scatter(x=np.array([1.4,2.4]), y=df_rank_independence_temp[mn].values,
                    color=c, marker=mt, label=mn)
        ax2.scatter(x=np.array([1.4,2.4]), y=df_rank_skill_prec[mn].values,
                    color=c, marker=mt, label=mn)
        ax4.scatter(x=np.array([1.4,2.4]), y=df_rank_independence_prec[mn].values,
                    color=c, marker=mt, label=mn)
        ax5.scatter(x=np.array([0.9,1.9,2.9]), y=df_rank_extreme_temp[mn].values,
                    color=c, marker=mt, label=mn)
        ax6.scatter(x=np.array([0.9,1.9,2.9]), y=df_rank_extreme_prec[mn].values,
                    color=c, marker=mt, label=mn)
    elif mn in ['BCC-CSM2-MR','EC-Earth3', 'GFDL-ESM4', 'KIOST-ESM',
                'MRI-ESM2-0', 'TaiESM1']:
        mt='s'
        ax1.scatter(x=np.array([1.6,2.6]), y=df_rank_skill_temp[mn].values,
                    color=c, marker=mt, label=mn)
        ax3.scatter(x=np.array([1.6,2.6]), y=df_rank_independence_temp[mn].values,
                    color=c, marker=mt, label=mn)
        ax2.scatter(x=np.array([1.6,2.6]), y=df_rank_skill_prec[mn].values,
                    color=c, marker=mt, label=mn)
        ax4.scatter(x=np.array([1.6,2.6]), y=df_rank_independence_prec[mn].values,
                    color=c, marker=mt, label=mn)
        ax5.scatter(x=np.array([1.1,2.1,3.1]), y=df_rank_extreme_temp[mn].values,
                    color=c, marker=mt, label=mn)
        ax6.scatter(x=np.array([1.1,2.1,3.1]), y=df_rank_extreme_prec[mn].values,
                    color=c, marker=mt, label=mn)
    elif mn in ['CanESM5', 'EC-Earth3-CC', 'INM-CM4-8', 'MIROC6', 'NESM3']:
        mt='D'
        ax1.scatter(x=np.array([1.8,2.8]), y=df_rank_skill_temp[mn].values,
                    color=c, marker=mt, label=mn)
        ax3.scatter(x=np.array([1.8,2.8]), y=df_rank_independence_temp[mn].values,
                    color=c, marker=mt, label=mn)
        ax2.scatter(x=np.array([1.8,2.8]), y=df_rank_skill_prec[mn].values,
                    color=c, marker=mt, label=mn)
        ax4.scatter(x=np.array([1.8,2.8]), y=df_rank_independence_prec[mn].values,
                    color=c, marker=mt, label=mn)
        ax5.scatter(x=np.array([1.3,2.3,3.3]), y=df_rank_extreme_temp[mn].values,
                    color=c, marker=mt, label=mn)
        ax6.scatter(x=np.array([1.3,2.3,3.3]), y=df_rank_extreme_prec[mn].values,
                    color=c, marker=mt, label=mn)

for a in (ax1,ax2,ax3,ax4):
    a.set_xlim([0.5,3.5])
    a.set_xticks([1.5,2.5])

for a in (ax1,ax2,ax3,ax4,ax5,ax6):
    a.invert_yaxis()
    a.set_yticks(np.arange(1,25,2))
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)

for a in (ax1,ax2):
    a.axhline(4.5,linewidth=1, color='k', ls='-', alpha=0.5)
    a.axvline(2, linewidth=1,  color='k', alpha=0.5)
    a.set_xticklabels([])

for a in (ax3,ax4):
    a.axhline(6.5,linewidth=1, color='k', ls='-', alpha=0.5)
    a.axvline(2, linewidth=1,  color='k', alpha=0.5)
    a.set_xticklabels(['Monthly', 'Annual'])

for a in (ax5,ax6):
    a.axhline(1.5,linewidth=1, color='k', ls='-', alpha=0.5)
    a.axhline(20.5,linewidth=1, color='k', ls='-', alpha=0.5)
    a.axvline(1.5, linewidth=1,  color='k', alpha=0.5)
    a.axvline(2.5, linewidth=1,  color='k', alpha=0.5)
    a.set_xlim([0.5,3.5])
    a.set_xticks([1,2,3])
    a.set_xticklabels(['Averages', 'Change', 'IAV'])

ax1.set_ylabel('Rank$_\mathrm{T, skill}$')
ax2.set_ylabel('Rank$_\mathrm{PPT, skill}$')
ax3.set_ylabel('Rank$_\mathrm{T, independence}$')
ax4.set_ylabel('Rank$_\mathrm{PPT, independence}$')
ax5.set_ylabel('Rank$_\mathrm{T, extremes}$')
ax6.set_ylabel('Rank$_\mathrm{PPT, extremes}$')

ax1.set_title('a)', loc='left')
ax2.set_title('b)', loc='left')
ax3.set_title('c)', loc='left')
ax4.set_title('d)', loc='left')
ax5.set_title('e)', loc='left')
ax6.set_title('f)', loc='left')

ax1.set_title('Temperature')
ax2.set_title('Precipitation')

ax6.legend(loc='upper center', bbox_to_anchor=(-0.2, -0.15), ncol=4, frameon=False)

plt.show()
# plt.savefig('rank_clim.pdf')
