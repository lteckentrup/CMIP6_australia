import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import text
import scipy.stats

fig = plt.figure(figsize=(8,11))

plt.rcParams['text.usetex'] = False
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.size'] = 11
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

fig.subplots_adjust(hspace=0.3)
fig.subplots_adjust(wspace=0.0)
fig.subplots_adjust(top=0.95)
fig.subplots_adjust(bottom=0.08)
fig.subplots_adjust(right=0.97)
fig.subplots_adjust(left=0.08)

# ax1 = fig.add_subplot(4,3,1)
ax2 = fig.add_subplot(4,3,2)
ax3 = fig.add_subplot(4,3,3)
ax4 = fig.add_subplot(4,3,4)
ax5 = fig.add_subplot(4,3,5)
ax6 = fig.add_subplot(4,3,6)
ax7 = fig.add_subplot(4,3,7)
ax8 = fig.add_subplot(4,3,8)
ax9 = fig.add_subplot(4,3,9)
ax10 = fig.add_subplot(4,3,10)
ax11 = fig.add_subplot(4,3,11)
ax12 = fig.add_subplot(4,3,12)

def scatter(model,method,fname,var,axis,test):
    df_daily = pd.read_csv('../../monthly_lpj_guess/runs_'+method+'/'+model+'/'+
                           fname+'.out',
                           delim_whitespace=True)
    df_monthly = pd.read_csv('../../monthly_lpj_guess/runs_'+method+'_'+test+'/'+
                             model+'/'+fname+'.out',
                           delim_whitespace=True)

    df_daily = df_daily.loc[df_daily['Year'].isin(np.arange(1901, 2018))]
    df_monthly = df_monthly.loc[df_monthly['Year'].isin(np.arange(1901, 2018))]

    corr = scipy.stats.pearsonr(df_daily[var], df_monthly[var])
    axis.scatter(df_daily[var]*(-1), df_monthly[var]*(-1), s=0.5, color='tab:grey',
                 alpha=0.5, label='r='+'{:.2f}'.format(corr[0]))

    lims = [
    np.min([axis.get_xlim(), axis.get_ylim()]),  # min of both axes
    np.max([axis.get_xlim(), axis.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    axis.plot(lims, lims, 'k-', alpha=0.5, zorder=0)
    axis.set_aspect('equal')
    axis.set_xlim(lims)
    axis.set_ylim(lims)

    if fname == 'cflux':
        axis.axhline(y=0,color='k', alpha=0.5)
        axis.axvline(x=0,color='k', alpha=0.5)

        axis.legend(loc='lower left', bbox_to_anchor=(-0.25,0.7), frameon=False)
    else:
        axis.legend(loc='lower right', frameon=False)

    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    if axis in (ax2,ax4):
        axis.set_ylabel('C$_\mathrm{Total}$ [kgC]\nLG$_\mathrm{'+model+',ERA5,daily}$')
    elif axis in (ax7,ax10):
        axis.set_ylabel('C$_\mathrm{Total}$ [kgC]\nLG$_\mathrm{'+model+',CRUJRA,monthly}$')
    if axis in (ax10,ax11,ax12):
        axis.set_xlabel('C$_\mathrm{Total}$ [kgC]\nLG$_\mathrm{'+model+',CRUJRA,daily}$')

methods = ['SCALING', 'QM', 'CDFt', 'R2D2', 'dOTC']
methods_title = ['Scaling', 'QM', 'CDF-t', 'R2D2', 'dOTC']
axes = [ax2,ax3,ax4,ax5,ax6]

model='EC-Earth3-Veg'
# model='INM-CM4-8'
# model='KIOST-ESM'
# model='MPI-ESM1-2-HR'
# model='NorESM2-MM'

fname='cpool'
var='Total'

for mt,mtt,a in zip(methods,methods_title,axes):
    scatter(model,mt,fname,var,a,'ERA5')
    a.set_title(mtt)

methods = ['original', 'SCALING', 'QM', 'CDFt', 'R2D2', 'dOTC']
methods_title = ['Raw','Scaling', 'QM', 'CDF-t', 'R2D2', 'dOTC']

axes = [ax7,ax8,ax9,ax10,ax11,ax12]
for mt,mtt,a in zip(methods,methods_title,axes):
    scatter(model,mt,fname,var,a,'monthly')
    a.set_title(mtt)

letters = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 'k)']
axes = [ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]

for a,l in zip(axes,letters):
    a.set_title(l, loc='left')

fig.align_ylabels()

plt.show()
# plt.savefig('scatter_'+var+'_'+model+'.png', dpi=400)
