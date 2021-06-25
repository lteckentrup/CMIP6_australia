import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
sns.set_theme(style='ticks', palette='pastel')

fig = plt.figure(figsize=(10,19))

fig.subplots_adjust(hspace=0.12)
fig.subplots_adjust(wspace=0.15)
fig.subplots_adjust(right=0.98)
fig.subplots_adjust(left=0.15)
fig.subplots_adjust(bottom=0.15)
fig.subplots_adjust(top=0.98)

plt.rcParams['text.usetex'] = False
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.size'] = 11
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

ax1 = fig.add_subplot(7,2,1)
ax2 = fig.add_subplot(7,2,2)
ax3 = fig.add_subplot(7,2,3)
ax4 = fig.add_subplot(7,2,4)
ax5 = fig.add_subplot(7,2,5)
ax6 = fig.add_subplot(7,2,6)
ax7 = fig.add_subplot(7,2,7)
ax8 = fig.add_subplot(7,2,8)
ax9 = fig.add_subplot(7,2,9)
ax10 = fig.add_subplot(7,2,10)
ax11 = fig.add_subplot(7,2,11)
ax12 = fig.add_subplot(7,2,12)
ax13 = fig.add_subplot(7,2,13)
ax14 = fig.add_subplot(7,2,14)

data_vegetation_mask = nc.Dataset('vegetation_mask.nc')
vegetation_mask = data_vegetation_mask.variables['land_cover'][:,:]

tropics_mask = vegetation_mask != 1
savanna_mask = vegetation_mask != 2
warm_temperate_mask = vegetation_mask != 3
cool_temperate_mask = vegetation_mask != 4
mediterranean_mask = vegetation_mask !=5
desert_mask = vegetation_mask !=6
total_mask = vegetation_mask > 6
total_mask = total_mask < 0

total_mask = vegetation_mask < 0
total_mask_new = vegetation_mask != 10 # australia

def readin(veg_mask,model,method):
    if method == 'original':
        dataset = nc.Dataset('../Australia/CMIP6/CTRL/'+model+
                             '/fpc_LPJ-GUESS_1850-2100.nc')
    elif method == 'CRUJRA':
        dataset = nc.Dataset('../Australia/reanalysis/CTRL/CRUJRA/fpc_LPJ-GUESS_1901-2018.nc')
    else:
        dataset = nc.Dataset('../Australia/LPJ_SBCK/'+method+'/fpc_'+model+
                             '_1851-2100.nc')

    BNE = dataset.variables['BNE'][:,:,:]
    BINE = dataset.variables['BINE'][:,:,:]
    BNS = dataset.variables['BNS'][:,:,:]
    TeNE = dataset.variables['TeNE'][:,:,:]
    TeBS = dataset.variables['TeBS'][:,:,:]
    IBS = dataset.variables['IBS'][:,:,:]
    TeBE = dataset.variables['TeBE'][:,:,:]
    TrBE = dataset.variables['TrBE'][:,:,:]
    TrIBE = dataset.variables['TrIBE'][:,:,:]
    TrBR = dataset.variables['TrBR'][:,:,:]
    C3G = dataset.variables['C3G'][:,:,:]
    C4G = dataset.variables['C4G'][:,:,:]

    Tree = BNE + BINE + BNS + TeNE + TeBS + IBS + TeBE + TrBE + TrIBE + TrBR
    Grass = C3G + C4G

    if method == 'original':
        Tree = Tree[51:-82,:,:]
        Grass = Grass[51:-82,:,:]
        Tree_masked = np.ma.array(Tree, mask = Tree*veg_mask[np.newaxis,:,:])
        Grass_masked = np.ma.array(Grass, mask = Grass*veg_mask[np.newaxis,:,:])
    elif method == 'CRUJRA':
        Tree_masked = np.ma.array(Tree, mask = Tree*veg_mask[np.newaxis,:,:])
        Grass_masked = np.ma.array(Grass, mask = Grass*veg_mask[np.newaxis,:,:])
    else:
        Tree = Tree[50:-82,:,:]
        Grass = Grass[50:-82,:,:]
        Tree_masked = np.ma.array(Tree, mask = Tree*veg_mask[np.newaxis,:,:])
        Grass_masked = np.ma.array(Grass, mask = Grass*veg_mask[np.newaxis,:,:])

    return(Tree_masked.flatten().compressed(), Grass_masked.flatten().compressed())

def grouped_boxplot(axis_grass, axis_tree, veg_mask, region):
    methods = ['CRUJRA', 'original', 'SCALING', 'MVA', 'QM', 'CDFt', 'MRec']

    df_can_tree = pd.DataFrame()
    df_can_grass = pd.DataFrame()
    df_inm_tree = pd.DataFrame()
    df_inm_grass = pd.DataFrame()
    df_mpi_tree = pd.DataFrame()
    df_mpi_grass = pd.DataFrame()
    df_nor_tree = pd.DataFrame()
    df_nor_grass = pd.DataFrame()

    for m in methods:
        Tree, Grass = readin(veg_mask, 'CanESM5', m)
        df_can_tree[m] = pd.Series(Tree)
        df_can_grass[m] = pd.Series(Grass)
    for m in methods:
        Tree, Grass = readin(veg_mask, 'INM-CM4-8', m)
        df_inm_tree[m] = pd.Series(Tree)
        df_inm_grass[m] = pd.Series(Grass)
    for m in methods:
        Tree, Grass = readin(veg_mask, 'MPI-ESM1-2-HR', m)
        df_mpi_tree[m] = pd.Series(Tree)
        df_mpi_grass[m] = pd.Series(Grass)
    for m in methods:
        Tree, Grass = readin(veg_mask, 'NorESM2-MM', m)
        df_nor_tree[m] = pd.Series(Tree)
        df_nor_grass[m] = pd.Series(Grass)

    df_can_grass = df_can_grass.assign(Model='CanESM5')
    df_can_tree = df_can_tree.assign(Model='CanESM5')
    df_inm_grass = df_inm_grass.assign(Model='INM-CM4-8')
    df_inm_tree = df_inm_tree.assign(Model='INM-CM4-8')
    df_mpi_grass = df_mpi_grass.assign(Model='MPI-ESM1-2-HR')
    df_mpi_tree = df_mpi_tree.assign(Model='MPI-ESM1-2-HR')
    df_nor_grass = df_nor_grass.assign(Model='NorESM2-MM')
    df_nor_tree = df_nor_tree.assign(Model='NorESM2-MM')

    df_grass = pd.concat([df_can_grass,df_inm_grass,df_mpi_grass,df_nor_grass])
    df_tree = pd.concat([df_can_tree,df_inm_tree,df_mpi_tree,df_nor_tree])
    
    df_grass_long = pd.melt(df_grass, 'Model', var_name='Method', 
                            value_name='FPC')
    df_tree_long = pd.melt(df_tree, 'Model', var_name='Method', 
                           value_name='FPC')

    axis_grass = sns.boxplot(x='Model', hue='Method', y='FPC', 
                             data=df_grass_long, showfliers = False, 
                             ax=axis_grass)
    axis_tree = sns.boxplot(x='Model', hue='Method', y='FPC', data=df_tree_long,
                            showfliers = False, ax=axis_tree)

    axis_grass.set_xlabel('')
    axis_tree.set_xlabel('')
    axis_grass.set_ylabel(region+'\n \n FPC')
    axis_tree.set_ylabel('')

    if region != 'Desert':
        axis_grass.set_xticklabels([])
        axis_tree.set_xticklabels([])
    if region == 'Desert':
        axis_grass.tick_params(labelrotation=90)
        axis_tree.tick_params(labelrotation=90)

    if region == 'Total':
        axis_grass.set_title('Grass')
        axis_tree.set_title('Tree')

grouped_boxplot(ax1, ax2, total_mask, 'Total')
grouped_boxplot(ax3, ax4, tropics_mask, 'Tropics')
grouped_boxplot(ax5, ax6, savanna_mask, 'Savanna')
grouped_boxplot(ax7, ax8, warm_temperate_mask, 'Warm temperate')
grouped_boxplot(ax9, ax10, cool_temperate_mask, 'Cool temperate')
grouped_boxplot(ax11, ax12, mediterranean_mask, 'Mediterranean')
grouped_boxplot(ax13, ax14, desert_mask, 'Desert')

axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13]
for ax in axes:
    ax.legend([],[], frameon=False)
ax14.legend()
plt.show()
# plt.savefig('NorESM2-MM.png')
