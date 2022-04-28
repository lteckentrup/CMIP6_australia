historical='historical'
scenario='ssp585'
realisation='r1i1p1f1'
temp_res='day'
var='tas'

var_lpj='temp'
standard_name='air_temperature'

pathway_ACCESS='/g/data/fs38/publications/CMIP6'
pathwayCMIP='/g/data/oi10/replicas/CMIP6'

pathwayHistorical=${historical}/${realisation}/${temp_res}/${var}
pathwayScenario=${scenario}/${realisation}/${temp_res}/${var}

suffixScenario_prel='ssp585_r1i1p1f1_gn_18500101-21001231_prel.nc'
suffixScenario='ssp585_r1i1p1f1_gn_20150101-21001231.nc'

#-------------------------------------------------------------------------------

inst='CSIRO-ARCCSS'
model='ACCESS-CM2'

cdo -L -chname,${var},${var_lpj} -sellonlatbox,110,155,-45,-9 \
       -sellonlatbox,-180,180,-90,90 -mergetime \
       ${pathway_ACCESS}/CMIP/${inst}/${model}/${pathwayHistorical}/gn/latest/* \
       ${pathway_ACCESS}/ScenarioMIP/${inst}/${model}/${pathwayScenario}/gn/latest/*_20* \
       ${var}_${scenario}/${model}/${var}_day_${model}_${suffixScenario_prel}

inst='CSIRO'
model='ACCESS-ESM1-5'

cdo -L -chname,${var},${var_lpj} -sellonlatbox,110,155,-45,-9 \
      -sellonlatbox,-180,180,-90,90 -mergetime \
      ${pathway_ACCESS}/CMIP/${inst}/${model}/${pathwayHistorical}/gn/latest/* \
      ${pathway_ACCESS}/ScenarioMIP/${inst}/${model}/${pathwayScenario}/gn/latest/*_20* \
      ${var}_${scenario}/${model}/${var}_day_${model}_${suffixScenario_prel}

inst='BCC'
model='BCC-CSM2-MR'

cdo -L -chname,${var},${var_lpj} -sellonlatbox,110,155,-45,-9 \
      -sellonlatbox,-180,180,-90,90 -mergetime \
      ${pathwayCMIP}/CMIP/${inst}/${model}/${pathwayHistorical}/gn/*/* \
      ${pathwayCMIP}/ScenarioMIP/${inst}/${model}/${pathwayScenario}/gn/*/*_20* \
      ${var}_${scenario}/${model}/${var}_day_${model}_${suffixScenario_prel}

inst='CCCma'
model='CanESM5'
version='v20190429'

cdo -L -chname,${var},${var_lpj} -sellonlatbox,110,155,-45,-9 \
      -sellonlatbox,-180,180,-90,90 -mergetime \
      ${pathwayCMIP}/CMIP/${inst}/${model}/${pathwayHistorical}/gn/${version}/* \
      ${pathwayCMIP}/ScenarioMIP/${inst}/${model}/${pathwayScenario}/gn/${version}/*_20* \
      ${var}_${scenario}/${model}/${var}_day_${model}_${suffixScenario_prel}

inst='NCAR'
model='CESM2-WACCM'
version='v20200702'

cdo -L -chname,${var},${var_lpj} -sellonlatbox,110,155,-45,-9 \
      -sellonlatbox,-180,180,-90,90 -mergetime \
      ${pathwayCMIP}/CMIP/${inst}/${model}/${pathwayHistorical}/gn/*/* \
      ${pathwayCMIP}/ScenarioMIP/${inst}/${model}/${pathwayScenario}/gn/${version}/*_20* \
      ${var}_${scenario}/${model}/${var}_day_${model}_${suffixScenario_prel}

inst='CMCC'
model='CMCC-CM2-SR5'

cdo -L -chname,${var},${var_lpj} -sellonlatbox,110,155,-45,-9 \
      -sellonlatbox,-180,180,-90,90 -mergetime \
      ${pathwayCMIP}/CMIP/${inst}/${model}/${pathwayHistorical}/gn/*/* \
      ${pathwayCMIP}/ScenarioMIP/${inst}/${model}/${pathwayScenario}/gn/*/*_20* \
      ${var}_${scenario}/${model}/${var}_day_${model}_${suffixScenario_prel}

inst='EC-Earth-Consortium'
model='EC-Earth3'

cdo -L -chname,${var},${var_lpj} -sellonlatbox,110,155,-45,-9 \
      -sellonlatbox,-180,180,-90,90 -mergetime \
      ${pathwayCMIP}/CMIP/${inst}/${model}/${pathwayHistorical}/gr/*/* \
      ${pathwayCMIP}/ScenarioMIP/${inst}/${model}/${pathwayScenario}/gr/*/*_20* \
      ${var}_${scenario}/${model}/${var}_day_${model}_${suffixScenario_prel}

inst='EC-Earth-Consortium'
model='EC-Earth3-Veg'

cdo -L -chname,${var},${var_lpj} -sellonlatbox,110,155,-45,-9 \
      -sellonlatbox,-180,180,-90,90 -mergetime \
      ${pathwayCMIP}/CMIP/${inst}/${model}/${pathwayHistorical}/gr/*/* \
      ${pathwayCMIP}/ScenarioMIP/${inst}/${model}/${pathwayScenario}/gr/*/*_20* \
      ${var}_${scenario}/${model}/${var}_day_${model}_${suffixScenario_prel}

inst='NOAA-GFDL'
model='GFDL-CM4'

cdo -L -chname,${var},${var_lpj} -sellonlatbox,110,155,-45,-9 \
      -sellonlatbox,-180,180,-90,90 -mergetime \
      ${pathwayCMIP}/CMIP/${inst}/${model}/${pathwayHistorical}/gr1/*/* \
      ${pathwayCMIP}/ScenarioMIP/${inst}/${model}/${pathwayScenario}/gr1/*/*_20* \
      ${var}_${scenario}/${model}/${var}_day_${model}_${suffixScenario_prel}

inst='NOAA-GFDL'
model='GFDL-ESM4'

cdo -L -chname,${var},${var_lpj} -sellonlatbox,110,155,-45,-9 \
      -sellonlatbox,-180,180,-90,90 -mergetime \
      ${pathwayCMIP}/CMIP/${inst}/${model}/${pathwayHistorical}/gr1/*/* \
      ${pathwayCMIP}/ScenarioMIP/${inst}/${model}/${pathwayScenario}/gr1/*/*_20* \
      ${var}_${scenario}/${model}/${var}_day_${model}_${suffixScenario_prel}

inst='INM'
model='INM-CM4-8'

cdo -L -chname,${var},${var_lpj} -sellonlatbox,110,155,-45,-9 \
      -sellonlatbox,-180,180,-90,90 -mergetime \
      ${pathwayCMIP}/CMIP/${inst}/${model}/${pathwayHistorical}/gr1/*/* \
      ${pathwayCMIP}/ScenarioMIP/${inst}/${model}/${pathwayScenario}/gr1/*/*_20* \
      ${var}_${scenario}/${model}/${var}_day_${model}_${suffixScenario_prel}

inst='INM'
model='INM-CM5-0'

cdo -L -chname,${var},${var_lpj} -sellonlatbox,110,155,-45,-9 \
      -sellonlatbox,-180,180,-90,90 -mergetime \
      ${pathwayCMIP}/CMIP/${inst}/${model}/${pathwayHistorical}/gr1/*/* \
      ${pathwayCMIP}/ScenarioMIP/${inst}/${model}/${pathwayScenario}/gr1/*/*_20* \
      ${var}_${scenario}/${model}/${var}_day_${model}_${suffixScenario_prel}

inst='IPSL'
model='IPSL-CM6A-LR'

cdo -L -chname,${var},${var_lpj} -sellonlatbox,110,155,-45,-9 \
      -sellonlatbox,-180,180,-90,90 -mergetime \
      ${pathwayCMIP}/CMIP/${inst}/${model}/${pathwayHistorical}/gr/*/* \
      ${pathwayCMIP}/ScenarioMIP/${inst}/${model}/${pathwayScenario}/gr/*/*_20* \
      ${var}_${scenario}/${model}/${var}_day_${model}_${suffixScenario_prel}

inst='KIOST'
model='KIOST-ESM'

cdo -L -chname,${var},${var_lpj} -sellonlatbox,110,155,-45,-9 \
      -sellonlatbox,-180,180,-90,90 -mergetime \
      ${pathwayCMIP}/CMIP/${inst}/${model}/${pathwayHistorical}/gr1/v2021*/* \
      ${pathwayCMIP}/ScenarioMIP/${inst}/${model}/${pathwayScenario}/gr1/v2021*/*_20* \
      ${var}_${scenario}/${model}/${var}_day_${model}_${suffixScenario_prel}

inst='MIROC'
model='MIROC6'

cdo -L -chname,${var},${var_lpj} -sellonlatbox,110,155,-45,-9 \
      -sellonlatbox,-180,180,-90,90 -mergetime \
      ${pathwayCMIP}/CMIP/${inst}/${model}/${pathwayHistorical}/gn/*/* \
      ${pathwayCMIP}/ScenarioMIP/${inst}/${model}/${pathwayScenario}/gn/*/*_20* \
      ${var}_${scenario}/${model}/${var}_day_${model}_${suffixScenario_prel}

inst='MPI-M'
model='MPI-ESM1-2-LR'

cdo -L -chname,${var},${var_lpj} -sellonlatbox,110,155,-45,-9 \
      -sellonlatbox,-180,180,-90,90 -mergetime \
      ${pathwayCMIP}/CMIP/${inst}/${model}/${pathwayHistorical}/gn/*/* \
      ${pathwayCMIP}/ScenarioMIP/${inst}/${model}/${pathwayScenario}/gn/*/*_20* \
      ${var}_${scenario}/${model}/${var}_day_${model}_${suffixScenario_prel}

inst='MRI'
model='MRI-ESM2-0'

cdo -L -chname,${var},${var_lpj} -sellonlatbox,110,155,-45,-9 \
      -sellonlatbox,-180,180,-90,90 -mergetime \
      ${pathwayCMIP}/CMIP/${inst}/${model}/${pathwayHistorical}/gn/*/* \
      ${pathwayCMIP}/ScenarioMIP/${inst}/${model}/${pathwayScenario}/gn/*/*_20* \
      ${var}_${scenario}/${model}/${var}_day_${model}_${suffixScenario_prel}

inst='NUIST'
model='NESM3'

cdo -L -chname,${var},${var_lpj} -sellonlatbox,110,155,-45,-9 \
      -sellonlatbox,-180,180,-90,90 -mergetime \
      ${pathwayCMIP}/CMIP/${inst}/${model}/${pathwayHistorical}/gn/*/* \
      ${pathwayCMIP}/ScenarioMIP/${inst}/${model}/${pathwayScenario}/gn/*/*_20* \
      ${var}_${scenario}/${model}/${var}_day_${model}_${suffixScenario_prel}

inst='NCC'
model='NorESM2-LM'

cdo -L -chname,${var},${var_lpj} -sellonlatbox,110,155,-45,-9 \
      -sellonlatbox,-180,180,-90,90 -mergetime \
      ${pathwayCMIP}/CMIP/${inst}/${model}/${pathwayHistorical}/gn/*/* \
      ${pathwayCMIP}/ScenarioMIP/${inst}/${model}/${pathwayScenario}/gn/*/*_20* \
      ${var}_${scenario}/${model}/${var}_day_${model}_${suffixScenario_prel}

inst='NCC'
model='NorESM2-MM'

cdo -L -chname,${var},${var_lpj} -sellonlatbox,110,155,-45,-9 \
      -sellonlatbox,-180,180,-90,90 -mergetime \
      ${pathwayCMIP}/CMIP/${inst}/${model}/${pathwayHistorical}/gn/*/* \
      ${pathwayCMIP}/ScenarioMIP/${inst}/${model}/${pathwayScenario}/gn/*/*_20* \
      ${var}_${scenario}/${model}/${var}_day_${model}_${suffixScenario_prel}

#-------------------------------------------------------------------------------

declare -a model_list=('ACCESS-CM2' 'ACCESS-ESM1-5' 'BCC-CSM2-MR' 'CanESM5'
                       'CESM2-WACCM' 'CMCC-CM2-SR5' 'EC-Earth3' 'EC-Earth3-Veg'
                       'GFDL-CM4' 'GFDL-ESM4' 'INM-CM4-8' 'INM-CM5-0'
                       'IPSL-CM6A-LR' 'KIOST-ESM' 'MIROC6' 'MPI-ESM1-2-HR'
                       'MPI-ESM1-2-LR' 'MRI-ESM2-0' 'NESM3' 'NorESM2-LM'
                       'NorESM2-MM')

for model in "${model_list[@]}"; do
    echo ${model}
    ncatted -a standard_name,${var_lpj},o,c,${standard_name} \
            ${var}_${scenario}/${model}/${var}_day_${model}_${suffixScenario_prel} \
            ${var}_${scenario}/${model}/${var}_${scenario}_r1i1p1f1_2015-2100_setgrid.nc
done
