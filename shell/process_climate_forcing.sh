insol_1850=historical_ssp245_r1i1p1f1_W_m-2_1850_2100_historical_ssp245_regrid_setgrid.nc
prec_1850=historical_ssp245_r1i1p1f1_mm_sec_1850_2100_historical_ssp245_regrid_setgrid.nc
temp_1850=historical_ssp245_r1i1p1f1_K_1850_2100_historical_ssp245_regrid_setgrid.nc

forcing_lpj=/g/data/w35/lt0205/research/lpj_guess/forcing/climdata/CMIP6

declare -a model_list=('CanESM5' 'CESM2-WACCM' 'CMCC-CM2-SR5' 'EC-Earth3' 
                       'EC-Earth3-Veg' 'GFDL-CM4' 'IITM-ESM' 'INM-CM4-8'
                       'INM-CM5-0' 'IPSL-CM6A-LR' 'KIOST-ESM' 'MIROC6' 
                       'MPI-ESM1-2-HR' 'MPI-ESM1-2-LR' 'MRI-ESM2-0' 
                       'NorESM2-LM' 'NorESM2-MM')
                       
for model in "${model_list[@]}"; do
    for var in pr; do
        cdo -L -b F64 -remapycon,ozgrid.txt \
            ${var}/${model}/r1i1p1f1/${var}_${model}_${prec_1850} \
            prec/prec_${model}_SSP245_r1i1p1f1_K_1850_2100.nc
        ncpdq -F -O -a lat,lon,time \
              prec/prec_${model}_SSP245_r1i1p1f1_K_1850_2100.nc
              ${forcing_lpj}/prec_${model}_SSP245_r1i1p1f1_K_1850_2100.nc
    done 
    for var in rsds; do
        cdo -L -b F64 -remapycon,ozgrid.txt \
            ${var}/${model}/r1i1p1f1/${var}_${model}_${insol_1850} \
            insol/insol_${model}_SSP245_r1i1p1f1_K_1850_2100.nc
        ncpdq -F -O -a lat,lon,time \
              insol/insol_${model}_SSP245_r1i1p1f1_K_1850_2100.nc \
              ${forcing_lpj}/insol_${model}_SSP245_r1i1p1f1_K_1850_2100.nc
    done 
    for var in tas; do
        cdo -L -b F64 -remapycon,ozgrid.txt \
            ${var}/${model}/r1i1p1f1/${var}_${model}_${temp_1850} \
            temp/temp_${model}_SSP245_r1i1p1f1_K_1850_2100.nc
        ncpdq -F -O -a lat,lon,time \
              temp/temp_${model}_SSP245_r1i1p1f1_K_1850_2100.nc \
              ${forcing_lpj}/temp_${model}_SSP245_r1i1p1f1_K_1850_2100.nc
    done 
done 
