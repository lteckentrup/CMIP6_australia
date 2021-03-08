declare -a model_list=('CABLE-POP' 'CLASS-CTEM' 'CLM5.0' 'DLEM' 'ISAM' 
                       'ISBA-CTRIP' 'JSBACH' 'JULES-ES' 'LPJ-GUESS' 'LPJ' 
                       'LPX-Bern' 'OCN' 'ORCHIDEE-CNP' 'ORCHIDEE' 'SDGVM'
                       'VISIT')

for exp in S1 S2; do
    for var in cVeg gpp nbp cSoil; do
        for model in CABLE-POP OCN ORCHIDEE-CNP ORCHIDEE SDGVM; do
            cdo -b F64 -L -selyear,1901/2018 -remapycon,fine_grid.txt \
                -invertlat ${var}/raw/${model}_${exp}_${var}.nc \
                 ${var}/processed/${model}_${exp}_${var}_1901-2018.nc
        done

        for model in CLASS-CTEM CLM5.0 ISBA-CTRIP JSBACH JULES-ES; do
            cdo -b F64 -L -selyear,1901/2018 -remapycon,fine_grid.txt \
                 ${var}/raw/${model}_${exp}_${var}.nc \
                 ${var}/processed/${model}_${exp}_${var}_1901-2018.nc
        done

        for model in ISAM LPJ-GUESS LPX-Bern; do
            cdo -b F64 -L -selyear,1901/2018 \
            ${var}/raw/${model}_${exp}_${var}.nc \
            ${var}/processed/${model}_${exp}_${var}_1901-2018.nc
        done

        for model in DLEM VISIT; do
            cdo -b F64 -L -selyear,1901/2018 -invertlat \
            ${var}/raw/${model}_${exp}_${var}.nc \
            ${var}/processed/${model}_${exp}_${var}_1901-2018.nc
        done
    done

    for var in gpp nbp; do
        for model in "${model_list[@]}"; do
            cdo -b F64 -L -divc,1e+12 -yearsum -mulc,86400 -muldpm \
                -sellonlatbox,112.25,153.75,-43.75,-10.25 \
                ${var}/processed/${model}_${exp}_${var}_1901-2018.nc \
                ${var}/processed/${model}_${exp}_${var}_australia_annual.nc

            cdo -b F64 -L -divc,1e+12 -mulc,86400 -muldpm \
                -settaxis,1901-01-01,00:00,1month \
                -sellonlatbox,112.25,153.75,-43.75,-10.25 \
                ${var}/processed/${model}_${exp}_${var}_1901-2018.nc \
                ${var}/processed/${model}_${exp}_${var}_australia_monthly.nc

            cdo mul ${var}/processed/${model}_${exp}_${var}_australia_annual.nc \
                    oz_mask.nc \
                    ${var}/processed/${model}_${exp}_${var}_australia_annual_mask.nc
            cdo mul ${var}/processed/${model}_${exp}_${var}_australia_monthly.nc \
                    oz_mask.nc \
                    ${var}/processed/${model}_${exp}_${var}_australia_monthly_mask.nc

            cdo -L -fldsum -mul \
                ${var}/processed/${model}_${exp}_${var}_australia_annual_mask.nc \
                -gridarea \
                ${var}/processed/${model}_${exp}_${var}_australia_annual_mask.nc \
                ${var}/processed/${model}_${exp}_${var}_australia_annual_oz.nc
            cdo -L -fldsum -mul \
                ${var}/processed/${model}_${exp}_${var}_australia_monthly_mask.nc \
                -gridarea \
                ${var}/processed/${model}_${exp}_${var}_australia_annual_mask.nc \
                ${var}/processed/${model}_${exp}_${var}_australia_monthly_oz.nc
        done

        for var in cVeg cSoil; do
            for model in "${model_list[@]}"; do
                cdo -b F64 -L -divc,1e+12 -yearsum \
                    -sellonlatbox,112.25,153.75,-43.75,-10.25 \
                    ${var}/processed/${model}_${exp}_${var}_1901-2018.nc \
                    ${var}/processed/${model}_${exp}_${var}_australia_annual.nc

                cdo -b F64 -L -divc,1e+12 \
                    -settaxis,1901-01-01,00:00,1month \
                    -sellonlatbox,112.25,153.75,-43.75,-10.25 \
                    ${var}/processed/${model}_${exp}_${var}_1901-2018.nc \
                    ${var}/processed/${model}_${exp}_${var}_australia_monthly.nc

                cdo mul ${var}/processed/${model}_${exp}_${var}_australia_annual.nc \
                        oz_mask.nc \
                        ${var}/processed/${model}_${exp}_${var}_australia_annual_mask.nc
                cdo mul ${var}/processed/${model}_${exp}_${var}_australia_monthly.nc \
                        oz_mask.nc \
                        ${var}/processed/${model}_${exp}_${var}_australia_monthly_mask.nc

                cdo -L -fldsum -mul \
                    ${var}/processed/${model}_${exp}_${var}_australia_annual_mask.nc \
                    -gridarea \
                    ${var}/processed/${model}_${exp}_${var}_australia_annual_mask.nc \
                    ${var}/processed/${model}_${exp}_${var}_australia_annual_oz.nc
                cdo -L -fldsum -mul \
                    ${var}/processed/${model}_${exp}_${var}_australia_monthly_mask.nc \
                    -gridarea \
                    ${var}/processed/${model}_${exp}_${var}_australia_annual_mask.nc \
                    ${var}/processed/${model}_${exp}_${var}_australia_monthly_oz.nc
        done
    done
done
