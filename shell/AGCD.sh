idir='/g/data/zv2/agcd/v1/precip/total/r005/01day/agcd_v1_precip_total_r005_daily_'

cdo -L -chname,precip,prec -chunit,mm,'kg m-2' -remapcon,grid.txt -mergetime \
        ${idir}1989.nc ${idir}1990.nc ${idir}1991.nc ${idir}1992.nc \
        ${idir}1993.nc ${idir}1994.nc ${idir}1995.nc ${idir}1996.nc \
        ${idir}1997.nc ${idir}1998.nc ${idir}1999.nc ${idir}2000.nc \
        ${idir}2001.nc ${idir}2002.nc ${idir}2003.nc ${idir}2004.nc \
        ${idir}2005.nc ${idir}2006.nc ${idir}2007.nc ${idir}2008.nc \
        ${idir}2009.nc ${idir}2010.nc AGCD_prec_1989-2010.nc
