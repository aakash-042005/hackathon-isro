import xarray as xr

ds = xr.open_dataset("g4.timeAvgMap.M2T1NXFLX_5_12_4_PRECTOT.20240101-20240630.67E_5N_97E_37N.nc")
print(ds.data_vars)  # Lists all variables
