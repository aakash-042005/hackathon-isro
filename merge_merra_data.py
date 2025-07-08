import xarray as xr
import pandas as pd
import numpy as np

# Define file paths (update with your actual downloaded .nc filenames)
files = {
    "temperature_2m_max": "T2MMAX.nc",
    "temperature_2m_min": "T2MMIN.nc",
    "u10": "U10M.nc",
    "v10": "V10M.nc",
    "precipitation": "PRECTOT.nc",
    # "aod_550": "AOD.nc",  # optional
}

# Function to load variable from NetCDF
def load_variable(file_path, variable_name):
    ds = xr.open_dataset(file_path)
    var = list(ds.data_vars)[0]
    arr = ds[var].values
    lats = ds["lat"].values
    lons = ds["lon"].values
    latlon = [(lat, lon) for lat in lats for lon in lons]
    flat_vals = arr.flatten()
    df = pd.DataFrame(latlon, columns=["Latitude", "Longitude"])
    df[variable_name] = flat_vals
    return df

# Load all variables and merge
merged_df = None
for var_name, file_path in files.items():
    df = load_variable(file_path, var_name)
    if merged_df is None:
        merged_df = df
    else:
        merged_df = pd.merge(merged_df, df, on=["Latitude", "Longitude"], how="inner")

# Compute wind speed if both U10 and V10 exist
if "u10" in merged_df.columns and "v10" in merged_df.columns:
    merged_df["windspeed_10m_max"] = np.sqrt(merged_df["u10"]**2 + merged_df["v10"]**2)
    merged_df.drop(columns=["u10", "v10"], inplace=True)

# Save to CSV
merged_df.to_csv("prediction_ready_grid.csv", index=False)
print("✅ Merged data saved to prediction_ready_grid.csv")
