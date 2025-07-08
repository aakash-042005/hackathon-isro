import pandas as pd

# ✅ Absolute path to your file
input_file = r"C:\Users\Dell\Downloads\g4.areaAvgTimeSeries.MYD08_D3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean.20240101-20240630.68E_6N_98E_37N.csv"

# ✅ Read CSV, skip metadata
df_raw = pd.read_csv(input_file, skiprows=12)

# ✅ Clean column names
df_raw.columns = ["Date", "AOD_550"]

# ✅ Convert date
df_raw["Date"] = pd.to_datetime(df_raw["Date"])
df_clean = df_raw[df_raw["AOD_550"] != -9999]

# ✅ Save cleaned CSV
df_clean.to_csv("daily_AOD_550_cleaned.csv", index=False)

print("✅ Done! Cleaned file saved.")
