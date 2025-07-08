import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load data and retrained model
df = pd.read_csv("prediction_ready_grid.csv")
model = joblib.load("random_forest_pm25_retrained.pkl")

# Use the 4-feature model
features = ["temperature_2m_max", "temperature_2m_min", "windspeed_10m_max", "precipitation_sum"]
# Rename if needed
if "precipitation" in df.columns:
    df.rename(columns={"precipitation": "precipitation_sum"}, inplace=True)
elif "M2T1NXFLX_5_12_4_PRECTOT" in df.columns:
    df.rename(columns={"M2T1NXFLX_5_12_4_PRECTOT": "precipitation_sum"}, inplace=True)

# Define features
features = ["temperature_2m_max", "temperature_2m_min", "windspeed_10m_max", "precipitation_sum"]
X = df[features]


# Predict
df["PM2.5_predicted"] = model.predict(X)

# Plot the predicted values
plt.figure(figsize=(10, 8))
sc = plt.scatter(df["Longitude"], df["Latitude"],
                 c=df["PM2.5_predicted"], cmap="plasma", s=40, edgecolors='k')
plt.colorbar(sc, label="Predicted PM2.5 (µg/m³)")
plt.title("Predicted Surface PM2.5 over India (Reduced Model)", fontsize=14)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.tight_layout()
plt.savefig("pm25_prediction_map_retrained.png", dpi=300)
plt.show()

print("✅ Map saved as 'pm25_prediction_map_retrained.png'")
