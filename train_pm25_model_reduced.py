import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load dataset (your full merged dataset with PM2.5)
df = pd.read_csv("final_merged_environmental_data 2.csv")  # replace with your dataset

# Use only selected features
features = ["temperature_2m_max", "temperature_2m_min", "windspeed_10m_max", "precipitation_sum"]
X = df[features]
y = df["PM2.5"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
import numpy as np
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

mae = mean_absolute_error(y_test, y_pred)

print(f"✅ Retrained model:")
print(f"📊 R² Score: {r2:.3f}")
print(f"📊 RMSE: {rmse:.2f} µg/m³")
print(f"📊 MAE: {mae:.2f} µg/m³")

# Save model
joblib.dump(model, "random_forest_pm25_retrained.pkl")

# Plot predictions
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Actual vs Predicted PM2.5 (Reduced Model)")
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted_retrained.png", dpi=300)
plt.show()
