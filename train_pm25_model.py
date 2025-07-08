import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Load dataset
df = pd.read_csv("final_merged_environmental_data 2.csv")

# Step 2: Drop missing values
df.dropna(subset=[
    "PM2.5", "AOD_550", "temperature_2m_max", "temperature_2m_min",
    "relative_humidity_2m_max", "windspeed_10m_max", "precipitation_sum"
], inplace=True)

# Step 3: Plot PM2.5 distribution using matplotlib
plt.figure(figsize=(8, 4))
plt.hist(df["PM2.5"], bins=50, color='skyblue', edgecolor='black')
plt.title("PM2.5 Distribution")
plt.xlabel("PM2.5 (µg/m³)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("pm25_distribution.png")
plt.close()

# Step 4: Correlation matrix heatmap
features_all = [
    "PM2.5", "AOD_550", "temperature_2m_max", "temperature_2m_min",
    "relative_humidity_2m_max", "windspeed_10m_max", "precipitation_sum"
]
corr = df[features_all].corr()

plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap='coolwarm', interpolation='none')
plt.colorbar(label='Correlation')
plt.xticks(range(len(features_all)), features_all, rotation=45, ha='right')
plt.yticks(range(len(features_all)), features_all)
for i in range(len(features_all)):
    for j in range(len(features_all)):
        plt.text(i, j, f"{corr.iloc[j, i]:.2f}", ha='center', va='center', color='black')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# Step 5: Feature setup with log-transformed target
features = features_all[1:]  # exclude PM2.5
X = df[features]
y = np.log1p(df["PM2.5"])

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Predict and back-transform
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_actual = np.expm1(y_test)

# Step 9: Evaluation
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mae = mean_absolute_error(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)

print("✅ Model Trained Successfully")
print(f"📊 R² Score: {r2:.3f}")
print(f"📊 RMSE: {rmse:.2f} µg/m³")
print(f"📊 MAE: {mae:.2f} µg/m³")

# Step 10: Actual vs Predicted Plot
plt.figure(figsize=(7, 5))
plt.scatter(y_actual, y_pred, alpha=0.6, edgecolors="k", color='green')
plt.xlabel("Actual PM2.5 (µg/m³)")
plt.ylabel("Predicted PM2.5 (µg/m³)")
plt.title("Actual vs Predicted PM2.5")
plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--')
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()

# Step 11: Feature Importance Plot
importances = model.feature_importances_
plt.figure(figsize=(7, 4))
plt.barh(features, importances, color="orange", edgecolor='black')
plt.xlabel("Importance Score")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# Step 12: Save model and predictions
results = pd.DataFrame({
    "Actual_PM2.5": y_actual,
    "Predicted_PM2.5": y_pred
})
results.to_csv("pm25_predictions.csv", index=False)
joblib.dump(model, "random_forest_pm25_model.pkl")

print("📁 Files Saved:")
print("- pm25_distribution.png")
print("- correlation_heatmap.png")
print("- actual_vs_predicted.png")
print("- feature_importance.png")
print("- pm25_predictions.csv")
print("- random_forest_pm25_model.pkl")
