import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load the Dataset
df = pd.read_csv('logical_taxi_fare_dataset2.csv')  # Replace with the correct file path

# 2. Add noise to the features for more randomness
np.random.seed(42)

# Add random noise (in the range of -10% to +10% of the original values)
df['distance_km'] += df['distance_km'] * np.random.uniform(-0.1, 0.1, size=len(df))
df['duration_min'] += df['duration_min'] * np.random.uniform(-0.1, 0.1, size=len(df))
df['hour_of_day'] += df['hour_of_day'] * np.random.uniform(-0.1, 0.1, size=len(df))
df['traffic_conditions'] += df['traffic_conditions'] * np.random.uniform(-0.1, 0.1, size=len(df))

# 3. Preprocess Data
X = df[["distance_km", "duration_min", "hour_of_day", "traffic_conditions"]]  # Features
y = df["fare_amount"]  # Target

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)

# 7. Visualize Feature Importances
feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})
print(feature_importances)

# 8. Visualize Predictions
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color="blue", label="Actual Fare")
plt.scatter(range(len(y_pred)), y_pred, color="red", label="Predicted Fare", alpha=0.7)
plt.title("Actual vs Predicted Fares")
plt.xlabel("Test Sample Index")
plt.ylabel("Fare Amount")
plt.legend()
plt.show()

# 9. Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals, color="purple", alpha=0.7)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Residual Plot")
plt.xlabel("Actual Fare")
plt.ylabel("Residuals")
plt.show()

# 10. Custom Prediction
# Custom Prediction with User Input
print("Enter custom trip details:")
distance_km = float(input("Distance (in km): "))
duration_min = float(input("Duration (in minutes): "))
hour_of_day = int(input("Hour of the day (0-23): "))
traffic_conditions = int(input("Traffic conditions (1: Low, 2: Moderate, 3: Heavy): "))

# Create a DataFrame for the custom trip
custom_trip = pd.DataFrame([[distance_km, duration_min, hour_of_day, traffic_conditions]],
                           columns=["distance_km", "duration_min", "hour_of_day", "traffic_conditions"])

# Predict fare for the custom trip
predicted_fare = model.predict(custom_trip)
print("\nPredicted Fare for the custom trip:", round(predicted_fare[0], 2))
