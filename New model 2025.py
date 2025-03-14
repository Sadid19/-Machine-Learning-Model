# Import necessary libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Load and preprocess dataset
file_path = 'E:\\7th Semester\\Math 6\\Project Details\\Data\\flooddata.csv'
  # Replace with your dataset path
data = pd.read_csv(file_path)

# Handle missing data
data['Flood?'] = data['Flood?'].fillna(data['Flood?'].mean())

# Create DateTime Index
data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(Day=1))
data.set_index('Date', inplace=True)

# Select numeric columns for resampling
numeric_data = data.select_dtypes(include=[np.number])
monthly_data = numeric_data.resample('MS').mean()

# Ensure target variable
flood_series = monthly_data['Flood?']
features = monthly_data.drop(columns=['Flood?'])

# Feature engineering
lag_features = ['Rainfall', 'Max_Temp', 'Min_Temp']
for feature in lag_features:
    for lag in range(1, 4):
        features[f'{feature}_lag{lag}'] = features[feature].shift(lag)
    features[f'{feature}_roll3'] = features[feature].rolling(window=3).mean()
features['Month_Sin'] = np.sin(2 * np.pi * features.index.month / 12)
features['Month_Cos'] = np.cos(2 * np.pi * features.index.month / 12)

# Drop NaN values caused by feature engineering
features = features.dropna()
flood_series = flood_series[features.index]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, flood_series, test_size=0.2, shuffle=False)

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SARIMA model
sarima_model = SARIMAX(y_train, order=(2, 1, 2), seasonal_order=(1, 1, 0, 12))
sarima_fit = sarima_model.fit(disp=False)
sarima_test_predictions = sarima_fit.predict(start=y_test.index[0], end=y_test.index[-1])

# Train XGBoost on residuals
residuals = y_train - sarima_fit.fittedvalues
xgb_model = XGBRegressor(n_estimators=200, max_depth=5, random_state=42)
xgb_model.fit(X_train_scaled, residuals)
xgb_test_residuals = xgb_model.predict(X_test_scaled)
final_predictions = sarima_test_predictions + xgb_test_residuals

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
mae = mean_absolute_error(y_test, final_predictions)
mape = np.mean(np.abs((y_test - final_predictions) / y_test.replace(0, np.nan))) * 100
accuracy = 100 - mape

# Display evaluation metrics
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"Accuracy: {accuracy:.2f}%")

# Visualization: Actual vs Predicted
plt.figure(figsize=(12, 6))
plt.plot(y_train.index, y_train, label="Training Data", color='blue')
plt.plot(y_test.index, y_test, label="Actual Test Data", color='orange')
plt.plot(y_test.index, final_predictions, label="Hybrid Model Predictions", color='green')
plt.axvline(x=X_test.index[0], color='red', linestyle='--', label="Test Start")
plt.title("Flood Prediction: Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Flood Occurrence")
plt.legend()
plt.show()

# Future forecast
future_steps = 120  # 10 years
future_index = pd.date_range(start=X_test.index[-1], periods=future_steps, freq='MS')
future_forecast = sarima_fit.get_forecast(steps=future_steps).predicted_mean

# Visualization: Future Forecast
plt.figure(figsize=(12, 6))
plt.plot(y_train.index, y_train, label="Training Data", color='blue')
plt.plot(y_test.index, y_test, label="Actual Test Data", color='orange')
plt.plot(y_test.index, final_predictions, label="Hybrid Model Predictions", color='green')
plt.plot(future_index, future_forecast, label="Future Forecast (2015-2024)", color='purple')
plt.axvline(x=future_index[0], color='red', linestyle='--', label="Forecast Start")
plt.title("Flood Prediction: Future Forecast")
plt.xlabel("Date")
plt.ylabel("Flood Occurrence")
plt.legend()
plt.show()
