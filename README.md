# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("air_quality.csv")  # Replace with your dataset path

# Basic preprocessing
df = df.dropna()  # Remove rows with missing values

# Feature selection
features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity']
target = 'AQI'

X = df[features]
y = df[target]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training
model = XGBRegressor()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("RMSE:", rmse)
print("R² Score:", r2)

# Save model and scaler
pickle.dump(model, open("xgb_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Streamlit App
st.title("Air Quality Index (AQI) Predictor")

# User inputs
pm25 = st.number_input("PM2.5")
pm10 = st.number_input("PM10")
no2 = st.number_input("NO2")
so2 = st.number_input("SO2")
co = st.number_input("CO")
o3 = st.number_input("O3")
temp = st.number_input("Temperature")
humidity = st.number_input("Humidity")

if st.button("Predict AQI"):
    data = np.array([[pm25, pm10, no2, so2, co, o3, temp, humidity]])
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)
    st.success(f"Predicted AQI: {prediction[0]:.2f}")
