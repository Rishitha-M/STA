# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 17:23:16 2025

@author: Omkar
"""
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load and preprocess the dataset
data_path = 'AAPL.csv'  # Replace with your file path
data = pd.read_csv(data_path)
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data.set_index('Date', inplace=True)

# Prepare features and target
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close']

# Train-test split (80% training, 20% testing)
train_size = int(0.8 * len(data))
X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

# Train the XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model.fit(X_train, y_train)

# Streamlit Application
st.title("Apple Stock Price Prediction")
st.write("This app predicts Apple stock prices using historical data.")

# Input fields for user
st.sidebar.header("Input Parameters")
open_price = st.sidebar.number_input("Open Price:", value=0.0, step=0.1)
high_price = st.sidebar.number_input("High Price:", value=0.0, step=0.1)
low_price = st.sidebar.number_input("Low Price:", value=0.0, step=0.1)
volume = st.sidebar.number_input("Volume:", value=0, step=1000)

# Predict button
if st.sidebar.button("Predict"):
    # Create input data
    input_data = pd.DataFrame([[open_price, high_price, low_price, volume]], 
                              columns=['Open', 'High', 'Low', 'Volume'])
    
    # Make prediction
    prediction = xgb_model.predict(input_data)
    
    # Display result
    st.write(f"Predicted Closing Price: **${prediction[0]:.2f}**")

