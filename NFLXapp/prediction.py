# NFLXapp prediction.py
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np
import os

# Get today's date dynamically
today = datetime.today().strftime('%Y-%m-%d')
# Fetch data from Yahoo Finance with dynamic end date
ticker = "NFLX"

# Define the linear regression model class
class LinearRegression(nn.Module):
    def __init__(self, input_features):
        super(LinearRegression, self).__init__()
        self.layer = nn.Linear(input_features, 1)

    def forward(self, X):
        return self.layer(X)

# Load the pre-trained model weights from the saved file
input_features = 3
model = LinearRegression(input_features)

# Get the path to the model file in ml_models folder
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml_models', 'nflx_model.pth')
model.load_state_dict(torch.load(model_path))
model.eval()

def normalise(features):
    data = yf.download(ticker, start="2002-01-01", end=today)
    data = data[['Open', 'High', 'Low']]
    features_array = np.array(features)
    data_min = data.min().values
    data_max = data.max().values
    normalized_features = (features_array - data_min) / (data_max - data_min)
    return normalized_features

def denormalise(normalised_res):
    data = yf.download(ticker, start="2002-01-01", end=today)
    data = data[['Close']]
    data_min = data.min().values[0]
    data_max = data.max().values[0]
    value = (normalised_res * (data_max - data_min)) + data_min
    return value

def predict_stock_price(features):
    try:
        features = normalise(features)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(features_tensor).item()
            prediction = denormalise(prediction)
        
        return round(prediction, 2)
    except Exception as e:
        raise ValueError(f"Error in predicting stock price: {e}")
