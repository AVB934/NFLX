# NFLXapp prediction.py
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import yfinance as yf
from datetime import datetime

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
model.load_state_dict(torch.load('nflx_model.pth'))
model.eval()

def normalise(features):
    data = yf.download(ticker, start="2002-01-01", end=today)
    data = data[['Open', 'High', 'Low']]
    normalized_df = (features - data.min()) / (data.max() - data.min())
    return normalized_df

def denormalise(normalised_res):
    data = yf.download(ticker, start="2002-01-01", end=today)
    data = data[['Close']]
    value = (normalised_res * (data.max() - data.min())) + data.min()
    return value

def predict_stock_price(features):
    try:
        features = normalise(features)
        features_tensor = torch.tensor(features.values, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(features_tensor).item()
            prediction = denormalise(prediction)
        
        return round(prediction, 2)
    except Exception as e:
        raise ValueError(f"Error in predicting stock price: {e}")