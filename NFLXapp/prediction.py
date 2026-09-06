# NFLXapp prediction.py
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf

# Yahoo Finance treats the end date as exclusive.
today = (date.today() + timedelta(days=1)).isoformat()
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
model_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "ml_models", "nflx_model.pth"
)
model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
model.eval()


def download_history():
    data = yf.download(ticker, start="2002-01-01", end=today, progress=False)
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Yahoo Finance returned no NFLX price data.")
    return data


def price_values(data, column):
    try:
        values = np.asarray(data[column], dtype=np.float64).reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Yahoo Finance data does not contain {column} prices."
        ) from exc

    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"Yahoo Finance returned invalid {column} prices.")
    return values


def normalise(features):
    data = download_history()
    features_array = np.asarray(features, dtype=np.float64)
    if features_array.shape != (3,):
        raise ValueError("Prediction requires Open, High, and Low prices.")

    historical_values = np.column_stack(
        [
            price_values(data, "Open"),
            price_values(data, "High"),
            price_values(data, "Low"),
        ]
    )
    data_min = historical_values.min(axis=0)
    data_max = historical_values.max(axis=0)
    if np.any(data_max == data_min):
        raise ValueError("Historical price data cannot be normalized.")
    normalized_features = (features_array - data_min) / (data_max - data_min)
    return normalized_features


def denormalise(normalised_res):
    data = download_history()
    close_values = price_values(data, "Close")
    data_min = close_values.min()
    data_max = close_values.max()
    if data_max == data_min:
        raise ValueError("Historical close data cannot be denormalized.")
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
